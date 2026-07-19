# tts_server.py — headless Chatterbox TTS API server.
#
# Serves the trained voices in voices/ over HTTP so other programs (the
# audiobook reader, the Discord voice bot) can synthesize speech without
# running the Gradio app.
#
#   GET  /health -> {"ok": true, resident/pinned/capacity/stats, loaded_voice}
#   GET  /voices -> ["Trump", "fallout_4", ...]
#   POST /warm   -> {"voice": "...", "pin": false}  preload a voice into RAM
#   POST /plan   -> {"narrator": "...", "voices": {"name": count, ...}}
#                   set up residency for a whole book at once
#   POST /tts    -> WAV bytes (24 kHz mono 16-bit)
#        body: {"text": "...", "voice": "Trump", "exaggeration": 0.5,
#               "cfg_weight": 0.5, "temperature": 0.8, "seed": null}
#
# Voice residency is managed by a VoiceRouter (see below): the narrator is
# pinned (loaded first, never evicted), character voices are loaded on
# demand and can be *warmed* ahead of when they're needed so speaking never
# stalls on a load; least-recently-used unpinned voices are evicted only
# when the resident set exceeds a RAM-derived capacity. Output loudness is
# matched to each voice's original training recordings.
#
# Run with run_tts_server.ps1 (listens on http://127.0.0.1:7861).

import io
import os
import re
import sys
import threading
import time
from collections import OrderedDict

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from voice_training import VOICES_DIR, get_voice_loudness, speech_rms_dbfs

MAX_CHUNK_CHARS = 250      # chatterbox is trained on short utterances
GAIN_LIMIT_DB = 12.0
PEAK_GUARD = 0.985

# Which engine synthesizes speech.
#
# "onnx" (default): the chatterbox-turbo ONNX model. Measured on this machine
# (Radeon 8060S / gfx1151) it runs at RTF ~0.5 - about twice as fast as
# realtime - so playback never has to wait for it. It clones a voice zero-shot
# from a reference wav, so there are no multi-GB per-voice weights to load and
# no load stall between characters.
#
# "pytorch": the per-voice fine-tuned weights (voices/<name>/t3_cfg.safetensors).
# Higher fidelity to a trained voice, but measured at RTF 2-5x (i.e. 12-24s to
# speak one sentence), which is far too slow to read a book aloud. Kept for the
# Voice Lab, where training and A/B testing a voice is the point and latency
# doesn't matter.
ENGINE = os.environ.get("CHATTERBOX_AUDIOBOOK_ENGINE", "onnx").lower()

# MIOpen on ROCm asks PyTorch for a conv workspace, gets 0 bytes back and falls
# back to naive solvers, which is a large part of why the pytorch path is slow
# (ROCm/rocm-libraries#4071 on gfx1151). FAST mode avoids that fallback.
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
REF_SECONDS = 15.0         # enough audio for a zero-shot clone

# Residency sizing. A loaded voice model is ~2-2.5 GB. Capacity is derived
# from free RAM (unified memory on APUs), leaving a reserve, and hard-capped
# so a huge cast doesn't try to hold everything at once. Override the whole
# thing with CHATTERBOX_TTS_MAX_VOICES (0 = auto).
PER_VOICE_GB = float(os.environ.get("CHATTERBOX_VOICE_RAM_GB", "2.5"))
RESERVE_GB = float(os.environ.get("CHATTERBOX_VOICE_RESERVE_GB", "12"))
HARD_MAX_VOICES = int(os.environ.get("CHATTERBOX_VOICE_HARD_MAX", "12"))
FORCED_MAX = int(os.environ.get("CHATTERBOX_TTS_MAX_VOICES", "0"))

app = FastAPI(title="Chatterbox headless TTS")

_GEN_LOCK = threading.Lock()   # serialize generation (one GPU job at a time)


def list_voices():
    out = []
    if os.path.isdir(VOICES_DIR):
        for name in sorted(os.listdir(VOICES_DIR)):
            d = os.path.join(VOICES_DIR, name)
            if os.path.isfile(os.path.join(d, "t3_cfg.safetensors")):
                out.append(name)
    return out


def _dataset_dir_for(voice):
    """The training-clip dir for a voice. Names don't always match case
    (voice 'Trump' vs datasets/trump)."""
    if not os.path.isdir(DATASETS_DIR):
        return None
    want = voice.lower()
    for d in sorted(os.listdir(DATASETS_DIR)):
        if d.lower() == want and os.path.isdir(os.path.join(DATASETS_DIR, d)):
            return os.path.join(DATASETS_DIR, d)
    return None


def ensure_reference_wav(voice):
    """Reference audio for the ONNX engine, which clones a voice from a wav
    rather than loading trained weights. Prefers an existing reference, else
    builds one once from the voice's own training clips."""
    vdir = os.path.join(VOICES_DIR, voice)
    p = os.path.join(vdir, "reference.wav")
    if os.path.isfile(p):
        return p

    # Build from the voice's training clips in preference to loudness_probe.wav:
    # the probe is only a few seconds of level-check audio, while the training
    # set is the material the voice was actually made from.
    ds = _dataset_dir_for(voice)
    if not ds:
        probe = os.path.join(vdir, "loudness_probe.wav")
        return probe if os.path.isfile(probe) else None
    clips = []
    for root, _dirs, files in os.walk(ds):
        for f in sorted(files):
            if f.lower().endswith(".wav"):
                clips.append(os.path.join(root, f))
    if not clips:
        probe = os.path.join(vdir, "loudness_probe.wav")
        return probe if os.path.isfile(probe) else None

    chunks, sr, total = [], None, 0.0
    for c in clips:
        try:
            x, csr = sf.read(c, dtype="float32", always_2d=False)
        except Exception:
            continue
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr is None:
            sr = csr
        elif csr != sr:
            continue          # don't resample here; just skip odd rates
        chunks.append(x)
        total += len(x) / float(sr)
        if total >= REF_SECONDS:
            break
    if not chunks:
        return None

    out = np.concatenate(chunks)[: int(REF_SECONDS * sr)]
    os.makedirs(vdir, exist_ok=True)
    path = os.path.join(vdir, "reference.wav")
    sf.write(path, out, sr, subtype="PCM_16")
    print(f"[onnx] built reference for '{voice}' from {len(chunks)} training "
          f"clip(s) ({total:.1f}s) -> {path}", flush=True)
    return path


class OnnxEngine:
    """One chatterbox-turbo model for every voice; voices differ only by the
    reference wav passed per request, so switching character costs nothing."""

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def ready(self):
        return self._model is not None

    def model(self):
        with self._lock:
            if self._model is None:
                from chatterbox_onnx import ChatterboxOnnxTTS
                t0 = time.time()
                print("[onnx] loading chatterbox-turbo ...", flush=True)
                self._model = ChatterboxOnnxTTS.from_pretrained(
                    lm_precision="q4", model_variant="chatterbox-turbo")
                print(f"[onnx] ready in {time.time() - t0:.1f}s", flush=True)
            return self._model


ONNX = OnnxEngine()


def _auto_capacity():
    if FORCED_MAX > 0:
        return FORCED_MAX
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
    except Exception:
        return 3
    return max(2, min(HARD_MAX_VOICES, int((avail_gb - RESERVE_GB) / PER_VOICE_GB)))


class VoiceRouter:
    """Decides which voice models live in RAM.

    - The narrator voice is pinned: loaded first and never evicted, because
      it speaks most of the book and must always be ready.
    - Other voices load on demand; warm() preloads one in the background so
      the load overlaps the currently-playing line instead of stalling it.
    - Loads are serialized and deduplicated (two requests for the same
      not-yet-loaded voice share one load); generation runs under a separate
      lock, so a voice can load *while* another voice is speaking.
    - When the resident set exceeds capacity, the least-recently-used
      *unpinned* voice is evicted.
    """

    def __init__(self):
        self._models = OrderedDict()      # name -> model, front = LRU
        self._pinned = set()
        self._loading = {}                # name -> Event (in-flight load)
        self._lock = threading.RLock()
        self._load_lock = threading.Lock()
        self._forced_cap = None
        self.stats = {}                   # name -> {loads,last_load_sec,uses}

    # -- capacity ---------------------------------------------------------
    def capacity(self):
        if self._forced_cap is not None:
            return self._forced_cap
        return _auto_capacity()

    def set_capacity(self, n):
        self._forced_cap = max(1, int(n))

    # -- helpers ----------------------------------------------------------
    def _device(self):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self, name):
        """Actually load a voice model. Overridable for testing."""
        from chatterbox.src.chatterbox.tts import ChatterboxTTS
        return ChatterboxTTS.from_local(
            os.path.join(VOICES_DIR, name), self._device())

    def _stat(self, name):
        return self.stats.setdefault(
            name, {"loads": 0, "last_load_sec": None, "uses": 0})

    def is_resident(self, name):
        with self._lock:
            return name in self._models

    def pin(self, name):
        with self._lock:
            self._pinned.add(name)

    # -- core: load / cache / evict --------------------------------------
    def ensure(self, name):
        """Return the model for `name`, loading synchronously if needed.
        Deduplicates concurrent loads of the same voice."""
        with self._lock:
            if name in self._models:
                self._models.move_to_end(name)
                self._stat(name)["uses"] += 1
                return self._models[name]
            ev = self._loading.get(name)
            mine = ev is None
            if mine:
                ev = self._loading[name] = threading.Event()

        if not mine:                      # another thread is loading it
            ev.wait()
            with self._lock:
                self._models.move_to_end(name)
                self._stat(name)["uses"] += 1
                return self._models[name]

        try:
            with self._load_lock:         # one heavy load at a time
                t0 = time.time()
                print(f"[router] loading '{name}' ...", flush=True)
                model = self._load_model(name)
                dt = time.time() - t0
            with self._lock:
                self._models[name] = model
                self._models.move_to_end(name)
                s = self._stat(name)
                s["loads"] += 1
                s["last_load_sec"] = round(dt, 2)
                s["uses"] += 1
                self._evict_locked()
                print(f"[router] '{name}' ready in {dt:.1f}s "
                      f"({len(self._models)}/{self.capacity()} resident, "
                      f"pinned={sorted(self._pinned)})", flush=True)
            return model
        finally:
            with self._lock:
                self._loading.pop(name, None)
            ev.set()

    def _evict_locked(self):
        import torch
        cap = self.capacity()
        for name in list(self._models.keys()):   # front = least recent
            if len(self._models) <= cap:
                break
            if name in self._pinned:
                continue
            del self._models[name]
            print(f"[router] evicted '{name}' (over capacity {cap})", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def warm(self, name, pin=False):
        """Ensure `name` is (being) loaded, in the background. Returns
        'resident' if already loaded, else 'warming'."""
        with self._lock:
            if pin:
                self._pinned.add(name)
            if name in self._models:
                return "resident"
            already = name in self._loading
        if not already:
            threading.Thread(target=self._warm_worker, args=(name,),
                             daemon=True, name=f"warm-{name}").start()
        return "warming"

    def _warm_worker(self, name):
        try:
            self.ensure(name)
        except Exception as e:            # a bad voice shouldn't kill the server
            print(f"[router] warm '{name}' failed: {e}", flush=True)

    def plan(self, narrator, counts):
        """Set up residency for a whole book: pin+load the narrator first,
        then preload the most-used voices up to capacity."""
        self.pin(narrator)
        cap = self.capacity()
        ordered = [narrator] + [
            v for v, _ in sorted(counts.items(), key=lambda kv: -kv[1])
            if v != narrator]
        for v in ordered[:cap]:
            self.warm(v, pin=(v == narrator))
        return self.status()

    def status(self):
        with self._lock:
            return {
                "resident": list(self._models.keys()),
                "pinned": sorted(self._pinned),
                "loading": list(self._loading.keys()),
                "capacity": self.capacity(),
                "stats": self.stats,
            }


ROUTER = VoiceRouter()


def _split_text(text):
    """Split long text into sentence groups the model handles well."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > MAX_CHUNK_CHARS:
            chunks.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        chunks.append(cur)
    return chunks or [text.strip()]


def _match_loudness(x, sr, target_db):
    """Gain the waveform so its speech RMS hits target_db (clamped, peak-safe)."""
    if target_db is None:
        return x
    cur = speech_rms_dbfs(x, sr)
    if cur is None:
        return x
    gain_db = max(-GAIN_LIMIT_DB, min(GAIN_LIMIT_DB, target_db - cur))
    y = x * (10.0 ** (gain_db / 20.0))
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > PEAK_GUARD:
        y = y * (PEAK_GUARD / peak)
    return y


class TTSRequest(BaseModel):
    text: str
    voice: str
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    seed: int | None = None


class WarmRequest(BaseModel):
    voice: str
    pin: bool = False


class PlanRequest(BaseModel):
    narrator: str
    voices: dict[str, int] = {}


@app.get("/health")
def health():
    st = ROUTER.status()
    st["ok"] = True
    st["engine"] = ENGINE
    if ENGINE == "onnx":
        # one model serves every voice, so residency/eviction don't apply
        st["model_ready"] = ONNX.ready()
    st["loaded_voices"] = st["resident"]
    st["loaded_voice"] = st["resident"][-1] if st["resident"] else None
    return st


@app.get("/voices")
def voices():
    return list_voices()


@app.post("/warm")
def warm(req: WarmRequest):
    if req.voice not in list_voices():
        raise HTTPException(404, f"unknown voice '{req.voice}'")
    if ENGINE == "onnx":
        # nothing per-voice to load; make sure the one model and this voice's
        # reference audio are ready so the first line doesn't pay for it
        ensure_reference_wav(req.voice)
        ONNX.model()
        return {"voice": req.voice, "state": "ready", "status": {"engine": ENGINE}}
    state = ROUTER.warm(req.voice, pin=req.pin)
    return {"voice": req.voice, "state": state, "status": ROUTER.status()}


@app.post("/plan")
def plan(req: PlanRequest):
    have = set(list_voices())
    if req.narrator not in have:
        raise HTTPException(404, f"unknown narrator voice '{req.narrator}'")
    if ENGINE == "onnx":
        # one shared model: just build any missing reference audio up front
        for v in [req.narrator] + list(req.voices):
            if v in have:
                ensure_reference_wav(v)
        ONNX.model()
        return {"engine": ENGINE, "narrator": req.narrator, "preloaded": ["(single model)"]}
    counts = {v: c for v, c in req.voices.items() if v in have}
    return ROUTER.plan(req.narrator, counts)


@app.post("/tts")
def tts(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    if req.voice not in list_voices():
        raise HTTPException(404, f"unknown voice '{req.voice}' (have: {list_voices()})")

    import torch

    if ENGINE == "onnx":
        ref = ensure_reference_wav(req.voice)
        if not ref:
            raise HTTPException(
                500, f"no reference audio for '{req.voice}' (looked in "
                     f"voices/{req.voice} and datasets/)")
        model = ONNX.model()
        with _GEN_LOCK:
            gen = None
            if req.seed is not None:
                gen = torch.Generator()
                gen.manual_seed(req.seed)
            pieces = []
            for chunk in _split_text(text):
                wav = model.generate(
                    chunk,
                    audio_prompt_path=ref,
                    exaggeration=req.exaggeration,
                    cfg_weight=req.cfg_weight,
                    temperature=req.temperature,
                    generator=gen,
                )
                pieces.append(wav.squeeze(0).cpu().numpy())
            sr = model.sr
    else:
        # load (or wait for a warm already in flight) OUTSIDE the generation
        # lock, so warming another voice doesn't block a voice that's speaking
        model = ROUTER.ensure(req.voice)

        with _GEN_LOCK:
            gen = None
            if req.seed is not None:
                gen = torch.Generator(device=model.device)
                gen.manual_seed(req.seed)
            pieces = []
            for chunk in _split_text(text):
                wav = model.generate(
                    chunk,
                    exaggeration=req.exaggeration,
                    cfg_weight=req.cfg_weight,
                    temperature=req.temperature,
                    generator=gen,
                )
                pieces.append(wav.squeeze(0).cpu().numpy())
            sr = model.sr

    x = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    x = _match_loudness(x, sr, get_voice_loudness(req.voice))

    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    print("[TTS] Starting headless TTS server on http://127.0.0.1:7861")
    print(f"[TTS] Voices available: {', '.join(list_voices()) or '(none)'}")
    print(f"[TTS] Voice residency capacity: {ROUTER.capacity()}")
    uvicorn.run(app, host="127.0.0.1", port=7861, log_level="warning")
