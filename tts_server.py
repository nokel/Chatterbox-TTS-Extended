# tts_server.py — headless Chatterbox TTS API server.
#
# Serves the trained voices in voices/ over HTTP so other programs (e.g. the
# Discord voice bot) can synthesize speech without running the Gradio app.
#
#   GET  /health  -> {"ok": true, "loaded_voice": "Trump" | null}
#   GET  /voices  -> ["Trump", "Trump-paced", ...]
#   POST /tts     -> WAV bytes (24 kHz mono 16-bit)
#        body: {"text": "...", "voice": "Trump",
#               "exaggeration": 0.5, "cfg_weight": 0.5,
#               "temperature": 0.8, "seed": null}
#
# Exactly one voice model is resident at a time; requesting a different voice
# swaps it out. Output speech loudness is matched to the voice's original
# training recordings (voices/<voice>/loudness.json), same as the main app.
#
# Run with run_tts_server.ps1 (listens on http://127.0.0.1:7861).

import io
import os
import re
import sys
import threading

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

app = FastAPI(title="Chatterbox headless TTS")

_LOCK = threading.Lock()
_MODEL = None
_LOADED_VOICE = None


def list_voices():
    out = []
    if os.path.isdir(VOICES_DIR):
        for name in sorted(os.listdir(VOICES_DIR)):
            d = os.path.join(VOICES_DIR, name)
            if os.path.isfile(os.path.join(d, "t3_cfg.safetensors")):
                out.append(name)
    return out


def _get_model(voice):
    """Load (or swap to) the requested voice. Caller holds _LOCK."""
    global _MODEL, _LOADED_VOICE
    if _LOADED_VOICE == voice and _MODEL is not None:
        return _MODEL
    import torch
    from chatterbox.src.chatterbox.tts import ChatterboxTTS

    if _MODEL is not None:
        print(f"[TTS] Evicting voice '{_LOADED_VOICE}'")
        _MODEL = None
        _LOADED_VOICE = None
        torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TTS] Loading voice '{voice}' on {device}...")
    _MODEL = ChatterboxTTS.from_local(os.path.join(VOICES_DIR, voice), device)
    _LOADED_VOICE = voice
    print(f"[TTS] Voice '{voice}' ready")
    return _MODEL


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


@app.get("/health")
def health():
    return {"ok": True, "loaded_voice": _LOADED_VOICE}


@app.get("/voices")
def voices():
    return list_voices()


@app.post("/tts")
def tts(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    if req.voice not in list_voices():
        raise HTTPException(404, f"unknown voice '{req.voice}' (have: {list_voices()})")

    import torch

    with _LOCK:
        model = _get_model(req.voice)
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
    uvicorn.run(app, host="127.0.0.1", port=7861, log_level="warning")
