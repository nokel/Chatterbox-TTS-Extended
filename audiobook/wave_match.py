"""
Wave-match auto-tuning: make a cloned voice measurably match its target.

Given a trained voice and reference audio of the real speaker, the tuner
synthesizes the *same passages the reference speaks* across a grid of
generation settings and scores every candidate against the reference
recording by acoustic measurement, not by ear:

  * speaker similarity - cosine distance between speaker embeddings from
    chatterbox's own VoiceEncoder (the wave "fingerprint" match),
  * long-term average spectrum - overall timbre/EQ contour match,
  * pitch - median F0 and F0 spread match.

The best-scoring settings are returned along with waveform traces of the
reference and the clone so the UI can overlay them. The passages and the
winning settings are chosen by this module - the person training the voice
does not pick them.

Reference material comes from the voice's training dataset
(datasets/<voice>/wavs + metadata.csv) when it exists, or from explicit
audio files (transcribed with the app's Whisper backend).
"""

import csv
import os

import numpy as np

from . import synth

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
VOICES_DIR = os.path.join(BASE_DIR, "voices")

VE_SR = 16000
REF_MAX_SEC = 60.0

# score = weighted sum of similarity components, each in [0, 1]
W_SPEAKER, W_SPECTRUM, W_PITCH = 0.65, 0.20, 0.15

DEFAULT_GRID = {
    "exaggeration": [0.35, 0.5, 0.65],
    "cfg_weight": [0.35, 0.5, 0.65],
    "temperature": [0.7, 0.9],
}
QUICK_GRID = {
    "exaggeration": [0.4, 0.6],
    "cfg_weight": [0.4, 0.6],
    "temperature": [0.8],
}

_VE_CACHE = {}


# ------------------------------------------------------------- features ----

def _load_ve(voice_name):
    """Chatterbox's VoiceEncoder from the voice folder (CPU, tiny model)."""
    if voice_name in _VE_CACHE:
        return _VE_CACHE[voice_name]
    import torch
    from safetensors.torch import load_file
    from chatterbox.src.chatterbox.models.voice_encoder import VoiceEncoder

    ve = VoiceEncoder()
    ve.load_state_dict(load_file(
        os.path.join(VOICES_DIR, voice_name, "ve.safetensors")))
    ve.to("cpu").eval()
    _VE_CACHE.clear()          # keep at most one resident
    _VE_CACHE[voice_name] = ve
    return ve


def _to_16k(x, sr):
    import librosa
    if sr != VE_SR:
        x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=VE_SR)
    return x.astype(np.float32)


def speaker_embed(voice_name, x, sr):
    import torch
    ve = _load_ve(voice_name)
    with torch.inference_mode():
        e = ve.embeds_from_wavs([_to_16k(x, sr)], sample_rate=VE_SR,
                                as_spk=True)
    e = np.asarray(e, dtype=np.float32).reshape(-1)
    return e / (np.linalg.norm(e) + 1e-9)


def ltas(x, sr):
    """Long-term average log-mel spectrum, mean-removed (timbre contour)."""
    import librosa
    m = librosa.feature.melspectrogram(y=x.astype(np.float32), sr=sr,
                                       n_mels=64, fmax=8000)
    v = np.log(m.mean(axis=1) + 1e-8)
    return v - v.mean()


def pitch_stats(x, sr):
    """(median_f0_hz, iqr_semitones) over voiced frames, or None."""
    import librosa
    f0 = librosa.yin(x.astype(np.float32), fmin=60, fmax=400, sr=sr)
    rms = librosa.feature.rms(y=x.astype(np.float32),
                              frame_length=2048, hop_length=512)[0]
    n = min(len(f0), len(rms))
    f0, rms = f0[:n], rms[:n]
    voiced = f0[(rms > rms.max() * 0.1) & (f0 > 60) & (f0 < 400)]
    if len(voiced) < 10:
        return None
    st = 12 * np.log2(voiced / 55.0)
    return float(np.median(voiced)), float(np.percentile(st, 75)
                                           - np.percentile(st, 25))


def wave_trace(x, sr, points=700):
    """Min/max amplitude envelope for drawing: [(lo, hi), ...]."""
    x = np.asarray(x, dtype=np.float32)
    if not len(x):
        return []
    step = max(1, len(x) // points)
    m = (len(x) // step) * step
    seg = x[:m].reshape(-1, step)
    return [[float(a), float(b)] for a, b in zip(seg.min(axis=1),
                                                 seg.max(axis=1))]


def _similarity(ref, cand_wav, sr, voice_name):
    """Component similarities of a candidate against reference features."""
    e = speaker_embed(voice_name, cand_wav, sr)
    s_spk = float(np.clip(np.dot(ref["embed"], e), 0.0, 1.0))

    d = np.linalg.norm(ref["ltas"] - ltas(cand_wav, sr)) / np.sqrt(
        len(ref["ltas"]))
    s_spec = float(np.exp(-d))

    s_pitch = 0.5
    cp = pitch_stats(cand_wav, sr)
    if cp is not None and ref["pitch"] is not None:
        d_med = abs(12 * np.log2(cp[0] / ref["pitch"][0]))   # semitones
        d_iqr = abs(cp[1] - ref["pitch"][1])
        s_pitch = float(np.exp(-(d_med / 2.0 + d_iqr / 4.0)))

    total = W_SPEAKER * s_spk + W_SPECTRUM * s_spec + W_PITCH * s_pitch
    return {"score": total, "speaker": s_spk, "spectrum": s_spec,
            "pitch": s_pitch}


# ------------------------------------------------------------ reference ----

def dataset_lines(voice_name):
    """[(wav_path, text)] from the voice's training dataset, if any."""
    meta = os.path.join(DATASETS_DIR, voice_name, "metadata.csv")
    wavs = os.path.join(DATASETS_DIR, voice_name, "wavs")
    out = []
    if os.path.isfile(meta):
        with open(meta, encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="|"):
                if len(row) >= 2:
                    p = os.path.join(wavs, row[0] + ".wav")
                    if os.path.isfile(p):
                        out.append((p, row[1].strip()))
    return out


def reference_features(voice_name, audio_paths=None, transcribe_fn=None):
    """Measure the target speaker. Returns dict with embed/ltas/pitch/trace,
    plus the passages (texts) the tuner will synthesize.

    audio_paths: explicit reference clips; otherwise the voice's dataset.
    transcribe_fn(path)->str: needed only for explicit clips.
    """
    import librosa

    if audio_paths:
        clips = [(p, transcribe_fn(p) if transcribe_fn else "") for p in
                 audio_paths]
    else:
        clips = dataset_lines(voice_name)
    if not clips:
        raise RuntimeError(
            f"No reference audio for '{voice_name}': supply clips or keep "
            f"its training dataset in datasets/{voice_name}/")

    # concat up to REF_MAX_SEC of audio for measurement
    parts, total, sr0 = [], 0.0, None
    for p, _ in clips:
        x, sr = librosa.load(p, sr=None, mono=True)
        if sr0 is None:
            sr0 = sr
        elif sr != sr0:
            x = librosa.resample(x, orig_sr=sr, target_sr=sr0)
        parts.append(x.astype(np.float32))
        total += len(x) / sr0
        if total >= REF_MAX_SEC:
            break
    ref_wav = np.concatenate(parts)

    # tuner-chosen passages: mid-length lines with real sentence structure
    texts = [t for _, t in clips if t]
    texts.sort(key=lambda t: abs(len(t) - 120))
    passages = texts[:2] or [""]

    return {
        "embed": speaker_embed(voice_name, ref_wav, sr0),
        "ltas": ltas(ref_wav, sr0),
        "pitch": pitch_stats(ref_wav, sr0),
        "trace": wave_trace(ref_wav, sr0),
        "sr": sr0,
        "passages": [p for p in passages if p],
        "seconds": float(len(ref_wav) / sr0),
    }


# --------------------------------------------------------------- tuning ----

def _grid_points(grid):
    pts = [{}]
    for key, values in grid.items():
        pts = [{**p, key: v} for p in pts for v in values]
    return pts


def auto_match(voice_name, ref=None, grid=None, tts_url=synth.DEFAULT_TTS_URL,
               progress=None, log=None, audio_paths=None, transcribe_fn=None):
    """Grid-search generation settings for the closest acoustic match.

    Returns {"best": {...}, "candidates": [...], "ref_trace": [...],
             "passages": [...]}; candidates are sorted best-first and carry
    their component scores. best["params"] plugs straight into casting.
    """
    log = log or (lambda *_: None)
    progress = progress or (lambda *_: None)
    if ref is None:
        ref = reference_features(voice_name, audio_paths, transcribe_fn)
    if not ref["passages"]:
        raise RuntimeError("Reference clips have no transcripts to compare "
                           "against - supply a transcribe function.")
    text = " ".join(ref["passages"])[:400]
    pts = _grid_points(grid or DEFAULT_GRID)
    log(f"Matching '{voice_name}' against {ref['seconds']:.0f}s of "
        f"reference audio, {len(pts)} candidates.")

    results = []
    for i, p in enumerate(pts):
        progress(i, len(pts), f"Candidate {i + 1}/{len(pts)}: {p}")
        params = {"voice": voice_name, "seed": 424242, **synth.DEFAULT_PARAMS,
                  **p}
        try:
            wav, sr = synth.synth_unit(text, params, tts_url)
        except Exception as e:
            log(f"  candidate {p} failed: {e}")
            continue
        sim = _similarity(ref, wav, sr, voice_name)
        log(f"  {p} -> score {sim['score']:.4f} "
            f"(spk {sim['speaker']:.3f} spec {sim['spectrum']:.3f} "
            f"pitch {sim['pitch']:.3f})")
        results.append({"params": p, **sim,
                        "trace": wave_trace(wav, sr), "wav": wav, "sr": sr})
    if not results:
        raise RuntimeError("Every candidate failed - is the TTS server up?")

    results.sort(key=lambda r: r["score"], reverse=True)
    # keep audio only for the winner to stay light
    for r in results[1:]:
        r.pop("wav", None)
    progress(len(pts), len(pts), "Done")
    return {"best": results[0], "candidates": results,
            "ref_trace": ref["trace"], "passages": ref["passages"]}
