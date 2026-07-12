"""
Kokoro engine for near-instant TTS.

Kokoro-82M is ~6x smaller than chatterbox-turbo and speaks almost
immediately, at the cost of voice cloning: it has a fixed set of built-in
voices, and the exaggeration / CFG / temperature / reference-audio controls
have no effect. Exposes the same .generate()/.sr interface as the other
engines so every tab works unchanged.

Model files (kokoro-v1.0.onnx + voices-v1.0.bin) are searched for in the
project tree and common locations; set KOKORO_MODEL_PATH / KOKORO_VOICES_PATH
to override.
"""

import os

import numpy as np
import torch

DEFAULT_VOICE = "af_heart"

# A stable subset of kokoro v1.0 voices for the UI (full list comes from the
# loaded voices file at runtime).
COMMON_VOICES = [
    "af_heart", "af_sarah", "af_bella", "af_nicole",
    "am_adam", "am_michael", "bf_emma", "bm_george",
]


def find_kokoro_files():
    """Locate kokoro model + voices files without downloading if they exist."""
    env_model = os.environ.get("KOKORO_MODEL_PATH")
    env_voices = os.environ.get("KOKORO_VOICES_PATH")
    if env_model and env_voices and os.path.isfile(env_model) and os.path.isfile(env_voices):
        return env_model, env_voices

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        here,
        os.path.join(here, "models"),
        os.path.dirname(here),                    # chatterbox-AI
        os.path.dirname(os.path.dirname(here)),   # AI crap
    ]
    for d in candidates:
        m = os.path.join(d, "kokoro-v1.0.onnx")
        v = os.path.join(d, "voices-v1.0.bin")
        if os.path.isfile(m) and os.path.isfile(v):
            return m, v

    # Not found locally: download once into the project's models dir.
    target = os.path.join(here, "models")
    os.makedirs(target, exist_ok=True)
    m = os.path.join(target, "kokoro-v1.0.onnx")
    v = os.path.join(target, "voices-v1.0.bin")
    base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
    import urllib.request
    for path, name in ((m, "kokoro-v1.0.onnx"), (v, "voices-v1.0.bin")):
        if not os.path.isfile(path):
            print(f"[KOKORO] Downloading {name} (one-time)...")
            urllib.request.urlretrieve(f"{base}/{name}", path)
    return m, v


class KokoroEngine:
    sr = 24000

    def __init__(self, voice=DEFAULT_VOICE):
        from kokoro_onnx import Kokoro
        model_path, voices_path = find_kokoro_files()
        print(f"[KOKORO] Loading {model_path}")
        self.kokoro = Kokoro(model_path, voices_path)
        self.voice = voice if voice else DEFAULT_VOICE
        self.device = "kokoro:onnx-cpu"

    def get_voices(self):
        try:
            return sorted(self.kokoro.get_voices())
        except Exception:
            return COMMON_VOICES

    def generate(self, text, audio_prompt_path=None, exaggeration=0.5,
                 cfg_weight=0.5, temperature=0.8, apply_watermark=False,
                 generator=None, **kwargs) -> torch.Tensor:
        # Voice cloning / CFG / exaggeration / temperature / seeds are not
        # applicable to kokoro; the built-in voice is used.
        samples, sr = self.kokoro.create(text, voice=self.voice, speed=1.0,
                                         lang="en-us")
        self.sr = sr
        data = np.asarray(samples, dtype=np.float32)
        return torch.from_numpy(data).unsqueeze(0)
