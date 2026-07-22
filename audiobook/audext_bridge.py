import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
AUDEXT = os.path.join(PROJECT, "audext")
AUDEXT_PY = os.path.join(AUDEXT, ".venv", "Scripts", "python.exe")

_RUNNER = (
    "import sys; from pathlib import Path; "
    "sys.path.insert(0, sys.argv[1]); "
    "from vocal_isolation import VocalIsolator; "
    "VocalIsolator(device=sys.argv[4]).isolate("
    "Path(sys.argv[2]), Path(sys.argv[3])); "
    "print('ISOLATE_OK ' + sys.argv[3])"
)


def available():
    return os.path.isfile(AUDEXT_PY) and os.path.isdir(AUDEXT)


def isolate_vocals(in_wav, out_wav, device="cpu", log=None, timeout=1800):
    log = log or (lambda *a: None)
    if not available():
        raise RuntimeError(f"audext interpreter not found at {AUDEXT_PY}")
    in_wav = os.path.abspath(in_wav)
    out_wav = os.path.abspath(out_wav)
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    proc = subprocess.run(
        [AUDEXT_PY, "-c", _RUNNER, AUDEXT, in_wav, out_wav, device],
        cwd=AUDEXT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout)
    if proc.returncode != 0 or not os.path.isfile(out_wav):
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise RuntimeError("audext vocal isolation failed: " +
                           (tail[-1] if tail else f"exit {proc.returncode}"))
    log(f"vocals isolated -> {os.path.basename(out_wav)}")
    return out_wav
