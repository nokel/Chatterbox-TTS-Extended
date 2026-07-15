"""
Headless audiobook engine — the process SumatraPDF launches when Read Aloud
is set to Chatterbox (Audiobook.UseChatterbox).

No UI: given a PDF it makes sure the TTS server is running, works out who
speaks each line (or reads everything as the narrator if no local LLM is
reachable), casts voices, and reads the whole book aloud while driving the
in-window highlight back into SumatraPDF via the AudiobookHighlight DDE
command. SumatraPDF stops it by terminating the process (and clears its own
highlight).

    engine.py --pdf <file> --sumatra-exe <SumatraPDF.exe>
              [--tts-port 7861] [--lm-url http://127.0.0.1:11434]
              [--narrator <voice>]

Run indirectly through SumatraPDF's Read Aloud button, not by hand.
"""

import argparse
import os
import subprocess
import sys
import threading
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from audiobook import analyze, pdfbook, synth          # noqa: E402
from audiobook.sumatra import SumatraBridge             # noqa: E402

LOG_DIR = os.path.join(BASE_DIR, "audiobook", "cache")


def _log_path():
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, "engine.log")


class Log:
    def __init__(self):
        try:
            self.f = open(_log_path(), "a", encoding="utf-8")
        except Exception:
            self.f = None

    def __call__(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()
        try:
            print(line, flush=True)
        except Exception:
            pass


def ensure_server(url, log):
    """Return True once the TTS server answers; start it if needed."""
    if synth.tts_available(url):
        log("TTS server already running.")
        return True
    log("Starting TTS server...")
    creation = 0
    if os.name == "nt":
        creation = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    try:
        subprocess.Popen(
            [sys.executable, os.path.join(BASE_DIR, "tts_server.py")],
            cwd=BASE_DIR, creationflags=creation,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log(f"Could not launch TTS server: {e}")
        return False
    for _ in range(90):
        if synth.tts_available(url):
            log("TTS server is up.")
            return True
        time.sleep(2)
    log("TTS server did not come up in time.")
    return False


def build_casting(pdf, characters, server_voices, narrator_arg, log):
    """Load saved casting, fill in a narrator + a voice per character, save."""
    casting = pdfbook.load_casting(pdf)
    if not server_voices:
        return casting, None
    narrator = (narrator_arg
                or casting.get("narrator", {}).get("voice")
                or server_voices[0])
    if narrator not in server_voices:
        narrator = server_voices[0]
    casting.setdefault("narrator", {})["voice"] = narrator

    others = [v for v in server_voices if v != narrator] or [narrator]
    chars = sorted(characters or {}, key=lambda c: -characters[c])
    i = 0
    for c in chars:
        slot = casting.setdefault("characters", {}).setdefault(c, {})
        if not slot.get("voice"):
            slot["voice"] = others[i % len(others)]
            i += 1
    pdfbook.save_casting(pdf, casting)
    cast_str = ", ".join(f"{c}->{casting['characters'][c]['voice']}"
                         for c in chars)
    log(f"cast: narrator={narrator}"
        + (f"; {cast_str}" if cast_str else ""))
    return casting, narrator


def get_units(pdf, lm_url, log):
    """Analyzed units, or a whole-book narration fallback without an LLM."""
    cached = pdfbook.load_analysis(pdf)
    if cached and cached.get("version") == 1:
        log("Using cached analysis.")
        return cached["units"], cached.get("characters", {})
    try:
        a = analyze.analyze_book(pdf, cfg={"lm_studio_url": lm_url},
                                 log=log, use_cache=True)
        return a["units"], a.get("characters", {})
    except analyze.AnalyzeError as e:
        log(f"No local LLM ({e}); reading the whole book as the narrator.")
        units = pdfbook.extract_units(pdf)
        for u in units:
            u["speaker"] = analyze.NARRATOR
        return units, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--sumatra-exe", default=None)
    ap.add_argument("--tts-port", type=int, default=7861)
    ap.add_argument("--lm-url", default="http://127.0.0.1:11434")
    ap.add_argument("--narrator", default=None)
    args = ap.parse_args()

    log = Log()
    log(f"=== engine start: {os.path.basename(args.pdf)} ===")
    url = f"http://127.0.0.1:{args.tts_port}"
    pdf = os.path.abspath(args.pdf)

    if not ensure_server(url, log):
        log("Aborting: no TTS server.")
        return 2

    units, characters = get_units(pdf, args.lm_url, log)
    voices = synth.list_server_voices(url)
    log(f"{len(units)} units; server voices: {voices}")
    if not voices:
        log("No trained voices available - nothing to read with.")
        return 3

    casting, narrator = build_casting(pdf, characters, voices,
                                      args.narrator, log)

    bridge = SumatraBridge(exe=args.sumatra_exe, log=log)
    log(f"sumatra bridge: {bridge.exe} (available={bridge.available})")

    done = threading.Event()

    def on_start(u):
        if bridge.available:
            bridge.highlight(pdf, u)

    def on_finish(err):
        if bridge.available:
            bridge.clear(pdf)
        log(f"playback finished: {err}")
        done.set()

    player = synth.Player(casting, tts_url=url,
                          on_unit_start=on_start, on_finish=on_finish,
                          log=log)
    log("reading...")
    player.play(units)
    # block until playback ends; SumatraPDF stops us by terminating the process
    while not done.wait(timeout=1.0):
        pass
    log("=== engine done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
