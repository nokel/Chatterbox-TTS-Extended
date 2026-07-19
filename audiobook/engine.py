"""
Audiobook engine — the process SumatraPDF launches when Read Aloud is set to
Chatterbox (Audiobook.UseChatterbox).

It is a resident service, not a playback job. It starts, makes sure the TTS
server is up, loads the cast, and then waits. Reading, analysing (working out
who speaks each line) and casting are all requests that arrive on its local
control API. It exits only when SumatraPDF does, or on /quit.

That split exists because both of the slow steps are opt-in. Analysis costs
minutes on a real book, so it happens when asked for rather than on the way to
reading; and the Characters panel needs the cast without wanting a word read
aloud, so opening it starts the engine in this idle state.

While reading it drives the in-window highlight back into SumatraPDF via the
AudiobookHighlight DDE command.

The engine has no UI of its own: SumatraPDF is the UI. Its Read Aloud commands
(Pause / Continue / Stop Reading) and its Characters panel drive playback and
casting through the control API (audiobook/control.py).

    engine.py --pdf <file> --sumatra-exe <SumatraPDF.exe>
              [--tts-port 7861] [--lm-url http://127.0.0.1:11434]
              [--narrator <voice>] [--control-port 7862]
              [--play] [--from-start] [--start-text <selection>]

Without --play it starts idle. Run it through SumatraPDF, not by hand.
"""

import argparse
import os
import subprocess
import sys
import threading
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from audiobook import analyze, control, pdfbook, synth  # noqa: E402
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


def watch_parent(pid, on_gone, log):
    """Exit when SumatraPDF does.

    We have no window of our own, so an orphaned engine would keep reading the
    book aloud with no way to stop it short of Task Manager. SumatraPDF stops us
    on its own when it closes cleanly; this also covers it being killed or
    crashing.
    """
    if not pid:
        return

    def alive():
        try:
            import psutil
            return psutil.pid_exists(pid)
        except Exception:
            pass
        # fall back to the Win32 API: a process we can't open is gone
        try:
            import ctypes
            k = ctypes.windll.kernel32
            h = k.OpenProcess(0x1000, False, pid)   # QUERY_LIMITED_INFORMATION
            if not h:
                return False
            code = ctypes.c_ulong()
            ok = k.GetExitCodeProcess(h, ctypes.byref(code))
            k.CloseHandle(h)
            return bool(ok) and code.value == 259   # STILL_ACTIVE
        except Exception:
            return True      # can't tell: don't kill ourselves on a guess

    def loop():
        while True:
            time.sleep(1.0)
            if not alive():
                log(f"SumatraPDF (pid {pid}) is gone - stopping")
                on_gone()
                return

    threading.Thread(target=loop, daemon=True, name="ab-parent-watch").start()


def _kill_stale_server(url, log):
    """Kill a TTS server that isn't the one we want.

    Whatever holds the port serves every request, so a server started from the
    wrong interpreter (system Python has CPU-only torch and no onnxruntime)
    makes synthesis 10x slower - and we'd never notice, because it answers
    /health perfectly well. Ours reports its engine; anything that doesn't is
    stale and has to go.
    """
    port = None
    try:
        port = int(url.rsplit(":", 1)[1].rstrip("/"))
    except (ValueError, IndexError):
        return False
    try:
        import psutil
    except ImportError:
        log("stale TTS server on the port, and psutil isn't available to "
            "clear it - stop it by hand")
        return False
    killed = False
    for c in psutil.net_connections(kind="inet"):
        if not c.laddr or c.laddr.port != port or c.status != psutil.CONN_LISTEN:
            continue
        if not c.pid:
            continue
        try:
            p = psutil.Process(c.pid)
            log(f"killing stale TTS server (pid {c.pid}, {p.exe()})")
            p.kill()
            p.wait(timeout=10)
            killed = True
        except Exception as e:
            log(f"could not kill pid {c.pid}: {e}")
    return killed


def ensure_server(url, log):
    """Return True once *our* TTS server answers; start it if needed."""
    health = synth.tts_health(url)
    if health is not None:
        # Don't just accept anything that answers: a server from the wrong
        # interpreter is CPU-only and 10x slower.
        if health.get("engine"):
            log(f"TTS server already running (engine={health['engine']}).")
            return True
        log("A TTS server is running but doesn't report its engine - it's "
            "stale (likely started from the wrong Python). Replacing it.")
        _kill_stale_server(url, log)
        time.sleep(1.0)
    log(f"Starting TTS server ({sys.executable})...")
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
        health = synth.tts_health(url)
        if health is not None and health.get("engine"):
            log(f"TTS server is up (engine={health['engine']}).")
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
    # "Unknown" isn't a character - it's every line nobody could name, or
    # hasn't been read yet. Giving it a voice of its own would put all of them
    # in one wrong voice; left uncast it falls through to the narrator, which
    # is the intended sound. (A part-analysed book is mostly Unknown, so this
    # is the common case, not an edge one.)
    skip = {analyze.UNKNOWN, analyze.NARRATOR, "narrator", None, ""}
    chars = sorted([c for c in (characters or {}) if c not in skip],
                   key=lambda c: -characters[c])
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


def load_units(pdf, log):
    """The book's lines: from a cached analysis if there is one.

    Returns (units, characters, analyzed, meta). Analysis is never run here -
    it costs minutes, and the caller may only want the cast or a narrator
    read. Un-analyzed, every line is the narrator's, which is exactly what a
    book with no LLM available sounds like anyway.
    """
    cached = pdfbook.load_analysis(pdf)
    if cached and cached.get("version") == 1:
        # may be a partial run (stopped, or only part of the book asked for):
        # usable as it stands, and Analyse will carry on from it
        meta = {"complete": cached.get("complete", True),
                "lines_read": cached.get("lines_read", 0),
                "lines_total": cached.get("lines_total", 0)}
        if meta["complete"]:
            log("Using cached analysis.")
        else:
            log(f"Using partial analysis ({meta['lines_read']}/"
                f"{meta['lines_total']} lines read).")
        return cached["units"], cached.get("characters", {}), True, meta
    units = pdfbook.extract_units(pdf)
    for u in units:
        u["speaker"] = analyze.NARRATOR
    log(f"not analyzed yet: {len(units)} lines, narrator only")
    return units, {}, False, {"complete": True, "lines_read": 0,
                              "lines_total": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--sumatra-exe", default=None)
    ap.add_argument("--tts-port", type=int, default=7861)
    ap.add_argument("--lm-url", default="http://127.0.0.1:11434")
    ap.add_argument("--lm-urls", default="",
                    help="comma-separated extra LLM endpoints to analyse with, "
                         "e.g. http://192.168.1.20:11434 - the book's chunks "
                         "are shared out across all of them")
    ap.add_argument("--lm-model", default="",
                    help="LLM for speaker attribution; empty = pick it")
    ap.add_argument("--narrator", default=None)
    ap.add_argument("--analyzer", default="llm", choices=["llm", "booknlp"],
                    help="who works out the speakers: the per-chunk LLM, or "
                         "BookNLP (local, no LLM server needed)")
    ap.add_argument("--control-port", type=int,
                    default=control.DEFAULT_CONTROL_PORT,
                    help="port SumatraPDF drives playback/casting through")
    ap.add_argument("--parent-pid", type=int, default=0,
                    help="stop when this process (SumatraPDF) exits")
    ap.add_argument("--from-start", action="store_true",
                    help="ignore saved progress and read from the beginning")
    ap.add_argument("--play", action="store_true",
                    help="start reading at once; without it the engine idles")
    ap.add_argument("--start-text", default=None,
                    help="start at the line holding this text (a selection)")
    ap.add_argument("--start-page", type=int, default=None,
                    help="page the --start-text selection is on (1-based)")
    args = ap.parse_args()

    log = Log()
    log(f"=== engine start: {os.path.basename(args.pdf)} ===")
    url = f"http://127.0.0.1:{args.tts_port}"
    pdf = os.path.abspath(args.pdf)
    extra_lm_urls = [u.strip() for u in (args.lm_urls or "").split(",")
                     if u.strip()]
    if extra_lm_urls:
        log(f"extra analysis endpoints: {', '.join(extra_lm_urls)}")

    if not ensure_server(url, log):
        log("Aborting: no TTS server.")
        return 2

    units, characters, analyzed, ameta = load_units(pdf, log)
    voices = synth.list_server_voices(url)
    log(f"{len(units)} units; server voices: {voices}")
    if not voices:
        log("No trained voices available - nothing to read with.")
        return 3

    casting, narrator = build_casting(pdf, characters, voices,
                                      args.narrator, log)

    bridge = SumatraBridge(exe=args.sumatra_exe, log=log)
    log(f"sumatra bridge: {bridge.exe} (available={bridge.available})")

    state = None

    def on_start(u):
        if bridge.available:
            bridge.highlight(pdf, u)
        if state:
            state.unit_started(u)

    def on_finish(err):
        # A skip/restart stops the Player and starts it again, which lands here
        # too - that isn't the end of the book, so don't act on it.
        if state and state.seeking.is_set():
            return
        if bridge.available:
            bridge.clear(pdf)
        log(f"playback finished: {err}")
        if state:
            state.playback_finished(err)
            # read to the end: forget the resume point, or the next Read Aloud
            # would "resume" at the last line
            if err is None and not state.stop_requested.is_set():
                pdfbook.clear_progress(pdf)
        # the engine stays up: the panel may still be open, and the model is
        # loaded. Only /quit or SumatraPDF closing ends it.

    player = synth.Player(casting, tts_url=url,
                          on_unit_start=on_start, on_finish=on_finish,
                          log=log)

    def do_analyze(set_status, should_stop, percent, force=False, analyzer=None):
        """Run speaker attribution, then re-cast. Called from /analyze.

        force throws the cached answer away and reads the book again. Without
        it, Re-analyse cannot do anything at all: a finished analysis is
        returned from the cache before a single chunk is sent, so the button
        would read the file back and report success having changed nothing.

        analyzer picks "llm" or "booknlp" for this run; None keeps the value
        the engine was launched with, so the panel can switch without a restart.
        """
        which = (analyzer or args.analyzer or "llm").lower()
        # the live list: the panel can add or drop a computer between runs
        cfg = {"lm_studio_url": args.lm_url,
               "lm_urls": list(state.extra_lm_urls) if state else extra_lm_urls}
        chosen = state.lm_model if state else ""
        if chosen:
            cfg["lm_model"] = chosen
        set_status("starting")
        if force:
            log("re-analysing from scratch (ignoring the cached analysis)")
        progress_cb = lambda done, total, msg: set_status(
            f"{msg} ({done}/{total})" if total else msg)
        if which == "booknlp":
            # BookNLP is a single local pass - no LLM server, no per-chunk
            # requests, so percent/should_stop don't apply and it always
            # completes. The rest of do_analyze doesn't change: same dict.
            from audiobook import booknlp_bridge
            a = booknlp_bridge.analyze_book(
                pdf, cfg=cfg, log=log, use_cache=not force,
                progress=progress_cb)
        else:
            a = analyze.analyze_book(
                pdf, cfg=cfg, log=log, use_cache=not force,
                should_stop=should_stop, percent=percent,
                progress=progress_cb)
        u = a["units"]
        c = a.get("characters", {})
        set_status("casting voices")
        cast, _ = build_casting(pdf, c, synth.list_server_voices(url),
                                args.narrator, log)
        meta = {"complete": a.get("complete", True),
                "lines_read": a.get("lines_read", 0),
                "lines_total": a.get("lines_total", 0)}
        return u, c, cast, meta

    # Carry on from where reading was stopped, like Continue Reading does for
    # Windows TTS. "Start Reading From Top" passes --from-start to override it.
    start_index = 0
    if args.from_start:
        pdfbook.clear_progress(pdf)
    else:
        start_index = pdfbook.load_progress(pdf)
        if start_index >= len(units):
            start_index = 0                     # finished last time: start over
        if start_index:
            log(f"resuming at unit {start_index}/{len(units)}")

    # SumatraPDF is the UI. It drives us through this local control API:
    # its Read Aloud commands and its Characters panel. We show no windows.
    state = control.EngineState(pdf, units, characters, casting, player,
                                url, log, start_index=start_index,
                                analyzed=analyzed, analyze_fn=do_analyze,
                                lm_url=args.lm_url, lm_model=args.lm_model,
                                extra_lm_urls=extra_lm_urls)
    state.analyze_complete = ameta["complete"]
    state.lines_read = ameta["lines_read"]
    state.lines_total = ameta["lines_total"]
    if analyzed and not ameta["complete"]:
        state.analyze_status = "partial"
    # keeps the model / computer lists current without /state ever waiting on
    # a network call (an absent machine costs the whole timeout)
    state.start_probe()

    try:
        control.start(state, port=args.control_port)
        log(f"control API on 127.0.0.1:{args.control_port}")
    except OSError as e:
        log(f"control API unavailable on port {args.control_port}: {e}")
        return 4

    # never outlive SumatraPDF: we have no UI, so an orphan would keep reading
    # with no way to stop it
    watch_parent(args.parent_pid, state.quit_requested.set, log)

    # Load the narrator's model now rather than on the first line. This is the
    # "initialise without reading" the panel wants: by the time anything is
    # asked for, the model is resident.
    if narrator:
        threading.Thread(target=synth.warm_voice, args=(narrator, True, url),
                         daemon=True, name="ab-warm").start()

    if args.play:
        state.play_request(unit=start_index, from_start=args.from_start,
                           text=args.start_text, page=args.start_page,
                           analyze_first=True)
    else:
        log("idle: waiting for SumatraPDF (panel open, nothing reading)")

    state.quit_requested.wait()
    log("quit requested")
    player.stop()
    if bridge.available:
        bridge.clear(pdf)
    log("=== engine done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
