"""Local control API for the resident audiobook engine.

SumatraPDF is the UI: its Read Aloud commands and its Characters panel drive
the engine through this. The engine has no windows of its own.

The engine is *resident*, not a playback job: it starts, loads the TTS model,
and then sits idle until asked to do something. Opening the Characters panel
starts it this way, so the panel works before anything has been read. Analysis
(working out who speaks each line) is a request too, not something that happens
on the way to reading -- it costs minutes on a real book, so it happens when
asked for, not by surprise.

Bound to 127.0.0.1 only.

    GET  /state            what's loaded, what's playing, and the full cast
    POST /analyze          work out who speaks each line (background);
                           {"percent": 25} does only the first quarter
    POST /analyze_stop     wind analysis up early, keeping what it has read
    POST /model            {"model": "..."} which LLM analysis uses ("" = auto)
    POST /endpoints        {"urls": [...]} other computers to analyse with
    POST /scan             look for LLM servers on the local network
    POST /scan_stop        give up on the scan early
    POST /play             start reading; {"unit"|"from_start"|"text"|"page",
                                           "analyze_first"}
    POST /pause            pause playback
    POST /resume           resume (play)
    POST /stop             stop, remembering the place; stays resident
    POST /restart          play from the beginning
    POST /prev             back one sentence
    POST /next             skip a sentence
    POST /page             skip a page; {"dir": -1} goes back one
    POST /cast             {"character": "Keeper", "voice": "fallout_4"}
    POST /test             {"character": "Keeper"}  speak one of their lines
    POST /quit             shut the engine down
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from audiobook import analyze, discover, pdfbook, synth

DEFAULT_CONTROL_PORT = 7862


# Quotes are dropped before matching, not because of the text but because of
# how the selection reaches us: it travels on a Windows command line, where a
# literal " has to be mangled to survive. Since novels are mostly dialogue,
# stripping quotes from both sides is what lets the two ends agree.
_DROP = str.maketrans("", "", "\"'“”‘’\\")


def _norm(s):
    """Whitespace/case/quote-insensitive form, for matching a selection."""
    return " ".join((s or "").translate(_DROP).split()).lower()


class EngineState:
    """Shared state between the player callbacks and the control API."""

    def __init__(self, pdf, units, characters, casting, player, tts_url, log,
                 start_index=0, analyzed=False, analyze_fn=None,
                 lm_url=None, lm_model="", extra_lm_urls=None):
        self.pdf = pdf
        self.units = units
        self.characters = characters or {}      # name -> line count
        self.casting = casting or {}
        self.player = player
        self.tts_url = tts_url
        self.log = log or (lambda *_: None)
        self.start_index = start_index          # where this session began
        self.index = 0                          # units spoken this session
        self.total = len(units)
        self.speaker = None
        self.voice = None
        self.finished = False
        self.error = None
        self._lock = threading.Lock()
        self._testing = False
        self.stop_requested = threading.Event()
        # A seek restarts the Player, and restarting it fires on_finish exactly
        # as reaching the end of the book does. Without this the engine would
        # treat every skip as "finished".
        self.seeking = threading.Event()

        # analysis is a request, not a startup step
        self.analyzed = analyzed
        self.analyzing = False
        self.analyze_error = None
        self.analyze_status = "done" if analyzed else ""
        self._analyze_fn = analyze_fn
        # Stop is not cancel: the run winds up and keeps what it has read.
        self.analyze_stop = threading.Event()
        self.analyze_percent = 100
        self.analyze_force = False
        self.analyze_analyzer = None    # None = the engine's launch default
        # a part-done analysis the next run can continue from
        self.analyze_complete = True
        self.lines_read = 0
        self.lines_total = 0

        # which local LLM works out who speaks each line. Empty = decide at
        # analysis time (the only one there is, or the one already loaded).
        self.lm_url = lm_url
        self.lm_model = lm_model or ""
        self._models = []
        self._models_at = 0.0
        # other computers to share the analysis with
        self.extra_lm_urls = list(extra_lm_urls or [])
        self._eps = []
        self._eps_at = 0.0
        # looking for other computers on the network
        self._scanning = False
        self._scan_stop = threading.Event()
        self._scan_found = []
        self._scan_done = 0
        self._scan_total = 0
        self._scan_msg = ""

        # the engine outlives playback; only this ends it
        self.quit_requested = threading.Event()

    def unit_started(self, unit):
        with self._lock:
            self.index += 1
            self.speaker = unit.get("speaker") or "Narrator"
            self.voice = synth.unit_voice(self.casting, self.speaker)

    # -- analysis --------------------------------------------------------
    def _set_analyze_status(self, msg):
        with self._lock:
            self.analyze_status = str(msg)

    def _run_analysis(self):
        """Analyze now, on the calling thread. Returns True on success."""
        with self._lock:
            if self.analyzing:
                return False
            self.analyzing = True
            self.analyze_error = None
            self.analyze_status = "starting"
        self.analyze_stop.clear()
        try:
            units, characters, casting, meta = self._analyze_fn(
                self._set_analyze_status, self.analyze_stop.is_set,
                self.analyze_percent, self.analyze_force, self.analyze_analyzer)
            with self._lock:
                self.units = units
                self.characters = characters or {}
                self.casting = casting or self.casting
                self.total = len(units)
                self.analyzed = True
                self.analyze_complete = bool(meta.get("complete", True))
                self.lines_read = int(meta.get("lines_read", 0))
                self.lines_total = int(meta.get("lines_total", 0))
                self.analyze_status = "done" if self.analyze_complete else "partial"
            # the Player reads the cast when it picks a voice per unit
            self.player.casting = self.casting
            self.log(f"analysis done: {len(self.characters)} characters")
            return True
        except Exception as e:
            # the panel shows this on one line, so it can't carry a traceback
            # or a pretty-printed JSON envelope
            with self._lock:
                self.analyze_error = " ".join(str(e).split())[:200]
                self.analyze_status = "failed"
            self.log(f"analysis failed: {e}")
            return False
        finally:
            with self._lock:
                self.analyzing = False

    def start_probe(self):
        """Keep the model/computer lists fresh in the background.

        /state is polled about once a second by a panel that asks for it on
        its UI thread, so /state must never wait on a network call. Probing an
        absent machine costs the full timeout - do that here, off to one side,
        and let /state answer instantly with the last known answer.
        """
        def loop():
            while not self.quit_requested.is_set():
                try:
                    self._probe_once()
                except Exception as e:
                    self.log(f"probe failed: {e!r}")
                self.quit_requested.wait(15.0)

        threading.Thread(target=loop, daemon=True, name="ab-probe").start()

    def _probe_once(self):
        local = (self.lm_url or "").strip().rstrip("/")
        models = []
        try:
            models = [m["id"] for m in analyze.list_models(local, timeout=3)]
        except Exception:
            pass

        out, seen = [], []
        for u in [local] + list(self.extra_lm_urls):
            u = (u or "").strip().rstrip("/")
            if not u or u in seen:
                continue
            seen.append(u)
            try:
                ms = analyze.list_models(u, timeout=3)
            except Exception:
                ms = []
            # Whatever analysis would actually use, worked out the same way it
            # works it out - otherwise the panel names one model and the run
            # uses another. On another computer that is the loaded one, not
            # the preferred one (see analyze.pick_remote_model).
            model = ""
            if ms:
                ids = [m["id"] for m in ms]
                loaded = [m["id"] for m in ms if m["loaded"]]
                if u != local:
                    model = analyze.pick_remote_model(ms, self.lm_model)
                elif self.lm_model in ids:
                    model = self.lm_model
                elif len(ids) == 1:
                    model = ids[0]
                elif len(loaded) == 1:
                    model = loaded[0]
                else:
                    model = ""      # ambiguous: the panel's dropdown asks
            out.append({"url": u, "ok": bool(ms), "model": model,
                        "local": u == local})
        with self._lock:
            if models:
                self._models = models
            self._eps = out
            self._models_at = self._eps_at = time.time()

    def lm_models(self):
        """Chat models this machine's LLM server has (last known)."""
        with self._lock:
            return list(self._models)

    def endpoints(self):
        """Every computer analysis would use, and whether it answers (last
        known - the probe thread keeps this current).

        Reachability is the whole point of showing this: LM Studio listens on
        127.0.0.1 unless told otherwise, so the usual outcome of adding a
        second machine is silence. Better to say so than to let the reader
        wonder why nothing got faster.
        """
        with self._lock:
            return list(self._eps)

    def set_endpoints(self, urls):
        urls = [u.strip().rstrip("/") for u in (urls or []) if (u or "").strip()]
        with self._lock:
            self.extra_lm_urls = urls
            # show it straight away as "checking"; the probe fills in the truth
            local = (self.lm_url or "").strip().rstrip("/")
            known = {e["url"]: e for e in self._eps}
            self._eps = [known.get(u, {"url": u, "ok": False, "model": "",
                                       "local": u == local})
                         for u in [local] + urls]
        self.log(f"analysis computers: {', '.join(urls) or '(this one only)'}")
        # probe the new one now rather than at the next tick
        threading.Thread(target=self._probe_once, daemon=True,
                         name="ab-probe-now").start()

    # -- finding other computers ------------------------------------------
    def scan_network(self):
        """Look for LLM servers on this machine's network, in the background.

        Like analysis, this is polled for, not waited on: a /24 takes a few
        seconds, and /state is answered on the panel's UI thread.
        """
        with self._lock:
            if self._scanning:
                return False, "already scanning"
            self._scanning = True
            self._scan_found = []
            self._scan_done = 0
            self._scan_total = 0
            self._scan_msg = "looking..."
        self._scan_stop.clear()

        def work():
            try:
                port = 0
                tail = (self.lm_url or "").rsplit(":", 1)
                if len(tail) == 2 and tail[1].split("/")[0].isdigit():
                    port = int(tail[1].split("/")[0])
                nets = ", ".join(str(n) for n, _ in discover.subnets()) or "(none)"
                self.log(f"scanning for LLM servers on {nets}")

                def progress(done, total):
                    with self._lock:
                        self._scan_done, self._scan_total = done, total

                found = discover.scan(
                    ports=[port] if port else [],
                    skip=[self.lm_url] + list(self.extra_lm_urls),
                    on_progress=progress,
                    should_stop=self._scan_stop.is_set)
                fresh = [f for f in found if not f["known"]]
                with self._lock:
                    self._scan_found = found
                    self._scan_msg = (
                        "no other computers found" if not found else
                        f"found {len(found)}" +
                        (f" ({len(found) - len(fresh)} already added)"
                         if len(fresh) != len(found) else ""))
                self.log(f"scan done: {', '.join(f['url'] for f in found) or 'nothing'}")
            except Exception as e:
                with self._lock:
                    self._scan_msg = " ".join(str(e).split())[:120]
                self.log(f"scan failed: {e!r}")
            finally:
                with self._lock:
                    self._scanning = False

        threading.Thread(target=work, daemon=True, name="ab-scan").start()
        return True, "scanning"

    def stop_scan(self):
        self._scan_stop.set()
        return True, "stopping"

    def scan_state(self):
        with self._lock:
            return {"scanning": self._scanning, "done": self._scan_done,
                    "total": self._scan_total, "message": self._scan_msg,
                    "found": list(self._scan_found)}

    def analyze(self, percent=100, force=False, analyzer=None):
        """Start analysis in the background. The panel polls /state for it.

        force = read the book again rather than returning the cached answer.
        It is what Re-analyse means; without it that button is a no-op.
        analyzer = "llm" or "booknlp"; None keeps the engine's launch default.
        """
        if not self._analyze_fn:
            return False, "analysis not available"
        with self._lock:
            if self.analyzing:
                return False, "already analysing"
            self.analyze_percent = max(1, min(int(percent or 100), 100))
            self.analyze_force = bool(force)
            self.analyze_analyzer = analyzer or None
        threading.Thread(target=self._run_analysis, daemon=True,
                         name="ab-analyze").start()
        return True, "analysing"

    def stop_analysis(self):
        """Wind the run up early and keep what it has read."""
        if not self.analyzing:
            return False, "not analysing"
        self.analyze_stop.set()
        self._set_analyze_status("stopping - keeping what's been read")
        self.log("analysis stop requested")
        return True, "stopping"

    # -- finding a place -------------------------------------------------
    def find_unit_for_text(self, text, page=None):
        """The unit a selection starts in, or None.

        The selection rarely lines up with a unit: it can sit inside one, or
        span several. So try the head of the selection inside a unit first,
        then the first unit that falls inside the selection.
        """
        want = _norm(text)
        if not want:
            return None
        head = want[:60]
        with self._lock:
            units = list(self.units)
        for i, u in enumerate(units):
            if page is not None and u.get("page") != page:
                continue
            if head and head in _norm(u.get("text")):
                return i
        for i, u in enumerate(units):
            ut = _norm(u.get("text"))
            if ut and ut in want:
                return i
        if page is not None:
            for i, u in enumerate(units):
                if u.get("page") == page:
                    return i
        return None

    # -- playing ---------------------------------------------------------
    def play_from(self, unit=None):
        if unit is None:
            unit = self.start_index
        unit = max(0, min(int(unit), max(0, self.total - 1)))
        if self.player.playing:
            return self.seek(unit)
        with self._lock:
            self.start_index = unit
            self.index = 0
            self.finished = False
            self.error = None
            units = self.units
        self.stop_requested.clear()
        self.player.play(units, start_index=unit)
        self.log(f"playing from unit {unit}/{self.total}")
        return unit

    def play_request(self, unit=None, from_start=False, text=None, page=None,
                     analyze_first=False):
        """Handle POST /play off-thread, so a slow analysis doesn't block it."""
        def work():
            if analyze_first and not self.analyzed and self._analyze_fn:
                self._run_analysis()
            idx = unit
            if text:
                found = self.find_unit_for_text(text, page)
                if found is None:
                    self.log("selection didn't match any line; reading on")
                else:
                    self.log(f"selection -> unit {found}")
                    idx = found
            elif from_start:
                idx = 0
            self.play_from(idx)

        threading.Thread(target=work, daemon=True, name="ab-play").start()
        return True

    # -- seeking ---------------------------------------------------------
    #
    # The Player has no seek: it can only start at an index. So a seek is a
    # stop + replay from the new unit. Prefetched audio for the old position is
    # dropped, which is why a skip costs one synth (~1-2s) before it speaks.
    def current_unit(self):
        with self._lock:
            return max(0, self.start_index + self.index - 1)

    def seek(self, unit):
        unit = max(0, min(int(unit), max(0, self.total - 1)))
        was_paused = self.player.paused
        self.seeking.set()
        try:
            self.player.stop()
            with self._lock:
                self.start_index = unit
                self.index = 0
                self.finished = False
                self.error = None
                units = self.units
            self.player.play(units, start_index=unit)
        finally:
            self.seeking.clear()
        if was_paused:
            self.player.pause()
        self.log(f"seek -> unit {unit}/{self.total}")
        return unit

    def seek_relative(self, delta):
        return self.seek(self.current_unit() + delta)

    def seek_page(self, delta):
        """Jump to the first unit of the next/previous page."""
        cur = self.current_unit()
        page = self.units[cur].get("page", 0) if cur < len(self.units) else 0
        want = page + delta
        pages = [u.get("page", 0) for u in self.units]
        if delta > 0:
            for i, p in enumerate(pages):
                if p >= want:
                    return self.seek(i)
            return self.seek(self.total - 1)
        for i in range(len(pages) - 1, -1, -1):
            if pages[i] <= want:
                # rewind to that page's first unit
                first = i
                while first > 0 and pages[first - 1] == pages[i]:
                    first -= 1
                return self.seek(first)
        return self.seek(0)

    def save_position(self):
        """Remember the line being spoken, so Continue resumes there.

        Saves the *current* line (index-1), not the next one: the line was
        interrupted mid-sentence, so it should be read again rather than
        skipped.
        """
        with self._lock:
            unit = max(0, self.start_index + self.index - 1)
        pdfbook.save_progress(self.pdf, unit)
        self.log(f"stopped at unit {unit}; Continue will resume there")
        return unit

    def playback_finished(self, err):
        with self._lock:
            self.finished = True
            self.error = str(err) if err else None

    # -- cast ------------------------------------------------------------
    def voice_of(self, name):
        # /state must never raise: it's the panel's only view of us, and a
        # handler that dies mid-response just closes the connection, so the
        # panel goes blank with no clue why. A nameless speaker reads as the
        # narrator, like any uncast line.
        if not isinstance(name, str) or not name:
            return self.casting.get("narrator", {}).get("voice")
        if name.lower() == "narrator":
            return self.casting.get("narrator", {}).get("voice")
        return self.casting.get("characters", {}).get(name, {}).get("voice")

    def set_voice(self, name, voice):
        if name.lower() == "narrator":
            self.casting.setdefault("narrator", {})["voice"] = voice
        else:
            self.casting.setdefault("characters", {})\
                .setdefault(name, {})["voice"] = voice
        pdfbook.save_casting(self.pdf, self.casting)
        self.log(f"cast {name} -> {voice}")

    def sample_line(self, name):
        for u in self.units:
            sp = (u.get("speaker") or "Narrator")
            if name.lower() == "narrator":
                if sp.lower() in ("narrator", ""):
                    return u["text"]
            elif sp == name:
                return u["text"]
        return "The quick brown fox jumps over the lazy dog."

    def test_voice(self, name):
        """Speak one of the character's own lines.

        Pauses the reading and *leaves it paused*: testing a voice is not a
        request to carry on reading, and having the book resume by itself after
        the sample is jarring. The reading continues only when told to.
        """
        voice = self.voice_of(name)
        if not voice:
            return False, f"'{name}' has no voice assigned"
        if self._testing:
            return False, "already testing a voice"
        self._testing = True
        text = self.sample_line(name)[:220]
        if self.player.playing and not self.player.paused:
            self.player.pause()

        def work():
            try:
                params = synth.voice_params(self.casting, name) or {}
                params = {**synth.DEFAULT_PARAMS, **params, "voice": voice}
                x, sr = synth.synth_unit(text, params, self.tts_url)
                import sounddevice as sd
                sd.play(x, sr)
                sd.wait()
            except Exception as e:
                self.log(f"test voice failed: {e}")
            finally:
                self._testing = False

        threading.Thread(target=work, daemon=True, name="ab-test").start()
        return True, f"testing {name} ({voice})"

    def snapshot(self):
        # Nothing here may wait on the network: the panel asks for /state on
        # its UI thread about once a second, so a slow answer freezes the
        # reader's window. The LLM lists come from the probe thread; the TTS
        # server is local and answers in milliseconds.
        voices = synth.list_server_voices(self.tts_url)
        models = self.lm_models()
        eps = self.endpoints()
        scan = self.scan_state()
        with self._lock:
            # "Unknown" is not a character: it's the lines nobody could name,
            # plus (on a part-analysed book) every line not read yet - often
            # most of the book. Sorted by line count it would head the list
            # looking like the lead role, and there's nothing to cast it to
            # anyway: uncast, it reads as the narrator. The status line is
            # where "how much is analysed" belongs.
            cast = []
            for name in sorted(self.characters,
                               key=lambda c: -self.characters.get(c, 0)):
                if not isinstance(name, str) or not name:
                    continue
                if name in ("Unknown", "Narrator") or name.lower() == "narrator":
                    continue
                cast.append({"name": name,
                             "lines": self.characters.get(name, 0),
                             "voice": self.voice_of(name) or ""})
            return {
                # Which book this engine has. There's one engine per machine
                # (one control port, one LLM, one set of TTS models) but every
                # SumatraPDF window has a panel polling it, so each one has to
                # be able to tell whether this engine is the one for *its*
                # document - otherwise it shows someone else's cast.
                "pdf": self.pdf,
                "playing": bool(self.player.playing),
                "paused": bool(self.player.paused),
                "finished": self.finished,
                "error": self.error,
                "index": self.start_index + self.index,
                "total": self.total,
                "start_index": self.start_index,
                "speaker": self.speaker or "",
                "voice": self.voice or "",
                "narrator": self.voice_of("narrator") or "",
                "analyzed": bool(self.analyzed),
                "analyzing": bool(self.analyzing),
                "analyze_status": self.analyze_status or "",
                "analyze_error": self.analyze_error or "",
                "analyze_complete": bool(self.analyze_complete),
                "lines_read": self.lines_read,
                "lines_total": self.lines_total,
                "lm_model": self.lm_model or "",
                "lm_models": models,
                "endpoints": eps,
                "scan": scan,
                "characters": cast,
                "voices": voices,
            }


class _Handler(BaseHTTPRequestHandler):
    state: EngineState = None       # set on the server instance

    def log_message(self, *_a):     # keep the console clean
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self):
        st = self.server.state
        if self.path.startswith("/state"):
            # An exception here kills the handler mid-response, so the panel
            # sees a dropped connection and reports the engine as gone - the
            # real fault buried in a thread nobody reads. Answer, and say so.
            try:
                self._send(st.snapshot())
            except Exception as e:
                st.log(f"/state failed: {e!r}")
                self._send({"error": f"state failed: {e}"}, 500)
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        st = self.server.state
        path = self.path.split("?")[0]
        if path == "/analyze":
            b = self._body()
            ok, msg = st.analyze(percent=b.get("percent", 100),
                                 force=bool(b.get("force")),
                                 analyzer=b.get("analyzer"))
            self._send({"ok": ok, "message": msg}, 200 if ok else 409)
        elif path == "/analyze_stop":
            ok, msg = st.stop_analysis()
            self._send({"ok": ok, "message": msg}, 200 if ok else 409)
        elif path == "/model":
            b = self._body()
            st.lm_model = (b.get("model") or "").strip()
            st.log(f"analysis model: {st.lm_model or '(auto)'}")
            self._send({"ok": True, "model": st.lm_model})
        elif path == "/scan":
            ok, msg = st.scan_network()
            self._send({"ok": ok, "message": msg}, 200 if ok else 409)
        elif path == "/scan_stop":
            ok, msg = st.stop_scan()
            self._send({"ok": ok, "message": msg})
        elif path == "/endpoints":
            b = self._body()
            st.set_endpoints(b.get("urls") or [])
            self._send({"ok": True, "endpoints": st.endpoints()})
        elif path == "/play":
            b = self._body()
            st.play_request(unit=b.get("unit"),
                            from_start=bool(b.get("from_start")),
                            text=b.get("text"),
                            page=b.get("page"),
                            analyze_first=bool(b.get("analyze_first")))
            self._send({"ok": True})
        elif path == "/pause":
            st.player.pause()
            self._send({"ok": True, "paused": True})
        elif path == "/resume":
            st.player.resume()
            self._send({"ok": True, "paused": False})
        elif path == "/stop":
            # stop the reading, keep the engine: the Characters panel is still
            # open and the model is loaded, so quitting here would make the
            # next Read Aloud pay the whole startup again
            unit = st.save_position()     # so Continue picks up here
            st.stop_requested.set()
            threading.Thread(target=st.player.stop, daemon=True).start()
            self._send({"ok": True, "resume_unit": unit})
        elif path == "/quit":
            st.quit_requested.set()
            self._send({"ok": True})
        elif path == "/restart":            # |<<  play from the beginning
            st.seek(0)
            st.player.resume()
            self._send({"ok": True, "unit": 0})
        elif path == "/prev":               # <<   previous sentence
            u = st.seek_relative(-1)
            st.player.resume()
            self._send({"ok": True, "unit": u})
        elif path == "/next":               # >>   skip sentence
            u = st.seek_relative(+1)
            st.player.resume()
            self._send({"ok": True, "unit": u})
        elif path == "/page":               # >>>  skip page (dir=-1 back)
            b = self._body()
            u = st.seek_page(int(b.get("dir", 1)))
            st.player.resume()
            self._send({"ok": True, "unit": u})
        elif path == "/cast":
            b = self._body()
            name, voice = b.get("character"), b.get("voice")
            if not name:
                self._send({"error": "character required"}, 400)
                return
            st.set_voice(name, voice or None)
            self._send({"ok": True, "character": name, "voice": voice})
        elif path == "/test":
            b = self._body()
            name = b.get("character")
            if not name:
                self._send({"error": "character required"}, 400)
                return
            ok, msg = st.test_voice(name)
            self._send({"ok": ok, "message": msg}, 200 if ok else 409)
        else:
            self._send({"error": "not found"}, 404)


def start(state, port=DEFAULT_CONTROL_PORT):
    """Run the control API in a daemon thread. Returns the server."""
    srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    srv.state = state
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True,
                     name="ab-control").start()
    return srv
