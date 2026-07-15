"""
Synthesis + playback engine for the audiobook reader.

Speaks analyzed units through the headless Chatterbox TTS server
(tts_server.py, port 7861), prefetching a few units ahead in a worker
thread while the current one plays through sounddevice. Emits callbacks on
unit start/finish so the UI can highlight the words being spoken.

Casting maps character names to voices:
    {
      "narrator": {"voice": "Trump-paced", "exaggeration": 0.5,
                   "cfg_weight": 0.5, "temperature": 0.8, "seed": null},
      "characters": {"Keeper": {"voice": "...", ...}, ...}
    }
Characters without an entry (and "Unknown") fall back to the narrator voice.
"""

import io
import queue
import threading
import time

import numpy as np
import requests
import soundfile as sf

DEFAULT_TTS_URL = "http://127.0.0.1:7861"
DEFAULT_PARAMS = {"exaggeration": 0.5, "cfg_weight": 0.5,
                  "temperature": 0.8, "seed": None}
GAP_UNIT_SEC = 0.18       # pause between units
GAP_PARA_SEC = 0.45       # extra pause at a paragraph change
PREFETCH = 3              # units synthesized ahead of playback


def voice_params(casting, speaker):
    """Resolve a speaker name -> synthesis params via the casting table."""
    chars = (casting or {}).get("characters", {})
    entry = chars.get(speaker) or (casting or {}).get("narrator") or {}
    p = {**DEFAULT_PARAMS, **entry}
    return p if p.get("voice") else None


def tts_available(tts_url=DEFAULT_TTS_URL):
    try:
        r = requests.get(tts_url.rstrip("/") + "/health", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def list_server_voices(tts_url=DEFAULT_TTS_URL):
    try:
        r = requests.get(tts_url.rstrip("/") + "/voices", timeout=5)
        return r.json() if r.status_code == 200 else []
    except requests.RequestException:
        return []


def synth_unit(text, params, tts_url=DEFAULT_TTS_URL, timeout=600):
    """Synthesize one unit -> (float32 mono waveform, sample_rate)."""
    body = {"text": text, "voice": params["voice"],
            "exaggeration": params["exaggeration"],
            "cfg_weight": params["cfg_weight"],
            "temperature": params["temperature"]}
    if params.get("seed") is not None:
        body["seed"] = params["seed"]
    r = requests.post(tts_url.rstrip("/") + "/tts", json=body,
                      timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"TTS HTTP {r.status_code}: {r.text[:200]}")
    x, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def warm_voice(voice, pin=False, tts_url=DEFAULT_TTS_URL):
    """Ask the server to preload a voice into RAM (non-blocking on the server)."""
    try:
        requests.post(tts_url.rstrip("/") + "/warm",
                      json={"voice": voice, "pin": pin}, timeout=10)
    except requests.RequestException:
        pass


def plan_book(narrator_voice, voice_counts, tts_url=DEFAULT_TTS_URL):
    """Tell the server a book's voice usage so it pins the narrator and
    preloads the most-used voices. voice_counts: {voice_name: n_units}."""
    try:
        requests.post(tts_url.rstrip("/") + "/plan",
                      json={"narrator": narrator_voice,
                            "voices": voice_counts}, timeout=15)
    except requests.RequestException:
        pass


def unit_voice(casting, speaker):
    """The voice name a speaker resolves to, or None."""
    p = voice_params(casting, speaker)
    return p["voice"] if p else None


class Player:
    """Plays a sequence of units with prefetch, pause/resume and stop.

    Callbacks (called from worker threads - marshal into your UI loop):
        on_unit_start(unit)   about to speak this unit
        on_unit_end(unit)     finished speaking it
        on_finish(error)      playback ended (error is None or Exception)
    """

    def __init__(self, casting, tts_url=DEFAULT_TTS_URL,
                 on_unit_start=None, on_unit_end=None, on_finish=None,
                 log=None):
        self.casting = casting
        self.tts_url = tts_url
        self.on_unit_start = on_unit_start or (lambda u: None)
        self.on_unit_end = on_unit_end or (lambda u: None)
        self.on_finish = on_finish or (lambda e: None)
        self.log = log or (lambda *_: None)
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._threads = []
        self._q = None

    # ------------------------------------------------------------ control --
    @property
    def playing(self):
        return any(t.is_alive() for t in self._threads)

    @property
    def paused(self):
        return self._paused.is_set()

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    def stop(self):
        self._stop.set()
        self._paused.clear()
        for t in self._threads:
            t.join(timeout=10)
        self._threads = []

    def play(self, units, start_index=0):
        """Start speaking units[start_index:] in the background."""
        self.stop()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._q = queue.Queue(maxsize=PREFETCH)
        todo = units[start_index:]
        self._send_plan(todo)
        tp = threading.Thread(target=self._prefetch, args=(todo,),
                              daemon=True, name="ab-prefetch")
        tb = threading.Thread(target=self._playback, daemon=True,
                              name="ab-playback")
        self._threads = [tp, tb]
        tp.start()
        tb.start()

    def _send_plan(self, units):
        """Tell the router the book's voice usage: pin the narrator, preload
        the most-used voices. Fire-and-forget in a thread so we don't wait."""
        narrator = (self.casting or {}).get("narrator", {}).get("voice")
        if not narrator:
            return
        counts = {}
        for u in units:
            v = unit_voice(self.casting, u["speaker"])
            if v:
                counts[v] = counts.get(v, 0) + 1
        threading.Thread(
            target=plan_book, args=(narrator, counts, self.tts_url),
            daemon=True, name="ab-plan").start()

    # ------------------------------------------------------------ workers --
    def _prefetch(self, units):
        try:
            for i, u in enumerate(units):
                if self._stop.is_set():
                    return
                # warm-ahead: start loading the voice a few units out so a
                # first-time voice is resident before we need to synth it
                # (belt-and-braces on top of the book plan, for the case
                # where capacity is too small to hold every voice at once)
                ahead = units[i + PREFETCH] if i + PREFETCH < len(units) else None
                if ahead is not None:
                    v = unit_voice(self.casting, ahead["speaker"])
                    if v:
                        warm_voice(v, tts_url=self.tts_url)
                params = voice_params(self.casting, u["speaker"])
                if params is None:
                    self._q.put(("error", u, RuntimeError(
                        f"No voice cast for '{u['speaker']}' and no "
                        "narrator voice set")))
                    return
                try:
                    t0 = time.time()
                    x, sr = synth_unit(u["text"], params, self.tts_url)
                    self.log(f"synth {time.time() - t0:4.1f}s "
                             f"[{u['speaker']}] {u['text'][:50]}")
                except Exception as e:
                    self._q.put(("error", u, e))
                    return
                while not self._stop.is_set():
                    try:
                        self._q.put(("wav", u, (x, sr)), timeout=0.25)
                        break
                    except queue.Full:
                        continue
            self._q.put(("eof", None, None))
        except Exception as e:  # pragma: no cover - safety net
            self._q.put(("error", None, e))

    def _playback(self):
        import sounddevice as sd
        error = None
        last_para = None
        stream = None
        try:
            while not self._stop.is_set():
                try:
                    kind, unit, payload = self._q.get(timeout=0.25)
                except queue.Empty:
                    continue
                if kind == "eof":
                    break
                if kind == "error":
                    error = payload
                    break
                x, sr = payload
                if stream is None or stream.samplerate != sr:
                    if stream is not None:
                        stream.stop()
                        stream.close()
                    stream = sd.OutputStream(samplerate=sr, channels=1,
                                             dtype="float32")
                    stream.start()
                if last_para is not None:
                    gap = GAP_PARA_SEC if unit["para"] != last_para \
                        else GAP_UNIT_SEC
                    self._write(stream, np.zeros(int(sr * gap), np.float32))
                last_para = unit["para"]
                self.on_unit_start(unit)
                self._write(stream, x)
                if not self._stop.is_set():
                    self.on_unit_end(unit)
        except Exception as e:
            error = e
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
            self._stop.set()  # also winds down the prefetcher
            self.on_finish(error)

    def _write(self, stream, x):
        """Write audio in small blocks, honoring pause/stop."""
        block = 2048
        for i in range(0, len(x), block):
            while self._paused.is_set() and not self._stop.is_set():
                time.sleep(0.05)
            if self._stop.is_set():
                return
            stream.write(np.ascontiguousarray(x[i:i + block]).reshape(-1, 1))
