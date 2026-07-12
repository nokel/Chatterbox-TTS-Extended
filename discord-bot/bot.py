# bot.py — Discord voice-chat bot with a Chatterbox voice.
#
# Sits in a voice channel, listens, and talks back:
#   * speech -> faster-whisper (GPU) -> text
#   * text   -> LM Studio (OpenAI-compatible API) -> reply
#   * reply  -> headless Chatterbox TTS server -> spoken in the channel
#
# Reply rules (from config):
#   * exactly ONE human in the channel  -> replies to everything they say
#   * more than one human               -> replies only when its name (or an
#     alias) is heard, e.g. "Trump, didn't you say..." / "isn't that right, Trump?"
#   Everything heard goes into the conversation history either way, so the bot
#   has context for "didn't I hear you say ..." questions.
#
# Text commands: !join !leave !say <text> !voice <name> !reset
#
# Run:      python bot.py --config config.json
# Selftest: python bot.py --selftest [some.wav]   (no Discord needed)
#
# Each bot instance = one config file + one Discord bot token, so you can run
# several named bots at once (different names, voices, personas).

import argparse
import asyncio
import io
import json
import os
import re
import sys
import tempfile
import threading
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import aiohttp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

DISCORD_SR = 48000          # what voice packets arrive as (16-bit stereo)
WHISPER_SR = 16000
MIN_UTT_SEC = 0.4           # ignore blips shorter than this
MAX_UTT_SEC = 30.0          # force-flush monologues
HISTORY_MAX = 24            # rolling conversation messages kept for the LLM


# ---------------------------------------------------------------- config ----

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("bot_name", "Bot")
    cfg.setdefault("name_aliases", [])
    cfg.setdefault("tts_voice", "Trump-paced")
    cfg.setdefault("tts_url", "http://127.0.0.1:7861")
    cfg.setdefault("lm_studio_url", "http://127.0.0.1:11434")
    cfg.setdefault("lm_model", "")
    cfg.setdefault("persona",
                   "You are {name}, a voice-chat companion. Reply in 1-3 short "
                   "spoken sentences. Never use emoji, markdown, or stage "
                   "directions - your words are read aloud exactly as written.")
    cfg.setdefault("whisper_model", "medium")
    cfg.setdefault("command_prefix", "!")
    cfg.setdefault("utterance_silence_ms", 700)
    cfg.setdefault("tts_exaggeration", 0.5)
    cfg.setdefault("tts_cfg_weight", 0.5)
    tok = os.environ.get("DISCORD_BOT_TOKEN")
    if tok:
        cfg["token"] = tok
    return cfg


# ------------------------------------------------------------------ audio ---

def pcm48s_to_wav16k(pcm_bytes):
    """48 kHz 16-bit stereo PCM -> 16 kHz mono float32 for whisper."""
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if len(x) % 2:
        x = x[:-1]
    mono = x.reshape(-1, 2).mean(axis=1)
    n = (len(mono) // 3) * 3
    return mono[:n].reshape(-1, 3).mean(axis=1)  # 48k -> 16k box decimation


class SpeechToText:
    """faster-whisper wrapper; one GPU transcription at a time."""

    def __init__(self, model_name):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            print(f"[STT] Loading faster-whisper '{self.model_name}'...")
            try:
                self._model = WhisperModel(self.model_name, device="cuda",
                                           compute_type="float16")
            except Exception as e:
                print(f"[STT] GPU load failed ({e}); using CPU int8")
                self._model = WhisperModel(self.model_name, device="cpu",
                                           compute_type="int8")
            print("[STT] Ready")
        return self._model

    def transcribe(self, audio16k):
        with self._lock:
            model = self._load()
            segs, _ = model.transcribe(audio16k, language="en",
                                       vad_filter=True, beam_size=1)
            return " ".join(s.text.strip() for s in segs).strip()


# ------------------------------------------------------------ LLM client ----

class Brain:
    """Talks to LM Studio; keeps a rolling voice-chat transcript."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.history = []   # list of {"role", "content"}
        self._lock = asyncio.Lock()

    def system_prompt(self):
        persona = self.cfg["persona"].replace("{name}", self.cfg["bot_name"])
        return (persona + "\nYou are in a live voice channel. Lines from "
                "humans are prefixed with the speaker's name.")

    def hear(self, speaker, text):
        self.history.append({"role": "user", "content": f"{speaker}: {text}"})
        del self.history[:-HISTORY_MAX]

    def said(self, text):
        self.history.append({"role": "assistant", "content": text})
        del self.history[:-HISTORY_MAX]

    def reset(self):
        self.history.clear()

    async def reply(self, session):
        async with self._lock:
            payload = {
                "messages": [{"role": "system", "content": self.system_prompt()}]
                            + list(self.history),
                "temperature": 0.8,
                "max_tokens": 220,
            }
            if self.cfg["lm_model"]:
                payload["model"] = self.cfg["lm_model"]
            url = self.cfg["lm_studio_url"].rstrip("/") + "/v1/chat/completions"
            async with session.post(url, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=120)) as r:
                if r.status != 200:
                    raise RuntimeError(f"LM Studio HTTP {r.status}: "
                                       f"{(await r.text())[:200]}")
                data = await r.json()
            text = data["choices"][0]["message"]["content"].strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # reasoning models
            text = re.sub(r"[*_#`]|\[.*?\]|\(laughs?\)", " ", text)      # not speakable
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                self.said(text)
            return text


# ----------------------------------------------------------------- TTS ------

async def synthesize(session, cfg, text):
    """Ask the headless Chatterbox server for a WAV of `text`."""
    url = cfg["tts_url"].rstrip("/") + "/tts"
    body = {"text": text, "voice": cfg["tts_voice"],
            "exaggeration": cfg["tts_exaggeration"],
            "cfg_weight": cfg["tts_cfg_weight"]}
    async with session.post(url, json=body,
                            timeout=aiohttp.ClientTimeout(total=300)) as r:
        if r.status != 200:
            raise RuntimeError(f"TTS server HTTP {r.status}: "
                               f"{(await r.text())[:200]}")
        return await r.read()


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()] or [text.strip()]


# ------------------------------------------------------- trigger logic ------

def name_patterns(cfg):
    names = [cfg["bot_name"]] + list(cfg["name_aliases"])
    return [re.compile(r"\b" + re.escape(n.lower()) + r"\b") for n in names if n]


def is_addressed(text, patterns):
    low = re.sub(r"[^a-z0-9' ]", " ", text.lower())
    return any(p.search(low) for p in patterns)


def should_reply(text, patterns, human_count):
    if human_count <= 1:
        return True
    return is_addressed(text, patterns)


# ------------------------------------------------------------- the bot ------

def run_bot(cfg):
    import discord
    from discord.ext import commands, voice_recv

    import dave_recv_patch
    dave_recv_patch.apply()

    intents = discord.Intents.default()
    intents.message_content = True
    intents.voice_states = True
    bot = commands.Bot(command_prefix=cfg["command_prefix"], intents=intents)

    stt = SpeechToText(cfg["whisper_model"])
    brain = Brain(cfg)
    patterns = name_patterns(cfg)
    utt_queue: asyncio.Queue = asyncio.Queue(maxsize=4)
    speak_queue: asyncio.Queue = asyncio.Queue()
    state = {"vc": None, "session": None}

    # -- capture: per-user buffers, flushed after a silence gap ------------
    buffers = {}          # user_id -> {"name", "chunks", "last", "started"}
    buf_lock = threading.Lock()

    def on_voice(user, data):
        if user is None or getattr(user, "bot", False):
            return
        with buf_lock:
            b = buffers.setdefault(user.id, {
                "name": getattr(user, "display_name", user.name),
                "chunks": [], "last": 0.0, "started": 0.0})
            if not b["chunks"]:
                b["started"] = time.monotonic()
            b["chunks"].append(data.pcm)
            b["last"] = time.monotonic()

    async def flusher():
        gap = cfg["utterance_silence_ms"] / 1000.0
        while True:
            await asyncio.sleep(0.15)
            now = time.monotonic()
            done = []
            with buf_lock:
                for uid, b in buffers.items():
                    if not b["chunks"]:
                        continue
                    if (now - b["last"] > gap
                            or now - b["started"] > MAX_UTT_SEC):
                        done.append((b["name"], b"".join(b["chunks"])))
                        b["chunks"] = []
            for name, pcm in done:
                dur = len(pcm) / (DISCORD_SR * 4)
                if dur < MIN_UTT_SEC:
                    continue
                try:
                    utt_queue.put_nowait((name, pcm))
                except asyncio.QueueFull:
                    print("[BOT] utterance queue full; dropping one")

    # -- understand + decide + reply ---------------------------------------
    async def thinker():
        loop = asyncio.get_running_loop()
        while True:
            name, pcm = await utt_queue.get()
            audio = pcm48s_to_wav16k(pcm)
            text = await loop.run_in_executor(None, stt.transcribe, audio)
            if not text:
                continue
            vc = state["vc"]
            humans = 0
            if vc and vc.channel:
                humans = sum(1 for m in vc.channel.members if not m.bot)
            print(f"[HEARD] {name}: {text}  (humans={humans})")
            brain.hear(name, text)
            if not should_reply(text, patterns, humans):
                continue
            try:
                reply = await brain.reply(state["session"])
            except Exception as e:
                print(f"[LLM] {e}")
                continue
            if not reply:
                continue
            print(f"[REPLY] {reply}")
            for sent in split_sentences(reply):
                await speak_queue.put(sent)

    # -- speak: fetch TTS per sentence, play sequentially -------------------
    async def speaker():
        while True:
            sent = await speak_queue.get()
            vc = state["vc"]
            if vc is None or not vc.is_connected():
                continue
            try:
                wav = await synthesize(state["session"], cfg, sent)
            except Exception as e:
                print(f"[TTS] {e}")
                continue
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            f.write(wav)
            f.close()
            done = asyncio.Event()
            loop = asyncio.get_running_loop()

            def after(err, _p=f.name):
                if err:
                    print(f"[PLAY] {err}")
                loop.call_soon_threadsafe(done.set)

            while vc.is_playing():
                await asyncio.sleep(0.1)
            vc.play(discord.FFmpegPCMAudio(f.name), after=after)
            await done.wait()
            try:
                os.unlink(f.name)
            except OSError:
                pass

    # -- connect helper (used by !join and auto-join) ------------------------
    async def connect_and_listen(ch, notify=None):
        if state["vc"] is not None:
            await state["vc"].disconnect(force=True)
        for attempt in (1, 2):
            try:
                vc = await ch.connect(cls=voice_recv.VoiceRecvClient, timeout=45)
                break
            except (asyncio.TimeoutError, TimeoutError):
                # A recently killed bot session can steal the voice events;
                # Discord expires it within a minute, so wait and retry once.
                print(f"[CMD] voice connect timed out (attempt {attempt})")
                if attempt == 2:
                    if notify:
                        await notify("Voice connection timed out twice - wait "
                                     "a minute and try !join again.")
                    return None
                if notify:
                    await notify("Voice is taking a moment, retrying...")
                await asyncio.sleep(10)
        vc.listen(voice_recv.BasicSink(on_voice))
        state["vc"] = vc
        print(f"[BOT] Listening in '{ch.name}'")
        return vc

    # -- commands ------------------------------------------------------------
    @bot.event
    async def on_ready():
        state["session"] = aiohttp.ClientSession()
        asyncio.create_task(flusher())
        asyncio.create_task(thinker())
        asyncio.create_task(speaker())
        print(f"[BOT] Logged in as {bot.user} — name trigger: "
              f"{[cfg['bot_name']] + cfg['name_aliases']}, "
              f"voice: {cfg['tts_voice']}")
        print(f"[BOT] Use {cfg['command_prefix']}join in a text channel "
              f"while you are in a voice channel.")
        auto_id = cfg.get("auto_join_channel_id")
        if auto_id:
            ch = bot.get_channel(int(auto_id))
            if ch is not None:
                print(f"[BOT] Auto-joining voice channel '{ch.name}'")
                await connect_and_listen(ch)

    @bot.event
    async def on_message(msg):
        if msg.author != bot.user:
            print(f"[MSG] #{msg.channel} {msg.author.display_name}: {msg.content!r}")
        await bot.process_commands(msg)

    @bot.event
    async def on_command_error(ctx, error):
        import traceback
        print(f"[CMDERR] {ctx.command}: {error!r}")
        traceback.print_exception(type(error), error, error.__traceback__,
                                  file=sys.stdout)

    @bot.command()
    async def join(ctx):
        print(f"[CMD] join from {ctx.author.display_name}; "
              f"voice state: {ctx.author.voice!r}")
        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.send("Join a voice channel first, then use !join.")
            return
        ch = ctx.author.voice.channel
        vc = await connect_and_listen(ch, notify=ctx.send)
        if vc is not None:
            await ctx.send(f"Listening in **{ch.name}** as **{cfg['bot_name']}** "
                           f"(voice: {cfg['tts_voice']}).")

    @bot.command()
    async def leave(ctx):
        if state["vc"] is not None:
            await state["vc"].disconnect(force=True)
            state["vc"] = None
            await ctx.send("Left the voice channel.")

    @bot.command()
    async def say(ctx, *, text: str):
        for sent in split_sentences(text):
            await speak_queue.put(sent)
        await ctx.send("Speaking.")

    @bot.command()
    async def voice(ctx, name: str = ""):
        async with state["session"].get(
                cfg["tts_url"].rstrip("/") + "/voices") as r:
            available = await r.json()
        if not name:
            await ctx.send(f"Current: **{cfg['tts_voice']}**. "
                           f"Available: {', '.join(available)}")
            return
        if name not in available:
            await ctx.send(f"No voice '{name}'. Available: {', '.join(available)}")
            return
        cfg["tts_voice"] = name
        await ctx.send(f"Voice switched to **{name}**.")

    @bot.command()
    async def reset(ctx):
        brain.reset()
        await ctx.send("Conversation history cleared.")

    token = cfg.get("token", "")
    if not token or "PUT-YOUR" in token:
        print("ERROR: no bot token. Put it in config.json (\"token\") or set "
              "the DISCORD_BOT_TOKEN environment variable.")
        sys.exit(1)
    import logging
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="[discord] %(levelname)s %(name)s: %(message)s")
    logging.getLogger("discord.gateway").setLevel(logging.WARNING)
    bot.run(token, log_handler=None)


# ------------------------------------------------------------- selftest -----

def selftest(cfg, wav_path):
    """Exercise STT -> trigger -> LLM -> TTS without Discord."""
    import soundfile as sf
    ok = True

    print(f"== 1. STT on {os.path.basename(wav_path)}")
    x, sr = sf.read(wav_path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    # resample to 16k the same crude way the bot does (via 48k path when possible)
    if sr != WHISPER_SR:
        idx = np.linspace(0, len(x) - 1, int(len(x) * WHISPER_SR / sr))
        x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    stt = SpeechToText(cfg["whisper_model"])
    text = stt.transcribe(x)
    print(f"   heard: {text!r}")
    if not text:
        print("   FAIL: empty transcript"); ok = False

    print("== 2. trigger logic")
    pats = name_patterns(cfg)
    cases = [
        (f"hey {cfg['bot_name']} what do you think", 3, True),
        ("nice weather today", 3, False),
        ("nice weather today", 1, True),
        (f"isn't that right {cfg['bot_name']}", 2, True),
    ]
    for t, humans, want in cases:
        got = should_reply(t, pats, humans)
        print(f"   humans={humans} {t!r} -> {got} (want {want})")
        if got != want:
            ok = False

    async def net():
        nonlocal ok
        async with aiohttp.ClientSession() as session:
            print("== 3. LM Studio")
            brain = Brain(cfg)
            brain.hear("Tester", text or "Say hello in one sentence.")
            try:
                reply = await brain.reply(session)
                print(f"   reply: {reply!r}")
            except Exception as e:
                print(f"   SKIPPED (LM Studio not reachable: {e})")
                reply = "This is a self test of the text to speech pipeline."
            print("== 4. TTS server")
            try:
                wav = await synthesize(session, cfg, reply[:300])
                out = os.path.join(HERE, "selftest_out.wav")
                with open(out, "wb") as f:
                    f.write(wav)
                data, osr = sf.read(io.BytesIO(wav))
                print(f"   OK: {len(data)/osr:.1f}s audio -> {out}")
            except Exception as e:
                print(f"   FAIL: {e}"); ok = False

    asyncio.run(net())
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ----------------------------------------------------------------- main -----

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--selftest", nargs="?", const="", metavar="WAV")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.selftest is not None:
        wav = args.selftest or os.path.join(
            os.path.dirname(HERE), "Chatterbox-TTS-Extended-main",
            "voices", "Trump-paced-test", "sample_original_pace.wav")
        sys.exit(selftest(cfg, wav))
    run_bot(cfg)
