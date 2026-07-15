# Chatterbox-TTS-Extended — ONNX fork addition with voice training & a Discord voice bot

A *power-user TTS pipeline* for advanced single and batch speech synthesis,
voice conversion, and artifact-reduced audio generation, based on
[Chatterbox-TTS](https://github.com/resemble-ai/chatterbox) via
[petermg/Chatterbox-TTS-Extended](https://github.com/petermg/Chatterbox-TTS-Extended).

**This fork adds:**

- **AMD GPU support on Windows** (ROCm PyTorch — full CUDA parity, see below)
- **ONNX Runtime engine** (WebGPU / Ryzen AI NPU / CPU) incl. chatterbox-turbo
- **Voice training**: fine-tune Chatterbox on your own recordings
- **Speaking-style (pacing) transfer** for trained voices
- **Pronunciation picker** for heteronyms (wind/wind, read/read, ...), IME-style
- **Live TTS tab** — speaks while generating, streamed to the browser
- **Headless TTS API server** — use trained voices from other programs
- **Discord voice bot** — talk to a trained voice in a voice channel (DAVE/E2EE-ready)
- **Audiobook reader** — turns a PDF ebook into a multi-voice audiobook: an LLM
  reads the book, works out who speaks every line, and each character gets their
  own trained voice; the reader highlights the words as they are spoken and
  hooks into SumatraPDF
- Fast startup: web page in seconds, model loads behind a self-hiding loading bar

Upstream features (all preserved): multi-file input & batch output, candidate
generation & Whisper/faster-whisper validation, rich audio post-processing
(pyrnnoise denoising, Auto-Editor, FFmpeg normalization), voice conversion
tab, persistent settings UI, parallel processing.

---

## Table of Contents

- [Installation](#installation)
- [Running](#running)
- [The web app](#the-web-app)
  - [Generation features (upstream)](#generation-features-upstream)
  - [Live TTS](#live-tts)
  - [Inference engines (PyTorch / ONNX)](#inference-engines-pytorch--onnx)
  - [Voice Training](#voice-training)
  - [Pronunciation picker](#pronunciation-picker)
  - [Voice Conversion](#voice-conversion)
- [Headless TTS API server](#headless-tts-api-server)
- [Discord voice bot](#discord-voice-bot)
- [Audiobook reader (PDF → audiobook)](#audiobook-reader-pdf--audiobook)
- [AMD implementation notes](#amd-implementation-notes)
- [Tips & troubleshooting](#tips--troubleshooting)

---

## Installation

One installer for every machine — right-click [install.ps1](install.ps1) →
**Run with PowerShell** (or from a terminal: `.\install.ps1`).

It **detects your graphics hardware** and installs the matching stack
(override with `.\install.ps1 -Hardware amd|nvidia|intel|cpu`):

| Detected | What gets installed |
|---|---|
| AMD (Radeon / Ryzen AI) | ROCm 7.2.1 SDK + ROCm PyTorch 2.9.1 from `repo.radeon.com`, OpenNMT's ROCm CTranslate2 wheel (GPU faster-whisper), `onnxruntime-webgpu` |
| NVIDIA | CUDA 12.8 PyTorch, PyPI CTranslate2 (its Windows wheel is CUDA-capable; needs cuDNN 9 for GPU whisper), `onnxruntime-gpu` |
| Intel | CPU PyTorch + `onnxruntime-openvino` with the paired `openvino` runtime (Intel GPU/NPU acceleration through the ONNX engine) |
| none of the above | CPU builds of everything |

**Fail-fast probe:** before committing to the multi-GB hardware stack, the
installer first installs the *smallest* builds of the three hardware-
sensitive runtimes — CPU torch (~200 MB), PyPI CTranslate2, PyPI
onnxruntime (~15 MB) — and runs real computations on them. A broken
Python/pip/network/VC-runtime setup fails in minutes, not after 15 GB.

Everything else: Python 3.12 and FFmpeg are auto-installed if missing
(via winget, or straight from python.org / gyan.dev on machines without
winget — with progress bars either way); steps already satisfied are
skipped on re-runs, and a `.venv-amd` left broken by a machine move or a
Python uninstall is detected and rebuilt automatically. The install ends
with a hardware-appropriate verification (`verify.py`) that actually
exercises the GPU/provider rather than just importing packages, then
offers to launch the app right away (`y`/`n`/`a` — `a` launches with
`--auto` so the browser opens itself).

Every install run is fully logged to `install.log` (pip output included),
and failure messages point there. The installer also strips Windows'
"downloaded from the internet" mark from the project's scripts, so
`run.ps1` doesn't trigger a security prompt on every launch.

AMD support covers the Ryzen AI MAX+ 395 "Strix Halo" (gfx1151) and recent
Radeon RX 7000/9000 GPUs — see AMD's
[compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/windows/windows_compatibility.html).
An up-to-date Adrenalin driver and ~15 GB of disk are required for that
branch.

## Running

Right-click `run.ps1` → **Run with PowerShell**, or from a terminal:

```powershell
./run.ps1 --auto
```

The webclient will be accessible within a few seconds by typing in
`localhost:7860` in your browser — with `--auto` a browser window opens
there automatically. The TTS model loads in the
background. A loading bar at the top of the page shows what is happening
("Loading TTS engine", "Compiling GPU kernels") with elapsed time, turns
green when everything is warmed (~25 s cold), then hides itself. Requests
made before that simply wait for the load to finish.

Other options are passed through to the app: `--host 0.0.0.0` (listen on
all interfaces), `--port 7861`, `--share` (public Gradio link). Every
launch is logged to `run.log` (overwritten per run); if the app exits
with an error, the script says so and points at the log.

---

## The web app

### Generation features (upstream)

| Feature | Notes |
| --- | --- |
| Text input | text box + multi-file `.txt` upload; merge into one audio or separate outputs per file |
| Reference audio | upload/record a sample; the engine mimics its style/timbre |
| Emotion exaggeration | 0 = flat, 1 = normal, 2 = exaggerated |
| CFG weight / pace | high = literal & monotone, low = expressive (and slower) |
| Temperature, seed | randomness control; fixed seed = reproducible output |
| Batching & chunking | sentence batching (~300 chars/chunk), smart-append of short sentences, recursive splitting of long ones, parallel workers |
| Text preprocessing | lowercase, whitespace normalization, "J.R.R." → "J R R", inline reference-number removal, sound-word remove/replace lists (`um`, `zzz=>sigh`) |
| Candidates & validation | N candidates per chunk, retries, per-candidate deterministic seeds; each candidate transcribed by Whisper/faster-whisper and compared to the intended text; fallback to longest transcript / best similarity |
| Post-processing | pyrnnoise (RNNoise) denoising → Auto-Editor silence/stutter trimming (threshold & margin in UI) → FFmpeg normalization (EBU R128 or peak) |
| Export | WAV / MP3 320k / FLAC; timestamped, seed-tagged filenames; per-output `.settings.json`/`.csv` artifacts |
| Persistent settings | UI state saved/restored automatically; import/export as JSON |
| Watermarking | disabled in this fork (upstream option) |

### Live TTS

Speaks text as it is generated instead of writing files: sentences are
synthesized on the GPU and streamed straight to the browser; playback starts
after the first sentence (~4 s) while the rest generates. Reference voice,
exaggeration, temperature, CFG and seed supported; the Stop button cancels
mid-stream. (Measured: ~2.5 s of audio per ~2.8 s of compute per sentence on
the PyTorch/ROCm engine — the stream keeps pace with playback.)

### Inference engines (PyTorch / ONNX)

Selectable at the top of the page:

- **PyTorch (CUDA / ROCm GPU)** — the default engine.
- **ONNX Runtime (NPU / OpenVINO / WebGPU / CPU)** — Providers are probed in order **VitisAI (Ryzen AI NPU) →
  OpenVINO (Intel CPU/GPU/NPU) → WebGPU (DirectX 12) → CPU**, dropping down
  automatically when a component fails validation on the actual hardware.
  Two models:
  - **chatterbox-turbo** (default): GPT-2-based 350M model, 1-step distilled
    vocoder, ~37 tok/s on the GPU — faster than realtime; supports tags like
    `[laugh]`; CFG/exaggeration sliders ignored.
  - **chatterbox**: the full 0.5B model with CFG and exaggeration, ~0.5×
    realtime on the ONNX path.

  The engine validates every component at load (the WebGPU provider silently
  miscomputes the fp32 conditional decoder, and turbo's LM overflows in fp16
  — both detected and remapped automatically, e.g. turbo fp16 → q4). Seeds
  are honored. Models (~2–3 GB) download from Hugging Face on first use and
  cache in `%USERPROFILE%\.cache\huggingface`.

**Intel machines (OpenVINO):** the installer's Intel branch sets this up
automatically (`onnxruntime-openvino` + the paired `openvino` runtime —
the versions must match: 1.24.x ↔ openvino 2025.4.*; a mismatched pair
loads but silently falls back to CPU, which the engine detects and
reports). Device selection is automatic (NPU > GPU > CPU from the devices
OpenVINO actually reports; OpenVINO's `AUTO:` list hard-fails on absent
devices, so the engine builds it from `Core().available_devices`).
Override with the `CHATTERBOX_OPENVINO_DEVICE` env var (`CPU` / `GPU` /
`GPU.1` / `NPU` / `AUTO:...`). Ops OpenVINO can't run (e.g.
GroupQueryAttention) partition to the CPU provider automatically, and the
usual load-time self-tests still validate every component. Verified on
ORT 1.24.1 + OpenVINO 2025.4.1 (CPU device; Intel GPU/NPU paths use the
same AUTO mechanism but were not hardware-tested).

Voice Conversion always uses the PyTorch engine.

### Voice Training

Fine-tunes Chatterbox on your own recordings and saves the result as a
selectable voice under `voices/<name>/`.

- **Dataset prep**: drop in audio files; they are split on silence into
  1–14.5 s utterances and transcribed with faster-whisper. Full transcripts
  are written to `datasets/<name>/metadata.csv` — review/edit before
  training if you want corrections.
- **Loudness**: each trained voice stores the speech loudness of its training
  audio (`voices/<name>/loudness.json`) and generation output is gain-matched
  to it, so trained voices come out as loud as the original recordings.
- **Speaking style (pacing) transfer**: give it clips of someone speaking at
  the pace you want and it rebuilds the dataset around how humans actually
  slow down — pauses between phrases are rescaled to the style clip's
  pause-to-speech ratio while the words themselves are stretched at most
  ±15 % (preventing the "slow-motion" artifact), then fine-tunes a copy of
  the voice (`<name>-paced`). 10 epochs shifted overall pace by ~13–18 % in
  testing with unchanged articulation and word-perfect Whisper transcripts.
  At generation time a lower CFG weight (~0.3) slows delivery further and
  stacks with a pace-trained voice.

### Pronunciation picker

Under the text box on the TTS and Live TTS tabs. **Scan** finds the next
heteronym (wind, read, lead, tear, bass, ...), offers IPA-labeled choices
like an IME, and **Apply** substitutes a forcing respelling (e.g. wind →
"wined"). Whisper validation canonicalizes respellings back to the real word
so accuracy checking still passes. The word list is user-editable in
`pronunciations.json`.

### Voice Conversion

Upload/record input audio and a target voice; get the same words in the
target voice. Long audio is split into overlapping chunks and recombined
with crossfades; pitch shift supported.

---

## Headless TTS API server

`run_tts_server.ps1` starts [tts_server.py](tts_server.py) on
`http://127.0.0.1:7861` — the trained voices without the web UI, for other
programs (the Discord bot uses this):

- `GET /health` — `{"ok": true, "resident": [...], "pinned": [...],
  "capacity": N, "stats": {...}, "loaded_voice": ...}`
- `GET /voices` — list trained voices
- `POST /tts` `{"text", "voice", "exaggeration", "cfg_weight", "temperature",
  "seed"}` — returns a loudness-matched 24 kHz WAV
- `POST /warm` `{"voice", "pin"}` — preload a voice into RAM in the background
- `POST /plan` `{"narrator", "voices": {name: count}}` — set up residency for a
  whole book at once (pin the narrator, preload the most-used voices)

**Voice router (multi-voice residency).** A book has many voices; loading one
cold takes ~5–10 s on a Ryzen AI MAX+ 395, so the server keeps several
resident at once instead of swapping on every line. The router:

- **pins the narrator** — loaded first and never evicted, since it speaks most
  of the book;
- **warms voices ahead of need** — the audiobook reader tells the server the
  book's cast (`/plan`) and the next voice it will need (`/warm`), so a
  character's model loads *while the previous line is still playing* — first
  use never stalls;
- **sizes itself to RAM** — capacity is derived from free memory (≈12 on a
  96 GB machine, so a whole book's cast usually stays resident with no
  churn); override with `CHATTERBOX_TTS_MAX_VOICES`;
- **evicts least-recently-used** unpinned voices only when over capacity.

Warm requests then run ~11 s per sentence with no load stall between speakers.

---

## Discord voice bot

`discord-bot/` contains a bot that sits in a Discord voice channel, listens,
and talks back with one of your trained voices:

- speech → **faster-whisper** (GPU) → text
- text → **LM Studio** (OpenAI-compatible local API) → reply
- reply → **headless TTS server** → spoken in the channel

**Reply rules:** with exactly one human in the channel it replies to
everything; with more people it replies only when it hears its name or an
alias (`bot_name` / `name_aliases`) — e.g. *"Bobbie, didn't I hear you say
xyz"* or *"...isn't that right, Bobbie?"*. Everything heard goes into its
conversation memory either way, so it has context when finally addressed.

### One-time setup

1. Create the bot at https://discord.com/developers/applications →
   *New Application*:
   - *Bot* tab → *Reset Token* → put the token in `discord-bot/config.json`
     (copy `config.example.json`; the real config is gitignored) or set the
     `DISCORD_BOT_TOKEN` environment variable.
   - Enable **Message Content Intent** (Privileged Gateway Intents).
   - Untick **Public Bot** if only you should be able to install it (also
     set *Installation → Install Link* to *None*).
   - *OAuth2 → URL Generator*: scope `bot`; permissions *View Channels*,
     *Send Messages*, *Connect*, *Speak*. Open the URL to invite it.
2. LM Studio: start the local server with a model loaded (*Developer* tab,
   or `lms server start` + `lms load <model> -y`). Put its URL in
   `config.json` → `lm_studio_url`.

### Running

1. `run_tts_server.ps1` (this folder)
2. LM Studio local server with a model loaded
3. `discord-bot\run_bot.ps1`
4. If `auto_join_channel_id` is set the bot joins that voice channel by
   itself; otherwise join a voice channel and type `!join`. Talk. First
   exchange is slower (whisper loads); after that expect ~15–25 s per
   exchange (STT + LLM + TTS sharing one GPU).

### Commands

| Command | What it does |
|---|---|
| `!join` / `!leave` | join your voice channel / leave |
| `!say <text>` | speak text directly (TTS test) |
| `!voice <name>` | switch Chatterbox voice (`!voice` lists them) |
| `!reset` | clear conversation memory |

### Multiple named bots

Each bot = one Discord application/token + one config file:
`python bot.py --config mybot2.json`. All bots share the one TTS server (it
swaps voices on demand — simultaneous different-voice bots take turns).

### Testing without Discord

`python bot.py --selftest` exercises STT → trigger logic → LM Studio → TTS
end-to-end and writes `selftest_out.wav`.

### E2EE voice (DAVE) — why dave_recv_patch.py exists

Since March 2026 Discord requires the DAVE end-to-end-encryption protocol on
voice connections. discord.py 2.7 handles DAVE for *sending* audio, but the
`discord-ext-voice-recv` extension (listening) predates enforcement and
would crash on encrypted incoming frames. `dave_recv_patch.py` (imported
automatically by bot.py) decrypts incoming frames through the same `davey`
DAVE session and drops any frame that still can't be decoded instead of
killing the audio pipeline. Expect the bot to be deaf for the first few
seconds after joining while the encryption group finishes its handshake.

### Config reference (config.json)

| Key | Meaning |
|---|---|
| `bot_name`, `name_aliases` | words that count as "spoken to" (case-insensitive) |
| `tts_voice` | folder name under `voices/` |
| `tts_url`, `lm_studio_url` | the two local servers |
| `lm_model` | LM Studio model id; `""` = whatever is loaded |
| `persona` | system prompt; `{name}` is replaced with `bot_name` |
| `whisper_model` | faster-whisper size (`medium` accurate, `small` faster) |
| `utterance_silence_ms` | silence gap that ends an utterance (default 700) |
| `auto_join_channel_id` | voice channel ID to auto-join on startup (`null` = off) |
| `tts_exaggeration`, `tts_cfg_weight` | delivery knobs (lower cfg = slower pace) |

### Bot troubleshooting

- **`!join` times out** — usually a just-restarted bot whose old Discord
  session hasn't expired (up to a minute). The bot retries once by itself;
  if it still fails, wait a minute and `!join` again.
- **Commands ignored** — the message must *start* with the prefix (`!join`,
  not `hello !join`), and Message Content Intent must be enabled.
- **Joins but never replies** — check the console: `[HEARD]` lines show what
  it transcribed; none means no audio is arriving (see DAVE note);
  `[LLM]`/`[TTS]` errors mean LM Studio or the TTS server isn't reachable.
- **Doesn't react to its name** — whisper must transcribe the name as you
  say it; add likely spellings to `name_aliases`.

---

## Audiobook reader (PDF → audiobook)

Turns a PDF ebook into a multi-voice audiobook: **open a PDF in SumatraPDF
and click Read Aloud** — it reads the whole book with a different trained
voice per character and highlights the words as they're spoken, inside
SumatraPDF's own window. Like a screen reader, but with a cast.

The Read Aloud button is a **setting**: it uses either the built-in Windows
TTS or the Chatterbox audiobook engine, your choice — no extra button, no
second window.

### One-time setup

1. **Build the patched SumatraPDF** — `.\build_sumatra.ps1` (needs VS 2022
   Build Tools; the script says so if they're missing). This produces
   `..\sumatrapdf\out\dbg64\SumatraPDF-dll.exe`, a SumatraPDF fork with an
   `Audiobook` settings section and the Read Aloud engine hook.
2. **Point SumatraPDF at this install** — `.\setup_audiobook.ps1`. It
   auto-detects this folder and its venv Python, finds
   `SumatraPDF-settings.txt`, and writes an `Audiobook` block with
   `UseChatterbox = true`. (Run `.\setup_audiobook.ps1 -Windows` to switch
   Read Aloud back to Windows TTS; or just edit the setting.)

That's it. Open any PDF in that SumatraPDF and press **Read Aloud** (the
speaker icon in the toolbar). Press it again to stop.

### What happens on click

SumatraPDF launches the headless engine ([audiobook/engine.py](audiobook/engine.py)),
which:

1. **Starts the TTS server** if it isn't already running.
2. **Works out who speaks each line.** An LLM (LM Studio, any
   OpenAI-compatible local server) reads the book once, attributes every
   quoted line to a character, and discovers the cast. The result is cached
   in `audiobook/cache/` by file hash, so it's a one-time cost per book.
   **No LM Studio? It still works** — the whole book is read in the
   narrator voice.
3. **Casts voices.** The narrator (most of the book) gets one voice;
   characters get the others. A saved casting is reused; anything unset is
   auto-assigned and saved so it's stable.
4. **Reads everything** — narration in the narrator voice, quotes in their
   character voices — while the [voice router](#headless-tts-api-server)
   keeps the narrator pinned and warms each character's voice *before* its
   first line, so switching speakers never stalls on a model load. The
   spoken words highlight in SumatraPDF and pages turn on their own.

The `Audiobook` settings (SumatraPDF-settings.txt): `UseChatterbox`,
`ChatterboxDir`, `PythonExe`, `TtsServerPort`, `LmStudioUrl`,
`NarratorVoice` (empty = first available voice).

### Making & tuning character voices — Voice Lab

Character voices are made and refined in **Voice Lab**, opened from the
helper window (`.\run_audiobook.ps1` → **Voice Lab…**):

- **Train a new voice from recordings** — the same VAD + Whisper + LoRA
  fine-tuning pipeline as the web app's Voice Training tab.
- **Wave match** — the tuner clones the *reference speaker's own passages*
  across a grid of generation settings and measures every candidate's
  waveform against the real recording: speaker-embedding similarity
  (chatterbox's own voice encoder), long-term spectrum (timbre), and pitch.
  The program picks the passages and the winning settings — not the ear of
  whoever is training — and draws the real-vs-clone wave traces overlaid;
  one click applies the winner to that character.
- **Test voice** — reads one or two of that character's actual lines so you
  hear the voice in context.

### Under the hood

The patched SumatraPDF adds two DDE commands the engine uses to paint the
highlight (same mechanism Sumatra uses for LaTeX forward search):

```
[AudiobookHighlight("<pdf>",<page>,"x0 y0 x1 y1;...")]   coordinates in PDF points
[AudiobookClear("<pdf>")]
```

(For a stock, unpatched SumatraPDF the helper window's **Add to SumatraPDF**
button still registers the reader as an external viewer under
**File → Open With** — the fallback before the fork existed.)

## AMD implementation notes

Every GPU code path works identically to an NVIDIA setup — ROCm PyTorch
implements the `torch.cuda` API, so the app reports e.g.
`Running on device: cuda (AMD Radeon(TM) 8060S Graphics, ROCm/HIP 7.2...)`.

- TTS generation, openai-whisper, and fine-tuning run on the AMD GPU
  unchanged (seeding, generators, SDPA attention, `empty_cache`).
- faster-whisper runs on the AMD GPU through OpenNMT's official ROCm
  CTranslate2 Windows wheels (v4.7.1+ ships native gfx1151 kernels), which
  load their GPU libraries from the pip-installed ROCm 7.2 SDK inside the
  venv. **Do not** substitute the standalone "HIP SDK" installer (7.1.1,
  incompatible DLL names).
- Unified-memory machines (Ryzen AI MAX+): "VRAM" is shared system RAM —
  raise the GPU memory allocation in BIOS/Adrenalin (e.g. 32 GB+) if you use
  large Whisper models alongside TTS.
- If Python aborts at startup with `OMP: Error #15`, set
  `KMP_DUPLICATE_LIB_OK=TRUE`.

---

## Tips & troubleshooting

- **Install failed?** The full log of the run is in `install.log` next to
  the script — include it when reporting an issue. App crashes land in
  `run.log` the same way.
- **"Do you want to run this software?" on every launch** — Windows marks
  browser-downloaded scripts. Run `install.ps1` once (it unblocks the
  project's scripts) or `Unblock-File .\run.ps1`.
- **Moved the folder to another PC?** Just run `install.ps1` there — it
  detects the broken venv and rebuilds everything that machine needs.
- **Background noise in output?** Enable pyrnnoise denoising (runs before
  Auto-Editor and normalization).
- **Out of VRAM or slow?** Lower parallel workers, pick a smaller Whisper
  model, reduce candidates.
- **Artifacts?** Increase candidates/retries, adjust Auto-Editor
  threshold/margin, refine sound-word replacements.
- **Choppy audio?** Increase Auto-Editor margin; lower threshold.
- **Reproducibility:** set a fixed seed.

Feedback and contributions: open an issue or pull request.
