# Chatterbox-TTS-Extended — AMD-ready fork with voice training & a Discord voice bot

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

Everything else: Python 3.12 and FFmpeg are auto-installed via winget if
missing; steps already satisfied are skipped on re-runs; the install ends
with a hardware-appropriate verification (`verify.py`) that actually
exercises the GPU/provider rather than just importing packages.

AMD support covers the Ryzen AI MAX+ 395 "Strix Halo" (gfx1151) and recent
Radeon RX 7000/9000 GPUs — see AMD's
[compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/windows/windows_compatibility.html).
An up-to-date Adrenalin driver and ~15 GB of disk are required for that
branch.

## Running

Right-click `run_amd.ps1` → **Run with PowerShell** (the name is historical —
it launches the app on every hardware branch), or from a terminal:

```powershell
.\run_amd.ps1
```

The web page opens within a few seconds; the TTS model loads in the
background. A loading bar at the top of the page shows what is happening
("Loading TTS engine", "Compiling GPU kernels") with elapsed time, turns
green when everything is warmed (~25 s cold), then hides itself. Requests
made before that simply wait for the load to finish.

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

- `GET /health` — `{"ok": true, "loaded_voice": ...}`
- `GET /voices` — list trained voices
- `POST /tts` `{"text", "voice", "exaggeration", "cfg_weight", "temperature",
  "seed"}` — returns a loudness-matched 24 kHz WAV

One voice model is resident at a time; requesting another voice swaps it.
Cold first request ~30 s (model load), warm requests ~11 s per sentence on a
Ryzen AI MAX+ 395.

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

- **Background noise in output?** Enable pyrnnoise denoising (runs before
  Auto-Editor and normalization).
- **Out of VRAM or slow?** Lower parallel workers, pick a smaller Whisper
  model, reduce candidates.
- **Artifacts?** Increase candidates/retries, adjust Auto-Editor
  threshold/margin, refine sound-word replacements.
- **Choppy audio?** Increase Auto-Editor margin; lower threshold.
- **Reproducibility:** set a fixed seed.

Feedback and contributions: open an issue or pull request.
