"""
Voice training pipeline: dataset agent + training agent.

Dataset agent (prepare_voice_dataset): takes raw audio files, splits them
into utterances with silero-VAD, transcribes any utterance without supplied
text using the app's Whisper backend, and writes an LJSpeech-style dataset
(datasets/<voice>/wavs + metadata.csv).

Training agent (run_training): drives the vendored chatterbox-finetuning
toolkit (finetuning/) as a subprocess — LoRA fine-tune of the T3 transformer
on the ROCm GPU — streaming its log, then merges the adapter and assembles a
complete voice model folder (voices/<voice>/) that the PyTorch engine can
load directly.
"""

import os
import re
import csv
import json
import shutil
import subprocess
import sys
import threading
import time
import queue as _queue

import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
VOICES_DIR = os.path.join(BASE_DIR, "voices")
FINETUNE_DIR = os.path.join(BASE_DIR, "finetuning")
CONFIG_PATH = os.path.join(FINETUNE_DIR, "src", "config.py")
CONFIG_ORIG = CONFIG_PATH + ".orig"

DATASET_SR = 24000
MIN_UTT_SEC = 1.0
MAX_UTT_SEC = 14.0

_vad_model = None
_train_proc = None
_train_lock = threading.Lock()


def list_voices():
    """Trained voice model folders usable by the PyTorch engine."""
    if not os.path.isdir(VOICES_DIR):
        return []
    out = []
    for name in sorted(os.listdir(VOICES_DIR)):
        d = os.path.join(VOICES_DIR, name)
        if os.path.isfile(os.path.join(d, "t3_cfg.safetensors")):
            out.append(name)
    return out


def speech_rms_dbfs(x, sr):
    """Speech-level loudness of a waveform in dBFS: RMS over 20 ms frames,
    ignoring frames more than 40 dB below the loudest (pauses/silence), so
    the value tracks how loud the *speech* is, not how much silence there is.
    Returns None when the clip is too short or silent."""
    x = np.asarray(x, dtype=np.float64)
    n = max(1, int(sr * 0.02))
    if len(x) < n * 3:
        return None
    m = (len(x) // n) * n
    frames = np.sqrt(np.mean(x[:m].reshape(-1, n) ** 2, axis=1))
    frames = frames[frames > 0]
    if not len(frames):
        return None
    speech = frames[frames > frames.max() * 10 ** (-40 / 20)]
    if not len(speech):
        return None
    return 20 * np.log10(float(np.mean(speech)) + 1e-12)


def get_voice_loudness(voice_name):
    """Target speech loudness (dBFS, see speech_rms_dbfs) of a trained
    voice's original training audio. Measured once and cached in
    voices/<voice>/loudness.json; generation matches its output level to
    this so voices are as loud as the recordings they were trained on."""
    voice_dir = os.path.join(VOICES_DIR, voice_name)
    jpath = os.path.join(voice_dir, "loudness.json")
    try:
        if os.path.isfile(jpath):
            with open(jpath, encoding="utf-8") as f:
                return float(json.load(f)["speech_rms_dbfs"])
    except Exception:
        pass

    import librosa
    srcs = [os.path.join(voice_dir, "reference.wav")]
    wav_dir = os.path.join(DATASETS_DIR, voice_name, "wavs")
    if os.path.isdir(wav_dir):
        srcs += [os.path.join(wav_dir, f)
                 for f in sorted(os.listdir(wav_dir))[:10]
                 if f.endswith(".wav")]
    vals = []
    for s in srcs:
        if not os.path.isfile(s):
            continue
        try:
            a, sr = librosa.load(s, sr=None, mono=True)
        except Exception:
            continue
        v = speech_rms_dbfs(a, sr)
        if v is not None:
            vals.append(v)
        if s.endswith("reference.wav") and vals:
            break  # the reference concat already covers ~12s of the dataset
    if not vals:
        return None
    out = float(np.mean(vals))
    try:
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump({"speech_rms_dbfs": out}, f)
    except Exception:
        pass
    return out


def _get_vad():
    global _vad_model
    if _vad_model is None:
        from silero_vad import load_silero_vad
        _vad_model = load_silero_vad()
    return _vad_model


def _vad_utterances(path):
    """Split an audio file into (start_sec, end_sec) utterances."""
    import librosa
    import torch
    from silero_vad import get_speech_timestamps

    wav16, _ = librosa.load(path, sr=16000, mono=True)
    ts = get_speech_timestamps(torch.from_numpy(wav16), _get_vad(),
                               sampling_rate=16000)
    segs = [(t["start"] / 16000.0, t["end"] / 16000.0) for t in ts]
    if not segs:
        return []

    # Merge speech bursts into utterances bounded by MAX_UTT_SEC.
    utts = []
    cur_s, cur_e = segs[0]
    for s, e in segs[1:]:
        if e - cur_s <= MAX_UTT_SEC and s - cur_e <= 0.6:
            cur_e = e
        else:
            utts.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    utts.append((cur_s, cur_e))

    # Merge too-short utterances into their neighbours where possible.
    merged = []
    for s, e in utts:
        if merged and (e - s) < MIN_UTT_SEC and \
                (e - merged[-1][0]) <= MAX_UTT_SEC:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(max(0.0, s - 0.15), e + 0.25) for s, e in merged
            if (e - s) >= MIN_UTT_SEC]


def prepare_voice_dataset(voice_name, audio_paths, supplied_text, transcribe_fn):
    """
    Generator yielding progress lines. Writes datasets/<voice>/wavs/*.wav and
    metadata.csv (id|text|text).

    supplied_text: optional transcript. It is used verbatim when a single
    short (<= MAX_UTT_SEC) clip is given; longer audio is always VAD-split
    and each utterance transcribed by transcribe_fn (the Whisper agent).
    """
    import librosa

    voice_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", voice_name.strip())
    if not voice_name:
        raise ValueError("Give the voice a name first.")
    if not audio_paths:
        raise ValueError("Add at least one audio file.")

    ds_dir = os.path.join(DATASETS_DIR, voice_name)
    wav_dir = os.path.join(ds_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    rows = []
    total_sec = 0.0
    utt_idx = 0

    for file_i, src in enumerate(audio_paths):
        name = os.path.basename(src)
        yield f"[{file_i + 1}/{len(audio_paths)}] Analyzing {name}..."
        wav24, _ = librosa.load(src, sr=DATASET_SR, mono=True)
        dur = len(wav24) / DATASET_SR

        if supplied_text and len(audio_paths) == 1 and dur <= MAX_UTT_SEC:
            spans = [(0.0, dur)]
            texts = [supplied_text.strip()]
            yield "Using the supplied transcript for this clip."
        else:
            spans = _vad_utterances(src)
            texts = [None] * len(spans)
            yield f"  VAD found {len(spans)} utterance(s) in {dur:.1f}s of audio."
            if supplied_text:
                yield ("  NOTE: supplied text ignored (audio longer than "
                       f"{MAX_UTT_SEC:.0f}s or multiple files); the Whisper "
                       "agent transcribes each utterance instead.")

        for (s, e), text in zip(spans, texts):
            seg = wav24[int(s * DATASET_SR):int(e * DATASET_SR)]
            if len(seg) < int(MIN_UTT_SEC * DATASET_SR):
                continue
            utt_idx += 1
            utt_id = f"{voice_name}_{utt_idx:04d}"
            out_path = os.path.join(wav_dir, utt_id + ".wav")
            sf.write(out_path, seg.astype(np.float32), DATASET_SR)

            if text is None:
                text = transcribe_fn(out_path).strip()
                yield f"  [whisper] {utt_id}: {text!r}"
            if not text:
                os.remove(out_path)
                utt_idx -= 1
                yield f"  Skipped silent/untranscribable segment at {s:.1f}s."
                continue
            rows.append((utt_id, text))
            total_sec += len(seg) / DATASET_SR

    if not rows:
        raise RuntimeError("No usable utterances were produced.")

    meta = os.path.join(ds_dir, "metadata.csv")
    with open(meta, "w", encoding="utf-8", newline="") as f:
        for utt_id, text in rows:
            clean = text.replace("|", " ").strip()
            f.write(f"{utt_id}|{clean}|{clean}\n")

    yield (f"DATASET READY: {len(rows)} utterances, {total_sec / 60.0:.1f} "
           f"minutes of audio -> {meta}")
    yield ("Every [whisper] line above is the COMPLETE transcript used for "
           f"training (nothing is shortened); all of them are saved to "
           f"{meta}. Open that file to check or correct any line before "
           "training.")
    yield ("Tip: 5+ minutes gives a recognizable voice; 30+ minutes gives a "
           "good one. You can now press Train.")


# --------------------------------------------------------------------- #

def _write_train_config(voice_name, epochs, batch_size, grad_accum,
                        learning_rate, model_dir=None):
    """Generate finetuning/src/config.py from the pristine original with our
    dataset paths and hyperparameters substituted. model_dir overrides the
    starting checkpoint (continue training from an existing voice)."""
    if not os.path.isfile(CONFIG_ORIG):
        shutil.copyfile(CONFIG_PATH, CONFIG_ORIG)
    text = open(CONFIG_ORIG, encoding="utf-8").read()

    ds_dir = os.path.join(DATASETS_DIR, voice_name).replace("\\", "/")
    out_dir = os.path.join(VOICES_DIR, voice_name, "training").replace("\\", "/")
    os.makedirs(out_dir, exist_ok=True)

    def set_field(name, value, current):
        pattern = rf"^(\s*{name}\s*(?::[^=]+)?=\s*).*$"
        repl = rf"\g<1>{value}"
        new, n = re.subn(pattern, repl, current, count=1, flags=re.MULTILINE)
        if n != 1:
            raise RuntimeError(f"Could not set config field {name}")
        return new

    text = set_field("csv_path", f'"{ds_dir}/metadata.csv"', text)
    text = set_field("wav_dir", f'"{ds_dir}/wavs"', text)
    text = set_field("preprocessed_dir", f'"{ds_dir}/preprocess"', text)
    text = set_field("output_dir", f'"{out_dir}"', text)
    text = set_field("ljspeech", "True", text)
    text = set_field("json_format", "False", text)
    text = set_field("preprocess", "True", text)
    text = set_field("is_turbo", "False", text)
    text = set_field("is_lora", "True", text)
    # We keep the original English tokenizer (704 tokens); the toolkit default
    # (2454) is for its add-a-new-language workflow and enlarges text_emb /
    # text_head so the checkpoint no longer matches the standard T3 config.
    text = set_field("new_vocab_size", "704", text)
    text = set_field("batch_size", str(int(batch_size)), text)
    text = set_field("grad_accum", str(int(grad_accum)), text)
    text = set_field("num_epochs", str(int(epochs)), text)
    text = set_field("learning_rate", str(float(learning_rate)), text)
    # Windows: dataloader worker subprocesses are fragile; keep in-process.
    text = set_field("dataloader_num_workers", "0", text)
    text = set_field("is_inference", "False", text)
    if model_dir:
        text = set_field("model_dir",
                         f'"{model_dir.replace(os.sep, "/")}"', text)

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    return out_dir


def _trim_t3_vocab(merged_path, base_t3_path, out_path):
    """Copy the merged T3 checkpoint, trimming any enlarged text-vocab rows
    back to the base model's size so ChatterboxTTS.from_local can load it.
    A no-op copy when the shapes already match."""
    from safetensors.torch import load_file, save_file
    merged = load_file(merged_path)
    base = load_file(base_t3_path)
    for key in ("text_emb.weight", "text_head.weight", "text_head.bias"):
        if key in merged and key in base and \
                merged[key].shape[0] != base[key].shape[0]:
            merged[key] = merged[key][: base[key].shape[0]].clone()
    save_file(merged, out_path)


def make_voice_conds(voice_name):
    """Build voices/<voice>/conds.pt from the voice's own training audio so
    the finished voice sounds like the trained speaker by default (instead of
    the base model's default voice). Run as a subprocess by run_training."""
    import torch
    import librosa
    from pathlib import Path
    from chatterbox.src.chatterbox.tts import ChatterboxTTS

    voice_dir = os.path.join(VOICES_DIR, voice_name)
    wav_dir = os.path.join(DATASETS_DIR, voice_name, "wavs")
    wavs = sorted(
        (os.path.join(wav_dir, f) for f in os.listdir(wav_dir)
         if f.endswith(".wav")),
        key=os.path.getsize, reverse=True)
    if not wavs:
        raise RuntimeError(f"No dataset audio found in {wav_dir}")

    # Concatenate the longest utterances into a ~12s reference clip.
    ref_parts, ref_len = [], 0
    for w in wavs:
        audio, _ = librosa.load(w, sr=DATASET_SR, mono=True)
        ref_parts.append(audio)
        ref_len += len(audio)
        if ref_len >= 12 * DATASET_SR:
            break
    ref_path = os.path.join(voice_dir, "reference.wav")
    ref = np.concatenate(ref_parts)
    sf.write(ref_path, ref, DATASET_SR)

    # Record how loud the training audio is so generation can match it
    # (survives the PyTorch-only path, which deletes reference.wav).
    lv = speech_rms_dbfs(ref, DATASET_SR)
    if lv is not None:
        with open(os.path.join(voice_dir, "loudness.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"speech_rms_dbfs": lv}, f)
        print(f"Training-audio loudness recorded: {lv:.1f} dBFS "
              f"(generated audio will be matched to this level).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTTS.from_local(Path(voice_dir), device)
    conds = model._build_conditionals(ref_path, exaggeration=0.5)
    conds.save(Path(voice_dir) / "conds.pt")
    print(f"conds.pt built from {len(ref_parts)} training utterance(s).")


def make_style_conds(voice_name, style_paths):
    """Re-style a trained voice: keep WHO is speaking (the voice's speaker
    embedding + vocoder timbre, both taken from its own training audio) but
    take HOW it speaks — pacing, pauses, rhythm — from the style recordings.
    The T3 model paces its output by continuing the style prompt tokens,
    while the vocoder renders everything in the trained voice's timbre, so
    the style speaker's voice itself is not copied.

    The previous conds.pt is backed up to conds_voice_only.pt (once) so the
    style can be reverted. Run as a subprocess by the UI handler.
    """
    import torch
    import librosa
    from pathlib import Path
    from chatterbox.src.chatterbox.tts import ChatterboxTTS, Conditionals
    from chatterbox.src.chatterbox.models.t3.modules.cond_enc import T3Cond

    voice_dir = os.path.join(VOICES_DIR, voice_name)
    if not os.path.isfile(os.path.join(voice_dir, "t3_cfg.safetensors")):
        raise RuntimeError(f"Voice '{voice_name}' not found in {VOICES_DIR}")
    style_paths = [p for p in (style_paths or []) if os.path.isfile(p)]
    if not style_paths:
        raise RuntimeError("Add at least one style recording.")

    def concat_clips(paths, sr, max_sec):
        parts, total = [], 0
        for p in paths:
            a, _ = librosa.load(p, sr=sr, mono=True)
            parts.append(a)
            total += len(a)
            if total >= max_sec * sr:
                break
        if not parts:
            raise RuntimeError("No audio could be loaded.")
        return np.concatenate(parts)

    # Timbre reference: the voice's own audio. Use reference.wav when the
    # voice has one; otherwise rebuild the concat from its dataset into a
    # TEMP file (PyTorch-only voices deliberately have no reference.wav so
    # the ONNX engine won't clone them — that choice must survive this).
    ref_path = os.path.join(voice_dir, "reference.wav")
    tmp_ref = None
    if not os.path.isfile(ref_path):
        wav_dir = os.path.join(DATASETS_DIR, voice_name, "wavs")
        if not os.path.isdir(wav_dir):
            raise RuntimeError(
                f"'{voice_name}' has no reference.wav and no dataset at "
                f"{wav_dir}; cannot build its timbre reference.")
        wavs = sorted((os.path.join(wav_dir, f) for f in os.listdir(wav_dir)
                       if f.endswith(".wav")),
                      key=os.path.getsize, reverse=True)
        ref = concat_clips(wavs, DATASET_SR, 12)
        tmp_ref = os.path.join(voice_dir, "_timbre_tmp.wav")
        sf.write(tmp_ref, ref, DATASET_SR)
        ref_path = tmp_ref

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading voice '{voice_name}'...")
        model = ChatterboxTTS.from_local(Path(voice_dir), device)
        base = model._build_conditionals(ref_path, exaggeration=0.5)

        # Style prompt: the model conditions on the first ~6s of speech
        # tokens, so that window of the style recordings drives the pacing.
        print(f"Reading the speaking style from {len(style_paths)} "
              f"recording(s)...")
        style16 = concat_clips(style_paths, 16000, 10)
        sf.write(os.path.join(voice_dir, "style.wav"),
                 style16.astype(np.float32), 16000)
        plen = model.t3.hp.speech_cond_prompt_len
        style_tokens, _ = model.s3gen.tokenizer.forward(
            [style16[: model.ENC_COND_LEN]], max_len=plen)
        style_tokens = torch.atleast_2d(style_tokens).to(model.device)

        t3 = T3Cond(speaker_emb=base.t3.speaker_emb,
                    cond_prompt_speech_tokens=style_tokens,
                    emotion_adv=base.t3.emotion_adv).to(device=model.device)
        conds = Conditionals(t3, base.gen)

        cpath = Path(voice_dir) / "conds.pt"
        backup = Path(voice_dir) / "conds_voice_only.pt"
        if cpath.exists() and not backup.exists():
            shutil.copyfile(cpath, backup)
        conds.save(cpath)
        print(f"STYLE APPLIED: '{voice_name}' keeps its own sound but now "
              f"paces itself like the style recording(s). Revert any time "
              f"(backup: conds_voice_only.pt).")
    finally:
        if tmp_ref and os.path.isfile(tmp_ref):
            os.remove(tmp_ref)


def _clip_pause_profile(y, sr):
    """Speech/pause structure of a clip via silero-VAD.
    Returns (segments, speech_sec, internal_gap_sec) where segments are
    (start, end) sample ranges at the clip's own sample rate."""
    import librosa
    import torch
    from silero_vad import get_speech_timestamps

    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
    # silero already pads each segment by ~30ms; padding again here would
    # eat the measured pauses (a 0.4s pause would read as 0.24s), which
    # badly understates how much a slow speaker actually pauses.
    ts = get_speech_timestamps(torch.from_numpy(y16), _get_vad(),
                               sampling_rate=16000)
    scale = sr / 16000.0
    segs = []
    for t in ts:
        s = max(0, int(t["start"] * scale))
        e = min(len(y), int(t["end"] * scale))
        if segs and s <= segs[-1][1]:
            segs[-1] = (segs[-1][0], e)
        else:
            segs.append((s, e))
    if not segs:
        return [], 0.0, 0.0
    speech = sum(e - s for s, e in segs) / sr
    gaps = sum(max(0, segs[i + 1][0] - segs[i][1])
               for i in range(len(segs) - 1)) / sr
    return segs, speech, gaps


def make_pace_dataset(voice_name, style_paths, out_name, transcribe_fn):
    """Generator: build datasets/<out_name> — the voice's own dataset
    retimed to the SPEAKING MANNER of the style recordings.

    People slow down mainly by pausing more and longer between words, not
    by dragging out every sound (speech research: articulation rate and
    pauses are separate dimensions). So this does two things, measured
    independently from the style recordings:
      1. pause structure: the silence between speech bursts is rescaled
         to match the style's pause-to-speech ratio (per-pause capped so
         it stays natural);
      2. articulation: the speech itself is stretched only mildly, hard
         clamped to ±15%, so words never sound slow-motion.
    Fine-tuning on the result teaches the voice to SPEAK with that pacing
    in its own voice; nothing of the style speaker's timbre is used."""
    import librosa

    src_ds = os.path.join(DATASETS_DIR, voice_name)
    meta = os.path.join(src_ds, "metadata.csv")
    if not os.path.isfile(meta):
        raise RuntimeError(
            f"Voice '{voice_name}' has no dataset ({meta}). Pace training "
            f"retimes the voice's own training data, so that data must "
            f"still exist.")
    style_paths = [p for p in (style_paths or []) if os.path.isfile(p)]
    if not style_paths:
        raise RuntimeError("Add at least one style recording.")

    # How the style recordings speak: articulation (words/s while actually
    # speaking) and pause structure (silence per second of speech).
    yield "Measuring the style recordings' manner (Whisper + VAD)..."
    st_words, st_speech, st_gaps = 0, 0.0, 0.0
    for p in style_paths:
        text = (transcribe_fn(p) or "").strip()
        y, sr = librosa.load(p, sr=None, mono=True)
        segs, speech, gaps = _clip_pause_profile(y, sr)
        if text and speech > 1.0:
            st_words += len(text.split())
            st_speech += speech
            st_gaps += gaps
            yield (f"  [style] {os.path.basename(p)}: "
                   f"{len(text.split())} words, {speech:.1f}s of speech, "
                   f"{gaps:.1f}s of pauses")
    if not st_words or st_speech <= 0:
        raise RuntimeError("Could not measure the style recordings' pace "
                           "(no transcribable speech found).")
    style_art = st_words / st_speech
    style_pause_ratio = st_gaps / st_speech

    # The same measurements for the voice's own dataset.
    rows = []
    for line in open(meta, encoding="utf-8").read().splitlines():
        if line.strip():
            parts = line.split("|")
            rows.append((parts[0], parts[1]))
    wav_dir = os.path.join(src_ds, "wavs")
    clips, profiles = {}, {}
    vw, vs, vg = 0, 0.0, 0.0
    for utt_id, text in rows:
        y, sr = librosa.load(os.path.join(wav_dir, utt_id + ".wav"),
                             sr=None, mono=True)
        clips[utt_id] = (y, sr)
        segs, speech, gaps = _clip_pause_profile(y, sr)
        profiles[utt_id] = segs
        vw += len(text.split())
        vs += speech
        vg += gaps
    if vs <= 0:
        raise RuntimeError("No speech found in the voice's dataset.")
    voice_art = vw / vs
    voice_pause_ratio = max(vg / vs, 0.02)

    # Words are stretched only mildly (hard clamp: never past ±15%, so no
    # slow-motion drawl); the rest of the pace difference is delivered the
    # way humans do it — by the pauses between words.
    art_rate = max(0.87, min(1.15, style_art / voice_art))
    # The model smooths pauses back out when generating (measured transfer:
    # it reproduces roughly 60% of the pause time trained into it), so the
    # dataset overshoots the pause target; generation then lands near the
    # style's actual manner.
    pause_scale = (style_pause_ratio / voice_pause_ratio) * 1.6
    yield (f"Style manner: {style_art:.2f} words/s while speaking, "
           f"{style_pause_ratio:.2f}s of pause per second of speech.")
    yield (f"Voice manner: {voice_art:.2f} words/s while speaking, "
           f"{voice_pause_ratio:.2f}s of pause per second of speech.")
    yield (f"Applying: words x{1 / art_rate:.2f} duration (clamped to stay "
           f"natural), pauses x{pause_scale:.1f}.")
    if abs(art_rate - 1.0) < 0.03 and abs(pause_scale - 1.0) < 0.15:
        yield ("NOTE: the voice already speaks in nearly this manner — "
               "training will change little.")

    GAP_CAP = 1.5      # s; single pauses longer than this sound broken
    MIN_SEG_STRETCH = 0.25  # s; segments shorter than this aren't stretched

    def _resize_gap(gap, want_sec):
        cur = len(gap) / sr
        if want_sec < cur - 0.01:      # shrink around the midpoint
            keep = max(int(want_sec * sr), int(0.04 * sr))
            h = keep // 2
            return np.concatenate([gap[:h], gap[len(gap) - (keep - h):]])
        if want_sec <= cur + 0.01:
            return gap
        extra = np.zeros(int((want_sec - cur) * sr), dtype=gap.dtype)
        mid = len(gap) // 2
        fade = min(int(0.015 * sr), mid, len(gap) - mid)
        out_pre, out_post = gap[:mid].copy(), gap[mid:].copy()
        if fade > 1:                   # soften the splice into silence
            out_pre[-fade:] *= np.linspace(1.0, 0.0, fade)
            out_post[:fade] *= np.linspace(0.0, 1.0, fade)
        return np.concatenate([out_pre, extra, out_post])

    out_ds = os.path.join(DATASETS_DIR, out_name)
    out_wavs = os.path.join(out_ds, "wavs")
    os.makedirs(out_wavs, exist_ok=True)
    # The training pipeline is built for utterances up to ~MAX_UTT_SEC;
    # longer inputs hang its GPU preprocessing, so longer results are
    # skipped.
    max_out = MAX_UTT_SEC + 0.5
    kept, skipped = [], 0
    for i, (utt_id, text) in enumerate(rows):
        y, sr = clips[utt_id]
        segs = profiles[utt_id]
        if not segs:
            skipped += 1
            continue
        parts = [y[: segs[0][0]]]      # leading audio unchanged
        for j, (s, e) in enumerate(segs):
            seg = y[s:e]
            if abs(1.0 - art_rate) > 0.03 and len(seg) > MIN_SEG_STRETCH * sr:
                seg = librosa.effects.time_stretch(seg, rate=art_rate)
            parts.append(seg)
            if j < len(segs) - 1:
                gap = y[e: segs[j + 1][0]]
                want = min(GAP_CAP, (len(gap) / sr) * pause_scale)
                parts.append(_resize_gap(gap, want))
        parts.append(y[segs[-1][1]:])  # trailing audio unchanged
        ys = np.concatenate(parts)
        if len(ys) / sr > max_out:
            skipped += 1
            continue
        sf.write(os.path.join(out_wavs, utt_id + ".wav"),
                 ys.astype(np.float32), sr)
        kept.append((utt_id, text))
        if len(kept) % 25 == 0:
            yield f"  retimed {len(kept)} clips ({i + 1}/{len(rows)} scanned)"
    if len(kept) < 20:
        raise RuntimeError(
            f"Only {len(kept)} clips fit under {max_out:.0f}s after "
            f"retiming — too few to train on. The pace difference is "
            f"probably too large.")
    with open(os.path.join(out_ds, "metadata.csv"), "w",
              encoding="utf-8", newline="") as f:
        for utt_id, text in kept:
            f.write(f"{utt_id}|{text}|{text}\n")
    if skipped:
        yield (f"  {skipped} clip(s) skipped (they would exceed "
               f"{max_out:.0f}s after retiming).")
    yield f"PACED DATASET READY: {len(kept)} clips -> {out_ds}"


def revert_style_conds(voice_name):
    """Restore the voice's own speaking style from the backup made by
    make_style_conds. Returns True if something was restored."""
    voice_dir = os.path.join(VOICES_DIR, voice_name)
    backup = os.path.join(voice_dir, "conds_voice_only.pt")
    if not os.path.isfile(backup):
        return False
    shutil.copyfile(backup, os.path.join(voice_dir, "conds.pt"))
    style = os.path.join(voice_dir, "style.wav")
    if os.path.isfile(style):
        os.remove(style)
    return True


def stop_training():
    global _train_proc
    with _train_lock:
        p = _train_proc
    if p is not None and p.poll() is None:
        p.terminate()
        return True
    return False


def _stream_subprocess(cmd, cwd):
    """Run cmd, yielding output lines; stores proc for stop_training()."""
    global _train_proc
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env=env)
    with _train_lock:
        _train_proc = p
    stream_err = None
    try:
        try:
            for line in p.stdout:
                yield line.rstrip()
        except Exception as e:
            # If reading dies while the child lives, the child eventually
            # blocks on the full pipe — surface the error, never hang.
            stream_err = repr(e)
    finally:
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            # Output ended but the child won't exit (blocked writer /
            # broken reader): kill it instead of deadlocking forever.
            p.kill()
            p.wait()
            if stream_err is None:
                stream_err = "child outlived its output stream; killed"
        with _train_lock:
            _train_proc = None
    if stream_err:
        yield f"[stream error: {stream_err}]"
    yield f"[exit code {p.returncode}]"
    if p.returncode != 0:
        raise RuntimeError(f"Process failed with exit code {p.returncode}")


def run_training(voice_name, epochs, batch_size, grad_accum, learning_rate,
                 base_voice=None, target="both", conds_from=None):
    """
    Generator yielding log lines: fine-tunes T3 with LoRA, merges the
    adapter, and assembles voices/<voice>/ as a complete model folder.

    base_voice: name of an existing trained voice to continue training from
    (its merged weights become the starting checkpoint) instead of the base
    model.

    target: "both" (default) or "pytorch" fine-tune the weights; "onnx"
    skips the GPU fine-tune entirely — ONNX models are frozen inference
    graphs that cannot be trained, so the ONNX engine always speaks a
    trained voice by cloning its reference audio. "pytorch" additionally
    omits the reference clip so the voice is PyTorch-engine-only.

    conds_from: name of an existing voice whose conds.pt / reference.wav /
    loudness.json are copied instead of building them from this voice's
    dataset. Used by pace training, whose dataset is time-stretched: the
    conditioning should come from the ORIGINAL unprocessed audio so the
    timbre stays clean.
    """
    voice_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", voice_name.strip())
    ds_meta = os.path.join(DATASETS_DIR, voice_name, "metadata.csv")
    if not os.path.isfile(ds_meta):
        raise RuntimeError(
            f"No dataset found for '{voice_name}'. Run Prepare Dataset first.")

    pretrained_dir = os.path.join(FINETUNE_DIR, "pretrained_models")
    start_dir = pretrained_dir
    if base_voice and base_voice not in ("(base model)", "(none)"):
        cand = os.path.join(VOICES_DIR, base_voice)
        if not os.path.isfile(os.path.join(cand, "t3_cfg.safetensors")):
            raise RuntimeError(f"Base voice '{base_voice}' not found; "
                               f"expected {cand}\\t3_cfg.safetensors")
        start_dir = cand
        yield (f"Continuing from existing voice '{base_voice}' — its weights "
               f"are the starting point and the new training is added on top.")

    py = sys.executable
    voice_dir = os.path.join(VOICES_DIR, voice_name)

    if target == "onnx":
        yield ("NOTE: ONNX models are frozen inference graphs — they cannot "
               "be trained. Building a clone-only voice package instead: the "
               "ONNX engine will imitate the speaker from the reference "
               "audio. Pick 'Both' or 'PyTorch only' to actually fine-tune "
               "the model weights.")
        os.makedirs(voice_dir, exist_ok=True)
        for fname in ["ve.safetensors", "s3gen.safetensors", "tokenizer.json",
                      "conds.pt", "t3_cfg.safetensors"]:
            src = os.path.join(start_dir, fname)
            dst = os.path.join(voice_dir, fname)
            # Never overwrite: if this voice was fine-tuned before, its
            # trained weights must survive a clone-only re-run.
            if not os.path.isfile(dst):
                shutil.copyfile(src, dst)
        yield "Building voice conditioning from the training audio..."
        for line in _stream_subprocess(
                [py, "-c",
                 f"import voice_training; "
                 f"voice_training.make_voice_conds('{voice_name}')"],
                BASE_DIR):
            yield line
        yield (f"VOICE READY: {voice_dir}\n"
               f"Select it under Inference Engine -> Custom Trained Voice "
               f"and generate (works on both engines; no weights were "
               f"fine-tuned).")
        return

    out_dir = _write_train_config(
        voice_name, epochs, batch_size, grad_accum, learning_rate,
        model_dir=start_dir if start_dir != pretrained_dir else None)
    yield f"Training config written (dataset: {ds_meta})."
    yield "Starting LoRA fine-tune of the T3 transformer on the GPU..."

    for line in _stream_subprocess([py, "train.py"], FINETUNE_DIR):
        yield line

    yield "Training finished. Merging LoRA adapter into a standalone model..."
    for line in _stream_subprocess([py, "merge_lora.py"], FINETUNE_DIR):
        yield line

    merged = os.path.join(out_dir, "t3_finetuned_merged.safetensors")
    if not os.path.isfile(merged):
        raise RuntimeError(f"Merged model not found at {merged}")

    yield "Assembling the voice model folder..."
    os.makedirs(voice_dir, exist_ok=True)
    for fname in ["ve.safetensors", "s3gen.safetensors", "tokenizer.json",
                  "conds.pt"]:
        dst = os.path.join(voice_dir, fname)
        if not os.path.isfile(dst):
            shutil.copyfile(os.path.join(start_dir, fname), dst)
    # Trim against the pristine base t3 so the checkpoint always matches the
    # standard T3 config, regardless of what we started training from.
    _trim_t3_vocab(merged, os.path.join(pretrained_dir, "t3_cfg.safetensors"),
                   os.path.join(voice_dir, "t3_cfg.safetensors"))

    if conds_from and os.path.isfile(
            os.path.join(VOICES_DIR, conds_from, "conds.pt")):
        yield (f"Copying voice conditioning from '{conds_from}' (original, "
               f"unprocessed timbre)...")
        for fname in ["conds.pt", "reference.wav", "loudness.json"]:
            src = os.path.join(VOICES_DIR, conds_from, fname)
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(voice_dir, fname))
    else:
        yield "Building voice conditioning from the training audio..."
        for line in _stream_subprocess(
                [py, "-c",
                 f"import voice_training; "
                 f"voice_training.make_voice_conds('{voice_name}')"],
                BASE_DIR):
            yield line

    if target == "pytorch":
        ref = os.path.join(voice_dir, "reference.wav")
        if os.path.isfile(ref):
            os.remove(ref)
        yield ("PyTorch-only package: reference clip omitted, so the ONNX "
               "engine will NOT imitate this voice (it uses its default "
               "voice instead).")

    yield (f"VOICE READY: {voice_dir}\n"
           f"Select it under Inference Engine -> Custom Trained Voice "
           f"and generate. The PyTorch engine runs the fine-tuned weights; "
           f"the ONNX engine clones the speaker from the reference audio.")
