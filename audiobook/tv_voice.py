import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
CHATTERBOX = os.path.dirname(HERE)
PROJECT = os.path.dirname(CHATTERBOX)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from voice_training import (VOICES_DIR, DATASETS_DIR, DATASET_SR,  # noqa: E402
                            MIN_UTT_SEC, MAX_UTT_SEC)
from audiobook import audext_bridge  # noqa: E402

REF_TARGET_SECONDS = 45.0
REF_MAX_SECONDS = 60.0
SEG_PAD = 0.12
GAP_SILENCE = 0.10
MERGE_GAP = 0.6
WORK_SR = 44100
TRAIN_EPOCHS = 10
TRAIN_BATCH = 8
TRAIN_GRAD_ACCUM = 2
TRAIN_LR = 1e-4


def _safe_voice_name(character):
    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", (character or "").strip())
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "TV_Voice"


def _cut_segment(video, start, end, out_wav):
    start = max(0.0, start - SEG_PAD)
    dur = (end + SEG_PAD) - start
    if dur <= 0:
        return False
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", video,
         "-t", f"{dur:.3f}", "-map", "0:a:0", "-ac", "2", "-ar", str(WORK_SR),
         "-c:a", "pcm_s16le", out_wav],
        capture_output=True)
    return r.returncode == 0 and os.path.isfile(out_wav)


def _merge_adjacent(segments):
    ordered = sorted(segments, key=lambda s: (s["episode"],
                                              float(s["start"])))
    merged = []
    for s in ordered:
        if merged:
            m = merged[-1]
            gap = float(s["start"]) - float(m["end"])
            if (s["episode"] == m["episode"] and 0.0 <= gap <= MERGE_GAP and
                    float(s["end"]) - float(m["start"]) <= MAX_UTT_SEC):
                m["end"] = float(s["end"])
                m["text"] = (m.get("text", "") + " " +
                             (s.get("text") or "")).strip()
                m["score"] = min(float(m.get("score", 0.0)),
                                 float(s.get("score", 0.0)))
                if s.get("source") != "sdh":
                    m["source"] = "voice"
                continue
        merged.append(dict(s))
    return merged


def _rank(segments):
    return sorted(
        segments,
        key=lambda s: (0 if s.get("source") == "sdh" else 1,
                       -float(s.get("score", 0.0)),
                       -(float(s["end"]) - float(s["start"]))))


def _pick_segments(segments, target_seconds):
    picked, total = [], 0.0
    for s in _rank(segments):
        dur = float(s["end"]) - float(s["start"])
        if dur < 0.6:
            continue
        picked.append(s)
        total += dur + GAP_SILENCE
        if total >= target_seconds:
            break
    return picked


def _pick_training_segments(segments):
    picked = []
    for s in _rank(segments):
        dur = float(s["end"]) - float(s["start"])
        if dur < MIN_UTT_SEC or dur > MAX_UTT_SEC:
            continue
        if not (s.get("text") or "").strip():
            continue
        picked.append(s)
    return picked


def _concat_cuts(picked, tmpdir, should_stop):
    gap = np.zeros(int(GAP_SILENCE * WORK_SR), dtype="float32")
    chunks, spans, used = [], [], []
    pos = 0
    for i, s in enumerate(picked):
        if should_stop():
            break
        seg_wav = os.path.join(tmpdir, f"seg_{i:04d}.wav")
        if not _cut_segment(s["episode"], float(s["start"]),
                            float(s["end"]), seg_wav):
            continue
        try:
            x, sr = sf.read(seg_wav, dtype="float32", always_2d=True)
        except Exception:
            continue
        finally:
            try:
                os.remove(seg_wav)
            except OSError:
                pass
        if sr != WORK_SR or not len(x):
            continue
        if chunks:
            chunks.append(np.stack([gap, gap], axis=1))
            pos += len(gap)
        spans.append((pos, pos + len(x)))
        used.append(s)
        chunks.append(x)
        pos += len(x)
    if not chunks:
        raise RuntimeError("no segments could be cut from the episode")
    return np.concatenate(chunks, axis=0), spans, used


def _isolate(combined, tmpdir, device, log):
    raw = os.path.join(tmpdir, "combined.wav")
    sf.write(raw, combined, WORK_SR, subtype="PCM_16")
    clean = os.path.join(tmpdir, "vocals.wav")
    isolated = True
    try:
        audext_bridge.isolate_vocals(raw, clean, device=device, log=log)
        src = clean
    except Exception as e:
        log(f"vocal isolation unavailable ({e}); using the raw cut")
        src = raw
        isolated = False
    y, sr = sf.read(src, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0:
        y = y * (0.97 / peak)
    return y, sr, isolated


def _slice(y, sr, span):
    scale = sr / float(WORK_SR)
    a = max(0, int(span[0] * scale))
    b = min(len(y), int(span[1] * scale))
    return y[a:b]


_GENDER_F0 = {"male": (80.0, 170.0), "female": (160.0, 330.0),
              "child": (200.0, 400.0)}
PURIFY_MIN_SIM = 0.55


def _clip16(y, sr, span):
    import librosa
    clip = _slice(y, sr, span).astype("float32")
    if sr != 16000 and len(clip):
        clip = librosa.resample(clip, orig_sr=sr, target_sr=16000)
    return clip


def _f0_median(clip16):
    import librosa
    if len(clip16) < int(16000 * 0.3):
        return None
    f0, _, _ = librosa.pyin(clip16, sr=16000, fmin=70, fmax=400,
                            frame_length=1024)
    v = f0[~np.isnan(f0)]
    return float(np.median(v)) if len(v) else None


def _dominant_cluster(embs_by_idx):
    from collections import Counter
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    idx = list(embs_by_idx)
    M = np.stack([embs_by_idx[i] for i in idx])
    D = np.clip(1.0 - M @ M.T, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=1.0 - PURIFY_MIN_SIM, criterion="distance")
    scores = {}
    for lab in set(labels):
        members = [idx[k] for k in range(len(idx)) if labels[k] == lab]
        sub = np.stack([embs_by_idx[i] for i in members])
        c = sub.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-9)
        tight = float(np.mean(sub @ c))
        scores[lab] = (len(members) * tight, members)
    best = max(scores.values(), key=lambda kv: kv[0])
    return best[1]


def _purify_spans(y, sr, spans, used, log, want_gender=None, model_path=None):
    import tempfile
    vid = None
    if model_path and os.path.exists(model_path):
        try:
            import find_book_lines as fbl
            vid = fbl.VoiceID(model_path)
        except Exception as e:
            log(f"purify: voice model unavailable ({e})")
    lo, hi = _GENDER_F0.get(want_gender, (0.0, 1e9))
    kept, embs = [], {}
    dropped_gender = 0
    for i, span in enumerate(spans):
        clip = _clip16(y, sr, span)
        f0 = _f0_median(clip)
        if want_gender and (f0 is None or not (lo <= f0 <= hi)):
            dropped_gender += 1
            continue
        kept.append(i)
        if vid is not None and len(clip) >= int(16000 * 0.8):
            tmp = tempfile.mktemp(suffix=".wav")
            try:
                sf.write(tmp, clip, 16000)
                e = vid.embed(tmp, 0.0, len(clip) / 16000.0)
                if e is not None:
                    embs[i] = e
            except Exception:
                pass
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    if not kept:
        log("purify: nothing matched the target voice; keeping all spans")
        return spans, used
    if len(embs) >= 4:
        coherent = set(_dominant_cluster(embs))
        kept = [i for i in kept if i not in embs or i in coherent]
    kept = sorted(set(kept))
    log(f"purify: {len(spans)} span(s) -> {len(kept)} kept; dropped "
        f"{dropped_gender} off-gender, "
        f"{len(spans) - len(kept) - dropped_gender} off-voice.")
    return [spans[i] for i in kept], [used[i] for i in kept]


def _write_reference(y, sr, spans, out_ref, target_seconds):
    gap = np.zeros(int(GAP_SILENCE * sr), dtype="float32")
    parts, total, used = [], 0.0, 0
    for span in spans:
        clip = _slice(y, sr, span)
        if len(clip) < int(0.6 * sr):
            continue
        if parts:
            parts.append(gap)
        parts.append(clip)
        total += len(clip) / float(sr)
        used += 1
        if total >= target_seconds:
            break
    if not parts:
        raise RuntimeError("no usable audio for the reference clip")
    ref = np.concatenate(parts)[: int(REF_MAX_SECONDS * sr)]
    os.makedirs(os.path.dirname(out_ref), exist_ok=True)
    sf.write(out_ref, ref, sr, subtype="PCM_16")
    return used, len(ref) / float(sr)


def _write_dataset(voice, y, sr, spans, segs, log):
    import librosa

    ds_dir = os.path.join(DATASETS_DIR, voice)
    wav_dir = os.path.join(ds_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    for f in os.listdir(wav_dir):
        if f.endswith(".wav"):
            try:
                os.remove(os.path.join(wav_dir, f))
            except OSError:
                pass

    rows, total = [], 0.0
    for span, s in zip(spans, segs):
        text = (s.get("text") or "").replace("|", " ").strip()
        if not text:
            continue
        clip = _slice(y, sr, span)
        dur = len(clip) / float(sr)
        if dur < MIN_UTT_SEC or dur > MAX_UTT_SEC + 0.5:
            continue
        if not len(clip) or float(np.max(np.abs(clip))) <= 1e-4:
            continue
        if sr != DATASET_SR:
            clip = librosa.resample(clip, orig_sr=sr, target_sr=DATASET_SR)
        utt_id = f"{voice}_{len(rows) + 1:04d}"
        sf.write(os.path.join(wav_dir, utt_id + ".wav"),
                 clip.astype(np.float32), DATASET_SR)
        rows.append((utt_id, text))
        total += dur

    if not rows:
        raise RuntimeError("no usable training utterances were produced")

    meta = os.path.join(ds_dir, "metadata.csv")
    with open(meta, "w", encoding="utf-8", newline="") as f:
        for utt_id, text in rows:
            f.write(f"{utt_id}|{text}|{text}\n")
    log(f"training dataset: {len(rows)} utterance(s), {total / 60.0:.1f} "
        f"minute(s) -> {meta}")
    return rows, total


def _train(voice, epochs, log, should_stop):
    import voice_training
    for msg in voice_training.run_training(voice, epochs, TRAIN_BATCH,
                                           TRAIN_GRAD_ACCUM, TRAIN_LR,
                                           target="both"):
        log(msg)
        if should_stop():
            voice_training.stop_training()
            raise RuntimeError("cancelled")


def auto_create_voice(book_path, episode_paths, character, voice_name=None,
                      device="cpu", target_seconds=REF_TARGET_SECONDS,
                      train=True, epochs=TRAIN_EPOCHS, voice_gender=None,
                      log=None, progress=None, should_stop=None):
    log = log or (lambda *a: None)
    progress = progress or (lambda *a: None)
    should_stop = should_stop or (lambda: False)

    import find_book_lines
    result = find_book_lines.analyze_for_voices(
        book_path, episode_paths, targets=[character],
        log=log, progress=progress, should_stop=should_stop)
    if should_stop():
        raise RuntimeError("cancelled")

    token = find_book_lines.speaker_key(character)
    entry = result.get(token) or (next(iter(result.values()))
                                  if len(result) == 1 else None)
    if not entry or not entry.get("segments"):
        raise RuntimeError(
            f"couldn't find clean lines for '{character}' in these episodes")

    return _build_from_entry(entry, character, voice_name, device,
                             target_seconds, train, epochs, log, should_stop,
                             voice_gender=voice_gender)


def _build_from_entry(entry, character, voice_name, device, target_seconds,
                      train, epochs, log, should_stop, voice_gender=None):
    import find_book_lines
    token = find_book_lines.speaker_key(character)

    segments = _merge_adjacent(entry["segments"])
    picked = _pick_training_segments(segments) if train \
        else _pick_segments(segments, target_seconds * 4)
    if not picked:
        raise RuntimeError(
            f"'{character}' has no lines long enough to train on")
    log(f"{character}: {len(segments)} candidate line(s); cutting the best "
        f"{len(picked)}.")

    voice = _safe_voice_name(voice_name or character)
    vdir = os.path.join(VOICES_DIR, voice)
    out_ref = os.path.join(vdir, "reference.wav")

    tmpdir = tempfile.mkdtemp(prefix="tvvoice_")
    try:
        combined, spans, used_segs = _concat_cuts(picked, tmpdir, should_stop)
        if should_stop():
            raise RuntimeError("cancelled")
        log(f"cut {len(used_segs)} clip(s); isolating the voice...")
        y, sr, isolated = _isolate(combined, tmpdir, device, log)
        if not isolated:
            log("WARNING: music/effects were NOT removed from this audio.")

        import find_book_lines as _fbl
        spans, used_segs = _purify_spans(y, sr, spans, used_segs, log,
                                         want_gender=voice_gender,
                                         model_path=_fbl.VOICE_MODEL)
        if not spans:
            raise RuntimeError(
                f"no clean '{character}' voice survived purification")

        rows, train_seconds = ([], 0.0)
        if train:
            rows, train_seconds = _write_dataset(voice, y, sr, spans,
                                                 used_segs, log)
        ref_used, ref_dur = _write_reference(y, sr, spans, out_ref,
                                             target_seconds)
        log(f"conditioning mask for '{voice}': {ref_used} clip(s), "
            f"{ref_dur:.1f}s (separate from the training data).")
    finally:
        for f in os.listdir(tmpdir):
            try:
                os.remove(os.path.join(tmpdir, f))
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass

    attribution = {
        "character": character,
        "book_token": entry.get("display", character),
        "threshold": entry.get("threshold"),
        "prints": entry.get("prints", {}),
        "used": [_seg_min(s) for s in used_segs],
        "candidates": [_seg_min(s) for s in segments],
    }
    try:
        with open(os.path.join(vdir, "attribution.json"), "w",
                  encoding="utf-8") as f:
            json.dump(attribution, f)
    except OSError as e:
        log(f"note: could not write attribution.json ({e})")

    trained = False
    if train:
        if should_stop():
            raise RuntimeError("cancelled")
        if train_seconds < 60.0:
            log(f"WARNING: only {train_seconds / 60.0:.1f} minute(s) of clean "
                f"speech - the trained voice will be rough. More episodes "
                f"give a better model.")
        log(f"=== Training '{voice}' on {len(rows)} utterance(s) "
            f"({epochs} epochs) ===")
        _train(voice, epochs, log, should_stop)
        trained = os.path.isfile(os.path.join(vdir, "t3_cfg.safetensors"))
        if not trained:
            raise RuntimeError(
                f"training finished but no model was written to {vdir}")
        log(f"=== Trained model ready: {vdir}\\t3_cfg.safetensors ===")

    return {"voice": voice, "character": character, "reference": out_ref,
            "token": token, "used": len(used_segs), "found": len(segments),
            "threshold": entry.get("threshold"),
            "trained": trained, "utterances": len(rows),
            "train_seconds": train_seconds,
            "segments": [_seg_min(s) for s in segments]}


def auto_create_all(book_path, episode_paths, show_title, roster=None,
                    book_title=None, device="cpu",
                    target_seconds=REF_TARGET_SECONDS, epochs=TRAIN_EPOCHS,
                    do_lookup=True, read_credits=False, credits_kw=None,
                    store=None, on_voice=None, log=None, progress=None,
                    should_stop=None):
    log = log or (lambda *a: None)
    progress = progress or (lambda *a: None)
    should_stop = should_stop or (lambda: False)

    import find_book_lines
    import cast_resolver

    log("Analyzing the whole cast from the show...")
    analysis = find_book_lines.analyze_for_voices(
        book_path, episode_paths, targets=None,
        log=log, progress=progress, should_stop=should_stop)
    if should_stop():
        raise RuntimeError("cancelled")

    plan_res = cast_resolver.resolve_cast(
        book_path, episode_paths, show_title, book_title=book_title,
        roster=roster, store=store, do_lookup=do_lookup,
        read_credits=read_credits, credits_kw=credits_kw, analysis=analysis,
        log=log, progress=progress, should_stop=should_stop)
    plan = plan_res["plan"]

    portrayed = sorted(
        [(n, p) for n, p in plan.items() if p["status"] == "portrayed"],
        key=lambda np: -np[1]["audio_seconds"])
    thin = [n for n, p in plan.items() if p["status"] == "thin"]
    absent = [n for n, p in plan.items() if p["status"] == "absent"]
    log(f"Cast plan: {len(portrayed)} to voice now, {len(thin)} thin, "
        f"{len(absent)} book-only (need your input).")

    outcomes = []
    for i, (name, p) in enumerate(portrayed):
        if should_stop():
            break
        progress(i, len(portrayed), f"voice {i + 1}/{len(portrayed)}: {name}")
        entry = analysis.get(p["token"])
        if not entry or not entry.get("segments"):
            outcomes.append({"character": name, "voice": None,
                             "status": "portrayed", "error": "no segments"})
            continue
        log(f"=== [{i + 1}/{len(portrayed)}] {name} "
            f"({p['audio_seconds']:.0f}s, actor {p.get('actor')}) ===")
        try:
            out = _build_from_entry(entry, name, None, device, target_seconds,
                                    True, epochs, log, should_stop)
            out["status"] = "portrayed"
            out["actor"] = p.get("actor")
            outcomes.append(out)
            if on_voice:
                on_voice(name, out)
        except Exception as e:
            if should_stop():
                break
            log(f"  '{name}' failed: {e}")
            outcomes.append({"character": name, "voice": None,
                             "status": "portrayed", "error": str(e)})

    for name in thin + absent:
        p = plan[name]
        outcomes.append({"character": name, "voice": None,
                         "status": p["status"], "actor": p.get("actor")})

    built = sum(1 for o in outcomes if o.get("voice"))
    log(f"Batch done: {built} voice(s) built and assigned.")
    return {"outcomes": outcomes, "plan": plan, "built": built,
            "thin": thin, "absent": absent}


def _seg_min(s):
    return {"episode": s["episode"], "episode_label": s.get("episode_label"),
            "start": float(s["start"]), "end": float(s["end"]),
            "text": s.get("text", ""), "score": float(s.get("score", 0.0)),
            "source": s.get("source", "voice")}
