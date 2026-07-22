"""
Voice Lab - train and tune character voices for the audiobook reader.

For the character picked from the book's cast you can:
  * train a brand-new voice from reference recordings (VAD split +
    Whisper transcription + LoRA fine-tune, the same pipeline as the
    main app's Voice Training tab),
  * run WAVE MATCH: the tuner synthesizes the reference speaker's own
    passages across a grid of settings and measures each candidate's
    waveform against the real recording (speaker embedding, spectrum,
    pitch) - the program picks the passages and the winner, not the ear
    of whoever is training. The reference trace and the clone's trace
    are drawn overlaid so you can see how closely they line up.
  * TEST VOICE: hear the voice read one or two of that character's
    actual lines from the book.
"""

import os
import re
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_DIR)

from audiobook import synth, wave_match  # noqa: E402
from audiobook.analyze import NARRATOR, UNKNOWN  # noqa: E402

_whisper = None


def transcribe(path):
    """Lazy faster-whisper transcription for reference clips."""
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel("medium", device="cpu",
                                compute_type="int8")
    segs, _ = _whisper.transcribe(path, vad_filter=True, beam_size=1)
    return " ".join(s.text.strip() for s in segs).strip()


class CastingSession:
    def __init__(self, pdf):
        from audiobook import pdfbook
        self.pdf = pdf if pdf and os.path.isfile(pdf) else None
        self.units = []
        self.characters = {}
        self.casting = {}
        if self.pdf:
            cached = pdfbook.load_analysis(self.pdf)
            if cached:
                self.units = cached.get("units", [])
                self.characters = cached.get("characters", {})
            self.casting = pdfbook.load_casting(self.pdf)

    def _voice_of(self, name):
        if name == NARRATOR:
            return self.casting.get("narrator", {}).get("voice")
        return self.casting.get("characters", {}).get(name, {}).get("voice")

    def _set_voice(self, name, voice):
        if name == NARRATOR:
            self.casting.setdefault("narrator", {})["voice"] = voice
        else:
            self.casting.setdefault("characters", {}).setdefault(
                name, {})["voice"] = voice
        if self.pdf:
            from audiobook import pdfbook
            pdfbook.save_casting(self.pdf, self.casting)

    def _rebuild_cast_panel(self):
        pass


class VoiceLab(tk.Toplevel):
    def __init__(self, parent, reader, character=None):
        super().__init__(parent)
        self.reader = reader
        self.title("Voice Lab")
        self.geometry("920x760")
        self.ref_files = []
        self.episode_paths = []
        self._auto_cancel = None
        self.match_result = None
        self._busy = False
        self._build_ui()
        self._refresh_characters()
        if character:
            wanted = character.strip().lower()
            values = list(self.char_cb.cget("values"))
            pick = next((c for c in values if c.lower() == wanted), None)
            if pick is None:
                pick = character.strip()
                self.char_cb.configure(values=values + [pick])
            self.char_var.set(pick)
            self._load_character()

    # ---------------------------------------------------------------- UI --
    def _build_ui(self):
        pad = {"padx": 6, "pady": 3}

        row = ttk.Frame(self)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="Character:").pack(side="left")
        self.char_var = tk.StringVar()
        self.char_cb = ttk.Combobox(row, textvariable=self.char_var,
                                    width=24, state="readonly")
        self.char_cb.pack(side="left", padx=6)
        self.char_cb.bind("<<ComboboxSelected>>",
                          lambda _e: self._load_character())
        ttk.Label(row, text="Voice:").pack(side="left", padx=(16, 0))
        self.voice_var = tk.StringVar()
        self.voice_cb = ttk.Combobox(row, textvariable=self.voice_var,
                                     width=20, state="readonly")
        self.voice_cb.pack(side="left", padx=6)
        self.voice_cb.bind("<<ComboboxSelected>>",
                           lambda _e: self._assign_voice())
        self.btn_test = ttk.Button(row, text="🔊 Test voice",
                                   command=self.test_voice)
        self.btn_test.pack(side="left", padx=12)

        refbox = ttk.Labelframe(
            self, text="Reference audio - the clip this voice clones from")
        refbox.pack(fill="x", **pad)
        r0 = ttk.Frame(refbox)
        r0.pack(fill="x", **pad)
        ttk.Button(r0, text="Set reference audio...",
                   command=self.set_reference).pack(side="left")
        self.btn_ref_hear = ttk.Button(r0, text="🔊 Hear reference",
                                       command=self.hear_reference,
                                       state="disabled")
        self.btn_ref_hear.pack(side="left", padx=6)
        self.btn_ref_rm = ttk.Button(r0, text="Remove",
                                     command=self.remove_reference,
                                     state="disabled")
        self.btn_ref_rm.pack(side="left")
        self.ref_lbl = ttk.Label(refbox, text="", foreground="#444")
        self.ref_lbl.pack(anchor="w", padx=8, pady=(0, 4))

        autobox = ttk.Labelframe(
            self, text="Auto-create from TV episode - build this character's "
                       "voice from the show (no reference clip needed)")
        autobox.pack(fill="x", **pad)
        a0 = ttk.Frame(autobox)
        a0.pack(fill="x", **pad)
        ttk.Button(a0, text="Add episode video(s)...",
                   command=self._pick_episodes).pack(side="left")
        ttk.Button(a0, text="Add whole folder...",
                   command=self._pick_episode_folder).pack(side="left", padx=6)
        self.ep_lbl = ttk.Label(autobox, text="(no episodes)",
                                foreground="#444")
        self.ep_lbl.pack(anchor="w", padx=8)
        arow = ttk.Frame(autobox)
        arow.pack(fill="x", **pad)
        ttk.Label(arow, text="TV show name:").pack(side="left")
        self.show_var = tk.StringVar(value=self._guess_show_name())
        ttk.Entry(arow, textvariable=self.show_var, width=22)\
            .pack(side="left", padx=6)
        self.credits_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(arow, text="Read end-credits (slow)",
                        variable=self.credits_var).pack(side="left", padx=10)

        a1 = ttk.Frame(autobox)
        a1.pack(fill="x", **pad)
        self.btn_auto = ttk.Button(
            a1, text="Create voice from the show", command=self.auto_create)
        self.btn_auto.pack(side="left")
        self.btn_auto_cast = ttk.Button(
            a1, text="🎬 Voice the whole cast", command=self.auto_create_cast)
        self.btn_auto_cast.pack(side="left", padx=6)
        self.btn_auto_cancel = ttk.Button(
            a1, text="Cancel", command=self._cancel_auto, state="disabled")
        self.btn_auto_cancel.pack(side="left", padx=6)

        # --- training a new voice -----------------------------------------
        box = ttk.Labelframe(self, text="Train a new voice from recordings")
        box.pack(fill="x", **pad)
        r1 = ttk.Frame(box)
        r1.pack(fill="x", **pad)
        ttk.Label(r1, text="New voice name:").pack(side="left")
        self.name_var = tk.StringVar()
        ttk.Entry(r1, textvariable=self.name_var, width=18)\
            .pack(side="left", padx=6)
        ttk.Button(r1, text="Add audio files...",
                   command=self._pick_files).pack(side="left", padx=6)
        self.files_lbl = ttk.Label(r1, text="(no files)")
        self.files_lbl.pack(side="left")
        r2 = ttk.Frame(box)
        r2.pack(fill="x", **pad)
        ttk.Label(r2, text="Epochs:").pack(side="left")
        self.epochs_var = tk.IntVar(value=10)
        ttk.Spinbox(r2, from_=1, to=100, textvariable=self.epochs_var,
                    width=5).pack(side="left", padx=(4, 16))
        self.btn_train = ttk.Button(r2, text="Prepare dataset + train",
                                    command=self.train_voice)
        self.btn_train.pack(side="left")

        # --- wave match ----------------------------------------------------
        box2 = ttk.Labelframe(
            self, text="Wave match - tune the clone to the real recording")
        box2.pack(fill="both", expand=True, **pad)
        r3 = ttk.Frame(box2)
        r3.pack(fill="x", **pad)
        self.quick_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r3, text="Quick (4 candidates)",
                        variable=self.quick_var).pack(side="left")
        self.btn_match = ttk.Button(r3, text="▶ Auto-match",
                                    command=self.auto_match)
        self.btn_match.pack(side="left", padx=10)
        self.btn_apply = ttk.Button(r3, text="Apply matched settings",
                                    command=self.apply_match,
                                    state="disabled")
        self.btn_apply.pack(side="left")
        self.btn_hear = ttk.Button(r3, text="🔊 Hear best match",
                                   command=self.hear_match,
                                   state="disabled")
        self.btn_hear.pack(side="left", padx=10)
        self.btn_hear_ref = ttk.Button(r3, text="🔊 Hear real voice",
                                       command=self.hear_reference,
                                       state="disabled")
        self.btn_hear_ref.pack(side="left")
        self.style_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(r3, text="Copy the real speaker's pacing",
                        variable=self.style_var).pack(side="left", padx=10)
        self.match_lbl = ttk.Label(box2, text="", foreground="#444")
        self.match_lbl.pack(anchor="w", padx=8)

        self.trace = tk.Canvas(box2, height=220, bg="#1c1c24",
                               highlightthickness=0)
        self.trace.pack(fill="both", expand=False, padx=6, pady=4)
        self.trace.bind("<Configure>", lambda _e: self._draw_traces())

        self.progress = ttk.Progressbar(box2, mode="determinate")
        self.progress.pack(fill="x", padx=6)

        self.log_box = tk.Text(self, height=10, state="disabled",
                               font=("Consolas", 8), wrap="word")
        self.log_box.pack(fill="both", expand=True, **pad)

    def log(self, msg):
        def _do():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", str(msg) + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.after(0, _do)

    def _progress(self, i, n, msg):
        def _do():
            self.progress.configure(maximum=max(n, 1), value=i)
            self.match_lbl.configure(text=msg)
        self.after(0, _do)

    # ------------------------------------------------------------ helpers --
    def _refresh_characters(self):
        chars = [NARRATOR] + sorted(self.reader.characters)
        self.char_cb.configure(values=chars)
        if not self.char_var.get() and chars:
            self.char_var.set(chars[0])
        self.voice_cb.configure(values=self._all_voice_names())
        self._load_character()

    def _all_voice_names(self):
        from voice_training import list_voices, VOICES_DIR
        names = list(list_voices())
        if os.path.isdir(VOICES_DIR):
            for n in sorted(os.listdir(VOICES_DIR)):
                d = os.path.join(VOICES_DIR, n)
                if n not in names and (
                        os.path.isfile(os.path.join(d, "reference.wav")) or
                        os.path.isfile(os.path.join(d, "t3_cfg.safetensors"))):
                    names.append(n)
        return names

    def _load_character(self):
        self.voice_var.set(self.reader._voice_of(self.char_var.get()) or "")
        self._refresh_ref_info()

    def _assign_voice(self):
        self.reader._set_voice(self.char_var.get(), self.voice_var.get())
        self.reader._rebuild_cast_panel()
        self._refresh_ref_info()

    def _ref_path(self):
        from voice_training import VOICES_DIR
        voice = self.voice_var.get()
        if not voice:
            return None
        return os.path.join(VOICES_DIR, voice, "reference.wav")

    def _refresh_ref_info(self):
        p = self._ref_path()
        if not p:
            self.ref_lbl.configure(
                text="Assign a voice to manage its reference audio.")
            self.btn_ref_hear.configure(state="disabled")
            self.btn_ref_rm.configure(state="disabled")
            return
        if os.path.isfile(p):
            try:
                import soundfile as sf
                info = sf.info(p)
                dur = info.frames / float(info.samplerate)
                txt = (f"{self.voice_var.get()}: reference.wav is set "
                       f"({dur:.1f}s). The ONNX engine clones this clip.")
            except Exception:
                txt = f"{self.voice_var.get()}: reference.wav is set."
            self.ref_lbl.configure(text=txt)
            self.btn_ref_hear.configure(state="normal")
            self.btn_ref_rm.configure(state="normal")
        else:
            self.ref_lbl.configure(
                text=f"{self.voice_var.get()}: no reference set - the TTS "
                     "server builds one from the voice's training clips.")
            self.btn_ref_hear.configure(state="disabled")
            self.btn_ref_rm.configure(state="disabled")

    def set_reference(self):
        voice = self.voice_var.get()
        if not voice:
            messagebox.showinfo("Voice Lab", "Assign a voice first.")
            return
        f = filedialog.askopenfilename(
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg")])
        if not f:
            return

        def work():
            import librosa
            import soundfile as sf
            from voice_training import VOICES_DIR
            y, sr = librosa.load(f, sr=None, mono=True)
            y = y[: int(30.0 * sr)]
            vdir = os.path.join(VOICES_DIR, voice)
            os.makedirs(vdir, exist_ok=True)
            sf.write(os.path.join(vdir, "reference.wav"), y, sr,
                     subtype="PCM_16")
            self.log(f"reference for '{voice}': {os.path.basename(f)} "
                     f"({len(y) / sr:.1f}s)")
            self.after(0, self._refresh_ref_info)
        self._run_bg(work)

    def hear_reference(self):
        p = self._ref_path()
        if not p or not os.path.isfile(p):
            return

        def work():
            import sounddevice as sd
            import soundfile as sf
            x, sr = sf.read(p, dtype="float32")
            sd.play(x, sr, blocking=True)
        self._run_bg(work)

    def remove_reference(self):
        p = self._ref_path()
        if p and os.path.isfile(p):
            os.remove(p)
            self.log(f"reference removed for '{self.voice_var.get()}'.")
        self._refresh_ref_info()

    def _pick_episodes(self):
        f = filedialog.askopenfilenames(
            filetypes=[("Video", "*.mkv *.mp4 *.avi *.m4v *.mov *.webm "
                                  "*.ts *.wmv")])
        if f:
            self.episode_paths = list(f)
            self.ep_lbl.configure(text=f"{len(f)} episode file(s)")

    def _pick_episode_folder(self):
        d = filedialog.askdirectory()
        if not d:
            return

        def work():
            import find_book_lines
            eps = find_book_lines.find_episodes(d)
            paths = [e["path"] for e in eps]

            def done():
                self.episode_paths = paths
                self.ep_lbl.configure(
                    text=f"{len(paths)} episode(s) in {os.path.basename(d)}")
            self.after(0, done)
        self._run_bg(work)

    def _cancel_auto(self):
        if self._auto_cancel:
            self._auto_cancel.set()
            self.log("cancelling auto-create...")

    def auto_create(self):
        character = self.char_var.get()
        if character == NARRATOR:
            messagebox.showinfo(
                "Voice Lab", "The narrator isn't a show character - pick a "
                             "cast member.")
            return
        if not character:
            messagebox.showinfo("Voice Lab", "Pick a character first.")
            return
        if not self.episode_paths:
            messagebox.showinfo("Voice Lab", "Add episode video(s) first.")
            return
        book = getattr(self.reader, "pdf", None)
        if not book:
            messagebox.showinfo(
                "Voice Lab", "Open the book first - the show voice is matched "
                             "to the book's cast.")
            return
        self._auto_cancel = threading.Event()
        self.btn_auto.configure(state="disabled")
        self.btn_auto_cancel.configure(state="normal")

        def restore():
            self.btn_auto.configure(state="normal")
            self.btn_auto_cancel.configure(state="disabled")

        def work():
            from audiobook import tv_voice
            try:
                res = tv_voice.auto_create_voice(
                    book, list(self.episode_paths), character,
                    log=self.log, progress=self._progress,
                    should_stop=self._auto_cancel.is_set)
            finally:
                self.after(0, restore)
            voice = res["voice"]
            if res.get("trained"):
                self.log(f"Done: voice '{voice}' trained on "
                         f"{res['utterances']} line(s) "
                         f"({res['train_seconds'] / 60.0:.1f} min) from "
                         f"{res['found']} found; assigned to {character}.")
            else:
                self.log(f"Done: voice '{voice}' built from {res['used']} of "
                         f"{res['found']} lines found; assigned to "
                         f"{character}.")

            def done():
                names = self._all_voice_names()
                if voice not in names:
                    names.append(voice)
                self.voice_cb.configure(values=names)
                self.voice_var.set(voice)
                self._assign_voice()
            self.after(0, done)
        self._run_bg(work)

    def _guess_show_name(self):
        book = getattr(self.reader, "pdf", None)
        if not book:
            return ""
        stem = os.path.splitext(os.path.basename(book))[0]
        stem = re.sub(r"[_\.]+", " ", stem)
        words = [w for w in stem.split() if w.lower() not in (
            "the", "a", "an", "saga", "omnibus", "book", "novel", "series")]
        return " ".join(words[:2]) if words else stem

    def auto_create_cast(self):
        if not self.episode_paths:
            messagebox.showinfo("Voice Lab", "Add episode video(s) first.")
            return
        book = getattr(self.reader, "pdf", None)
        if not book:
            messagebox.showinfo("Voice Lab", "Open the book first.")
            return
        roster = sorted(self.reader.characters) if self.reader.characters \
            else None
        if not roster:
            messagebox.showinfo(
                "Voice Lab", "Analyze the book first so I know the cast.")
            return
        show = self.show_var.get().strip()
        if not messagebox.askyesno(
                "Voice the whole cast",
                f"Build voices for the cast of '{show or '(unknown show)'}' "
                f"from {len(self.episode_paths)} episode(s)?\n\n"
                f"Characters the show portrays get a trained voice; "
                f"thin/book-only ones are listed for you to handle after."):
            return

        self._auto_cancel = threading.Event()
        self.btn_auto.configure(state="disabled")
        self.btn_auto_cast.configure(state="disabled")
        self.btn_auto_cancel.configure(state="normal")
        read_credits = bool(self.credits_var.get())

        def restore():
            self.btn_auto.configure(state="normal")
            self.btn_auto_cast.configure(state="normal")
            self.btn_auto_cancel.configure(state="disabled")

        def on_voice(name, out):
            voice = out.get("voice")
            if not voice:
                return

            def assign():
                self.reader._set_voice(name, voice)
                names = self._all_voice_names()
                if voice not in names:
                    names.append(voice)
                self.voice_cb.configure(values=names)
                self.reader._rebuild_cast_panel()
            self.after(0, assign)

        def work():
            from audiobook import tv_voice
            try:
                res = tv_voice.auto_create_all(
                    book, list(self.episode_paths), show, roster=roster,
                    read_credits=read_credits, on_voice=on_voice,
                    log=self.log, progress=self._progress,
                    should_stop=self._auto_cancel.is_set)
            finally:
                self.after(0, restore)
            need = res.get("thin", []) + res.get("absent", [])
            self.log(f"Done: {res['built']} voice(s) built and assigned. "
                     f"{len(need)} character(s) need your input "
                     f"(thin audio or not in the show): "
                     f"{', '.join(need[:20])}"
                     f"{' ...' if len(need) > 20 else ''}")
            self.after(0, self._refresh_characters)
        self._run_bg(work)

    def _character_lines(self, max_lines=2):
        """1-2 of this character's actual lines from the book."""
        name = self.char_var.get()
        want = (lambda u: u["speaker"] == name) if name != NARRATOR else \
            (lambda u: u["speaker"] in (NARRATOR, UNKNOWN))
        lines = [u["text"] for u in self.reader.units if want(u)]
        lines.sort(key=lambda t: abs(len(t) - 110))
        return lines[:max_lines]

    def _run_bg(self, fn):
        if self._busy:
            messagebox.showinfo("Voice Lab", "Something is already running.")
            return
        self._busy = True

        def wrap():
            try:
                fn()
            except Exception as e:
                self.log(f"ERROR: {e}")
                self.after(0, lambda: messagebox.showerror("Voice Lab",
                                                           str(e)))
            finally:
                self._busy = False
        threading.Thread(target=wrap, daemon=True,
                         name="voicelab").start()

    # --------------------------------------------------------- test voice --
    def test_voice(self):
        lines = self._character_lines()
        voice = self.voice_var.get()
        if not voice:
            messagebox.showinfo("Voice Lab", "Assign a voice first.")
            return
        if not lines:
            messagebox.showinfo("Voice Lab",
                                "No lines for this character - analyze the "
                                "book first.")
            return
        params = {**synth.DEFAULT_PARAMS,
                  **(self.reader.casting.get("characters", {})
                     .get(self.char_var.get(), {})
                     if self.char_var.get() != NARRATOR
                     else self.reader.casting.get("narrator", {})),
                  "voice": voice}

        def work():
            import sounddevice as sd
            for text in lines:
                self.log(f"test: [{voice}] {text[:70]}")
                x, sr = synth.synth_unit(text, params)
                sd.play(x, sr, blocking=True)
            self.log("test done.")
        self._run_bg(work)

    # ------------------------------------------------------------- train --
    def _pick_files(self):
        f = filedialog.askopenfilenames(
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg")])
        if f:
            self.ref_files = list(f)
            self.files_lbl.configure(text=f"{len(f)} file(s)")

    def train_voice(self):
        import voice_training
        name = self.name_var.get().strip()
        if not name:
            name = self.voice_var.get().strip()
        has_dataset = name and os.path.isfile(os.path.join(
            voice_training.DATASETS_DIR,
            re.sub(r"[^a-zA-Z0-9_\-]", "_", name), "metadata.csv"))
        if not name or (not self.ref_files and not has_dataset):
            messagebox.showinfo(
                "Voice Lab",
                "Set a voice name and add audio files (or pick a voice that "
                "already has a dataset, e.g. one auto-created from the show).")
            return
        epochs = int(self.epochs_var.get())

        def work():
            if self.ref_files:
                self.log(f"=== Preparing dataset for '{name}' ===")
                for msg in voice_training.prepare_voice_dataset(
                        name, self.ref_files, "", transcribe):
                    self.log(msg)
            else:
                self.log(f"=== Using the existing dataset for '{name}' ===")
            self.log(f"=== Training '{name}' ({epochs} epochs) ===")
            for msg in voice_training.run_training(name, epochs, 8, 2, 1e-4):
                self.log(msg)
            self.log("=== Training finished ===")

            def done():
                from voice_training import list_voices
                self.voice_cb.configure(values=list_voices())
                self.voice_var.set(name)
                self._assign_voice()
            self.after(0, done)
        self._run_bg(work)

    # -------------------------------------------------------- wave match --
    def auto_match(self):
        voice = self.voice_var.get()
        if not voice:
            messagebox.showinfo("Voice Lab", "Assign a voice first.")
            return
        grid = wave_match.QUICK_GRID if self.quick_var.get() \
            else wave_match.DEFAULT_GRID
        audio = self.ref_files or None

        def work():
            res = wave_match.auto_match(
                voice, grid=grid, log=self.log, progress=self._progress,
                audio_paths=audio,
                transcribe_fn=transcribe if audio else None)
            self.match_result = res
            b = res["best"]

            def done():
                self.match_lbl.configure(text=(
                    f"Best: {b['params']}  |  match {b['score']:.3f} "
                    f"(speaker {b['speaker']:.3f}, spectrum "
                    f"{b['spectrum']:.3f}, pitch {b['pitch']:.3f}, "
                    f"rhythm {b.get('rhythm', 0.5):.3f})"))
                self.btn_apply.configure(state="normal")
                self.btn_hear.configure(state="normal")
                self.btn_hear_ref.configure(state="normal")
                self._draw_traces()
            self.after(0, done)
        self._run_bg(work)

    def apply_match(self):
        if not self.match_result:
            return
        name = self.char_var.get()
        voice = self.voice_var.get()
        params = self.match_result["best"]["params"]
        casting = self.reader.casting
        slot = casting.setdefault("narrator", {}) if name == NARRATOR else \
            casting.setdefault("characters", {}).setdefault(name, {})
        slot.update(params)
        slot["voice"] = voice
        if self.reader.pdf:
            from audiobook import pdfbook
            pdfbook.save_casting(self.reader.pdf, casting)
        self.log(f"applied to {name}: {params}")
        if self.style_var.get():
            self._copy_pacing(voice)

    def _copy_pacing(self, voice):
        import voice_training
        vdir = os.path.join(voice_training.VOICES_DIR, voice)
        if not os.path.isfile(os.path.join(vdir, "t3_cfg.safetensors")):
            self.log("pacing copy skipped: voice has no trained model "
                     "(PyTorch engine only).")
            return
        clips = list(self.ref_files)
        if not clips:
            wav_dir = os.path.join(voice_training.DATASETS_DIR,
                                   re.sub(r"[^a-zA-Z0-9_\-]", "_", voice),
                                   "wavs")
            if os.path.isdir(wav_dir):
                clips = sorted(
                    (os.path.join(wav_dir, f) for f in os.listdir(wav_dir)
                     if f.endswith(".wav")),
                    key=os.path.getsize, reverse=True)[:6]
        if not clips:
            self.log("pacing copy skipped: no real recordings to take the "
                     "pacing from.")
            return

        def work():
            import json
            import subprocess
            self.log(f"copying the real speaker's pacing into '{voice}' "
                     f"from {len(clips)} recording(s)...")
            code = ("import json, sys, voice_training; "
                    "voice_training.make_style_conds(sys.argv[1], "
                    "json.loads(sys.argv[2]))")
            r = subprocess.run(
                [sys.executable, "-c", code, voice, json.dumps(clips)],
                cwd=voice_training.BASE_DIR, capture_output=True, text=True,
                encoding="utf-8", errors="replace")
            for line in (r.stdout or "").splitlines():
                if line.strip():
                    self.log(line)
            if r.returncode != 0:
                self.log(f"pacing copy failed: "
                         f"{(r.stderr or '').strip().splitlines()[-1:]}")
            else:
                self.log("pacing copied (the PyTorch engine now paces like "
                         "the real recordings; revert via "
                         "conds_voice_only.pt).")
        self._run_bg(work)

    def hear_match(self):
        if not self.match_result:
            return
        b = self.match_result["best"]
        if "wav" not in b:
            return

        def work():
            import sounddevice as sd
            sd.play(b["wav"], b["sr"], blocking=True)
        self._run_bg(work)

    def hear_reference(self):
        r = self.match_result
        if not r or "ref_wav" not in r:
            return

        def work():
            import sounddevice as sd
            sd.play(r["ref_wav"], r["ref_sr"], blocking=True)
        self._run_bg(work)

    def _spec_band(self, spec, w, h, mode):
        import numpy as np
        a = np.asarray(spec, dtype=np.float32)[::-1, :]
        idx_t = np.linspace(0, a.shape[1] - 1, w).astype(int)
        idx_f = np.linspace(0, a.shape[0] - 1, h).astype(int)
        a = a[np.ix_(idx_f, idx_t)]
        v = np.clip(a * 255.0, 0, 255).astype(np.uint8)
        z = np.zeros_like(v)
        if mode == "ref":
            return np.stack([z, v, v], axis=-1)
        if mode == "clone":
            return np.stack([v, (v * 0.55).astype(np.uint8), z], axis=-1)
        return v

    def _draw_traces(self):
        c = self.trace
        c.delete("all")
        r = self.match_result
        if not r or "ref_spec" not in r:
            c.create_text(10, 10, anchor="nw", fill="#888",
                          text="Run Auto-match to see the real voice's "
                               "spectrogram, the clone's, and their overlay.")
            return
        import numpy as np
        from PIL import Image, ImageTk
        w = max(c.winfo_width(), 100)
        h = max(c.winfo_height(), 120)
        band = max(30, h // 3)

        ref = self._spec_band(r["ref_spec"], w, band, "ref")
        clone = self._spec_band(r["best"]["spec"], w, band, "clone")
        overlay = np.clip(ref.astype(int) + clone.astype(int),
                          0, 255).astype(np.uint8)
        img = np.concatenate([ref, overlay, clone], axis=0)
        self._spec_photo = ImageTk.PhotoImage(Image.fromarray(img))
        c.create_image(0, 0, anchor="nw", image=self._spec_photo)
        f = ("Segoe UI", 8, "bold")
        c.create_text(8, 4, anchor="nw", fill="#7fe7e7",
                      text="real voice", font=f)
        c.create_text(8, band + 4, anchor="nw", fill="#ffffff",
                      text="overlay (white = match)", font=f)
        c.create_text(8, 2 * band + 4, anchor="nw", fill="#ffb35c",
                      text="clone (best match)", font=f)
