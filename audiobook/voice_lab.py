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
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

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


class VoiceLab(tk.Toplevel):
    def __init__(self, parent, reader):
        super().__init__(parent)
        self.reader = reader
        self.title("Voice Lab")
        self.geometry("920x760")
        self.ref_files = []
        self.match_result = None
        self._busy = False
        self._build_ui()
        self._refresh_characters()

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
        from voice_training import list_voices
        chars = [NARRATOR] + sorted(self.reader.characters)
        self.char_cb.configure(values=chars)
        if not self.char_var.get() and chars:
            self.char_var.set(chars[0])
        self.voice_cb.configure(values=list_voices())
        self._load_character()

    def _load_character(self):
        self.voice_var.set(self.reader._voice_of(self.char_var.get()) or "")

    def _assign_voice(self):
        self.reader._set_voice(self.char_var.get(), self.voice_var.get())
        self.reader._rebuild_cast_panel()

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
        if not name or not self.ref_files:
            messagebox.showinfo("Voice Lab",
                                "Set a voice name and add audio files.")
            return
        epochs = int(self.epochs_var.get())

        def work():
            self.log(f"=== Preparing dataset for '{name}' ===")
            for msg in voice_training.prepare_voice_dataset(
                    name, self.ref_files, "", transcribe):
                self.log(msg)
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
                    f"{b['spectrum']:.3f}, pitch {b['pitch']:.3f})"))
                self.btn_apply.configure(state="normal")
                self.btn_hear.configure(state="normal")
                self._draw_traces()
            self.after(0, done)
        self._run_bg(work)

    def apply_match(self):
        if not self.match_result:
            return
        name = self.char_var.get()
        params = self.match_result["best"]["params"]
        casting = self.reader.casting
        slot = casting.setdefault("narrator", {}) if name == NARRATOR else \
            casting.setdefault("characters", {}).setdefault(name, {})
        slot.update(params)
        slot["voice"] = self.voice_var.get()
        if self.reader.pdf:
            from audiobook import pdfbook
            pdfbook.save_casting(self.reader.pdf, casting)
        self.log(f"applied to {name}: {params}")

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

    def _draw_traces(self):
        c = self.trace
        c.delete("all")
        if not self.match_result:
            c.create_text(10, 10, anchor="nw", fill="#888",
                          text="Run Auto-match to see the reference wave "
                               "trace overlaid with the clone's.")
            return
        w = max(c.winfo_width(), 100)
        h = max(c.winfo_height(), 80)
        half = h / 2

        def draw(traceset, color, label, y0):
            if not traceset:
                return
            n = len(traceset)
            amp = max(max(abs(lo), abs(hi)) for lo, hi in traceset) or 1.0
            mid = y0 + half / 2
            scale = (half / 2 - 8) / amp
            for i, (lo, hi) in enumerate(traceset):
                x = 4 + i * (w - 8) / n
                c.create_line(x, mid - hi * scale, x, mid - lo * scale,
                              fill=color)
            c.create_text(8, y0 + 4, anchor="nw", fill=color, text=label,
                          font=("Segoe UI", 8, "bold"))

        draw(self.match_result["ref_trace"], "#9e9e9e",
             "reference (real recording)", 0)
        draw(self.match_result["best"]["trace"], "#4fc3f7",
             "clone (best match)", half)
