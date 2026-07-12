"""
Audiobook reader window.

Shows the PDF page and reads it aloud with the cast character voices,
highlighting the words being spoken like a screen reader. Click any text to
start reading from there.

Usage:   reader.py [book.pdf]
Normally launched via run_audiobook.ps1 or from SumatraPDF's right-click
"Open with... -> Audiobook Reader" once registered (Tools menu here).

Requires the headless TTS server (run_tts_server.ps1). Character analysis
additionally requires LM Studio's local server with a model loaded.
"""

import json
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from audiobook import analyze, pdfbook, synth  # noqa: E402

ZOOM = 2.0
HL_FILL = "#ffd54a"
NARRATOR = analyze.NARRATOR
UNKNOWN = analyze.UNKNOWN

SUMATRA_SETTINGS = [
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "SumatraPDF",
                 "SumatraPDF-settings.txt"),
    os.path.join(os.environ.get("APPDATA", ""), "SumatraPDF",
                 "SumatraPDF-settings.txt"),
]


def local_voices():
    from voice_training import list_voices
    return list_voices()


class ReaderApp:
    def __init__(self, root, pdf_path=None):
        self.root = root
        self.pdf = None
        self.units = []
        self.page_units = {}      # page -> [unit]
        self.characters = {}      # name -> line count
        self.casting = {}
        self.page_no = 0
        self.n_pages = 0
        self.current_unit = None
        self.player = None
        self.photo = None         # keep a reference or Tk drops the image
        self._busy = False

        root.title("Chatterbox Audiobook Reader")
        root.geometry("1150x820")
        self._build_ui()
        if pdf_path:
            root.after(100, lambda: self.open_book(pdf_path))

    # ---------------------------------------------------------------- UI --
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=4)
        top.pack(side="top", fill="x")
        ttk.Button(top, text="Open PDF...", command=self.open_dialog)\
            .pack(side="left")
        self.btn_analyze = ttk.Button(top, text="Analyze characters",
                                      command=self.run_analysis,
                                      state="disabled")
        self.btn_analyze.pack(side="left", padx=(6, 12))
        self.btn_play = ttk.Button(top, text="▶ Play", width=9,
                                   command=self.toggle_play,
                                   state="disabled")
        self.btn_play.pack(side="left")
        self.btn_stop = ttk.Button(top, text="■ Stop", width=8,
                                   command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=(4, 12))
        ttk.Button(top, text="◀", width=3, command=lambda: self.goto_page(
            self.page_no - 1)).pack(side="left")
        self.lbl_page = ttk.Label(top, text="- / -", width=9,
                                  anchor="center")
        self.lbl_page.pack(side="left")
        ttk.Button(top, text="▶", width=3, command=lambda: self.goto_page(
            self.page_no + 1)).pack(side="left", padx=(0, 12))
        ttk.Button(top, text="Voice Lab...", command=self.open_voice_lab)\
            .pack(side="left")
        ttk.Button(top, text="Add to SumatraPDF",
                   command=self.register_sumatra).pack(side="right")

        main = ttk.Panedwindow(self.root, orient="horizontal")
        main.pack(fill="both", expand=True)

        # page view
        view = ttk.Frame(main)
        self.canvas = tk.Canvas(view, bg="#606060",
                                highlightthickness=0)
        vsb = ttk.Scrollbar(view, orient="vertical",
                            command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(
            -e.delta // 120, "units"))
        main.add(view, weight=4)

        # cast panel
        side = ttk.Frame(main, padding=6)
        main.add(side, weight=0)
        ttk.Label(side, text="Cast", font=("Segoe UI", 11, "bold"))\
            .pack(anchor="w")
        self.cast_frame = ttk.Frame(side)
        self.cast_frame.pack(fill="x", pady=(4, 8))
        self.progress = ttk.Progressbar(side, mode="determinate")
        self.progress.pack(fill="x", pady=(4, 2))
        self.status = tk.StringVar(value="Open a PDF to begin.")
        ttk.Label(side, textvariable=self.status, wraplength=260,
                  foreground="#444").pack(anchor="w", pady=(0, 4))
        self.log_box = tk.Text(side, width=38, height=14, state="disabled",
                               font=("Consolas", 8), wrap="word")
        self.log_box.pack(fill="both", expand=True)

    def log(self, msg):
        def _do():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", str(msg) + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.root.after(0, _do)

    def set_status(self, msg):
        self.root.after(0, lambda: self.status.set(msg))

    # -------------------------------------------------------------- book --
    def open_dialog(self):
        p = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if p:
            self.open_book(p)

    def open_book(self, path):
        self.stop()
        self.pdf = path
        self.n_pages = pdfbook.page_count(path)
        self.root.title(f"Chatterbox Audiobook Reader - "
                        f"{os.path.basename(path)}")
        cached = pdfbook.load_analysis(path)
        if cached:
            self._apply_analysis(cached)
            self.set_status("Loaded cached analysis.")
        else:
            self.units = pdfbook.extract_units(path)
            for u in self.units:
                u["speaker"] = NARRATOR if u["kind"] == "narration" \
                    else UNKNOWN
            self._index_units()
            self.characters = {}
            self.set_status("Text extracted. Run character analysis to "
                            "give characters their own voices.")
        self.casting = pdfbook.load_casting(path)
        if not self.casting.get("narrator", {}).get("voice"):
            voices = local_voices()
            if voices:
                self.casting["narrator"] = {"voice": voices[0]}
        self.btn_analyze.configure(state="normal")
        self.btn_play.configure(state="normal")
        self.btn_stop.configure(state="normal")
        self._rebuild_cast_panel()
        self.goto_page(0)

    def _index_units(self):
        self.page_units = {}
        for u in self.units:
            self.page_units.setdefault(u["page"], []).append(u)

    def _apply_analysis(self, analysis):
        self.units = analysis["units"]
        self.characters = analysis.get("characters", {})
        self._index_units()

    # ---------------------------------------------------------- analysis --
    def run_analysis(self):
        if self._busy or not self.pdf:
            return
        self._busy = True
        self.btn_analyze.configure(state="disabled")
        self.set_status("Reading book with the language model...")

        def work():
            try:
                a = analyze.analyze_book(
                    self.pdf, use_cache=False, log=self.log,
                    progress=self._on_progress)
                self.root.after(0, lambda: self._analysis_done(a, None))
            except Exception as e:
                self.root.after(0, lambda: self._analysis_done(None, e))
        threading.Thread(target=work, daemon=True,
                         name="ab-analyze").start()

    def _on_progress(self, i, n, msg):
        def _do():
            self.progress.configure(maximum=max(n, 1), value=i)
            self.status.set(msg)
        self.root.after(0, _do)

    def _analysis_done(self, analysis, error):
        self._busy = False
        self.btn_analyze.configure(state="normal")
        if error:
            self.set_status(f"Analysis failed: {error}")
            messagebox.showerror("Analysis failed", str(error))
            return
        self._apply_analysis(analysis)
        self._rebuild_cast_panel()
        self.render_page()
        self.set_status(f"Found {len(self.characters)} speaking "
                        "characters. Assign voices, then press Play.")

    # -------------------------------------------------------------- cast --
    def _rebuild_cast_panel(self):
        for w in self.cast_frame.winfo_children():
            w.destroy()
        voices = local_voices()
        rows = [(NARRATOR, sum(1 for u in self.units
                               if u["speaker"] == NARRATOR))]
        rows += sorted(self.characters.items(), key=lambda kv: -kv[1])
        for name, count in rows:
            row = ttk.Frame(self.cast_frame)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=f"{name} ({count})", width=20)\
                .pack(side="left")
            var = tk.StringVar(value=self._voice_of(name) or "")
            cb = ttk.Combobox(row, textvariable=var, values=voices,
                              width=14, state="readonly")
            cb.pack(side="right")
            cb.bind("<<ComboboxSelected>>",
                    lambda _e, n=name, v=var: self._set_voice(n, v.get()))

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
        pdfbook.save_casting(self.pdf, self.casting)
        self.log(f"cast: {name} -> {voice}")

    # -------------------------------------------------------------- page --
    def goto_page(self, page_no):
        if not self.pdf or not (0 <= page_no < self.n_pages):
            return
        self.page_no = page_no
        self.render_page()

    def render_page(self):
        png, w, h, _z = pdfbook.render_page(self.pdf, self.page_no, ZOOM)
        self.photo = tk.PhotoImage(data=png)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw",
                                 tags="page")
        self.canvas.configure(scrollregion=(0, 0, w, h))
        self.lbl_page.configure(
            text=f"{self.page_no + 1} / {self.n_pages}")
        if self.current_unit and self.current_unit["page"] == self.page_no:
            self._draw_highlight(self.current_unit)

    def _draw_highlight(self, unit):
        self.canvas.delete("hl")
        for x0, y0, x1, y1 in unit["rects"]:
            self.canvas.create_rectangle(
                x0 * ZOOM, y0 * ZOOM, x1 * ZOOM, y1 * ZOOM,
                fill=HL_FILL, stipple="gray50", outline="", tags="hl")
        # keep highlight under nothing (image is beneath), scroll into view
        ys = [r[1] for r in unit["rects"]] + [r[3] for r in unit["rects"]]
        if ys:
            _, _, _, total = map(float, self.canvas.cget(
                "scrollregion").split())
            mid = (min(ys) + max(ys)) / 2 * ZOOM
            view_h = self.canvas.winfo_height()
            self.canvas.yview_moveto(
                max(0.0, (mid - view_h / 2) / max(total, 1)))

    # ---------------------------------------------------------- playback --
    def on_click(self, event):
        if not self.units:
            return
        x = self.canvas.canvasx(event.x) / ZOOM
        y = self.canvas.canvasy(event.y) / ZOOM
        for u in self.page_units.get(self.page_no, []):
            for x0, y0, x1, y1 in u["rects"]:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self.play_from(u)
                    return

    def toggle_play(self):
        if self.player and self.player.playing:
            if self.player.paused:
                self.player.resume()
                self.btn_play.configure(text="⏸ Pause")
            else:
                self.player.pause()
                self.btn_play.configure(text="▶ Resume")
            return
        # start from current unit or top of the visible page
        start = self.current_unit
        if start is None:
            page = self.page_units.get(self.page_no, [])
            start = page[0] if page else (self.units[0] if self.units
                                          else None)
        if start is not None:
            self.play_from(start)

    def play_from(self, unit):
        self.stop()
        if not synth.tts_available():
            messagebox.showerror(
                "TTS server not running",
                "Start the TTS server first (run_tts_server.ps1).")
            return
        if not self.casting.get("narrator", {}).get("voice"):
            messagebox.showerror(
                "No narrator voice",
                "Assign a narrator voice in the Cast panel first.")
            return
        self.player = synth.Player(
            self.casting,
            on_unit_start=lambda u: self.root.after(0, self._unit_start, u),
            on_unit_end=lambda u: None,
            on_finish=lambda e: self.root.after(0, self._play_done, e),
            log=self.log)
        self.player.play(self.units, self.units.index(unit))
        self.btn_play.configure(text="⏸ Pause")
        self.set_status("Reading...")

    def _unit_start(self, unit):
        self.current_unit = unit
        if unit["page"] != self.page_no:
            self.goto_page(unit["page"])
        else:
            self._draw_highlight(unit)

    def _play_done(self, error):
        self.btn_play.configure(text="▶ Play")
        self.canvas.delete("hl")
        if error:
            self.set_status(f"Playback stopped: {error}")
            messagebox.showerror("Playback error", str(error))
        else:
            self.set_status("Finished.")

    def stop(self):
        if self.player:
            self.player.stop()
            self.player = None
        self.btn_play.configure(text="▶ Play")
        self.canvas.delete("hl")

    # --------------------------------------------------------- voice lab --
    def open_voice_lab(self):
        from audiobook.voice_lab import VoiceLab
        VoiceLab(self.root, self)

    # ----------------------------------------------------------- sumatra --
    def register_sumatra(self, settings_path=None):
        """Add this reader as an ExternalViewer in SumatraPDF settings."""
        path = settings_path or next(
            (p for p in SUMATRA_SETTINGS if os.path.isfile(p)), None)
        if not path:
            messagebox.showinfo(
                "SumatraPDF not found",
                "SumatraPDF-settings.txt was not found. Install/run "
                "SumatraPDF once, then try again.")
            return False
        pythonw = os.path.join(BASE_DIR, ".venv-amd", "Scripts",
                               "pythonw.exe")
        script = os.path.abspath(__file__)
        entry = (
            "\t[\n"
            f'\t\tCommandLine = "{pythonw}" "{script}" "%1"\n'
            "\t\tName = Audiobook Reader\n"
            "\t\tFilter = *.pdf\n"
            "\t]\n"
        )
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if "Audiobook Reader" in text:
            messagebox.showinfo("SumatraPDF",
                                "Already registered in SumatraPDF.")
            return True
        with open(path + ".bak", "w", encoding="utf-8") as f:
            f.write(text)
        marker = "ExternalViewers [\n"
        if marker in text:
            text = text.replace(marker, marker + entry, 1)
        else:
            text = text.rstrip() + "\n\nExternalViewers [\n" + entry + "]\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        messagebox.showinfo(
            "SumatraPDF",
            "Registered. Close and reopen SumatraPDF (it must not be "
            "running while settings change), then use\n"
            "File -> Open With -> Audiobook Reader.")
        return True


def main():
    pdf = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].strip() else None
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.4)
    except tk.TclError:
        pass
    app = ReaderApp(root, pdf)
    def on_close():
        app.stop()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
