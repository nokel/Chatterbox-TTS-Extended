"""
Bridge to the patched SumatraPDF build.

Our SumatraPDF fork (sumatrapdf/ next to this repo) adds two DDE commands:

    [AudiobookHighlight("<pdf>",<page1based>,"x0 y0 x1 y1;...")]   points, y down
    [AudiobookClear("<pdf>")]

They draw/remove a persistent forward-search-style mark and scroll it into
view, which lets the audiobook reader highlight the words being spoken
inside SumatraPDF's own window. Commands are delivered with the stock
`SumatraPDF.exe -dde "<cmd>"` mechanism, so no DDE client code is needed.

Word rects are merged into one bar per text line before sending - shorter
commands and a cleaner highlight.
"""

import os
import subprocess

ENV_EXE = "CHATTERBOX_SUMATRA_EXE"

_CANDIDATES = [
    # our patched build living next to the Chatterbox repo
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "sumatrapdf", "out", "dbg64",
        "SumatraPDF-dll.exe"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "sumatrapdf", "out", "rel64",
        "SumatraPDF.exe"),
]


def find_exe():
    """Path to the patched SumatraPDF, or None."""
    p = os.environ.get(ENV_EXE)
    if p and os.path.isfile(p):
        return p
    for c in _CANDIDATES:
        if os.path.isfile(c):
            return c
    return None


def merge_line_rects(rects, y_tol=2.0):
    """Merge word rects into one rect per text line."""
    out = []
    for x0, y0, x1, y1 in rects:
        for r in out:
            if abs(r[1] - y0) <= y_tol and abs(r[3] - y1) <= y_tol:
                r[0] = min(r[0], x0)
                r[2] = max(r[2], x1)
                break
        else:
            out.append([x0, y0, x1, y1])
    return out


class SumatraBridge:
    def __init__(self, exe=None, log=None):
        self.exe = exe or find_exe()
        self.log = log or (lambda *_: None)
        self._proc = None

    @property
    def available(self):
        return self.exe is not None

    def _dde(self, cmd):
        # -dde forwards to the running instance (or starts one)
        try:
            subprocess.run([self.exe, "-dde", cmd], timeout=10,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        except (OSError, subprocess.TimeoutExpired) as e:
            self.log(f"sumatra: {e}")

    def open(self, pdf_path, extra_args=()):
        """Make sure the book is open in the patched SumatraPDF."""
        pdf_path = os.path.abspath(pdf_path)
        self._dde(f'[Open("{pdf_path}",0,1,0)]')

    def highlight(self, pdf_path, unit):
        """Highlight a unit's words (unit page is 0-based; Sumatra's 1-based)."""
        pdf_path = os.path.abspath(pdf_path)
        rects = merge_line_rects(unit["rects"])
        blob = ";".join(f"{x0:.1f} {y0:.1f} {x1:.1f} {y1:.1f}"
                        for x0, y0, x1, y1 in rects)
        self._dde(f'[AudiobookHighlight("{pdf_path}",{unit["page"] + 1},'
                  f'"{blob}")]')

    def clear(self, pdf_path):
        pdf_path = os.path.abspath(pdf_path)
        self._dde(f'[AudiobookClear("{pdf_path}")]')
