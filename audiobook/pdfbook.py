"""
PDF ebook extraction for the audiobook reader.

Loads a PDF with PyMuPDF and turns it into a list of speakable *units*.
A unit is the smallest thing the reader speaks and highlights in one voice:
either a stretch of narration or one quoted utterance. Every unit keeps the
bounding rectangles of its words so the reader can highlight exactly the
text being spoken, screen-reader style.

Unit dict:
    id      int                unique, in reading order
    page    int                0-based page number
    para    int                paragraph index (global, for LLM context)
    text    str                the text to speak
    kind    "dialogue"|"narration"
    rects   [[x0,y0,x1,y1]..]  word rectangles in PDF coordinates
    speaker str|None           filled in by analyze.py ("Narrator" for narration)

Analysis artifacts are cached per book in audiobook/cache/<sha1>/.
"""

import hashlib
import json
import os
import re

import fitz  # PyMuPDF

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT = os.path.join(BASE_DIR, "audiobook", "cache")

OPEN_QUOTES = "“‘«"
CLOSE_QUOTES = "”’»"
STRAIGHT = "\""

_ABBREV = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Prof|St|Sr|Jr|vs|etc|i\.e|e\.g|No)\.$", re.I)


def book_hash(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def cache_dir(path):
    d = os.path.join(CACHE_ROOT, book_hash(path))
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- extract ---

def _page_words(page):
    """Words in reading order: (rect, text)."""
    words = page.get_text("words")  # x0,y0,x1,y1, word, block, line, wno
    words.sort(key=lambda w: (w[5], w[6], w[7]))
    return [([w[0], w[1], w[2], w[3]], w[4]) for w in words if w[4].strip()]


def _paragraphs(page):
    """Group a page's words into paragraphs using text blocks."""
    words = page.get_text("words")
    words.sort(key=lambda w: (w[5], w[6], w[7]))
    paras, cur, cur_block = [], [], None
    for w in words:
        if not w[4].strip():
            continue
        if cur_block is not None and w[5] != cur_block and cur:
            paras.append(cur)
            cur = []
        cur_block = w[5]
        cur.append(([w[0], w[1], w[2], w[3]], w[4]))
    if cur:
        paras.append(cur)
    return paras


def _is_sentence_end(word_text):
    t = word_text.rstrip(CLOSE_QUOTES + STRAIGHT + "')")
    if not t or t[-1] not in ".!?":
        return False
    if _ABBREV.search(t):
        return False
    return True


def _flush(units, buf, page_no, para_no, kind):
    text = " ".join(w for _, w in buf).strip()
    # Strip surrounding quote marks from dialogue: the voice actor doesn't
    # pronounce them, and the LLM sees clean utterances.
    if kind == "dialogue":
        text = text.strip(OPEN_QUOTES + CLOSE_QUOTES + STRAIGHT).strip()
    if text:
        units.append({
            "id": len(units), "page": page_no, "para": para_no,
            "text": text, "kind": kind,
            "rects": [r for r, _ in buf], "speaker": None,
        })
    buf.clear()


def extract_units(pdf_path):
    """Whole book -> list of units in reading order."""
    doc = fitz.open(pdf_path)
    units = []
    para_no = -1
    for page_no in range(len(doc)):
        for para in _paragraphs(doc[page_no]):
            para_no += 1
            buf, in_quote = [], False
            for rect, word in para:
                first, last = word[:1], word.rstrip()[-1:]
                opened_here = False
                if not in_quote and (first in OPEN_QUOTES or first in STRAIGHT):
                    _flush(units, buf, page_no, para_no, "narration")
                    in_quote = opened_here = True
                buf.append((rect, word))
                if in_quote:
                    closes = (last in CLOSE_QUOTES
                              or (last in STRAIGHT
                                  and not (opened_here and len(word) == 1)))
                    if closes:
                        _flush(units, buf, page_no, para_no, "dialogue")
                        in_quote = False
                elif _is_sentence_end(word):
                    _flush(units, buf, page_no, para_no, "narration")
            # paragraph ended: flush whatever is left (unclosed quotes count
            # as dialogue - fiction re-opens quotes on the next paragraph)
            _flush(units, buf, page_no, para_no,
                   "dialogue" if in_quote else "narration")
    doc.close()
    return units


def page_count(pdf_path):
    doc = fitz.open(pdf_path)
    n = len(doc)
    doc.close()
    return n


def render_page(pdf_path, page_no, zoom=2.0):
    """Render one page -> (png_bytes, pixel_w, pixel_h, zoom)."""
    doc = fitz.open(pdf_path)
    page = doc[page_no]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    data = pix.tobytes("png")
    size = (pix.width, pix.height)
    doc.close()
    return data, size[0], size[1], zoom


# ------------------------------------------------------------------ cache ---

def load_analysis(pdf_path):
    p = os.path.join(cache_dir(pdf_path), "analysis.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_analysis(pdf_path, data):
    p = os.path.join(cache_dir(pdf_path), "analysis.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_casting(pdf_path):
    p = os.path.join(cache_dir(pdf_path), "casting.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_casting(pdf_path, data):
    p = os.path.join(cache_dir(pdf_path), "casting.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
