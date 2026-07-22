"""Attribute speakers with BookNLP, as a drop-in for the LLM analyser.

The rest of the pipeline is built around one thing: `pdfbook.extract_units`,
whose units carry the on-page rectangles the reader highlights as it speaks.
BookNLP works from plain text and knows nothing about the PDF, so it can't
produce those. It doesn't need to. This bridge keeps extract_units as the
source of units and uses BookNLP only for the one thing it's for - deciding who
speaks each line - then writes those speakers back onto the units the reader
already has.

The join is by position in the text. The units are turned back into a document
- one paragraph per unit-paragraph, each dialogue unit re-wrapped in quotation
marks - and every dialogue unit's character span in that document is recorded.
BookNLP reports each quote's span into the same document, so a quote maps to the
unit whose span it falls in. Same text in, so the spans line up.

The spans are in CHARACTERS. BookNLP's .tokens columns are named byte_onset /
byte_offset but they are code-point offsets, not UTF-8 byte offsets (verified:
a smart quote "u201c" reports a width of 1, not 3). Counting bytes here - as an
earlier version did, on the plausible-but-wrong assumption that a name like that
meant bytes - drifts by the multibyte surplus at the first curly quote and mis-
assigns every quote after it. So build and overlap the spans in character space,
exactly as BookNLP counts.

Output is the exact dict `analyze.analyze_book` returns, so casting, caching and
playback don't know or care which analyser ran.
"""

import os
import shutil
import tempfile

from audiobook import analyze, cast_dedup, pdfbook
from audiobook.attribution import IdentityMatrix
from audiobook.booknlp_attribution import BookNLPDoc, attribute

NARRATOR = analyze.NARRATOR
UNKNOWN = analyze.UNKNOWN


def _build_document(units):
    """Units -> (text, spans). text is a real document with paragraph breaks;
    spans maps each dialogue unit's id to its (start, end) CHARACTER range in
    text.

    Dialogue is re-wrapped in quotes so BookNLP sees it as speech; the units
    dropped the marks when they were extracted. The string is assembled piece
    by piece and offsets read off the buffer as it grows, so the spans are
    exact by construction rather than recomputed.

    Offsets are characters, matching BookNLP's own token offsets (see the module
    docstring - its "byte" columns are actually code-point offsets).
    """
    spans = {}
    buf = []
    nchars = 0
    cur_para = None

    for u in units:
        if u["para"] != cur_para:
            if cur_para is not None:
                buf.append("\n\n")
                nchars += 2
            cur_para = u["para"]
        elif buf and not buf[-1].endswith("\n"):
            buf.append(" ")
            nchars += 1

        piece = '"' + u["text"] + '"' if u["kind"] == "dialogue" else u["text"]
        if u["kind"] == "dialogue":
            spans[u["id"]] = (nchars, nchars + len(piece))
        buf.append(piece)
        nchars += len(piece)

    return "".join(buf), spans


def _char_range(doc, quote):
    """The quote's (start, end) CHARACTER offsets into the document.

    BookNLP's byte_onset/byte_offset columns are code-point offsets despite the
    name (see module docstring), and _build_document numbers spans the same way,
    so these line up directly.
    """
    a = doc.by_id.get(int(quote["quote_start"]))
    b = doc.by_id.get(int(quote["quote_end"]))
    if not a or not b:
        return None
    try:
        return int(a["byte_onset"]), int(b["byte_offset"])
    except (TypeError, ValueError, KeyError):
        return None


def _assign(units, doc, marked):
    """Write each attributed quote's speaker onto the unit it overlaps."""
    _, spans = _build_document(units)
    by_id = {u["id"]: u for u in units}
    # dialogue unit ids in text order, with their spans
    ordered = sorted(spans.items(), key=lambda kv: kv[1][0])

    assigned = 0
    for q, m in zip(doc.quotes, marked):
        br = _char_range(doc, q)
        if br is None:
            continue
        qs, qe = br
        # the dialogue unit whose span best overlaps this quote's byte range
        best, best_ov = None, 0
        for uid, (s, e) in ordered:
            ov = min(qe, e) - max(qs, s)
            if ov > best_ov:
                best, best_ov = uid, ov
        if best is not None and best_ov > 0 and m.get("speaker"):
            by_id[best]["speaker"] = m["speaker"]
            assigned += 1
    return assigned


def analyze_book(pdf_path, cfg=None, progress=None, log=None,
                 use_cache=True, should_stop=None, percent=100):
    """BookNLP speaker attribution. Same signature and return as the LLM path.

    percent/should_stop are accepted for interface parity. BookNLP runs the
    whole book in one pass - there are no per-chunk requests to stop between or
    to limit - so a partial run isn't meaningful here; it always completes.
    """
    cfg = cfg or {}
    log = log or (lambda *_: None)
    progress = progress or (lambda *_: None)

    cached = pdfbook.load_analysis(pdf_path) if use_cache else None
    if cached and cached.get("version") == 1 and cached.get("complete", True) \
            and cached.get("analyzer") == "booknlp":
        log("Using cached analysis.")
        return cached

    progress(0, 4, "Reading the book")
    units = pdfbook.extract_units(pdf_path)
    for u in units:
        if u["kind"] == "narration":
            u["speaker"] = NARRATOR
    text, _ = _build_document(units)

    progress(1, 4, "Finding characters and quotes (BookNLP)")
    out_dir = tempfile.mkdtemp(prefix="booknlp_")
    try:
        from audiobook import booknlp_compat
        booknlp_compat.apply()
        from booknlp.booknlp import BookNLP
        model = cfg.get("booknlp_model", "big")
        in_path = os.path.join(out_dir, "book.txt")
        with open(in_path, "w", encoding="utf-8") as f:
            f.write(text)
        bn = BookNLP("en", {"pipeline": "entity,quote,coref", "model": model})
        bn.process(in_path, out_dir, "book")

        progress(2, 4, "Assigning speakers")
        doc = BookNLPDoc(out_dir, "book")
        identity = IdentityMatrix()
        marked = attribute(doc, identity=identity)
        _assign(units, doc, marked)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    # anything left unspoken-for reads as the narrator, same as the LLM path
    for u in units:
        if u["kind"] == "dialogue" and not u.get("speaker"):
            u["speaker"] = UNKNOWN

    progress(3, 4, "Reviewing the cast")
    cast_dedup.dedup(units, identity, cfg=cfg, log=log)

    characters = {}
    for u in units:
        if u["kind"] == "dialogue":
            characters[u["speaker"]] = characters.get(u["speaker"], 0) + 1
    n_dialogue = sum(1 for u in units if u["kind"] == "dialogue")

    analysis = {
        "version": 1,
        "analyzer": "booknlp",
        "units": units,
        "characters": characters,
        "complete": True,
        "lines_read": n_dialogue,
        "lines_total": n_dialogue,
    }
    pdfbook.save_analysis(pdf_path, analysis)
    progress(4, 4, "Done")
    log(f"BookNLP: {len(characters)} characters over {n_dialogue} lines")
    return analysis
