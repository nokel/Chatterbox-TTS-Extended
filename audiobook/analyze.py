"""
Character / speaker attribution for the audiobook reader.

Reads the whole book once, in order, through the local LM Studio server
(OpenAI-compatible /v1/chat/completions). For every chunk of paragraphs the
model is shown the running character roster plus the passage with its quoted
lines numbered, and returns who speaks each line. A final pass merges
duplicate names ("the keeper" / "The Keeper" / "Old Tom").

Narration units are always attributed to "Narrator". Unattributable quotes
become "Unknown" (cast to the narrator's voice by default).

Entry point:  analyze_book(pdf_path, cfg, progress=fn, log=fn) -> analysis dict
Analysis dict: {"units": [...], "characters": {name: n_lines}, "version": 1}
Cached via pdfbook.save_analysis / load_analysis.
"""

import json
import re

import requests

from . import pdfbook

NARRATOR = "Narrator"
UNKNOWN = "Unknown"

DEFAULTS = {
    "lm_studio_url": "http://127.0.0.1:11434",
    "lm_model": "",
    "chunk_chars": 3500,
}

SYSTEM_PROMPT = (
    "You annotate novels for audiobook production. You are given a passage "
    "from a book. Quoted dialogue lines are marked with numbered tags like "
    "[line 12]. Decide which character speaks each numbered line, using "
    "evidence from the passage (dialogue tags such as 'said X', turn-taking, "
    "who is being addressed, context). Use each character's canonical name, "
    "consistently - prefer a proper name ('Marta') over a description "
    "('the woman') when the passage reveals it. If a line's speaker is "
    "genuinely unidentifiable, use \"Unknown\". "
    "Respond with ONLY a JSON object, no prose, in this exact shape: "
    '{"speakers": {"12": "Name", ...}, "new_characters": ["Name", ...]} '
    "where new_characters lists any speaking characters not already in the "
    "known-characters list you were given."
)

MERGE_PROMPT = (
    "These names were collected as speaking characters of one novel. Some "
    "may be the same person under different names or spellings. Respond "
    "with ONLY a JSON object mapping every alias to its canonical name, "
    'e.g. {"merge": {"the keeper": "The Keeper"}}. Only include real '
    "duplicates; if all names are distinct people return {\"merge\": {}}. "
    'Never merge distinct people, and never merge anything into "Unknown".'
)


class AnalyzeError(RuntimeError):
    pass


def _chat(cfg, messages, max_tokens=3000):
    payload = {
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if cfg.get("lm_model"):
        payload["model"] = cfg["lm_model"]
    url = cfg["lm_studio_url"].rstrip("/") + "/v1/chat/completions"
    try:
        r = requests.post(url, json=payload, timeout=900)
    except requests.ConnectionError as e:
        raise AnalyzeError(
            f"Cannot reach LM Studio at {cfg['lm_studio_url']}. Start its "
            "local server (lms server start) and load a model.") from e
    if r.status_code != 200:
        raise AnalyzeError(f"LM Studio HTTP {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def _parse_json(text):
    """Extract the first balanced {...} object from a model reply."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON in reply: {text[:200]}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"unbalanced JSON in reply: {text[:200]}")


def _chunks(units, chunk_chars):
    """Split units into chunks along paragraph boundaries."""
    out, cur, size = [], [], 0
    cur_para = None
    for u in units:
        if cur and u["para"] != cur_para and size >= chunk_chars:
            out.append(cur)
            cur, size = [], 0
        cur.append(u)
        cur_para = u["para"]
        size += len(u["text"])
    if cur:
        out.append(cur)
    return out


def _render_chunk(chunk):
    lines, last_para = [], None
    for u in chunk:
        if last_para is not None and u["para"] != last_para:
            lines.append("")
        last_para = u["para"]
        if u["kind"] == "dialogue":
            lines.append(f'[line {u["id"]}] "{u["text"]}"')
        else:
            lines.append(u["text"])
    return "\n".join(lines)


def analyze_book(pdf_path, cfg=None, progress=None, log=None,
                 use_cache=True):
    """Full pipeline: extract -> attribute -> merge names -> cache."""
    cfg = {**DEFAULTS, **(cfg or {})}
    log = log or (lambda *_: None)
    progress = progress or (lambda *_: None)

    if use_cache:
        cached = pdfbook.load_analysis(pdf_path)
        if cached and cached.get("version") == 1:
            log("Using cached analysis.")
            return cached

    log("Extracting text...")
    units = pdfbook.extract_units(pdf_path)
    for u in units:
        if u["kind"] == "narration":
            u["speaker"] = NARRATOR

    chunks = _chunks(units, cfg["chunk_chars"])
    roster = []
    n_dialogue = sum(1 for u in units if u["kind"] == "dialogue")
    log(f"{len(units)} units, {n_dialogue} dialogue lines, "
        f"{len(chunks)} chunks.")

    by_id = {u["id"]: u for u in units}
    for ci, chunk in enumerate(chunks):
        progress(ci, len(chunks), f"Reading chunk {ci + 1}/{len(chunks)}")
        ids = [u["id"] for u in chunk if u["kind"] == "dialogue"]
        if not ids:
            continue
        user = (f"KNOWN CHARACTERS SO FAR: "
                f"{json.dumps(roster) if roster else '(none yet)'}\n\n"
                f"PASSAGE:\n{_render_chunk(chunk)}\n\n"
                f"Attribute lines: {ids}")
        reply = _chat(cfg, [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user}])
        try:
            data = _parse_json(reply)
        except ValueError as e:
            log(f"Chunk {ci + 1}: bad model reply ({e}); lines left Unknown.")
            data = {}
        speakers = data.get("speakers", {})
        for name in data.get("new_characters", []):
            name = str(name).strip()
            if name and name not in roster and name != UNKNOWN:
                roster.append(name)
        for uid in ids:
            name = str(speakers.get(str(uid), UNKNOWN)).strip() or UNKNOWN
            by_id[uid]["speaker"] = name
            if name not in roster and name not in (UNKNOWN, NARRATOR):
                roster.append(name)

    # ---- merge duplicate names -------------------------------------------
    counts = {}
    for u in units:
        if u["kind"] == "dialogue":
            counts[u["speaker"]] = counts.get(u["speaker"], 0) + 1
    real = [n for n in counts if n != UNKNOWN]
    if len(real) > 1:
        progress(len(chunks), len(chunks), "Merging character names")
        try:
            reply = _chat(cfg, [
                {"role": "system", "content": MERGE_PROMPT},
                {"role": "user", "content": json.dumps(
                    {"names": {n: counts[n] for n in real}})}],
                max_tokens=1000)
            merge = _parse_json(reply).get("merge", {})
        except (AnalyzeError, ValueError) as e:
            log(f"Name-merge pass skipped: {e}")
            merge = {}
        merge = {a: b for a, b in merge.items()
                 if a in counts and b != UNKNOWN and a != b}
        if merge:
            log("Merged names: " + ", ".join(f"{a} -> {b}"
                                             for a, b in merge.items()))
            for u in units:
                u["speaker"] = merge.get(u["speaker"], u["speaker"])

    characters = {}
    for u in units:
        if u["kind"] == "dialogue":
            characters[u["speaker"]] = characters.get(u["speaker"], 0) + 1

    analysis = {"version": 1, "units": units, "characters": characters}
    pdfbook.save_analysis(pdf_path, analysis)
    progress(len(chunks), len(chunks), "Done")
    log(f"Characters: {json.dumps(characters)}")
    return analysis
