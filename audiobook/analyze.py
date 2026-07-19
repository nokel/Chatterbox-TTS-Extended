"""
Character / speaker attribution for the audiobook reader.

Reads the whole book once through a local LLM server -- LM Studio, Ollama, or
anything else speaking OpenAI's /v1/chat/completions -- and over several at
once if more than one is configured. For every chunk of paragraphs the model is
shown the running character roster plus the passage with its quoted lines
numbered, and returns who speaks each line. A final pass merges duplicate names
("the keeper" / "The Keeper" / "Old Tom"), which is also what makes it safe to
run the chunks out of order across machines.

Narration units are always attributed to "Narrator". Unattributable quotes
become "Unknown" (cast to the narrator's voice by default).

Entry point:  analyze_book(pdf_path, cfg, progress=fn, log=fn) -> analysis dict
Analysis dict: {"units": [...], "characters": {name: n_lines}, "version": 1}
Cached via pdfbook.save_analysis / load_analysis.
"""

import json
import re
import threading

import requests

from . import pdfbook

NARRATOR = "Narrator"
UNKNOWN = "Unknown"

DEFAULTS = {
    "lm_studio_url": "http://127.0.0.1:11434",
    # Extra OpenAI-compatible endpoints to spread the work over. Reading a
    # novel is hundreds of independent chunk requests, so a second machine
    # roughly halves the wall clock.
    "lm_urls": [],
    "lm_model": "",
    "chunk_chars": 3500,
    # how much of the preceding text to show with each chunk, so an exchange
    # that starts on a chunk boundary still knows who is in the scene
    "context_chars": 900,
    # Reasoning models (gpt-oss et al) otherwise spend the whole token budget
    # thinking and return empty content - measured on gpt-oss-120b: "medium"
    # burned all 400 tokens and answered nothing, "low" answered and ran
    # faster. Working out who speaks a tagged line doesn't need deep
    # reasoning. Ignored by models that don't support the parameter.
    "reasoning_effort": "low",
}

SYSTEM_PROMPT = (
    "You annotate novels for audiobook production. You are given a passage "
    "from a book. Quoted dialogue lines are marked with numbered tags like "
    "[line 12]. Decide which character speaks each numbered line.\n"
    "\n"
    "Work it out in this order:\n"
    "1. A tag naming the speaker: 'said Marta', 'Marta shouted'.\n"
    "2. A tag using a pronoun: 'she said', 'he asked'. Resolve the pronoun "
    "from the surrounding narration and the known characters, and answer with "
    "that character's NAME.\n"
    "3. An action beat instead of a tag: 'Marta folded her arms. \"Fine.\"' "
    "The line belongs to whoever the beat is about.\n"
    "4. No tag at all. Dialogue alternates: in a two-person exchange the turns "
    "go back and forth, so find the nearest line you are sure of and count the "
    "turns from there. Use CONTEXT to see who is in the scene and who spoke "
    "last. A reply that answers a question belongs to the other person.\n"
    "\n"
    "Traps to avoid:\n"
    "- A name INSIDE a quoted line is almost always the person being "
    "ADDRESSED, not the speaker. In '\"What are you playing at, Marie?\"' the "
    "speaker is whoever is talking TO Marie - it is not Marie. Greetings are "
    "the same: '\"Bernard.\"' is someone greeting Bernard.\n"
    "- Never answer with a pronoun ('He', 'She', 'They'). A pronoun is not a "
    "name: resolve it, or answer \"Unknown\".\n"
    "- Never answer with a bare role or description ('the young man', 'the "
    "worker') if the passage or the known-characters list lets you name the "
    "person. Only use a description when the character is genuinely never "
    "named, and then use the same wording every time.\n"
    "- Use each character's canonical name consistently, and reuse the exact "
    "spelling from the known-characters list when it is the same person.\n"
    "- Answer \"Unknown\" only as a last resort, when the speaker really "
    "cannot be worked out. Guessing a plausible name is worse than Unknown, "
    "but so is giving up on a line the alternation makes obvious.\n"
    "\n"
    "CONTEXT, when given, is the text immediately before the passage. Use it "
    "to work out who is present and whose turn it is. Do NOT attribute lines "
    "in CONTEXT - only the numbered lines listed at the end.\n"
    "\n"
    "Respond with ONLY a JSON object, no prose, in this exact shape: "
    '{"speakers": {"12": "Name", ...}, "new_characters": ["Name", ...]} '
    "where new_characters lists any speaking characters not already in the "
    "known-characters list you were given. Give an answer for every line "
    "listed."
)

MERGE_PROMPT = (
    "These names were collected as the speaking characters of one novel, with "
    "the number of lines each was given. Some are the same person written "
    "different ways. Merge:\n"
    "- titles and honorifics: 'Mayor Jahns' = 'Jahns', 'Dr. Sneed' = 'Sneed', "
    "'Senator Thurman' = 'Thurman'\n"
    "- full names against the short form: 'Peter Billings' = 'Peter'\n"
    "- spelling and case variants: 'the keeper' = 'The Keeper'\n"
    "- a description that is plainly the same person as a name\n"
    "\n"
    "Do NOT merge:\n"
    "- different people who share a surname: if 'Thurman' is one character, "
    "'Anna Thurman' may well be his daughter - leave both alone\n"
    "- a relationship word that could be several people ('Father', 'Mother')\n"
    "- anything into 'Unknown'\n"
    "\n"
    "Merge INTO the form a listener would recognise, which is usually the one "
    "with the most lines. Respond with ONLY a JSON object mapping each alias "
    'to its canonical name, e.g. {"merge": {"Mayor Jahns": "Jahns"}}. Include '
    'every duplicate you find. If all names are distinct return {"merge": {}}.'
)

# Never a character, however confidently the model says so. A pronoun stands
# in for a name the passage already gave; kept as a speaker it becomes one
# voice shared by every man in the book. Unknown at least reads as narration.
_NOT_NAMES = {
    "he", "she", "they", "him", "her", "them", "it", "i", "you", "we", "us",
    "his", "hers", "theirs", "someone", "somebody", "anyone", "everyone",
    "no one", "nobody", "a voice", "the voice", "voice", "both", "all",
    "another", "the other", "the others", "unknown", "unnamed", "unidentified",
    "narrator", "n/a", "none", "null", "?",
}


def _is_name(s):
    return isinstance(s, str) and s.strip().lower().strip(".") not in _NOT_NAMES


class AnalyzeError(RuntimeError):
    pass


class OutOfRoom(AnalyzeError):
    """The model spent its context thinking and never answered.

    Its own class because the cure is specific and worth naming: give it a
    bigger context, or let it think less. Retrying the same ask cannot help.
    """


def _lmstudio_models(base, timeout):
    """LM Studio's own API. Says which models are loaded and what each *is*.

    The type matters: an embedding model is listed alongside the chat models
    and cannot answer a chat request at all.
    """
    r = requests.get(base + "/api/v0/models", timeout=timeout)
    if r.status_code != 200:
        return None
    out = []
    for m in r.json().get("data", []):
        if m.get("type") not in ("llm", "vlm"):
            continue                # embeddings can't chat
        if m.get("id"):
            # What it was loaded with, not what it could take: a model whose
            # weights allow 131k is still whatever the person at that machine
            # loaded it as, and that is the wall we actually hit.
            ctx = m.get("loaded_context_length") or m.get("max_context_length")
            out.append({"id": m["id"], "loaded": m.get("state") == "loaded",
                        "ctx": int(ctx) if ctx else 0, "kind": "lmstudio"})
    return out


def _ollama_models(base, timeout):
    """Ollama's own API.

    Ollama answers /v1/models too, but only its native API can say which
    models are in memory (/api/ps) and which can actually hold a conversation
    (/api/show -> capabilities). Both matter here: "which is loaded" is how a
    machine with several models picks one without asking, and an embedding
    model that reaches the chunk loop fails every request.
    """
    r = requests.get(base + "/api/tags", timeout=timeout)
    if r.status_code != 200:
        return None
    names = [m.get("name") or m.get("model") for m in r.json().get("models", [])]
    names = [n for n in names if n]
    if not names:
        return []

    loaded = set()
    try:
        r = requests.get(base + "/api/ps", timeout=timeout)
        if r.status_code == 200:
            for m in r.json().get("models", []):
                n = m.get("name") or m.get("model")
                if n:
                    loaded.add(n)
    except Exception:
        pass

    # Ask what each model can do, but don't turn a listing into a stampede:
    # older Ollama has no capabilities field, and on a long library this is a
    # request per model. When it can't tell us, keep the model - a wrong
    # exclusion is worse than a wrong inclusion, which only costs one failed
    # chunk.
    out = []
    for n in names:
        chat = True
        if len(names) <= 32:
            try:
                r = requests.post(base + "/api/show", json={"model": n},
                                  timeout=timeout)
                if r.status_code == 200:
                    caps = r.json().get("capabilities")
                    if caps is not None:
                        chat = "completion" in caps
            except Exception:
                pass
        if chat:
            out.append({"id": n, "loaded": n in loaded, "ctx": 0,
                        "kind": "ollama"})
    return out


def list_models(lm_url, timeout=10):
    """Chat-capable models a server has. Returns [{"id","loaded","kind"}].

    Tries each server's native API before the lowest common denominator,
    because the native ones answer the two questions that matter - what's
    loaded, and what can chat - and /v1/models answers neither. LM Studio and
    Ollama both get a real implementation; anything else OpenAI-compatible
    (llama.cpp, vLLM, ...) still works through the fallback.

    timeout is short when this is just a "is that machine there?" check: an
    absent host costs the full timeout once per API tried, and that wait ends
    up in front of whoever asked.
    """
    base = (lm_url or "").rstrip("/")
    for probe in (_lmstudio_models, _ollama_models):
        try:
            out = probe(base, timeout)
        except Exception:
            out = None
        if out:
            return out
    try:
        r = requests.get(base + "/v1/models", timeout=timeout)
        if r.status_code == 200:
            return [{"id": m.get("id"), "loaded": False, "ctx": 0,
                     "kind": "openai"}
                    for m in r.json().get("data", []) if m.get("id")]
    except Exception:
        pass
    return []


def server_kind(lm_url, timeout=3):
    """What sort of server this is, for showing to the reader ("" if none)."""
    ms = list_models(lm_url, timeout=timeout)
    return ms[0].get("kind", "") if ms else ""


# How hard each model is allowed to think, matched against its name. A small
# model has to reason its way to what a big one sees at a glance, so it earns
# the thinking time; the big one doesn't need it and is slower for it.
#
# The patterns are regexes, not substrings, and deliberately so: "gpt-oss-120b"
# *contains* "20b", so a plain `"20b" in name` would quietly put the 120b on
# high reasoning and halve the speed of the main machine for no gain.
REASONING_BY_MODEL = (
    (r"(?<!\d)20b", "high"),
    (r"120b", "low"),
)


def _less_effort(effort):
    """The next rung down, or None at the bottom."""
    order = ["high", "medium", "low"]
    if effort in order:
        i = order.index(effort)
        return order[i + 1] if i + 1 < len(order) else None
    return "low" if effort else None


def effort_for(model, cfg):
    """The reasoning_effort to send for a given model ("" = don't send one)."""
    table = cfg.get("reasoning_by_model")
    if table is None:
        table = REASONING_BY_MODEL
    name = (model or "").lower()
    for pat, eff in table:
        if re.search(pat, name):
            return eff
    return cfg.get("reasoning_effort", DEFAULTS["reasoning_effort"])


def pick_remote_model(models, want):
    """Which model to ask another computer for.

    On this machine the reader's choice is an instruction. On someone else's
    it can only be a preference: the panel's dropdown lists the models *here*,
    and the reader was never asked what the machine in the study should run.

    So a model already in that machine's memory wins over the preferred one.
    A server lists every model on its disk, not every model it can actually
    run - naming one it merely has makes LM Studio try to load it, and a
    machine that keeps a 20b loaded because a 120b won't fit will grind, swap,
    or fail. Loaded means someone chose it and it fits.
    """
    ids = [m["id"] for m in models]
    loaded = [m["id"] for m in models if m["loaded"]]
    if want and want in loaded:
        return want             # the preference, and it's ready to go
    if loaded:
        return loaded[0]        # what that machine is set up to run
    if want and want in ids:
        return want             # nothing loaded: the preference can JIT-load
    return ids[0] if ids else ""


def resolve_model(cfg):
    """Which model to analyse with, on this machine.

    Naming a model is what lets LM Studio load it on demand, so this never
    asks the reader to go and load one by hand. Choosing is only necessary
    when the machine has several and none is obviously the one.
    """
    want = (cfg.get("lm_model") or "").strip()
    models = list_models(cfg.get("lm_studio_url"))
    if not models:
        return want or ""       # can't enumerate: let the server decide
    ids = [m["id"] for m in models]
    if want:
        if want in ids:
            return want
        raise AnalyzeError(
            f"The chosen model '{want}' isn't available. Pick one of: "
            + ", ".join(ids))
    if len(ids) == 1:
        return ids[0]           # only one: no choice to make
    loaded = [m["id"] for m in models if m["loaded"]]
    if len(loaded) == 1:
        return loaded[0]        # one is already in memory: obviously that one
    raise AnalyzeError(
        "Several models are available - choose one in the Characters panel: "
        + ", ".join(ids))


def worker_cfgs(cfg, log=None):
    """One config per reachable endpoint, each with its own model resolved.

    Machines don't have the same models installed, so the configured model is
    only a preference: an endpoint that hasn't got it falls back to whatever
    it can pick for itself rather than dropping out of the pool. An endpoint
    that can't be reached at all is dropped, with a note - a dead machine in
    the list shouldn't fail the whole analysis.
    """
    log = log or (lambda *_: None)
    local = (cfg.get("lm_studio_url") or "").strip().rstrip("/")
    urls = []
    for u in [local] + list(cfg.get("lm_urls") or []):
        u = (u or "").strip().rstrip("/")
        if u and u not in urls:
            urls.append(u)

    out = []
    for u in urls:
        c = {**cfg, "lm_studio_url": u, "lm_urls": []}
        models = list_models(u)
        if not models:
            log(f"analysis: skipping {u} (not reachable, or no chat models)")
            continue
        if u == local:
            try:
                c["_resolved_model"] = resolve_model(c)
            except AnalyzeError as e:
                # the preferred model isn't here any more: let it choose
                c2 = {**c, "lm_model": ""}
                try:
                    c2["_resolved_model"] = resolve_model(c2)
                    c = c2
                except AnalyzeError:
                    log(f"analysis: skipping {u} ({e})")
                    continue
        else:
            c["_resolved_model"] = pick_remote_model(
                models, (cfg.get("lm_model") or "").strip())
        c["_ctx"] = next((m.get("ctx") or 0 for m in models
                          if m["id"] == c["_resolved_model"]), 0)
        c["reasoning_effort"] = effort_for(c["_resolved_model"], cfg)
        # Thinking is spent from the context, so a hard-thinking model in a
        # small context can never reach the answer. Measured: gpt-oss-20b on
        # high burns ~6.7k tokens before it starts writing. Say so here, where
        # it's fixable (load it with a bigger context), rather than letting
        # every chunk fail and step itself down.
        if c["reasoning_effort"] == "high" and 0 < c["_ctx"] < 12000:
            log(f"note: {c['_resolved_model']} at {u} is loaded with only "
                f"{c['_ctx']} tokens of context - not enough to think hard in. "
                "Load it with 16k+ in LM Studio to use high reasoning; "
                "until then it drops to medium/low by itself.")
        out.append(c)
        model_name = c["_resolved_model"] or "(default)"
        effort_name = c["reasoning_effort"] or "default"
        ctx_note = f", {c['_ctx']} ctx" if c["_ctx"] else ""
        log(f"analysis worker: {u} -> {model_name} "
            f"(reasoning {effort_name}{ctx_note})")
    if not out:
        raise AnalyzeError(
            "No usable LLM server. Start LM Studio's local server, or Ollama, "
            f"at {cfg.get('lm_studio_url')}.")
    return out


def _fit_budget(cfg, messages, max_tokens):
    """Don't ask for more tokens than the model has room for.

    A model's context is what it was *loaded* with, and the prompt is spent
    from the same pot as the answer. Asking for 8000 completion tokens from a
    model loaded at 8192 with a 1500-token prompt is asking for something that
    cannot happen: it thinks until it hits the wall and returns nothing, and
    the only clue is finish_reason=length. Ask for what's actually there.
    """
    ctx = cfg.get("_ctx") or 0
    if not ctx:
        return max_tokens           # server won't say; let it decide
    # ~3.5 chars per token, rounded against us, plus room for the chat wrapper
    est = sum(len(str(m.get("content", ""))) for m in messages) // 3 + 96
    room = ctx - est - 64
    return max(256, min(max_tokens, room))


def _chat(cfg, messages, max_tokens=3000):
    max_tokens = _fit_budget(cfg, messages, max_tokens)
    payload = {
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    effort = cfg.get("reasoning_effort", DEFAULTS["reasoning_effort"])
    if effort:
        payload["reasoning_effort"] = effort
    # Always name the model: LM Studio loads it just-in-time when asked for by
    # name, so a server with nothing loaded still works. Without this it
    # answers "No models loaded" and expects the user to go and load one.
    model = cfg.get("_resolved_model")
    if model is None:
        model = resolve_model(cfg)
        cfg["_resolved_model"] = model
    if model:
        payload["model"] = model
    url = cfg["lm_studio_url"].rstrip("/") + "/v1/chat/completions"
    try:
        r = requests.post(url, json=payload, timeout=900)
    except requests.ConnectionError as e:
        raise AnalyzeError(
            f"Cannot reach LM Studio at {cfg['lm_studio_url']}. Start its "
            "local server (lms server start).") from e
    if r.status_code != 200:
        raise AnalyzeError(f"LM Studio: {_error_text(r)}")
    choice = r.json()["choices"][0]
    content = choice["message"].get("content") or ""
    if not content.strip() and choice.get("finish_reason") == "length":
        # It thought until it ran out and never got to the answer. Worth its
        # own message: "bad model reply" sends you looking at the prompt, when
        # what happened is the thinking ate the budget.
        ctx = cfg.get("_ctx")
        raise OutOfRoom(
            f"{cfg.get('_resolved_model') or 'the model'} spent all "
            f"{max_tokens} tokens thinking and never answered"
            + (f" (loaded with {ctx} of context)" if ctx else ""))
    return content


def _error_text(r):
    """The human part of an LM Studio error.

    It answers with a JSON envelope, and dumping that raw gives a multi-line
    blob with the useful sentence buried in it - no good in the one-line status
    the Characters panel shows. Pull out .error.message when it's there.
    """
    try:
        msg = r.json().get("error", {}).get("message")
        if msg:
            return " ".join(str(msg).split())[:200]
    except Exception:
        pass
    return f"HTTP {r.status_code}: {' '.join(r.text.split())[:200]}"


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


def _render_context(prev_units, max_chars):
    """The tail of the preceding text, unnumbered.

    A chunk boundary lands in the middle of conversations, and an exchange
    that opens a chunk has nothing above it to attribute from - the model sees
    a bare '"Marie."' with no idea who is in the room. This is the fix: the
    text before, purely to read, never to attribute. It is textual, so it does
    not depend on the previous chunk having been attributed yet - which
    matters, because chunks run out of order across machines.
    """
    out, size = [], 0
    for u in reversed(prev_units):
        t = f'"{u["text"]}"' if u["kind"] == "dialogue" else u["text"]
        if out and size + len(t) > max_chars:
            break
        out.append(t)
        size += len(t)
    return "\n".join(reversed(out))


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
                 use_cache=True, should_stop=None, percent=100):
    """Full pipeline: extract -> attribute -> merge names -> cache.

    percent      analyse only the first N% of the book; the rest keeps its
                 Unknown speakers and is read by the narrator.
    should_stop  checked between chunks; when it returns True the run winds up
                 early and *keeps* what it has (merged, cached, usable). The
                 result is marked incomplete, so a later run resumes rather
                 than redoing the chunks already paid for.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    log = log or (lambda *_: None)
    progress = progress or (lambda *_: None)
    should_stop = should_stop or (lambda: False)

    cached = pdfbook.load_analysis(pdf_path) if use_cache else None
    # A cache written by the other analyser (BookNLP) is not ours to reuse or
    # resume from - its units are attributed differently. Ignore it here; the
    # BookNLP path guards its own cache the same way.
    if cached and cached.get("analyzer", "llm") != "llm":
        cached = None
    if cached and cached.get("version") == 1:
        # old analyses predate partial runs and are complete by definition
        if cached.get("complete", True):
            log("Using cached analysis.")
            return cached

    log("Extracting text...")
    units = pdfbook.extract_units(pdf_path)
    for u in units:
        if u["kind"] == "narration":
            u["speaker"] = NARRATOR

    # Resume: a stopped or partial run left attributions behind. Seeding from
    # them is what makes Analyse continue instead of starting over.
    # "attr" marks a line the model has already looked at - which is not the
    # same as a line it could name. Some lines are genuinely unattributable
    # and stay Unknown; keyed on the speaker alone, every resume would ask
    # about those same lines again forever.
    resumed = 0
    if cached and cached.get("version") == 1:
        prev = {u.get("id"): u for u in cached.get("units", [])}
        for u in units:
            pu = prev.get(u["id"])
            if pu and pu.get("attr"):
                u["attr"] = True
                u["speaker"] = pu.get("speaker") or UNKNOWN
                if u["kind"] == "dialogue":
                    resumed += 1
        if resumed:
            log(f"resuming: {resumed} lines already read")

    chunks = _chunks(units, cfg["chunk_chars"])
    n_dialogue = sum(1 for u in units if u["kind"] == "dialogue")
    log(f"{len(units)} units, {n_dialogue} dialogue lines, "
        f"{len(chunks)} chunks.")

    # how much of the book to look at
    percent = max(1, min(int(percent or 100), 100))
    limit_units = len(units) if percent >= 100 else max(
        1, len(units) * percent // 100)
    if percent < 100:
        log(f"analysing the first {percent}% ({limit_units}/{len(units)} lines)")

    by_id = {u["id"]: u for u in units}

    def todo(chunk):
        """Chunk still needing work: in range, has dialogue, not yet read."""
        ids = [u["id"] for u in chunk if u["kind"] == "dialogue"]
        if not ids:
            return []
        if chunk[0]["id"] >= limit_units:
            return []
        return [i for i in ids if not by_id[i].get("attr")]

    pending = [ci for ci, c in enumerate(chunks) if todo(c)]
    workers = worker_cfgs(cfg, log)

    roster = []
    for u in units:
        s = u.get("speaker")
        if u["kind"] == "dialogue" and s and s not in (UNKNOWN, NARRATOR) \
                and s not in roster:
            roster.append(s)

    lock = threading.Lock()
    nxt = [0]
    done = [0]
    stopped = [False]
    failed = []       # chunks the model wouldn't answer for; retried next run

    def run_chunk(ci, wcfg):
        chunk = chunks[ci]
        ids = [u["id"] for u in chunk if u["kind"] == "dialogue"]
        with lock:
            known = list(roster)      # a snapshot: chunks run out of order now
        ctx = _render_context(chunks[ci - 1], cfg["context_chars"]) if ci else ""
        user = (f"KNOWN CHARACTERS SO FAR: "
                f"{json.dumps(known) if known else '(none yet)'}\n\n"
                + (f"CONTEXT (do not attribute):\n{ctx}\n\n" if ctx else "")
                + f"PASSAGE:\n{_render_chunk(chunk)}\n\n"
                f"Attribute lines: {ids}")
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user}]

        # Ask twice before giving up. A reasoning model sometimes spends the
        # whole budget thinking and answers with nothing, and a long chunk can
        # run the reply out of tokens mid-JSON; both are fixed by asking again
        # with more room. On the Silo omnibus this was 42 of 665 chunks -
        # ~6% of the book left Unknown for want of a retry.
        #
        # The budget starts higher when the model is thinking hard, because
        # thinking is spent from it: a small model on high reasoning would
        # otherwise run out on the first ask every single time, and pay for
        # two calls per chunk to get one answer.
        budget = 8000 if wcfg.get("reasoning_effort") == "high" else 3000
        effort = wcfg.get("reasoning_effort")
        data = None
        for attempt in (1, 2):
            try:
                data = _parse_json(_chat({**wcfg, "reasoning_effort": effort},
                                         messages, max_tokens=budget))
                break
            except OutOfRoom as e:
                # More room is exactly what it cannot have - the context is
                # the wall. The other lever is to make it think less.
                nxt_effort = _less_effort(effort)
                log(f"Chunk {ci + 1}: {e}"
                    + (f"; retrying at reasoning={nxt_effort}."
                       if nxt_effort and attempt == 1 else ""))
                if not nxt_effort:
                    break
                effort = nxt_effort
            except ValueError as e:
                log(f"Chunk {ci + 1}: bad model reply ({e})"
                    + ("; retrying with more room." if attempt == 1 else ""))
                budget *= 2
        if data is None:
            # Deliberately not marked as read: marking it would make this
            # chunk Unknown forever, since resume skips anything already
            # attributed. Left alone, Continue analysing picks it up.
            with lock:
                failed.append(ci)
            return

        speakers = data.get("speakers", {})
        with lock:
            for name in data.get("new_characters", []):
                name = str(name).strip()
                if name and name not in roster and _is_name(name):
                    roster.append(name)
            for uid in ids:
                name = str(speakers.get(str(uid), UNKNOWN)).strip() or UNKNOWN
                if not _is_name(name):
                    name = UNKNOWN      # a pronoun is not an answer
                by_id[uid]["speaker"] = name
                by_id[uid]["attr"] = True     # read, even if left Unknown
                if name not in roster and name not in (UNKNOWN, NARRATOR):
                    roster.append(name)

    def work(wcfg):
        while True:
            with lock:
                if should_stop():
                    stopped[0] = True
                    return
                if nxt[0] >= len(pending):
                    return
                ci = pending[nxt[0]]
                nxt[0] += 1
            try:
                run_chunk(ci, wcfg)
            except AnalyzeError as e:
                # One endpoint dying shouldn't lose the whole book - but it
                # must not pass for a finished one either. Unrecorded, a run
                # where every request failed still called itself complete, and
                # cached a book of Unknowns that Re-analyse then declined to
                # look at again.
                log(f"chunk {ci + 1} failed on {wcfg['lm_studio_url']}: {e}")
                with lock:
                    failed.append(ci)
            with lock:
                done[0] += 1
                progress(done[0], len(pending),
                         f"Reading chunk {done[0]}/{len(pending)}")

    if pending:
        threads = [threading.Thread(target=work, args=(w,), daemon=True,
                                    name=f"ab-analyze-{i}")
                   for i, w in enumerate(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    if should_stop():
        stopped[0] = True
    if stopped[0]:
        log(f"analysis stopped after {done[0]}/{len(pending)} chunks; "
            "keeping what's attributed so far")
    if failed:
        log(f"{len(failed)} of {len(pending)} chunks got no usable answer, "
            "even after a retry; Continue analysing will try them again")

    # Lines nobody got to - stopped early, or past the percent limit - still
    # have the speaker=None that extract_units gave them. Name them Unknown
    # (which is cast to the narrator) so that "not read yet" and "read but
    # unattributable" end up in the same, handled place. Left as None they
    # become a character literally called "null" with hundreds of lines, and
    # every consumer that lower()s a speaker name falls over.
    for u in units:
        if u["kind"] == "dialogue" and not u.get("speaker"):
            u["speaker"] = UNKNOWN

    # ---- merge duplicate names -------------------------------------------
    # This matters more now than it did: chunks no longer run in order, so a
    # worker can meet a character before the roster naming them has reached
    # it. This pass is what reconciles the aliases that produces.
    # A pronoun that survived the per-chunk check (an old cached analysis, or
    # a model that ignored the rule) is caught here rather than becoming a
    # character. "He" had 98 lines on the Silo omnibus.
    for u in units:
        if u["kind"] == "dialogue" and not _is_name(u.get("speaker")):
            u["speaker"] = UNKNOWN

    counts = {}
    for u in units:
        if u["kind"] == "dialogue":
            counts[u["speaker"]] = counts.get(u["speaker"], 0) + 1
    real = [n for n in counts if n != UNKNOWN]
    if len(real) > 1:
        progress(len(pending), len(pending), "Merging character names")
        # Sorted by line count, so if the reply does get cut off it's the
        # walk-ons that are lost, not the leads. The budget is sized for a
        # real cast: the omnibus reached 219 names, and 1000 tokens could not
        # have expressed the answer even if the model had found every alias.
        names = {n: counts[n] for n in sorted(real, key=lambda n: -counts[n])}
        try:
            reply = _chat(workers[0], [
                {"role": "system", "content": MERGE_PROMPT},
                {"role": "user", "content": json.dumps({"names": names})}],
                max_tokens=6000)
            merge = _parse_json(reply).get("merge", {})
        except (AnalyzeError, ValueError) as e:
            log(f"Name-merge pass skipped: {e}")
            merge = {}
        merge = {a: b for a, b in merge.items()
                 if a in counts and a != b and _is_name(b)}
        # Follow chains ("Mayor Jahns" -> "Jahns" -> "Marie Jahns") and refuse
        # to go round in circles, which a model occasionally asks for.
        def resolve(n, seen=None):
            seen = seen or {n}
            t = merge.get(n)
            if t is None or t in seen:
                return n
            seen.add(t)
            return resolve(t, seen)

        merge = {a: resolve(a) for a in merge}
        merge = {a: b for a, b in merge.items() if a != b}
        if merge:
            log("Merged names: " + ", ".join(f"{a} -> {b}"
                                             for a, b in merge.items()))
            for u in units:
                u["speaker"] = merge.get(u["speaker"], u["speaker"])

    characters = {}
    for u in units:
        if u["kind"] == "dialogue":
            characters[u["speaker"]] = characters.get(u["speaker"], 0) + 1

    # A stopped or part-of-the-book run is still worth keeping: it's real work
    # and the book reads better with it than without. Marking it incomplete is
    # what lets the next Analyse pick up rather than start again.
    read = sum(1 for u in units if u["kind"] == "dialogue" and u.get("attr"))
    # Chunks that never answered leave the book genuinely unfinished. Calling
    # it complete would bury them: the panel would offer "Re-analyse" (start
    # over) rather than "Continue analysing" (retry just those).
    complete = (not stopped[0]) and percent >= 100 and not failed
    analysis = {
        "version": 1,
        "analyzer": "llm",
        "units": units,
        "characters": characters,
        "complete": complete,
        "lines_read": read,
        "lines_total": n_dialogue,
    }
    pdfbook.save_analysis(pdf_path, analysis)
    progress(len(pending), len(pending), "Done")
    log(f"Characters: {json.dumps(characters)}")
    if not complete:
        log(f"partial analysis: {read}/{n_dialogue} lines read")
    return analysis
