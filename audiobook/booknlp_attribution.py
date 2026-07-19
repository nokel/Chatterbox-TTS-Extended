"""Speaker attribution from BookNLP output, the way the standard taxonomy does.

Quotation attribution research (the Project Dialogism Novel Corpus and the work
around it) sorts every quote into three kinds, in falling order of certainty:

  explicit   a speech verb with a NAMED subject     "...," said Bernard.
  anaphoric  a speech verb with a pronoun/role       "...," she asked.
  implicit   no speech tag at all                    "...," and the beat, if
                                                      any, is a reaction

BookNLP is near-perfect on explicit, good on anaphoric (it resolves the pronoun
by coreference), and weak on implicit - which is the hard, unavoidable residue
where a reader falls back on turn-taking. So this module trusts BookNLP exactly
where BookNLP is strong and no further:

  * It reads the tag itself off the dependency parse BookNLP already produced.
    A speech verb (say/ask/...) next to the quote whose subject is a name is
    explicit; whose subject is a pronoun is anaphoric, and *that pronoun's*
    coreference - BookNLP's own - names the speaker.
  * A beat is not a tag. "...playing at, Marie?" / "Jahns felt her temperature
    rise." has no speech verb, so it is implicit, and the name in the beat
    ("Jahns") is precisely the trap to avoid: she is reacting, not speaking.
  * Only the implicit quotes fall through to alternation, and only within a
    paragraph structure that actually marks the turns (see pdfbook, and feed
    BookNLP text with real paragraph breaks so its paragraph_ID is meaningful).

Names are canonicalised through an [[IdentityMatrix]], so aliases and a later
reveal read as one voice.
"""

import csv
import os
import re
from collections import Counter

from audiobook.attribution import IdentityMatrix

csv.field_size_limit(10 ** 7)

PRONOUNS = {"he", "she", "they", "him", "her", "them", "i", "you", "we", "it",
            "someone", "somebody", "who", "that", "this"}


def _read_tsv(path):
    # QUOTE_NONE is essential: these files contain tokens that are literally a
    # double-quote character (every line of dialogue has them), and the default
    # reader treats " as a field quote - it swallows the delimiter and shifts
    # every following column, so byte offsets come back as POS tags. Turning
    # quote handling off reads each tab-separated field verbatim.
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


class BookNLPDoc:
    """The three output files, indexed for attribution."""

    def __init__(self, out_dir, idd):
        self.tokens = _read_tsv(os.path.join(out_dir, idd + ".tokens"))
        self.quotes = _read_tsv(os.path.join(out_dir, idd + ".quotes"))
        self.entities = _read_tsv(os.path.join(out_dir, idd + ".entities"))

        self.by_id = {}
        for t in self.tokens:
            tid = _int(t["token_ID_within_document"])
            if tid is not None:
                self.by_id[tid] = t

        # token id -> coref id, from every entity mention span
        self.coref_of = {}
        for e in self.entities:
            s, en = _int(e["start_token"]), _int(e["end_token"])
            if s is None or en is None:
                continue
            for tid in range(s, en + 1):
                self.coref_of[tid] = e["COREF"]

        # coref id -> display name (its most frequent proper mention)
        counts = {}
        for e in self.entities:
            if e["cat"] == "PER" and e["prop"] == "PROP":
                counts.setdefault(e["COREF"], {})
                counts[e["COREF"]][e["text"]] = counts[e["COREF"]].get(e["text"], 0) + 1
        self.name_of = {c: max(d, key=d.get) for c, d in counts.items()}

    def display(self, coref):
        return self.name_of.get(coref)

    def paragraph(self, token_id):
        t = self.by_id.get(token_id)
        return _int(t["paragraph_ID"]) if t else None


def _int(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


# A quote whose attributed mention sits within this many tokens carries a real
# tag - a reader would see the speaker right there. Beyond it, BookNLP had to
# reach, which is the signature of an implicit quote.
TAG_DISTANCE = 4


def classify(doc, q):
    """One quote -> (kind, coref_id_or_None).

    The signal is BookNLP's own quote-mention link, not a re-derived one.
    BookNLP is a trained, benchmarked tagger; re-detecting tags with a verb
    list is both redundant and weaker (it missed more than half of them). So
    read the answer off what BookNLP already linked:

      * the mention it chose is right next to the quote -> a tag. A proper name
        makes it explicit; a pronoun or common noun makes it anaphoric, already
        coreference-resolved to a character.
      * the mention is far away -> BookNLP had nothing adjacent to work with,
        which is exactly an implicit quote. Hand it to alternation.

    kind is 'explicit', 'anaphoric' or 'implicit'; coref is BookNLP's entity id
    for the tagged speaker, or None when implicit.
    """
    qs, qe = _int(q["quote_start"]), _int(q["quote_end"])
    ms, me = _int(q["mention_start"]), _int(q["mention_end"])
    coref = q.get("char_id") or None

    if ms is None or me is None:
        return "implicit", None
    dist = min(abs(ms - qe), abs(qs - me))
    if dist > TAG_DISTANCE:
        return "implicit", None

    phrase = (q.get("mention_phrase") or "").strip()
    head = doc.by_id.get(ms) or doc.by_id.get(me)
    is_name = bool(phrase[:1].isupper()) and phrase.lower() not in PRONOUNS
    if head is not None:
        is_name = head["POS_tag"] == "PROPN" or head["fine_POS_tag"] == "NNP"
    return ("explicit" if is_name else "anaphoric"), coref


def _is_break(para_a, para_b):
    """A structural break between two paragraph indices ends a conversation.

    A jump of more than a couple of paragraphs is a scene change or a stretch
    of pure narration - either way the back-and-forth is over and alternation
    must not reach across it.
    """
    if para_a is None or para_b is None:
        return False
    return (para_b - para_a) > 3


def _vocative_addressee(text, roster):
    """If a quote is just an address to someone ("Bernard,"), return that name.

    A line whose whole content is a character's name is a greeting or a
    summons, spoken *to* them - so the speaker is the other party, and the
    addressee is a participant in the exchange. That is the seed a run of
    untagged dialogue needs when neither speaker is tagged early.
    """
    bare = text.strip().strip(".,!?—-–\"'“”‘’ ").strip()
    for name in roster:
        if bare.lower() == name.lower() or bare.lower() == name.split()[0].lower():
            return name
    return None


def _vocatives(text, names):
    found = []
    for name in names:
        first = name.split()[0]
        for n in {name, first}:
            pat = r"(?:^|[,;!?—–-]\s*|\b[Mm]y dear\s+)" + re.escape(n) + r"(?=\s*(?:[,.!?;:—–-]|$))"
            if re.search(pat, text):
                found.append(name)
                break
    return found


def _learn_aliases(marked, identity):
    tagged = {m["tag"] for m in marked if m["tag"]}
    pairs = {}
    for i, m in enumerate(marked[:-1]):
        nxt = marked[i + 1]
        if nxt["para"] is None or m["para"] is None or nxt["para"] - m["para"] > 2:
            continue
        if not nxt["tag"]:
            continue
        for v in m["vocs"]:
            if v != nxt["tag"]:
                pairs.setdefault(v, Counter())[nxt["tag"]] += 1
    for v, cnt in pairs.items():
        if v in tagged:
            continue
        (top, n), = cnt.most_common(1)
        if n >= 2 and len(cnt) == 1:
            identity.link(v, top)


def _para_context(doc):
    qspans = {}
    for q in doc.quotes:
        qs, qe = _int(q["quote_start"]), _int(q["quote_end"])
        if qs is None or qe is None:
            continue
        qspans.setdefault(doc.paragraph(qs), []).append((qs, qe))

    ptoks = {}
    for t in doc.tokens:
        p, tid = _int(t["paragraph_ID"]), _int(t["token_ID_within_document"])
        if p is not None and tid is not None:
            ptoks.setdefault(p, []).append(tid)

    def inq(p, tid):
        return any(s <= tid <= e for s, e in qspans.get(p, ()))

    ments = {}
    for e in doc.entities:
        if e["cat"] != "PER":
            continue
        s = _int(e["start_token"])
        if s is None:
            continue
        p = doc.paragraph(s)
        if inq(p, s):
            continue
        ments.setdefault(p, []).append((s, doc.name_of.get(e["COREF"])))
    for p in ments:
        ments[p].sort(key=lambda x: x[0])

    bare, lastw = {}, {}
    for p, toks in ptoks.items():
        bare[p] = not any(not inq(p, tid)
                          and doc.by_id[tid]["POS_tag"] != "PUNCT"
                          for tid in toks)
        words = [tid for tid in toks if doc.by_id[tid]["POS_tag"] != "PUNCT"]
        lastw[p] = max(words) if words else max(toks)

    return {"ments": ments, "bare": bare, "lastw": lastw}


def attribute(doc, identity=None):
    """Every quote -> a speaker name.

    Tags (explicit and anaphoric) are trusted first. The implicit remainder
    goes through the deterministic sieves of Muzny et al. 2017, in falling
    order of precision: single mention in the quote's own paragraph, vocative
    in the preceding quote, final mention before a paragraph-ending quote,
    strict two-apart conversational pattern, and only then run alternation.
    Vocatives also teach the identity matrix aliases: a name that is only ever
    addressed, whose replies are consistently tagged to one speaker, is that
    speaker.
    """
    identity = identity or IdentityMatrix()
    roster = [n for n in doc.name_of.values() if n]
    ctx = _para_context(doc)

    marked = []
    for q in doc.quotes:
        kind, coref = classify(doc, q)
        qs, qe = _int(q["quote_start"]), _int(q["quote_end"])
        para = doc.paragraph(qs)
        name = doc.display(coref) if coref else None
        text = q["quote"].replace("“", "").replace("”", "").strip()
        vocs = _vocatives(text, roster)
        bare_v = _vocative_addressee(text, roster)
        if bare_v and bare_v not in vocs:
            vocs.append(bare_v)
        pm = ctx["ments"].get(para, [])
        beat = pm[0][1] if len(pm) == 1 and pm[0][1] else None
        pfinal = None
        if qe is not None and qe >= ctx["lastw"].get(para, -1):
            before = [n for s, n in pm if s < (qs or 0) and n]
            pfinal = before[-1] if before else None
        marked.append({
            "text": text,
            "para": para,
            "kind": kind,
            "speaker": name if kind != "implicit" else None,
            "tag": name if kind != "implicit" else None,
            "vocs": vocs,
            "addressee": bare_v,
            "beat": beat,
            "pfinal": pfinal,
            "bare": ctx["bare"].get(para, False),
        })

    _learn_aliases(marked, identity)

    def canon(name, at):
        if not name:
            return None
        cid = identity.resolve(name, at=at)
        return identity.display(cid) if cid is not None else name

    for m in marked:
        at = m["para"]
        for k in ("tag", "speaker", "beat", "pfinal", "addressee"):
            m[k] = canon(m[k], at)
        m["vocs"] = [canon(v, at) for v in m["vocs"]]

    run, prev = [], None
    for m in marked:
        if run and _is_break(prev, m["para"]):
            _resolve_run(run)
            run = []
        run.append(m)
        prev = m["para"]
    if run:
        _resolve_run(run)
    return marked


def _not_addressee(m, name, fallback=None):
    if name and name in m["vocs"]:
        return fallback
    return name


def _resolve_run(run):
    """Fill one conversational run: sieves first, alternation last."""
    for m in run:
        if not m["speaker"] and m["beat"]:
            m["speaker"] = _not_addressee(m, m["beat"])
    for i, m in enumerate(run):
        if m["speaker"] or i == 0:
            continue
        prev = run[i - 1]
        if prev["vocs"] and m["para"] is not None and prev["para"] is not None \
                and 0 <= m["para"] - prev["para"] <= 2:
            m["speaker"] = _not_addressee(m, prev["vocs"][-1])
    for m in run:
        if not m["speaker"] and m["pfinal"]:
            m["speaker"] = _not_addressee(m, m["pfinal"])

    by_para = {}
    for m in run:
        if m["para"] is not None:
            by_para.setdefault(m["para"], []).append(m)
    changed = True
    while changed:
        changed = False
        for m in run:
            if m["speaker"] or not m["bare"] or m["para"] is None:
                continue
            for d in (-2, 2):
                nb = by_para.get(m["para"] + d)
                mid = by_para.get(m["para"] + d // 2)
                if nb and mid and nb[0]["speaker"]:
                    s = _not_addressee(m, nb[0]["speaker"])
                    if s:
                        m["speaker"] = s
                        changed = True
                        break

    freq = Counter(m["speaker"] for m in run if m["speaker"])
    for m in run:
        for v in m["vocs"]:
            freq[v] += 0
        if m["addressee"]:
            freq[m["addressee"]] += 0
    parts = [s for s, _ in freq.most_common()]

    if len(parts) >= 2:
        last = None
        for m in run:
            addressed = set(m["vocs"]) | ({m["addressee"]} if m["addressee"] else set())
            if m["speaker"]:
                if m["speaker"] in addressed:
                    m["speaker"] = None
                else:
                    last = m["speaker"]
                    continue
            cand = [s for s in parts if s not in addressed] or list(parts)
            pick = next((s for s in cand if s != last), cand[0])
            m["speaker"] = pick
            last = pick
        return

    a = parts[0] if parts else None
    b = None
    last = None
    for m in run:
        if m["speaker"]:
            last = m["speaker"]
            continue
        if m["vocs"]:
            m["speaker"] = a if b in m["vocs"] else b
        elif last is not None:
            m["speaker"] = a if last == b else b
        else:
            m["speaker"] = a
        last = m["speaker"]
