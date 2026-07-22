import re
from collections import Counter

from audiobook import analyze

UNKNOWN = analyze.UNKNOWN
NARRATOR = analyze.NARRATOR

BLOCKLIST = {
    "hello", "hi", "hey", "hiya", "howdy", "greetings", "goodbye", "bye",
    "good morning", "good evening", "good night", "goodnight", "morning",
    "hm", "hmm", "hmmm", "hrm", "mm", "mmm", "mhm", "mmhmm", "um", "umm",
    "uh", "uhh", "er", "erm", "ah", "ahh", "aah", "oh", "ohh", "ooh", "oof",
    "whoa", "woah", "wow", "huh", "ugh", "argh", "aargh", "shh", "shhh",
    "psst", "phew", "ha", "hah", "heh", "hee", "yay", "ow", "ouch",
    "yeah", "yep", "yup", "nope", "nah", "aye", "okay", "ok", "alright",
    "yes", "no", "well", "what", "why", "wait", "stop", "sorry", "please",
    "thanks", "thank you",
    "sir", "maam", "madam", "madame", "mister", "miss", "missus", "sonny",
    "yessir", "nossir", "yessum", "shit", "damn", "dammit", "damnit", "hell",
    "crap", "ratshit", "gosh", "jeez", "geez", "christ", "jesus", "god",
    "oh god", "my god", "oh my god",
}

TITLES = {
    "mr", "mrs", "ms", "miss", "dr", "doctor", "sheriff", "deputy", "mayor",
    "governor", "senator", "congressman", "congresswoman", "judge", "officer",
    "captain", "lieutenant", "sergeant", "corporal", "major", "colonel",
    "general", "admiral", "professor", "reverend", "pastor", "lord", "lady",
    "sir", "dame", "master", "mistress", "king", "queen", "prince",
    "princess", "old", "young", "the",
}

MAIN_LINES = 100
SMALL_LINES = 30
FEW_LINES = 5
ANCHOR_SHARE = 0.60
ANCHOR_MIN = 3

PAIR_PROMPT = (
    "Automatic speaker attribution for one novel produced two cast entries "
    "that may be one character split in two - a nickname and the full name, "
    "a title variant, or a wrong surname the software attached to a first "
    "name. You are shown both entries with excerpts, and the full cast so "
    "you can recognise the book. Decide: one person, or two?\n"
    "How to read the excerpts: the speaker labels come from the same "
    "imperfect software, so one person's lines in a scene can flip between "
    "the two labels - alternating labels do NOT prove two people. Trust the "
    "quoted words and the narration. Narration that names one entry around "
    "lines labelled as the other ('Juliette took the scroll' right after a "
    "line labelled Jules) is the classic signature of one split character.\n"
    "Answer \"different\" when the excerpts show two entries as genuinely "
    "different people - talking to each other as two people, relatives "
    "sharing a surname, a child named after someone - or when the book "
    "deliberately runs the two names as separate identities until a late "
    "reveal (an alias, a secret identity: the reveal is handled elsewhere, "
    "keep them apart).\n"
    "Compare each entry's company: one character split in two keeps the "
    "same companions, scenes and relationships under both labels, while "
    "two different people move through different scenes with different "
    "companions - a different spouse or confidant means a different "
    "person, whatever the names suggest.\n"
    "A wrong merge ruins the audiobook more than a missed one: answer "
    "\"same\" only when the excerpts or your knowledge of this novel "
    "identify the two as one person outright.\n"
    "Weigh both hypotheses against every excerpt before answering: if these "
    "are one person, does every excerpt of both entries read consistently? "
    "If they are two people, who is each?\n"
    "Answer with ONLY JSON: {\"why\": \"one or two sentences weighing the "
    "evidence\", \"verdict\": \"same\" or \"different\"}."
)

REASSIGN_PROMPT = (
    "Automatic speaker attribution for one novel produced a cast entry that "
    "may not be a real character: a name the software misread from a few "
    "lines. You are shown every line of the entry with the surrounding "
    "text, and the book's cast. For each line, decide who really speaks "
    "it.\n"
    "The true speaker is in the scene: named by the narration around the "
    "line, or the other party of the conversation. A name inside a quoted "
    "line is almost always the person being ADDRESSED, not the speaker. "
    "Never pick a character who is not present in the excerpt's scene.\n"
    "For each line answer with a name from BOOK CAST, or \"Unknown\" when "
    "it cannot be worked out, or \"keep\" when the entry is a real minor "
    "character and its own label is right for that line.\n"
    "Answer with ONLY JSON: {\"lines\": {\"<line id>\": \"Name\"}}. "
    "Answer for every line listed."
)


def _norm(name):
    return re.sub(r"[^a-z0-9 ]", "", (name or "").lower()).strip()


def _tokens(name):
    return _norm(name).split()


def _bare(name):
    toks = _tokens(name)
    while toks and toks[0] in TITLES:
        toks = toks[1:]
    return toks


def _parts(name):
    toks = _tokens(name)
    bare = _bare(name)
    titled = len(toks) > len(bare)
    if not bare or len(bare) > 3:
        return None
    single = bare[0] if len(bare) == 1 and not titled else None
    last = bare[-1] if (len(bare) > 1 or titled) else None
    return {"bare": " ".join(bare), "single": single, "last": last,
            "titled": titled}


def _nick(short, long):
    if not short or not long or short == long or len(short) >= len(long):
        return False
    for suf in ("", "s", "es", "e", "ie", "ey", "y"):
        if suf and not short.endswith(suf):
            continue
        base = short[: len(short) - len(suf)] if suf else short
        if len(base) >= 3 and long.startswith(base):
            return True
    return False


def _reveal_pair(identity, a, b):
    ia, ib = identity.id_for(a), identity.id_for(b)
    if ia is None or ib is None:
        return False
    for name, own, other in ((a, ia, ib), (b, ib, ia)):
        r = identity.resolve(name, at=float("inf"))
        if r is not None and r != own and r == other:
            return True
    return False


def _bigrams(units):
    counts = Counter()
    disp = {}
    pat = re.compile(r"\b([A-Z][a-zA-Z'’-]+)[ ]([A-Z][a-zA-Z'’-]+)\b")
    for u in units:
        for m in pat.finditer(u.get("text") or ""):
            f, l = _norm(m.group(1)), _norm(m.group(2))
            if not f or not l or f in TITLES or l in TITLES:
                continue
            counts[(f, l)] += 1
            disp.setdefault((f, l), m.group(0))
    return counts, disp


def _mention_token(name):
    words = [w for w in re.findall(r"[A-Za-z'’-]+", name or "")
             if _norm(w) not in TITLES and len(w) >= 3]
    return words[-1] if words else None


def _mention_pat(name):
    tok = _mention_token(name)
    return re.compile(r"\b%s\b" % re.escape(tok)) if tok else None


def _near_narration(by, uid, pat):
    for j in (uid - 2, uid - 1, uid + 1, uid + 2):
        u = by.get(j)
        if u and u.get("kind") == "narration" and pat.search(u["text"]):
            return True
    return False


def _support(by, alias_ids, full):
    pat = _mention_pat(full)
    if not pat:
        return 0
    return sum(1 for i in alias_ids if _near_narration(by, i, pat))


def _pairs(units, by, cast, ids):
    names = [n for n in cast if _parts(n)]
    parts = {n: _parts(n) for n in names}
    grams, gram_disp = _bigrams(units)

    evidence = {}

    def flag(a, b, why, kind):
        evidence.setdefault(frozenset((a, b)), []).append((kind, why))

    for i, a in enumerate(names):
        pa = parts[a]
        for b in names[i + 1:]:
            pb = parts[b]
            if pa["bare"] == pb["bare"]:
                flag(a, b, "the same name apart from a title", "title")
                continue
            if len(pa["bare"].split()) == 1 and len(pb["bare"].split()) == 1 \
                    and (_nick(pa["bare"], pb["bare"])
                         or _nick(pb["bare"], pa["bare"])):
                short, long = (a, b) if len(pa["bare"]) < len(pb["bare"]) \
                    else (b, a)
                flag(a, b, "%s could be a short form of %s" % (short, long),
                     "nick")
                continue
            if pa["last"] and pb["last"] and pa["last"] == pb["last"]:
                flag(a, b, "share a surname", "surname")
                continue
            if (pa["last"] and pb["single"] == pa["last"]) \
                    or (pb["last"] and pa["single"] == pb["last"]):
                flag(a, b, "share a surname", "surname")
                continue
            for x, y in ((a, b), (b, a)):
                key = (parts[x]["single"], parts[y]["last"])
                if key[0] and key[1] and grams.get(key, 0) >= 2:
                    flag(x, y, "the text says '%s' %d times"
                         % (gram_disp[key], grams[key]), "bigram")
                    break

    for alias in names:
        n = cast[alias]
        if n > SMALL_LINES:
            continue
        aids = ids.get(alias, [])
        for full in names:
            if full == alias or cast[full] <= n:
                continue
            if parts[alias]["bare"][:1] != parts[full]["bare"][:1]:
                continue
            s = _support(by, aids, full)
            if s >= 2 and s >= 0.15 * n:
                flag(alias, full,
                     "the narration around %d of %s's %d lines names %s"
                     % (s, alias, n, full), "narration")

    return evidence, parts, grams, gram_disp


def _excerpt(by, uid, radius, mark_id=False):
    lines = []
    for i in range(uid - radius, uid + radius + 1):
        u = by.get(i)
        if not u:
            continue
        text = (u.get("text") or "")[:100]
        mark = ">" if i == uid else " "
        head = "[line %d] " % i if (mark_id and i == uid) else ""
        if u.get("kind") == "dialogue":
            lines.append('%s %s%s: "%s"' % (mark, head,
                                            u.get("speaker") or "?", text))
        else:
            lines.append("%s %s%s" % (mark, head, text))
    return "\n".join(lines)


def _sample_ids(uids, by, other, k, spread=False, avoid=None):
    if avoid:
        far = [i for i in uids if all(abs(i - j) > 20 for j in avoid)]
        uids = far or uids
    pat = _mention_pat(other)
    strong = [i for i in uids if pat and _near_narration(by, i, pat)]
    quoted = [i for i in uids if pat and i not in strong
              and pat.search(by[i].get("text") or "")]
    picks = (strong + quoted)[: max(1, k - 3) if spread else k]
    for i in (uids[0], uids[len(uids) // 3], uids[2 * len(uids) // 3],
              uids[-1]):
        if len(picks) >= k:
            break
        if i not in picks:
            picks.append(i)
    return sorted(set(picks))


def _profile(by, uids, name, radius=5):
    c = Counter()
    for i in uids:
        for j in range(i - radius, i + radius + 1):
            if j == i:
                continue
            u = by.get(j)
            if u and u.get("kind") == "dialogue":
                s = u.get("speaker")
                if s and s not in (name, UNKNOWN, NARRATOR):
                    c[s] += 1
    return c


def _partners(by, uids, name):
    c = _profile(by, uids, name)
    return ", ".join("%s (%d)" % (n, k) for n, k in c.most_common(5)) \
        or "nobody"


def _anchor_veto(by, ids, small, big):
    pb = _profile(by, ids[big], big)
    pb.pop(small, None)
    total = sum(pb.values())
    if not total:
        return None
    who, n = pb.most_common(1)[0]
    if n / total < ANCHOR_SHARE:
        return None
    ps = _profile(by, ids[small], small)
    ps.pop(big, None)
    if sum(ps.values()) < ANCHOR_MIN or ps.get(who):
        return None
    return who


def _cast_line(cast, top=80):
    ranked = sorted(cast, key=lambda n: -cast[n])[:top]
    return ", ".join("%s: %d" % (n, cast[n]) for n in ranked)


def _pair_user(a, b, cast, ids, by, whys):
    out = ["BOOK CAST (name: number of lines):", _cast_line(cast), "",
           "ENTRY A: %s (%d lines); shares scenes with %s"
           % (a, cast[a], _partners(by, ids[a], a)),
           "ENTRY B: %s (%d lines); shares scenes with %s"
           % (b, cast[b], _partners(by, ids[b], b)),
           "WHY FLAGGED: %s" % "; ".join(w for _, w in whys), ""]
    for label, name, other, k, spread in (("A", a, b, 4, True),
                                          ("B", b, a, 4, False)):
        out.append("EXCERPTS OF %s (%s):" % (label, name))
        avoid = ids[other] if spread else None
        for uid in _sample_ids(ids[name], by, other, k, spread, avoid):
            out.append("--- around line %d:" % uid)
            out.append(_excerpt(by, uid, 3))
            out.append("")
    out.append("Same person, or different?")
    return "\n".join(out)


def _in_scene(by, uid, target, radius=25):
    pat = _mention_pat(target)
    for i in range(uid - radius, uid + radius + 1):
        u = by.get(i)
        if not u:
            continue
        if u.get("speaker") == target:
            return True
        if pat and pat.search(u.get("text") or ""):
            return True
    return False


def _reassign_user(name, cast, ids, by):
    out = ["BOOK CAST (name: number of lines):", _cast_line(cast), "",
           "SUSPECT ENTRY: %s (%d lines)" % (name, cast[name]), ""]
    for uid in ids[name]:
        out.append("--- line %d:" % uid)
        out.append(_excerpt(by, uid, 4, mark_id=True))
        out.append("")
    out.append("Who really speaks each line?")
    return "\n".join(out)


def _ask(worker, system, user, log):
    for effort in ("high", "high", "medium", "low"):
        try:
            return analyze._parse_json(analyze._chat(
                {**worker, "reasoning_effort": effort},
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                max_tokens=8000))
        except (analyze.AnalyzeError, ValueError) as e:
            log("cast dedup: no usable answer (%s)" % e)
    return None


def _canonical(component, cast, parts, grams, gram_disp):
    biggest = max(component, key=lambda n: cast[n])
    if cast[biggest] >= MAIN_LINES:
        return biggest
    best = None
    for f in component:
        if not parts[f]["single"]:
            continue
        for (bf, bl), g in grams.items():
            if bf == parts[f]["single"] and g >= 5 \
                    and (best is None or g > best[0]):
                best = (g, gram_disp[(bf, bl)])
    if best:
        return best[1]
    return biggest


def _fallback_merges(by, cast, ids, parts, identity):
    merges = {}
    for alias in cast:
        pa = parts.get(alias)
        if not pa or not pa["single"] or cast[alias] > SMALL_LINES:
            continue
        for full in cast:
            pb = parts.get(full)
            if not pb or not pb["single"] or cast[full] <= cast[alias]:
                continue
            if not _nick(pa["single"], pb["single"]):
                continue
            if _reveal_pair(identity, alias, full):
                continue
            if _anchor_veto(by, ids, alias, full):
                continue
            if _support(by, ids.get(alias, []), full) >= 2:
                merges[alias] = full
                break
    return merges


def dedup(units, identity, cfg=None, log=None):
    log = log or (lambda *_: None)
    cfg = {**analyze.DEFAULTS, **(cfg or {})}
    by = {u["id"]: u for u in units}

    counts = Counter(u.get("speaker") for u in units
                     if u.get("kind") == "dialogue" and u.get("speaker"))
    blocked = {n for n in counts
               if n not in (UNKNOWN, NARRATOR) and _norm(n) in BLOCKLIST}
    for u in units:
        if u.get("kind") == "dialogue" and u.get("speaker") in blocked:
            u["speaker"] = UNKNOWN
    for n in sorted(blocked):
        log("cast dedup: %r is not a name; %d lines -> %s"
            % (n, counts[n], UNKNOWN))

    cast = {n: c for n, c in counts.items()
            if n not in blocked and n not in (UNKNOWN, NARRATOR)}
    ids = {}
    for u in units:
        if u.get("kind") == "dialogue" and u.get("speaker") in cast:
            ids.setdefault(u["speaker"], []).append(u["id"])

    evidence, parts, grams, gram_disp = _pairs(units, by, cast, ids)

    parent = {n: n for n in cast}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    try:
        worker = analyze.worker_cfgs(cfg, log)[0]
    except analyze.AnalyzeError as e:
        log("cast dedup: no LLM to adjudicate with (%s); applying only "
            "evidence-backed nickname links" % e)
        worker = None

    if worker:
        same, diff = [], []
        for pair, whys in sorted(evidence.items(),
                                 key=lambda kv: -max(cast[n] for n in kv[0])):
            a, b = sorted(pair, key=lambda x: -cast[x])
            kinds = {k for k, _ in whys}
            if cast[a] >= MAIN_LINES and cast[b] >= MAIN_LINES:
                continue
            if cast[b] <= FEW_LINES and parts[b]["last"] \
                    and "bigram" not in kinds:
                continue
            if _reveal_pair(identity, a, b):
                log("cast dedup: %s vs %s not asked (reveal pair)" % (a, b))
                continue
            anchor = _anchor_veto(by, ids, b, a)
            if anchor:
                diff.append((a, b))
                log("cast dedup: %s vs %s: different (%s is almost always "
                    "with %s, and %s never is)" % (a, b, a, anchor, b))
                continue
            reply = _ask(worker, PAIR_PROMPT,
                         _pair_user(a, b, cast, ids, by, whys), log)
            verdict = str((reply or {}).get("verdict", "")).strip().lower()
            log("cast dedup: %s vs %s: %s" % (a, b, verdict or "no answer"))
            (same if verdict == "same" else diff).append((a, b))
        for a, b in same:
            snapshot = dict(parent)
            parent[find(a)] = find(b)
            joined = {}
            for n in cast:
                if cast[n] >= SMALL_LINES:
                    joined.setdefault(find(n), []).append(n)
            crowded = any(len(g) > 1 for g in joined.values())
            if crowded or any(find(x) == find(y) for x, y in diff):
                parent = snapshot
                log("cast dedup: not joining %s and %s (%s)"
                    % (a, b, "two established characters"
                       if crowded else "conflicts with a 'different' verdict"))
    else:
        for alias, full in _fallback_merges(by, cast, ids, parts,
                                            identity).items():
            parent[find(alias)] = find(full)
            log("cast dedup: %s = %s" % (alias, full))

    components = {}
    for n in cast:
        components.setdefault(find(n), []).append(n)

    merged = set()
    for group in components.values():
        if len(group) < 2:
            continue
        canon = _canonical(group, cast, parts, grams, gram_disp)
        for n in group:
            if n != canon:
                identity.link(n, canon)
                merged.add(n)
                log("cast dedup: %s -> %s (%d lines)" % (n, canon, cast[n]))

    if merged:
        for u in units:
            if u.get("kind") != "dialogue":
                continue
            s = u.get("speaker")
            if not s or s in (UNKNOWN, NARRATOR):
                continue
            cid = identity.resolve(s, at=u.get("para"))
            disp = identity.display(cid) if cid is not None else None
            if disp:
                u["speaker"] = disp

    if not worker:
        return

    flagged = {n for pair in evidence for n in pair}
    suspects = [n for n in cast
                if n in flagged and n not in merged
                and cast[n] <= FEW_LINES and parts[n]["last"]]
    for name in suspects:
        reply = _ask(worker, REASSIGN_PROMPT,
                     _reassign_user(name, cast, ids, by), log)
        for k, v in ((reply or {}).get("lines") or {}).items():
            m = re.search(r"\d+", str(k))
            v = str(v).strip()
            if not m:
                continue
            uid = int(m.group())
            if uid not in ids[name]:
                continue
            if v.lower() == "keep" or _norm(v) == _norm(name):
                continue
            if v not in cast and v != UNKNOWN:
                continue
            if v != UNKNOWN and not _in_scene(by, uid, v):
                log("cast dedup: line %d not moved to %s (not in the scene)"
                    % (uid, v))
                continue
            u = by[uid]
            cid = identity.resolve(v, at=u.get("para"))
            disp = (identity.display(cid) if cid is not None else None) or v
            log("cast dedup: line %d: %s -> %s" % (uid, name, disp))
            u["speaker"] = disp
