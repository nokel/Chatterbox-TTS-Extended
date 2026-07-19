"""Turn BookNLP's raw attribution into speaker-per-quote, the way a reader does.

BookNLP is strong at two things and blind to a third. It clusters the cast
(every "she"/"him"/"the mayor" that means Jahns lands on one id), and it nails a
line that carries its own tag ("said Bernard"). But its speaker model binds each
quote to the nearest resolved mention, which is not how an unmarked exchange
works. In

    "What games are you playing at, Marie?"
    Jahns felt her temperature rise.
    "Playing at?"

the nearest mention to the first line is "Jahns", so BookNLP gives her the line
Bernard speaks. A reader doesn't do that: they know two people are talking and
the turn just passed, so this line is the *other* one's.

That turn-taking is the rule this module adds. BookNLP stays the substrate - its
clusters and its confidently-tagged lines are trusted - and alternation only
fills the gaps between them:

  * A quote whose paragraph carries a real dialogue tag naming a character keeps
    that speaker. A name *inside* the quotation is the person addressed, not the
    speaker ("Marie" in the line above is who Bernard is talking to), so tags are
    only read from the narration around the quote, never its contents.
  * A quote with no such tag is assigned by alternation: in an established
    two-person exchange it goes to whoever didn't just speak.
  * The exchange resets at a structural break - a new chapter or a scene divider
    - and whenever a tag names a third person, because three in a room is no
    longer a simple back-and-forth.

The identity matrix ([[IdentityMatrix]]) sits underneath all of this: names are
resolved to a stable character id, so "the mayor" and "Jahns" alternate as one
person, and a later reveal can retie an early alias to its true identity.
"""

import re

# Verbs that mark a line of speech. A tag is one of these next to the quote,
# in the narration - not a word that merely appears inside it.
_SAY = (r"said|says|asked|replied|answered|shouted|whispered|muttered|called"
        r"|added|cried|snapped|murmured|breathed|hissed|barked|growled|roared"
        r"|continued|offered|admitted|agreed|insisted|repeated|began|told"
        r"|demanded|observed|noted|remarked|responded|countered|urged|warned"
        r"|pleaded|sighed|laughed|spat|drawled|put in|went on|wondered")

_NAME = r"[A-Z][a-zA-Z'’.-]+(?:\s+[A-Z][a-zA-Z'’.-]+)?"
# "said Bernard" / "Bernard said" / "Bernard, still angry, said"
_TAG_AFTER = re.compile(r"^[\s,]*(?:%s)\s+(%s)\b" % (_SAY, _NAME))
_TAG_BEFORE_END = re.compile(r"\b(%s)\s+(?:%s)\s*[.,!?]?\s*$" % (_NAME, _SAY))
_TAG_BEFORE_START = re.compile(r"^[\s,]*(%s)\s+(?:%s)\b" % (_NAME, _SAY))


class IdentityMatrix:
    """Names to a stable character id, with reveals over the book's timeline.

    A character is one id however many names they wear. Linking is
    monotonic - two ids that turn out to be one are merged, and every name
    that pointed at either points at the survivor - so "the mayor", "Mayor
    Jahns" and "Jahns" all resolve alike no matter which was seen first.

    Reveals are the harder case a novel actually needs. A character can speak
    for chapters under a mask - a cover name, "the stranger", a false
    identity - before the book says who they were all along. `reveal(alias,
    real, at)` records that: lines attributed to the alias *after* the reveal
    point fold into the real character, while the ones before it stay under
    the mask, so the twist reads as written and isn't spoiled early by a voice
    that already knows.
    """

    def __init__(self):
        self._id_of = {}          # canonical name key -> id
        self._name = {}           # id -> display name
        self._next = 1
        self._reveals = []        # (alias_id, real_id, at_marker)

    @staticmethod
    def _key(name):
        return re.sub(r"[^a-z0-9 ]", "", (name or "").lower()).strip()

    def id_for(self, name):
        k = self._key(name)
        if not k:
            return None
        if k not in self._id_of:
            self._id_of[k] = self._next
            self._name[self._next] = name
            self._next += 1
        return self._id_of[k]

    def link(self, a, b):
        """Declare two names the same character; returns the surviving id.

        `b` is treated as the more canonical name - the one a listener would
        recognise - so linking an alias to it keeps b's display, not whichever
        id happened to be created first. ("Marie" links to "Jahns" and the
        voice is announced as Jahns, not the given name Bernard alone uses.)
        """
        ida, idb = self.id_for(a), self.id_for(b)
        if ida is None or idb is None or ida == idb:
            return ida if idb is None else idb
        keep, drop, keep_name = idb, ida, self._name.get(idb)
        for k, v in list(self._id_of.items()):
            if v == drop:
                self._id_of[k] = keep
        self._name[keep] = keep_name
        self._name.pop(drop, None)
        return keep

    def reveal(self, alias, real, at):
        """After marker `at`, lines under `alias` belong to `real`."""
        self._reveals.append((self.id_for(alias), self.id_for(real), at))

    def resolve(self, name, at=None):
        """The character id for a name, applying any reveal reached by `at`."""
        cid = self.id_for(name)
        if cid is None:
            return None
        if at is not None:
            for alias_id, real_id, marker in self._reveals:
                if cid == alias_id and at >= marker:
                    cid = real_id
        return cid

    def display(self, cid):
        return self._name.get(cid, None)


def _tag_speaker(before, after, roster):
    """A speaker named by the narration touching a quote, or None.

    `before` is the narration ending at the quote, `after` the narration that
    opens right after it. Only names the roster knows are accepted, so a stray
    capitalised word isn't mistaken for a speaker.
    """
    def known(m):
        if not m:
            return None
        name = m.group(1).strip(" ,.'’-")
        return name if any(name.split()[0] == r.split()[0] for r in roster) else None

    # "..." said Bernard   |   "..." Bernard said
    for pat in (_TAG_AFTER, _TAG_BEFORE_START):
        who = known(pat.search(after or ""))
        if who:
            return who
    # Bernard said, "..."
    who = known(_TAG_BEFORE_END.search(before or ""))
    if who:
        return who
    return None


def refine(quotes, roster, identity=None, is_break=None):
    """Assign a speaker to every quote, tags first then alternation.

    quotes    ordered list of dicts with:
                para   int    paragraph index (a real one - geometry, not a
                              layout block; see pdfbook._paragraphs)
                text   str    the quoted words, no surrounding narration
                before str    narration ending at the quote (same paragraph)
                after  str    narration right after the quote
                hint   str    BookNLP's guess, used only as a last resort
    roster    known character names, for validating tags
    identity  an IdentityMatrix, or None to make a fresh one
    is_break  para -> bool, True at a chapter/scene boundary that ends an
              exchange. Defaults to never.

    Returns the same list with a "speaker" on each quote.
    """
    identity = identity or IdentityMatrix()
    is_break = is_break or (lambda para: False)
    for r in roster:
        identity.id_for(r)

    # pass 1: confident, tag-based speakers.
    # A caller working from BookNLP already has a coref-resolved tag - it knows
    # "she asked" means Jahns, which a regex never will - and passes it as
    # "tag". Trust that when present; fall back to reading the narration only
    # for the plain-PDF path that has no coref to lean on.
    for q in quotes:
        who = q.get("tag")
        if who is None:
            who = _tag_speaker(q.get("before", ""), q.get("after", ""), roster)
        q["_tag"] = who
        q["speaker"] = who

    # pass 2: alternation across the gaps
    # `pair` is the two people currently trading turns; `last` is who spoke
    # last. A break clears both - a new scene starts a new conversation.
    pair = []
    last = None
    prev_para = None
    for q in quotes:
        if prev_para is not None:
            for p in range(prev_para + 1, q["para"] + 1):
                if is_break(p):
                    pair, last = [], None
                    break
        prev_para = q["para"]

        tag = q["_tag"]
        if tag:
            # a third voice ends the two-way exchange and reseeds it
            if tag not in pair:
                pair = ([last, tag] if last and last != tag else [tag])
            last = tag
            _register(pair, tag)
            continue

        # no tag: alternate to whoever isn't `last`, if we have a pair
        other = _other(pair, last)
        if other:
            q["speaker"] = other
            last = other
        elif last:
            # only one voice known so far: a reply implies a second, unnamed
            # for now; keep BookNLP's hint if it offers a different name
            hint = q.get("hint")
            q["speaker"] = hint if hint and hint != last else last
            last = q["speaker"]
        else:
            q["speaker"] = q.get("hint")
            last = q["speaker"]
        _register(pair, q["speaker"])

    # fold names to canonical display, so aliases read as one voice
    for q in quotes:
        cid = identity.resolve(q.get("speaker"), at=q["para"]) if q.get("speaker") else None
        q["char_id"] = cid
        if cid is not None:
            q["speaker"] = identity.display(cid) or q["speaker"]
    return quotes


def _register(pair, who):
    if who and who not in pair:
        pair.append(who)
        if len(pair) > 2:
            pair.pop(0)


def _other(pair, last):
    """The member of the trading pair who isn't `last`."""
    if len(pair) == 2 and last in pair:
        return pair[0] if pair[1] == last else pair[1]
    return None
