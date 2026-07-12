"""
Heteronym pronunciation picker.

English words like "wind" (wɪnd, moving air) and "wind" (waɪnd, to turn)
are spelled the same but spoken differently. Chatterbox reads letters, not
phonemes — there is no IPA input — so the only reliable way to force a
pronunciation is to respell the word the way it sounds ("wined the clock").

This module supplies a dictionary of such words. Each option is a
*forcing respelling*: text written that way can only be read one way.
Leaving the word as typed lets the model decide from context.

The dictionary is written to pronunciations.json on first use and read from
there afterwards, so you can edit/extend it: each entry is
  "word": [["IPA", "meaning — 'example'", "respelling"], ...]
"""

import json
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "pronunciations.json")

# word -> list of [ipa, meaning hint, forcing respelling]
DEFAULTS = {
    "wind":    [["wɪnd",      "moving air — 'the wind blew'",            "winned"],
                ["waɪnd",     "to turn or coil — 'wind the clock'",      "wined"]],
    "read":    [["riːd",      "present tense — 'I read every day'",      "reed"],
                ["rɛd",       "past tense — 'I read it yesterday'",      "red"]],
    "lead":    [["liːd",      "to guide / be ahead — 'lead the way'",    "leed"],
                ["lɛd",       "the metal — 'a lead pipe'",               "led"]],
    "tear":    [["tɪr",       "from crying — 'a tear rolled down'",      "teer"],
                ["tɛr",       "to rip — 'tear the paper'",               "tair"]],
    "live":    [["lɪv",       "to be alive — 'I live here'",             "liv"],
                ["laɪv",      "broadcast / in person — 'a live show'",   "lyve"]],
    "lives":   [["lɪvz",      "resides — 'he lives here'",               "livz"],
                ["laɪvz",     "plural of life — 'nine lives'",           "lyves"]],
    "bow":     [["baʊ",       "bend forward / ship's front — 'take a bow'", "bough"],
                ["boʊ",       "ribbon / weapon — 'bow and arrow'",       "beau"]],
    "wound":   [["wuːnd",     "an injury — 'a deep wound'",              "woond"],
                ["waʊnd",     "past of wind — 'wound the clock'",        "wowned"]],
    "bass":    [["beɪs",      "low sound / instrument — 'bass guitar'",  "base"],
                ["bæs",       "the fish",                                "bas"]],
    "close":   [["kloʊs",     "near — 'close by'",                       "cloce"],
                ["kloʊz",     "to shut — 'close the door'",              "cloze"]],
    "desert":  [["ˈdɛzərt",   "sandy place — 'the Sahara desert'",       "dezzurt"],
                ["dɪˈzɜːrt",  "to abandon — 'desert the army'",          "dessert"]],
    "dove":    [["dʌv",       "the bird",                                "duv"],
                ["doʊv",      "past of dive — 'she dove in'",            "dohve"]],
    "minute":  [["ˈmɪnɪt",    "60 seconds — 'wait a minute'",            "minnit"],
                ["maɪˈnuːt",  "tiny — 'a minute amount'",                "mynoot"]],
    "object":  [["ˈɒbdʒɪkt",  "a thing — 'a shiny object'",              "objekt"],
                ["əbˈdʒɛkt",  "to protest — 'I object!'",                "ubject"]],
    "present": [["ˈprɛzənt",  "gift / here now — 'a nice present'",      "prezzunt"],
                ["prɪˈzɛnt",  "to show — 'present the findings'",        "prizent"]],
    "produce": [["ˈproʊduːs", "vegetables — 'fresh produce'",            "prohduce"],
                ["prəˈduːs",  "to make — 'produce results'",             "pruhdoose"]],
    "record":  [["ˈrɛkərd",   "a stored item — 'a world record'",        "reckurd"],
                ["rɪˈkɔːrd",  "to capture — 'record a video'",           "ricord"]],
    "refuse":  [["rɪˈfjuːz",  "to say no — 'I refuse'",                  "refyooz"],
                ["ˈrɛfjuːs",  "garbage — 'refuse collection'",           "refyoos"]],
    "use":     [["juːz",      "verb — 'use the tool'",                   "yooz"],
                ["juːs",      "noun — 'no use crying'",                  "yoose"]],
    "house":   [["haʊs",      "noun — 'a big house'",                    "howse"],
                ["haʊz",      "verb — 'house the visitors'",             "howze"]],
    "polish":  [["ˈpɒlɪʃ",    "to shine — 'polish the car'",             "pollish"],
                ["ˈpoʊlɪʃ",   "from Poland — 'Polish food'",             "poelish"]],
    "resume":  [["rɪˈzuːm",   "to continue — 'resume playback'",         "rizoom"],
                ["ˈrɛzjʊmeɪ", "a CV — 'send your resume'",               "rezoomay"]],
}

_dict_cache = None
_dict_mtime = None


def get_dictionary():
    """The heteronym dictionary, from pronunciations.json (created from
    DEFAULTS on first use so the user can edit it)."""
    global _dict_cache, _dict_mtime
    if not os.path.isfile(JSON_PATH):
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULTS, f, ensure_ascii=False, indent=2)
    mtime = os.path.getmtime(JSON_PATH)
    if _dict_cache is None or mtime != _dict_mtime:
        try:
            with open(JSON_PATH, encoding="utf-8") as f:
                _dict_cache = {k.lower(): v for k, v in json.load(f).items()}
            _dict_mtime = mtime
        except Exception as e:
            print(f"[PRON] pronunciations.json unreadable ({e}); "
                  f"using built-in defaults.")
            _dict_cache = DEFAULTS
    return _dict_cache


def _word_re():
    words = sorted(get_dictionary().keys(), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b",
                      re.IGNORECASE)


def find_next(text, cursor=0):
    """First heteronym occurrence at/after character offset `cursor`.
    Returns (start, end, word_lower, options) or None."""
    if not text:
        return None
    m = _word_re().search(text, max(0, int(cursor)))
    if not m:
        return None
    word = m.group(1).lower()
    return (m.start(1), m.end(1), word, get_dictionary()[word])


def apply_choice(text, start, end, respelling):
    """Replace text[start:end] with the respelling, keeping capitalization.
    Returns (new_text, end_of_replacement)."""
    original = text[start:end]
    out = respelling
    if original[:1].isupper():
        out = out[:1].upper() + out[1:]
    if original.isupper() and len(original) > 1:
        out = out.upper()
    return text[:start] + out + text[end:], start + len(out)


_canon_re = None
_canon_map = None


def canonicalize(text):
    """Map every known respelling back to its normally-spelled word.
    Applied to BOTH the target text and the Whisper transcript during
    validation, so respelled words never cause a false mismatch."""
    global _canon_re, _canon_map
    if _canon_re is None:
        m = {}
        for word, options in get_dictionary().items():
            for _ipa, _hint, say in options:
                if say.lower() != word:
                    m[say.lower()] = word
        _canon_map = m
        _canon_re = re.compile(
            r"\b(" + "|".join(re.escape(s) for s in
                              sorted(m, key=len, reverse=True)) + r")\b",
            re.IGNORECASE)
    if not text:
        return text
    return _canon_re.sub(lambda mo: _canon_map[mo.group(1).lower()], text)
