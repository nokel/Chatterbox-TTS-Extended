"""Make BookNLP run on Windows.

BookNLP works out which pretrained BERT to fetch from the *filename* of the
model it just downloaded:

    base_model = re.sub("google_bert", "google/bert", model_file.split("/")[-1])
    base_model = re.sub(".model", "", base_model)

Two bugs, both fatal here and neither reachable on the author's machine:

1. `split("/")` splits on forward slash only. Windows paths are backslashed,
   so the "filename" comes back as the whole path.
2. `.model` is a regex, and `.` matches any character - so it also eats the
   `_model` inside the directory name `booknlp_models`.

Together they turn
    C:\\Users\\me\\booknlp_models\\entities_google_bert_uncased_L-6_...-v1.0.model
into
    C:\\Users\\me\\booknlps\\entities_google/bert_uncased_L-6_...
which transformers then tries to fetch from the Hub, and BookNLP dies before
reading a word.

Both regexes are correct if they are handed a forward-slashed path: the
basename then comes out clean and the only "model" left is the extension. So
rather than patching site-packages - which any reinstall would undo - the
three constructors are wrapped to normalise the path argument on the way in.
Upstream is untouched, and this whole module is a no-op on Linux and Mac.
"""

import os

_patched = False

# BookNLP's checkpoints were saved under an older transformers, where BERT kept
# a "position_ids" buffer in its state dict. transformers >= 4.31 dropped that
# buffer, so a strict load now rejects the checkpoint over one stale key that
# the current model computes on the fly anyway. These are the keys safe to
# ignore either way round (present in the file but not the model, or vice
# versa) - all of them derived, none of them learned.
_IGNORABLE = ("position_ids", "token_type_ids")


def _slashed(p):
    """A path BookNLP's own regexes can read. Still opens fine: Windows
    accepts forward slashes everywhere."""
    if isinstance(p, str) and "\\" in p:
        return p.replace("\\", "/")
    return p


def _tolerant_load_state_dict(cls):
    """Load a checkpoint that differs only in derived buffers.

    Wraps this model class's load_state_dict so a mismatch confined to the
    known derived buffers (position_ids and friends) is tolerated, while any
    other missing or unexpected key - a real architecture mismatch - is still
    raised. It is not a blanket strict=False: that would hide a genuinely
    wrong checkpoint and let the model run on random weights.
    """
    orig = cls.load_state_dict
    if getattr(orig, "_booknlp_win_patched", False):
        return

    def load_state_dict(self, state_dict, strict=True, **kw):
        try:
            return orig(self, state_dict, strict=strict, **kw)
        except RuntimeError as e:
            res = orig(self, state_dict, strict=False, **kw)
            leftover = [k for k in (list(res.missing_keys) + list(res.unexpected_keys))
                        if not any(tok in k for tok in _IGNORABLE)]
            if leftover:
                raise RuntimeError(
                    f"{cls.__name__}: checkpoint mismatch beyond derived "
                    f"buffers: {leftover}") from e
            return res

    load_state_dict._booknlp_win_patched = True
    cls.load_state_dict = load_state_dict


def _wrap_first_arg(cls):
    """Normalise the model path, whether it's passed by position or name."""
    orig = cls.__init__
    if getattr(orig, "_booknlp_win_patched", False):
        return

    def __init__(self, *args, **kwargs):
        args = list(args)
        if args:
            args[0] = _slashed(args[0])
        for k in ("model_file", "modelFile"):
            if k in kwargs:
                kwargs[k] = _slashed(kwargs[k])
        return orig(self, *args, **kwargs)

    __init__._booknlp_win_patched = True
    cls.__init__ = __init__


def apply():
    """Patch BookNLP in-process. Safe to call more than once."""
    global _patched
    if _patched or os.name != "nt":
        _patched = True
        return
    from booknlp.english.bert_qa import QuotationAttribution, BERTSpeakerID
    from booknlp.english.entity_tagger import LitBankEntityTagger
    from booknlp.english.litbank_coref import LitBankCoref
    from booknlp.english.bert_coref_quote_pronouns import BERTCorefTagger
    from booknlp.english.tagger import Tagger
    # the outer classes take the model path (the Windows bug)
    for cls in (LitBankEntityTagger, QuotationAttribution, LitBankCoref):
        _wrap_first_arg(cls)
    # the inner nn.Modules take the checkpoint (the version-skew bug)
    for cls in (Tagger, BERTSpeakerID, BERTCorefTagger):
        _tolerant_load_state_dict(cls)
    _utf8_input_read()
    _patched = True


def _utf8_input_read():
    """Read every text file BookNLP opens as UTF-8.

    BookNLP opens files with a bare open() - no encoding - so on Windows they
    decode as cp1252 and die on the first smart quote, which in a novel is the
    first line of dialogue. It isn't one call: the input reader, the gender
    hyperparameter reader, and others each have their own. Every *output* it
    writes already names utf-8; only the reads were missed. So rather than
    hunt each module's open(), give the whole package one that defaults to
    utf-8, and leave the real builtin alone everywhere outside BookNLP.
    """
    import builtins
    import importlib
    import pkgutil

    import booknlp
    real_open = builtins.open

    def open_utf8(file, mode="r", *a, **kw):
        if "b" not in mode and kw.get("encoding") is None and "encoding" not in kw:
            kw["encoding"] = "utf-8"
        return real_open(file, mode, *a, **kw)

    for mod in pkgutil.walk_packages(booknlp.__path__, booknlp.__name__ + "."):
        try:
            m = importlib.import_module(mod.name)
        except Exception:
            continue
        # only rebind modules that actually reference the builtin open
        if getattr(m, "open", None) is real_open or "open" not in vars(m):
            m.open = open_utf8
