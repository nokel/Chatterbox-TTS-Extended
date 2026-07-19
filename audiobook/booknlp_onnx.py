"""Run BookNLP's speaker-attribution model through ONNX Runtime.

Why this model and not the others: of BookNLP's three BERT models, speaker
attribution is the one whose forward pass is a clean tensor pipeline - encoder,
two matmuls against mention-position matrices, two linear layers - with no
Python control flow, no CRF, no layered decode. That makes it the one that
exports to ONNX whole, and it is also the one doing the job we care about most:
deciding who speaks each quote.

The point is a lighter, faster inference path. The BERT encoder is where nearly
all the compute goes, and ONNX Runtime's WebGPU provider runs it on the AMD GPU
without dragging the training-shaped torch graph along at inference time. The
torch model stays the source of truth for the export and the numeric check; it
is not needed to *run* once the .onnx file exists.

The shapes, read off speaker_attribution.get_batches:
    toks   (batch, seq)          wordpiece ids
    mask   (batch, seq)          attention mask
    cands  (batch, 10, seq)      up to 10 candidate speakers, each row a
                                 uniform average over that mention's wordpieces
    quote  (batch, 10, seq)      the quote's own span, repeated per candidate
  -> logits (batch, 10, 1); argmax over the 10 candidates picks the speaker.
"""

import os

import numpy as np


class _SpeakerNet:
    """Tensor-in, tensor-out wrapper so the dict-based forward can be traced."""

    def __init__(self, model):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.bert = m.bert
                self.fc = m.fc
                self.fc2 = m.fc2
                self.tanh = m.tanh

            def forward(self, toks, mask, cands, quote):
                import torch
                _, _, hidden = self.bert(
                    toks, token_type_ids=None, attention_mask=mask,
                    output_hidden_states=True, return_dict=False)
                out = hidden[-1]
                combined = torch.cat(
                    (torch.matmul(cands, out), torch.matmul(quote, out)), axis=2)
                return self.fc2(self.tanh(self.fc(combined)))

        self.net = Net(model).eval()


def export(model, path, device="cpu"):
    """Trace BERTSpeakerID to an ONNX file at `path`. Returns the path.

    A tiny dummy batch is enough to trace: ONNX records the operations, and the
    dynamic axes let the real batch and sequence lengths differ at run time.
    """
    import torch

    net = _SpeakerNet(model).net.to(device)
    seq, cand = 24, 10
    toks = torch.ones(1, seq, dtype=torch.long, device=device)
    mask = torch.ones(1, seq, dtype=torch.long, device=device)
    cands = torch.zeros(1, cand, seq, device=device)
    quote = torch.zeros(1, cand, seq, device=device)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    kw = dict(
        input_names=["toks", "mask", "cands", "quote"],
        output_names=["logits"],
        dynamic_axes={"toks": {0: "b", 1: "s"}, "mask": {0: "b", 1: "s"},
                      "cands": {0: "b", 2: "s"}, "quote": {0: "b", 2: "s"},
                      "logits": {0: "b"}},
        opset_version=17, do_constant_folding=True)
    # torch >= 2.5 defaults to the dynamo exporter, which needs onnxscript.
    # The TorchScript exporter handles this graph fine and needs nothing extra,
    # so ask for it explicitly where the argument exists.
    try:
        torch.onnx.export(net, (toks, mask, cands, quote), path,
                          dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(net, (toks, mask, cands, quote), path, **kw)
    return path


class OnnxSpeaker:
    """Drop-in for BERTSpeakerID.forward, backed by ONNX Runtime.

    Prefers the GPU provider (WebGPU on this AMD box), falls back to CPU. The
    forward takes the same two dicts the torch model does, so the call site in
    QuotationAttribution doesn't change.
    """

    def __init__(self, onnx_path, providers=None):
        import onnxruntime as ort
        if providers is None:
            avail = ort.get_available_providers()
            providers = [p for p in ("WebGpuExecutionProvider",
                                     "DmlExecutionProvider",
                                     "CUDAExecutionProvider",
                                     "CPUExecutionProvider") if p in avail]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.provider = self.sess.get_providers()[0]

    def __call__(self, batch_x, batch_m):
        import torch
        toks = batch_x["toks"].cpu().numpy().astype(np.int64)
        mask = batch_x["mask"].cpu().numpy().astype(np.int64)
        cands = batch_m["cands"].cpu().numpy().astype(np.float32)
        quote = batch_m["quote"].cpu().numpy().astype(np.float32)
        out = self.sess.run(["logits"], {"toks": toks, "mask": mask,
                                         "cands": cands, "quote": quote})[0]
        return torch.from_numpy(out)
