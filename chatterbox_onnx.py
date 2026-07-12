"""
ONNX Runtime engine for Chatterbox TTS.

Runs the official ONNX export of Chatterbox (onnx-community/chatterbox-ONNX,
exported from ResembleAI/chatterbox) through ONNX Runtime execution providers:

    VitisAIExecutionProvider   AMD Ryzen AI NPU (requires Ryzen AI Software's
                               onnxruntime build)
    OpenVINOExecutionProvider  Intel CPU/GPU/NPU via OpenVINO
                               (pip install onnxruntime-openvino - replaces
                               onnxruntime-webgpu; device picked via AUTO,
                               override with CHATTERBOX_OPENVINO_DEVICE)
    WebGpuExecutionProvider    Any DirectX 12 GPU
                               (pip install onnxruntime-webgpu)
    DmlExecutionProvider       Any DirectX 12 GPU via DirectML
                               (pip install onnxruntime-directml)
    ROCm/MIGraphX/CUDA         Linux GPU builds of onnxruntime
    CPUExecutionProvider       Always available fallback

The public class `ChatterboxOnnxTTS` mirrors the interface of the PyTorch
`ChatterboxTTS` that Chatter.py uses: `.generate(text, audio_prompt_path=...,
exaggeration=..., temperature=..., cfg_weight=..., apply_watermark=...,
generator=...)` returning a (1, N) torch tensor, plus `.sr` and `.device`.

The sampling loop is a numpy re-implementation of
chatterbox.models.t3.T3._sample_loop with identical semantics and ordering:
CFG logit blend -> temperature -> repetition penalty (2.0) -> top-p (0.8)
-> softmax -> seeded multinomial sampling. Seeds are honored per call via the
`generator` argument, so candidate generation is deterministic per seed.
"""

import os
import threading
import time

import numpy as np
import torch

S3GEN_SR = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
SILENCE_TOKEN = 4299

REPO_ID = "onnx-community/chatterbox-onnx"  # kept for backward compat

# Two official ONNX exports are supported:
#  - "chatterbox": the full 0.5B model. Supports CFG (cfg_weight) and emotion
#    exaggeration exactly like the PyTorch engine, at 2x batch cost for CFG
#    and a 10-step vocoder.
#  - "chatterbox-turbo": Resemble AI's 350M low-latency model with a 1-step
#    distilled vocoder and no CFG batch doubling — much faster, but the
#    cfg_weight / exaggeration sliders have no effect; it supports
#    paralinguistic tags like [laugh] and [cough] in the text instead.
# component_precision maps the user-requested precision to the file variant
# actually loaded, per component. These substitutions are the result of
# hardware validation on WebGPU (Radeon 8060S):
#  - turbo's GPT-2 language model is numerically broken in fp16 (activation
#    overflow -> endless rambling); q4 keeps fp32 activations and is correct
#    AND faster (37 tok/s measured), so fp16/q4f16 requests map to q4.
#  - turbo's conditional decoder is wrong in fp16 on GPU (garbled words) and
#    silent in fp32 on WebGPU; it is pinned to fp32 and the load-time
#    self-test routes it to CPU, where its single distilled step is fast.
MODEL_VARIANTS = {
    "chatterbox": {
        "repo_id": "onnx-community/chatterbox-onnx",
        "supports_cfg": True,
        "supports_exaggeration": True,
        "embed_takes_positions": True,
        "lm_takes_position_ids": False,
        "component_precision": {
            "language_model": {"fp32": "fp32", "fp16": "fp16", "q8": "q4",
                               "q4": "q4", "q4f16": "q4f16"},
        },
        "repetition_penalty": 2.0,
        "silence_pad_tokens": 0,
        # 10-step CFM decoder: too slow to re-run per token block
        "stream_capable": False,
        "default_voice_repo": "onnx-community/chatterbox-onnx",
    },
    "chatterbox-turbo": {
        "repo_id": "ResembleAI/chatterbox-turbo-ONNX",
        "supports_cfg": False,
        "supports_exaggeration": False,
        "embed_takes_positions": False,
        "lm_takes_position_ids": True,
        "component_precision": {
            "language_model": {"fp32": "fp32", "fp16": "q4", "q8": "q8",
                               "q4": "q4", "q4f16": "q4"},
        },
        "repetition_penalty": 1.2,
        "silence_pad_tokens": 3,
        # 1-step distilled decoder: cheap enough to re-run per token block,
        # enabling sub-second first-audio latency in generate_stream
        "stream_capable": True,
        # turbo repo ships no default voice; reuse the classic one
        "default_voice_repo": "onnx-community/chatterbox-onnx",
    },
}

# Precision -> filename suffix used by both exports
PRECISION_SUFFIX = {
    "fp32": "",
    "fp16": "_fp16",
    "q8": "_quantized",
    "q4": "_q4",
    "q4f16": "_q4f16",
}

# Execution provider priority: NPU first, then GPU, then CPU.
# WebGPU ranks above DirectML because the Chatterbox export uses
# com.microsoft.GroupQueryAttention, which the WebGPU EP implements for
# dynamic KV caches while the DML EP does not (fails with
# "The parameter is incorrect"). Install with: pip install onnxruntime-webgpu
# OpenVINO (Intel CPU/GPU/NPU) ranks above WebGPU: it is Intel's optimized
# runtime and is only present when the user installed onnxruntime-openvino
# (which CONFLICTS with onnxruntime-webgpu - both ship the `onnxruntime`
# package, so an Intel machine installs one or the other). Ops OpenVINO
# cannot run (e.g. GroupQueryAttention) partition to the CPU EP automatically;
# the load-time probes below still verify every component actually executes.
EP_PRIORITY = [
    "VitisAIExecutionProvider",
    "OpenVINOExecutionProvider",
    "WebGpuExecutionProvider",
    "DmlExecutionProvider",
    "MIGraphXExecutionProvider",
    "ROCMExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]

_OPENVINO_DEVICE_CACHE = ["unset"]
_OPENVINO_FALLBACK_WARNED = [False]


def _openvino_device():
    """Pick the OpenVINO device_type from the devices that actually exist.
    OpenVINO's AUTO:dev1,dev2 syntax hard-fails if ANY listed device is
    absent ("Device NPU is not available" - verified), so the candidate
    list must be built from Core().available_devices, preferring
    NPU > GPU > CPU. Overridable via CHATTERBOX_OPENVINO_DEVICE
    (CPU / GPU / GPU.1 / NPU / AUTO:... / HETERO:...). Returns None to use
    the EP's default (CPU) when the device list can't be queried."""
    if _OPENVINO_DEVICE_CACHE[0] != "unset":
        return _OPENVINO_DEVICE_CACHE[0]
    device = os.environ.get("CHATTERBOX_OPENVINO_DEVICE")
    if not device:
        try:
            from openvino import Core
            devs = Core().available_devices  # e.g. ['CPU', 'GPU', 'NPU']
            order = [d for d in ("NPU", "GPU", "CPU")
                     if any(x == d or x.startswith(d + ".") for x in devs)]
            if len(order) > 1:
                device = "AUTO:" + ",".join(order)
            elif order:
                device = order[0]
        except Exception:
            device = None
    if device:
        print(f"[ONNX] OpenVINO device selection: {device}")
    _OPENVINO_DEVICE_CACHE[0] = device
    return device


def _expand_provider_options(provider_list):
    """Attach per-provider options where needed (ORT accepts mixed
    'Name' and ('Name', {options}) entries)."""
    out = []
    for p in provider_list:
        if p == "OpenVINOExecutionProvider" and _openvino_device():
            out.append((p, {"device_type": _openvino_device()}))
        else:
            out.append(p)
    return out

_ORT_TO_NP_DTYPE = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}

_WATERMARK_LOCK = threading.Lock()


def _preload_openvino_libs():
    """On Windows the OpenVINO EP DLL links against openvino.dll from the
    separate `openvino` pip package; onnxruntime-openvino ships a helper that
    puts those DLLs on PATH. Without this the EP is listed as available but
    session creation silently falls back to CPU."""
    try:
        import onnxruntime.tools.add_openvino_win_libs as _ovutils
        _ovutils.add_openvino_libs_to_path()
    except Exception:
        pass


def _materialize_model(path):
    """ORT 1.24+ validates external-data paths and rejects models whose
    .onnx_data resolves through a symlink (the Hugging Face cache uses
    symlinks when Windows developer mode is on): the canonical blob path
    'escapes' the model directory. Copy the model and its data file to a
    real directory once and load from there. Returns the new path, or the
    original if nothing needed copying."""
    import shutil
    src_dir = os.path.dirname(path)
    base = os.path.basename(path)
    candidates = [base, base + "_data"]
    if not any(os.path.islink(os.path.join(src_dir, c))
               for c in candidates if os.path.exists(os.path.join(src_dir, c))):
        return path
    # e.g. .../models--X--Y/snapshots/<hash>/onnx/file.onnx -> use the hash
    snap = os.path.basename(os.path.dirname(src_dir)) or "models"
    dst_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "onnx_models", snap)
    os.makedirs(dst_dir, exist_ok=True)
    for name in candidates:
        src = os.path.join(src_dir, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(dst_dir, name)
        if not (os.path.exists(dst) and
                os.path.getsize(dst) == os.path.getsize(src)):
            print(f"[ONNX] Copying {name} out of the symlinked HF cache "
                  f"({os.path.getsize(src) / 1e6:.0f} MB, one-time)...")
            shutil.copyfile(src, dst)  # follows the symlink
    return os.path.join(dst_dir, base)


def _select_providers(requested=None):
    import onnxruntime as ort
    available = ort.get_available_providers()
    if "OpenVINOExecutionProvider" in available:
        _preload_openvino_libs()
    if requested:
        providers = [p for p in requested if p in available]
    else:
        providers = [p for p in EP_PRIORITY if p in available]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
    return providers


def _apply_repetition_penalty(logits, generated_ids, penalty):
    # Same math as transformers.RepetitionPenaltyLogitsProcessor
    score = np.take_along_axis(logits, generated_ids, axis=1)
    score = np.where(score < 0, score * penalty, score / penalty)
    out = logits.copy()
    np.put_along_axis(out, generated_ids, score, axis=1)
    return out


def _apply_top_p(logits, top_p):
    # Same semantics as transformers.TopPLogitsWarper (min_tokens_to_keep=1)
    sorted_indices = np.argsort(logits, axis=-1)  # ascending
    sorted_logits = np.take_along_axis(logits, sorted_indices, axis=-1)
    shifted = sorted_logits - sorted_logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    cumulative = np.cumsum(probs, axis=-1)
    to_remove = cumulative <= (1.0 - top_p)
    to_remove[..., -1] = False  # always keep the top token
    masked = np.where(to_remove, -np.inf, sorted_logits)
    out = np.full_like(logits, -np.inf)
    np.put_along_axis(out, sorted_indices, masked, axis=-1)
    return out


class ChatterboxOnnxTTS:
    sr = S3GEN_SR

    def __init__(self, speech_encoder, embed_tokens, language_model,
                 conditional_decoder, tokenizer, providers, lm_precision,
                 model_variant="chatterbox"):
        self.speech_encoder = speech_encoder
        self.embed_tokens = embed_tokens
        self.language_model = language_model
        self.conditional_decoder = conditional_decoder
        self.tokenizer = tokenizer
        self.providers = providers
        self.lm_precision = lm_precision
        self.model_variant = model_variant
        self.variant = MODEL_VARIANTS[model_variant]
        self.device = f"onnx:{providers[0]}"
        self._warned_unsupported = set()

        lm_inputs = {i.name: i.type for i in language_model.get_inputs()}
        self.embeds_dtype = _ORT_TO_NP_DTYPE.get(
            lm_inputs.get("inputs_embeds", "tensor(float)"), np.float32)
        self.kv_dtype = _ORT_TO_NP_DTYPE.get(
            lm_inputs.get("past_key_values.0.key", "tensor(float)"), np.float32)
        self.kv_input_names = [n for n in lm_inputs if n.startswith("past_key_values.")]
        # Map LM outputs back to the matching past-KV inputs BY NAME
        # (present.X.key -> past_key_values.X.key); positional matching is
        # unsafe if the graph orders inputs and outputs differently.
        self.lm_output_names = [o.name for o in language_model.get_outputs()]
        self.present_to_past = {}
        for out_name in self.lm_output_names:
            if out_name.startswith("present"):
                past_name = out_name.replace("present", "past_key_values", 1)
                if past_name in lm_inputs:
                    self.present_to_past[out_name] = past_name

        # Infer KV-cache geometry from the graph instead of hardcoding
        kv0 = next(i for i in language_model.get_inputs()
                   if i.name == "past_key_values.0.key")
        # shape: [batch, num_kv_heads, seq, head_dim] with symbolic dims
        self.num_kv_heads = kv0.shape[1] if isinstance(kv0.shape[1], int) else 16
        self.head_dim = kv0.shape[3] if isinstance(kv0.shape[3], int) else 64

        self._voice_cache = {}
        self._voice_cache_lock = threading.Lock()
        self._watermarker = None

    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(cls, lm_precision="fp16", providers=None,
                        model_variant="chatterbox") -> "ChatterboxOnnxTTS":
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        if model_variant not in MODEL_VARIANTS:
            print(f"[ONNX] Unknown model variant '{model_variant}', using chatterbox.")
            model_variant = "chatterbox"
        variant = MODEL_VARIANTS[model_variant]
        repo_id = variant["repo_id"]

        if lm_precision not in PRECISION_SUFFIX:
            print(f"[ONNX] Unknown precision '{lm_precision}', using fp16.")
            lm_precision = "fp16"

        providers = _select_providers(providers)
        print(f"[ONNX] Model: {model_variant} | precision: {lm_precision}")
        print(f"[ONNX] Execution providers (in priority order): {providers}")

        def fetch(name):
            # Resolve the actual file variant through the per-component
            # validation map; components without an entry are pinned to fp32.
            resolved = variant["component_precision"].get(name, {}).get(
                lm_precision, "fp32")
            if resolved != lm_precision and name in variant["component_precision"]:
                print(f"[ONNX] {name}: using {resolved} instead of {lm_precision} "
                      f"(validated-correct variant for this model)")
            filename = f"{name}{PRECISION_SUFFIX[resolved]}.onnx"
            try:
                path = hf_hub_download(repo_id=repo_id, filename=filename,
                                       subfolder="onnx")
                hf_hub_download(repo_id=repo_id, filename=f"{filename}_data",
                                subfolder="onnx")
            except Exception:
                if filename == f"{name}.onnx":
                    raise
                print(f"[ONNX] {filename} not available; falling back to {name}.onnx")
                path = hf_hub_download(repo_id=repo_id, filename=f"{name}.onnx",
                                       subfolder="onnx")
                hf_hub_download(repo_id=repo_id, filename=f"{name}.onnx_data",
                                subfolder="onnx")
            return path

        print("[ONNX] Downloading/locating Chatterbox ONNX models "
              "(first run downloads several GB)...")
        speech_encoder_path = fetch("speech_encoder")
        embed_tokens_path = fetch("embed_tokens")
        decoder_path = fetch("conditional_decoder")
        lm_path = fetch("language_model")

        def session(path, provider_list):
            so = ort.SessionOptions()
            if "DmlExecutionProvider" in provider_list:
                # DirectML requires memory pattern off and sequential execution
                so.enable_mem_pattern = False
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            def finish(sess):
                # ORT does not raise when an EP fails to initialize - it
                # silently drops to the next provider. Surface that once for
                # OpenVINO, whose usual cause is a missing or version-
                # mismatched `openvino` package.
                if ("OpenVINOExecutionProvider" in provider_list
                        and "OpenVINOExecutionProvider" not in sess.get_providers()
                        and not _OPENVINO_FALLBACK_WARNED[0]):
                    _OPENVINO_FALLBACK_WARNED[0] = True
                    print("[ONNX] WARNING: the OpenVINO provider could not "
                          "initialize; running on the next provider instead. "
                          "Usual fix: install the `openvino` package version "
                          "matching onnxruntime-openvino (1.24.x pairs with "
                          "openvino==2025.4.*).")
                return sess

            def create(model_path):
                try:
                    return finish(ort.InferenceSession(
                        model_path, sess_options=so,
                        providers=_expand_provider_options(provider_list)))
                except Exception as e:
                    # Provider options can be rejected (e.g. an OpenVINO
                    # device_type naming a device this driver stack doesn't
                    # expose). Retry with bare provider names - default
                    # options - rather than losing the provider entirely.
                    if (provider_list != _expand_provider_options(provider_list)
                            and "External data" not in str(e)):
                        print(f"[ONNX] Session with provider options failed "
                              f"({str(e).splitlines()[0][:120]}); retrying "
                              f"with default provider options.")
                        return finish(ort.InferenceSession(
                            model_path, sess_options=so,
                            providers=provider_list))
                    raise

            try:
                return create(path)
            except Exception as e:
                # ORT 1.24+ rejects external data reached through HF-cache
                # symlinks; copy the model to a real directory and retry.
                if "External data" in str(e):
                    real = _materialize_model(path)
                    if real != path:
                        return create(real)
                raise

        print("[ONNX] Creating inference sessions...")
        speech_encoder = session(speech_encoder_path, providers)
        embed_tokens = session(embed_tokens_path, providers)
        conditional_decoder = session(decoder_path, providers)

        # The language model is the provider-sensitive part (it uses the
        # GroupQueryAttention contrib op). Probe each provider tier with a
        # dummy prefill and drop to the next tier if execution fails, so a
        # provider that loads but cannot actually run the graph (e.g. DML)
        # is skipped instead of crashing during generation.
        language_model = None
        lm_providers = list(providers)
        while lm_providers:
            candidate = session(lm_path, lm_providers)
            try:
                cls._probe_language_model(candidate)
                language_model = candidate
                break
            except Exception as e:
                failed = lm_providers[0]
                if failed == "CPUExecutionProvider":
                    raise
                print(f"[ONNX] {failed} cannot execute the language model "
                      f"({str(e).splitlines()[0][:160]}); trying next provider.")
                del candidate
                lm_providers = lm_providers[1:]
        if language_model is None:
            raise RuntimeError("No ONNX execution provider could run the language model.")
        if lm_providers != providers:
            print(f"[ONNX] Language model providers: {lm_providers}")

        tokenizer = AutoTokenizer.from_pretrained(repo_id)

        model = cls(speech_encoder, embed_tokens, language_model,
                    conditional_decoder, tokenizer, lm_providers, lm_precision,
                    model_variant=model_variant)

        # Functional self-test: render the default voice's own tokens through
        # the decoder and check the output is audible. Execution providers can
        # load a graph fine yet miscompute it (verified on WebGPU: the classic
        # conditional_decoder produces pure silence) — only an end-to-end
        # audio check catches that. On failure, rebuild the decoder on the
        # next provider tier and retest.
        dec_providers = list(providers)
        while True:
            try:
                if model._decoder_self_test():
                    break
                reason = "produces silent audio"
            except Exception as e:
                reason = f"failed: {str(e).splitlines()[0][:120]}"
            failed = dec_providers[0]
            if failed == "CPUExecutionProvider" or len(dec_providers) == 1:
                print(f"[ONNX] WARNING: decoder self-test {reason} even on "
                      f"{failed}; audio output may be degraded.")
                break
            dec_providers = dec_providers[1:]
            print(f"[ONNX] {failed} decoder {reason}; retrying on {dec_providers[0]}.")
            model.conditional_decoder = session(decoder_path, dec_providers)
        if dec_providers != providers:
            print(f"[ONNX] Decoder providers: {dec_providers}")

        print(f"[ONNX] Ready ({model_variant}, precision: {lm_precision}).")
        return model

    def _decoder_self_test(self) -> bool:
        """Render the default voice's own tokens; True if audibly non-silent."""
        cond_emb, prompt_token, ref_x_vector, prompt_feat = \
            self._get_voice_conditioning(None)
        prompt_region = prompt_feat.shape[1] // 2
        front_pad = max(0, prompt_region - prompt_token.shape[1])
        parts = []
        if front_pad:
            parts.append(np.full((1, front_pad), SILENCE_TOKEN, dtype=np.int64))
        parts += [prompt_token, prompt_token]
        tokens = np.concatenate(parts, axis=1).astype(np.int64)
        wav = self.conditional_decoder.run(None, {
            "speech_tokens": tokens,
            "speaker_embeddings": ref_x_vector,
            "speaker_features": prompt_feat,
        })[0]
        wav = np.asarray(wav, dtype=np.float32).squeeze()
        rms = float(np.sqrt(np.mean(wav ** 2)))
        print(f"[ONNX] Decoder self-test RMS: {rms:.4f} "
              f"({'ok' if rms > 5e-3 else 'silent'})")
        return rms > 5e-3

    @staticmethod
    def _probe_language_model(lm_session):
        """Run a tiny dummy prefill to verify the provider can execute the graph."""
        lm_inputs = {i.name: i for i in lm_session.get_inputs()}
        embeds_info = lm_inputs["inputs_embeds"]
        hidden = embeds_info.shape[2] if isinstance(embeds_info.shape[2], int) else 1024
        embeds_dtype = _ORT_TO_NP_DTYPE.get(embeds_info.type, np.float32)
        kv0 = lm_inputs["past_key_values.0.key"]
        kv_dtype = _ORT_TO_NP_DTYPE.get(kv0.type, np.float32)
        num_kv_heads = kv0.shape[1] if isinstance(kv0.shape[1], int) else 16
        head_dim = kv0.shape[3] if isinstance(kv0.shape[3], int) else 64

        seq = 8
        feed = {
            "inputs_embeds": np.zeros((1, seq, hidden), dtype=embeds_dtype),
            "attention_mask": np.ones((1, seq), dtype=np.int64),
        }
        if "position_ids" in lm_inputs:  # turbo variant
            feed["position_ids"] = np.arange(seq, dtype=np.int64)[np.newaxis, :]
        for name in lm_inputs:
            if name.startswith("past_key_values."):
                feed[name] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=kv_dtype)
        lm_session.run(None, feed)

    # ------------------------------------------------------------------ #

    def _get_voice_conditioning(self, audio_prompt_path):
        """Run (and cache) the speech encoder for a reference voice."""
        import librosa

        if audio_prompt_path:
            key = (os.path.abspath(audio_prompt_path),
                   os.path.getmtime(audio_prompt_path))
            path = audio_prompt_path
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=self.variant["default_voice_repo"],
                                   filename="default_voice.wav")
            key = ("__default__", 0)

        with self._voice_cache_lock:
            if key in self._voice_cache:
                return self._voice_cache[key]

        audio_values, _ = librosa.load(path, sr=S3GEN_SR, mono=True)
        audio_values = audio_values[np.newaxis, :].astype(np.float32)
        cond_emb, prompt_token, ref_x_vector, prompt_feat = self.speech_encoder.run(
            None, {"audio_values": audio_values})

        result = (cond_emb, prompt_token.astype(np.int64), ref_x_vector, prompt_feat)
        with self._voice_cache_lock:
            self._voice_cache[key] = result
        return result

    def _embed(self, input_ids, position_ids, exaggeration):
        if not self.variant["embed_takes_positions"]:  # turbo: input_ids only
            return self.embed_tokens.run(None, {"input_ids": input_ids})[0]
        return self.embed_tokens.run(None, {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "exaggeration": np.array([exaggeration], dtype=np.float32),
        })[0]

    def _warn_once(self, key, message):
        if key not in self._warned_unsupported:
            self._warned_unsupported.add(key)
            print(message)

    # ------------------------------------------------------------------ #

    def generate(
        self,
        text: str,
        audio_prompt_path=None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        apply_watermark: bool = False,
        generator=None,
        max_new_tokens: int = 1000,
        top_p: float = 0.8,
        repetition_penalty: float = None,
        **kwargs,
    ) -> torch.Tensor:
        variant = self.variant
        if repetition_penalty is None:
            repetition_penalty = variant["repetition_penalty"]

        # Deterministic per-call RNG, honoring the caller's torch.Generator seed
        if generator is not None:
            seed = int(generator.initial_seed())
        else:
            seed = int(torch.initial_seed())
        rng = np.random.default_rng(seed & 0xFFFFFFFFFFFFFFFF)

        cond_emb, prompt_token, ref_x_vector, prompt_feat = \
            self._get_voice_conditioning(audio_prompt_path)

        input_ids = self.tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)
        seq_len = input_ids.shape[1]
        position_ids = np.where(
            input_ids >= START_SPEECH_TOKEN,
            0,
            np.arange(seq_len, dtype=np.int64)[np.newaxis, :] - 1,
        ).astype(np.int64)

        text_embeds = self._embed(input_ids, position_ids, exaggeration)

        if not variant["supports_exaggeration"] and abs(exaggeration - 0.5) > 1e-6:
            self._warn_once("exaggeration",
                            "[ONNX] chatterbox-turbo does not support the exaggeration "
                            "control; the slider is ignored (use tags like [laugh] instead).")
        use_cfg = variant["supports_cfg"] and cfg_weight > 0.0
        if not variant["supports_cfg"] and cfg_weight > 0.0:
            self._warn_once("cfg",
                            "[ONNX] chatterbox-turbo does not use CFG; the CFG weight "
                            "slider is ignored (this is also why it is ~2x faster).")
        batch = 2 if use_cfg else 1

        cond_row = np.concatenate([cond_emb, text_embeds], axis=1)
        if use_cfg:
            # Uncond row: same voice conditioning, text embeddings zeroed —
            # exactly what T3.prepare_input_embeds does for CFG (rows 1,3,5...)
            uncond_text = text_embeds.copy()
            uncond_text[input_ids < START_SPEECH_TOKEN] = 0.0
            uncond_row = np.concatenate([cond_emb, uncond_text], axis=1)
            inputs_embeds = np.concatenate([cond_row, uncond_row], axis=0)
        else:
            inputs_embeds = cond_row

        past_key_values = {
            name: np.zeros((batch, self.num_kv_heads, 0, self.head_dim),
                           dtype=self.kv_dtype)
            for name in self.kv_input_names
        }
        attention_mask = np.ones((batch, inputs_embeds.shape[1]), dtype=np.int64)
        generated_ids = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)

        # The turbo LM takes explicit position_ids (0..seq-1 over the whole
        # prefix, then +1 per generated token).
        lm_position_ids = None
        if variant["lm_takes_position_ids"]:
            lm_position_ids = np.arange(inputs_embeds.shape[1],
                                        dtype=np.int64)[np.newaxis, :]
            lm_position_ids = np.repeat(lm_position_ids, batch, axis=0)

        for i in range(max_new_tokens):
            feed = dict(
                inputs_embeds=inputs_embeds.astype(self.embeds_dtype),
                attention_mask=attention_mask,
                **past_key_values,
            )
            if lm_position_ids is not None:
                feed["position_ids"] = lm_position_ids
            outputs = self.language_model.run(None, feed)
            logits, present_key_values = outputs[0], outputs[1:]
            step_logits = logits[:, -1, :].astype(np.float64)

            # Identical ordering to T3._sample_loop:
            if use_cfg:
                logits_cond = step_logits[0:1]
                logits_uncond = step_logits[1:2]
                step_logits = logits_cond + cfg_weight * (logits_cond - logits_uncond)
            if temperature != 1.0:
                step_logits = step_logits / temperature
            step_logits = _apply_repetition_penalty(
                step_logits, generated_ids, repetition_penalty)
            step_logits = _apply_top_p(step_logits, top_p)

            shifted = step_logits - step_logits.max(axis=-1, keepdims=True)
            probs = np.exp(shifted)
            probs = np.where(np.isfinite(probs), probs, 0.0)
            probs /= probs.sum(axis=-1, keepdims=True)
            next_token = int(rng.choice(probs.shape[-1], p=probs[0]))

            generated_ids = np.concatenate(
                [generated_ids, np.array([[next_token]], dtype=np.int64)], axis=1)
            if next_token == STOP_SPEECH_TOKEN:
                break

            token_embed = self._embed(
                np.array([[next_token]], dtype=np.int64),
                np.full((1, 1), i + 1, dtype=np.int64),
                exaggeration,
            )
            inputs_embeds = np.repeat(token_embed, batch, axis=0)
            attention_mask = np.concatenate(
                [attention_mask, np.ones((batch, 1), dtype=np.int64)], axis=1)
            if lm_position_ids is not None:
                lm_position_ids = lm_position_ids[:, -1:] + 1
            present_by_name = dict(zip(self.lm_output_names[1:], present_key_values))
            past_key_values = {
                self.present_to_past[out_name]: value
                for out_name, value in present_by_name.items()
                if out_name in self.present_to_past
            }

        # START/STOP stripped; drop any invalid (>= 6561) tokens like the
        # PyTorch path does, then prepend the reference prompt tokens
        # (turbo also appends a few silence tokens for a clean ending).
        speech_tokens = generated_ids[0, 1:]
        speech_tokens = speech_tokens[speech_tokens < START_SPEECH_TOKEN]
        # The decoder renders only the frames after the reference-prompt
        # region, which spans mel_frames/2 token positions. When the encoder
        # returns fewer prompt tokens than that (chatterbox-turbo: 34 tokens
        # vs 186 frames), left-pad with silence tokens so the trim boundary
        # falls exactly at the start of the generated speech — otherwise the
        # first (mel_frames/2 - len(prompt)) tokens of speech are cut off.
        prompt_region = prompt_feat.shape[1] // 2
        front_pad = max(0, prompt_region - prompt_token.shape[1])
        parts = []
        if front_pad:
            parts.append(np.full((1, front_pad), SILENCE_TOKEN, dtype=np.int64))
        parts += [prompt_token, speech_tokens[np.newaxis, :]]
        if variant["silence_pad_tokens"]:
            parts.append(np.full((1, variant["silence_pad_tokens"]),
                                 SILENCE_TOKEN, dtype=np.int64))
        speech_tokens = np.concatenate(parts, axis=1).astype(np.int64)

        wav = self.conditional_decoder.run(None, {
            "speech_tokens": speech_tokens,
            "speaker_embeddings": ref_x_vector,
            "speaker_features": prompt_feat,
        })[0]
        wav = np.asarray(wav, dtype=np.float32).squeeze()

        self.last_stats = {
            "new_tokens": int(generated_ids.shape[1] - 1),
            "prompt_tokens": int(prompt_token.shape[1]),
            "decoder_tokens": int(speech_tokens.shape[1]),
            "wav_seconds": float(wav.shape[-1]) / self.sr,
        }

        if apply_watermark:
            wav = self._watermark(wav)

        return torch.from_numpy(wav).unsqueeze(0)

    # ------------------------------------------------------------------ #

    def _decode_speech(self, speech_tokens_1d, prompt_token, ref_x_vector,
                       prompt_feat, end_pad):
        """Decode generated speech tokens to a float32 waveform, applying the
        prompt-region front padding (see generate() for the explanation)."""
        prompt_region = prompt_feat.shape[1] // 2
        front_pad = max(0, prompt_region - prompt_token.shape[1])
        parts = []
        if front_pad:
            parts.append(np.full((1, front_pad), SILENCE_TOKEN, dtype=np.int64))
        parts += [prompt_token, speech_tokens_1d[np.newaxis, :]]
        if end_pad:
            parts.append(np.full((1, end_pad), SILENCE_TOKEN, dtype=np.int64))
        tokens = np.concatenate(parts, axis=1).astype(np.int64)
        wav = self.conditional_decoder.run(None, {
            "speech_tokens": tokens,
            "speaker_embeddings": ref_x_vector,
            "speaker_features": prompt_feat,
        })[0]
        return np.asarray(wav, dtype=np.float32).squeeze()

    def generate_stream(
        self,
        text: str,
        audio_prompt_path=None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        generator=None,
        max_new_tokens: int = 1000,
        top_p: float = 0.8,
        repetition_penalty: float = None,
        first_block_tokens: int = 25,
        block_tokens: int = 38,
        crossfade_sec: float = 0.10,
        **kwargs,
    ):
        """
        Streaming synthesis: yields (sr, float32 waveform) increments while
        tokens are still being generated. First audio is emitted after
        ~first_block_tokens tokens (~1s of speech) instead of after the whole
        sentence. Only variants with a 1-step decoder support this; others
        fall back to one full-sentence chunk.

        The decoder is non-causal, so each incremental decode re-renders the
        whole sequence; previously emitted audio may differ slightly at the
        junction, which a short crossfade smooths over.
        """
        variant = self.variant
        if not variant.get("stream_capable"):
            wav = self.generate(
                text, audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration, cfg_weight=cfg_weight,
                temperature=temperature, apply_watermark=False,
                generator=generator, max_new_tokens=max_new_tokens,
                top_p=top_p, repetition_penalty=repetition_penalty)
            yield (self.sr, wav.squeeze(0).numpy().astype(np.float32))
            return

        if repetition_penalty is None:
            repetition_penalty = variant["repetition_penalty"]
        if generator is not None:
            seed = int(generator.initial_seed())
        else:
            seed = int(torch.initial_seed())
        rng = np.random.default_rng(seed & 0xFFFFFFFFFFFFFFFF)

        cond_emb, prompt_token, ref_x_vector, prompt_feat = \
            self._get_voice_conditioning(audio_prompt_path)

        input_ids = self.tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)
        seq_len = input_ids.shape[1]
        position_ids = np.where(
            input_ids >= START_SPEECH_TOKEN,
            0,
            np.arange(seq_len, dtype=np.int64)[np.newaxis, :] - 1,
        ).astype(np.int64)
        text_embeds = self._embed(input_ids, position_ids, exaggeration)

        # Same sampling loop as generate() (turbo: single batch, no CFG).
        inputs_embeds = np.concatenate([cond_emb, text_embeds], axis=1)
        past_key_values = {
            name: np.zeros((1, self.num_kv_heads, 0, self.head_dim),
                           dtype=self.kv_dtype)
            for name in self.kv_input_names
        }
        attention_mask = np.ones((1, inputs_embeds.shape[1]), dtype=np.int64)
        generated_ids = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)
        lm_position_ids = None
        if variant["lm_takes_position_ids"]:
            lm_position_ids = np.arange(inputs_embeds.shape[1],
                                        dtype=np.int64)[np.newaxis, :]

        # Incremental emission state
        cf = max(1, int(self.sr * crossfade_sec))
        sent_upto = 0        # decoded-timeline index already yielded
        prev_tail = None     # held-back cf samples from the previous decode
        next_emit = first_block_tokens
        valid_tokens = []

        def emit(full, final=False):
            nonlocal sent_upto, prev_tail
            end = len(full) if final else max(sent_upto, len(full) - cf)
            if end <= sent_upto and not final:
                return None
            if prev_tail is not None and sent_upto < len(full):
                n = min(len(prev_tail), len(full) - sent_upto, end - sent_upto)
                if n > 0:
                    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
                    blended = prev_tail[:n] * (1.0 - ramp) + \
                        full[sent_upto:sent_upto + n] * ramp
                    out = np.concatenate([blended, full[sent_upto + n:end]])
                else:
                    out = full[sent_upto:end]
            else:
                out = full[sent_upto:end]
            prev_tail = None if final else full[end:end + cf].copy()
            sent_upto = end
            return out if len(out) else None

        # The LM (GPU) and vocoder (CPU) run CONCURRENTLY: the token loop
        # runs in a worker thread while this generator decodes and yields.
        # Alternating them serially drops net production to ~1x realtime,
        # which makes playback pause; overlapped, the full LM rate is kept.
        import threading as _threading
        lm_stop = _threading.Event()
        lm_done = _threading.Event()
        tokens_lock = _threading.Lock()

        def _lm_loop():
            nonlocal inputs_embeds, attention_mask, lm_position_ids, \
                past_key_values, generated_ids
            try:
                for i in range(max_new_tokens):
                    if lm_stop.is_set():
                        break
                    feed = dict(
                        inputs_embeds=inputs_embeds.astype(self.embeds_dtype),
                        attention_mask=attention_mask,
                        **past_key_values,
                    )
                    if lm_position_ids is not None:
                        feed["position_ids"] = lm_position_ids
                    outputs = self.language_model.run(None, feed)
                    logits, present_key_values = outputs[0], outputs[1:]
                    step_logits = logits[:, -1, :].astype(np.float64)

                    if temperature != 1.0:
                        step_logits = step_logits / temperature
                    step_logits = _apply_repetition_penalty(
                        step_logits, generated_ids, repetition_penalty)
                    step_logits = _apply_top_p(step_logits, top_p)
                    shifted = step_logits - step_logits.max(axis=-1, keepdims=True)
                    probs = np.exp(shifted)
                    probs = np.where(np.isfinite(probs), probs, 0.0)
                    probs /= probs.sum(axis=-1, keepdims=True)
                    next_token = int(rng.choice(probs.shape[-1], p=probs[0]))

                    generated_ids = np.concatenate(
                        [generated_ids, np.array([[next_token]], dtype=np.int64)],
                        axis=1)
                    if next_token == STOP_SPEECH_TOKEN:
                        break
                    if next_token < START_SPEECH_TOKEN:
                        with tokens_lock:
                            valid_tokens.append(next_token)

                    token_embed = self._embed(
                        np.array([[next_token]], dtype=np.int64),
                        np.full((1, 1), i + 1, dtype=np.int64),
                        exaggeration,
                    )
                    inputs_embeds = token_embed
                    attention_mask = np.concatenate(
                        [attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1)
                    if lm_position_ids is not None:
                        lm_position_ids = lm_position_ids[:, -1:] + 1
                    present_by_name = dict(
                        zip(self.lm_output_names[1:], present_key_values))
                    past_key_values = {
                        self.present_to_past[out_name]: value
                        for out_name, value in present_by_name.items()
                        if out_name in self.present_to_past
                    }
            finally:
                lm_done.set()

        lm_thread = _threading.Thread(target=_lm_loop, daemon=True)
        lm_thread.start()

        try:
            while True:
                with tokens_lock:
                    n_tokens = len(valid_tokens)
                if n_tokens >= next_emit:
                    with tokens_lock:
                        snapshot = np.asarray(valid_tokens, dtype=np.int64)
                    full = self._decode_speech(
                        snapshot, prompt_token, ref_x_vector, prompt_feat,
                        end_pad=0)
                    out = emit(full)
                    if out is not None:
                        yield (self.sr, out)
                    next_emit = len(snapshot) + block_tokens
                elif lm_done.is_set():
                    break
                else:
                    time.sleep(0.02)

            # Final decode with the end-silence padding for a clean tail.
            with tokens_lock:
                snapshot = np.asarray(valid_tokens, dtype=np.int64)
            if len(snapshot):
                full = self._decode_speech(
                    snapshot, prompt_token, ref_x_vector, prompt_feat,
                    end_pad=variant["silence_pad_tokens"])
                out = emit(full, final=True)
                if out is not None:
                    yield (self.sr, out)
        finally:
            lm_stop.set()

    # ------------------------------------------------------------------ #

    def _watermark(self, wav):
        try:
            with _WATERMARK_LOCK:
                if self._watermarker is None:
                    import perth
                    self._watermarker = perth.PerthImplicitWatermarker()
                return self._watermarker.apply_watermark(wav, sample_rate=self.sr)
        except Exception as e:
            print(f"[ONNX] Watermarking failed ({e}); returning unwatermarked audio.")
            return wav
