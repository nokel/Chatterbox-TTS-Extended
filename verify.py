"""Install verification for Chatterbox-TTS-Extended. Run by install.ps1.

Usage: verify.py <mode>
    probe   - after the minimal-build phase: prove torch, ctranslate2 and
              onnxruntime import and actually compute on CPU before the
              multi-GB hardware-specific downloads begin.
    amd     - full check, GPU expected through ROCm (torch.cuda + CTranslate2
              GPU + WebGPU ONNX provider).
    nvidia  - full check, GPU expected through CUDA (CTranslate2 GPU is a
              warning only: it additionally needs cuDNN on PATH).
    intel   - full check, OpenVINO ONNX provider expected; torch runs on CPU.
    cpu     - full check, everything on CPU.
"""
import importlib
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "cpu"
failures = []
warnings = []


def check(name, fn, warn_only=False):
    try:
        print(f"  {name}: {fn()}")
    except Exception as e:
        (warnings if warn_only else failures).append(name)
        tag = "WARNING" if warn_only else "FAILED"
        print(f"  {name}: {tag} - {e}")


def torch_cpu():
    import torch
    x = (torch.ones(8, 8) @ torch.ones(8, 8)).sum().item()
    if x != 512.0:
        raise RuntimeError(f"matmul wrong result: {x}")
    return f"{torch.__version__} | CPU matmul ok"


def torch_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available through torch.cuda")
    backend = (f"ROCm/HIP {torch.version.hip}" if torch.version.hip
               else f"CUDA {torch.version.cuda}")
    return f"{torch.__version__} | {torch.cuda.get_device_name(0)} | {backend}"


def ct2_cpu():
    import ctranslate2
    types = ctranslate2.get_supported_compute_types("cpu")
    if not types:
        raise RuntimeError("no CPU compute types reported")
    return f"{ctranslate2.__version__} | CPU types {types}"


def ct2_gpu():
    import ctranslate2
    n = ctranslate2.get_cuda_device_count()
    if n < 1:
        raise RuntimeError("no GPU visible to CTranslate2")
    types = ctranslate2.get_supported_compute_types("cuda")
    return f"{ctranslate2.__version__} | GPU count {n} | types {types}"


def ort_with(expected):
    def _check():
        import onnxruntime as ort
        provs = ort.get_available_providers()
        if expected not in provs:
            raise RuntimeError(f"{expected} missing: {provs}")
        return f"{ort.__version__} | {provs}"
    return _check


if mode == "probe":
    print("Probe: verifying the toolchain with minimal CPU builds...")
    check("torch (CPU)", torch_cpu)
    check("ctranslate2 (CPU)", ct2_cpu)
    check("onnxruntime (CPU)", ort_with("CPUExecutionProvider"))
else:
    if mode in ("amd", "nvidia"):
        check("torch (GPU)", torch_gpu)
        check("ctranslate2 (GPU)", ct2_gpu, warn_only=(mode == "nvidia"))
    else:
        check("torch (CPU)", torch_cpu)
        check("ctranslate2 (CPU)", ct2_cpu)
    expected_ep = {
        "amd": "WebGpuExecutionProvider",
        "nvidia": "CUDAExecutionProvider",
        "intel": "OpenVINOExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }[mode]
    check(f"onnxruntime ({expected_ep})", ort_with(expected_ep))
    check("numpy", lambda: importlib.import_module("numpy").__version__)
    check("gradio", lambda: importlib.import_module("gradio").__version__)
    check("faster_whisper", lambda: importlib.import_module("faster_whisper").__version__)
    check("transformers", lambda: importlib.import_module("transformers").__version__)
    check("nltk", lambda: importlib.import_module("nltk").__version__)
    check("soundfile", lambda: importlib.import_module("soundfile").__version__)

if warnings:
    print(f"\nWarnings (not fatal): {warnings}")
    if mode == "nvidia" and "ctranslate2 (GPU)" in warnings:
        print("  faster-whisper needs cuDNN 9 for NVIDIA GPUs: "
              "pip install nvidia-cudnn-cu12, or it will run on CPU.")
if failures:
    print(f"\nVERIFICATION FAILED: {failures}")
    sys.exit(1)
print("\nAll checks passed.")
