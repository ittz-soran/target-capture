"""Use an NVIDIA GPU when the GPU libraries are installed (setup_gpu_windows.bat), else the CPU.

YOLO, EasyOCR and the car-model network run on PyTorch and pick the GPU by themselves when
torch.cuda is available. The plate reader and face models run on ONNX Runtime, which needs
onnxruntime-gpu and the CUDA/cuDNN libraries; the CUDA build of PyTorch ships those, so they
are loaded from there.
"""
from __future__ import annotations

import functools
import logging

log = logging.getLogger(__name__)
CPU = ["CPUExecutionProvider"]


@functools.lru_cache(maxsize=None)
def torch_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def half() -> dict:
    """YOLO options: half precision on an NVIDIA card (about twice as fast), nothing on CPU."""
    if not torch_cuda():
        return {}
    return {"quantize": 16} if _has_quantize() else {"half": True}


@functools.lru_cache(maxsize=None)
def _has_quantize() -> bool:
    """Newer Ultralytics renamed 'half' to 'quantize' and warns on the old name."""
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT

        return "quantize" in DEFAULT_CFG_DICT
    except Exception:
        return False


@functools.lru_cache(maxsize=None)
def onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except Exception:
        return CPU
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return CPU
    if torch_cuda() and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()  # CUDA + cuDNN DLLs from the PyTorch install
        except Exception as e:
            log.warning("could not load CUDA libraries for ONNX Runtime: %s", e)
    # HEURISTIC: no slow search for the best kernels on every new input size; kSameAsRequested:
    # grow GPU memory only as needed (4 GB cards such as the GTX 1650 run out otherwise, and Windows
    # then quietly moves work to much slower shared memory)
    return [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC",
                                       "arena_extend_strategy": "kSameAsRequested"}), *CPU]


def onnx_on_gpu() -> bool:
    return onnx_providers()[0] != "CPUExecutionProvider"


@functools.lru_cache(maxsize=None)
def gpu_memory_gb() -> float:
    """Memory of the NVIDIA card in GB (0 without one)."""
    if not torch_cuda():
        return 0.0
    import torch

    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def gpu_memory_used() -> str | None:
    """'2.1 of 4.0 GB' used on the card right now (all programs), None without a card."""
    if not torch_cuda():
        return None
    try:
        import torch

        free, total = torch.cuda.mem_get_info()
        return f"{(total - free) / 1024 ** 3:.1f} of {total / 1024 ** 3:.1f} GB"
    except Exception:
        return None


def describe() -> str:
    """One line for the screen: what will do the work."""
    name = None
    if torch_cuda():
        import torch

        name = torch.cuda.get_device_name(0)
    if name and onnx_on_gpu():
        return f"GPU: {name} ({gpu_memory_gb():.0f} GB)"
    if name:
        return f"GPU for detection: {name} (plate reader and faces on CPU: Settings, Install graphics-card support)"
    return "CPU only (for an NVIDIA GPU: Settings, Install graphics-card support)"
