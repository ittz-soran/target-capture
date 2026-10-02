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
    return {"half": True} if torch_cuda() else {}


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
    return ["CUDAExecutionProvider", *CPU]


def describe() -> str:
    """One line for the screen: what will do the work."""
    name = None
    if torch_cuda():
        import torch

        name = torch.cuda.get_device_name(0)
    if name and onnx_providers()[0] != "CPUExecutionProvider":
        return f"GPU: {name}"
    if name:
        return f"GPU for detection: {name} (plate reader and faces on CPU, run setup_gpu_windows.bat)"
    return "CPU only (for an NVIDIA GPU run setup_gpu_windows.bat)"
