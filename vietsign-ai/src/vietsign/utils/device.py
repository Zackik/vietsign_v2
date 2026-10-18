"""Safe runtime device selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeDevice:
    """Selected runtime device information."""

    type: str

    def __str__(self) -> str:
        return self.type


def _import_torch():
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    return torch


def select_device(
    preference: str = "auto",
    *,
    allow_cuda: bool = True,
    allow_mps: bool = True,
) -> RuntimeDevice:
    """Select CUDA, MPS, or CPU without crashing when accelerators are unavailable."""

    normalized = preference.lower()
    if normalized not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("preference must be one of: auto, cuda, mps, cpu")

    torch = _import_torch()
    if normalized == "cpu" or torch is None:
        return RuntimeDevice("cpu")

    cuda_available = bool(
        allow_cuda and hasattr(torch, "cuda") and torch.cuda.is_available()
    )
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(
        allow_mps and mps_backend is not None and mps_backend.is_available()
    )

    if normalized == "cuda":
        return RuntimeDevice("cuda" if cuda_available else "cpu")
    if normalized == "mps":
        return RuntimeDevice("mps" if mps_available else "cpu")
    if cuda_available:
        return RuntimeDevice("cuda")
    if mps_available:
        return RuntimeDevice("mps")
    return RuntimeDevice("cpu")

