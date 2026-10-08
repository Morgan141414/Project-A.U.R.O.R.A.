"""Device selection and memory reporting."""

import torch


def system_vitals() -> dict[str, float]:
    """Return host vitals for the UI without relying on CUDA-only APIs."""
    try:
        import psutil
        return {"cpu_percent": float(psutil.cpu_percent(interval=None)), "memory_percent": float(psutil.virtual_memory().percent)}
    except ImportError:
        return {"cpu_percent": 0.0, "memory_percent": 0.0}


def get_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def memory_usage(device: torch.device) -> str:
    if device.type == "cuda":
        return f"{torch.cuda.memory_allocated(device) / 1024**2:.1f} MB"
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        return f"{torch.mps.current_allocated_memory() / 1024**2:.1f} MB"
    return "n/a"
