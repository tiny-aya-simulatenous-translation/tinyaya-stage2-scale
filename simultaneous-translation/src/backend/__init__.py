"""Backend abstraction for GPU/TPU training."""
import os


def detect_backend() -> str:
    """Detect available backend. Returns 'tpu' or 'gpu'."""
    explicit = os.environ.get("DEVICE_BACKEND", "").lower()
    if explicit in ("tpu", "gpu"):
        return explicit

    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        return "tpu"
    except Exception:
        return "gpu"


def get_backend(backend_type: str | None = None):
    """Get backend instance. Auto-detects if backend_type is None."""
    if backend_type is None:
        backend_type = detect_backend()

    if backend_type == "tpu":
        from src.backend.tpu_backend import TPUBackend
        return TPUBackend()
    else:
        from src.backend.gpu_backend import GPUBackend
        return GPUBackend()
