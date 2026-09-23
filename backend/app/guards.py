"""Memory guards. Running out of RAM cannot damage hardware, but it makes the machine swap or the
process get killed mid-run, so every heavy step checks first and refuses or stops cleanly."""
from typing import Any

import psutil

# headroom kept free for the OS and everything else
RESERVE_MB = 1500
# stop a running job if free memory falls below this
CRITICAL_MB = 800
# PCM16 @ 16 kHz
BYTES_PER_AUDIO_SECOND = 32 * 1024


class NotEnoughMemory(RuntimeError):
    pass


def free_mb() -> float:
    return psutil.virtual_memory().available / 2**20


def gpu_free_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(0)
            return free / 2**20
    except Exception:
        pass
    return None


def model_need_mb(params: float, method: str, device: str) -> float:
    """Rough working-set estimate. fp32 on CPU, fp16 on GPU; full training also holds
    gradients + Adam moments (~4x weights), LoRA holds little beyond the weights."""
    bytes_per_param = 2 if device == "cuda" else 4
    weights = params * bytes_per_param / 2**20
    factor = 4.0 if method == "full" else 1.4
    return weights * factor + 800  # + activations/feature extraction headroom


def check(need_mb: float, what: str, device: str = "cpu") -> None:
    if device == "cuda":
        g = gpu_free_mb()
        if g is not None and need_mb > g:
            raise NotEnoughMemory(
                f"{what} needs about {need_mb / 1024:.1f} GB of VRAM but only {g / 1024:.1f} GB is free. "
                "Use LoRA instead of a full fine-tune, pick a smaller model, or lower the batch size."
            )
        return
    free = free_mb()
    if need_mb + RESERVE_MB > free:
        raise NotEnoughMemory(
            f"{what} needs about {need_mb / 1024:.1f} GB of RAM but only {free / 1024:.1f} GB is free. "
            "Pick a smaller model, use LoRA instead of a full fine-tune, lower the batch size, or close other programs."
        )


def critical() -> bool:
    return free_mb() < CRITICAL_MB


def estimate_prepare_mb(clips: int, avg_seconds: float = 6.0) -> float:
    return clips * avg_seconds * BYTES_PER_AUDIO_SECOND / 2**20


def summary() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {"free_mb": vm.available / 2**20, "total_mb": vm.total / 2**20,
            "reserve_mb": RESERVE_MB, "critical_mb": CRITICAL_MB, "gpu_free_mb": gpu_free_mb()}
