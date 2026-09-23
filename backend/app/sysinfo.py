"""RAM / VRAM / CPU snapshot (NVIDIA, Apple Metal, or CPU-only)."""
import os
import platform
import shutil
import subprocess
from typing import Any

import psutil


def device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _gpus() -> list[dict[str, Any]]:
    try:
        import torch

        if torch.cuda.is_available():
            out = []
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                out.append({"index": i, "name": torch.cuda.get_device_name(i), "total_mb": total / 2**20,
                            "used_mb": (total - free) / 2**20, "free_mb": free / 2**20})
            return out
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            total = psutil.virtual_memory().total / 2**20
            used = torch.mps.driver_allocated_memory() / 2**20 if hasattr(torch.mps, "driver_allocated_memory") else 0.0
            return [{"index": 0, "name": "Apple GPU (Metal, unified memory)", "total_mb": total, "used_mb": used, "free_mb": total - used}]
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=3)
            out = []
            for line in r.stdout.strip().splitlines():
                idx, name, total, used = [x.strip() for x in line.split(",")]
                out.append({"index": int(idx), "name": name, "total_mb": float(total), "used_mb": float(used),
                            "free_mb": float(total) - float(used)})
            return out
        except Exception:
            pass
    return []


def quick_mem() -> dict[str, float | None]:
    rss = psutil.Process(os.getpid()).memory_info().rss / 2**20
    vram = None
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            vram = (total - free) / 2**20
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            vram = torch.mps.driver_allocated_memory() / 2**20
    except Exception:
        pass
    return {"rss_mb": rss, "vram_mb": vram}


def snapshot() -> dict[str, Any]:
    proc = psutil.Process(os.getpid())
    vm = psutil.virtual_memory()
    gpus = _gpus()
    dev = device()
    note = None
    if dev == "cpu" and gpus:
        note = "A GPU is present but PyTorch cannot use it (driver / CUDA build mismatch): training and transcription run on CPU."
    elif dev == "cpu":
        note = "No GPU detected: fine-tuning runs on CPU, which is slow. Use Whisper tiny/small and few steps, or run this app on a GPU machine."
    return {
        "os": f"{platform.system()} {platform.release()}",
        "device": dev,
        "cpu_percent": psutil.cpu_percent(interval=None),
        "cpu_count": psutil.cpu_count(logical=True),
        "process": {"rss_mb": proc.memory_info().rss / 2**20},
        "ram": {"total_mb": vm.total / 2**20, "used_mb": (vm.total - vm.available) / 2**20,
                "free_mb": vm.available / 2**20, "percent": vm.percent},
        "gpus": gpus,
        "note": note,
    }
