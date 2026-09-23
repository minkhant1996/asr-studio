"""ASR base models. `finetune` says what this app can actually train, not what is theoretically possible."""
from typing import Any

MODELS: dict[str, dict[str, Any]] = {
    "whisper-large-v3-turbo": {
        "path": "openai/whisper-large-v3-turbo", "family": "whisper", "params": "809M",
        "label": "Whisper large-v3-turbo", "finetune": "lora+full", "vram_full_gb": 18, "vram_lora_gb": 8, "ram_cpu_gb": 6,
        "note": "Fast multilingual Whisper. Best accuracy/effort trade-off for Burmese fine-tuning.",
        "url": "https://huggingface.co/openai/whisper-large-v3-turbo",
    },
    "whisper-small": {
        "path": "openai/whisper-small", "family": "whisper", "params": "244M",
        "label": "Whisper small", "finetune": "lora+full", "vram_full_gb": 8, "vram_lora_gb": 5, "ram_cpu_gb": 3,
        "note": "Trains on a modest GPU; a reasonable CPU-only experiment.",
        "url": "https://huggingface.co/openai/whisper-small",
    },
    "whisper-tiny": {
        "path": "openai/whisper-tiny", "family": "whisper", "params": "39M",
        "label": "Whisper tiny", "finetune": "lora+full", "vram_full_gb": 3, "vram_lora_gb": 2, "ram_cpu_gb": 1,
        "note": "Small enough to fine-tune on CPU. Use it to check the pipeline before spending GPU time.",
        "url": "https://huggingface.co/openai/whisper-tiny",
    },
    "qwen3-asr-1.7b": {
        "path": "Qwen/Qwen3-ASR-1.7B", "family": "qwen3_asr", "params": "1.7B",
        "label": "Qwen3-ASR 1.7B", "finetune": "lora", "vram_full_gb": 40, "vram_lora_gb": 14, "ram_cpu_gb": 10,
        "note": "Audio encoder + LLM decoder. Fine-tuned here with LoRA on the decoder; full fine-tuning needs a large GPU.",
        "url": "https://huggingface.co/Qwen/Qwen3-ASR-1.7B",
    },
    "vibevoice-asr-streaming-7b": {
        "path": "microsoft/VibeVoice-ASR-Streaming-7B", "family": "vibevoice", "params": "7B",
        "label": "VibeVoice ASR Streaming 7B", "finetune": "lora", "vram_full_gb": 120, "vram_lora_gb": 24, "ram_cpu_gb": 32,
        "note": "Streaming ASR, ~15 GB of weights. LoRA only, and only on a large GPU.",
        "url": "https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B",
    },
    "nemotron-3.5-asr-streaming-0.6b": {
        "path": "nvidia/nemotron-3.5-asr-streaming-0.6b", "family": "nemotron_rnnt", "params": "0.6B",
        "label": "NVIDIA Nemotron 3.5 ASR streaming 0.6B", "finetune": "no", "vram_full_gb": 8, "vram_lora_gb": 0, "ram_cpu_gb": 3,
        "note": "RNN-Transducer trained with NVIDIA NeMo. Usable here for transcription and evaluation; RNNT fine-tuning needs the NeMo toolkit.",
        "url": "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b",
    },
}

DEFAULT_MODEL = "whisper-small"


def catalogue() -> list[dict[str, Any]]:
    return [{"id": k, **v} for k, v in MODELS.items()]


def resolve(model_id: str) -> dict[str, Any]:
    if model_id in MODELS:
        return {"id": model_id, **MODELS[model_id]}
    # any Hugging Face model id: assume Whisper-style unless the config says otherwise
    return {"id": model_id, "path": model_id, "family": "auto", "label": model_id, "params": "?",
            "finetune": "lora+full", "note": "Custom Hugging Face model id.", "url": f"https://huggingface.co/{model_id}"}


def supports_training(spec: dict[str, Any]) -> bool:
    return spec.get("finetune") in ("lora", "lora+full", "full")
