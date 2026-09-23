"""ASR base models.

Two fields matter more than size. `burmese` says whether the model was trained with Burmese at all
(only Whisper was), and `finetune` says what this app can actually train, not what is theoretically
possible. A model without Burmese can still be fine-tuned on Burmese audio in principle, but it has no
Burmese tokens to build on, so it needs far more data than adapting Whisper does.
"""
from typing import Any

MODELS: dict[str, dict[str, Any]] = {
    "whisper-large-v3-turbo": {
        "path": "openai/whisper-large-v3-turbo", "family": "whisper", "params": "809M", "via": "transformers",
        "label": "Whisper large-v3-turbo", "finetune": "lora+full", "vram_full_gb": 18, "vram_lora_gb": 8, "ram_cpu_gb": 6,
        "languages": 112, "burmese": True, "status": "ok",
        "note": "Multilingual Whisper with Burmese among its 112 languages. The best base for Burmese fine-tuning.",
        "url": "https://huggingface.co/openai/whisper-large-v3-turbo",
    },
    "whisper-small": {
        "path": "openai/whisper-small", "family": "whisper", "params": "244M", "via": "transformers",
        "label": "Whisper small", "finetune": "lora+full", "vram_full_gb": 8, "vram_lora_gb": 5, "ram_cpu_gb": 3,
        "languages": 112, "burmese": True, "status": "ok",
        "note": "Knows Burmese, trains on a modest GPU, and is a reasonable CPU-only experiment.",
        "url": "https://huggingface.co/openai/whisper-small",
    },
    "whisper-tiny": {
        "path": "openai/whisper-tiny", "family": "whisper", "params": "39M", "via": "transformers",
        "label": "Whisper tiny", "finetune": "lora+full", "vram_full_gb": 3, "vram_lora_gb": 2, "ram_cpu_gb": 1,
        "languages": 112, "burmese": True, "status": "ok",
        "note": "Knows Burmese in principle but is too small to do it well. Fine-tunes on CPU: use it to check the pipeline before spending GPU time.",
        "url": "https://huggingface.co/openai/whisper-tiny",
    },
    "qwen3-asr-1.7b": {
        "path": "Qwen/Qwen3-ASR-1.7B", "family": "qwen3_asr", "params": "1.7B", "via": "qwen-asr",
        "label": "Qwen3-ASR 1.7B", "finetune": "no", "vram_full_gb": 40, "vram_lora_gb": 14, "ram_cpu_gb": 10,
        "languages": 30, "burmese": False, "status": "needs_package",
        "note": "Strong on its 30 languages, but Burmese is not one of them. Its checkpoint also uses weight names "
                "plain transformers does not map (everything loads randomly initialised), so it needs Qwen's own "
                "`qwen-asr` package. Listed for completeness, not usable for Burmese here.",
        "url": "https://huggingface.co/Qwen/Qwen3-ASR-1.7B",
    },
    "vibevoice-asr-streaming-7b": {
        "path": "microsoft/VibeVoice-ASR-Streaming-7B", "family": "vibevoice", "params": "7B", "via": "transformers",
        "label": "VibeVoice ASR Streaming 7B", "finetune": "lora", "vram_full_gb": 120, "vram_lora_gb": 24, "ram_cpu_gb": 32,
        "languages": 10, "burmese": False, "status": "untested",
        "note": "Streaming ASR that also labels who spoke. 10 languages, Burmese not among them, and ~26 GB of weights: "
                "a large-GPU model. Not verified on this machine.",
        "url": "https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B",
    },
    "nemotron-3.5-asr-streaming-0.6b": {
        "path": "nvidia/nemotron-3.5-asr-streaming-0.6b", "family": "nemotron_rnnt", "params": "0.6B", "via": "pipeline",
        "label": "NVIDIA Nemotron 3.5 ASR streaming 0.6B", "finetune": "no", "vram_full_gb": 8, "vram_lora_gb": 3, "ram_cpu_gb": 3,
        "languages": 35, "burmese": False, "status": "ok",
        "note": "Cache-aware FastConformer RNN-Transducer, verified working here for transcription. Covers 35 language-locales "
                "but not Burmese, and RNNT fine-tuning needs NVIDIA's NeMo toolkit, so it is transcription only.",
        "url": "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b",
    },
}

DEFAULT_MODEL = "whisper-small"


def catalogue() -> list[dict[str, Any]]:
    return [{"id": k, **v} for k, v in MODELS.items()]


def can_transcribe(spec: dict[str, Any]) -> bool:
    return spec.get("status") not in ("needs_package",)


def resolve(model_id: str) -> dict[str, Any]:
    if model_id in MODELS:
        return {"id": model_id, **MODELS[model_id]}
    # any Hugging Face model id: assume Whisper-style unless the config says otherwise
    return {"id": model_id, "path": model_id, "family": "auto", "label": model_id, "params": "?", "via": "transformers",
            "finetune": "lora+full", "languages": None, "burmese": None, "status": "unknown",
            "note": "Custom Hugging Face model id.", "url": f"https://huggingface.co/{model_id}"}


def supports_training(spec: dict[str, Any]) -> bool:
    return spec.get("finetune") in ("lora", "lora+full", "full")
