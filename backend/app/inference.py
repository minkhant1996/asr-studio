"""Loading models for transcription / evaluation: base models from the catalogue, any Hugging Face
id, or a fine-tuned run (full weights or a LoRA adapter)."""
import asyncio
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import guards, models_catalog, sysinfo, training
from .config import get_hf_token

_load_lock = threading.Lock()


def resolve_target(target: dict[str, Any]) -> dict[str, Any]:
    """target: {"kind": "base"|"run", "model"|"run_id": ...} -> {path, label, base_path, adapter}"""
    kind = target.get("kind", "base")
    if kind == "run":
        run = training.get_run(target["run_id"])
        model_dir = Path(run.get("model_dir") or (training.run_dir(run["id"]) / "model"))
        base = models_catalog.resolve(run["model"])
        is_adapter = (model_dir / "adapter_config.json").exists()
        return {"path": str(model_dir), "label": f"{run['name']} (fine-tuned)", "base_path": base["path"],
                "adapter": is_adapter, "language": run.get("config", {}).get("language", "burmese"), "run_id": run["id"]}
    spec = models_catalog.resolve(target.get("model") or models_catalog.DEFAULT_MODEL)
    return {"path": spec["path"], "label": spec["label"], "base_path": spec["path"], "adapter": False,
            "language": target.get("language", "burmese"), "family": spec.get("family")}


@lru_cache(maxsize=2)
def _load(path: str, base_path: str, adapter: bool, language: str):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device = sysinfo.device()
    token = get_hf_token() or None
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        processor = AutoProcessor.from_pretrained(path, language=language, task="transcribe", token=token)
    except Exception:
        processor = AutoProcessor.from_pretrained(base_path, language=language, task="transcribe", token=token)
    if adapter:
        from peft import PeftModel

        model = AutoModelForSpeechSeq2Seq.from_pretrained(base_path, dtype=dtype, token=token)
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(path, dtype=dtype, token=token)
    # Stale forced_decoder_ids in a checkpoint's generation config silently override the language
    # flag, which makes multilingual Whisper transcribe Burmese as something else.
    gc = getattr(model, "generation_config", None)
    if gc is not None:
        try:
            gc.forced_decoder_ids = None
            if language:
                gc.language = language
                gc.task = "transcribe"
        except Exception:
            pass
    model.to(device).eval()
    return model, processor, device


def load_target(target: dict[str, Any]):
    t = resolve_target(target)
    spec = models_catalog.resolve(target.get("model") or "") if target.get("kind") != "run" else {}
    from .training import _params_of

    params = _params_of({**spec, "id": target.get("model")}) if spec else 800e6
    device = sysinfo.device()
    if _load.cache_info().currsize == 0:   # nothing cached yet: this load really allocates
        guards.check(guards.model_need_mb(params, "inference", device), f"{t['label']} (transcription)", device)
    with _load_lock:
        return _load(t["path"], t["base_path"], t["adapter"], t.get("language") or "burmese"), t


def _transcribe_sync(model, processor, device, arrays: list, language: str, max_new_tokens: int = 200) -> list[str]:
    import torch

    fe, tok = processor.feature_extractor, processor.tokenizer
    out: list[str] = []
    for arr in arrays:
        feats = fe(arr, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        if device == "cuda":
            feats = feats.half()
        try:
            with torch.no_grad():
                ids = model.generate(feats, max_new_tokens=max_new_tokens, language=language, task="transcribe")
        except Exception:
            with torch.no_grad():
                ids = model.generate(feats, max_new_tokens=max_new_tokens)
        out.append(tok.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return out


async def transcribe_arrays(target: dict[str, Any], arrays: list, language: str = "burmese") -> tuple[list[str], dict[str, Any]]:
    (model, processor, device), t = await asyncio.to_thread(load_target, target)
    texts = await asyncio.to_thread(_transcribe_sync, model, processor, device, arrays, language)
    return texts, {**t, "device": device}
