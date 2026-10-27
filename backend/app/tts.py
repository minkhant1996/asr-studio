"""Text-to-speech: the models here run the other way, text in and audio out.

They live outside models_catalog on purpose. That catalogue is what Fine-tune, Evaluate and Transcribe
offer as ASR targets, and a TTS model cannot transcribe anything.

Two engines, picked per model by `engine`:

* `voxcpm`  - the `voxcpm` package, installed with --no-deps (see setup.sh): its metadata pins
  `datasets<4` and pulls in gradio, funasr and modelscope, none of which synthesis needs, and the pin
  would downgrade the datasets library the rest of the app depends on. Synthesis only needs `einops`
  on top of what is already here.
* `vibevoice` - native transformers (5.17 added the architecture), so no extra package beyond
  `diffusers`, which supplies the noise scheduler the diffusion head needs.

Only one model is held in memory at a time: a second set of weights would not fit beside the first on
a 24 GB card, so loading one unloads the other.
"""
import asyncio
import gc
import importlib.util
import io
import os
import re
import tempfile
import threading
import time
from typing import Any

from . import guards, sysinfo
from .config import get_hf_token

VOXCPM_HINT = "pip install --no-deps voxcpm && pip install einops"
VIBEVOICE_HINT = "pip install diffusers"

MODELS: dict[str, dict[str, Any]] = {
    "voxcpm2": {
        "id": "voxcpm2", "path": "openbmb/VoxCPM2", "label": "VoxCPM2", "params": "2B", "engine": "voxcpm",
        "languages": 30, "burmese": True, "sample_rate": 48000, "license": "Apache-2.0", "need_mb": 8000,
        "clone": True, "design": True, "advanced": True, "weights_gb": 4.7,
        "note": "Tokenizer-free diffusion-autoregressive TTS. Burmese and Thai are among its 30 languages; "
                "no language tag is needed. About 4.7 GB of weights, downloaded on first use.",
        "url": "https://huggingface.co/openbmb/VoxCPM2",
    },
    "vibevoice-1.5b": {
        "id": "vibevoice-1.5b", "path": "vibevoice/VibeVoice-1.5B-hf", "label": "VibeVoice 1.5B", "params": "1.5B",
        "engine": "vibevoice", "languages": 2, "burmese": False, "sample_rate": 24000, "license": "MIT",
        "need_mb": 7000, "clone": True, "design": False, "advanced": False, "weights_gb": 5.4,
        "note": "Microsoft's long-form conversational TTS, in the transformers-native conversion. English and "
                "Chinese only, so it cannot read Burmese: it is here as an English voice and a cloning "
                "comparison against VoxCPM2. Clones from a reference clip; it has no voice-description mode.",
        "url": "https://huggingface.co/vibevoice/VibeVoice-1.5B-hf",
    },
    # Listed, not loadable. Its checkpoint declares model_type `vibevoice_streaming`, which transformers
    # does not know, and Microsoft's standalone `vibevoice` package cannot supply it here: that package
    # pins transformers<5 and its config classes collide with the ones transformers 5.x already registers
    # ("... is already used by a Transformers model"). Running it needs its own environment.
    "vibevoice-realtime-0.5b": {
        "id": "vibevoice-realtime-0.5b", "path": "microsoft/VibeVoice-Realtime-0.5B", "label": "VibeVoice Realtime 0.5B",
        "params": "0.5B", "engine": "vibevoice_streaming", "languages": 1, "burmese": False, "sample_rate": 24000,
        "license": "MIT", "need_mb": 5000, "clone": False, "design": False, "advanced": False, "weights_gb": 2.0,
        "status": "unsupported",
        "note": "Streaming English TTS, ~200 ms to first audio. Not loadable here: its `vibevoice_streaming` "
                "architecture is not in transformers, and Microsoft's package that implements it pins "
                "transformers<5 and collides with the VibeVoice classes transformers already registers. "
                "It also cannot clone a voice - its speakers ship as fixed embeddings. Run it in its own "
                "environment if you want it.",
        "url": "https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B",
    },
}

DEFAULT_MODEL = "voxcpm2"
MAX_TEXT = 1000
MAX_REFERENCE_SECONDS = 30

# back-compat: the Speak tab used to be VoxCPM2 only
MODEL = MODELS[DEFAULT_MODEL]
NEED_MB = MODEL["need_mb"]
INSTALL_HINT = VOXCPM_HINT

_loaded: dict[str, Any] = {}
# one generation at a time: VoxCPM2 keeps a single key-value cache between steps, and two models do
# not fit in memory together
_lock = threading.Lock()


def resolve(model_id: str | None) -> dict[str, Any]:
    spec = MODELS.get(model_id or DEFAULT_MODEL)
    if spec is None:
        raise KeyError(f"unknown speech model {model_id!r}")
    return spec


def install_hint(spec: dict[str, Any]) -> str:
    return VOXCPM_HINT if spec["engine"] == "voxcpm" else VIBEVOICE_HINT


def installed(model_id: str | None = None) -> bool:
    """Whether this model's engine can run in the backend environment."""
    spec = resolve(model_id)
    engine = spec["engine"]
    if engine == "voxcpm":
        return importlib.util.find_spec("voxcpm") is not None and importlib.util.find_spec("einops") is not None
    if engine == "vibevoice":
        return importlib.util.find_spec("diffusers") is not None
    return False  # vibevoice_streaming: no in-process engine, see the note on that entry


def _resident_mb() -> int:
    """Memory the currently loaded model would release, since loading another unloads it."""
    return sum(MODELS[m]["need_mb"] for m in _loaded if m in MODELS)


def _status_one(spec: dict[str, Any], device: str, free: int | None, reserve: int) -> dict[str, Any]:
    need = spec["need_mb"]
    loaded = spec["id"] in _loaded
    return {
        **spec,
        "status": spec.get("status", "ok"),
        "installed": installed(spec["id"]),
        "install_hint": install_hint(spec),
        "loaded": loaded,
        "device": device,
        "need_mb": need,
        "free_mb": free,
        # a loaded model already fits; otherwise count what the currently loaded one would give back
        "fits": bool(loaded or (free is not None and need + reserve < free + _resident_mb())),
        "max_text": MAX_TEXT,
        "max_reference_seconds": MAX_REFERENCE_SECONDS,
    }


def status(model_id: str | None = None) -> dict[str, Any]:
    """Status of one model, plus the catalogue, so the tab can offer a choice."""
    device = sysinfo.device()
    free = guards.gpu_free_mb() if device == "cuda" else guards.free_mb()
    reserve = 0 if device == "cuda" else guards.RESERVE_MB
    models = [_status_one(s, device, free, reserve) for s in MODELS.values()]
    chosen = resolve(model_id)
    return {**_status_one(chosen, device, free, reserve), "models": models, "default": DEFAULT_MODEL}


def _unload_others(keep: str) -> None:
    for mid in list(_loaded):
        if mid != keep:
            _loaded.pop(mid, None)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load(spec: dict[str, Any]):
    mid = spec["id"]
    if mid in _loaded:
        return _loaded[mid]
    if not installed(mid):
        raise RuntimeError(f"{spec['label']} needs a package that is not installed. Run: {install_hint(spec)}")
    _unload_others(mid)
    guards.check(spec["need_mb"], f"{spec['label']} (speech synthesis)", sysinfo.device())

    from huggingface_hub import snapshot_download

    token = get_hf_token() or None
    if spec["engine"] == "voxcpm":
        from voxcpm import VoxCPM

        path = snapshot_download(spec["path"], token=token)
        # No denoiser: it needs modelscope. No torch.compile: the warm-up takes minutes and needs a C
        # compiler, which is a poor trade for a tab used a sentence at a time.
        _loaded[mid] = VoxCPM.from_pretrained(path, load_denoiser=False, optimize=False)
    else:
        import torch
        from transformers import AutoModelForTextToWaveform, AutoProcessor

        device = sysinfo.device()
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(spec["path"], token=token)
        model = AutoModelForTextToWaveform.from_pretrained(spec["path"], dtype=dtype, token=token)
        model.to(device).eval()
        _loaded[mid] = (processor, model)
    return _loaded[mid]


# ---------------------------------------------------------------- engines


def _speak_voxcpm(model, text: str, voice: str, ref_path: str | None, prompt_text: str,
                  cfg: float, steps: int) -> tuple[Any, int, str]:
    kwargs: dict[str, Any] = {}
    if ref_path:
        kwargs["reference_wav_path"] = ref_path
        if prompt_text:
            # reference plus its transcript: the model continues the clip, and a voice description
            # would be read as part of the text, so it is dropped
            kwargs.update(prompt_wav_path=ref_path, prompt_text=prompt_text)
            voice = ""
    wav = model.generate(text=f"({voice}){text}" if voice else text, cfg_value=cfg,
                         inference_timesteps=steps, **kwargs)
    mode = "clone+transcript" if "prompt_text" in kwargs else "clone" if ref_path else "design" if voice else "default"
    return wav, int(model.tts_model.sample_rate), mode


def _speak_vibevoice(bundle, text: str, ref_path: str | None) -> tuple[Any, int, str]:
    import numpy as np
    import torch

    processor, model = bundle
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if ref_path:
        content.append({"type": "audio", "url": ref_path})
    conversation = [{"role": "0", "content": content}]
    inputs = processor.apply_chat_template(conversation, return_dict=True, tokenize=True,
                                           add_generation_prompt=True).to(model.device, model.dtype)
    with torch.no_grad():
        audio = model.generate(**inputs)
    wav = audio[0] if isinstance(audio, (list, tuple)) else audio
    if hasattr(wav, "detach"):
        wav = wav.detach().to(torch.float32).cpu().numpy()
    wav = np.asarray(wav, dtype="float32").squeeze()
    sr = int(getattr(processor.feature_extractor, "sampling_rate", 24000))
    return wav, sr, "clone" if ref_path else "default"


def _synthesize_sync(model_id: str | None, text: str, voice: str, reference, prompt_text: str,
                     cfg: float, steps: int) -> dict[str, Any]:
    import soundfile as sf

    spec = resolve(model_id)
    # "(description)text" is VoxCPM2's own prompt format for designing or steering a voice
    voice = re.sub(r"[()（）]", "", voice or "").strip()
    prompt_text = (prompt_text or "").strip()
    if not spec["design"]:
        voice = ""
    if not spec["clone"]:
        reference = None

    ref_path = None
    with _lock:
        engine = _load(spec)
        try:
            if reference is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    ref_path = tmp.name
                sf.write(ref_path, reference, 16000)
            t0 = time.perf_counter()
            if spec["engine"] == "voxcpm":
                wav, sr, mode = _speak_voxcpm(engine, text, voice, ref_path, prompt_text, cfg, steps)
            else:
                wav, sr, mode = _speak_vibevoice(engine, text, ref_path)
            took = time.perf_counter() - t0
        finally:
            if ref_path and os.path.exists(ref_path):
                os.unlink(ref_path)

    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    dur = len(wav) / sr
    return {"wav": buf.getvalue(), "sampling_rate": sr, "duration": round(dur, 2), "ms": round(took * 1000, 1),
            "rtf": round(took / dur, 3) if dur else None, "mode": mode, "model": spec["label"],
            "model_id": spec["id"], "device": sysinfo.device()}


async def synthesize(text: str, voice: str = "", reference=None, prompt_text: str = "",
                     cfg: float = 2.0, steps: int = 10, model_id: str | None = None) -> dict[str, Any]:
    """reference: optional 16 kHz mono float array of the voice to clone."""
    return await asyncio.to_thread(_synthesize_sync, model_id, text, voice, reference, prompt_text, cfg, steps)
