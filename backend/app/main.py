import asyncio
import io
import json
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from . import datasets_catalog as dc
from . import evaluation, guards, inference, models_catalog, prepare, secrets_store, sysinfo, training
from .config import get_hf_token, settings

app = FastAPI(title="ASR Studio API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5175", "http://127.0.0.1:5175"],
                   allow_methods=["*"], allow_headers=["*"])

MAX_UPLOAD = 200 * 1024 * 1024


def ndjson(gen):
    async def stream():
        try:
            async for ev in gen:
                yield json.dumps(ev, ensure_ascii=False, default=str) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": f"{type(e).__name__}: {e}"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.on_event("startup")
async def _startup() -> None:
    n = training.sweep_stale_runs()
    if n:
        print(f"[asr-studio] marked {n} interrupted run(s) from a previous process")


@app.get("/api/health")
async def health():
    return {"ok": True, "device": sysinfo.device(), "hf_token_set": bool(get_hf_token())}


@app.get("/api/system")
async def system():
    snap = await asyncio.to_thread(sysinfo.snapshot)
    return {**snap, "guards": guards.summary()}


@app.post("/api/estimate")
async def estimate(body: dict[str, Any]):
    """What a planned job would need, so the UI can warn before anything is loaded."""
    kind = body.get("kind", "prepare")
    device = sysinfo.device()
    if kind == "prepare":
        clips = int(body.get("clips") or 0)
        need = 200 + prepare.SHARD_ROWS * 6 * guards.BYTES_PER_AUDIO_SECOND / 2**20
        disk = guards.estimate_prepare_mb(clips)
        return {"kind": kind, "need_mb": need, "disk_mb": disk, "free_mb": guards.free_mb(),
                "ok": need + guards.RESERVE_MB < guards.free_mb(),
                "note": f"Clips are written to disk every {prepare.SHARD_ROWS}, so memory stays flat; "
                        f"{clips:,} clips need roughly {disk / 1024:.1f} GB of disk."}
    spec = models_catalog.resolve(body.get("model") or models_catalog.DEFAULT_MODEL)
    from .training import _params_of

    need = guards.model_need_mb(_params_of(spec), body.get("method", "lora"), device)
    free = guards.gpu_free_mb() if device == "cuda" else guards.free_mb()
    return {"kind": kind, "need_mb": need, "free_mb": free, "device": device,
            "ok": bool(free is not None and need + (0 if device == "cuda" else guards.RESERVE_MB) < free)}


# ------------------------------------------------------------------ datasets
@app.get("/api/datasets")
async def datasets():
    return dc.catalogue()


class PreviewRequest(BaseModel):
    dataset_id: str | None = None
    path: str | None = None
    config: str | None = None
    data_dir: str | None = None
    split: str = "train"
    text_field: str | None = None
    limit: int = Field(default=3, ge=1, le=10)


@app.post("/api/datasets/preview")
async def preview(req: PreviewRequest):
    """Stream a few clips from a catalogue entry or any Hugging Face dataset, with playable audio."""
    spec = dc.CATALOG.get(req.dataset_id or "")
    if spec:
        path, config, data_dir = spec.path, spec.config, spec.data_dir
        split = req.split if req.split in spec.splits else spec.splits[0]
        text_field = spec.text_field
    else:
        if not req.path:
            raise HTTPException(400, "dataset_id or path required")
        path, config = dc.parse_hf_ref(req.path)
        config = req.config or config
        data_dir, split, text_field = req.data_dir, req.split, req.text_field
    try:
        it = await dc.stream(path, config=config, data_dir=data_dir, split=split)
    except Exception as e:
        raise HTTPException(502, f"could not open dataset: {e}")

    import base64

    items, columns = [], []
    for _ in range(req.limit):
        row = await asyncio.to_thread(next, it, None)
        if row is None:
            break
        if not columns:
            columns = list(row.keys())
        rec = dc.normalize(row, text_field)
        entry: dict[str, Any] = {"text": rec["text"], "duration": rec["duration"]}
        if rec["audio"]:
            try:
                wav, sr, dur = await asyncio.to_thread(dc.audio_to_wav_bytes, rec["audio"], 30)
                entry.update({"audio_b64": base64.b64encode(wav).decode(), "sampling_rate": sr,
                              "duration": round(entry["duration"] or dur, 2)})
            except Exception as e:
                entry["audio_error"] = str(e)
        items.append(entry)
    return {"path": path, "split": split, "columns": columns, "items": items,
            "text_field": text_field, "data_dir": data_dir,
            "transcripts": bool(spec.transcripts) if spec else any(i.get("text") for i in items)}


class PrepareRequest(BaseModel):
    sources: list[dict[str, Any]] = Field(min_length=1, max_length=10)
    name: str = ""
    min_seconds: float = Field(default=0.4, ge=0.1, le=60)
    max_seconds: float = Field(default=30.0, ge=1, le=60)
    test_fraction: float = Field(default=0.1, ge=0, lt=0.9)
    seed: int = 42


@app.post("/api/prepare/stream")
async def prepare_stream(req: PrepareRequest):
    return ndjson(prepare.prepare(req.sources, name=req.name, min_seconds=req.min_seconds,
                                  max_seconds=req.max_seconds, test_fraction=req.test_fraction, seed=req.seed))


@app.get("/api/prepared")
async def prepared_list():
    return prepare.list_prepared()


@app.get("/api/prepared/{pid}")
async def prepared_get(pid: str):
    try:
        return prepare.get_prepared(pid)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.delete("/api/prepared/{pid}")
async def prepared_delete(pid: str):
    return {"ok": prepare.delete_prepared(pid)}


@app.get("/api/prepared/{pid}/clip/{split}/{index}")
async def prepared_clip(pid: str, split: str, index: int):
    """WAV bytes for one clip of a prepared dataset, for playback in the browser."""
    try:
        dsd = await asyncio.to_thread(prepare.load_prepared, pid)
    except Exception as e:
        raise HTTPException(404, str(e))
    ds = dsd[split] if split in dsd else dsd["train"]
    if index < 0 or index >= len(ds):
        raise HTTPException(404, "clip out of range")
    row = ds[index]
    return Response(content=row["wav"], media_type="audio/wav", headers={"Cache-Control": "no-store"})


@app.get("/api/prepared/{pid}/rows")
async def prepared_rows(pid: str, split: str = "train", offset: int = 0, limit: int = 25):
    try:
        dsd = await asyncio.to_thread(prepare.load_prepared, pid)
    except Exception as e:
        raise HTTPException(404, str(e))
    ds = dsd[split] if split in dsd else dsd["train"]
    end = min(offset + limit, len(ds))
    rows = [{"index": i, "text": ds[i]["text"], "duration": ds[i]["duration"], "source": ds[i].get("source")}
            for i in range(offset, end)]
    return {"split": split, "total": len(ds), "rows": rows}


# ------------------------------------------------------------------ models & training
@app.get("/api/models")
async def models():
    snap = await asyncio.to_thread(sysinfo.snapshot)
    gpu = snap["gpus"][0]["total_mb"] / 1024 if snap["gpus"] else 0
    free_ram = snap["ram"]["free_mb"] / 1024
    out = []
    for m in models_catalog.catalogue():
        can_lora = (gpu >= m.get("vram_lora_gb", 0)) if gpu else (free_ram >= m.get("ram_cpu_gb", 0))
        can_full = (gpu >= m.get("vram_full_gb", 0)) if gpu else (free_ram >= m.get("ram_cpu_gb", 0) * 2)
        out.append({**m, "can_train": models_catalog.supports_training(m),
                    "fits_lora": bool(can_lora), "fits_full": bool(can_full)})
    return {"models": out, "device": snap["device"], "gpu_gb": round(gpu, 1), "free_ram_gb": round(free_ram, 1)}


class TrainRequest(BaseModel):
    dataset_id: str
    model: str = models_catalog.DEFAULT_MODEL
    language: str = "burmese"
    method: Literal["lora", "full"] = "lora"
    epochs: float = Field(default=1.0, gt=0, le=50)
    max_steps: int = Field(default=0, ge=0, le=100000)
    batch_size: int = Field(default=2, ge=1, le=64)
    grad_accum: int = Field(default=4, ge=1, le=64)
    learning_rate: float = Field(default=1e-5, gt=0, le=1e-2)
    warmup_steps: int = Field(default=10, ge=0, le=5000)
    eval_steps: int = Field(default=0, ge=0, le=10000)
    eval_clips: int = Field(default=8, ge=1, le=64)
    fp16: bool = True
    freeze_encoder: bool = False
    lora_r: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1, le=512)
    seed: int = 42
    name: str = ""


@app.post("/api/train/stream")
async def train_stream(req: TrainRequest):
    try:
        prepare.get_prepared(req.dataset_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    spec = models_catalog.resolve(req.model)
    from .training import _params_of

    try:
        guards.check(guards.model_need_mb(_params_of(spec), req.method, sysinfo.device()),
                     f"{spec['label']} ({req.method} fine-tuning)", sysinfo.device())
    except guards.NotEnoughMemory as e:
        raise HTTPException(507, str(e))
    cfg = training.TrainConfig(**req.model_dump())
    return ndjson(training.train(cfg))


@app.get("/api/runs")
async def runs():
    return training.list_runs()


@app.get("/api/runs/{run_id}")
async def run_get(run_id: str):
    try:
        return training.get_run(run_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.post("/api/runs/{run_id}/stop")
async def run_stop(run_id: str):
    return {"ok": training.stop_run(run_id)}


@app.delete("/api/runs/{run_id}")
async def run_delete(run_id: str):
    return {"ok": training.delete_run(run_id)}


# ------------------------------------------------------------------ evaluation
class EvaluateRequest(BaseModel):
    dataset_id: str
    targets: list[dict[str, Any]] = Field(min_length=1, max_length=4)
    split: str = "test"
    limit: int = Field(default=25, ge=1, le=2000)
    language: str = "burmese"
    title: str = ""


@app.post("/api/evaluate/stream")
async def evaluate_stream(req: EvaluateRequest):
    return ndjson(evaluation.evaluate(req.dataset_id, req.targets, split=req.split, limit=req.limit,
                                      language=req.language, title=req.title))


@app.get("/api/evals")
async def evals():
    return evaluation.list_evals()


@app.get("/api/evals/{eid}")
async def eval_get(eid: str):
    try:
        return evaluation.get_eval(eid)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.delete("/api/evals/{eid}")
async def eval_delete(eid: str):
    return {"ok": evaluation.delete_eval(eid)}


# ------------------------------------------------------------------ transcribe
@app.post("/api/transcribe")
async def transcribe(file: UploadFile, target: str = "{}", language: str = "burmese"):
    """Transcribe an uploaded audio file with a base model or a fine-tuned run."""
    raw = await file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "file larger than 200 MB")
    try:
        tgt = json.loads(target) or {"kind": "base", "model": models_catalog.DEFAULT_MODEL}
    except Exception:
        raise HTTPException(400, "target must be JSON")

    import librosa

    def _decode():
        arr, sr = librosa.load(io.BytesIO(raw), sr=16000, mono=True)
        return arr, sr

    try:
        arr, _ = await asyncio.to_thread(_decode)
    except Exception as e:
        raise HTTPException(400, f"could not decode audio: {e}")
    if len(arr) == 0:
        raise HTTPException(400, "empty audio")

    chunk = 30 * 16000
    chunks = [arr[i:i + chunk] for i in range(0, len(arr), chunk)] or [arr]
    import time as _t

    t0 = _t.perf_counter()
    try:
        texts, info = await inference.transcribe_arrays(tgt, chunks, language)
    except guards.NotEnoughMemory as e:
        raise HTTPException(507, str(e))
    except Exception as e:
        raise HTTPException(500, f"transcription failed: {e}")
    took = (_t.perf_counter() - t0) * 1000
    dur = len(arr) / 16000
    return {"text": " ".join(t for t in texts if t).strip(), "chunks": texts, "duration": round(dur, 2),
            "ms": round(took, 1), "rtf": round((took / 1000) / dur, 3) if dur else None,
            "model": info["label"], "device": info["device"], "filename": file.filename}


# ------------------------------------------------------------------ settings
@app.get("/api/settings")
async def get_settings():
    tok = get_hf_token()
    return {"hf_token_set": bool(tok), "hf_token_masked": secrets_store.mask(tok) if tok else None,
            "prefs": secrets_store.get_prefs(), "cache_dir": settings.cache_dir, "runs_dir": settings.runs_dir}


class TokenUpdate(BaseModel):
    token: str = Field(min_length=8, max_length=200, pattern=r"^[A-Za-z0-9_\-\.]+$")


@app.put("/api/settings/hf")
async def set_hf_token(body: TokenUpdate):
    """Verify the token with the Hub, then store it encrypted."""
    from huggingface_hub import HfApi

    try:
        who = await asyncio.to_thread(lambda: HfApi().whoami(token=body.token))
    except Exception as e:
        raise HTTPException(400, f"Hugging Face rejected this token: {e}")
    secrets_store.set_secret("hf_token", body.token)
    return {"ok": True, "user": who.get("name"), "masked": secrets_store.mask(body.token)}


@app.delete("/api/settings/hf")
async def delete_hf_token():
    secrets_store.delete_secret("hf_token")
    return {"ok": True}


class PrefsUpdate(BaseModel):
    prefs: dict[str, Any] = {}


@app.put("/api/settings/prefs")
async def set_prefs(body: PrefsUpdate):
    return {"ok": True, "prefs": secrets_store.set_prefs(**body.prefs)}
