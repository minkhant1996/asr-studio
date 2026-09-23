"""Turn (possibly huge, streamed) source datasets into a small local training set.

Streams N examples per source, decodes and resamples audio to 16 kHz mono, filters by duration and
empty transcripts, then saves one Arrow dataset under data/cache/<id>/ with a manifest.
"""
import asyncio
import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np

from . import datasets_catalog as dc
from . import guards
from .config import settings

CACHE = Path(settings.cache_dir)
TARGET_SR = 16000
SHARD_ROWS = 500          # flush to disk this often: memory stays flat no matter how many clips


def _manifest_path(pid: str) -> Path:
    return CACHE / pid / "manifest.json"


def list_prepared() -> list[dict[str, Any]]:
    out = []
    if not CACHE.exists():
        return out
    for d in CACHE.iterdir():
        m = d / "manifest.json"
        if m.is_file():
            try:
                out.append(json.loads(m.read_text()))
            except Exception:
                continue
    return sorted(out, key=lambda x: -x.get("created", 0))


def get_prepared(pid: str) -> dict[str, Any]:
    p = _manifest_path(pid)
    if not p.exists():
        raise FileNotFoundError("prepared dataset not found")
    return json.loads(p.read_text())


def delete_prepared(pid: str) -> bool:
    d = CACHE / pid
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False


def load_prepared(pid: str):
    from datasets import load_from_disk

    return load_from_disk(str(CACHE / pid / "data"))


def _to_wav(arr: np.ndarray) -> bytes:
    """PCM16 WAV bytes: half the size of float32 and decodable without torchcodec."""
    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.clip(arr, -1.0, 1.0), TARGET_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def to_array(row: dict[str, Any]) -> np.ndarray:
    """Prepared row -> float32 mono array at 16 kHz."""
    import io

    import soundfile as sf

    arr, _ = sf.read(io.BytesIO(row["wav"]), dtype="float32", always_2d=False)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=1)
    return np.asarray(arr, dtype="float32")


def _resample(arr: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return arr.astype("float32")
    import librosa

    return librosa.resample(arr.astype("float32"), orig_sr=sr, target_sr=TARGET_SR)


def _decode(audio: dict[str, Any]) -> tuple[np.ndarray, int]:
    import io

    import librosa

    arr = audio.get("array")
    sr = int(audio.get("sampling_rate") or 0)
    if arr is None:
        raw, path = audio.get("bytes"), audio.get("path")
        if not raw and path:
            raw = dc.fetch_bytes(path)
        if raw:
            arr, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        else:
            raise ValueError("no audio payload")
    arr = np.asarray(arr, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr, sr or TARGET_SR


async def prepare(sources: list[dict[str, Any]], *, name: str = "", min_seconds: float = 0.4,
                  max_seconds: float = 30.0, test_fraction: float = 0.1, seed: int = 42) -> AsyncIterator[dict[str, Any]]:
    """Yield progress events; the final one is {"type": "done", "manifest": {...}}.

    Each source: {dataset_id | path, config?, data_dir?, split, take, text_field?}
    """
    pid = uuid.uuid4().hex[:12]
    out_dir = CACHE / pid
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []          # current shard only
    shards: list[str] = []
    shard_dir = out_dir / "shards"
    kept = 0
    stats = {"seconds": 0.0, "chars": 0}
    samples: list[dict[str, Any]] = []
    low_memory = False

    def _flush() -> None:
        """Write the buffered rows as an Arrow shard and drop them from memory."""
        nonlocal rows
        if not rows:
            return
        from datasets import Dataset

        shard_dir.mkdir(parents=True, exist_ok=True)
        path = shard_dir / f"{len(shards):05d}"
        Dataset.from_list(rows).save_to_disk(str(path))
        shards.append(str(path))
        rows = []

    per_source: list[dict[str, Any]] = []
    total_target = sum(int(s.get("take") or 0) for s in sources)
    t0 = time.perf_counter()
    skipped = 0

    for si, src in enumerate(sources):
        spec = dc.CATALOG.get(src.get("dataset_id") or "")
        path = spec.path if spec else src["path"]
        label = spec.name if spec else path
        take = int(src.get("take") or 100)
        split = src.get("split") or (spec.splits[0] if spec else "train")
        text_field = src.get("text_field") or (spec.text_field if spec else None)
        data_dir = src.get("data_dir") or (spec.data_dir if spec else None)
        config = src.get("config") or (spec.config if spec else None)

        if spec and not spec.transcripts:
            yield {"type": "error", "message": f"{label} ships audio without transcripts, so it cannot be used for supervised training."}
            return
        yield {"type": "status", "message": f"opening {label} ({split})", "source": si}
        try:
            it = await dc.stream(path, config=config, data_dir=data_dir, split=split)
        except Exception as e:
            yield {"type": "error", "message": f"{label}: {e}"}
            return

        got = 0
        seconds = 0.0
        while got < take:
            try:
                row = await asyncio.to_thread(next, it, None)
            except Exception as e:
                yield {"type": "status", "message": f"{label}: stream error after {got} clips ({e})"}
                break
            if row is None:
                break
            rec = dc.normalize(row, text_field)
            text = (rec["text"] or "").strip()
            if not text or rec["audio"] is None:
                skipped += 1
                continue
            try:
                arr, sr = await asyncio.to_thread(_decode, rec["audio"])
            except Exception:
                skipped += 1
                continue
            dur = len(arr) / sr if sr else 0.0
            if dur < min_seconds or dur > max_seconds:
                skipped += 1
                continue
            arr = await asyncio.to_thread(_resample, arr, sr)
            rows.append({"wav": _to_wav(arr), "sampling_rate": TARGET_SR, "text": text,
                         "duration": round(len(arr) / TARGET_SR, 3), "source": label})
            kept += 1
            got += 1
            seconds += len(arr) / TARGET_SR
            stats["seconds"] += len(arr) / TARGET_SR
            stats["chars"] += len(text)
            if len(samples) < 5:
                samples.append({"text": text, "duration": round(len(arr) / TARGET_SR, 3), "source": label})
            if len(rows) >= SHARD_ROWS:
                await asyncio.to_thread(_flush)
            if guards.critical():
                await asyncio.to_thread(_flush)
                if guards.critical():
                    low_memory = True
                    yield {"type": "status", "message": f"stopping early: only {guards.free_mb():.0f} MB of RAM left "
                                                        f"({kept} clips kept and saved)"}
                    break
            if got % 5 == 0 or got == take:
                done = kept
                el = time.perf_counter() - t0
                yield {"type": "progress", "i": done, "n": total_target, "source": label, "source_index": si,
                       "clips": got, "seconds": round(seconds, 1), "elapsed": round(el, 1),
                       "eta": round(el / max(done, 1) * max(total_target - done, 0), 1), "skipped": skipped,
                       "free_mb": round(guards.free_mb())}
        per_source.append({"label": label, "path": path, "split": split, "clips": got, "seconds": round(seconds, 1)})
        if low_memory:
            break

    await asyncio.to_thread(_flush)
    if not shards:
        yield {"type": "error", "message": "no usable clips found (check the split, or raise the duration limits)"}
        return

    yield {"type": "status", "message": f"assembling {kept} clips from {len(shards)} shard(s)"}

    def _build():
        from datasets import concatenate_datasets, load_from_disk

        # Arrow shards are memory-mapped, so this concatenation does not load the audio into RAM.
        parts_ds = [load_from_disk(sp) for sp in shards]
        ds = (concatenate_datasets(parts_ds) if len(parts_ds) > 1 else parts_ds[0]).shuffle(seed=seed)
        if test_fraction > 0 and len(ds) >= 10:
            parts = ds.train_test_split(test_size=test_fraction, seed=seed)
        else:
            parts = {"train": ds, "test": ds.select(range(0))}
        from datasets import DatasetDict

        DatasetDict({"train": parts["train"], "test": parts["test"]}).save_to_disk(str(out_dir / "data"))
        n = len(parts["train"]), len(parts["test"])
        del parts_ds, ds, parts
        shutil.rmtree(shard_dir, ignore_errors=True)
        return n

    n_train, n_test = await asyncio.to_thread(_build)
    total_seconds = stats["seconds"]
    manifest = {
        "id": pid, "name": name or " + ".join(s["label"] for s in per_source),
        "created": time.time(), "clips": kept, "train": n_train, "test": n_test, "low_memory_stop": low_memory,
        "seconds": round(total_seconds, 1), "hours": round(total_seconds / 3600, 3),
        "sampling_rate": TARGET_SR, "sources": per_source, "skipped": skipped,
        "min_seconds": min_seconds, "max_seconds": max_seconds,
        "chars": stats["chars"],
        "avg_duration": round(total_seconds / kept, 2) if kept else 0,
        "samples": samples,
    }
    _manifest_path(pid).write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    yield {"type": "done", "manifest": manifest}
