"""WER / CER evaluation of a base model or a fine-tuned run, on a prepared test set.

For Burmese, word error rate is unreliable because the script does not put spaces between words the
way Latin scripts do, so the UI leads with CER and reports WER alongside it.
"""
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from . import inference, prepare, sysinfo
from .config import DATA

EVALS = DATA / "evals"


def _norm(s: str) -> str:
    import re
    import unicodedata

    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"[၊။,.!?;:\"'“”‘’()\[\]{}]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def metrics(refs: list[str], hyps: list[str]) -> dict[str, float]:
    import jiwer

    r = [_norm(x) for x in refs]
    h = [_norm(x) for x in hyps]
    pairs = [(a, b) for a, b in zip(r, h) if a]
    if not pairs:
        return {"wer": 0.0, "cer": 0.0, "n": 0}
    r2, h2 = [a for a, _ in pairs], [b for _, b in pairs]
    out = {"wer": float(jiwer.wer(r2, h2)), "cer": float(jiwer.cer(r2, h2)), "n": len(pairs)}
    try:
        cm = jiwer.process_characters(r2, h2)
        out.update({"cer_sub": cm.substitutions, "cer_del": cm.deletions, "cer_ins": cm.insertions,
                    "ref_chars": sum(len(x) for x in r2)})
    except Exception:
        pass
    return out


def list_evals() -> list[dict[str, Any]]:
    if not EVALS.exists():
        return []
    out = []
    for p in EVALS.glob("*.json"):
        try:
            d = json.loads(p.read_text())
            out.append({k: d.get(k) for k in ("id", "created", "title", "dataset_name", "targets", "summary")})
        except Exception:
            continue
    return sorted(out, key=lambda e: -(e.get("created") or 0))


def get_eval(eid: str) -> dict[str, Any]:
    p = EVALS / f"{eid}.json"
    if not p.exists():
        raise FileNotFoundError("evaluation not found")
    return json.loads(p.read_text())


def delete_eval(eid: str) -> bool:
    p = EVALS / f"{eid}.json"
    if p.exists():
        p.unlink()
        return True
    return False


async def evaluate(dataset_id: str, targets: list[dict[str, Any]], *, split: str = "test",
                   limit: int = 50, language: str = "burmese", title: str = "") -> AsyncIterator[dict[str, Any]]:
    """Run one or more models over the same clips. Yields progress, then a 'done' with all results."""
    manifest = prepare.get_prepared(dataset_id)
    dsd = prepare.load_prepared(dataset_id)
    ds = dsd[split] if split in dsd else dsd["train"]
    if len(ds) == 0:
        yield {"type": "error", "message": f"the '{split}' split of this dataset is empty"}
        return
    n = min(limit, len(ds))
    subset = ds.select(range(n))
    refs = [r["text"] for r in subset]
    arrays = [prepare.to_array(r) for r in subset]
    durations = [float(r.get("duration") or len(a) / 16000) for r, a in zip(subset, arrays)]

    results: list[dict[str, Any]] = []
    for ti, target in enumerate(targets):
        t = inference.resolve_target(target)
        yield {"type": "status", "message": f"loading {t['label']}", "target_index": ti, "label": t["label"]}
        try:
            (runner, kind, device), t = await asyncio.to_thread(inference.load_target, target)
        except Exception as e:
            yield {"type": "error", "message": f"{t['label']}: {e}"}
            return
        yield {"type": "status", "message": f"transcribing {n} clips with {t['label']} on {device}",
               "target_index": ti, "label": t["label"]}

        hyps: list[str] = []
        times: list[float] = []
        t0 = time.perf_counter()
        for i, arr in enumerate(arrays):
            s0 = time.perf_counter()
            try:
                txt = await asyncio.to_thread(inference._transcribe_sync, runner, kind, device, [arr], language)
            except Exception as e:
                yield {"type": "error", "message": f"{t['label']} failed on clip {i + 1}: {e}"}
                return
            ms = (time.perf_counter() - s0) * 1000
            hyps.append(txt[0])
            times.append(ms)
            m = sysinfo.quick_mem()
            el = time.perf_counter() - t0
            running = metrics(refs[: i + 1], hyps)
            yield {"type": "progress", "target_index": ti, "label": t["label"], "i": i + 1, "n": n,
                   "cer": running["cer"], "wer": running["wer"], "elapsed": round(el, 1),
                   "eta": round(el / (i + 1) * (n - i - 1), 1), "avg_ms": round(sum(times) / len(times), 1),
                   "ram_mb": round(m["rss_mb"], 1), "vram_mb": round(m["vram_mb"], 1) if m["vram_mb"] is not None else None,
                   "row": {"reference": refs[i], "hypothesis": hyps[i], "duration": durations[i], "ms": round(ms, 1),
                           "cer": metrics([refs[i]], [hyps[i]])["cer"]}}

        mt = metrics(refs, hyps)
        audio_s = sum(durations)
        proc_s = sum(times) / 1000
        results.append({
            "label": t["label"], "target": target, "device": device, **mt,
            "avg_ms": round(sum(times) / len(times), 1), "total_s": round(proc_s, 1),
            "rtf": round(proc_s / audio_s, 3) if audio_s else None, "audio_s": round(audio_s, 1),
            "rows": [{"reference": r, "hypothesis": h, "duration": d, "ms": round(ms, 1), "cer": metrics([r], [h])["cer"]}
                     for r, h, d, ms in zip(refs, hyps, durations, times)],
        })
        yield {"type": "target_done", "target_index": ti, "result": {k: v for k, v in results[-1].items() if k != "rows"}}

    best = min(results, key=lambda r: r["cer"])["label"] if results else None
    record = {
        "id": uuid.uuid4().hex[:12], "created": time.time(),
        "title": title or f"{manifest['name']} · {' vs '.join(r['label'] for r in results)}",
        "dataset_id": dataset_id, "dataset_name": manifest["name"], "split": split, "n": n, "language": language,
        "targets": [r["label"] for r in results], "best": best,
        "summary": [{k: v for k, v in r.items() if k != "rows"} for r in results],
        "results": results,
    }
    EVALS.mkdir(parents=True, exist_ok=True)
    (EVALS / f"{record['id']}.json").write_text(json.dumps(record, ensure_ascii=False))
    yield {"type": "done", "evaluation": record}
