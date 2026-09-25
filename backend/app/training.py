"""Fine-tuning. Whisper-family models train end to end (or with LoRA); other families are
attempted through AutoModelForSpeechSeq2Seq and refused early if they cannot produce a loss."""
import asyncio
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from . import guards, models_catalog, prepare, sysinfo
from .config import get_hf_token, settings

RUNS = Path(settings.runs_dir)
_stop_flags: dict[str, threading.Event] = {}


def run_dir(run_id: str) -> Path:
    return RUNS / run_id


def sweep_stale_runs() -> int:
    """A run marked 'running' cannot survive a backend restart: mark those interrupted on startup so
    the history does not show a job that is no longer going anywhere."""
    n = 0
    for r in list_runs():
        if r.get("status") == "running":
            r["status"] = "interrupted"
            r["message"] = "the backend restarted while this run was in progress"
            save_run(r)
            n += 1
    return n


def list_runs() -> list[dict[str, Any]]:
    out = []
    if not RUNS.exists():
        return out
    for d in RUNS.iterdir():
        m = d / "run.json"
        if m.is_file():
            try:
                out.append(json.loads(m.read_text()))
            except Exception:
                continue
    return sorted(out, key=lambda r: -r.get("created", 0))


def get_run(run_id: str) -> dict[str, Any]:
    p = run_dir(run_id) / "run.json"
    if not p.exists():
        raise FileNotFoundError("run not found")
    return json.loads(p.read_text())


def save_run(run: dict[str, Any]) -> None:
    d = run_dir(run["id"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1))


def delete_run(run_id: str) -> bool:
    import shutil

    d = run_dir(run_id)
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False


def stop_run(run_id: str) -> bool:
    ev = _stop_flags.get(run_id)
    if ev:
        ev.set()
        return True
    return False


@dataclass
class TrainConfig:
    dataset_id: str
    model: str = models_catalog.DEFAULT_MODEL
    language: str = "burmese"
    task: str = "transcribe"
    method: str = "lora"           # "lora" | "full"
    epochs: float = 1.0
    max_steps: int = 0             # 0 = derive from epochs
    batch_size: int = 2
    grad_accum: int = 4
    learning_rate: float = 1e-5
    warmup_steps: int = 10
    eval_steps: int = 0            # run a held-out CER/WER check every N steps (0 = off)
    eval_clips: int = 8            # how many held-out clips each check transcribes
    save_steps: int = 0
    fp16: bool = True
    freeze_encoder: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 42
    name: str = ""


_PARAMS = {"whisper-tiny": 39e6, "whisper-small": 244e6, "whisper-large-v3-turbo": 809e6,
           "qwen3-asr-1.7b": 1.7e9, "vibevoice-asr-streaming-7b": 7e9, "nemotron-3.5-asr-streaming-0.6b": 0.6e9}


def _params_of(spec: dict[str, Any]) -> float:
    if spec.get("id") in _PARAMS:
        return _PARAMS[spec["id"]]
    txt = str(spec.get("params", "")).strip().upper()
    try:
        if txt.endswith("B"):
            return float(txt[:-1]) * 1e9
        if txt.endswith("M"):
            return float(txt[:-1]) * 1e6
    except ValueError:
        pass
    return 800e6


def _language_for(processor, language: str) -> str | None:
    try:
        tok = getattr(processor, "tokenizer", processor)
        langs = getattr(tok, "supported_languages", None) or {}
        _ = langs
    except Exception:
        pass
    return language or None


class _Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        _ = torch
        return batch


def _quick_eval(model, processor, rows, language: str, max_new_tokens: int = 220) -> dict[str, Any]:
    """Transcribe a few held-out clips and score them. Runs inside the training thread."""
    import torch

    from .evaluation import metrics

    fe, tok = processor.feature_extractor, processor.tokenizer
    was_training = model.training
    model.eval()
    refs, hyps = [], []
    try:
        for row in rows:
            arr = prepare.to_array(row)
            feats = fe(arr, sampling_rate=prepare.TARGET_SR, return_tensors="pt").input_features.to(model.device)
            if next(model.parameters()).dtype == torch.float16:
                feats = feats.half()
            with torch.no_grad():
                try:
                    ids = model.generate(feats, max_new_tokens=max_new_tokens, language=language, task="transcribe")
                except Exception:
                    ids = model.generate(feats, max_new_tokens=max_new_tokens)
            hyps.append(tok.batch_decode(ids, skip_special_tokens=True)[0].strip())
            refs.append(row["text"])
    finally:
        if was_training:
            model.train()
    m = metrics(refs, hyps)
    return {"cer": m["cer"], "wer": m["wer"], "n": m["n"],
            "sample": {"reference": refs[0] if refs else "", "hypothesis": hyps[0] if hyps else ""}}


def _build_processor(path: str, language: str, task: str):
    from transformers import AutoProcessor

    kw = {"token": get_hf_token() or None}
    try:
        return AutoProcessor.from_pretrained(path, language=language, task=task, **kw)
    except Exception:
        return AutoProcessor.from_pretrained(path, **kw)


def _map_dataset(ds, processor, language: str, task: str, max_labels: int = 448, on_truncate=None):
    """Extract log-mel features and label ids.

    Whisper's decoder is capped at 448 positions, and Burmese costs roughly 2.9 tokens per character
    against about 0.3 for English, so a long clip can overflow that limit. Labels are truncated to fit
    rather than failing the whole run.
    """
    fe, tok = processor.feature_extractor, processor.tokenizer
    try:
        tok.set_prefix_tokens(language=language, task=task)
    except Exception:
        pass
    truncated = {"n": 0}

    def _prep(batch):
        arr = prepare.to_array(batch)
        batch["input_features"] = fe(arr, sampling_rate=prepare.TARGET_SR).input_features[0]
        ids = tok(batch["text"]).input_ids
        if len(ids) > max_labels:
            truncated["n"] += 1
            eos = tok.eos_token_id
            ids = ids[: max_labels - 1] + ([eos] if eos is not None else [])
        batch["labels"] = ids
        return batch

    out = ds.map(_prep, remove_columns=ds.column_names, desc="extracting features")
    if truncated["n"] and on_truncate:
        on_truncate(truncated["n"], max_labels)
    return out


def _train_sync(cfg: TrainConfig, run_id: str, q: "queue.Queue[dict[str, Any]]", stop: threading.Event) -> None:
    """Runs in a worker thread and pushes progress dicts into `q`. Final push is a 'done' or 'error'."""
    spec0 = models_catalog.resolve(cfg.model)
    if spec0.get("family") == "nemotron_rnnt":
        from .rnnt_training import train_rnnt

        return train_rnnt(cfg, run_id, q, stop, run_dir, save_run)
    try:
        import torch
        from transformers import (AutoModelForSpeechSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                                  TrainerCallback)

        spec = models_catalog.resolve(cfg.model)
        if not models_catalog.supports_training(spec):
            raise RuntimeError(f"{spec['label']} cannot be fine-tuned in this app. {spec.get('note', '')} "
                               "Pick a Whisper model, which supports Burmese and trains here.")
        if cfg.language and spec.get("burmese") is False:
            q.put({"type": "status", "message": f"note: {spec['label']} was not trained on Burmese "
                                                f"({spec.get('languages')} languages, Burmese not among them), so it has no "
                                                "Burmese tokens to build on and will need far more data than Whisper."})

        device = sysinfo.device()
        params = _params_of(spec)
        need = guards.model_need_mb(params, cfg.method, device)
        guards.check(need, f"{spec['label']} ({cfg.method} fine-tuning)", device)
        q.put({"type": "status", "stage": "loading",
               "message": f"loading {spec['label']} on {device} (needs ~{need / 1024:.1f} GB, "
                          f"{guards.free_mb() / 1024:.1f} GB free)"})
        processor = _build_processor(spec["path"], cfg.language, cfg.task)
        dtype = torch.float16 if (cfg.fp16 and device == "cuda") else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(spec["path"], dtype=dtype, token=get_hf_token() or None)
        model.config.use_cache = False
        if hasattr(model, "generation_config"):
            try:
                model.generation_config.language = cfg.language
                model.generation_config.task = cfg.task
                model.generation_config.forced_decoder_ids = None
            except Exception:
                pass

        if cfg.method == "lora":
            from peft import LoraConfig, get_peft_model

            targets = ["q_proj", "v_proj", "k_proj", "out_proj"]
            peft_cfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                                  target_modules=targets, bias="none")
            model = get_peft_model(model, peft_cfg)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            q.put({"type": "status", "message": f"LoRA: training {trainable / 1e6:.1f}M of {total / 1e6:.0f}M parameters"})
        elif cfg.freeze_encoder and hasattr(model, "freeze_encoder"):
            model.freeze_encoder()
            q.put({"type": "status", "message": "encoder frozen: only the decoder is trained"})

        q.put({"type": "status", "stage": "features", "message": "extracting log-mel features"})
        dsd = prepare.load_prepared(cfg.dataset_id)
        max_labels = int(getattr(model.config, "max_target_positions", 448) or 448)

        def _warn(n, limit):
            q.put({"type": "status", "message": f"{n} transcript(s) longer than the model's {limit}-token limit were truncated "
                                                "(Burmese costs ~2.9 tokens per character in Whisper's vocabulary)"})

        train_ds = _map_dataset(dsd["train"], processor, cfg.language, cfg.task, max_labels, _warn)
        eval_ds = _map_dataset(dsd["test"], processor, cfg.language, cfg.task, max_labels) if len(dsd["test"]) else None

        out = run_dir(run_id)
        out.mkdir(parents=True, exist_ok=True)
        steps_per_epoch = max(1, len(train_ds) // max(1, cfg.batch_size * cfg.grad_accum))
        max_steps = cfg.max_steps or max(1, int(steps_per_epoch * cfg.epochs))

        args = Seq2SeqTrainingArguments(
            output_dir=str(out / "checkpoints"),
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            learning_rate=cfg.learning_rate,
            warmup_steps=min(cfg.warmup_steps, max(0, max_steps - 1)),
            max_steps=max_steps,
            fp16=(cfg.fp16 and device == "cuda"),
            logging_steps=1,
            save_steps=cfg.save_steps or max_steps,
            save_total_limit=1,
            eval_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=0,
            seed=cfg.seed,
            label_names=["labels"],
        )

        t0 = time.perf_counter()
        eval_rows = [dsd["test"][i] for i in range(min(cfg.eval_clips, len(dsd["test"])))] if len(dsd["test"]) else []
        eval_every = cfg.eval_steps if cfg.eval_steps else (0 if max_steps < 20 else max(10, max_steps // 6))
        if eval_rows and eval_every:
            q.put({"type": "status", "message": f"held-out check every {eval_every} steps on {len(eval_rows)} clips "
                                                "(character error rate is the one to watch for Burmese)"})

        class _Cb(TrainerCallback):
            def on_log(self, a, state, control, logs=None, **kw):
                if not logs:
                    return
                mem = sysinfo.quick_mem()
                el = time.perf_counter() - t0
                step = int(state.global_step or 0)
                q.put({"type": "step", "step": step, "max_steps": max_steps,
                       "loss": logs.get("loss"), "lr": logs.get("learning_rate"),
                       "grad_norm": logs.get("grad_norm"), "epoch": logs.get("epoch"),
                       "elapsed": round(el, 1),
                       "eta": round(el / max(step, 1) * max(max_steps - step, 0), 1),
                       "ram_mb": round(mem["rss_mb"], 1),
                       "vram_mb": round(mem["vram_mb"], 1) if mem["vram_mb"] is not None else None})

            def on_step_end(self, a, state, control, **kw):
                step = int(state.global_step or 0)
                if eval_rows and eval_every and step > 0 and (step % eval_every == 0 or step == max_steps):
                    try:
                        e0 = time.perf_counter()
                        r = _quick_eval(model, processor, eval_rows, cfg.language)
                        q.put({"type": "eval", "step": step, **r, "seconds": round(time.perf_counter() - e0, 1)})
                    except Exception as ex:
                        q.put({"type": "status", "message": f"held-out check failed at step {step}: {ex}"})
                if stop.is_set():
                    control.should_training_stop = True
                elif guards.critical():
                    q.put({"type": "status", "message": f"stopping early: only {guards.free_mb():.0f} MB of RAM left. "
                                                        "Lower the batch size or use a smaller model."})
                    control.should_training_stop = True
                return control

        trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                                 data_collator=_Collator(processor), callbacks=[_Cb()])
        q.put({"type": "start", "max_steps": max_steps, "train_clips": len(train_ds),
               "eval_clips": len(eval_ds) if eval_ds is not None else 0, "device": device,
               "steps_per_epoch": steps_per_epoch})
        result = trainer.train()

        q.put({"type": "status", "stage": "saving", "message": "saving the fine-tuned model"})
        model_dir = out / "model"
        trainer.save_model(str(model_dir))
        processor.save_pretrained(str(model_dir))
        q.put({"type": "done", "train_runtime": round(result.metrics.get("train_runtime", 0), 1),
               "train_loss": result.metrics.get("train_loss"), "steps": int(result.metrics.get("step", max_steps)),
               "model_dir": str(model_dir), "stopped_early": stop.is_set()})
    except Exception as e:  # surfaced to the UI
        import traceback

        q.put({"type": "error", "message": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})


async def train(cfg: TrainConfig) -> AsyncIterator[dict[str, Any]]:
    run_id = uuid.uuid4().hex[:12]
    spec = models_catalog.resolve(cfg.model)
    ds_manifest = prepare.get_prepared(cfg.dataset_id)
    run: dict[str, Any] = {
        "id": run_id, "created": time.time(), "status": "running", "name": cfg.name or f"{spec['label']} · {ds_manifest['name']}",
        "model": cfg.model, "model_label": spec["label"], "dataset_id": cfg.dataset_id, "dataset_name": ds_manifest["name"],
        "clips": ds_manifest["clips"], "config": cfg.__dict__, "device": sysinfo.device(), "losses": [], "evals": [],
    }
    save_run(run)
    yield {"type": "run", "run": run}

    q: "queue.Queue[dict[str, Any]]" = queue.Queue()
    stop = threading.Event()
    _stop_flags[run_id] = stop
    worker = threading.Thread(target=_train_sync, args=(cfg, run_id, q, stop), daemon=True)
    worker.start()

    try:
        while True:
            try:
                ev = await asyncio.to_thread(q.get, True, 1.0)
            except queue.Empty:
                if not worker.is_alive():
                    break
                continue
            ev["run_id"] = run_id
            if ev["type"] == "step" and ev.get("loss") is not None:
                run["losses"].append({"step": ev["step"], "loss": ev["loss"]})
            if ev["type"] == "eval":
                run["evals"].append({k: ev[k] for k in ("step", "cer", "wer", "n")})
                save_run(run)
            if ev["type"] in ("done", "error"):
                run["status"] = "done" if ev["type"] == "done" else "error"
                run["finished"] = time.time()
                run.update({k: v for k, v in ev.items() if k in ("train_runtime", "train_loss", "steps", "model_dir", "message", "stopped_early")})
                save_run(run)
                yield ev
                break
            yield ev
    finally:
        _stop_flags.pop(run_id, None)
        if run.get("status") == "running":
            run["status"] = "stopped"
            save_run(run)
