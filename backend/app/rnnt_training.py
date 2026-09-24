"""Fine-tuning for RNN-Transducer ASR models (NVIDIA Nemotron).

Three things the stock transformers path gets wrong for this checkpoint, all handled here:

* its config declares no ``loss_type``, so transformers substitutes a causal-LM loss; the real
  transducer loss comes from ``torchaudio.functional.rnnt_loss``
* its processor emits a start-of-sequence id equal to ``vocab_size`` (13088), one past the end of the
  decoder embedding, so any forward pass crashes; resizing the vocabulary makes that id valid
* its 13k vocabulary has no Myanmar script, so Burmese text encodes to nothing; the characters that
  appear in the training set are added to the tokenizer and the embedding and joint head are grown
  to match, with the existing rows preserved
"""
import json
import math
import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import guards, prepare, sysinfo
from .config import get_hf_token

BLANK_FALLBACK = 13087


def _script_chars(texts: list[str]) -> list[str]:
    """Every character in the corpus that the tokenizer is likely to be missing."""
    out: set[str] = set()
    for t in texts:
        for c in t:
            if c.isascii() or c.isspace():
                continue
            out.add(c)
    return sorted(out)


def _grow_embedding(emb, new_num: int):
    import torch

    old_num, dim = emb.weight.shape
    if new_num <= old_num:
        return emb
    new = torch.nn.Embedding(new_num, dim)
    with torch.no_grad():
        new.weight[:old_num] = emb.weight
        # new rows start near the mean of the existing ones rather than at random scale
        new.weight[old_num:] = emb.weight.mean(0, keepdim=True) + 0.01 * torch.randn(new_num - old_num, dim)
    return new


def _grow_linear(lin, new_out: int):
    import torch

    old_out, in_f = lin.weight.shape
    if new_out <= old_out:
        return lin
    new = torch.nn.Linear(in_f, new_out, bias=lin.bias is not None)
    with torch.no_grad():
        new.weight[:old_out] = lin.weight
        new.weight[old_out:] = lin.weight.mean(0, keepdim=True) + 0.01 * torch.randn(new_out - old_out, in_f)
        if lin.bias is not None:
            new.bias[:old_out] = lin.bias
            new.bias[old_out:] = lin.bias.mean()
    return new


def _set_module(root, dotted: str, value) -> None:
    parts = dotted.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def train_rnnt(cfg, run_id: str, q: "queue.Queue[dict[str, Any]]", stop: threading.Event,
               run_dir_fn, save_run_fn) -> None:
    """Runs in a worker thread; pushes progress dicts into `q`."""
    try:
        import torch
        from torchaudio.functional import rnnt_loss
        from transformers import AutoModel, AutoProcessor

        from . import models_catalog

        spec = models_catalog.resolve(cfg.model)
        device = sysinfo.device()
        guards.check(guards.model_need_mb(0.6e9, cfg.method, device), f"{spec['label']} (RNNT fine-tuning)", device)

        q.put({"type": "status", "stage": "loading", "message": f"loading {spec['label']} on {device}"})
        token = get_hf_token() or None
        model = AutoModel.from_pretrained(spec["path"], token=token)
        processor = AutoProcessor.from_pretrained(spec["path"], token=token)
        tok = processor.tokenizer
        blank = int(getattr(model.config, "blank_token_id", BLANK_FALLBACK))

        dsd = prepare.load_prepared(cfg.dataset_id)
        train_ds, eval_ds = dsd["train"], dsd["test"]
        texts = [r["text"] for r in train_ds]

        # --- grow the vocabulary to cover the script in this dataset
        missing = [c for c in _script_chars(texts) if len(tok.tokenize(c)) == 0 or tok.decode(tok(c).input_ids, skip_special_tokens=True).strip() == ""]
        added = tok.add_tokens(missing) if missing else 0
        new_vocab = len(tok)
        old_vocab = model.decoder.embedding.weight.shape[0]
        if new_vocab > old_vocab:
            _set_module(model, "decoder.embedding", _grow_embedding(model.decoder.embedding, new_vocab))
            head_name = None
            for n, mod in model.named_modules():
                if isinstance(mod, torch.nn.Linear) and mod.weight.shape[0] == old_vocab:
                    head_name = n
                    break
            if head_name is None:
                raise RuntimeError("could not find the joint output layer to resize")
            _set_module(model, head_name, _grow_linear(dict(model.named_modules())[head_name], new_vocab))
            model.config.vocab_size = new_vocab
        q.put({"type": "status", "message": f"vocabulary: added {added} characters, {old_vocab} → {new_vocab} tokens "
                                            f"(blank stays {blank}); embedding and joint head resized"})

        # verify the script now round-trips, otherwise training has no target to learn
        probe = texts[0]
        back = tok.decode(tok(probe).input_ids, skip_special_tokens=True).strip()
        if not back:
            raise RuntimeError("the tokenizer still encodes this script to nothing after extension")
        q.put({"type": "status", "message": f"round-trip check: {'exact' if back == probe.strip() else 'lossy but non-empty'}"})

        # --- what to train
        if cfg.method == "lora":
            for p in model.parameters():
                p.requires_grad = False
            trainable_names = ("decoder", "joint")
            for n, p in model.named_parameters():
                if n.startswith(trainable_names):
                    p.requires_grad = True
            note = "decoder + joint only (encoder frozen)"
        else:
            note = "all parameters"
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        q.put({"type": "status", "message": f"training {n_train / 1e6:.1f}M of {n_all / 1e6:.0f}M parameters: {note}"})

        model.to(device).train()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg.learning_rate)
        steps = cfg.max_steps or max(1, int(len(train_ds) / max(1, cfg.batch_size) * cfg.epochs))
        warm = max(0, min(cfg.warmup_steps, max(steps - 1, 0)))

        def _lr(step_i: int) -> float:
            if warm and step_i < warm:
                return (step_i + 1) / warm
            rest = max(steps - warm, 1)
            return max(0.05, 0.5 * (1 + math.cos(math.pi * min((step_i - warm) / rest, 1.0))))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)

        def make_batch(indices: list[int]):
            feats, labs = [], []
            for i in indices:
                row = train_ds[int(i)]
                arr = prepare.to_array(row)
                b = processor(audio=arr, sampling_rate=16000, text=row["text"], return_tensors="pt")
                feats.append(b["input_features"][0])
                labs.append(b["labels"][0])
            T = max(f.shape[0] for f in feats)
            U = max(len(x) for x in labs)
            x = torch.zeros(len(feats), T, feats[0].shape[1])
            mask = torch.zeros(len(feats), T, dtype=torch.long)
            y = torch.full((len(labs), U), blank, dtype=torch.long)
            ylen = torch.zeros(len(labs), dtype=torch.int32)
            for k, (f, lb) in enumerate(zip(feats, labs)):
                x[k, : f.shape[0]] = f
                mask[k, : f.shape[0]] = 1
                y[k, : len(lb)] = lb
                ylen[k] = len(lb)
            dec_in = torch.cat([torch.full((len(labs), 1), blank, dtype=torch.long), y], dim=1)
            return x, mask, y, ylen, dec_in

        out_dir = run_dir_fn(run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(cfg.seed)
        order = rng.permutation(len(train_ds))
        pos = 0
        t0 = time.perf_counter()
        q.put({"type": "start", "max_steps": steps, "train_clips": len(train_ds),
               "eval_clips": len(eval_ds), "device": device, "steps_per_epoch": max(1, len(train_ds) // max(1, cfg.batch_size))})

        for step in range(1, steps + 1):
            if stop.is_set():
                q.put({"type": "status", "message": f"stopped after {step - 1} steps"})
                break
            if guards.critical():
                q.put({"type": "status", "message": f"stopping: only {guards.free_mb():.0f} MB of RAM left"})
                break
            if pos + cfg.batch_size > len(order):
                order = rng.permutation(len(train_ds))
                pos = 0
            idx = order[pos : pos + cfg.batch_size]
            pos += cfg.batch_size

            x, mask, y, ylen, dec_in = make_batch(list(idx))
            x, mask, y, ylen, dec_in = x.to(device), mask.to(device), y.to(device), ylen.to(device), dec_in.to(device)
            out = model(input_features=x, attention_mask=mask, decoder_input_ids=dec_in)
            logits = out.logits.float()                       # (B, T, U+1, V)
            B, T, U1, V = logits.shape
            xlen = torch.full((B,), T, dtype=torch.int32, device=device)
            loss = rnnt_loss(logits, y.int(), xlen, ylen.to(device), blank=blank, reduction="mean")
            if not torch.isfinite(loss):
                q.put({"type": "status", "message": f"step {step}: non-finite loss, skipped"})
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            mem = sysinfo.quick_mem()
            el = time.perf_counter() - t0
            q.put({"type": "step", "step": step, "max_steps": steps, "loss": float(loss.detach()),
                   "lr": float(sched.get_last_lr()[0]), "epoch": step * cfg.batch_size / max(1, len(train_ds)),
                   "elapsed": round(el, 1), "eta": round(el / step * (steps - step), 1),
                   "ram_mb": round(mem["rss_mb"], 1),
                   "vram_mb": round(mem["vram_mb"], 1) if mem["vram_mb"] is not None else None})

        q.put({"type": "status", "stage": "saving", "message": "saving the fine-tuned model"})
        model_dir = out_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(model_dir))
        processor.save_pretrained(str(model_dir))
        (model_dir / "rnnt.json").write_text(json.dumps({"blank": blank, "vocab": new_vocab, "base": spec["path"]}))
        q.put({"type": "done", "train_runtime": round(time.perf_counter() - t0, 1), "steps": steps,
               "model_dir": str(model_dir), "stopped_early": stop.is_set(), "family": "nemotron_rnnt"})
    except Exception as e:
        import traceback

        q.put({"type": "error", "message": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
