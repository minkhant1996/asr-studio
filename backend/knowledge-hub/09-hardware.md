# Hardware, memory and time

## GPU or CPU

CPU training works and is the right way to prove a pipeline, but it is slow: expect seconds per step
where a GPU takes tens of milliseconds. Two reasons compound. There is no `fp16` on CPU, so everything
runs in float32 at double the memory, and there is no parallel matrix hardware.

A practical rule: use CPU for runs measured in minutes, and a GPU for anything measured in hours.

## Estimating memory

Roughly, for a model with P parameters:

- **Inference:** P × 2 bytes in fp16, or P × 4 on CPU
- **LoRA training:** a little above the inference figure, since the frozen weights dominate
- **Full fine-tuning:** about four times the weights, because gradients and the optimiser's two moment
  estimates each take their own copy

So a 244M model needs about 1.4 GB for LoRA and 2.6 GB for full fine-tuning on a GPU, and roughly
double on CPU. Batch size and clip length add activation memory on top.

## When CUDA will not engage

A common failure: the GPU is present, `nvidia-smi` shows it, but the framework reports no device. The
usual cause is a PyTorch build compiled for a newer CUDA than the installed driver provides. The fix is
to match them — install a PyTorch wheel built for your driver's CUDA version, or update the driver.
Until then everything silently runs on CPU.

## Disk

Prepared 16 kHz mono audio costs about 32 KB per second, so roughly 115 MB per hour, or 2 GB per
10,000 short clips. Model weights are separate: a small Whisper is about 1 GB, a large one 3 GB or more.
Streaming datasets rather than downloading them whole is what keeps this manageable.

## Time budgeting

Work in steps, not epochs. Measure seconds per step on a short run, multiply by your planned steps, and
decide whether that fits your evening or your week before starting.
