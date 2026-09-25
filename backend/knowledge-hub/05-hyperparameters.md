# Settings that actually change the outcome

## Learning rate

The single most important number.

- **LoRA:** 1e-3 is a sensible starting point. Adapters are small and tolerate a high rate.
- **Full fine-tuning:** 1e-5 to 5e-5. Higher, and you erase what the model already knew.
- **A model learning a new script:** start lower, around 5e-5 even with LoRA, because freshly
  initialised embeddings make the early loss violently unstable.

If the loss spikes or turns to NaN, the learning rate is too high.

## Batch size and gradient accumulation

Effective batch = batch size × accumulation steps. Larger effective batches give steadier gradients.
When memory is tight, keep the batch small and raise accumulation: the arithmetic is identical, the
memory is not.

## Steps and epochs

One epoch is `train_clips ÷ effective_batch` steps. Two to five epochs is typical for fine-tuning.
Far more than that on a small corpus memorises it.

## Warmup

A few dozen steps of gradually increasing learning rate avoids wrecking the model in the first few
batches. More matters when the learning rate is high or the model is learning new vocabulary.

## Precision

`fp16` halves memory and roughly doubles speed on a GPU. On CPU it is unavailable, so everything runs
in float32 — one reason CPU training is slow as well as memory-hungry.

## A reasonable starting point

For LoRA on a mid-size Whisper with a few thousand clips: rank 16, learning rate 1e-3, effective batch
8 to 16, 2 to 3 epochs, warmup 50, early stopping on held-out error rate with patience 3.
