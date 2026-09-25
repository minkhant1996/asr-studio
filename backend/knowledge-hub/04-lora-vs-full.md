# LoRA or full fine-tuning

## What each does

**Full fine-tuning** updates every weight. Maximum capacity to change the model, maximum memory, and
the largest artifact — a copy of the whole model per experiment.

**LoRA** freezes the original weights and trains small low-rank adapter matrices injected into the
attention projections. Typically around 1–2% of the parameters are trainable. Memory roughly halves,
the saved artifact is megabytes rather than gigabytes, and the base model cannot be damaged, because
it is never modified.

## Choosing between them

| Situation | Prefer |
|---|---|
| Under a few thousand clips | LoRA: full fine-tuning will overfit |
| Adapting an accent, a domain vocabulary, a recording condition | LoRA |
| Large shift, such as a language the model writes in the wrong script | full, once you have the data |
| Limited VRAM | LoRA |
| Many experiments to keep and compare | LoRA: small artifacts |
| Squeezing out the last few points of accuracy | full |

The crossover is usually in the low thousands of clips. Below it, LoRA's inability to wreck the base
model is a genuine advantage; above it, full fine-tuning's extra capacity starts to pay.

## LoRA settings that matter

- **Rank (r).** Capacity. 8–16 for light adaptation, 32–64 when you are asking for a bigger change
  such as a new script. Higher rank costs little memory but can overfit small data.
- **Alpha.** Scaling, conventionally 2×r.
- **Target modules.** The attention projections (q, k, v, output) are the standard choice.

## A middle option

Freezing the encoder and training only the decoder sits between the two. The encoder already models
speech acoustics well; the decoder is what writes your language. This is often the right choice when
the acoustics are familiar but the output language is not.
