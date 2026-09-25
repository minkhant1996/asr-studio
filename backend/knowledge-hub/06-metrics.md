# Measuring whether it worked

## WER and CER

**Word error rate** counts word-level insertions, deletions and substitutions against the reference,
divided by the number of reference words. **Character error rate** does the same over characters.

Both can exceed 100%, because insertions are unbounded. A model that emits a long repeated string
against a short reference can score several hundred percent. That is a signal in itself: it usually
means the model is producing the wrong script or has fallen into a repetition loop.

## Which to trust

**For languages written without spaces between words — Burmese, Thai, Chinese, Japanese, Lao, Khmer —
character error rate is the metric to trust.** Word error rate depends entirely on how the reference
happened to be segmented, and two equally good transcripts can differ by tens of points simply because
one inserted spaces differently.

For space-separated languages, word error rate is the conventional headline and character error rate
is the tie-breaker.

## Normalisation

Before scoring, decide what you are measuring. Punctuation, casing and digit formatting are usually
normalised away, because a model that writes "5" for "five" is rarely wrong in a way you care about.
Report what you normalised: an unnormalised score and a normalised one are not comparable.

## Watch the held-out set, not the loss

Training loss falling means the model fits the training data better. It does not mean the transcripts
improved. Score held-out clips periodically during training and use that curve to decide when to stop.

## Other numbers worth keeping

- **Real-time factor:** processing time ÷ audio duration. Below 1.0 is faster than real time.
- **Latency per clip**, which matters for interactive use.
- **Per-label or per-source breakdown**, which reveals that an average hides one terrible source.
