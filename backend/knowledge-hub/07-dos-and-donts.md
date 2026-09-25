# Do and don't

## Do

- **Listen to your data before training on it.** Play twenty random clips and read their transcripts.
  Misalignment and paraphrasing are common in subtitle-derived corpora and invisible in the numbers.
- **Fix a held-out split once and never touch it.** Comparable results depend on it.
- **Run the smallest model first.** A ten-minute run that proves the pipeline saves a wasted overnight one.
- **Watch held-out error rate, not loss.** They diverge exactly when it matters.
- **Keep the best checkpoint, not the last.** Models often peak mid-run and drift afterwards.
- **Change one thing at a time.** Two changes and a better number tell you nothing.
- **Mix sources.** Several registers generalise better than one large homogeneous corpus.
- **Record what you ran.** Model, data, learning rate, steps. An unrepeatable good result is folklore.
- **Check the licence** of every dataset before publishing a model trained on it.

## Don't

- **Don't judge by the training loss.** It falls while the model memorises.
- **Don't evaluate on training data.** The number will be excellent and meaningless.
- **Don't train a model that has no tokens for your script** and expect a small corpus to fix it.
- **Don't use word error rate as the headline for Burmese, Thai, Chinese or Japanese.** Segmentation
  dominates the score.
- **Don't raise the learning rate to make the loss fall faster.** It erases what the model knew.
- **Don't train for many epochs on a small corpus.** You are memorising, not learning.
- **Don't compare runs on different data or different held-out splits.**
- **Don't ignore clip-length limits.** Whisper's decoder caps at 448 tokens, and scripts that tokenise
  expensively — Burmese costs roughly 2.9 tokens per character against about 0.3 for English — can
  overflow that on a surprisingly short clip.
- **Don't trust a single aggregate number.** Break it down by source and by speaker.
