# Choosing and preparing data

## How much do you need

A rough ladder, for adapting a model that already knows the language a little:

| Audio | What to expect |
|---|---|
| under 1 hour | the pipeline works; the model starts writing the right script; error rates stay high |
| 5–20 hours | clearly useful for a narrow domain or a single speaker |
| 50–200 hours | a general-purpose model for one language |
| 500+ hours | competitive with commercial systems, given a good base model |

For a language the base model does *not* know at all, multiply those figures several times over.

## What makes data good

- **Accurate transcripts.** Subtitle-aligned data is convenient but often drifts a few hundred
  milliseconds, or paraphrases speech. Spot-check by listening.
- **Varied speakers.** One speaker for twenty hours teaches the model that speaker. Ten speakers for
  five hours generalises better.
- **Clip length between roughly 1 and 20 seconds.** Whisper truncates at 30 seconds. Very short clips
  carry little context; very long ones waste padding.
- **Matched conditions.** If you will transcribe phone calls, train on phone-quality audio.
- **Mixed registers.** News reading, conversation and scripted speech stress different things. A mix
  generalises better than any single source.

## Formats you will meet

- **Parquet with an audio column.** The easiest: everything is in one place, splits are defined.
- **Audiofolder.** Media files plus a metadata CSV. Streaming these is slow because each clip is a
  separate HTTP request; fine for hundreds, painful for thousands.
- **WebDataset.** Tar archives of paired `clip.mp3` and `clip.json`. Streams very fast because it is
  sequential, and it is how most large speech corpora are shipped.

## Splitting

Hold out a test split before you train and never touch it. This app fixes the split when a dataset is
prepared, with a fixed shuffle seed, so every run and every evaluation scores the same clips. A
result you cannot compare against another result is not a result.

## Licensing

Check it before you build anything you intend to publish. Public-domain marks (CC0, PDDL) let you do
anything. Share-alike (CC BY-SA) requires attribution and propagates terms. Anything marked
non-commercial, or scraped from a platform under a fair-use rationale, restricts a model trained on
it, whatever the training code's own licence says.
