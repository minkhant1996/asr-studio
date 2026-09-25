# Picking a base model

## The only question that matters first

**Does this model already know your language?** Everything else is secondary. A model with no tokens
for your script has to learn the script, the vocabulary and the acoustics at once, and will need
orders of magnitude more data than one that merely transcribes your language badly.

Check the model card's language list. If your language is absent, expect to add vocabulary and to
need a large corpus.

## Families you will encounter

**Whisper (OpenAI).** Encoder–decoder transformer, 112 languages, sizes from 39M to 1.5B. The default
choice for most fine-tuning: well supported, forgiving, and its weak languages respond quickly to a few
hours of audio. It is not streaming; it works on 30-second windows.

**RNN-Transducer (NVIDIA Nemotron, Parakeet, most production systems).** An encoder, a prediction
network and a joint network trained with the transducer loss. Decodes as audio arrives, so it is the
architecture for live captioning and low-latency applications. Fine-tuning needs the RNNT loss
specifically; a cross-entropy loss silently optimises the wrong objective.

**Audio-LLM hybrids (Qwen3-ASR and similar).** An audio encoder feeding a language model. Strong on
the languages they cover, and flexible about prompting, but heavier and often dependent on the
vendor's own runtime package.

## Size versus practicality

Bigger models know your language better before you start, which means less data to reach a given error
rate. They also cost more memory and time. A practical order:

1. Prove the pipeline on the smallest model in the family. Minutes, not hours.
2. Move to a mid-size model for real experiments.
3. Spend GPU time on the largest model only once the data and settings are settled.
