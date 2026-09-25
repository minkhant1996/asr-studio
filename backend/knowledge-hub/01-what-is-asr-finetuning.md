# What fine-tuning an ASR model actually means

A speech recognition model maps audio to text. The large open models — Whisper, Qwen3-ASR,
NVIDIA's Nemotron — were trained on enormous multilingual corpora, so they already know how speech
works in general. Fine-tuning does not teach a model to hear. It adjusts a model that already hears
so that it writes what *you* need: your language, your accent, your vocabulary, your recording
conditions.

## When fine-tuning helps

- **The language is supported but weak.** Whisper lists 112 languages, but the long tail is poor.
  Burmese out of the box produces Khmer or Chinese characters. The knowledge is there but thin, and a
  few hours of audio moves it a long way.
- **A domain vocabulary is missing.** Medical terms, product names, place names, call-centre jargon.
  The base model hears the sounds and writes the wrong words.
- **The audio is unusual.** Telephone bandwidth, noisy field recordings, a single distinctive speaker.
- **A dialect differs from the standard written form.** Thai Korat, Kham Mueang and Pattani speech is
  not transcribed the way central Thai is.

## When it does not help

- **The language is absent from the model entirely.** Nemotron covers 35 language-locales and Burmese
  is not one of them; its tokenizer turns Burmese into empty strings. You would be adding a new script
  and a new vocabulary, which is closer to training a new model than adapting one.
- **You have very little data.** Under an hour, expect a demonstration rather than a usable system.
- **The transcripts are wrong.** Fine-tuning faithfully reproduces whatever errors are in the labels.
  Bad data is worse than no data.

## The honest shape of the work

Most of the effort is data, not training. Collecting, aligning, cleaning and checking transcripts is
where a usable model is won or lost. The training loop is largely a solved problem you can run from
this app; the corpus is the part only you can supply.
