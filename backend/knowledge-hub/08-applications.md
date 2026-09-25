# What people build with fine-tuned ASR

## Transcription and subtitling

The direct use: turning recordings into text. Media archives, lecture capture, subtitle generation.
Accuracy matters more than latency, so a large offline model like Whisper is the right shape.

## Live captioning and voice interfaces

Text must appear as the person speaks. This needs a streaming architecture — an RNN-Transducer —
because an encoder–decoder model that waits for a 30-second window cannot feel responsive. Accuracy is
traded for latency deliberately.

## Call-centre and meeting analytics

Transcription feeding search, summarisation or quality monitoring. Domain vocabulary matters far more
than general accuracy: product names and jargon are exactly what a base model gets wrong and what
fine-tuning fixes cheaply.

## Language preservation and accessibility

For under-resourced languages, an ASR model is infrastructure. It makes archives searchable, supports
captioning for deaf and hard-of-hearing speakers, and creates a path to other language technology.
This is often where fine-tuning matters most, because commercial systems do not cover the language at all.

## Data creation for other models

A reasonable ASR model transcribes untranscribed audio to produce training data for a better one, or
for text-to-speech. Audio-only corpora become useful this way. Pseudo-labels carry the model's errors,
so filter by confidence and expect diminishing returns.

## Dialect and regional work

Standard-language models often transcribe dialect speech into the standard written form, erasing
exactly what you wanted to capture. Corpora that carry both a dialect transcript and a standard one
let you choose which you are training toward.
