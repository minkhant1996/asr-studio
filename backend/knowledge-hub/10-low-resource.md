# Working with a low-resource language

Most of the world's languages are poorly served by speech technology, not because the problem is hard
but because nobody has assembled the data. If that is your situation, the work looks different.

## Establish the baseline first

Run the base model on your held-out clips before training anything. You need to know whether it
produces the wrong words, the wrong script, or nothing usable at all. Those three cases call for
different amounts of data and different methods, and the number you record becomes the thing every
later result is measured against.

## Check tokenisation before anything else

Encode a sentence of your language and decode it back. If what returns is empty or mangled, the model
has no tokens for your script and fine-tuning as normally practised will not work. You would need to
extend the vocabulary and resize the output layer, which puts you in new-model territory.

Also measure how expensively your script tokenises. Some scripts cost many tokens per character, which
eats into decoder length limits and makes long clips fail.

## Find data in unobvious places

Public broadcasters, civic and government media, educational channels, parliamentary records, religious
recordings and subtitled video. Subtitle-aligned material is the most common practical source. Check
alignment drift, because subtitles are timed for reading rather than for exactness.

## Audio without transcripts is still useful

It cannot be used for supervised training, but it supports self-supervised pretraining, voice activity
detection, and pseudo-labelling once you have a reasonable model. Do not discard it.

## Dialects deserve a decision

Decide early whether you are transcribing what was said or what the standard orthography would write.
Both are legitimate; mixing them in one training set teaches the model to be inconsistent.

## Expect the first result to look bad

Going from a completely unusable model to one that writes the correct script with many errors is real
progress, even when the error rate is still high. Measure the trend, not the absolute number.
