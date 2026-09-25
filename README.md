# ASR Studio

An open-source app for fine-tuning and evaluating **speech recognition models on Burmese**, built
around seven public Myanmar speech datasets. Stream a dataset, build a training set, fine-tune a
model, measure character and word error rate against the original base model, and transcribe new
audio — all from a browser, all on your own machine.

FastAPI + PyTorch backend, Vite + React + TypeScript frontend. Docker ready. MIT licensed.

## Why

Burmese is badly served by general ASR models. Out of the box, Whisper tiny transcribes Burmese
speech as Chinese characters and Whisper small emits Khmer and IPA fragments: roughly 100% character
error rate. The data to fix that is public, so this app makes the loop from *dataset* to *fine-tuned
model* to *measured improvement* short enough to actually run.

## The four tabs

**1 · Data**
- Seven curated Burmese datasets, each previewable with a playable clip and its transcript before you
  commit to anything.
- Mix several sources with a per-source clip count. Mixed sources generalise better than one alone.
- Any other Hugging Face speech dataset by `owner/name` or URL.
- Everything is **streamed**: preparing 200 clips from a 115-hour set downloads 200 clips, not 115 hours.
  Clips are decoded, resampled to 16 kHz mono, filtered by duration, split into train/test and cached locally.
- Browse any prepared set: transcripts with an inline audio player per row.

**2 · Fine-tune**
- Base models: Whisper (tiny / small / large-v3-turbo), Qwen3-ASR 1.7B, VibeVoice ASR Streaming 7B,
  NVIDIA Nemotron 3.5 ASR streaming 0.6B. Each card shows parameter count, whether it can be trained
  here, and whether it fits in this machine's RAM or VRAM.
- **LoRA adapters** (light, the default) or a full fine-tune. LoRA on Whisper tiny trains 0.6M of 38M
  parameters and runs on a CPU.
- Steps, batch size, gradient accumulation, learning rate, warmup. The UI shows the effective batch
  size and how many steps make one epoch over your data.
- Live progress: a loss curve, step counter, elapsed time, ETA, and RAM/VRAM. Stop cleanly after the
  current step, or abort.
- **Held-out metrics while it trains.** Every few steps the run transcribes clips it has never seen and
  scores them, so you watch character and word error rate move rather than guessing from the loss. It
  gets its own chart, and the latest reference and model output are shown side by side. The interval is
  automatic (about six checks per run) or you can set it; each check costs a few seconds per clip.
- Every run is saved with its loss history and checkpoint, reopenable and deletable.

**3 · Evaluate**
- Run up to four models over the same held-out clips: base models, your fine-tuned runs, or both.
- **CER leads, WER follows.** Burmese is written without spaces between most words, so word error rate
  depends on how the reference happens to be segmented; character error rate is the number to trust.
- Live per-clip progress with running CER/WER, then a comparison table with CER, WER, latency per clip
  and real-time factor, plus every reference and hypothesis side by side.
- Saved to a history you can reopen.

**4 · Transcribe**
- Drag an audio file onto the drop zone, or click to browse, and transcribe it with a base model or one
  of your fine-tuned runs. Long files are chunked at 30 seconds. Shows duration, wall time and real-time
  factor, with the audio playable next to the transcript.

## Datasets

| Dataset | Size | Licence | Notes |
|---|---|---|---|
| [MIG Burmese Audio Transcription](https://huggingface.co/datasets/Ko-Yin-Maung/mig-burmese-audio-transcription) | 2,822 clips, ~2 h | see card | Parquet, 16 kHz, speaker/sex/age metadata, has a test split. **Best starting point.** |
| [FOEIM Academy 3 h](https://huggingface.co/datasets/freococo/3hr_myanmar_asr_raw_audio) | 3,200 clips, ~2.9 h | MIT | Educational media, subtitle-aligned, one consistent speaker |
| [PVTV News 115 h](https://huggingface.co/datasets/freococo/115hours_pvtv_myanmar_asr) | 156,262 clips, ~115 h | CC0-1.0 | News broadcast, WebDataset |
| [NUG Myanmar ASR](https://huggingface.co/datasets/freococo/nug_myanmar_asr) | ~310,000 clips, ~212 h | CC0-1.0 | Civic and educational speech |
| [Media Queen Entertainment](https://huggingface.co/datasets/freococo/media_queen_entertaiment_voices) | 100K+ clips | other, see card | Conversational register |
| [Khit Thit News](https://huggingface.co/datasets/freococo/khit_thit_news_voices) | 10K+ clips | other, see card | News voices with rich metadata |
| [VOA Burmese](https://huggingface.co/datasets/freococo/voa_myanmar_asr_audio_1) | 1M+ clips | PDDL | **Audio only, no transcripts.** Cannot be used for supervised training; useful for listening, transcription, or pseudo-labelling |

Three different on-disk shapes (Parquet, audiofolder, WebDataset `.tar`/`.tar.gz`) are normalised to
one record of `{audio, text, duration}`, so they mix freely.

## Models

**Only Whisper actually knows Burmese.** That is the single most important fact when choosing a base
model here, so the app shows it on every model card.

| Model | Params | Languages | Burmese? | Fine-tune here | VRAM (LoRA / full) |
|---|---|---|---|---|---|
| [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 809M | 112 | **yes** | LoRA or full | 2.9 / 6.8 GB |
| [Whisper small](https://huggingface.co/openai/whisper-small) | 244M | 112 | **yes** | LoRA or full | 1.4 / 2.6 GB |
| [Whisper tiny](https://huggingface.co/openai/whisper-tiny) | 39M | 112 | **yes** | LoRA or full | 0.9 / 1.1 GB |
| [NVIDIA Nemotron 3.5 ASR streaming](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | 0.6B | 35 | no (added on the fly) | RNNT | 2.3 GB |
| [Qwen3-ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 1.7B | 30 | no | not here | 5.2 GB |
| [VibeVoice ASR Streaming 7B](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B) | 7B | 10 | no | LoRA | 19 / 53 GB |

On CPU the same models need roughly double those figures in RAM, because weights are float32 rather
than float16. Any other Hugging Face speech-sequence-to-sequence model id also works.

Caveats found by testing each one, rather than trusting the cards:

- **Nemotron** is a cache-aware FastConformer RNN-Transducer, and fine-tuning it needed three fixes that
  `app/rnnt_training.py` now applies automatically:
  - its config declares no `loss_type`, so transformers silently substitutes a causal-LM loss; the real
    transducer loss comes from `torchaudio.functional.rnnt_loss`
  - its processor emits a start-of-sequence id of 13088 while the decoder embedding has rows 0–13087, so
    every forward pass crashed with an index error; growing the vocabulary makes that id valid
  - its 13,088-token vocabulary contains no Myanmar script, so Burmese encoded to nothing. The trainer
    collects the characters present in your dataset, adds them to the tokenizer, and grows the decoder
    embedding and joint head to match, preserving the original rows. On the 400-clip set that is 58
    characters, after which Burmese round-trips exactly.

  It trains the decoder and joint network with the encoder frozen: 23.8M of 638M parameters, and the
  fine-tuned copy loads back for transcription with its grown vocabulary. Because it starts from no
  Burmese at all — the added characters begin as random embeddings, and its encoder has never heard the
  language — expect it to need far more data than adapting Whisper, whose Burmese is merely bad rather
  than absent. Its advantage is elsewhere: RNN-Transducers decode in a streaming fashion, so this is the
  architecture to pick if you need low-latency live transcription rather than the best error rate per clip.
- **Qwen3-ASR** ships weight names (`thinker.*`) that transformers 5.x does not map, so loading it that
  way silently produces a randomly initialised model. It needs Qwen's own `qwen-asr` package — and
  installing that downgrades transformers to 4.x, which removes support for Nemotron and VibeVoice.
  `requirements.txt` therefore pins `transformers>=5.17`, and the app refuses Qwen3-ASR with an
  explanation rather than returning nonsense. Run it in a separate environment if you need it.
- **VibeVoice** needs ~19 GB of VRAM just to load, so it could not be verified here.

## Memory safety

Running out of RAM cannot damage hardware, but it makes the machine swap or the process get killed
mid-run. Three things prevent that:

- **Preparation never accumulates.** Clips are written to disk in shards of 500 and the buffer is
  dropped, so memory stays flat whether you take 100 clips or 100,000. Only disk grows: roughly
  190 MB per 1,000 clips at 6 s average. The size is shown before you start.
- **Loading is refused, not attempted.** Before a model loads, the backend estimates its working set
  (weights, plus gradients and optimiser state for a full fine-tune) and compares it with free memory,
  keeping 1.5 GB in reserve. If it will not fit, you get a clear message naming the number instead of
  a frozen machine. The Fine-tune tab shows the estimate live and disables the button.
- **Running jobs stop themselves.** Preparation and training check free memory as they go and stop
  cleanly below 800 MB, keeping whatever was already produced.

On a 64 GB machine with ~34 GB free, this is what the guards report:

| Job | Needs | Verdict |
|---|---|---|
| Prepare 50,000 clips | 0.3 GB RAM, 9.2 GB disk | runs |
| Whisper small, LoRA | 2.1 GB | runs |
| Whisper small, full fine-tune | 4.4 GB | runs |
| VibeVoice 7B, LoRA | 37 GB | refused |
| VibeVoice 7B, full fine-tune | 105 GB | refused |

## Hardware

- **A GPU is optional but strongly recommended for real training.** On CPU, a LoRA fine-tune of
  Whisper tiny runs at roughly 0.6 s/step, which is fine for checking the pipeline; Whisper
  large-v3-turbo on a real dataset is a GPU job.
- Works on NVIDIA (CUDA), Apple Silicon (Metal) and CPU-only machines. The header and Settings tab
  show which device is in use and live RAM/VRAM.
- Disk: model weights land in `~/.cache/huggingface` (Whisper tiny ~150 MB, small ~1 GB,
  large-v3-turbo ~1.6 GB). Prepared clips are stored as 16 kHz PCM16, about 32 KB per second of audio.

## Quick start

```bash
./setup.sh     # venv + deps (torch is large), npm install
./start.sh     # backend on :8767, frontend on :5175
```
Windows: `setup.bat` then `start.bat`. Then open http://localhost:5175.

### Docker

```bash
./docker.sh install   # installs Docker if missing
./docker.sh up        # build and start both containers
./docker.sh logs      # follow    ./docker.sh down   # stop
```
The image ships CPU-only PyTorch. To train on an NVIDIA GPU, install the NVIDIA Container Toolkit,
switch the torch wheel in `backend/Dockerfile`, and uncomment the `deploy.resources` block in
`docker-compose.yml`.

### A first run that finishes in minutes

1. **Data**: click *MIG Burmese Audio Transcription*, set it to 60 clips, press *Prepare dataset*.
2. **Fine-tune**: pick that dataset, Whisper tiny, LoRA, 30 steps. Watch the loss fall.
3. **Evaluate**: compare *Whisper tiny* against your run on the test split. Both will be poor at this
   size — the point is that the loop works end to end. Scale up to Whisper small or large-v3-turbo with
   thousands of clips and a few thousand steps on a GPU for a model that is actually useful.

## API

Interactive docs at http://127.0.0.1:8767/docs.

| Area | Endpoints |
|---|---|
| Health / hardware | `GET /api/health`, `GET /api/system` |
| Datasets | `GET /api/datasets`, `POST /api/datasets/preview` |
| Prepare | `POST /api/prepare/stream` (NDJSON), `GET /api/prepared`, `GET/DELETE /api/prepared/{id}`, `GET /api/prepared/{id}/rows`, `GET /api/prepared/{id}/clip/{split}/{i}` |
| Models | `GET /api/models` |
| Training | `POST /api/train/stream` (NDJSON), `GET /api/runs`, `GET/DELETE /api/runs/{id}`, `POST /api/runs/{id}/stop` |
| Evaluation | `POST /api/evaluate/stream` (NDJSON), `GET /api/evals`, `GET/DELETE /api/evals/{id}` |
| Transcribe | `POST /api/transcribe` (multipart) |
| Settings | `GET /api/settings`, `PUT/DELETE /api/settings/hf` |

## Data & privacy

Everything stays on your machine, in `backend/data/`: prepared clips, training checkpoints, evaluation
history and the encrypted Hugging Face token. That folder is git-ignored. A token is optional since
every dataset and model listed here is public; when you do add one it is verified with the Hub, then
encrypted at rest (Fernet, 0600 permissions) and never returned to the browser.

## Notes on the implementation

- `datasets` 5.x routes every `Audio` column through `torchcodec`. This app casts those columns to raw
  bytes and decodes with soundfile/librosa/ffmpeg instead, which keeps all three dataset formats
  working without that dependency.
- Prepared clips are stored as PCM16 WAV bytes rather than float arrays: half the size, and decodable
  anywhere.
- Whisper checkpoints often carry a stale `forced_decoder_ids` in their generation config that
  silently overrides the language flag. It is cleared on load so Burmese is actually forced.

## License

MIT © 2026 Min Khant Soe. See [LICENSE](LICENSE).

Dataset and model licences are their own: check each card before redistributing anything you train.
