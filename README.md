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
- Every run is saved with its loss history and checkpoint, reopenable and deletable.

**3 · Evaluate**
- Run up to four models over the same held-out clips: base models, your fine-tuned runs, or both.
- **CER leads, WER follows.** Burmese is written without spaces between most words, so word error rate
  depends on how the reference happens to be segmented; character error rate is the number to trust.
- Live per-clip progress with running CER/WER, then a comparison table with CER, WER, latency per clip
  and real-time factor, plus every reference and hypothesis side by side.
- Saved to a history you can reopen.

**4 · Transcribe**
- Upload any audio file and transcribe it with a base model or one of your fine-tuned runs. Long files
  are chunked at 30 seconds. Shows duration, wall time and real-time factor.

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

| Model | Params | Fine-tuning here | Rough memory |
|---|---|---|---|
| [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 809M | LoRA or full | ~8 GB VRAM (LoRA) |
| [Whisper small](https://huggingface.co/openai/whisper-small) | 244M | LoRA or full | ~5 GB VRAM, or CPU |
| [Whisper tiny](https://huggingface.co/openai/whisper-tiny) | 39M | LoRA or full | CPU-friendly |
| [Qwen3-ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 1.7B | LoRA | ~14 GB VRAM |
| [VibeVoice ASR Streaming 7B](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B) | 7B | LoRA | ~24 GB VRAM |
| [NVIDIA Nemotron 3.5 ASR streaming](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | 0.6B | **inference only** | ~3 GB |

Nemotron is an RNN-Transducer trained with NVIDIA NeMo; this app can transcribe and evaluate with it,
but RNNT fine-tuning needs the NeMo toolkit, so the UI marks it inference-only rather than pretending
otherwise. Any other Hugging Face speech-sequence-to-sequence model id also works.

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
