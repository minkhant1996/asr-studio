"""Burmese ASR datasets: a curated catalogue plus any Hugging Face dataset or local upload.

The three on-disk shapes are normalised to one record: {audio, text, duration}.
  * parquet with an `audio` feature + a transcript column (mig-burmese)
  * audiofolder: mp3 files + metadata.csv (3hr FOEIM)
  * webdataset: `<key>.mp3` + `<key>.json` inside .tar / .tar.gz (PVTV, NUG, VOA, Media Queen, Khit Thit)
"""
import asyncio
import io
import re
import json
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Any

from .config import get_hf_token

AUDIO_KEYS = ("audio", "mp3", "wav", "flac", "ogg", "m4a", "opus", "audio_filepath")
TEXT_KEYS = ("transcription", "transcript", "sentence", "text", "caption", "normalized_text", "raw_text")


@dataclass
class DatasetSpec:
    id: str
    name: str
    path: str
    hours: str
    clips: str
    license: str
    description: str
    config: str | None = None
    data_dir: str | None = None
    splits: list[str] = field(default_factory=lambda: ["train"])
    text_field: str | None = None          # where the transcript lives (may be inside a json column)
    streaming_only: bool = False
    transcripts: bool = True               # False = audio only, cannot be used for supervised training
    language: str = "Burmese"              # what Whisper should be told to transcribe
    lang_code: str = "my"
    slow_stream: bool = False              # one HTTP request per clip (audiofolder): fine for hundreds, not thousands


CATALOG: dict[str, DatasetSpec] = {
    d.id: d
    for d in [
        DatasetSpec(
            id="mig_burmese", name="MIG Burmese Audio Transcription", path="Ko-Yin-Maung/mig-burmese-audio-transcription",
            hours="~2", clips="2,822", license="see card", splits=["train", "test"], text_field="transcription",
            description="Parquet, 16 kHz, with speaker / sex / age metadata. Small and clean: the best starting point.",
        ),
        DatasetSpec(
            id="foeim_3hr", name="FOEIM Academy 3 h", path="freococo/3hr_myanmar_asr_raw_audio",
            hours="~2.9", clips="3,200", license="MIT", text_field="transcript", slow_stream=True,
            description="Educational media, subtitle-aligned mp3 clips of 0.25–19 s, one consistent speaker.",
        ),
        DatasetSpec(
            id="pvtv_115h", name="PVTV News 115 h", path="freococo/115hours_pvtv_myanmar_asr", data_dir="train",
            hours="~115", clips="156,262", license="CC0-1.0", streaming_only=True, text_field="transcript",
            description="News broadcast, WebDataset tars of 1–15 s segments. Large: stream a subset.",
        ),
        DatasetSpec(
            id="nug", name="NUG Myanmar ASR", path="freococo/nug_myanmar_asr", data_dir="train",
            hours="~212", clips="310,000", license="CC0-1.0", streaming_only=True, text_field="transcript",
            description="Two volumes of civic / educational speech in WebDataset format.",
        ),
        DatasetSpec(
            id="voa", name="VOA Burmese (audio only)", path="freococo/voa_myanmar_asr_audio_1", data_dir="train",
            hours="very large", clips="1M+", license="PDDL", streaming_only=True, text_field="transcript", transcripts=False,
            description="Voice of America Burmese broadcasts, 122 WebDataset tars. Ships audio and broadcast metadata but "
                        "no transcripts, so it cannot be used for supervised fine-tuning: use it to listen, to transcribe, "
                        "or to build pseudo-labels with a trained model.",
        ),
        DatasetSpec(
            id="media_queen", name="Media Queen Entertainment", path="freococo/media_queen_entertaiment_voices", data_dir="train",
            hours="large", clips="100K+", license="other (see card)", streaming_only=True, text_field="transcript",
            description="Entertainment media voices, WebDataset tar.gz. Conversational register.",
        ),
        DatasetSpec(
            id="khit_thit", name="Khit Thit News", path="freococo/khit_thit_news_voices", data_dir="train",
            hours="medium", clips="10K+", license="other (see card)", streaming_only=True, text_field="transcript",
            description="News voices with rich per-clip metadata (title, source video, duration).",
        ),
        # ---------------------------------------------------------------- Thai
        DatasetSpec(
            id="porjai_central", name="Porjai Thai · Central", path="CMKL/Porjai-Thai-voice-dataset-central",
            hours="large", clips="100K+", license="CC BY-SA 4.0", streaming_only=True, text_field="sentence",
            language="Thai", lang_code="th",
            description="CMKL's Porjai corpus, standard central Thai. The largest and most general of the set.",
        ),
        DatasetSpec(
            id="porjai_korat", name="Porjai Thai · Korat (Isan)", path="CMKL/Porjai-Thai-voice-dataset-korat",
            hours="medium", clips="10K+", license="CC BY-SA 4.0", streaming_only=True, text_field="sentence",
            language="Thai", lang_code="th",
            description="North-eastern (Korat/Isan) dialect. Carries both the dialect transcript and a standard Thai one.",
        ),
        DatasetSpec(
            id="porjai_khummuang", name="Porjai Thai · Kham Mueang (Northern)", path="CMKL/Porjai-Thai-voice-dataset-khummuang",
            hours="medium", clips="10K+", license="CC BY-NC-SA 4.0 (non-commercial)", streaming_only=True, text_field="sentence",
            language="Thai", lang_code="th",
            description="Northern Thai / Lanna dialect. Non-commercial licence.",
        ),
        DatasetSpec(
            id="porjai_pattani", name="Porjai Thai · Pattani (Southern)", path="CMKL/Porjai-Thai-voice-dataset-pattani",
            hours="medium", clips="10K+", license="CC BY-NC-SA 4.0 (non-commercial)", streaming_only=True, text_field="sentence",
            language="Thai", lang_code="th",
            description="Southern/Pattani Malay-influenced dialect. Non-commercial licence.",
        ),
        DatasetSpec(
            id="thai_cv17", name="Thai Common Voice 17 (130k)", path="Porameht/processed-cv-17-th-130k",
            hours="large", clips="130,000", license="CC0-1.0", streaming_only=True, text_field="sentence",
            language="Thai", lang_code="th",
            description="Cleaned Common Voice 17 Thai. Public domain and the best licensed starting point for Thai.",
        ),
        DatasetSpec(
            id="thai_voice_169k", name="Thai voice 169k", path="Porameht/processed-voice-th-169k",
            hours="large", clips="169,000", license="CC BY-SA 4.0", streaming_only=True, text_field="sentence",
            language="Thai", lang_code="th",
            description="Large mixed-source Thai speech corpus, cleaned and segmented.",
        ),
        DatasetSpec(
            id="thai_smarthome", name="Thai smart-home commands", path="Porameht/processed-smarthome-th",
            hours="small", clips="1K–10K", license="CC BY-SA 4.0", streaming_only=True, text_field="sentence",
            language="Thai", lang_code="th",
            description="Short spoken smart-home commands. Useful for command-and-control rather than general speech.",
        ),
        DatasetSpec(
            id="thai_common_voice_edited", name="Thai Common Voice (edited)", path="lunarlist/edited_common_voice",
            hours="medium", clips="10K+", license="MIT", streaming_only=True, text_field="text",
            language="Thai", lang_code="th",
            description="Re-segmented Common Voice Thai with per-clip durations.",
        ),
        DatasetSpec(
            id="thai_som_tts", name="PyThaiNLP som TTS", path="pythainlp/som_tts_dataset",
            hours="medium", clips="10K+", license="CC BY 4.0", streaming_only=True, text_field="text",
            language="Thai", lang_code="th",
            description="Single-speaker studio recordings built for TTS; clean audio that also trains ASR well.",
        ),
        DatasetSpec(
            id="thaimos", name="Typhoon ThaiMOS TTS annotation", path="typhoon-ai/thaimos-tts-annotation",
            hours="small", clips="1K–10K", license="see card", streaming_only=True, text_field="text",
            language="Thai", lang_code="th",
            description="Thai speech with MOS quality annotations. Small, and carries sound-quality labels.",
        ),
    ]
}


@lru_cache(maxsize=128)
def source_info(path: str, config: str | None = None,
                fallback_splits: tuple[str, ...] = ()) -> dict[str, Any]:
    """Get real split names and row limits without downloading the dataset.

    The viewer does not publish sizes for every dataset. In that case a limit is
    unknown, rather than guessed from the catalogue's approximate clip label.
    """
    headers = {"Accept": "application/json"}
    if get_hf_token():
        headers["Authorization"] = f"Bearer {get_hf_token()}"

    def query(endpoint: str) -> dict[str, Any]:
        url = f"https://datasets-server.huggingface.co/{endpoint}?" + urllib.parse.urlencode({"dataset": path})
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=8) as response:
            return json.load(response)

    rows: list[dict[str, Any]] = []
    try:
        rows = query("size").get("size", {}).get("splits", [])
    except (urllib.error.URLError, TimeoutError, ValueError):
        pass
    if not rows:
        try:
            rows = query("splits").get("splits", [])
        except (urllib.error.URLError, TimeoutError, ValueError):
            pass

    if config:
        rows = [r for r in rows if r.get("config") == config]
    elif rows:
        configs = {r.get("config") for r in rows}
        chosen = "default" if "default" in configs else rows[0].get("config")
        rows = [r for r in rows if r.get("config") == chosen]

    splits: dict[str, int | None] = {}
    for row in rows:
        name = row.get("split")
        if not isinstance(name, str) or not name:
            continue
        count = row.get("num_rows")
        splits[name] = count if isinstance(count, int) and count >= 0 else None
    if not splits:
        splits = {name: None for name in fallback_splits or ("train",)}
    return {"splits": [{"name": name, "max_clips": count} for name, count in splits.items()]}


def source_limit(info: dict[str, Any], split: str) -> int | None:
    counts = {item["name"]: item["max_clips"] for item in info["splits"]}
    if split == "both":
        train, test = counts.get("train"), counts.get("test")
        return train + test if train is not None and test is not None else None
    return counts.get(split)


def split_options(info: dict[str, Any]) -> list[str]:
    return [item["name"] for item in info["splits"]]


def catalogue() -> list[dict[str, Any]]:
    return [
        {**{k: v for k, v in spec.__dict__.items()}, "url": f"https://huggingface.co/datasets/{spec.path}"}
        for spec in CATALOG.values()
    ]


# ------------------------------------------------------------------ normalisation
def _first_key(row: dict[str, Any], keys) -> str | None:
    lower = {k.lower(): k for k in row}
    for k in keys:
        if k in lower:
            return lower[k]
    return None


def find_text(row: dict[str, Any], hint: str | None = None) -> str:
    """Transcript from a flat column, or from a nested json/metadata dict (WebDataset)."""
    if hint and hint in row and isinstance(row[hint], str):
        return row[hint]
    k = _first_key(row, TEXT_KEYS)
    if k and isinstance(row[k], str):
        return row[k]
    for key in ("json", "metadata", "meta"):
        blob = row.get(key)
        if isinstance(blob, (bytes, str)):
            import json as _json

            try:
                blob = _json.loads(blob)
            except Exception:
                blob = None
        if isinstance(blob, dict):
            if hint and isinstance(blob.get(hint), str):
                return blob[hint]
            kk = _first_key(blob, TEXT_KEYS)
            if kk:
                return str(blob[kk])
    return ""


def find_audio(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return {'array': np.ndarray, 'sampling_rate': int} or {'bytes': ..., 'path': ...}."""
    k = _first_key(row, AUDIO_KEYS)
    if k is None:
        return None
    v = row[k]
    if isinstance(v, dict) and ("array" in v or "bytes" in v or "path" in v):
        return v
    if isinstance(v, (bytes, bytearray)):
        return {"bytes": bytes(v), "path": None}
    if isinstance(v, str):
        return {"bytes": None, "path": v}
    return None


def find_duration(row: dict[str, Any]) -> float | None:
    for key in ("duration", "length", "seconds"):
        if isinstance(row.get(key), (int, float)):
            return float(row[key])
    blob = row.get("json")
    if isinstance(blob, (bytes, str)):
        import json as _json

        try:
            blob = _json.loads(blob)
        except Exception:
            blob = None
    if isinstance(blob, dict) and isinstance(blob.get("duration"), (int, float)):
        return float(blob["duration"])
    return None


def normalize(row: dict[str, Any], text_hint: str | None = None) -> dict[str, Any]:
    return {"audio": find_audio(row), "text": find_text(row, text_hint), "duration": find_duration(row)}


# ------------------------------------------------------------------ loading
def _cast_audio(ds, how: str):
    from datasets import Audio, Value

    feats = getattr(ds, "features", None) or {}
    for name, feat in list(feats.items()):
        if isinstance(feat, Audio):
            try:
                ds = ds.cast_column(name, Audio(decode=False) if how == "audio" else Value("binary"))
            except Exception:
                pass
    return ds


def _undecode_audio(ds):
    """datasets>=5 routes every Audio column through torchcodec, even with decode=False. This app
    decodes with soundfile/librosa (+ffmpeg) instead, so Audio columns are cast to raw binary."""
    from datasets import Audio, Value

    feats = getattr(ds, "features", None) or {}
    for name, feat in list(feats.items()):
        if isinstance(feat, Audio):
            try:
                ds = ds.cast_column(name, Value("binary"))
            except Exception:
                pass
    return ds


def _load_sync(path: str, *, config: str | None, data_dir: str | None, split: str, streaming: bool):
    from datasets import load_dataset

    kw: dict[str, Any] = {"split": split, "streaming": streaming}
    if config:
        kw["name"] = config
    if data_dir:
        kw["data_dir"] = data_dir
    if get_hf_token():
        kw["token"] = get_hf_token()
    return _undecode_audio(load_dataset(path, **kw))


def _stream_sync(path: str, *, config: str | None, data_dir: str | None, split: str):
    """Iterator over raw rows, choosing the audio cast that this storage actually supports.

    Parquet keeps audio as struct<bytes,path> (needs Audio(decode=False)); WebDataset keeps raw
    bytes (needs Value("binary")). Both would otherwise be routed through torchcodec.
    """
    import itertools

    from datasets import load_dataset

    kw: dict[str, Any] = {"split": split, "streaming": True}
    if config:
        kw["name"] = config
    if data_dir:
        kw["data_dir"] = data_dir
    if get_hf_token():
        kw["token"] = get_hf_token()

    last: Exception | None = None
    for how in ("audio", "binary", "none"):
        try:
            ds = load_dataset(path, **kw)
            if how != "none":
                ds = _cast_audio(ds, how)
            it = iter(ds)
            first = next(it)
            return itertools.chain([first], it)
        except StopIteration:
            return iter(())
        except Exception as e:  # try the next strategy
            last = e
    raise last or RuntimeError("could not open dataset")


async def stream(path: str, *, config: str | None = None, data_dir: str | None = None, split: str = "train"):
    if split == "both":
        import itertools

        train = await asyncio.to_thread(_stream_sync, path, config=config, data_dir=data_dir, split="train")
        test = await asyncio.to_thread(_stream_sync, path, config=config, data_dir=data_dir, split="test")
        return itertools.chain(train, test)
    return await asyncio.to_thread(_stream_sync, path, config=config, data_dir=data_dir, split=split)


async def load(path: str, *, config: str | None = None, data_dir: str | None = None,
               split: str = "train", streaming: bool = True):
    return await asyncio.to_thread(_load_sync, path, config=config, data_dir=data_dir, split=split, streaming=streaming)


def parse_hf_ref(ref: str) -> tuple[str, str | None]:
    ref = ref.strip()
    m = re.match(r"^https?://huggingface\.co/datasets/([^/\s?#]+/[^/\s?#]+)(?:/.*)?$", ref)
    if m:
        ref = m.group(1)
    config = None
    if ":" in ref:
        ref, config = ref.split(":", 1)
    if not re.match(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$", ref):
        raise ValueError("expected 'owner/name' or a huggingface.co/datasets URL")
    return ref, config or None


def fetch_bytes(path: str) -> bytes:
    """Read an audio file that streaming handed us as a URI (hf://…, http(s)://…, or local)."""
    if path.startswith(("hf://", "http://", "https://", "s3://", "gs://")):
        import fsspec

        token = get_hf_token() or None
        opts = {"token": token} if path.startswith("hf://") and token else {}
        with fsspec.open(path, "rb", **opts) as f:
            return f.read()
    with open(path, "rb") as f:
        return f.read()


def audio_to_wav_bytes(audio: dict[str, Any], max_seconds: float | None = None) -> tuple[bytes, int, float]:
    """Any decoded/undecoded audio dict -> (wav bytes, sampling_rate, duration_seconds)."""
    import numpy as np
    import soundfile as sf

    arr = audio.get("array") if isinstance(audio, dict) else None
    sr = int(audio.get("sampling_rate") or 16000) if isinstance(audio, dict) else 16000
    if arr is None:
        raw = audio.get("bytes")
        path = audio.get("path")
        import librosa

        if not raw and path:
            raw = fetch_bytes(path)
        if raw:
            arr, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        else:
            raise ValueError("no audio payload")
    arr = np.asarray(arr, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if max_seconds:
        arr = arr[: int(max_seconds * sr)]
    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), sr, len(arr) / sr if sr else 0.0
