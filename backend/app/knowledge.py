"""Learn tab: answers questions about ASR fine-tuning using only backend/knowledge-hub.

Two steps, so the model never sees the whole corpus at once and cannot wander outside it:
  select  - given titles and summaries, pick the files worth reading
  answer  - read only those and reply with citations
"""
import json
import re
from pathlib import Path
from typing import Any, AsyncIterator

from . import openrouter

HUB = Path(__file__).resolve().parent.parent / "knowledge-hub"
MAX_FILES = 3
MAX_CHARS = 26000

SELECT_PROMPT = """You are a librarian for a small handbook on fine-tuning speech-recognition models.
You get the user's question and an index of documents (file, title, summary). Pick the documents most
likely to answer it — prefer fewer and more specific. Reply with ONLY JSON:
{"files": ["<file>", ...], "reason": "<short>"} with at most %d files.
Questions may be in any language, including romanised forms. If the latest message is a follow-up or a
request to rephrase, translate or go deeper, pick documents for the PREVIOUS topic."""

ANSWER_PROMPT = """You are the Learn assistant inside ASR Studio, an app for fine-tuning speech-recognition
models. Answer using ONLY the documents below; they are your only source. Be concrete and practical:
give numbers and concrete settings where the documents do. Cite sources inline as [n] by document
number. Keep it under about 300 words unless more is asked for. If the documents do not cover the
question, say so and name the topic that is covered instead. Never invent figures.

LANGUAGE: reply in the language the user wrote in, whatever it is. Romanised requests such as
"myanmar lo pyaw" or "phasa thai" are language-switch requests about the previous question, not new
questions. Keep technical terms (LoRA, WER, CER, Whisper, RNN-Transducer, epoch, learning rate) in
English, and write everything else in the user's language and script."""


def index() -> list[dict[str, Any]]:
    f = HUB / "index.json"
    return json.loads(f.read_text()) if f.exists() else []


def _allowed(name: str) -> bool:
    return any(d["file"] == name for d in index()) and (HUB / name).is_file()


def read(name: str) -> str:
    if not _allowed(name):
        raise ValueError(f"'{name}' is not in the knowledge hub")
    return (HUB / name).read_text(encoding="utf-8", errors="ignore")[:MAX_CHARS]


def _keyword_fallback(question: str) -> list[str]:
    words = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 3}
    scored = []
    for d in index():
        hay = f"{d['title']} {d['summary']} {d['file']}".lower()
        scored.append((sum(1 for w in words if w in hay), d["file"]))
    scored.sort(reverse=True)
    return [f for s, f in scored[:2] if s > 0] or [index()[0]["file"]]


async def ask(question: str, history: list[dict[str, str]] | None = None,
              language: str | None = None) -> AsyncIterator[dict[str, Any]]:
    docs = index()
    if not docs:
        yield {"type": "error", "message": "the knowledge hub is empty"}
        return
    model = openrouter.get_model()

    yield {"type": "status", "stage": "selecting", "message": f"choosing documents with {model}"}
    listing = "\n".join(f"- {d['file']}: {d['title']} — {d['summary']}" for d in docs)
    hist = "".join(f"{m['role']}: {m['content'][:300]}\n" for m in (history or [])[-4:])
    try:
        sel = await openrouter.chat(
            [{"role": "system", "content": SELECT_PROMPT % MAX_FILES},
             {"role": "user", "content": (f"Previous conversation:\n{hist}\n" if hist else "")
                                         + f"Latest message: {question}\n\nIndex:\n{listing}"}],
            json_mode=True, temperature=0)
        files = [f for f in (sel.get("files") or []) if isinstance(f, str) and _allowed(f)][:MAX_FILES]
        reason = str(sel.get("reason", ""))
    except Exception as e:
        files, reason = [], f"selection failed ({e}); fell back to keywords"
    if not files:
        files = _keyword_fallback(question)

    chosen = [d for d in docs if d["file"] in files]
    yield {"type": "status", "stage": "reading", "message": "reading " + ", ".join(d["title"] for d in chosen),
           "files": files, "reason": reason}

    context = "\n\n".join(f"===== [{i + 1}] {d['title']} =====\n{read(d['file'])}" for i, d in enumerate(chosen))
    lang_rule = (f"\n\nOUTPUT LANGUAGE OVERRIDE: the user chose '{language}'. Answer in {language} "
                 "(native script) whatever language the question is in.") if language and language.lower() != "auto" else ""
    msgs = [{"role": "system", "content": ANSWER_PROMPT + lang_rule + "\n\nDOCUMENTS:\n" + context}]
    for m in (history or [])[-6:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": question})

    yield {"type": "status", "stage": "answering", "message": f"answering with {model}"}
    try:
        answer = await openrouter.chat(msgs, temperature=0.2)
    except Exception as e:
        yield {"type": "error", "message": f"answer failed: {e}"}
        return
    yield {"type": "done", "answer": str(answer),
           "sources": [{"n": i + 1, "title": d["title"], "file": d["file"]} for i, d in enumerate(chosen)]}
