"""OpenRouter client: the text model that powers the Learn tab."""
import json
import re
import time
from typing import Any

import httpx

from . import secrets_store

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-sonnet-5"
_models_cache: dict[str, Any] = {"at": 0.0, "data": []}


def get_key() -> str:
    import os

    return secrets_store.get_secret("openrouter_api_key") or os.environ.get("OPENROUTER_API_KEY", "")


def get_model() -> str:
    return secrets_store.get_prefs().get("openrouter_model") or DEFAULT_MODEL


def _extract_json(text: str) -> Any:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1:
        raise ValueError(f"no JSON object in model output: {text[:160]}")
    return json.loads(text[start : end + 1])


async def chat(messages: list[dict[str, str]], *, json_mode: bool = False, temperature: float = 0.2,
               max_tokens: int = 4000) -> Any:
    key = get_key()
    if not key:
        raise RuntimeError("no OpenRouter key: add one in Settings to use the Learn tab")
    model = get_model()
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "HTTP-Referer": "http://localhost:5175",
                     "X-Title": "ASR Studio"},
            json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                  **({"response_format": {"type": "json_object"}} if json_mode else {})},
        )
        if r.status_code >= 400:
            raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:200]}")
        body = r.json()
    content = body["choices"][0]["message"]["content"]
    _ = time.perf_counter() - t0
    return _extract_json(content) if json_mode else content


async def validate_key(key: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(f"{BASE_URL}/auth/key", headers={"Authorization": f"Bearer {key}"})
    if r.status_code == 401:
        raise ValueError("OpenRouter rejected this key (401)")
    r.raise_for_status()
    return r.json().get("data", {})


async def list_models(force: bool = False) -> list[dict[str, Any]]:
    if not force and _models_cache["data"] and time.time() - _models_cache["at"] < 600:
        return _models_cache["data"]
    headers = {"Authorization": f"Bearer {get_key()}"} if get_key() else {}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{BASE_URL}/models", headers=headers)
        r.raise_for_status()
    out = []
    for m in r.json().get("data", []):
        pr = m.get("pricing") or {}
        out.append({"id": m.get("id"), "name": m.get("name") or m.get("id"),
                    "context": m.get("context_length"),
                    "prompt_price": float(pr.get("prompt") or 0) * 1e6,
                    "completion_price": float(pr.get("completion") or 0) * 1e6})
    out.sort(key=lambda m: m["id"])
    _models_cache.update(at=time.time(), data=out)
    return out
