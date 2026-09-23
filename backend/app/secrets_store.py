"""Encrypted-at-rest storage for tokens set from the UI (same scheme as system-one-playground)."""
import json
import os
import threading
from pathlib import Path

from cryptography.fernet import Fernet

from .config import DATA

SECRET_FILE = DATA / ".secret"
STORE_FILE = DATA / "secrets.enc.json"
PREFS_FILE = DATA / "prefs.json"
_lock = threading.Lock()


def _write_private(path: Path, data: bytes) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.chmod(path, 0o600)


def _fernet() -> Fernet:
    if not SECRET_FILE.exists():
        _write_private(SECRET_FILE, Fernet.generate_key())
    return Fernet(SECRET_FILE.read_bytes().strip())


def _read() -> dict[str, str]:
    try:
        return json.loads(STORE_FILE.read_text()) if STORE_FILE.exists() else {}
    except Exception:
        return {}


def set_secret(name: str, value: str) -> None:
    with _lock:
        store = _read()
        store[name] = _fernet().encrypt(value.encode()).decode()
        _write_private(STORE_FILE, json.dumps(store).encode())


def get_secret(name: str) -> str | None:
    with _lock:
        token = _read().get(name)
    if not token:
        return None
    try:
        return _fernet().decrypt(token.encode()).decode()
    except Exception:
        return None


def delete_secret(name: str) -> None:
    with _lock:
        store = _read()
        if store.pop(name, None) is not None:
            _write_private(STORE_FILE, json.dumps(store).encode())


def mask(value: str) -> str:
    return f"…{value[-4:]}" if len(value) >= 8 else "…"


def get_prefs() -> dict:
    try:
        return json.loads(PREFS_FILE.read_text()) if PREFS_FILE.exists() else {}
    except Exception:
        return {}


def set_prefs(**kv) -> dict:
    with _lock:
        p = get_prefs()
        p.update({k: v for k, v in kv.items() if v is not None})
        DATA.mkdir(parents=True, exist_ok=True)
        PREFS_FILE.write_text(json.dumps(p))
    return p
