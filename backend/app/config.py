from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    hf_token: str = ""
    # where materialised dataset subsets, checkpoints and logs go
    cache_dir: str = str(DATA / "cache")
    runs_dir: str = str(DATA / "runs")
    uploads_dir: str = str(DATA / "uploads")


settings = Settings()


def get_hf_token() -> str:
    from . import secrets_store

    return secrets_store.get_secret("hf_token") or settings.hf_token
