from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "settings.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(rel: str) -> Path:
    return ROOT / rel


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "-").replace(".", "-")


def collection_name(strategy: str, model_name: str) -> str:
    return f"{strategy}__{model_slug(model_name)}"
