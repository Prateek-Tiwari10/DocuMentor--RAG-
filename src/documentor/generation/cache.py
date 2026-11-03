import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

KEY_FIELDS = ("model", "temperature", "reasoning_effort", "max_completion_tokens", "include_reasoning")


def cache_key(params: dict, messages: list[dict]) -> str:
    payload = {f: params.get(f) for f in KEY_FIELDS} | {"messages": messages}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ResponseCache:
    def __init__(self, path):
        self.root = Path(path)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str):
        from documentor.generation.llm import LLMResult

        path = self._path(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        data.pop("created_at", None)
        return replace(LLMResult(**data), cache_hit=True)

    def put(self, key: str, result) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(result) | {"cache_hit": False, "created_at": datetime.now().isoformat(timespec="seconds")}
        path.write_text(json.dumps(data, indent=2))
