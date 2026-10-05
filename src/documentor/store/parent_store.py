import json
from pathlib import Path


class ParentStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._parents: dict[str, dict] | None = None

    def save(self, parents: list[dict], merge: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = dict(self._load()) if merge else {}
        self._parents = existing | {p["parent_id"]: p for p in parents}
        self.path.write_text(json.dumps(self._parents, ensure_ascii=False))

    def _load(self) -> dict[str, dict]:
        if self._parents is None:
            self._parents = json.loads(self.path.read_text()) if self.path.exists() else {}
        return self._parents

    def get(self, parent_id: str) -> dict | None:
        return self._load().get(parent_id)

    def get_many(self, ids: list[str]) -> list[dict]:
        parents = self._load()
        return [parents[i] for i in ids if i in parents]
