from typing import Callable

import torch
from sentence_transformers import CrossEncoder

from documentor.embed.embedder import resolve_device


class Reranker:
    def __init__(self, model_name: str, device: str = "auto", max_length: int = 512, batch_size: int = 16):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        print(f"Reranker: model={model_name} device={self.device}")
        # The library's default differs by model: sigmoid for bge, raw logits for MS MARCO. Passing it here gives
        # both models exactly one sigmoid, so scores are in 0-1 and comparable across questions.
        self.model = CrossEncoder(model_name, device=self.device, max_length=max_length,
                                  activation_fn=torch.nn.Sigmoid())
        if self.device == "cuda":
            self.model.half()

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict([(query, text) for text in texts], batch_size=self.batch_size,
                                    show_progress_bar=False)
        return [float(s) for s in scores]


class FakeReranker:
    model_name = "fake-reranker"

    def __init__(self, score_fn: Callable[[str, str], float]):
        self.score_fn = score_fn

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [self.score_fn(query, text) for text in texts]


_loaded: dict[tuple[str, str], Reranker] = {}


def get_reranker(cfg: dict, which: str = "full") -> Reranker:
    """`which` is "full" or "fast". Each model loads once per process."""
    if which not in ("full", "fast"):
        raise ValueError(f"unknown reranker: {which}")
    r = cfg["retrieval"]
    name = r["reranker_model"] if which == "full" else r["fast_reranker_model"]
    device = cfg["embedding"]["device"]
    key = (name, device)
    if key not in _loaded:
        _loaded[key] = Reranker(name, device, r["reranker_max_length"], r["reranker_batch_size"])
    return _loaded[key]
