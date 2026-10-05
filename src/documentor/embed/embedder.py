import torch
from sentence_transformers import SentenceTransformer


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class Embedder:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 16):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        print(f"Embedder: model={model_name} device={self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        if self.device == "cuda":
            self.model.half()  # bge-m3 in fp32 is tight on a 4 GB GPU

    def _encode(self, texts: list[str], progress: bool) -> list[list[float]]:
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                    show_progress_bar=progress)
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, progress=True)

    def embed_query(self, text: str) -> list[float]:
        # bge-m3 needs no query prefix; models that do would add it here.
        return self._encode([text], progress=False)[0]
