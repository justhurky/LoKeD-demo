from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class OllamaError(RuntimeError):
    pass


@dataclass(slots=True)
class OllamaEmbeddingClient:
    host: str = "http://127.0.0.1:11434"
    model: str = "nomic-embed-text"
    timeout: int = 60

    def _url(self, path: str) -> str:
        return f"{self.host.rstrip('/')}{path}"

    def _post(self, path: str, payload: dict[str, Any], timeout: int | None = None) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._url(path),
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaError(f"Ollama embedding hiba: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama ervenytelen JSON valaszt adott embeddingre.") from exc

    def list_models(self, timeout: int = 5) -> list[str]:
        try:
            with urllib.request.urlopen(self._url("/api/tags"), timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []
        return [item.get("name", "") for item in data.get("models", []) if item.get("name")]

    def is_model_available(self) -> bool:
        names = self.list_models()
        return any(name == self.model or name.startswith(f"{self.model}:") for name in names)

    def embed(self, text: str) -> list[float]:
        text = text.strip()
        if not text:
            raise ValueError("Ures szoveg nem embeddingelheto.")

        try:
            data = self._post("/api/embed", {"model": self.model, "input": text})
            embeddings = data.get("embeddings") or []
            if embeddings:
                return [float(value) for value in embeddings[0]]
        except OllamaError:
            pass

        data = self._post("/api/embeddings", {"model": self.model, "prompt": text})
        embedding = data.get("embedding")
        if not embedding:
            raise OllamaError("Az Ollama nem adott vissza embedding vektort.")
        return [float(value) for value in embedding]


@dataclass(slots=True)
class DeterministicEmbeddingClient:
    """Small deterministic embedding client for tests and offline smoke checks."""

    dimensions: int = 64

    def embed(self, text: str) -> list[float]:
        import hashlib
        import math

        vector = [0.0] * self.dimensions
        for word in text.lower().split():
            digest = hashlib.sha256(word.encode("utf-8", errors="ignore")).digest()
            index = int.from_bytes(digest[:2], "big") % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
