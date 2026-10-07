"""Text embeddings for the knowledge base, behind a small provider protocol.

`EMBEDDING_DIMENSION` is fixed by the `knowledge_document.embedding vector(1024)` column. Both providers
produce 1024-dimensional vectors, so switching provider needs re-embedding but no migration.
"""

import asyncio
import math
from collections.abc import Sequence
from typing import Any, Protocol

from app.core.config import Settings

EMBEDDING_DIMENSION = 1024
DEFAULT_LOCAL_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_VOYAGE_MODEL = "voyage-4"


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class LocalEmbeddingProvider:
    """sentence-transformers model run in-process (CPU by default). Loaded lazily on first use.

    E5 models expect "query: " / "passage: " prefixes; set both to "" for models that do not.
    Needs the optional dependencies in requirements-local-embeddings.txt.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL,
        *,
        dimension: int = EMBEDDING_DIMENSION,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        batch_size: int = 16,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._batch_size = batch_size
        self._model: Any = None
        self._lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy optional import

            self._model = SentenceTransformer(self._model_name)
            actual = self._model.get_embedding_dimension()
            if actual != self._dimension:
                raise ValueError(f"{self._model_name} produces {actual}-dim vectors, expected {self._dimension}")
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(texts, batch_size=self._batch_size, normalize_embeddings=True)
        return [[float(x) for x in vector] for vector in vectors]

    async def _run(self, texts: list[str]) -> list[list[float]]:
        # One model per process; encoding is CPU-bound, so keep it off the event loop and serialized.
        async with self._lock:
            return await asyncio.to_thread(self._encode, texts)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._run([self._passage_prefix + text for text in texts]) if texts else []

    async def embed_query(self, text: str) -> list[float]:
        return (await self._run([self._query_prefix + text]))[0]


class VoyageEmbeddingProvider:
    """Voyage AI embeddings API (multilingual `voyage-4` family), with query/document input types."""

    def __init__(
        self, api_key: str, model_name: str = DEFAULT_VOYAGE_MODEL, *, dimension: int = EMBEDDING_DIMENSION
    ) -> None:
        import voyageai

        self._client = voyageai.AsyncClient(api_key=api_key)  # type: ignore[attr-defined]
        self._model_name = model_name
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    async def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 128):  # API batch limit
            result = await self._client.embed(
                list(texts[start : start + 128]),
                model=self._model_name,
                input_type=input_type,
                output_dimension=self._dimension,
            )
            vectors.extend([float(x) for x in vector] for vector in result.embeddings)
        return vectors

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, "document") if texts else []

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], "query"))[0]


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "voyage":
        if settings.voyage_api_key is None:
            raise ValueError("EMBEDDING_PROVIDER=voyage needs VOYAGE_API_KEY")
        return VoyageEmbeddingProvider(settings.voyage_api_key.get_secret_value(), settings.voyage_model)
    return LocalEmbeddingProvider(settings.local_embedding_model)
