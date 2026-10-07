"""In-memory stand-in for `KnowledgeService.search`, so the evals run without PostgreSQL.

It uses the production chunker, embedding provider, query terms and Reciprocal Rank Fusion. Two parts
are approximations of what Postgres does:
- vector search is exact cosine similarity (pgvector's HNSW index is approximate, but exact on a table
  this small);
- full-text ranking orders chunks by distinct matched query terms, then total matches. Postgres uses
  ts_rank_cd over to_tsvector('simple', ...), which also weighs term proximity.
"""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.schemas.agent import RetrievedDoc
from app.services.chunking import chunk_text
from app.services.embeddings import EmbeddingProvider, cosine_similarity
from app.services.retrieval import RRF_K, reciprocal_rank_fusion, search_terms

Mode = Literal["hybrid", "vector", "lexical"]
_WORD = re.compile(r"[^\W_]+")


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: int
    filename: str
    chunk_index: int
    text: str
    tokens: tuple[str, ...]


class EmbeddingCache:
    """Embeddings on disk keyed by model, input type and text, so reruns do not re-encode."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._data: dict[str, list[float]] = {}
        if path is not None and path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))
        self._dirty = False

    @staticmethod
    def key(model: str, kind: str, text: str) -> str:
        return hashlib.sha256(f"{model}\0{kind}\0{text}".encode()).hexdigest()

    async def documents(self, embedder: EmbeddingProvider, texts: Sequence[str]) -> list[list[float]]:
        keys = [self.key(embedder.model_name, "document", t) for t in texts]
        missing = [t for t, k in zip(texts, keys, strict=True) if k not in self._data]
        for text, vector in zip(missing, await embedder.embed_documents(missing), strict=True):
            self._data[self.key(embedder.model_name, "document", text)] = vector
            self._dirty = True
        return [self._data[k] for k in keys]

    async def query(self, embedder: EmbeddingProvider, text: str) -> list[float]:
        key = self.key(embedder.model_name, "query", text)
        if key not in self._data:
            self._data[key] = await embedder.embed_query(text)
            self._dirty = True
        return self._data[key]

    def save(self) -> None:
        if self._path is not None and self._dirty:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data), encoding="utf-8")
            self._dirty = False


class InMemoryHybridIndex:
    def __init__(
        self,
        documents: dict[str, str],
        embedder: EmbeddingProvider | None,
        cache: EmbeddingCache,
        *,
        candidates: int = 20,
        mode: Mode = "hybrid",
    ) -> None:
        self._embedder = embedder
        self._cache = cache
        self._candidates = candidates
        self.mode: Mode = mode if embedder is not None else "lexical"
        self.chunks: list[IndexedChunk] = []
        for filename, text in documents.items():
            for chunk in chunk_text(text):
                tokens = tuple(_WORD.findall(chunk.text.lower()))
                self.chunks.append(IndexedChunk(len(self.chunks) + 1, filename, chunk.chunk_index, chunk.text, tokens))
        self._vectors: dict[int, list[float]] = {}

    async def build(self) -> "InMemoryHybridIndex":
        if self._embedder is not None:
            vectors = await self._cache.documents(self._embedder, [c.text for c in self.chunks])
            self._vectors = {c.chunk_id: v for c, v in zip(self.chunks, vectors, strict=True)}
            self._cache.save()
        return self

    def with_mode(self, mode: Mode) -> "InMemoryHybridIndex":
        clone = InMemoryHybridIndex.__new__(InMemoryHybridIndex)
        clone.__dict__.update(self.__dict__)
        clone.mode = mode
        return clone

    def _lexical(self, query: str) -> list[tuple[int, float]]:
        terms = search_terms(query)
        scored: list[tuple[int, float]] = []
        for chunk in self.chunks:
            distinct = total = 0
            for term in terms:
                prefix = term.removesuffix(":*")
                hits = sum(1 for t in chunk.tokens if (t.startswith(prefix) if term.endswith(":*") else t == term))
                distinct += hits > 0
                total += hits
            if distinct:
                scored.append((chunk.chunk_id, distinct + total / 100))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[: self._candidates]

    async def _semantic(self, query: str) -> list[tuple[int, float]]:
        assert self._embedder is not None
        query_vector = await self._cache.query(self._embedder, query)
        scored = [(cid, cosine_similarity(query_vector, vector)) for cid, vector in self._vectors.items()]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[: self._candidates]

    async def ranked(self, query: str, limit: int) -> list[RetrievedDoc]:
        semantic = await self._semantic(query) if self.mode in ("hybrid", "vector") else []
        lexical = self._lexical(query) if self.mode in ("hybrid", "lexical") else []
        rankings = {"semantic": [cid for cid, _ in semantic], "lexical": [cid for cid, _ in lexical]}
        similarity, text_rank = dict(semantic), dict(lexical)
        by_id = {c.chunk_id: c for c in self.chunks}
        return [
            RetrievedDoc(
                chunk_id=cid,
                content=by_id[cid].text,
                filename=by_id[cid].filename,
                chunk_index=by_id[cid].chunk_index,
                score=score,
                vector_similarity=similarity.get(cid),
                text_rank=text_rank.get(cid),
            )
            for cid, score in reciprocal_rank_fusion(rankings, RRF_K)[:limit]
        ]

    async def search(self, user_email: str, query: str, limit: int = 3) -> list[RetrievedDoc]:
        """`DocumentSearcher` interface used by the orchestrator."""
        docs = await self.ranked(query, limit)
        self._cache.save()
        return docs
