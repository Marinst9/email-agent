"""Knowledge base (RAG): document storage, background embedding and hybrid search in PostgreSQL.

Upload (web request): extract text per page, chunk it, store chunks with embedding_status "pending".
Embedding (Celery worker): `embed_pending` fills `embedding` for pending chunks in batches.
Search (worker, while drafting): one SQL query runs pgvector cosine search and full-text search over the
generated `search_vector` column, then merges both rankings with Reciprocal Rank Fusion. Chunks that
are still pending are found by full-text search only.
"""

import asyncio
import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pypdf import PdfReader
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import KnowledgeDocument
from app.models.enums import EmbeddingStatus
from app.models.knowledge import FTS_CONFIG
from app.schemas.agent import RetrievedDoc
from app.services.chunking import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS, chunk_pages
from app.services.embeddings import EmbeddingProvider
from app.services.retrieval import RRF_K, search_terms, to_tsquery_text

logger = logging.getLogger(__name__)

Page = tuple[int | None, str]


def _pdf_pages(data: bytes) -> list[Page]:
    reader = PdfReader(io.BytesIO(data))
    return [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, start=1)]


async def extract_pages(filename: str, data: bytes) -> list[Page]:
    """(page number, text) pairs: 1-based pages for PDFs, a single (None, text) for anything else."""
    if filename.lower().endswith(".pdf"):
        return await asyncio.to_thread(_pdf_pages, data)
    return [(None, data.decode("utf-8", errors="ignore"))]


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"


def hybrid_search_sql(*, semantic: bool, lexical: bool) -> str:
    """SQL for hybrid search. Each side keeps its own top `:candidates`; RRF uses rank, not raw scores."""
    ctes: list[str] = []
    if semantic:
        ctes.append(
            """semantic AS (
    SELECT id,
           1 - (embedding <=> CAST(:query_vector AS vector)) AS vector_similarity,
           row_number() OVER (ORDER BY embedding <=> CAST(:query_vector AS vector), id) AS rnk
    FROM knowledge_document
    WHERE user_email = :user_email AND embedding IS NOT NULL
    ORDER BY embedding <=> CAST(:query_vector AS vector), id
    LIMIT :candidates
)"""
        )
    if lexical:
        ctes.append(
            f"""lexical AS (
    SELECT d.id,
           ts_rank_cd(d.search_vector, q.query) AS text_rank,
           row_number() OVER (ORDER BY ts_rank_cd(d.search_vector, q.query) DESC, d.id) AS rnk
    FROM knowledge_document AS d, to_tsquery('{FTS_CONFIG}', :tsquery) AS q(query)
    WHERE d.user_email = :user_email AND d.search_vector @@ q.query
    ORDER BY text_rank DESC, d.id
    LIMIT :candidates
)"""
        )
    if semantic and lexical:
        fused = """fused AS (
    SELECT coalesce(s.id, l.id) AS id, s.vector_similarity, l.text_rank,
           coalesce(1.0 / (:rrf_k + s.rnk), 0) + coalesce(1.0 / (:rrf_k + l.rnk), 0) AS score
    FROM semantic AS s FULL OUTER JOIN lexical AS l ON s.id = l.id
)"""
    elif semantic:
        fused = (
            "fused AS (SELECT id, vector_similarity, NULL::real AS text_rank, "
            "1.0 / (:rrf_k + rnk) AS score FROM semantic)"
        )
    else:
        fused = (
            "fused AS (SELECT id, NULL::float8 AS vector_similarity, text_rank, "
            "1.0 / (:rrf_k + rnk) AS score FROM lexical)"
        )
    return f"""WITH {", ".join([*ctes, fused])}
SELECT d.id, d.content, d.filename, d.chunk_index, d.page, f.score, f.vector_similarity, f.text_rank
FROM fused AS f JOIN knowledge_document AS d ON d.id = f.id
ORDER BY f.score DESC, d.id
LIMIT :limit"""


@dataclass(frozen=True)
class DocumentSummary:
    filename: str
    chunks: int
    pending: int


class KnowledgeService:
    def __init__(
        self,
        session: AsyncSession,
        embedder: EmbeddingProvider | None = None,
        *,
        candidates: int = 20,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    ) -> None:
        self._session = session
        self._embedder = embedder
        self._candidates = candidates
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens

    async def add_document(self, user_email: str, filename: str, pages: Sequence[Page]) -> int:
        """Replace any previous version of `filename` with fresh pending chunks. Returns the chunk count."""
        await self._session.execute(
            delete(KnowledgeDocument).where(
                KnowledgeDocument.user_email == user_email, KnowledgeDocument.filename == filename
            )
        )
        chunks = chunk_pages(pages, self._max_tokens, self._overlap_tokens)
        self._session.add_all(
            KnowledgeDocument(
                user_email=user_email,
                filename=filename,
                chunk_index=chunk.chunk_index,
                page=chunk.page,
                token_count=chunk.token_count,
                content=chunk.text,
                embedding_status=EmbeddingStatus.PENDING.value,
            )
            for chunk in chunks
        )
        await self._session.commit()
        return len(chunks)

    async def embed_pending(
        self,
        embedder: EmbeddingProvider,
        user_email: str | None = None,
        *,
        batch: int = 32,
        max_chunks: int | None = None,
    ) -> int:
        """Embed pending chunks batch by batch until none are left (or `max_chunks` were done). Returns the count.

        Rows are locked with SKIP LOCKED, so concurrent workers split the work instead of duplicating it,
        and a chunk deleted or re-uploaded meanwhile is simply not found.
        """
        done = 0
        while max_chunks is None or done < max_chunks:
            query = (
                select(KnowledgeDocument.id, KnowledgeDocument.content)
                .where(KnowledgeDocument.embedding_status == EmbeddingStatus.PENDING.value)
                .order_by(KnowledgeDocument.id)
                .limit(batch)
                .with_for_update(skip_locked=True)
            )
            if user_email is not None:
                query = query.where(KnowledgeDocument.user_email == user_email)
            rows = (await self._session.execute(query)).all()
            if not rows:
                await self._session.commit()
                return done
            vectors = await embedder.embed_documents([content or "" for _, content in rows])
            for (chunk_id, _), vector in zip(rows, vectors, strict=True):
                await self._session.execute(
                    update(KnowledgeDocument)
                    .where(KnowledgeDocument.id == chunk_id)
                    .values(
                        embedding=vector,
                        embedding_status=EmbeddingStatus.READY.value,
                        embedding_model=embedder.model_name,
                    )
                )
            await self._session.commit()
            done += len(rows)
        return done

    async def search(self, user_email: str, query: str, limit: int = 3) -> list[RetrievedDoc]:
        """Hybrid (vector + full-text) search with Reciprocal Rank Fusion, entirely in SQL."""
        terms = search_terms(query)
        semantic = self._embedder is not None
        if not semantic and not terms:
            return []
        params: dict[str, Any] = {
            "user_email": user_email,
            "candidates": max(self._candidates, limit),
            "rrf_k": RRF_K,
            "limit": limit,
        }
        if semantic:
            assert self._embedder is not None
            params["query_vector"] = vector_literal(await self._embedder.embed_query(query))
        if terms:
            params["tsquery"] = to_tsquery_text(terms)
        result = await self._session.execute(text(hybrid_search_sql(semantic=semantic, lexical=bool(terms))), params)
        return [
            RetrievedDoc(
                chunk_id=row.id,
                content=row.content or "",
                filename=row.filename,
                chunk_index=row.chunk_index,
                page=row.page,
                score=float(row.score),
                vector_similarity=None if row.vector_similarity is None else float(row.vector_similarity),
                text_rank=None if row.text_rank is None else float(row.text_rank),
            )
            for row in result
        ]

    async def list_documents(self, user_email: str) -> list[DocumentSummary]:
        pending = func.count().filter(KnowledgeDocument.embedding_status == EmbeddingStatus.PENDING.value)
        result = await self._session.execute(
            select(KnowledgeDocument.filename, func.count(), pending)
            .where(KnowledgeDocument.user_email == user_email, KnowledgeDocument.filename.is_not(None))
            .group_by(KnowledgeDocument.filename)
            .order_by(KnowledgeDocument.filename)
        )
        return [DocumentSummary(name, total, waiting) for name, total, waiting in result if name]

    async def delete_document(self, user_email: str, filename: str) -> None:
        await self._session.execute(
            delete(KnowledgeDocument).where(
                KnowledgeDocument.user_email == user_email, KnowledgeDocument.filename == filename
            )
        )
        await self._session.commit()
