"""Hybrid search against a real PostgreSQL with pgvector. Skipped unless TEST_DATABASE_URL is set.

The database must already be migrated (`alembic upgrade head`); CI does this against pgvector/pgvector:pg16.
Embeddings come from a deterministic fake, so the test checks the SQL, not the embedding model.
"""

import asyncio
import hashlib
import math
import os
import sys
from collections.abc import AsyncIterator, Sequence

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.services.embeddings import EMBEDDING_DIMENSION
from app.services.knowledge import KnowledgeService

DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL is not set")

USER, OTHER = "pg-test@lumenprint.mk", "pg-test-other@lumenprint.mk"
TOPICS = ("shipping", "price", "payment", "рекламации")


class TopicEmbedder:
    """Vectors from topic keywords: texts about the same topic are close, others are not."""

    model_name = "topic-fake"
    dimension = EMBEDDING_DIMENSION

    def _vector(self, text_: str) -> list[float]:
        lowered = text_.lower()
        vector = [0.0] * EMBEDDING_DIMENSION
        for i, topic in enumerate(TOPICS):
            vector[i] = float(lowered.count(topic))
        # A small text-specific component keeps vectors distinct.
        vector[len(TOPICS) + int(hashlib.sha256(lowered.encode()).hexdigest(), 16) % 100] = 0.1
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def embed_query(self, text_: str) -> list[float]:
        return self._vector(text_)


@pytest.fixture(scope="module")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    # psycopg's async driver needs a selector loop on Windows (asyncpg, used in CI, does not care).
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()  # type: ignore[attr-defined,unused-ignore]
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(DATABASE_URL)
    async with async_sessionmaker(engine, expire_on_commit=False)() as db:
        await db.execute(text("DELETE FROM knowledge_document WHERE user_email IN (:a, :b)"), {"a": USER, "b": OTHER})
        await db.commit()
        yield db
        await db.execute(text("DELETE FROM knowledge_document WHERE user_email IN (:a, :b)"), {"a": USER, "b": OTHER})
        await db.commit()
    await engine.dispose()


async def _upload(service: KnowledgeService) -> None:
    await service.add_document(
        USER, "shipping.md", [(None, "Shipping to Serbia via DHL costs 25 EUR. Shipping takes 3 days.")]
    )
    await service.add_document(USER, "price.md", [(None, "Price list: 100 business cards price 900 MKD.")])
    await service.add_document(USER, "payment.md", [(None, "Payment by card or bank transfer. No PayPal payment.")])
    await service.add_document(USER, "returns.md", [(1, "Рекламации се примаат во рок од 7 дена.")])
    await service.add_document(OTHER, "other.md", [(None, "Shipping to Serbia is free for this other user.")])


async def test_upload_embed_and_hybrid_search(session: AsyncSession) -> None:
    embedder = TopicEmbedder()
    service = KnowledgeService(session, embedder)
    await _upload(service)

    statuses = await session.execute(
        text("SELECT embedding_status, count(*) FROM knowledge_document WHERE user_email = :u GROUP BY 1"), {"u": USER}
    )
    assert dict(statuses.all()) == {"pending": 4}
    assert await service.embed_pending(embedder, USER) == 4
    assert await service.embed_pending(embedder, USER) == 0  # idempotent

    docs = await service.search(USER, "How much is shipping to Serbia?", limit=3)

    assert docs[0].filename == "shipping.md"
    assert docs[0].vector_similarity is not None and docs[0].vector_similarity > 0.9
    assert docs[0].text_rank is not None and docs[0].text_rank > 0
    assert docs[0].score == pytest.approx(2 / 61)  # first in both rankings
    assert all(d.filename != "other.md" for d in docs)  # other users' documents are never returned
    assert [d.score for d in docs] == sorted((d.score for d in docs), reverse=True)


async def test_pending_chunks_are_found_by_full_text_search(session: AsyncSession) -> None:
    service = KnowledgeService(session, TopicEmbedder())
    await _upload(service)  # nothing embedded yet

    docs = await service.search(USER, "Do you accept PayPal?", limit=3)

    assert [d.filename for d in docs] == ["payment.md"]
    assert docs[0].vector_similarity is None and docs[0].text_rank is not None


async def test_macedonian_full_text_search(session: AsyncSession) -> None:
    folds = await session.scalar(text("SELECT to_tsvector('simple', 'Рекламации') @@ 'рекламации'::tsquery"))
    if not folds:
        message = "this database's locale does not lower-case Cyrillic (see migration 0006)"
        if os.environ.get("REQUIRE_CYRILLIC_FTS"):
            pytest.fail(message)
        pytest.skip(message)
    service = KnowledgeService(session)  # full-text only
    await _upload(service)

    docs = await service.search(USER, "Имам рекламација за банерот", limit=3)

    assert [(d.filename, d.page) for d in docs] == [("returns.md", 1)]  # "рекламација" -> "реклама:*"
