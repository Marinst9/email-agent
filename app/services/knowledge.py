"""Knowledge base (RAG) storage and keyword retrieval over document chunks."""

import asyncio
import io

from pypdf import PdfReader
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import KnowledgeDocument

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks: list[str] = []
    for start in range(0, len(text), size - overlap):
        chunk = text[start : start + size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def score_chunk(query_words: list[str], content: str) -> int:
    content_lower = content.lower()
    return sum(1 for word in query_words if word in content_lower)


def _pdf_to_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "".join((page.extract_text() or "") + "\n" for page in reader.pages)


async def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        return await asyncio.to_thread(_pdf_to_text, data)
    return data.decode("utf-8", errors="ignore")


class KnowledgeService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_document(self, user_email: str, filename: str, text: str) -> int:
        """Replace any previous version of `filename` with freshly chunked `text`. Returns the chunk count."""
        await self._session.execute(
            delete(KnowledgeDocument).where(
                KnowledgeDocument.user_email == user_email, KnowledgeDocument.filename == filename
            )
        )
        chunks = chunk_text(text)
        self._session.add_all(
            KnowledgeDocument(user_email=user_email, filename=filename, chunk_index=i, content=chunk)
            for i, chunk in enumerate(chunks)
        )
        await self._session.commit()
        return len(chunks)

    async def search(self, user_email: str, query: str, limit: int = 3) -> list[str]:
        query_words = query.lower().split()
        result = await self._session.scalars(
            select(KnowledgeDocument.content).where(KnowledgeDocument.user_email == user_email)
        )
        scored = [
            (score, content) for content in result if content and (score := score_chunk(query_words, content)) > 0
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [content for _, content in scored[:limit]]

    async def list_filenames(self, user_email: str) -> list[str]:
        result = await self._session.scalars(
            select(KnowledgeDocument.filename)
            .where(KnowledgeDocument.user_email == user_email, KnowledgeDocument.filename.is_not(None))
            .distinct()
            .order_by(KnowledgeDocument.filename)
        )
        return [name for name in result if name]

    async def delete_document(self, user_email: str, filename: str) -> None:
        await self._session.execute(
            delete(KnowledgeDocument).where(
                KnowledgeDocument.user_email == user_email, KnowledgeDocument.filename == filename
            )
        )
        await self._session.commit()
