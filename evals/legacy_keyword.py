"""The keyword retrieval the app used before hybrid search, kept only as a baseline for the retrieval eval.

500-character chunks with 50 characters of overlap, ranked by how many whitespace-split query words
occur as substrings (stopwords and punctuation included), from the subject plus the first 200 body chars.
"""

from collections.abc import Iterable

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_chars(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    return [chunk for start in range(0, len(text), size - overlap) if (chunk := text[start : start + size]).strip()]


def rank_chunks(query: str, contents: Iterable[str], limit: int) -> list[str]:
    words = query.lower().split()
    scored = [(sum(1 for w in words if w in c.lower()), c) for c in contents]
    scored = [item for item in scored if item[0] > 0]
    scored.sort(key=lambda item: item[0], reverse=True)  # stable: ties keep document order
    return [content for _, content in scored[:limit]]


def legacy_query(subject: str, body: str) -> str:
    return f"{subject} {body[:200]}"
