"""Split documents into retrieval chunks on paragraph and sentence boundaries.

Sizes are measured in approximate tokens (words and punctuation marks), which keeps chunking free of
any model-specific tokenizer. Defaults stay well under the 512-token input limit of multilingual-e5
even though its subword tokenizer produces more tokens than this count for Macedonian text.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

DEFAULT_MAX_TOKENS = 200
DEFAULT_OVERLAP_TOKENS = 40

_TOKEN = re.compile(r"\w+|[^\w\s]")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_HEADING = re.compile(r"^#{1,6}\s+\S")


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_index: int
    page: int | None
    token_count: int


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


def _units(paragraph: str, max_tokens: int) -> list[str]:
    """Sentences of a paragraph (each line separately, so list items stay intact); long ones split by words."""
    units: list[str] = []
    for line in paragraph.splitlines():
        for sentence in _SENTENCE_END.split(line.strip()):
            if not sentence:
                continue
            if count_tokens(sentence) <= max_tokens:
                units.append(sentence)
                continue
            words: list[str] = []
            for word in sentence.split():
                if words and count_tokens(" ".join([*words, word])) > max_tokens:
                    units.append(" ".join(words))
                    words = []
                words.append(word)
            if words:
                units.append(" ".join(words))
    return units


def _chunk_page(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    heading = ""

    def emit() -> None:
        body = "\n".join(current)
        # Repeat the section heading so a chunk from the middle of a section still says what it is about.
        chunks.append(body if not heading or body.startswith(heading) else f"{heading}\n{body}")

    for paragraph in _PARAGRAPH_BREAK.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if _HEADING.match(paragraph):
            if current:
                emit()
                current = []
            heading = paragraph.splitlines()[0]
        budget = max_tokens - count_tokens(heading)
        for unit in _units(paragraph, budget):
            if current and count_tokens("\n".join([*current, unit])) > budget:
                emit()
                # Start the next chunk with the trailing sentences of this one, up to the overlap budget.
                carried: list[str] = []
                for previous in reversed(current):
                    if count_tokens("\n".join([previous, *carried])) > overlap_tokens:
                        break
                    carried.insert(0, previous)
                # Never let the overlap push a chunk over budget.
                fits = count_tokens("\n".join([*carried, unit])) <= budget
                current = carried if fits else []
            current.append(unit)
    if current:
        emit()
    return chunks


def chunk_pages(
    pages: Iterable[tuple[int | None, str]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk each page separately so every chunk has an exact page number (None for non-paginated text)."""
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")
    texts = [(page, chunk) for page, text in pages for chunk in _chunk_page(text, max_tokens, overlap_tokens)]
    return [Chunk(text, i, page, count_tokens(text)) for i, (page, text) in enumerate(texts)]


def chunk_text(
    text: str, max_tokens: int = DEFAULT_MAX_TOKENS, overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
) -> list[Chunk]:
    return chunk_pages([(None, text)], max_tokens, overlap_tokens)
