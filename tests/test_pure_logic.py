"""Unit tests for logic that needs no database or network."""

import pytest

from app.core.config import DatabaseSettings
from app.schemas.agent import Citation, Classification, DraftAction, DraftResult
from app.schemas.rules import BlockedWordForm, ReplyTemplateRead
from app.services.ai_agents import ReviewAgent, parse_classification, parse_draft_action
from app.services.rules import find_matching_template, is_blocked_sender


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("postgres://u:p@h:5432/db", "postgresql+asyncpg://u:p@h:5432/db"),
        ("postgresql://u:p@h/db", "postgresql+asyncpg://u:p@h/db"),
        ("postgresql+asyncpg://u:p@h/db", "postgresql+asyncpg://u:p@h/db"),
    ],
)
def test_database_url_uses_async_driver(raw: str, expected: str) -> None:
    assert DatabaseSettings(database_url=raw).database_url == expected


def test_template_keywords_are_split_and_blank_entries_dropped() -> None:
    template = ReplyTemplateRead.model_validate({"id": 1, "name": "x", "keywords": " a, ,b,", "response": "r"})
    assert template.keywords == ["a", "b"]


def test_find_matching_template_is_case_insensitive() -> None:
    templates = [
        ReplyTemplateRead(id=1, name="Meeting", keywords=["Meeting"], response="r1"),
        ReplyTemplateRead(id=2, name="Help", keywords=["urgent"], response="r2"),
    ]
    assert find_matching_template("URGENT question", "", templates) == templates[1]
    assert find_matching_template("hello", "nothing here", templates) is None


def test_blocked_sender_matching() -> None:
    assert is_blocked_sender("No-Reply <no-reply@example.com>", ["no-reply"])
    assert not is_blocked_sender("Ana <ana@example.com>", ["no-reply"])


def test_blocked_word_form_normalizes() -> None:
    assert BlockedWordForm(word="  Amazon ").word == "amazon"


def test_parse_classification_handles_code_fences_and_garbage() -> None:
    fenced = '```json\n{"category": "SPAM", "priority": "LOW", "language": "en", "sentiment": "negative"}\n```'
    assert parse_classification(fenced).category == "SPAM"
    assert parse_classification("not json") == Classification()


def test_parse_draft_action() -> None:
    assert parse_draft_action("АКЦИЈА: ОДГОВОР\nПОРАКА: hi") is DraftAction.REPLY
    assert parse_draft_action("АКЦИЈА: ПРЕПРАЌАЊЕ") is DraftAction.FORWARD
    assert parse_draft_action("something else") is DraftAction.IGNORE


CITED = [Citation(chunk_id=7, filename="pricing.md", snippet="Визит карти ...")]


def _draft(citations: list[Citation]) -> DraftResult:
    return DraftResult(raw="", action=DraftAction.REPLY, response_text="", citations=citations, reasoning="")


@pytest.mark.parametrize(
    ("classification", "citations", "needs_review"),
    [
        (Classification(category="COMPLAINT"), CITED, True),
        (Classification(priority="HIGH"), CITED, True),
        # A reply that cites nothing from the knowledge base is not grounded: a human checks it.
        (Classification(), [], True),
        (Classification(), CITED, False),
    ],
)
def test_review_agent(classification: Classification, citations: list[Citation], needs_review: bool) -> None:
    decision = ReviewAgent().execute(_draft(citations), classification)
    assert decision.needs_review is needs_review
    assert decision.auto_send is not needs_review
