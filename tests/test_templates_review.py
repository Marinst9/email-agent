"""Template replies are classified first and pass through the same review rules as AI drafts."""

import pytest

from app.models.enums import InboundStatus
from app.schemas.agent import Classification
from app.schemas.rules import ReplyTemplateRead
from app.services.ai_agents import CLASSIFICATION_PROMPT
from app.services.rules import DEFAULT_TEMPLATES, find_matching_template
from tests.fakes import PipelineHarness, make_row

MEETING = ReplyTemplateRead(id=1, name="Консултации", keywords=["состанок"], response="Слободен сум во среда.")


def _harness(classification: Classification, auto_send: bool = True) -> PipelineHarness:
    row = make_row(subject="Барам состанок", body="Може ли состанок утре?", auto_send=auto_send)
    return PipelineHarness(row, classification, draft_text="unused", templates=[MEETING])


@pytest.mark.parametrize(
    "classification",
    [
        Classification(category="COMPLAINT", priority="MEDIUM"),
        Classification(category="URGENT_HUMAN", priority="MEDIUM"),
        Classification(category="INQUIRY", priority="HIGH"),
    ],
)
async def test_template_reply_to_complaint_or_urgent_email_waits_for_review(classification: Classification) -> None:
    harness = _harness(classification)

    assert await harness.run() is InboundStatus.AWAITING_REVIEW

    row = harness.row
    assert row.needs_review and row.review_reason
    assert row.source == "Шаблон: Консултации" and row.response == MEETING.response
    assert (row.category, row.priority) == (classification.category, classification.priority)
    assert harness.gmail_service.sent == []


async def test_calm_template_match_is_auto_sent_without_drafting() -> None:
    harness = _harness(Classification(category="INQUIRY", priority="LOW"))

    assert await harness.run() is InboundStatus.SENT

    assert len(harness.gmail_service.sent) == 1
    # Only the classification call was made; the template replaced the drafting call.
    assert [call["system"] for call in harness.ai.calls] == [CLASSIFICATION_PROMPT]


async def test_spam_is_ignored_even_when_a_template_matches() -> None:
    harness = _harness(Classification(category="SPAM"))

    assert await harness.run() is InboundStatus.IGNORED
    assert harness.gmail_service.sent == []


@pytest.mark.parametrize(
    "text",
    [
        "URGENT: please help, the invoice is wrong",
        "Итно ми треба помош",
        "Request for a refund",
        "Имам барање за поврат на пари",
    ],
)
def test_default_templates_do_not_match_generic_words(text: str) -> None:
    templates = [
        ReplyTemplateRead(id=i, name=t.name, keywords=t.keywords, response=t.response)
        for i, t in enumerate(DEFAULT_TEMPLATES)
    ]
    assert find_matching_template(text, "", templates) is None
