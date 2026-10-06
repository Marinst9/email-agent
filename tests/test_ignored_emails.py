"""Ignored emails are labelled (not silently read), and rate-limited auto replies go to human review."""

from app.models.enums import EmailStatus, InboundStatus
from app.schemas.agent import Classification
from app.schemas.rules import ReplyTemplateRead
from tests.fakes import FakeGmailService, PipelineHarness, gmail_client, make_row

IGNORE_DRAFT = "АКЦИЈА: ИГНОРИРАЈ\nАКО ПРЕПРАЌАЊЕ ДО: НИКОЈ\nПОРАКА: "
OFFER = ReplyTemplateRead(id=1, name="Понуда", keywords=["понуда"], response="Благодарам, ќе ви се јавиме.")


def _label_calls(harness: PipelineHarness) -> list[dict[str, list[str]]]:
    return [body for _, body in harness.gmail_service.modified]


async def test_ai_ignored_email_is_labelled_and_left_unread_in_manual_mode() -> None:
    harness = PipelineHarness(make_row(auto_send=False), Classification(priority="LOW"), IGNORE_DRAFT)

    assert await harness.run() is InboundStatus.IGNORED

    (body,) = _label_calls(harness)
    label_id = next(label["id"] for label in harness.gmail_service.label_list if label["name"] == "AI-Ignored")
    assert body == {"addLabelIds": [label_id]}  # no removeLabelIds: stays UNREAD
    assert harness.log.records[-1].status is EmailStatus.IGNORED


async def test_spam_in_auto_mode_is_labelled_and_marked_read() -> None:
    harness = PipelineHarness(make_row(auto_send=True), Classification(category="SPAM"))

    assert await harness.run() is InboundStatus.IGNORED

    (body,) = _label_calls(harness)
    assert body["removeLabelIds"] == ["UNREAD"] and len(body["addLabelIds"]) == 1


async def test_blocked_sender_in_manual_mode_stays_unread() -> None:
    harness = PipelineHarness(make_row(auto_send=False), blocked=True)

    assert await harness.run() is InboundStatus.IGNORED
    (body,) = _label_calls(harness)
    assert "removeLabelIds" not in body
    assert harness.ai.calls == []  # blocked senders are filtered before any LLM call


async def test_rate_limited_auto_reply_goes_to_review_instead_of_being_ignored() -> None:
    # A calm template match would normally be auto-sent (it passes every review rule).
    harness = PipelineHarness(
        make_row(auto_send=True), Classification(priority="LOW"), templates=[OFFER], rate_allowed=False
    )

    assert await harness.run() is InboundStatus.AWAITING_REVIEW

    row = harness.row
    assert row.needs_review and "лимитот" in (row.review_reason or "")
    assert row.response == "Благодарам, ќе ви се јавиме."
    assert harness.gmail_service.sent == [] and harness.gmail_service.modified == []


async def test_label_is_created_once_and_reused() -> None:
    service = FakeGmailService()
    gmail = gmail_client(service)
    await gmail.add_label("m1", "AI-Ignored", mark_read=False)
    await gmail.add_label("m2", "AI-Ignored", mark_read=False)
    assert [label["name"] for label in service.label_list].count("AI-Ignored") == 1
    assert service.modified[0][1] == service.modified[1][1]

    # An existing label is found by name instead of being created again.
    other = FakeGmailService()
    other.label_list.append({"id": "Label_9", "name": "AI-Ignored"})
    await gmail_client(other).add_label("m1", "AI-Ignored", mark_read=True)
    assert other.modified == [("m1", {"addLabelIds": ["Label_9"], "removeLabelIds": ["UNREAD"]})]
