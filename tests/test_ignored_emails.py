"""Ignored emails are labelled (not silently read), only machine mail is ignored without review, and
rate-limited auto replies go to human review."""

import pytest

from app.models.enums import DraftAction, EmailStatus, InboundStatus
from app.schemas.agent import Classification, DraftResult
from app.schemas.email import GmailMessage
from app.schemas.rules import ReplyTemplateRead
from app.services.ai_agents import ReviewAgent, is_automated_email
from tests.fakes import FakeGmailService, FakeInbound, FakeLog, PipelineHarness, gmail_client, make_delivery, make_row

IGNORE_DRAFT = "АКЦИЈА: ИГНОРИРАЈ\nАКО ПРЕПРАЌАЊЕ ДО: НИКОЈ\nПОРАКА: "
OFFER = ReplyTemplateRead(id=1, name="Понуда", keywords=["понуда"], response="Благодарам, ќе ви се јавиме.")


def _label_calls(harness: PipelineHarness) -> list[dict[str, list[str]]]:
    return [body for _, body in harness.gmail_service.modified]


async def test_ai_ignored_email_is_labelled_and_left_unread_in_manual_mode() -> None:
    row = make_row(auto_send=False, sender="Shop <noreply@shop.mk>", body="Your order has shipped.")
    harness = PipelineHarness(row, Classification(priority="LOW"), IGNORE_DRAFT)

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


# --- Only clear machine mail is ignored without a human -------------------------------------------


@pytest.mark.parametrize(
    ("sender", "body", "automated"),
    [
        ("DHL Express <noreply@dhl.com>", "Your shipment has been delivered.", True),
        ("Google <no-reply@accounts.google.com>", "Security alert", True),
        ("LinkedIn <notifications-noreply@linkedin.com>", "You appeared in 12 searches.", True),
        ("Комора <bilten@komora.mk>", "Билтен бр. 42", True),
        ("Shop <hello@shop.mk>", "15% off this week. Click here to unsubscribe.", True),
        ("Хартија Плус <info@hartijaplus.mk>", "Попусти на хартија. За одјава кликнете тука.", True),
        ("Хартија Плус <smetki@hartijaplus.mk>", "Во прилог ви ја доставуваме фактурата бр. 2026-0815.", False),
        ("Procurement <procurement@alpha-pharma.com>", "We would like a quote for 20,000 catalogs.", False),
        ("Maya <maya.noreply.fan@gmail.com>", "I'd like to apply for a designer position.", False),
        ("Општина Центар <javni.nabavki@centar.gov.mk>", "Покана за тендер.", False),
    ],
)
def test_is_automated_email(sender: str, body: str, automated: bool) -> None:
    email = GmailMessage(id="g", thread_id="t", sender=sender, subject="s", body=body)
    assert is_automated_email(email) is automated


def _ignore_draft() -> DraftResult:
    return DraftResult(raw="", action=DraftAction.IGNORE, response_text="", confidence=0.99, docs_used=[], reasoning="")


def test_review_agent_only_lets_machine_mail_be_ignored() -> None:
    calm = Classification(category="INQUIRY", priority="LOW")
    noreply = GmailMessage(id="g", thread_id="t", sender="noreply@dhl.com", subject="s", body="Delivered.")
    invoice = GmailMessage(id="g", thread_id="t", sender="smetki@firma.mk", subject="Фактура", body="Фактура бр. 1")

    assert not ReviewAgent().execute(_ignore_draft(), calm, noreply).needs_review
    decision = ReviewAgent().execute(_ignore_draft(), calm, invoice)
    assert decision.needs_review and not decision.auto_send and "игнорирање" in decision.reason
    # Without the email to inspect, the safe default is a human look.
    assert ReviewAgent().execute(_ignore_draft(), calm).needs_review


@pytest.mark.parametrize(
    ("sender", "body"),
    [
        ("Хартија Плус <smetki@hartijaplus.mk>", "Во прилог е фактурата бр. 2026-0815 во износ од 84.300 ден."),
        ("Општина Центар <javni.nabavki@centar.gov.mk>", "Ве покануваме да учествувате на тендер."),
        ("Accounts <ar@inkworld-supplies.com>", "Invoice INK-5521 is 10 days overdue. Please arrange payment."),
        ("Maya <maya.thompson@gmail.com>", "I'd like to apply for a graphic designer position. CV attached."),
    ],
)
async def test_business_email_the_ai_wants_to_ignore_goes_to_review(sender: str, body: str) -> None:
    harness = PipelineHarness(make_row(auto_send=True, sender=sender, body=body), Classification(), IGNORE_DRAFT)

    assert await harness.run() is InboundStatus.AWAITING_REVIEW

    row = harness.row
    assert row.action == DraftAction.IGNORE.value and row.needs_review
    assert "игнорирање" in (row.review_reason or "")
    # Not labelled, not marked read, not logged as ignored, nothing sent.
    assert harness.gmail_service.modified == [] and harness.gmail_service.sent == []
    assert harness.log.records == []


async def test_spam_classification_is_still_ignored_without_review() -> None:
    row = make_row(auto_send=True, sender="Rank Boosters <hello@rankboosters-seo.biz>", body="Get #1 on Google!")
    harness = PipelineHarness(row, Classification(category="SPAM", priority="LOW"))

    assert await harness.run() is InboundStatus.IGNORED
    assert not harness.row.needs_review


async def test_approving_a_proposed_ignore_needs_a_written_reply() -> None:
    row = make_row(status=InboundStatus.AWAITING_REVIEW.value, action=DraftAction.IGNORE.value, response="")
    service = FakeGmailService()
    delivery = make_delivery(FakeInbound(row), FakeLog())

    for empty in (None, "", "   "):
        assert not await delivery.approve(gmail_client(service), row.user_email, row.id, empty)
    assert service.sent == [] and row.status == InboundStatus.AWAITING_REVIEW.value

    assert await delivery.approve(gmail_client(service), row.user_email, row.id, "Благодарам, ја примивме фактурата.")
    assert service.sent_mime()["To"] == "Ana <ana@klient.mk>"
    assert (row.status, row.action) == (InboundStatus.SENT.value, DraftAction.REPLY.value)


async def test_rejecting_a_proposed_ignore_sends_nothing() -> None:
    row = make_row(status=InboundStatus.AWAITING_REVIEW.value, action=DraftAction.IGNORE.value, response="")
    service = FakeGmailService()
    delivery = make_delivery(FakeInbound(row), FakeLog())

    assert await delivery.reject(gmail_client(service), row.user_email, row.id)
    assert service.sent == [] and row.status == InboundStatus.REJECTED.value
