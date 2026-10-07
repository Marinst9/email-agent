"""FORWARD decisions are parsed, stored, never auto-sent, and delivered as a real forward after approval."""

import pytest

from app.models.enums import DraftAction, InboundStatus
from app.schemas.agent import ApproveForm, Citation, Classification, DraftResult
from app.schemas.email import GmailMessage
from app.services.ai_agents import ReviewAgent, parse_forward_to
from app.services.gmail import forward_subject
from tests.fakes import (
    FakeGmailService,
    FakeInbound,
    FakeLog,
    PipelineHarness,
    gmail_client,
    make_delivery,
    make_row,
    mime_text,
)

FORWARD_DRAFT = """АКЦИЈА: ПРЕПРАЌАЊЕ
АКО ПРЕПРАЌАЊЕ ДО: Сметководство <smetki@firma.mk>
ПОРАКА: Ве молам погледнете ја оваа фактура."""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (FORWARD_DRAFT, "smetki@firma.mk"),
        ("АКЦИЈА: ПРЕПРАЌАЊЕ\nАКО ПРЕПРАЌАЊЕ ДО: pravno@firma.mk\nПОРАКА: x", "pravno@firma.mk"),
        ("АКЦИЈА: ПРЕПРАЌАЊЕ\nАКО ПРЕПРАЌАЊЕ ДО: НИКОЈ\nПОРАКА: x", None),
        ("АКЦИЈА: ОДГОВОР\nПОРАКА: x", None),
        # Formats the model produces in practice (found by the evals).
        ("АКЦИЈА: ПРЕПРАЌАЊЕ\nДО: finance@lumenprint.mk\n\nПОРАКА:\nx", "finance@lumenprint.mk"),
        ("АКЦИЈА: ПРЕПРАЌАЊЕ\nПРЕПРАЌАЊЕ ДО: hr@firma.mk\nПОРАКА: x", "hr@firma.mk"),
        ("АКЦИЈА: ПРЕПРАЌАЊЕ\nдо: legal@firma.mk\nПОРАКА: x", "legal@firma.mk"),
        ("АКЦИЈА: ПРЕПРАЌАЊЕ\nTO: sales@firma.mk\nПОРАКА: x", "sales@firma.mk"),
        ("АКЦИЈА: ПРЕПРАЌАЊЕ\nForward to: Sales <sales@firma.mk>.\nПОРАКА: x", "sales@firma.mk"),
        ("АКЦИЈА: ПРЕПРАЌАЊЕ\n**FORWARD TO:** sales@firma.mk\nПОРАКА: x", "sales@firma.mk"),
        ("АКЦИЈА: ПРЕПРАЌАЊЕ\n- **ДО**: finance@firma.mk\nПОРАКА: x", "finance@firma.mk"),
        # Present but not a valid address.
        ("АКЦИЈА: ПРЕПРАЌАЊЕ\nДО: сметководство\nПОРАКА: x", None),
        ("АКЦИЈА: ПРЕПРАЌАЊЕ\nTO: finance@\nПОРАКА: x", None),
        # A "To:" inside the drafted message is not the recipient; neither are lookalike labels.
        ("АКЦИЈА: ПРЕПРАЌАЊЕ\nПОРАКА: Препратено.\nTo: customer@klient.mk", None),
        ("АКЦИЈА: ПРЕПРАЌАЊЕ\nОД: a@klient.mk\nTOPIC: b@klient.mk\nПОРАКА: x", None),
    ],
)
def test_parse_forward_to(text: str, expected: str | None) -> None:
    assert parse_forward_to(text) == expected


def _draft(action: DraftAction, forward_to: str | None = None) -> DraftResult:
    return DraftResult(raw="", action=action, response_text="", citations=[], reasoning="", forward_to=forward_to)


def test_review_agent_always_flags_forwards() -> None:
    calm = Classification(category="INQUIRY", priority="LOW")
    decision = ReviewAgent().execute(_draft(DraftAction.FORWARD, "smetki@firma.mk"), calm)
    assert decision.needs_review and not decision.auto_send
    assert "не е наведен примач" in ReviewAgent().execute(_draft(DraftAction.FORWARD), calm).reason
    cited = _draft(DraftAction.REPLY).model_copy(
        update={"citations": [Citation(chunk_id=1, filename="faq.md", snippet="...")]}
    )
    assert not ReviewAgent().execute(cited, calm).needs_review


async def test_pipeline_stores_forward_and_never_auto_sends_it() -> None:
    harness = PipelineHarness(make_row(auto_send=True), Classification(priority="LOW"), FORWARD_DRAFT)

    assert await harness.run() is InboundStatus.AWAITING_REVIEW

    row = harness.row
    assert (row.action, row.forward_to, row.needs_review) == ("ПРЕПРАЌАЊЕ", "smetki@firma.mk", True)
    assert row.response == "Ве молам погледнете ја оваа фактура."
    assert harness.gmail_service.sent == []


async def test_auto_send_refuses_a_forward_even_if_not_flagged() -> None:
    row = make_row(status=InboundStatus.DRAFTED.value, action="ПРЕПРАЌАЊЕ", forward_to="x@firma.mk", response="r")
    service = FakeGmailService()
    delivery = make_delivery(FakeInbound(row), FakeLog())

    assert await delivery.auto_send(gmail_client(service), row) is InboundStatus.AWAITING_REVIEW
    assert service.sent == []


async def test_approving_a_forward_sends_it_to_the_recipient_not_the_sender() -> None:
    row = make_row(
        status=InboundStatus.AWAITING_REVIEW.value, action="ПРЕПРАЌАЊЕ", forward_to="smetki@firma.mk", response="Види."
    )
    service, log = FakeGmailService(), FakeLog()
    delivery = make_delivery(FakeInbound(row), log)

    assert await delivery.approve(gmail_client(service), row.user_email, row.id, None)

    sent = service.sent_mime()
    assert sent["To"] == "smetki@firma.mk"
    assert sent["Subject"] == "Fwd: Прашање за понуда"
    body = mime_text(sent)
    assert body.startswith("Види.")
    assert "From: Ana <ana@klient.mk>" in body and row.body in body
    assert row.status == InboundStatus.SENT.value
    assert log.threads == []  # a forward is not a turn of the conversation with the sender


async def test_reviewer_can_set_the_recipient_and_forward_without_one_is_refused() -> None:
    row = make_row(status=InboundStatus.AWAITING_REVIEW.value, action="ПРЕПРАЌАЊЕ", forward_to=None, response="r")
    service = FakeGmailService()
    delivery = make_delivery(FakeInbound(row), FakeLog())

    assert not await delivery.approve(gmail_client(service), row.user_email, row.id, None)
    assert service.sent == [] and row.status == InboundStatus.AWAITING_REVIEW.value

    form = ApproveForm(forward_to="Правно <pravno@firma.mk>")
    assert await delivery.approve(gmail_client(service), row.user_email, row.id, None, form.forward_to)
    assert service.sent_mime()["To"] == "pravno@firma.mk"
    assert row.forward_to == "pravno@firma.mk"


def test_approve_form_rejects_a_non_address() -> None:
    with pytest.raises(ValueError):
        ApproveForm(forward_to="not an address")
    assert ApproveForm(forward_to="  ").forward_to is None


def test_forward_subject_is_not_doubled() -> None:
    assert forward_subject("Фактура") == "Fwd: Фактура"
    assert forward_subject("Fwd: Фактура") == "Fwd: Фактура"
    assert forward_subject("FW: Фактура") == "FW: Фактура"


async def test_gmail_forward_quotes_the_original() -> None:
    service = FakeGmailService()
    original = GmailMessage(id="g1", thread_id="t1", sender="a@b.mk", subject="Тема", body="Оригинал")
    await gmail_client(service).forward("c@d.mk", original, "")
    body = mime_text(service.sent_mime())
    assert body.startswith("---------- Forwarded message ---------")
    assert "Оригинал" in body
