"""Replies are sent into the original Gmail thread with proper threading headers."""

import pytest

from app.models.enums import InboundStatus
from app.services.gmail import reply_subject
from tests.fakes import FakeGmailService, FakeInbound, FakeLog, gmail_client, make_delivery, make_row


@pytest.mark.parametrize(
    ("subject", "expected"),
    [
        ("Понуда", "Re: Понуда"),
        ("Re: Понуда", "Re: Понуда"),
        ("RE: Понуда", "RE: Понуда"),
        ("re : Понуда", "re : Понуда"),
        ("Re[2]: Понуда", "Re[2]: Понуда"),
        ("Одг: Понуда", "Одг: Понуда"),
        ("Return policy", "Re: Return policy"),
    ],
)
def test_reply_subject_is_never_doubled(subject: str, expected: str) -> None:
    assert reply_subject(subject) == expected


async def test_send_reply_sets_thread_and_headers() -> None:
    service = FakeGmailService()
    await gmail_client(service).send_reply(
        "a@b.mk", "Re: Понуда", "Одговор", thread_id="t42", in_reply_to="<m2@b.mk>", references="<m1@b.mk>"
    )

    assert service.sent[0]["threadId"] == "t42"
    sent = service.sent_mime()
    assert sent["Subject"] == "Re: Понуда"
    assert sent["In-Reply-To"] == "<m2@b.mk>"
    assert sent["References"] == "<m1@b.mk> <m2@b.mk>"


async def test_send_reply_without_message_id_omits_threading_headers() -> None:
    service = FakeGmailService()
    await gmail_client(service).send_reply("a@b.mk", "Тема", "Текст")
    assert "threadId" not in service.sent[0]
    sent = service.sent_mime()
    assert sent["In-Reply-To"] is None and sent["References"] is None


async def test_approved_reply_goes_into_the_original_thread() -> None:
    row = make_row(status=InboundStatus.AWAITING_REVIEW.value, subject="Re: Понуда", response="Еве ја понудата.")
    service = FakeGmailService()
    delivery = make_delivery(FakeInbound(row), FakeLog())

    assert await delivery.approve(gmail_client(service), row.user_email, row.id, None)

    assert service.sent[0]["threadId"] == "t1"
    sent = service.sent_mime()
    assert sent["To"] == "Ana <ana@klient.mk>"
    assert sent["Subject"] == "Re: Понуда"
    assert sent["In-Reply-To"] == "<CAB123@mail.gmail.com>"
    assert sent["References"] == "<first@klient.mk> <CAB123@mail.gmail.com>"


async def test_get_message_reads_threading_headers_case_insensitively() -> None:
    raw = {
        "id": "g1",
        "threadId": "t1",
        "payload": {
            "mimeType": "text/plain",
            "filename": "",
            "headers": [
                {"name": "From", "value": "a@b.mk"},
                {"name": "Subject", "value": "Тема"},
                {"name": "Message-Id", "value": "<abc@b.mk>"},
                {"name": "REFERENCES", "value": "<x@b.mk> <y@b.mk>"},
            ],
            "body": {"data": "VGV4dA"},
        },
    }
    message = await gmail_client(FakeGmailService({"g1": raw})).get_message("g1")
    assert (message.message_id_header, message.references, message.body) == ("<abc@b.mk>", "<x@b.mk> <y@b.mk>", "Text")
