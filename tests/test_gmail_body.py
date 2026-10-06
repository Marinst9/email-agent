"""Body extraction from realistic Gmail API `format=full` payloads."""

import base64
from typing import Any

from app.services.gmail import _extract_plain_text, html_to_text
from tests.fakes import FakeGmailService, gmail_client


def b64(text: str, charset: str = "utf-8") -> str:
    # Gmail returns unpadded base64url.
    return base64.urlsafe_b64encode(text.encode(charset)).decode().rstrip("=")


def text_part(mime_type: str, text: str, charset: str = "UTF-8", part_id: str = "0") -> dict[str, Any]:
    return {
        "partId": part_id,
        "mimeType": mime_type,
        "filename": "",
        "headers": [
            {"name": "Content-Type", "value": f'{mime_type}; charset="{charset}"'},
            {"name": "Content-Transfer-Encoding", "value": "quoted-printable"},
        ],
        "body": {"size": len(text), "data": b64(text, charset)},
    }


def container(mime_type: str, *parts: dict[str, Any]) -> dict[str, Any]:
    return {
        "partId": "",
        "mimeType": mime_type,
        "filename": "",
        "headers": [{"name": "Content-Type", "value": f'{mime_type}; boundary="000000000000abc"'}],
        "body": {"size": 0},
        "parts": list(parts),
    }


HTML = (
    "<html><head><style>p{color:red}</style><title>x</title></head><body>"
    '<div dir="ltr">Здраво,<br>ве молам за&nbsp;понуда &amp; цени.</div>'
    "<p>Поздрав,<br>Ана</p><script>alert(1)</script></body></html>"
)


def test_single_part_plain_text() -> None:
    assert _extract_plain_text(text_part("text/plain", "Само текст.\r\n")) == "Само текст."


def test_multipart_alternative_prefers_plain_text() -> None:
    payload = container(
        "multipart/alternative",
        text_part("text/plain", "Здраво,\nве молам за понуда.", part_id="0"),
        text_part("text/html", HTML, part_id="1"),
    )
    assert _extract_plain_text(payload) == "Здраво,\nве молам за понуда."


def test_nested_mixed_related_alternative_with_attachment() -> None:
    # What Gmail sends for a reply with an inline image and a PDF attachment.
    attachment = {
        "partId": "1",
        "mimeType": "text/plain",
        "filename": "notes.txt",
        "headers": [{"name": "Content-Disposition", "value": 'attachment; filename="notes.txt"'}],
        "body": {"attachmentId": "ANGjdJ8", "size": 1200},
    }
    pdf = {"partId": "2", "mimeType": "application/pdf", "filename": "faktura.pdf", "body": {"attachmentId": "X"}}
    payload = container(
        "multipart/mixed",
        container(
            "multipart/related",
            container(
                "multipart/alternative",
                text_part("text/plain", "Текст во длабочина.", part_id="0.0.0"),
                text_part("text/html", HTML, part_id="0.0.1"),
            ),
            {"partId": "0.1", "mimeType": "image/png", "filename": "logo.png", "body": {"attachmentId": "IMG"}},
        ),
        attachment,
        pdf,
    )
    assert _extract_plain_text(payload) == "Текст во длабочина."


def test_html_only_falls_back_to_stripped_text() -> None:
    payload = container("multipart/alternative", text_part("text/html", HTML))
    assert _extract_plain_text(payload) == "Здраво,\nве молам за понуда & цени.\n\nПоздрав,\nАна"


def test_empty_plain_part_falls_back_to_html() -> None:
    payload = container(
        "multipart/alternative", text_part("text/plain", "  \r\n"), text_part("text/html", "<p>Од HTML</p>")
    )
    assert _extract_plain_text(payload) == "Од HTML"


def test_non_utf8_charset_is_respected() -> None:
    assert _extract_plain_text(text_part("text/plain", "Почитувани", charset="windows-1251")) == "Почитувани"


def test_unknown_charset_falls_back_to_utf8() -> None:
    part = text_part("text/plain", "Текст")
    part["headers"][0]["value"] = 'text/plain; charset="x-unknown"'
    assert _extract_plain_text(part) == "Текст"


def test_no_text_parts_returns_empty_string() -> None:
    payload = container("multipart/mixed", {"partId": "0", "mimeType": "image/jpeg", "filename": "a.jpg", "body": {}})
    assert _extract_plain_text(payload) == ""


def test_html_to_text_handles_entities_and_whitespace() -> None:
    assert html_to_text("<p>A&nbsp;&lt;b&gt;   c</p>\n\n\n<p>d</p>") == "A <b> c\n\nd"


async def test_get_message_uses_recursive_extraction() -> None:
    raw = {
        "id": "g1",
        "threadId": "t1",
        "payload": {
            **container("multipart/alternative", text_part("text/html", "<b>Само HTML</b>")),
            "headers": [{"name": "From", "value": "a@b.mk"}, {"name": "Subject", "value": "Тема"}],
        },
    }
    message = await gmail_client(FakeGmailService({"g1": raw})).get_message("g1")
    assert (message.sender, message.subject, message.body) == ("a@b.mk", "Тема", "Само HTML")
