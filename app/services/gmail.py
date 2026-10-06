"""Async facade over the (blocking) Google API client.

Every SDK call runs in a worker thread via `asyncio.to_thread` so it never blocks the event loop.
Callers await each call before issuing the next, so the underlying httplib2 connection is never
used concurrently.
"""

import asyncio
import base64
import re
from collections.abc import Mapping
from email.message import EmailMessage
from html.parser import HTMLParser
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from app.core.config import GMAIL_SCOPE, Settings
from app.schemas.email import GmailMessage

GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"
FORWARD_SEPARATOR = "---------- Forwarded message ---------"
_FORWARD_PREFIX = re.compile(r"^\s*(fwd?|пр)\s*:", re.IGNORECASE)
_CHARSET = re.compile(r"charset\s*=\s*\"?([\w.:-]+)\"?", re.IGNORECASE)
_BLANK_LINES = re.compile(r"\n{3,}")


class GmailClient:
    def __init__(self, service: Any) -> None:
        self._service = service
        self._label_ids: dict[str, str] = {}

    @classmethod
    async def from_token(cls, token: Mapping[str, Any], settings: Settings) -> "GmailClient":
        creds = Credentials(
            token=token["access_token"],
            refresh_token=token.get("refresh_token"),
            token_uri=GOOGLE_TOKEN_URI,
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret.get_secret_value(),
            scopes=[GMAIL_SCOPE],
        )
        service = await asyncio.to_thread(build, "gmail", "v1", credentials=creds, cache_discovery=False)
        return cls(service)

    async def list_unread_ids(self, query: str) -> list[str]:
        request = self._service.users().messages().list(userId="me", q=query)
        result: dict[str, Any] = await asyncio.to_thread(request.execute)
        return [m["id"] for m in result.get("messages", [])]

    async def get_message(self, message_id: str) -> GmailMessage:
        request = self._service.users().messages().get(userId="me", id=message_id, format="full")
        msg: dict[str, Any] = await asyncio.to_thread(request.execute)
        payload: dict[str, Any] = msg["payload"]
        headers: list[dict[str, str]] = payload.get("headers", [])
        return GmailMessage(
            id=message_id,
            thread_id=msg.get("threadId", ""),
            subject=next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject"),
            sender=next((h["value"] for h in headers if h["name"] == "From"), ""),
            body=_extract_plain_text(payload),
        )

    async def send_reply(self, to: str, subject: str, text: str) -> None:
        message = EmailMessage()
        message["To"] = to
        message["Subject"] = f"Re: {subject}"
        message.set_content(text)
        await self._send(message)

    async def forward(self, to: str, original: GmailMessage, note: str) -> None:
        """Forward `original` to `to`, with `note` above the quoted original message."""
        message = EmailMessage()
        message["To"] = to
        message["Subject"] = forward_subject(original.subject)
        intro = f"{note.strip()}\n\n" if note.strip() else ""
        message.set_content(
            f"{intro}{FORWARD_SEPARATOR}\nFrom: {original.sender}\nSubject: {original.subject}\n\n{original.body}"
        )
        await self._send(message)

    async def _send(self, message: EmailMessage, thread_id: str | None = None) -> None:
        body: dict[str, str] = {"raw": base64.urlsafe_b64encode(message.as_bytes()).decode()}
        if thread_id:
            body["threadId"] = thread_id
        request = self._service.users().messages().send(userId="me", body=body)
        await asyncio.to_thread(request.execute)

    async def add_label(self, message_id: str, label_name: str, *, mark_read: bool) -> None:
        """Apply a user label (created on first use); also remove UNREAD only when `mark_read`."""
        body: dict[str, list[str]] = {"addLabelIds": [await self._label_id(label_name)]}
        if mark_read:
            body["removeLabelIds"] = ["UNREAD"]
        request = self._service.users().messages().modify(userId="me", id=message_id, body=body)
        await asyncio.to_thread(request.execute)

    async def _label_id(self, name: str) -> str:
        if name not in self._label_ids:
            listing: dict[str, Any] = await asyncio.to_thread(self._service.users().labels().list(userId="me").execute)
            existing = next((label["id"] for label in listing.get("labels", []) if label["name"] == name), None)
            if existing is None:
                body = {"name": name, "labelListVisibility": "labelShow", "messageListVisibility": "show"}
                request = self._service.users().labels().create(userId="me", body=body)
                created: dict[str, Any] = await asyncio.to_thread(request.execute)
                existing = created["id"]
            self._label_ids[name] = str(existing)
        return self._label_ids[name]

    async def mark_as_read(self, message_id: str) -> None:
        request = self._service.users().messages().modify(
            userId="me", id=message_id, body={"removeLabelIds": ["UNREAD"]}
        )
        await asyncio.to_thread(request.execute)


def forward_subject(subject: str) -> str:
    return subject if _FORWARD_PREFIX.match(subject) else f"Fwd: {subject}"


class _HtmlToText(HTMLParser):
    _SKIPPED = frozenset({"script", "style", "head", "title"})
    _BLOCKS = frozenset({"br", "p", "div", "li", "tr", "table", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIPPED:
            self._skipping += 1
        elif tag in self._BLOCKS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIPPED:
            self._skipping = max(self._skipping - 1, 0)
        elif tag in self._BLOCKS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self._chunks.append(data)

    def text(self) -> str:
        lines = [" ".join(line.split()) for line in "".join(self._chunks).splitlines()]
        return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def html_to_text(html: str) -> str:
    parser = _HtmlToText()
    parser.feed(html)
    parser.close()
    return parser.text()


def _charset(part: Mapping[str, Any]) -> str:
    content_type = next(
        (h["value"] for h in part.get("headers", []) if h.get("name", "").lower() == "content-type"), ""
    )
    match = _CHARSET.search(content_type)
    return match.group(1) if match else "utf-8"


def _decode(data: str, charset: str = "utf-8") -> str:
    raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:  # unknown charset name in the header
        return raw.decode("utf-8", errors="replace")


def _collect_text_parts(part: Mapping[str, Any], found: dict[str, str]) -> None:
    """Depth-first walk of a Gmail payload; keeps the first text/plain and text/html body (not attachments)."""
    if part.get("filename"):
        return
    for child in part.get("parts", []):
        _collect_text_parts(child, found)
    mime_type = str(part.get("mimeType", "")).lower()
    data = part.get("body", {}).get("data")
    if data and mime_type in ("text/plain", "text/html") and mime_type not in found:
        found[mime_type] = _decode(data, _charset(part))


def _extract_plain_text(payload: Mapping[str, Any]) -> str:
    """The message text: text/plain if present, otherwise text/html converted to text."""
    found: dict[str, str] = {}
    _collect_text_parts(payload, found)
    plain = found.get("text/plain", "").strip()
    return plain or html_to_text(found.get("text/html", ""))
