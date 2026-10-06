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
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from app.core.config import GMAIL_SCOPE, Settings
from app.schemas.email import GmailMessage

GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"
FORWARD_SEPARATOR = "---------- Forwarded message ---------"
_FORWARD_PREFIX = re.compile(r"^\s*(fwd?|пр)\s*:", re.IGNORECASE)


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


def _decode(data: str) -> str:
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")


def _extract_plain_text(payload: Mapping[str, Any]) -> str:
    body = ""
    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain" and "data" in part.get("body", {}):
                body = _decode(part["body"]["data"])
    elif "data" in payload.get("body", {}):
        body = _decode(payload["body"]["data"])
    return body
