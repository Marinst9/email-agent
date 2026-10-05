"""Standalone terminal agent: polls Gmail and asks for confirmation before replying or forwarding.

Run with `python main.py`. This is not the web app; that is `app/main.py` (uvicorn app.main:app).
It runs a single task, so the blocking Gmail and input() calls inside the async loop are acceptable.
"""

import asyncio
import base64
import os
from email.mime.text import MIMEText
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from app.agents import DEFAULT_MODEL, StructuredOutputError, create_validated, schema_instructions
from app.schemas.agent import DraftAction, OrchestrationResult

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
MODEL = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
MAX_CORRECTION_ATTEMPTS = 3
POLL_INTERVAL_SECONDS = 60

# Reads ANTHROPIC_API_KEY from the environment (.env is loaded above).
client = AsyncAnthropic()

SYSTEM_PROMPT = f"""Ти си AI агент за мејлови. Анализирај го мејлот и одлучи:

ПРАВИЛА:
- Ако мејлот е автоматска нотификација, реклама или newsletter -> action: ИГНОРИРАЈ
- Ако мејлот е од реална личност и бара одговор -> action: ОДГОВОР
- Ако мејлот треба да се препрати на некој друг -> action: ПРЕПРАЌАЊЕ

How to fill the fields:
- classification: category is INQUIRY, COMPLAINT, URGENT_HUMAN or SPAM; priority is HIGH, MEDIUM or LOW;
  language is mk, en or other; sentiment is positive, neutral or negative.
- retrieved_docs: always [] (no knowledge base is available here).
- draft: null when action is ИГНОРИРАЈ. Otherwise draft.action equals action, draft.response_text is the
  complete message to send, draft.forward_to is the recipient address for ПРЕПРАЌАЊЕ (null otherwise),
  draft.raw is "" and draft.docs_used is [].
- review: needs_review is true for complaints and urgent matters; auto_send is the opposite of needs_review.
- confidence: a number between 0 and 1. reasoning: one short sentence explaining the decision.

{schema_instructions(OrchestrationResult)}"""


def get_gmail_service() -> Any:
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def get_unread_emails(service: Any) -> list[dict[str, Any]]:
    results = service.users().messages().list(userId="me", q="is:unread").execute()
    return results.get("messages", [])


def get_email_content(service: Any, msg_id: str) -> tuple[str, str, str]:
    msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
    headers = msg["payload"]["headers"]
    subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
    sender = next((h["value"] for h in headers if h["name"] == "From"), "")

    body = ""
    if "parts" in msg["payload"]:
        for part in msg["payload"]["parts"]:
            if part["mimeType"] == "text/plain":
                body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
    elif "body" in msg["payload"] and "data" in msg["payload"]["body"]:
        body = base64.urlsafe_b64decode(msg["payload"]["body"]["data"]).decode("utf-8")

    return subject, sender, body


def is_automated_email(sender: str) -> bool:
    automated = ["noreply", "no-reply", "donotreply", "do-not-reply",
                 "newsletter", "notifications", "notification", "mailer",
                 "automated", "bounce", "support@courseking", "contact@kariera",
                 "zara", "pinterest", "binance", "linkedin", "upwork"]
    sender_lower = sender.lower()
    return any(word in sender_lower for word in automated)


async def ai_decide(subject: str, sender: str, body: str) -> OrchestrationResult:
    """Ask Claude for a decision as schema-validated JSON, letting it self-correct invalid output."""
    return await create_validated(
        client,
        model=MODEL,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Од: {sender}\nНаслов: {subject}\nСодржина: {body}"}],
        output_model=OrchestrationResult,
        max_attempts=MAX_CORRECTION_ATTEMPTS,
    )


def send_reply(service: Any, sender: str, subject: str, message_text: str) -> None:
    msg = MIMEText(message_text)
    msg["To"] = sender
    msg["Subject"] = f"Re: {subject}"
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()
    print(f"✅ Одговорено на: {sender}")


def forward_email(service: Any, to: str, subject: str, body: str) -> None:
    msg = MIMEText(body)
    msg["To"] = to
    msg["Subject"] = f"Fwd: {subject}"
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()
    print(f"✅ Препратено до: {to}")


def mark_as_read(service: Any, msg_id: str) -> None:
    service.users().messages().modify(
        userId="me", id=msg_id,
        body={"removeLabelIds": ["UNREAD"]}
    ).execute()


async def run_agent() -> None:
    print("🤖 Агентот е активен...")
    service = get_gmail_service()

    while True:
        emails = get_unread_emails(service)
        print(f"📬 Непрочитани мејлови: {len(emails)}")

        for email in emails:
            subject, sender, body = get_email_content(service, email["id"])
            print(f"📧 Обработувам: {subject} од {sender}")

            if is_automated_email(sender):
                print(f"⏭️ Прескокнувам автоматски мејл од: {sender}")
                mark_as_read(service, email["id"])
                continue

            try:
                decision = await ai_decide(subject, sender, body)
            except StructuredOutputError as exc:
                # Left unread so it is retried on the next poll.
                print(f"⚠️ AI не врати валидна одлука: {exc}")
                continue

            print(f"🧠 AI одлука: {decision.action} — {decision.reasoning}")
            draft = decision.draft

            if decision.action is DraftAction.IGNORE or draft is None:
                print("⏭️ AI одлучи да игнорира")

            elif decision.action is DraftAction.REPLY:
                print(f"\n📝 Предлог одговор до {sender}:")
                print(f"---\n{draft.response_text}\n---")
                potvrda = input("Испрати го овој одговор? (da/ne): ").strip().lower()
                if potvrda == "da":
                    send_reply(service, sender, subject, draft.response_text)
                else:
                    print("❌ Одговорот е откажан")

            elif decision.action is DraftAction.FORWARD:
                to = draft.forward_to
                if to and to != "НИКОЈ":
                    print(f"\n📝 Препрати до {to}:")
                    print(f"---\n{draft.response_text}\n---")
                    potvrda = input("Препрати го овој мејл? (da/ne): ").strip().lower()
                    if potvrda == "da":
                        forward_email(service, to, subject, draft.response_text)
                    else:
                        print("❌ Препраќањето е откажано")
                else:
                    print("⚠️ AI предложи препраќање, но без примач")

            mark_as_read(service, email["id"])

        print(f"⏳ Чекам {POLL_INTERVAL_SECONDS} секунди...")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(run_agent())
