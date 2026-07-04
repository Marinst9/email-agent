import os
import base64
import time
from email.mime.text import MIMEText
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import anthropic

load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
ai_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))



def get_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def get_unread_emails(service):
    results = service.users().messages().list(userId='me', q='is:unread').execute()
    return results.get('messages', [])

def get_email_content(service, msg_id):
    msg = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
    headers = msg['payload']['headers']
    subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
    sender = next((h['value'] for h in headers if h['name'] == 'From'), '')
    
    body = ''
    if 'parts' in msg['payload']:
        for part in msg['payload']['parts']:
            if part['mimeType'] == 'text/plain':
                body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
    elif 'body' in msg['payload'] and 'data' in msg['payload']['body']:
        body = base64.urlsafe_b64decode(msg['payload']['body']['data']).decode('utf-8')
    
    return subject, sender, body

def is_automated_email(sender):
    automated = ['noreply', 'no-reply', 'donotreply', 'do-not-reply',
                 'newsletter', 'notifications', 'notification', 'mailer',
                 'automated', 'bounce', 'support@courseking', 'contact@kariera',
                 'zara', 'pinterest', 'binance', 'linkedin', 'upwork']
    sender_lower = sender.lower()
    return any(word in sender_lower for word in automated)

def ai_decide(subject, sender, body):
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system="""Ти си AI агент за мејлови. Анализирај го мејлот и одлучи:

ПРАВИЛА:
- Ако мејлот е автоматска нотификација, реклама или newsletter -> АКЦИЈА: ИГНОРИРАЈ
- Ако мејлот е од реална личност и бара одговор -> АКЦИЈА: ОДГОВОР
- Ако мејлот треба да се препрати на некој друг -> АКЦИЈА: ПРЕПРАЌАЊЕ

Врати го одговорот САМО во овој формат:
АКЦИЈА: ОДГОВОР или ПРЕПРАЌАЊЕ или ИГНОРИРАЈ
АКО ПРЕПРАЌАЊЕ ДО: (email адреса или НИКОЈ)
ПОРАКА: (текстот на одговорот)""",
        messages=[{
            "role": "user",
            "content": f"Од: {sender}\nНаслов: {subject}\nСодржина: {body}"
        }]
    )
    return response.content[0].text

def send_reply(service, sender, subject, message_text):
    msg = MIMEText(message_text)
    msg['To'] = sender
    msg['Subject'] = f"Re: {subject}"
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    service.users().messages().send(userId='me', body={'raw': raw}).execute()
    print(f"✅ Одговорено на: {sender}")

def forward_email(service, to, subject, body):
    msg = MIMEText(body)
    msg['To'] = to
    msg['Subject'] = f"Fwd: {subject}"
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    service.users().messages().send(userId='me', body={'raw': raw}).execute()
    print(f"✅ Препратено до: {to}")

def mark_as_read(service, msg_id):
    service.users().messages().modify(
        userId='me', id=msg_id,
        body={'removeLabelIds': ['UNREAD']}
    ).execute()

def run_agent():
    print("🤖 Агентот е активен...")
    service = get_gmail_service()
    
    while True:
        emails = get_unread_emails(service)
        print(f"📬 Непрочитани мејлови: {len(emails)}")
        
        for email in emails:
            subject, sender, body = get_email_content(service, email['id'])
            print(f"📧 Обработувам: {subject} од {sender}")
            
            if is_automated_email(sender):
                print(f"⏭️ Прескокнувам автоматски мејл од: {sender}")
                mark_as_read(service, email['id'])
                continue
            
            decision = ai_decide(subject, sender, body)
            print(f"🧠 AI одлука:\n{decision}")
            
            if 'АКЦИЈА: ИГНОРИРАЈ' in decision:
                print("⏭️ AI одлучи да игнорира")

            elif 'АКЦИЈА: ОДГОВОР' in decision:
                poraka = decision.split('ПОРАКА:')[-1].strip()
                print(f"\n📝 Предлог одговор до {sender}:")
                print(f"---\n{poraka}\n---")
                potvrda = input("Испрати го овој одговор? (da/ne): ").strip().lower()
                if potvrda == 'da':
                    send_reply(service, sender, subject, poraka)
                else:
                    print("❌ Одговорот е откажан")

            elif 'АКЦИЈА: ПРЕПРАЌАЊЕ' in decision:
                lines = decision.split('\n')
                to = next((l.split('ДО:')[-1].strip() for l in lines if 'ДО:' in l), None)
                poraka = decision.split('ПОРАКА:')[-1].strip()
                if to and to != 'НИКОЈ':
                    print(f"\n📝 Препрати до {to}:")
                    print(f"---\n{poraka}\n---")
                    potvrda = input("Препрати го овој мејл? (da/ne): ").strip().lower()
                    if potvrda == 'da':
                        forward_email(service, to, subject, poraka)
                    else:
                        print("❌ Препраќањето е откажано")
            
            mark_as_read(service, email['id'])
        
        print("⏳ Чекам 60 секунди...")
        time.sleep(60)

if __name__ == "__main__":
    run_agent()