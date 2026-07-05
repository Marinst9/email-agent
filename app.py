from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import json
import threading
from dotenv import load_dotenv
from authlib.integrations.flask_client import OAuth
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import anthropic
from flask_session import Session

load_dotenv()

app = Flask(__name__)
app.secret_key = 'marina-email-agent-2026-super-secret-key'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = '/tmp/flask_sessions'
Session(app)
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile https://www.googleapis.com/auth/gmail.modify'
    }
)

ai_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

agent_state = {
    'running': False,
    'auto_mode': False,
    'processed': [],
    'pending': []
}

templates_file = 'templates_data.json'

def load_templates():
    if os.path.exists(templates_file):
        with open(templates_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return [
        {"id": 1, "name": "Консултации", "keywords": ["консултации", "consultation", "meeting", "состанок"], "response": "Здраво, благодарам за пораката. Слободен/на сум за консултации во следните термини: понеделник и среда 10-12ч."},
        {"id": 2, "name": "Потврда за прием", "keywords": ["барање", "request", "апликација", "application"], "response": "Здраво, Ви потврдувам дека Вашето барање е примено. Ќе Ви одговориме во рок од 2 работни дена."},
        {"id": 3, "name": "Недостапност", "keywords": ["итно", "urgent", "помош", "help"], "response": "Здраво, во моментов не сум достапен/на. За итни работи контактирајте нè на друга адреса."}
    ]

def save_templates(templates):
    with open(templates_file, 'w', encoding='utf-8') as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)

def is_automated_email(sender):
    automated = ['noreply', 'no-reply', 'donotreply', 'do-not-reply',
             'newsletter', 'notifications', 'notification', 'mailer',
             'automated', 'bounce', 'zara', 'pinterest', 'binance', 
             'linkedin', 'upwork', 'airbnb', 'express@airbnb']
    return any(word in sender.lower() for word in automated)

def find_matching_template(subject, body, templates):
    text = (subject + ' ' + body).lower()
    for template in templates:
        for keyword in template['keywords']:
            if keyword.lower() in text:
                return template
    return None

def ai_decide(subject, sender, body):
    response = ai_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system="""Ти си AI агент за мејлови. Одговарај на јазикот на мејлот.
ПРАВИЛА:
- Автоматска нотификација, реклама, newsletter -> АКЦИЈА: ИГНОРИРАЈ
- Реална личност бара одговор -> АКЦИЈА: ОДГОВОР
- Треба препраќање -> АКЦИЈА: ПРЕПРАЌАЊЕ

Формат:
АКЦИЈА: ОДГОВОР или ПРЕПРАЌАЊЕ или ИГНОРИРАЈ
АКО ПРЕПРАЌАЊЕ ДО: (email или НИКОЈ)
ПОРАКА: (текст на одговорот)""",
        messages=[{"role": "user", "content": f"Од: {sender}\nНаслов: {subject}\nСодржина: {body}"}]
    )
    return response.content[0].text

def get_gmail_service_from_token(token):
    creds = Credentials(
        token=token['access_token'],
        refresh_token=token.get('refresh_token'),
        token_uri='https://oauth2.googleapis.com/token',
        client_id=os.getenv('GOOGLE_CLIENT_ID'),
        client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
        scopes=['https://www.googleapis.com/auth/gmail.modify']
    )
    return build('gmail', 'v1', credentials=creds)

import base64
from email.mime.text import MIMEText

def get_unread_emails(service):
    results = service.users().messages().list(userId='me', q='is:unread newer_than:1d').execute()

    return results.get('messages', [])

def get_email_content(service, msg_id):
    msg = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
    headers = msg['payload']['headers']
    subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
    sender = next((h['value'] for h in headers if h['name'] == 'From'), '')
    body = ''
    if 'parts' in msg['payload']:
        for part in msg['payload']['parts']:
            if part['mimeType'] == 'text/plain' and 'data' in part.get('body', {}):
              body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')

    elif 'body' in msg['payload'] and 'data' in msg['payload']['body']:
       body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')

    return subject, sender, body

def send_reply(service, sender, subject, message_text):
    msg = MIMEText(message_text)
    msg['To'] = sender
    msg['Subject'] = f"Re: {subject}"
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    service.users().messages().send(userId='me', body={'raw': raw}).execute()

def mark_as_read(service, msg_id):
    service.users().messages().modify(
        userId='me', id=msg_id,
        body={'removeLabelIds': ['UNREAD']}
    ).execute()

def agent_loop(token):
    import time
    service = get_gmail_service_from_token(token)
    while agent_state['running']:
        try:
            emails = get_unread_emails(service)
            templates = load_templates()
            for email in emails:
                subject, sender, body = get_email_content(service, email['id'])
                if is_automated_email(sender):
                    mark_as_read(service, email['id'])
                    continue
                matched = find_matching_template(subject, body, templates)
                if matched:
                    response_text = matched['response']
                    source = f"Шаблон: {matched['name']}"
                else:
                    decision = ai_decide(subject, sender, body)
                    if 'ИГНОРИРАЈ' in decision:
                        mark_as_read(service, email['id'])
                        continue
                    response_text = decision.split('ПОРАКА:')[-1].strip()
                    source = "AI генериран"
                email_data = {
                    'id': email['id'],
                    'sender': sender,
                    'subject': subject,
                    'body': body[:300],
                    'response': response_text,
                    'source': source
                }
                if agent_state['auto_mode']:
                    send_reply(service, sender, subject, response_text)
                    mark_as_read(service, email['id'])
                    agent_state['processed'].append({**email_data, 'status': 'испратено'})
                else:
                    if not any(p['id'] == email['id'] for p in agent_state['pending']):
                        agent_state['pending'].append(email_data)
            time.sleep(60)
        except Exception as e:
            print(f"Грешка: {e}")
            time.sleep(30)

@app.route('/')
def index():
    user = session.get('user')
    return render_template('index.html', user=user)


@app.route('/login')
def login():

    redirect_uri = 'https://web-production-ec2fb.up.railway.app/callback'
    return google.authorize_redirect(redirect_uri)


@app.route('/callback')
def callback():
    token = google.authorize_access_token()
    user_info = token.get('userinfo')
    session['user'] = user_info
    session['token'] = token
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    agent_state['running'] = False
    session.clear()
    return redirect(url_for('index'))

@app.route('/start', methods=['POST'])
def start():
    if not session.get('user'):
        return redirect(url_for('login'))
    auto_mode = request.form.get('mode') == 'auto'
    agent_state['auto_mode'] = auto_mode
    agent_state['running'] = True
    agent_state['pending'] = []
    agent_state['processed'] = []
    token = session.get('token')
    thread = threading.Thread(target=agent_loop, args=(token,), daemon=True)
    thread.start()
    return redirect(url_for('dashboard'))

@app.route('/stop', methods=['POST'])
def stop():
    agent_state['running'] = False
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if not session.get('user'):
        return redirect(url_for('login'))
    return render_template('dashboard.html', state=agent_state, user=session.get('user'))

@app.route('/approve/<email_id>', methods=['POST'])
def approve(email_id):
    token = session.get('token')
    service = get_gmail_service_from_token(token)
    for email in agent_state['pending']:
        if email['id'] == email_id:
            send_reply(service, email['sender'], email['subject'], email['response'])
            mark_as_read(service, email_id)
            agent_state['pending'].remove(email)
            agent_state['processed'].append({**email, 'status': 'испратено'})
            break
    return redirect(url_for('dashboard'))

@app.route('/reject/<email_id>', methods=['POST'])
def reject(email_id):
    token = session.get('token')
    service = get_gmail_service_from_token(token)
    for email in agent_state['pending']:
        if email['id'] == email_id:
            mark_as_read(service, email_id)
            agent_state['pending'].remove(email)
            agent_state['processed'].append({**email, 'status': 'одбиено'})
            break
    return redirect(url_for('dashboard'))

@app.route('/templates', methods=['GET', 'POST'])
def manage_templates():
    if not session.get('user'):
        return redirect(url_for('login'))
    templates = load_templates()
    if request.method == 'POST':
        name = request.form.get('name')
        keywords = [k.strip() for k in request.form.get('keywords', '').split(',')]
        response = request.form.get('response')
        new_id = max([t['id'] for t in templates], default=0) + 1
        templates.append({'id': new_id, 'name': name, 'keywords': keywords, 'response': response})
        save_templates(templates)
        return redirect(url_for('manage_templates'))
    return render_template('templates.html', templates=templates)

@app.route('/templates/delete/<int:template_id>', methods=['POST'])

def delete_template(template_id):
    templates = load_templates()
    templates = [t for t in templates if t['id'] != template_id]
    save_templates(templates)
    return redirect(url_for('manage_templates'))

if __name__ == '__main__':
  app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
