from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import json
import threading
import time
from dotenv import load_dotenv
from authlib.integrations.flask_client import OAuth
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import anthropic
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

load_dotenv()

app = Flask(__name__)
app.secret_key = 'marina-email-agent-2026-super-secret-key'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = '/tmp/flask_sessions'
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///emails.db')app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
Session(app)
oauth = OAuth(app)

class EmailLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(100))
    sender = db.Column(db.String(200))
    subject = db.Column(db.String(300))
    response = db.Column(db.Text)
    source = db.Column(db.String(50))
    status = db.Column(db.String(20))
    category = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class ThreadMemory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(100))
    thread_id = db.Column(db.String(200))
    sender = db.Column(db.String(200))
    subject = db.Column(db.String(300))
    body = db.Column(db.Text)
    response = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class UserTemplate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(100))
    name = db.Column(db.String(100))
    keywords = db.Column(db.String(500))
    response = db.Column(db.Text)

class BlockedSender(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(100))
    word = db.Column(db.String(100))

with app.app_context():
    db.create_all()

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
user_states = {}
reply_counter = {}

def get_user_state(user_email):
    if user_email not in user_states:
        user_states[user_email] = {
            'running': False,
            'auto_mode': False,
            'processed': [],
            'pending': []
        }
    return user_states[user_email]

def can_reply_to(sender_email):
    now = time.time()
    hour_ago = now - 3600
    if sender_email not in reply_counter:
        reply_counter[sender_email] = []
    reply_counter[sender_email] = [t for t in reply_counter[sender_email] if t > hour_ago]
    if len(reply_counter[sender_email]) >= 3:
        return False
    reply_counter[sender_email].append(now)
    return True

def init_user_defaults(user_email):
    with app.app_context():
        # Default блокирани
        if not BlockedSender.query.filter_by(user_email=user_email).first():
            defaults = ['noreply', 'no-reply', 'donotreply', 'newsletter',
                       'notifications', 'notification', 'mailer', 'automated', 'bounce', 'railway']
            for word in defaults:
                db.session.add(BlockedSender(user_email=user_email, word=word))
            db.session.commit()

        # Default шаблони
        if not UserTemplate.query.filter_by(user_email=user_email).first():
            defaults = [
                {"name": "Консултации", "keywords": "консултации,consultation,meeting,состанок",
                 "response": "Здраво, благодарам за пораката. Слободен/на сум за консултации во следните термини: понеделник и среда 10-12ч."},
                {"name": "Потврда за прием", "keywords": "барање,request,апликација,application",
                 "response": "Здраво, Ви потврдувам дека Вашето барање е примено. Ќе Ви одговориме во рок од 2 работни дена."},
                {"name": "Недостапност", "keywords": "итно,urgent,помош,help",
                 "response": "Здраво, во моментов не сум достапен/на. За итни работи контактирајте нè на друга адреса."}
            ]
            for d in defaults:
                db.session.add(UserTemplate(user_email=user_email, name=d['name'],
                                           keywords=d['keywords'], response=d['response']))
            db.session.commit()

def load_blocked_for_user(user_email):
    blocked = BlockedSender.query.filter_by(user_email=user_email).all()
    return [b.word for b in blocked]

def load_templates_for_user(user_email):
    templates = UserTemplate.query.filter_by(user_email=user_email).all()
    return [{'id': t.id, 'name': t.name,
             'keywords': t.keywords.split(','), 'response': t.response} for t in templates]

def is_automated_email(sender, user_email):
    blocked = load_blocked_for_user(user_email)
    return any(word in sender.lower() for word in blocked)

def find_matching_template(subject, body, templates):
    text = (subject + ' ' + body).lower()
    for template in templates:
        for keyword in template['keywords']:
            if keyword.strip().lower() in text:
                return template
    return None

def get_thread_history(user_email, thread_id):
    memories = ThreadMemory.query.filter_by(
        user_email=user_email, thread_id=thread_id
    ).order_by(ThreadMemory.timestamp.desc()).limit(5).all()
    return memories

def categorize_email(subject, sender, body):
    response = ai_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        system="""Категоризирај го мејлот во ЕДНА од категориите:
INQUIRY - општо прашање
COMPLAINT - жалба или незадоволство
URGENT_HUMAN - итно, бара човечка интервенција
SPAM - нежелена пошта
Врати САМО категоријата, ништо друго.""",
        messages=[{"role": "user", "content": f"Од: {sender}\nНаслов: {subject}\nСодржина: {body[:200]}"}]
    )
    return response.content[0].text.strip()

def ai_decide(subject, sender, body, thread_history=None):
    history_text = ""
    if thread_history:
        history_text = "\nПРЕТХОДНИ ПОРАКИ ВО КОНВЕРЗАЦИЈАТА:\n"
        for mem in reversed(thread_history):
            history_text += f"- Примено: {mem.body[:100]}\n- Одговорено: {mem.response[:100]}\n\n"

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
        messages=[{"role": "user", "content": f"{history_text}Од: {sender}\nНаслов: {subject}\nСодржина: {body}"}]
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
    thread_id = msg.get('threadId', '')
    body = ''
    if 'parts' in msg['payload']:
        for part in msg['payload']['parts']:
            if part['mimeType'] == 'text/plain' and 'data' in part.get('body', {}):
                body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
    elif 'body' in msg['payload'] and 'data' in msg['payload']['body']:
        body = base64.urlsafe_b64decode(msg['payload']['body']['data']).decode('utf-8', errors='ignore')
    return subject, sender, body, thread_id

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

def save_to_log(user_email, sender, subject, response_text, source, status, category):
    with app.app_context():
        log = EmailLog(user_email=user_email, sender=sender, subject=subject,
                      response=response_text, source=source, status=status, category=category)
        db.session.add(log)
        db.session.commit()

def save_thread_memory(user_email, thread_id, sender, subject, body, response_text):
    with app.app_context():
        mem = ThreadMemory(user_email=user_email, thread_id=thread_id, sender=sender,
                          subject=subject, body=body[:500], response=response_text[:500])
        db.session.add(mem)
        db.session.commit()

def agent_loop(token, user_email):
    service = get_gmail_service_from_token(token)
    state = get_user_state(user_email)

    while state['running']:
        try:
            emails = get_unread_emails(service)

            with app.app_context():
                templates = load_templates_for_user(user_email)

            for email in emails:
                subject, sender, body, thread_id = get_email_content(service, email['id'])

                with app.app_context():
                    if is_automated_email(sender, user_email):
                        mark_as_read(service, email['id'])
                        save_to_log(user_email, sender, subject, '', 'Автоматски', 'игнориран', 'AUTO')
                        continue

                if not can_reply_to(sender):
                    print(f"⚠️ Rate limit за {sender}")
                    mark_as_read(service, email['id'])
                    continue

                category = categorize_email(subject, sender, body)

                if 'URGENT_HUMAN' in category or 'COMPLAINT' in category:
                    email_data = {
                        'id': email['id'],
                        'sender': sender,
                        'subject': subject,
                        'body': body[:300],
                        'response': '⚠️ Овој мејл бара човечка интервенција!',
                        'source': f'Категорија: {category}',
                        'user_email': user_email,
                        'category': category
                    }
                    if not any(p['id'] == email['id'] for p in state['pending']):
                        state['pending'].append(email_data)
                    save_to_log(user_email, sender, subject, '', f'Категорија: {category}', 'чека_човек', category)
                    continue

                if 'SPAM' in category:
                    mark_as_read(service, email['id'])
                    save_to_log(user_email, sender, subject, '', 'Автоматски', 'спам', 'SPAM')
                    continue

                with app.app_context():
                    thread_history = get_thread_history(user_email, thread_id)
                    matched = find_matching_template(subject, body, templates)

                if matched:
                    response_text = matched['response']
                    source = f"Шаблон: {matched['name']}"
                else:
                    decision = ai_decide(subject, sender, body, thread_history)
                    if 'ИГНОРИРАЈ' in decision:
                        mark_as_read(service, email['id'])
                        save_to_log(user_email, sender, subject, '', 'AI', 'игнориран', category)
                        continue
                    response_text = decision.split('ПОРАКА:')[-1].strip()
                    source = "AI генериран"

                email_data = {
                    'id': email['id'],
                    'sender': sender,
                    'subject': subject,
                    'body': body[:300],
                    'response': response_text,
                    'source': source,
                    'user_email': user_email,
                    'category': category,
                    'thread_id': thread_id
                }

                if state['auto_mode']:
                    send_reply(service, sender, subject, response_text)
                    mark_as_read(service, email['id'])
                    state['processed'].append({**email_data, 'status': 'испратено'})
                    save_to_log(user_email, sender, subject, response_text, source, 'испратено', category)
                    save_thread_memory(user_email, thread_id, sender, subject, body, response_text)
                else:
                    if not any(p['id'] == email['id'] for p in state['pending']):
                        state['pending'].append(email_data)

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
    with app.app_context():
        init_user_defaults(user_info['email'])
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    user_email = session.get('user', {}).get('email', '')
    if user_email in user_states:
        user_states[user_email]['running'] = False
    session.clear()
    return redirect(url_for('index'))

@app.route('/start', methods=['POST'])
def start():
    if not session.get('user'):
        return redirect(url_for('login'))
    user_email = session.get('user', {}).get('email', '')
    auto_mode = request.form.get('mode') == 'auto'
    state = get_user_state(user_email)
    state['auto_mode'] = auto_mode
    state['running'] = True
    state['pending'] = []
    state['processed'] = []
    token = session.get('token')
    thread = threading.Thread(target=agent_loop, args=(token, user_email), daemon=True)
    thread.start()
    return redirect(url_for('dashboard'))

@app.route('/stop', methods=['POST'])
def stop():
    user_email = session.get('user', {}).get('email', '')
    state = get_user_state(user_email)
    state['running'] = False
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if not session.get('user'):
        return redirect(url_for('login'))
    user_email = session.get('user', {}).get('email', '')
    state = get_user_state(user_email)
    return render_template('dashboard.html', state=state, user=session.get('user'))

@app.route('/approve/<email_id>', methods=['POST'])
def approve(email_id):
    user_email = session.get('user', {}).get('email', '')
    state = get_user_state(user_email)
    token = session.get('token')
    service = get_gmail_service_from_token(token)
    custom_response = request.form.get('custom_response')
    for email in state['pending']:
        if email['id'] == email_id:
            final_response = custom_response if custom_response else email['response']
            send_reply(service, email['sender'], email['subject'], final_response)
            mark_as_read(service, email_id)
            state['pending'].remove(email)
            state['processed'].append({**email, 'status': 'испратено'})
            save_to_log(user_email, email['sender'], email['subject'],
                       final_response, email['source'], 'испратено', email.get('category', ''))
            save_thread_memory(user_email, email.get('thread_id', ''), email['sender'],
                             email['subject'], email['body'], final_response)
            break
    return redirect(url_for('dashboard'))

@app.route('/reject/<email_id>', methods=['POST'])
def reject(email_id):
    user_email = session.get('user', {}).get('email', '')
    state = get_user_state(user_email)
    token = session.get('token')
    service = get_gmail_service_from_token(token)
    for email in state['pending']:
        if email['id'] == email_id:
            mark_as_read(service, email_id)
            state['pending'].remove(email)
            state['processed'].append({**email, 'status': 'одбиено'})
            save_to_log(user_email, email['sender'], email['subject'],
                       email['response'], email['source'], 'одбиено', email.get('category', ''))
            break
    return redirect(url_for('dashboard'))

@app.route('/templates', methods=['GET', 'POST'])
def manage_templates():
    if not session.get('user'):
        return redirect(url_for('login'))
    user_email = session.get('user', {}).get('email', '')
    templates = UserTemplate.query.filter_by(user_email=user_email).all()
    if request.method == 'POST':
        name = request.form.get('name')
        keywords = request.form.get('keywords')
        response = request.form.get('response')
        db.session.add(UserTemplate(user_email=user_email, name=name,
                                   keywords=keywords, response=response))
        db.session.commit()
        return redirect(url_for('manage_templates'))
    return render_template('templates.html', templates=templates)

@app.route('/templates/delete/<int:template_id>', methods=['POST'])
def delete_template(template_id):
    template = UserTemplate.query.get(template_id)
    if template:
        db.session.delete(template)
        db.session.commit()
    return redirect(url_for('manage_templates'))

@app.route('/history')
def history():
    if not session.get('user'):
        return redirect(url_for('login'))
    user_email = session.get('user', {}).get('email', '')
    logs = EmailLog.query.filter_by(user_email=user_email).order_by(EmailLog.timestamp.desc()).all()
    return render_template('history.html', logs=logs)

@app.route('/blocked', methods=['GET', 'POST'])
def manage_blocked():
    if not session.get('user'):
        return redirect(url_for('login'))
    user_email = session.get('user', {}).get('email', '')
    blocked = BlockedSender.query.filter_by(user_email=user_email).all()
    if request.method == 'POST':
        word = request.form.get('word', '').strip().lower()
        if word and not BlockedSender.query.filter_by(user_email=user_email, word=word).first():
            db.session.add(BlockedSender(user_email=user_email, word=word))
            db.session.commit()
        return redirect(url_for('manage_blocked'))
    return render_template('blocked.html', blocked=[b.word for b in blocked])

@app.route('/blocked/delete/<word>', methods=['POST'])
def delete_blocked(word):
    user_email = session.get('user', {}).get('email', '')
    b = BlockedSender.query.filter_by(user_email=user_email, word=word).first()
    if b:
        db.session.delete(b)
        db.session.commit()
    return redirect(url_for('manage_blocked'))

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))