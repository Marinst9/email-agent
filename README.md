# Enterprise AI Email Assistant & Automation Pipeline

A production-grade, multi-agent AI email processing and response system built with **FastAPI**, **Claude 3.5 Sonnet**, **pgvector (RAG)**, **Celery/Redis**, and **Docker**.

---

## 🌟 Key Features

- **Structured AI Outputs**: Pydantic-enforced responses with automatic fallback and self-correction retry loops.
- **Retrieval-Augmented Generation (RAG)**: Context-aware draft generation using `pgvector` for domain knowledge retrieval.
- **Asynchronous Task Architecture**: Background execution for heavy agent processing via Celery and Redis.
- **Enterprise Security & Policy Enforcement**: Rule-based prompt injection detection, sensitive data scrubbing, and tenant isolation.
- **Human-in-the-Loop Workflow**: Review and approval system for sensitive or low-confidence AI draft suggestions.
- **Production-Ready CI/CD**: Automated testing, linting, and container builds via GitHub Actions.

---

## 🏗️ System Architecture

[Inbound Email]
       │
       ▼
[FastAPI Gateway] ──► [Rule Engine / Validation]
       │
       ▼
 [Celery Queue] ──► [RAG Retrieval (pgvector)] ──► [Claude 3.5 Sonnet]
       │                                                   │
       ▼                                                   ▼
[PostgreSQL DB] ◄──────── [Structured Draft] ───────────────┘

---

## 🛠️ Tech Stack

- **Backend Framework**: FastAPI (Python 3.11+)
- **LLM Integration**: Anthropic Claude 3.5 Sonnet
- **Vector Database**: PostgreSQL with `pgvector` extension
- **Database ORM & Migrations**: SQLAlchemy 2.0 & Alembic
- **Task Queue**: Celery + Redis
- **Containerization**: Multi-stage Docker builds & `docker-compose`
- **CI/CD Pipeline**: GitHub Actions

---

## 🚀 Quick Start (Local Development)

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (optional, for local non-Docker development)

### Running with Docker Compose

1. Clone the repository:
   git clone https://github.com/Marinst9/email-agent.git
   cd email-agent

2. Configure environment variables:
   cp .env.example .env
   # Add your ANTHROPIC_API_KEY and database credentials in .env

3. Spin up the stack:
   docker-compose up --build -d

4. Verify running services:
   - API Docs (Swagger): http://localhost:8000/docs
   - Health Check: http://localhost:8000/healthz

---

## ⚙️ Running the Worker

Incoming emails are only ingested by the web app; classification, drafting and sending happen in the
Celery worker. **Without a running worker, emails stay queued and nothing is ever answered.**

Locally (with Redis and PostgreSQL running and `.env` filled in):

    uvicorn app.main:app --reload                                          # web
    celery -A app.worker.celery_app worker --loglevel=INFO -Q emails       # worker

`docker-compose up` starts both (`web` and `worker` services).

---

## 🚂 Deploying on Railway

The project needs four Railway services in one project:

| Service    | Source                     | Config file                                             |
| ---------- | -------------------------- | ------------------------------------------------------- |
| `web`      | this repo                  | `railway.toml` (default; runs migrations, then uvicorn) |
| `worker`   | this repo                  | `railway.worker.toml` (Celery worker, no public port)   |
| PostgreSQL | Railway PostgreSQL         |                                                         |
| Redis      | Railway Redis              |                                                         |

To add the worker:

1. In the project, **New → GitHub Repo** and pick this repository again; name the service `worker`.
2. In the worker's **Settings → Config-as-code**, set **Railway Config File** to `/railway.worker.toml`.
3. Give the worker the same variables as `web`: `DATABASE_URL` and `REDIS_URL` (as references to the
   PostgreSQL and Redis services), `SECRET_KEY`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`,
   `ANTHROPIC_API_KEY` (and `ANTHROPIC_MODEL` if you override it). Shared variables make this easier.
4. Do not generate a public domain for the worker; it only talks to Redis and PostgreSQL.

Keep the `web` service at one replica: the inbox pollers keep their running state in process memory.
The worker can be scaled with replicas or by changing `--concurrency` in `railway.worker.toml`.

---

## 🧪 Running Tests

To run unit and integration test suites locally:

pytest

---

## 📜 License

This project is open-source and available under the MIT License.
