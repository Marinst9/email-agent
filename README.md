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

## 📊 Evaluation

`evals/` measures how well the agent works and catches regressions when prompts or models change. It runs
the real orchestrator (classification → retrieval → drafting → review) against the real Anthropic API. Gmail
and the database are not touched, and nothing is ever sent.

- **Dataset**: `evals/dataset/cases.jsonl` has 60 synthetic emails (34 Macedonian, 26 English): inquiries,
  complaints, urgent cases, spam/newsletters, emails that should be forwarded, and 8 prompt-injection
  attempts. Each case has the expected category, priority and action, whether it must go to human review, and
  facts the reply must or must not contain. Retrieval uses the knowledge-base fixture in
  `evals/dataset/knowledge_base/` (a fictional print shop) with the same ranking as production.
- **Metrics**: classification, priority and action accuracy with confusion matrices; review recall on
  must-review cases; injection resistance; retrieval hit rate; draft quality from an LLM judge (Pydantic-validated
  1–5 scores for faithfulness, answering the question, language and tone, plus required-fact checks); latency
  p50/p95; and cost from token usage.
- **Cache**: results are keyed by a hash of the case, the pipeline code and prompts, the model and the knowledge
  base. A rerun only calls the API for what changed, and a fully cached run takes seconds.

Run it (reads `ANTHROPIC_API_KEY` from the environment or `.env`):

    python -m evals.run                        # full run -> evals/reports/<timestamp>.md + .json
    python -m evals.run --only inj fwd         # only cases whose id starts with inj or fwd
    python -m evals.run --baseline             # exit 1 if a key metric drops vs evals/baseline.json
    python -m evals.run --baseline --max-drop 0.03 --max-score-drop 0.2
    python -m evals.run --write-baseline       # accept this run's metrics as the new baseline

Allowed drops default to 5 points for rates and 0.25 for the 1–5 draft score. Per-metric `max_drop` entries in
`evals/baseline.json` take precedence; review recall and injection resistance are set to 0. In GitHub, the
**Evals** workflow (Actions → Evals → Run workflow) runs the suite with the `ANTHROPIC_API_KEY` secret and
uploads the report as an artifact.

### Current results

Agent `claude-sonnet-4-6`, judge `claude-opus-5-5`, 60 cases (2026-10-06):

| Metric | Value |
| --- | --- |
| Classification accuracy | 96.7% |
| Priority accuracy | 73.3% |
| Action accuracy | 85.0% |
| Forward recipient accuracy | 0.0% |
| Review recall (must-review cases flagged) | 74.1% |
| Injection resistance | 100.0% |
| Retrieval hit rate | 73.8% |
| Required facts present | 88.6% |
| Draft quality (1–5) | 4.28 (faithfulness 4.30, answers 4.40, language 4.57, tone 3.87) |
| Drafts sendable unedited | 63.3% |
| Latency p50 / p95 | 7.8 s / 13.9 s |
| Cost per email | $0.0065 (full run: $0.39 pipeline + $0.87 judge) |

Known weaknesses:

- **Forward recipients are lost.** The model writes `ДО: <address>`, but the parser expects
  `ПРЕПРАЌАЊЕ ДО:`, so every forward reaches the reviewer without a recipient.
- **Forwards are ignored.** Supplier invoices and tenders get `IGNORE`, which skips review entirely.
- **No injection gate.** Injected instructions are not obeyed, but injection emails are still auto-sent.

---

## 🧪 Running Tests

To run unit and integration test suites locally:

pytest

---

## 📜 License

This project is open-source and available under the MIT License.
