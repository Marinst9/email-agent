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
   - Health Check: http://localhost:8000/health

---

## 🧪 Running Tests

To run unit and integration test suites locally:

pytest

---

## 📜 License

This project is open-source and available under the MIT License.
