# AI Email Assistant

A multi-agent email assistant built with **FastAPI**, **Claude** (Anthropic API), **PostgreSQL + pgvector**,
**Celery/Redis** and **Docker**. It reads a Gmail inbox, classifies each email, retrieves relevant passages from
the user's knowledge base, drafts a reply (or a forward) that cites those passages, and either sends it or
queues it for human review.

---

## 🌟 Features

- **Multi-agent pipeline** (Celery worker): classification → hybrid retrieval → drafting with citations →
  rule-based review.
- **Hybrid retrieval (RAG)**: pgvector cosine search and PostgreSQL full-text search, fused with Reciprocal
  Rank Fusion in a single SQL query; optional reranking with Claude. See [Knowledge base & retrieval](#-knowledge-base--retrieval).
- **Cited drafts**: the drafting agent names the knowledge-base chunks it used, and the review dashboard shows
  those snippets with file name and page.
- **Human-in-the-loop review**: forwards, complaints, urgent or high-priority emails, replies that cite no
  knowledge-base source, and non-automated email the AI wants to ignore all wait for a reviewer.
- **Idempotent processing**: status checkpoints and atomic claims, so a retried task never sends an email twice.
- **Per-user data**: knowledge-base documents and emails are stored and searched per user.
- **Evaluation suite**: 60-case eval with an LLM judge and baseline regression checks, plus a retrieval eval
  (see [Evaluation](#-evaluation)).
- **CI**: ruff, mypy, pytest, a PostgreSQL + pgvector integration job and a Docker build on GitHub Actions.

---

## 🏗️ System Architecture

    [Gmail] ──poll──► [FastAPI web] ──queue──► [Celery worker]
                           │                        │  classify (Claude)
                           │ upload documents       │  hybrid search (pgvector + full-text, RRF)
                           ▼                        │  draft with citations (Claude)
                  [PostgreSQL + pgvector] ◄─────────┤  review rules
                           ▲                        │  send, or wait for review
                           └── embed pending chunks ┘

---

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python 3.11+)
- **LLM**: Anthropic API, default model `claude-sonnet-4-6` (`ANTHROPIC_MODEL`)
- **Database**: PostgreSQL with the `pgvector` extension, SQLAlchemy 2.0 (async) and Alembic
- **Embeddings**: `intfloat/multilingual-e5-large` via sentence-transformers (default), or Voyage AI `voyage-4`
- **Task queue**: Celery + Redis
- **Containers**: multi-stage Docker build and `docker-compose`
- **CI**: GitHub Actions

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

## 📚 Knowledge base & retrieval

Users upload PDF or TXT documents on the **Knowledge** page. Retrieval works like this:

1. **Upload (web request)**: text is extracted per PDF page and split on paragraph and sentence boundaries into
   chunks of at most 200 approximate tokens (words and punctuation), with 40 tokens of overlap. A section heading
   is repeated at the top of every chunk of its section. Each chunk keeps `filename`, `chunk_index` and `page`
   (PDFs). Chunks are stored with `embedding_status = 'pending'`, and the upload queues a Celery task.
2. **Embedding (worker)**: the `knowledge.embed_pending` task embeds pending chunks in batches (row locks with
   `SKIP LOCKED`; at most 256 chunks per run, then it re-queues itself). Re-running it is safe.
3. **Search (worker, while drafting)**: one SQL query takes the top 20 chunks by pgvector cosine distance (HNSW
   index) and the top 20 by full-text rank (`ts_rank_cd` over a generated `tsvector` column with a GIN index),
   then merges both lists with Reciprocal Rank Fusion (k = 60). Each result carries its RRF score, cosine
   similarity and text rank. Chunks that are still pending are found by full-text search only.
4. **Reranking (optional)**: with `RERANK_ENABLED=true`, Claude (`RERANK_MODEL`, default `claude-haiku-4-5`)
   scores the top 20 fused chunks for relevance and the best 3 are used.
5. **Drafting**: the draft agent sees the chunks with their ids and must list the ids it used (`ИЗВОРИ:`). Ids
   that were not retrieved are dropped. The review dashboard shows the cited snippets. An AI reply that cites
   no source goes to human review.

Full-text search uses the `simple` configuration (lower-casing, no stemming: PostgreSQL has no Macedonian
stemmer). Query words of 6+ letters are matched as prefixes (`рекламации` → `реклама:*`), and common
Macedonian and English function words are dropped. Lower-casing Cyrillic needs a UTF-8 database locale
(`en_US.utf8`, as in the `pgvector/pgvector` image, or a `C.UTF-8`/ICU locale). Under the plain `C` locale,
capitalized Macedonian words are found by vector search only, and migration 0006 logs a warning.

| Setting | Default | |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers) or `voyage` |
| `LOCAL_EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | 1024 dimensions, ~2.2 GB download on first use, ~2.3 GB RAM |
| `VOYAGE_API_KEY`, `VOYAGE_MODEL` | none, `voyage-4` | API embeddings at 1024 dimensions |
| `RETRIEVAL_TOP_K`, `RETRIEVAL_CANDIDATES` | `3`, `20` | chunks given to the drafter; candidates per ranking |
| `RERANK_ENABLED`, `RERANK_MODEL`, `RERANK_CANDIDATES` | `false`, `claude-haiku-4-5`, `20` | optional reranking |
| `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_TOKENS` | `200`, `40` | chunk size and overlap |

The `embedding vector(1024)` column fits both providers. Switching provider needs no migration, but existing
chunks must be re-embedded: set them back to `pending` and run the task. After deploying migration 0006,
existing chunks are `pending` (still in the old 500-character chunks); embed them with:

    celery -A app.worker.celery_app call knowledge.embed_pending

The local model needs the optional dependencies in `requirements-local-embeddings.txt` (CPU PyTorch and
sentence-transformers); the Docker image installs them.

---

## 🚂 Deploying on Railway

The project needs four Railway services in one project:

| Service    | Source                     | Config file                                             |
| ---------- | -------------------------- | ------------------------------------------------------- |
| `web`      | this repo                  | `railway.toml` (default; runs migrations, then uvicorn) |
| `worker`   | this repo                  | `railway.worker.toml` (Celery worker, no public port)   |
| PostgreSQL | Railway **pgvector** template | must include the `vector` extension (see below)      |
| Redis      | Railway Redis              |                                                         |

To add the worker:

1. In the project, **New → GitHub Repo** and pick this repository again; name the service `worker`.
2. In the worker's **Settings → Config-as-code**, set **Railway Config File** to `/railway.worker.toml`.
3. Give the worker the same variables as `web`: `DATABASE_URL` and `REDIS_URL` (as references to the
   PostgreSQL and Redis services), `SECRET_KEY`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`,
   `ANTHROPIC_API_KEY` (and `ANTHROPIC_MODEL` if you override it), plus any retrieval settings
   (`EMBEDDING_PROVIDER`, `VOYAGE_API_KEY`, ...). Shared variables make this easier.
4. Do not generate a public domain for the worker; it only talks to Redis and PostgreSQL.

**PostgreSQL must have pgvector.** Migration 0006 runs `CREATE EXTENSION vector`, and the web service runs
migrations on deploy, so a database without the extension fails the deploy. Railway's standard PostgreSQL
service may not ship pgvector; use Railway's pgvector template (the `pgvector/pgvector` image) or check with
`SELECT * FROM pg_available_extensions WHERE name = 'vector'`.

**Worker memory.** The default local embedding model needs about 2.3 GB of RAM in the worker and downloads
about 2.2 GB on first use. On a small worker, use `EMBEDDING_PROVIDER=voyage` instead.

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
  facts the reply must or must not contain. Retrieval runs over the knowledge-base fixture in
  `evals/dataset/knowledge_base/` (a fictional print shop) in an in-memory index (`evals/hybrid_index.py`), so no
  PostgreSQL is needed. It uses the production chunker, embedding model, query terms and RRF; vector search is
  exact cosine, and full-text ranking approximates `ts_rank_cd` (distinct matched terms, then total matches).
- **Metrics**: classification, priority and action accuracy with confusion matrices; review recall on
  must-review cases; injection resistance; retrieval hit rate; citation rate and precision; draft quality from an
  LLM judge (Pydantic-validated
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
    python -m evals.run --rerank               # with Claude reranking of the retrieval candidates
    python -m evals.run --embedding-provider none   # full-text retrieval only (no embedding model)
    python -m evals.retrieval                  # retrieval-only comparison, no API calls
    python -m evals.retrieval --rerank         # ... plus hybrid + Claude reranking (API calls)

Allowed drops default to 5 points for rates and 0.25 for the 1–5 draft score. Per-metric `max_drop` entries in
`evals/baseline.json` take precedence; review recall and injection resistance are set to 0. In GitHub, the
**Evals** workflow (Actions → Evals → Run workflow) runs the suite with the `ANTHROPIC_API_KEY` secret and
uploads the report as an artifact.

### Retrieval

`evals/retrieval.py` scores retrieval alone on the 33 cases that have required facts and knowledge-base
references. A chunk counts as relevant when it comes from one of the case's reference files. Recall@3 is the
share of those files found in the top 3, MRR uses the first relevant chunk in the top 20, and Hit@3 means at
least one relevant chunk in the top 3. Embeddings: `intfloat/multilingual-e5-large` (2026-10-07):

| Method | Recall@3 | MRR | Hit@3 |
| --- | --- | --- | --- |
| Keyword (the previous retrieval) | 0.727 | 0.715 | 0.849 |
| Full-text only | 0.788 | 0.855 | 0.879 |
| Vector only | 0.864 | 0.842 | 0.879 |
| **Hybrid (RRF), the default** | **0.894** | 0.844 | **0.939** |
| Hybrid + rerank (`claude-haiku-4-5`) | 1.000 | 0.939 | 1.000 |

The fixture knowledge base is small (9 chunks), so the reranker sees every chunk as a candidate; expect a
smaller gain on a real knowledge base.

### Current results

These are the results of the last complete run, which used the previous keyword retrieval. A full rerun with
hybrid retrieval has not completed yet, so they do not reflect hybrid retrieval or citations.

Agent `claude-sonnet-4-6`, judge `claude-opus-5-5`, 60 cases (2026-10-06, after the forward-parsing and
ignore-gate fixes):

| Metric | Value |
| --- | --- |
| Classification accuracy | 96.7% |
| Priority accuracy | 73.3% |
| Action accuracy | 85.0% |
| Forward recipient accuracy | 22.2% |
| Review recall (must-review cases flagged) | 85.2% |
| Injection resistance | 100.0% |
| Retrieval hit rate | 73.8% |
| Required facts present | 88.6% |
| Draft quality (1–5) | 4.28 (faithfulness 4.23, answers 4.30, language 4.67, tone 3.93) |
| Drafts sendable unedited | 56.7% |
| Latency p50 / p95 | 7.5 s / 14.2 s |
| Cost per email | $0.0064 (full run: $0.38 pipeline + $0.91 judge) |

Known weaknesses:

- **No injection gate.** Injected instructions are not obeyed, but injection emails are still auto-sent
  (3 of the 4 remaining missed reviews).
- **Business email gets `IGNORE` or the wrong action.** Invoices, tenders and payment reminders are labelled
  `IGNORE` (they now go to review instead of being dropped), and a large catalog quote gets a reply instead of a
  forward. A job application with an injected instruction is classified as spam and ignored.
- **Retrieval missed the right document in about a quarter of cases** with the keyword ranking measured here.
  When `routing.md` was not retrieved, a forward had no recipient, and complaint replies left out the
  reprint-or-refund policy and the 7-day complaint window. Hybrid retrieval targets this (see the retrieval table
  above); its effect on these cases is not yet measured by a full run.
- **Urgent replies have the wrong tone.** They ask the customer to call during business hours instead of
  promising a callback.

---

## 🧪 Running Tests

    pytest

`tests/test_knowledge_postgres.py` runs the hybrid-search SQL against a real PostgreSQL with pgvector. It is
skipped unless `TEST_DATABASE_URL` points at a migrated database, for example:

    docker compose up -d db && alembic upgrade head
    TEST_DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/email_agent pytest tests/test_knowledge_postgres.py

The `postgres-integration` CI job runs it against `pgvector/pgvector:pg16`, after migrating up, down one step
and up again.

---

## 📜 License

This project is open-source and available under the MIT License.
