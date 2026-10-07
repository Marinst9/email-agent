"""Retrieval eval: recall@k and MRR of keyword vs full-text vs vector vs hybrid (vs hybrid + rerank) search.

    python -m evals.retrieval                 # local embeddings, no API calls
    python -m evals.retrieval --rerank        # also hybrid + Claude reranking (API calls)

Cases: every eval case with required facts and knowledge-base references. A retrieved chunk is relevant
when it comes from one of the case's `kb_refs` files.
- recall@k: share of the case's relevant files that appear among the top k chunks.
- MRR: 1 / rank of the first relevant chunk among the top `--candidates` (0 if none).
- hit@k: at least one relevant chunk in the top k (what the main eval calls retrieval hit rate).
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic

from app.schemas.email import GmailMessage
from app.services.ai_agents import retrieval_query
from app.services.embeddings import LocalEmbeddingProvider
from app.services.retrieval import LLMReranker
from evals.harness import KnowledgeBase, load_cases
from evals.legacy_keyword import chunk_chars, legacy_query, rank_chunks
from evals.run import EVALS_DIR, _env
from evals.schema import EvalCase

Ranker = Callable[[EvalCase], Awaitable[list[str]]]  # case -> filenames of ranked chunks


def metrics(ranked_files: list[str], relevant: set[str], k: int) -> dict[str, float]:
    top = ranked_files[:k]
    first = next((i for i, name in enumerate(ranked_files, start=1) if name in relevant), None)
    return {
        "recall": len(relevant & set(top)) / len(relevant),
        "mrr": 1 / first if first else 0.0,
        "hit": float(any(name in relevant for name in top)),
    }


async def evaluate(cases: list[EvalCase], rankers: dict[str, Ranker], k: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, ranker in rankers.items():
        per_case = {case.id: metrics(await ranker(case), set(case.kb_refs), k) for case in cases}
        out[name] = {
            f"recall@{k}": round(statistics.mean(m["recall"] for m in per_case.values()), 4),
            "mrr": round(statistics.mean(m["mrr"] for m in per_case.values()), 4),
            f"hit@{k}": round(statistics.mean(m["hit"] for m in per_case.values()), 4),
            "per_case": per_case,
        }
        print(f"{name:<16} done", file=sys.stderr)
    return out


def render(results: dict[str, dict[str, Any]], k: int, n: int, model: str) -> str:
    lines = [
        f"# Retrieval eval — {n} cases, embeddings `{model}`",
        "",
        f"| Method | Recall@{k} | MRR | Hit@{k} |",
        "| --- | --- | --- | --- |",
    ]
    for name, r in results.items():
        lines.append(f"| {name} | {r[f'recall@{k}']:.3f} | {r['mrr']:.3f} | {r[f'hit@{k}']:.3f} |")
    return "\n".join(lines) + "\n"


async def main_async(args: argparse.Namespace) -> int:
    cases = [c for c in load_cases(args.dataset) if c.kb_refs and c.facts.must_contain]
    embedder = LocalEmbeddingProvider(args.embedding_model)
    kb = await KnowledgeBase.load(args.knowledge_base, embedder, args.cache_dir)
    index = kb.index
    legacy_chunks = [(name, chunk) for name, text in kb.documents.items() for chunk in chunk_chars(text)]

    def email(case: EvalCase) -> GmailMessage:
        return GmailMessage(id=case.id, thread_id=case.id, sender=case.sender, subject=case.subject, body=case.body)

    async def keyword(case: EvalCase) -> list[str]:
        ranked = rank_chunks(legacy_query(case.subject, case.body), [c for _, c in legacy_chunks], args.candidates)
        source = {chunk: name for name, chunk in legacy_chunks}
        return [source[c] for c in ranked]

    def mode_ranker(mode: Any) -> Ranker:
        async def rank(case: EvalCase) -> list[str]:
            docs = await index.with_mode(mode).ranked(retrieval_query(email(case)), args.candidates)
            return [d.filename or "?" for d in docs]

        return rank

    rankers: dict[str, Ranker] = {
        "keyword (before)": keyword,
        "full-text": mode_ranker("lexical"),
        "vector": mode_ranker("vector"),
        "hybrid (RRF)": mode_ranker("hybrid"),
    }
    client: AsyncAnthropic | None = None
    if args.rerank:
        api_key = _env("ANTHROPIC_API_KEY")
        if not api_key:
            print("--rerank needs ANTHROPIC_API_KEY (environment or .env).", file=sys.stderr)
            return 2
        client = AsyncAnthropic(api_key=api_key, max_retries=6)
        reranker = LLMReranker(client, args.rerank_model)

        async def reranked(case: EvalCase) -> list[str]:
            query = retrieval_query(email(case))
            candidates = await index.with_mode("hybrid").ranked(query, args.candidates)
            docs = await reranker.rerank(query, candidates, args.candidates)
            return [d.filename or "?" for d in docs]

        rankers[f"hybrid + rerank ({args.rerank_model})"] = reranked

    try:
        results = await evaluate(cases, rankers, args.k)
    finally:
        if client is not None:
            await client.close()
    report = render(results, args.k, len(cases), embedder.model_name)
    print(report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (args.out_dir / f"retrieval-{stamp}.md").write_text(report, encoding="utf-8")
    (args.out_dir / f"retrieval-{stamp}.json").write_text(
        json.dumps(
            {"k": args.k, "cases": len(cases), "embedding_model": embedder.model_name, "results": results}, indent=2
        ),
        encoding="utf-8",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=EVALS_DIR / "dataset" / "cases.jsonl")
    parser.add_argument("--knowledge-base", type=Path, default=EVALS_DIR / "dataset" / "knowledge_base")
    parser.add_argument(
        "--embedding-model", default=os.environ.get("LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-model", default="claude-haiku-4-5")
    parser.add_argument("--cache-dir", type=Path, default=EVALS_DIR / ".cache")
    parser.add_argument("--out-dir", type=Path, default=EVALS_DIR / "reports")
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
