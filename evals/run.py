"""Run the email agent eval suite against the real Anthropic API.

    python -m evals.run                      # full run, cached
    python -m evals.run --baseline           # also fail (exit 1) on regressions vs evals/baseline.json
    python -m evals.run --write-baseline     # store this run's key metrics as the new baseline

Results are cached in evals/.cache by a hash of the case, the pipeline code and prompts, the model and the
knowledge base, so a rerun only calls the API for what changed.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import dotenv_values

from app.core.config import Settings
from app.services.embeddings import (
    DEFAULT_LOCAL_MODEL,
    DEFAULT_VOYAGE_MODEL,
    EmbeddingProvider,
    LocalEmbeddingProvider,
    VoyageEmbeddingProvider,
)
from evals.harness import (
    KnowledgeBase,
    ResultCache,
    judge_key,
    load_cases,
    needs_judging,
    pipeline_fingerprint,
    pipeline_key,
    run_judge,
    run_pipeline,
)
from evals.metrics import (
    DEFAULT_MAX_DROP,
    DEFAULT_MAX_SCORE_DROP,
    baseline_from_summary,
    compare_to_baseline,
    compute_summary,
    render_report,
)
from evals.schema import CaseResult, EvalCase, JudgeVerdict, PipelineOutput

EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_JUDGE_MODEL = "claude-opus-5-5"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=EVALS_DIR / "dataset" / "cases.jsonl")
    parser.add_argument("--knowledge-base", type=Path, default=EVALS_DIR / "dataset" / "knowledge_base")
    parser.add_argument("--model", default=None, help="Agent model (default: ANTHROPIC_MODEL or the app default)")
    parser.add_argument("--judge-model", default=os.environ.get("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--only", nargs="*", default=None, help="Run only cases whose id starts with one of these")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge scoring")
    parser.add_argument(
        "--embedding-provider",
        choices=["local", "voyage", "none"],
        default=None,
        help="Embeddings for retrieval (default: EMBEDDING_PROVIDER or local); none = full-text only",
    )
    parser.add_argument("--rerank", action="store_true", help="Rerank retrieval candidates with Claude")
    parser.add_argument("--rerank-model", default=os.environ.get("RERANK_MODEL", "claude-haiku-4-5"))
    parser.add_argument("--no-cache", action="store_true", help="Ignore and do not write cached results")
    parser.add_argument("--cache-dir", type=Path, default=EVALS_DIR / ".cache")
    parser.add_argument("--out-dir", type=Path, default=EVALS_DIR / "reports")
    parser.add_argument(
        "--baseline",
        type=Path,
        nargs="?",
        const=EVALS_DIR / "baseline.json",
        default=None,
        help="Compare against a baseline file (default evals/baseline.json) and exit 1 on regressions",
    )
    parser.add_argument("--max-drop", type=float, default=DEFAULT_MAX_DROP, help="Allowed drop for rate metrics")
    parser.add_argument(
        "--max-score-drop", type=float, default=DEFAULT_MAX_SCORE_DROP, help="Allowed drop of the 1-5 draft quality"
    )
    parser.add_argument("--write-baseline", action="store_true", help="Write this run's metrics to evals/baseline.json")
    return parser.parse_args(argv)


def _embedding_provider(args: argparse.Namespace) -> str:
    return str(args.embedding_provider or _env("EMBEDDING_PROVIDER") or "local")


def embedding_model_name(args: argparse.Namespace) -> str:
    if _embedding_provider(args) == "voyage":
        return _env("VOYAGE_MODEL") or DEFAULT_VOYAGE_MODEL
    return _env("LOCAL_EMBEDDING_MODEL") or DEFAULT_LOCAL_MODEL


def build_embedder(args: argparse.Namespace) -> EmbeddingProvider | None:
    provider = _embedding_provider(args)
    if provider == "none":
        return None
    if provider == "voyage":
        api_key = _env("VOYAGE_API_KEY")
        if not api_key:
            raise SystemExit("--embedding-provider voyage needs VOYAGE_API_KEY (environment or .env).")
        return VoyageEmbeddingProvider(api_key, embedding_model_name(args))
    return LocalEmbeddingProvider(embedding_model_name(args))


def _env(name: str) -> str | None:
    """Environment first, then the project's .env file. Values are never logged."""
    return os.environ.get(name) or dotenv_values(EVALS_DIR.parent / ".env").get(name) or None


async def evaluate_case(
    case: EvalCase,
    *,
    client: AsyncAnthropic,
    model: str,
    judge_model: str | None,
    kb: KnowledgeBase,
    fingerprint: str,
    rerank_model: str | None,
    cache: ResultCache,
    semaphore: asyncio.Semaphore,
) -> CaseResult:
    async with semaphore:
        result = CaseResult(case_id=case.id)
        key = pipeline_key(case, fingerprint)
        output = cache.get(key, PipelineOutput)
        result.pipeline_cached = output is not None
        if output is None:
            try:
                output = await run_pipeline(case, client, model, kb, rerank_model)
            except Exception as exc:  # report the failure and keep evaluating the other cases
                result.error = f"pipeline: {type(exc).__name__}: {exc}"
                return result
            cache.put(key, output)
        result.output = output

        if judge_model is None or not needs_judging(output):
            return result
        jkey = judge_key(case, output, judge_model)
        cached_verdict = cache.get(jkey, JudgeVerdict)
        if cached_verdict is not None:
            result.judge, result.judge_cached = cached_verdict, True
            return result
        try:
            result.judge, result.judge_calls = await run_judge(case, output, client, judge_model, kb)
        except Exception as exc:
            result.error = f"judge: {type(exc).__name__}: {exc}"
            return result
        cache.put(jkey, result.judge)
        return result


async def evaluate(args: argparse.Namespace, api_key: str, model: str) -> tuple[list[EvalCase], list[CaseResult]]:
    cases = load_cases(args.dataset)
    if args.only:
        cases = [c for c in cases if c.id.startswith(tuple(args.only))]
    kb = await KnowledgeBase.load(args.knowledge_base, build_embedder(args), args.cache_dir)
    rerank_model = args.rerank_model if args.rerank else None
    cache = ResultCache(None if args.no_cache else args.cache_dir)
    semaphore = asyncio.Semaphore(args.concurrency)
    fingerprint = pipeline_fingerprint(model, kb, rerank_model)

    async with AsyncAnthropic(api_key=api_key, max_retries=6) as client:
        done = 0

        async def tracked(case: EvalCase) -> CaseResult:
            nonlocal done
            result = await evaluate_case(
                case,
                client=client,
                model=model,
                judge_model=None if args.no_judge else args.judge_model,
                kb=kb,
                fingerprint=fingerprint,
                rerank_model=rerank_model,
                cache=cache,
                semaphore=semaphore,
            )
            done += 1
            status = "error" if result.error else ("cached" if result.pipeline_cached else "ok")
            print(f"[{done:>3}/{len(cases)}] {case.id:<8} {status}", file=sys.stderr)
            return result

        results = await asyncio.gather(*(tracked(case) for case in cases))
    return cases, list(results)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    api_key = _env("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY is not set (environment or .env).", file=sys.stderr)
        return 2
    model = args.model or _env("ANTHROPIC_MODEL") or str(Settings.model_fields["anthropic_model"].default)

    cases, results = asyncio.run(evaluate(args, api_key, model))

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary: dict[str, Any] = {
        "timestamp": timestamp,
        "model": model,
        "judge_model": None if args.no_judge else args.judge_model,
        "embedding_model": None if args.embedding_provider == "none" else embedding_model_name(args),
        "rerank_model": args.rerank_model if args.rerank else None,
        **compute_summary(cases, results),
    }
    summary["cost"]["fresh_api_calls_this_run"] = sum(
        len(r.output.calls) for r in results if r.output is not None and not r.pipeline_cached
    ) + sum(len(r.judge_calls) for r in results)

    comparison = None
    baseline: dict[str, Any] | None = None
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        comparison = compare_to_baseline(summary["metrics"], baseline, args.max_drop, args.max_score_drop)
        summary["baseline_comparison"] = comparison

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / f"{timestamp}.md"
    report_path.write_text(render_report(summary, cases, results, comparison), encoding="utf-8")
    json_path = args.out_dir / f"{timestamp}.json"
    json_path.write_text(
        json.dumps({**summary, "results": [r.model_dump(mode="json") for r in results]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Report: {report_path}\nSummary: {json_path}")

    if args.write_baseline:
        path = EVALS_DIR / "baseline.json"
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        path.write_text(json.dumps(baseline_from_summary(summary, previous), indent=2) + "\n", encoding="utf-8")
        print(f"Baseline written: {path}")

    if comparison is not None:
        regressions = [row for row in comparison if row["regressed"]]
        for row in regressions:
            print(f"REGRESSION {row['metric']}: {row['baseline']} -> {row['current']}", file=sys.stderr)
        if regressions:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
