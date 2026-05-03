"""
Evaluation runner — orchestrates one full eval run for a strategy.

For each ticker in scope:
  1. Bucketed retrieval (current production approach) → run model → report A
  2. Plain retrieval   (baseline, no sentiment filter) → run model → report B
  3. LLM judge scores both on faithfulness / coverage / actionability
  4. Save result to eval_results table

The two reports are generated with *identical* prompts and the same model tier
so the only variable is the retrieval strategy.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from sky_finance.config import load_stock_config
from sky_finance.evaluation.judge import judge, score_avg
from sky_finance.evaluation.retrieval import plain_rag_fetch
from sky_finance.storage.db import get_connection
from sky_finance.strategies.engine import (
    build_prompt,
    rag_fetch,
    resolve_tickers,
    run_with_model,
)
from sky_finance.strategies.repository import get_strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_eval(
    strategy_id: int,
    ticker: str | None = None,
    judge_model: str = "claude-sonnet-4-6",
    model_tier: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Run evaluation for one strategy (all tickers, or a single ticker).

    Args:
        strategy_id: Strategy to evaluate.
        ticker: Evaluate a single ticker; ``None`` evaluates all tickers in scope.
        judge_model: Anthropic model used to score the two reports.
        model_tier: Override the strategy's model tier for report generation.
            Useful when the strategy uses a paid tier (``nano``, ``advanced``,
            ``claude``) but you want to run evals with the free local model.
            Defaults to the strategy's own ``model_tier``.

    Returns a list of result dicts — one per ticker evaluated.
    Results are also persisted to the ``eval_results`` table.
    """
    with get_connection() as conn:
        strategy = get_strategy(conn, strategy_id)
    if not strategy:
        raise ValueError(f"Strategy {strategy_id} not found")

    tickers = [ticker] if ticker else resolve_tickers(strategy)
    if not tickers:
        raise ValueError(f"No tickers resolved for strategy {strategy_id}")
    if limit is not None:
        tickers = tickers[:limit]

    effective_tier = model_tier or strategy["model_tier"]
    logger.info(
        "Starting eval | strategy=%r | tickers=%s | report_tier=%s | judge=%s",
        strategy["name"],
        tickers,
        effective_tier,
        judge_model,
    )

    results = []
    for t in tickers:
        try:
            result = _eval_ticker(strategy, t, judge_model, effective_tier)
            results.append(result)
        except Exception as exc:
            logger.error(
                "Skipping ticker=%s — eval failed: %s: %s",
                t,
                type(exc).__name__,
                exc,
            )

    bucketed_wins = sum(1 for r in results if r["winner"] == "bucketed")
    plain_wins = sum(1 for r in results if r["winner"] == "plain")
    logger.info(
        "Eval complete | strategy=%r | bucketed_wins=%d | plain_wins=%d | ties=%d",
        strategy["name"],
        bucketed_wins,
        plain_wins,
        len(results) - bucketed_wins - plain_wins,
    )
    return results


# ---------------------------------------------------------------------------
# Per-ticker eval
# ---------------------------------------------------------------------------


def _eval_ticker(
    strategy: dict[str, Any], ticker: str, judge_model: str, model_tier: str
) -> dict[str, Any]:
    cfg = load_stock_config(ticker)
    company = cfg.get("name", "") if cfg else ""
    query = strategy["rag_query_template"].replace("{ticker}", ticker)

    logger.info(
        "Evaluating ticker=%s strategy=%r tier=%s",
        ticker,
        strategy["name"],
        model_tier,
    )

    # Total chunk budget — same for both methods so the comparison is fair
    total_k = (
        strategy.get("rag_top_k_positive", 20)
        + strategy.get("rag_top_k_neutral", 20)
        + strategy.get("rag_top_k_negative", 20)
    )
    threshold = strategy.get("rag_threshold", 0.55)

    # ── Retrieval ──────────────────────────────────────────────────────────
    with get_connection() as conn:
        bucketed_ctx, bucketed_rows = rag_fetch(
            conn,
            strategy["rag_query_template"],
            ticker,
            company_name=company,
            threshold=threshold,
            top_k_positive=strategy.get("rag_top_k_positive", 20),
            top_k_neutral=strategy.get("rag_top_k_neutral", 20),
            top_k_negative=strategy.get("rag_top_k_negative", 20),
        )
        plain_ctx, plain_rows = plain_rag_fetch(
            conn,
            strategy["rag_query_template"],
            ticker,
            company_name=company,
            threshold=threshold,
            top_k=total_k,
        )

    # ── Sentiment distribution of bucketed vs plain (logged for insight) ──
    if bucketed_rows:
        b_dist = _sentiment_dist(bucketed_rows)
        logger.debug("Bucketed sentiment dist: %s", b_dist)
    if plain_rows:
        p_dist = _sentiment_dist(plain_rows)
        logger.debug("Plain sentiment dist: %s", p_dist)

    # ── Report generation (same model tier, same system prompt) ───────────
    system_prompt = strategy["prompt_template"]

    prompt_a = build_prompt(system_prompt, {ticker: bucketed_ctx}, [ticker], {ticker: company})
    report_a, model_id, _ = run_with_model(model_tier, system_prompt, prompt_a)

    prompt_b = build_prompt(system_prompt, {ticker: plain_ctx}, [ticker], {ticker: company})
    report_b, _, _2 = run_with_model(model_tier, system_prompt, prompt_b)

    # ── Judge ──────────────────────────────────────────────────────────────
    verdict = judge(
        ticker=ticker,
        company=company,
        query=query,
        report_a=report_a,
        report_b=report_b,
        a_chunks=len(bucketed_rows),
        b_chunks=len(plain_rows),
        model=judge_model,
    )

    scores_a = verdict.get("A", {})
    scores_b = verdict.get("B", {})
    winner_letter = verdict.get("winner", "tie")
    winner = "bucketed" if winner_letter == "A" else "plain" if winner_letter == "B" else "tie"

    row: dict[str, Any] = {
        "strategy_id": strategy["id"],
        "strategy_name": strategy["name"],
        "ticker": ticker,
        "query": query,
        "bucketed_n_chunks": len(bucketed_rows),
        "plain_n_chunks": len(plain_rows),
        "bucketed_report": report_a,
        "plain_report": report_b,
        "bucketed_score": score_avg(scores_a),
        "plain_score": score_avg(scores_b),
        "bucketed_scores": scores_a,
        "plain_scores": scores_b,
        "winner": winner,
        "judge_reasoning": verdict.get("reasoning", ""),
        "judge_model": judge_model,
        "model_used": model_id,
        "ran_at": datetime.now(UTC),
    }

    with get_connection() as conn:
        row["id"] = _save(conn, row)
        conn.commit()

    logger.info(
        "Ticker eval done | ticker=%s winner=%s bucketed=%.1f plain=%.1f chunks=%d/%d",
        ticker,
        winner,
        row["bucketed_score"],
        row["plain_score"],
        len(bucketed_rows),
        len(plain_rows),
    )
    return row


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------


def _save(conn: psycopg.Connection, row: dict[str, Any]) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO eval_results
                (strategy_id, strategy_name, ticker, query,
                 bucketed_n_chunks, plain_n_chunks,
                 bucketed_report, plain_report,
                 bucketed_score, plain_score,
                 bucketed_scores, plain_scores,
                 winner, judge_reasoning, judge_model, ran_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (
                row["strategy_id"],
                row["strategy_name"],
                row["ticker"],
                row["query"],
                row["bucketed_n_chunks"],
                row["plain_n_chunks"],
                row["bucketed_report"],
                row["plain_report"],
                row["bucketed_score"],
                row["plain_score"],
                Jsonb(row["bucketed_scores"]),
                Jsonb(row["plain_scores"]),
                row["winner"],
                row["judge_reasoning"],
                row["judge_model"],
                row["ran_at"],
            ),
        )
        result_row = cur.fetchone()
        assert result_row is not None
        return int(result_row[0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sentiment_dist(rows: list[dict[str, Any]]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for r in rows:
        s = r.get("sentiment", "unknown") or "unknown"
        dist[s] = dist.get(s, 0) + 1
    return dist
