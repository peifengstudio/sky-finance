"""Evaluation dashboard routes — list results and detail view."""

import logging
from typing import Any

import psycopg
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from sky_finance.dashboard._templates import templates
from sky_finance.storage.db import get_connection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/eval")

_PAGE_SIZE = 30


# ---------------------------------------------------------------------------
# DB helpers (eval-specific, kept local to avoid polluting queries.py)
# ---------------------------------------------------------------------------


def _list_results(
    conn: psycopg.Connection, limit: int = 30, offset: int = 0
) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, strategy_id, strategy_name, ticker, query,
                   bucketed_n_chunks, plain_n_chunks,
                   bucketed_score, plain_score,
                   bucketed_scores, plain_scores,
                   winner, judge_reasoning, judge_model, ran_at
            FROM eval_results
            ORDER BY ran_at DESC
            LIMIT %s OFFSET %s
            """,
            (limit, offset),
        )
        rows = cur.fetchall()
    return [_row_to_dict(r) for r in rows]


def _get_result(conn: psycopg.Connection, eval_id: int) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, strategy_id, strategy_name, ticker, query,
                   bucketed_n_chunks, plain_n_chunks,
                   bucketed_report, plain_report,
                   bucketed_score, plain_score,
                   bucketed_scores, plain_scores,
                   winner, judge_reasoning, judge_model, ran_at
            FROM eval_results
            WHERE id = %s
            """,
            (eval_id,),
        )
        row = cur.fetchone()
    return _row_to_dict(row, include_reports=True) if row else None


def _summary_stats(conn: psycopg.Connection) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*)                                            AS total,
                COUNT(*) FILTER (WHERE winner = 'bucketed')        AS bucketed_wins,
                COUNT(*) FILTER (WHERE winner = 'plain')           AS plain_wins,
                COUNT(*) FILTER (WHERE winner = 'tie')             AS ties,
                ROUND(AVG(bucketed_score)::NUMERIC, 1)             AS avg_bucketed,
                ROUND(AVG(plain_score)::NUMERIC, 1)                AS avg_plain,
                ROUND(AVG(bucketed_score - plain_score)::NUMERIC, 1) AS avg_delta
            FROM eval_results
            """)
        row = cur.fetchone()
    if not row or row[0] == 0:
        return {
            "total": 0,
            "bucketed_wins": 0,
            "plain_wins": 0,
            "ties": 0,
            "avg_bucketed": 0,
            "avg_plain": 0,
            "avg_delta": 0,
            "win_rate_pct": 0,
        }
    total = row[0]
    bucketed_wins = row[1]
    return {
        "total": total,
        "bucketed_wins": bucketed_wins,
        "plain_wins": row[2],
        "ties": row[3],
        "avg_bucketed": float(row[4] or 0),
        "avg_plain": float(row[5] or 0),
        "avg_delta": float(row[6] or 0),
        "win_rate_pct": round(bucketed_wins / total * 100) if total else 0,
    }


def _row_to_dict(row: tuple[Any, ...], include_reports: bool = False) -> dict[str, Any]:
    keys = [
        "id",
        "strategy_id",
        "strategy_name",
        "ticker",
        "query",
        "bucketed_n_chunks",
        "plain_n_chunks",
    ]
    if include_reports:
        keys += ["bucketed_report", "plain_report"]
    keys += [
        "bucketed_score",
        "plain_score",
        "bucketed_scores",
        "plain_scores",
        "winner",
        "judge_reasoning",
        "judge_model",
        "ran_at",
    ]
    return dict(zip(keys, row))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("", response_class=HTMLResponse)
async def eval_list(request: Request) -> HTMLResponse:
    with get_connection() as conn:
        stats = _summary_stats(conn)
        results = _list_results(conn, limit=_PAGE_SIZE)
    return templates.TemplateResponse(
        request,
        "eval.html",
        {
            "request": request,
            "stats": stats,
            "results": results,
            "active_page": "eval",
        },
    )


@router.get("/{eval_id}", response_class=HTMLResponse)
async def eval_detail(request: Request, eval_id: int) -> HTMLResponse:
    with get_connection() as conn:
        result = _get_result(conn, eval_id)
    if result is None:
        return HTMLResponse("Eval result not found", status_code=404)
    return templates.TemplateResponse(
        request,
        "eval_result.html",
        {
            "request": request,
            "result": result,
            "active_page": "eval",
        },
    )
