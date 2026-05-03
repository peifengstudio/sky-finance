"""
Strategy repository — DB CRUD for the strategies table and strategy_results.

All functions accept an open psycopg3 connection and do NOT commit.
The caller owns the transaction.
"""

import logging
from datetime import datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# strategies table
# ---------------------------------------------------------------------------


_SELECT_COLS = """
    id, name, description, scope, scope_value,
    rag_query_template, prompt_template, model_tier,
    schedule, enabled, created_at, updated_at,
    COALESCE(rag_threshold, 0.55)           AS rag_threshold,
    COALESCE(rag_top_k_positive, 20)        AS rag_top_k_positive,
    COALESCE(rag_top_k_neutral,  20)        AS rag_top_k_neutral,
    COALESCE(rag_top_k_negative, 20)        AS rag_top_k_negative,
    COALESCE(retrieval_mode, 'hybrid')      AS retrieval_mode
"""


def list_strategies(conn: psycopg.Connection, enabled_only: bool = False) -> list[dict[str, Any]]:
    where = "WHERE enabled = true" if enabled_only else ""
    with conn.cursor() as cur:
        cur.execute(f"SELECT {_SELECT_COLS} FROM strategies {where} ORDER BY name")
        rows = cur.fetchall()
    return [_row_to_strategy(r) for r in rows]


def get_strategy(conn: psycopg.Connection, strategy_id: int) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {_SELECT_COLS} FROM strategies WHERE id = %s",
            (strategy_id,),
        )
        row = cur.fetchone()
    return _row_to_strategy(row) if row else None


def create_strategy(conn: psycopg.Connection, data: dict[str, Any]) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO strategies
                (name, description, scope, scope_value,
                 rag_query_template, prompt_template, model_tier,
                 schedule, enabled,
                 rag_threshold, rag_top_k_positive, rag_top_k_neutral, rag_top_k_negative,
                 retrieval_mode)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """,
            (
                data["name"],
                data.get("description", ""),
                data.get("scope", "global"),
                data.get("scope_value") or None,
                data.get("rag_query_template", ""),
                data.get("prompt_template", ""),
                data.get("model_tier", "local"),
                data.get("schedule") or None,
                data.get("enabled", True),
                data.get("rag_threshold", 0.55),
                data.get("rag_top_k_positive", 20),
                data.get("rag_top_k_neutral", 20),
                data.get("rag_top_k_negative", 20),
                data.get("retrieval_mode", "hybrid"),
            ),
        )
        row = cur.fetchone()
        assert row is not None
        row_id: int = row[0]
    logger.info("Created strategy id=%d name=%r", row_id, data["name"])
    return row_id


def update_strategy(conn: psycopg.Connection, strategy_id: int, data: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE strategies SET
                name                = %s,
                description         = %s,
                scope               = %s,
                scope_value         = %s,
                rag_query_template  = %s,
                prompt_template     = %s,
                model_tier          = %s,
                schedule            = %s,
                enabled             = %s,
                rag_threshold       = %s,
                rag_top_k_positive  = %s,
                rag_top_k_neutral   = %s,
                rag_top_k_negative  = %s,
                retrieval_mode      = %s,
                updated_at          = now()
            WHERE id = %s
        """,
            (
                data["name"],
                data.get("description", ""),
                data.get("scope", "global"),
                data.get("scope_value") or None,
                data.get("rag_query_template", ""),
                data.get("prompt_template", ""),
                data.get("model_tier", "local"),
                data.get("schedule") or None,
                data.get("enabled", True),
                data.get("rag_threshold", 0.55),
                data.get("rag_top_k_positive", 20),
                data.get("rag_top_k_neutral", 20),
                data.get("rag_top_k_negative", 20),
                data.get("retrieval_mode", "hybrid"),
                strategy_id,
            ),
        )
    logger.info("Updated strategy id=%d", strategy_id)


def delete_strategy(conn: psycopg.Connection, strategy_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM strategies WHERE id = %s", (strategy_id,))
    logger.info("Deleted strategy id=%d", strategy_id)


def _row_to_strategy(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "name": row[1],
        "description": row[2],
        "scope": row[3],
        "scope_value": row[4],
        "rag_query_template": row[5],
        "prompt_template": row[6],
        "model_tier": row[7],
        "schedule": row[8],
        "enabled": row[9],
        "created_at": row[10],
        "updated_at": row[11],
        "rag_threshold": float(row[12]),
        "rag_top_k_positive": int(row[13]),
        "rag_top_k_neutral": int(row[14]),
        "rag_top_k_negative": int(row[15]),
        "retrieval_mode": row[16],
    }


# ---------------------------------------------------------------------------
# strategy_results table
# ---------------------------------------------------------------------------


def save_strategy_result(
    conn: psycopg.Connection,
    strategy_id: int,
    strategy_name: str,
    tickers: list[str],
    report: str,
    model: str,
    started_at: datetime | None = None,
    duration_seconds: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO strategy_results
                (strategy_id, strategy, tickers, report, model,
                 started_at, duration_seconds, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """,
            (
                strategy_id,
                strategy_name,
                tickers,
                report,
                model,
                started_at,
                duration_seconds,
                Jsonb(metadata or {}),
            ),
        )
        row = cur.fetchone()
        assert row is not None
        row_id: int = row[0]
    logger.info(
        "Saved strategy_result id=%d strategy=%r tickers=%s",
        row_id,
        strategy_name,
        tickers,
    )
    return row_id


def get_strategy_result(conn: psycopg.Connection, result_id: int) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, strategy_id, strategy, tickers, report, model, ran_at, metadata,
                   started_at, duration_seconds
            FROM strategy_results
            WHERE id = %s
            """,
            (result_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "strategy_id": row[1],
        "strategy": row[2],
        "tickers": row[3],
        "report": row[4],
        "model": row[5],
        "ran_at": row[6],
        "metadata": row[7],
        "started_at": row[8],
        "duration_seconds": row[9],
    }


def count_strategy_results(
    conn: psycopg.Connection,
    strategy_id: int | None = None,
) -> int:
    where = "WHERE strategy_id = %s" if strategy_id is not None else ""
    params = [strategy_id] if strategy_id is not None else []
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM strategy_results {where}", params)
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def list_strategy_results(
    conn: psycopg.Connection,
    strategy_id: int | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    params: list[Any] = []
    where = ""
    if strategy_id is not None:
        where = "WHERE strategy_id = %s"
        params.append(strategy_id)
    params.extend([limit, offset])

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, strategy_id, strategy, tickers, report, model, ran_at, metadata,
                   started_at, duration_seconds
            FROM strategy_results
            {where}
            ORDER BY ran_at DESC
            LIMIT %s OFFSET %s
        """,
            params,
        )
        rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "strategy_id": r[1],
            "strategy": r[2],
            "tickers": r[3],
            "report": r[4],
            "model": r[5],
            "ran_at": r[6],
            "metadata": r[7],
            "started_at": r[8],
            "duration_seconds": r[9],
        }
        for r in rows
    ]
