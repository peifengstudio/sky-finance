"""Strategies CRUD + manual trigger routes."""

import json
import logging
import math
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse

from sky_finance.dashboard import queries
from sky_finance.dashboard._templates import templates
from sky_finance.storage.db import get_connection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/strategies")

_SCOPES = ["global", "group", "ticker"]
_TIERS = ["local", "nano", "advanced", "claude"]
_PAGE_SIZE = 15


def _build_stock_groups() -> dict[str, list[dict[str, Any]]]:
    """Return {market: [{ticker, name}, ...]} from config, sorted by ticker."""
    from sky_finance.config import list_stock_configs

    groups: dict[str, list[dict[str, Any]]] = {}
    for cfg in list_stock_configs():
        market = cfg.get("market", "us")
        groups.setdefault(market, []).append({"ticker": cfg["ticker"], "name": cfg.get("name", "")})
    for stocks in groups.values():
        stocks.sort(key=lambda s: s["ticker"])
    return dict(sorted(groups.items()))


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


@router.get("", response_class=HTMLResponse)
async def strategies_list(request: Request) -> HTMLResponse:
    with get_connection() as conn:
        strategies = queries.list_strategies(conn)
    return templates.TemplateResponse(
        request,
        "strategies.html",
        {
            "request": request,
            "strategies": strategies,
            "active_page": "strategies",
        },
    )


# ---------------------------------------------------------------------------
# Create (GET new / POST)
# ---------------------------------------------------------------------------


@router.get("/new", response_class=HTMLResponse)
async def strategy_new(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "strategy_detail.html",
        {
            "request": request,
            "strategy": None,
            "results": [],
            "scopes": _SCOPES,
            "tiers": _TIERS,
            "stock_groups": _build_stock_groups(),
            "active_page": "strategies",
        },
    )


@router.post("", response_class=HTMLResponse)
async def strategy_create(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    scope: str = Form("global"),
    scope_value: str = Form(""),
    rag_query_template: str = Form(""),
    prompt_template: str = Form(""),
    model_tier: str = Form("local"),
    schedule: str = Form(""),
    enabled: str = Form("off"),
    rag_threshold: float = Form(0.55),
    rag_top_k_positive: int = Form(20),
    rag_top_k_neutral: int = Form(20),
    rag_top_k_negative: int = Form(20),
    retrieval_mode: str = Form("hybrid"),
) -> RedirectResponse:
    data = {
        "name": name.strip(),
        "description": description.strip(),
        "scope": scope,
        "scope_value": scope_value.strip() or None,
        "rag_query_template": rag_query_template.strip(),
        "prompt_template": prompt_template.strip(),
        "model_tier": model_tier,
        "schedule": schedule.strip() or None,
        "enabled": enabled == "on",
        "rag_threshold": max(0.0, min(1.0, rag_threshold)),
        "rag_top_k_positive": max(0, rag_top_k_positive),
        "rag_top_k_neutral": max(0, rag_top_k_neutral),
        "rag_top_k_negative": max(0, rag_top_k_negative),
        "retrieval_mode": retrieval_mode if retrieval_mode in ("hybrid", "vector") else "hybrid",
    }
    with get_connection() as conn:
        strategy_id = queries.create_strategy(conn, data)
        conn.commit()
    return RedirectResponse(f"/strategies/{strategy_id}", status_code=303)


# ---------------------------------------------------------------------------
# Scheduler toggle  (MUST be before /{strategy_id})
# ---------------------------------------------------------------------------


def _scheduler_badge(enabled: bool) -> str:
    track = "bg-emerald-500" if enabled else "bg-slate-700"
    thumb = "translate-x-5 bg-white" if enabled else "translate-x-0.5 bg-slate-400"
    label = "text-slate-200" if enabled else "text-slate-500"
    tip = "Disable scheduled auto-run" if enabled else "Enable scheduled auto-run"
    return (
        f'<button id="scheduler-badge"'
        f' hx-post="/strategies/scheduler/toggle"'
        f' hx-target="#scheduler-badge"'
        f' hx-swap="outerHTML"'
        f' title="{tip}"'
        f' class="flex items-center gap-2.5 cursor-pointer group">'
        f'<span class="text-xs font-medium {label} select-none transition-colors">Scheduler</span>'
        f'<span class="relative inline-flex h-5 w-9 flex-shrink-0 items-center rounded-full'
        f' {track} transition-colors duration-200 ease-in-out">'
        f'<span class="inline-block h-3.5 w-3.5 rounded-full shadow'
        f' {thumb} transition-transform duration-200 ease-in-out"></span>'
        f"</span>"
        f"</button>"
    )


@router.get("/scheduler/status", response_class=HTMLResponse)
async def scheduler_status(request: Request) -> HTMLResponse:
    from sky_finance.strategies.tasks import scheduler_enabled

    return HTMLResponse(_scheduler_badge(scheduler_enabled()))


@router.post("/scheduler/toggle", response_class=HTMLResponse)
async def scheduler_toggle(request: Request) -> HTMLResponse:
    from sky_finance.strategies.tasks import scheduler_enabled, set_scheduler_enabled

    new_state = not scheduler_enabled()
    set_scheduler_enabled(new_state)
    logger.info("Scheduler %s via web UI", "enabled" if new_state else "disabled")
    return HTMLResponse(_scheduler_badge(new_state))


# ---------------------------------------------------------------------------
# All results — paginated table  (MUST be before /{strategy_id})
# ---------------------------------------------------------------------------


@router.get("/results", response_class=HTMLResponse)
async def all_results_table(
    request: Request,
    page: int = Query(default=1, ge=1),
) -> HTMLResponse:
    offset = (page - 1) * _PAGE_SIZE
    with get_connection() as conn:
        total = queries.count_strategy_results(conn)
        results = queries.list_strategy_results(conn, limit=_PAGE_SIZE, offset=offset)
    total_pages = max(1, math.ceil(total / _PAGE_SIZE))
    ctx = {
        "request": request,
        "results": results,
        "page": page,
        "total_pages": total_pages,
        "total": total,
        "active_page": "strategies",
    }
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(request, "partials/results_table.html", ctx)
    return templates.TemplateResponse(request, "strategy_results_page.html", ctx)


# ---------------------------------------------------------------------------
# Result detail page  (MUST be before /{strategy_id})
# ---------------------------------------------------------------------------


@router.get("/results/{result_id}", response_class=HTMLResponse)
async def result_detail(request: Request, result_id: int) -> HTMLResponse:
    with get_connection() as conn:
        result = queries.get_strategy_result(conn, result_id)
    if result is None:
        return HTMLResponse("Result not found", status_code=404)

    from sky_finance.config import load_stock_config

    ticker_names: dict[str, str] = {}
    for t in result["tickers"]:
        cfg = load_stock_config(t)
        if cfg:
            ticker_names[t] = cfg.get("name", "")

    return templates.TemplateResponse(
        request,
        "strategy_result_detail.html",
        {
            "request": request,
            "result": result,
            "ticker_names": ticker_names,
            "active_page": "strategies",
        },
    )


# ---------------------------------------------------------------------------
# Task status polling  (MUST be before /{strategy_id})
# ---------------------------------------------------------------------------


@router.get("/task/{task_id}/status", response_class=HTMLResponse)
async def task_status(request: Request, task_id: str) -> HTMLResponse:
    from celery.result import AsyncResult

    result = AsyncResult(task_id)
    state = result.state

    if state == "SUCCESS":
        return HTMLResponse('<span class="text-xs text-emerald-400">✓ Done</span>')
    if state == "FAILURE":
        err = str(result.result)[:80]
        return HTMLResponse(f'<span class="text-xs text-red-400" title="{err}">✗ Failed</span>')
    if state == "RETRY":
        label, colour = "⟳ Retrying…", "text-amber-400"
    elif state == "STARTED":
        label, colour = "⏳ Running…", "text-amber-400"
    else:
        label, colour = "⏳ Queued…", "text-slate-500"

    return HTMLResponse(f"""<span id="task-{task_id[:8]}" class="text-xs {colour}"
          hx-get="/strategies/task/{task_id}/status"
          hx-trigger="every 3s"
          hx-swap="outerHTML">{label}</span>""")


# ---------------------------------------------------------------------------
# Live streaming run — SSE  (must come before bare /{strategy_id})
# ---------------------------------------------------------------------------


@router.get("/{strategy_id}/run/stream")
async def strategy_run_stream(request: Request, strategy_id: int) -> StreamingResponse:
    """
    Server-Sent Events endpoint that streams the LLM output token-by-token.

    Event envelope (JSON in the `data:` field):
        {"type": "status",  "text": "…"}          — progress messages
        {"type": "chunk",   "text": "…"}           — one or more tokens
        {"type": "done",    "model": "…",
                            "tickers": […],
                            "result_id": 42,
                            "duration": 5.2}       — stream finished, result saved
        {"type": "error",   "message": "…"}        — unrecoverable failure
    """
    from sky_finance.config import load_stock_config
    from sky_finance.strategies.engine import (
        _load_model_cfg,
        astream_with_model,
        build_prompt,
        rag_fetch,
        resolve_tickers,
    )
    from sky_finance.strategies.repository import save_strategy_result

    with get_connection() as conn:
        strategy = queries.get_strategy(conn, strategy_id)
    if strategy is None:
        return StreamingResponse(
            iter([f"data: {json.dumps({'type': 'error', 'message': 'Strategy not found'})}\n\n"]),
            media_type="text/event-stream",
        )

    def _sse(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    async def generate() -> Any:
        try:
            yield _sse({"type": "status", "text": f"Resolving tickers for '{strategy['name']}'…"})

            # ── Phase 1: sync DB setup (no yield inside with-blocks) ──────────
            with get_connection() as conn:
                tickers = resolve_tickers(strategy)
                if not tickers:
                    yield _sse(
                        {"type": "error", "message": "No tickers resolved for this strategy."}
                    )
                    return
                ticker_names: dict[str, str] = {
                    t: ((load_stock_config(t) or {}).get("name", "")) for t in tickers
                }

            yield _sse(
                {"type": "status", "text": f"Fetching RAG context for {', '.join(tickers)}…"}
            )

            with get_connection() as conn:
                ticker_contexts: dict[str, str] = {}
                for ticker in tickers:
                    text, _ = rag_fetch(
                        conn,
                        strategy["rag_query_template"],
                        ticker,
                        company_name=ticker_names.get(ticker, ""),
                        threshold=strategy.get("rag_threshold", 0.55),
                        top_k_positive=strategy.get("rag_top_k_positive", 20),
                        top_k_neutral=strategy.get("rag_top_k_neutral", 20),
                        top_k_negative=strategy.get("rag_top_k_negative", 20),
                        retrieval_mode=strategy.get("retrieval_mode", "hybrid"),
                        rrf_k=strategy.get("rrf_k", 60),
                    )
                    ticker_contexts[ticker] = text

            user_content = build_prompt(
                strategy["prompt_template"],
                ticker_contexts,
                tickers,
                ticker_names=ticker_names,
            )
            model_id = _load_model_cfg(strategy["model_tier"])["model"]
            yield _sse({"type": "status", "text": f"Streaming {model_id}…"})

            # ── Phase 2: async model stream — each token is a real await point ─
            started_at = datetime.now(UTC)
            t0 = time.monotonic()
            full_chunks: list[str] = []
            usage_out: list[Any] = []

            async for chunk in astream_with_model(
                strategy["model_tier"],
                strategy["prompt_template"],
                user_content,
                usage_out=usage_out,
            ):
                full_chunks.append(chunk)
                yield _sse({"type": "chunk", "text": chunk})

            duration = round(time.monotonic() - t0, 1)
            report = "".join(full_chunks)

            # ── Phase 3: persist result ───────────────────────────────────────
            usage_dict = usage_out[0].to_dict() if usage_out else None
            with get_connection() as conn:
                result_id = save_strategy_result(
                    conn,
                    strategy_id=strategy_id,
                    strategy_name=strategy["name"],
                    tickers=tickers,
                    report=report,
                    model=model_id,
                    started_at=started_at,
                    duration_seconds=duration,
                    metadata={"streamed": True, "usage": usage_dict},
                )
                conn.commit()

            logger.info(
                "Stream complete strategy_id=%d result_id=%d model=%s tickers=%s duration=%.1fs",
                strategy_id,
                result_id,
                model_id,
                tickers,
                duration,
            )
            yield _sse(
                {
                    "type": "done",
                    "model": model_id,
                    "tickers": tickers,
                    "result_id": result_id,
                    "duration": duration,
                }
            )

        except Exception:
            logger.exception("Strategy stream failed strategy_id=%d", strategy_id)
            import traceback

            yield _sse({"type": "error", "message": traceback.format_exc(limit=3)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # prevent nginx from buffering SSE
        },
    )


# ---------------------------------------------------------------------------
# Detail / Edit  (parametric — must come after all literal GET paths)
# ---------------------------------------------------------------------------


@router.get("/{strategy_id}", response_class=HTMLResponse)
async def strategy_detail(request: Request, strategy_id: int) -> HTMLResponse:
    with get_connection() as conn:
        strategy = queries.get_strategy(conn, strategy_id)
        results = queries.list_strategy_results(conn, strategy_id=strategy_id, limit=20)
    if strategy is None:
        return HTMLResponse("Strategy not found", status_code=404)
    return templates.TemplateResponse(
        request,
        "strategy_detail.html",
        {
            "request": request,
            "strategy": strategy,
            "results": results,
            "scopes": _SCOPES,
            "tiers": _TIERS,
            "stock_groups": _build_stock_groups(),
            "active_page": "strategies",
        },
    )


@router.post("/{strategy_id}", response_class=HTMLResponse)
async def strategy_update(
    request: Request,
    strategy_id: int,
    name: str = Form(...),
    description: str = Form(""),
    scope: str = Form("global"),
    scope_value: str = Form(""),
    rag_query_template: str = Form(""),
    prompt_template: str = Form(""),
    model_tier: str = Form("local"),
    schedule: str = Form(""),
    enabled: str = Form("off"),
    rag_threshold: float = Form(0.55),
    rag_top_k_positive: int = Form(20),
    rag_top_k_neutral: int = Form(20),
    rag_top_k_negative: int = Form(20),
    retrieval_mode: str = Form("hybrid"),
) -> RedirectResponse:
    data = {
        "name": name.strip(),
        "description": description.strip(),
        "scope": scope,
        "scope_value": scope_value.strip() or None,
        "rag_query_template": rag_query_template.strip(),
        "prompt_template": prompt_template.strip(),
        "model_tier": model_tier,
        "schedule": schedule.strip() or None,
        "enabled": enabled == "on",
        "rag_threshold": max(0.0, min(1.0, rag_threshold)),
        "rag_top_k_positive": max(0, rag_top_k_positive),
        "rag_top_k_neutral": max(0, rag_top_k_neutral),
        "rag_top_k_negative": max(0, rag_top_k_negative),
        "retrieval_mode": retrieval_mode if retrieval_mode in ("hybrid", "vector") else "hybrid",
    }
    with get_connection() as conn:
        queries.update_strategy(conn, strategy_id, data)
        conn.commit()
    return RedirectResponse(f"/strategies/{strategy_id}", status_code=303)


# ---------------------------------------------------------------------------
# Delete / Run / Results partial
# ---------------------------------------------------------------------------


@router.post("/{strategy_id}/delete", response_class=HTMLResponse)
async def strategy_delete(request: Request, strategy_id: int) -> RedirectResponse:
    with get_connection() as conn:
        queries.delete_strategy(conn, strategy_id)
        conn.commit()
    return RedirectResponse("/strategies", status_code=303)


@router.post("/{strategy_id}/run", response_class=HTMLResponse)
async def strategy_run(request: Request, strategy_id: int) -> HTMLResponse:
    from sky_finance.strategies.tasks import run_strategy_task

    task = run_strategy_task.apply_async(args=[strategy_id])
    logger.info("Manually triggered strategy_id=%d task_id=%s", strategy_id, task.id)

    return HTMLResponse(f"""<span id="task-{task.id[:8]}" class="text-xs text-amber-400"
          hx-get="/strategies/task/{task.id}/status"
          hx-trigger="every 3s"
          hx-swap="outerHTML">⏳ Running…</span>""")


@router.get("/{strategy_id}/results", response_class=HTMLResponse)
async def strategy_results_partial(
    request: Request,
    strategy_id: int,
    page: int = Query(default=1, ge=1),
) -> HTMLResponse:
    offset = (page - 1) * _PAGE_SIZE
    with get_connection() as conn:
        total = queries.count_strategy_results(conn, strategy_id=strategy_id)
        results = queries.list_strategy_results(
            conn, strategy_id=strategy_id, limit=_PAGE_SIZE, offset=offset
        )
    total_pages = max(1, math.ceil(total / _PAGE_SIZE))
    return templates.TemplateResponse(
        request,
        "partials/strategy_results.html",
        {
            "request": request,
            "results": results,
            "strategy_id": strategy_id,
            "page": page,
            "total_pages": total_pages,
            "total": total,
        },
    )
