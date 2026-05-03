"""
Strategy engine — resolve tickers, fetch RAG context, call model, return report.

Entry point: run_strategy(strategy_id)
"""

import json
import logging
import time
import tomllib
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg

from sky_finance.config import list_stock_configs, load_stock_config
from sky_finance.strategies.costs import UsageStats, compute_cost

logger = logging.getLogger(__name__)

_SETTINGS_PATH = Path(__file__).parents[3] / "config" / "settings.toml"


def _load_model_cfg(tier: str) -> dict[str, Any]:
    with _SETTINGS_PATH.open("rb") as f:
        cfg = tomllib.load(f)
    models = cfg.get("models", {})
    if tier not in models:
        raise ValueError(f"Unknown model tier {tier!r} — expected one of {list(models)}")
    result: dict[str, Any] = models[tier]
    return result


# ---------------------------------------------------------------------------
# Ticker resolution
# ---------------------------------------------------------------------------


def resolve_tickers(strategy: dict[str, Any]) -> list[str]:
    """
    Return the list of tickers a strategy should run against.

    scope='global'  → all enabled tickers
    scope='group'   → scope_value is a market name ('us' | 'japan')
    scope='ticker'  → scope_value is a single ticker symbol
    """
    scope = strategy.get("scope", "global")
    value = strategy.get("scope_value")

    if scope == "global":
        return [c["ticker"] for c in list_stock_configs()]
    elif scope == "group":
        if not value:
            return []
        # Market name → filter by market (e.g. "us" or "japan")
        if value in ("us", "japan"):
            return [c["ticker"] for c in list_stock_configs(market=value)]
        # Comma-separated list of specific tickers
        enabled = {c["ticker"] for c in list_stock_configs()}
        return [t.strip() for t in value.split(",") if t.strip() in enabled]
    elif scope == "ticker":
        if not value:
            raise ValueError("scope='ticker' requires scope_value to be set")
        cfg = load_stock_config(value)
        return [cfg["ticker"]] if cfg and cfg.get("enabled", True) else []
    else:
        raise ValueError(f"Unknown scope {scope!r}")


# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------


_RAG_QUERY = """
    SELECT d.id, d.title, LEFT(d.body, 400), d.sentiment,
           1 - (e.embedding <=> %(vec)s::vector) AS score
    FROM embeddings e
    JOIN documents d ON d.id = e.document_id
    WHERE e.ticker = %(ticker)s
      AND d.sentiment = %(sentiment)s
      AND 1 - (e.embedding <=> %(vec)s::vector) >= %(threshold)s
    ORDER BY e.embedding <=> %(vec)s::vector
    LIMIT %(top_k)s
"""

# BM25-style keyword search via PostgreSQL full-text search.
# body_tsv is a GENERATED ALWAYS AS STORED tsvector column (migration 007).
# ts_rank_cd uses cover density — rewards documents where query terms appear
# close together, which tends to surface more relevant financial news chunks.
_BM25_QUERY = """
    SELECT d.id, d.title, LEFT(d.body, 400), d.sentiment,
           ts_rank_cd(d.body_tsv, plainto_tsquery('english', %(query_text)s)) AS score
    FROM documents d
    WHERE d.ticker = %(ticker)s
      AND d.sentiment = %(sentiment)s
      AND d.body_tsv @@ plainto_tsquery('english', %(query_text)s)
    ORDER BY score DESC
    LIMIT %(top_k)s
"""


def _rrf_fuse(
    vec_rows: list[tuple[Any, ...]],
    bm25_rows: list[tuple[Any, ...]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[tuple[Any, ...]]:
    """
    Reciprocal Rank Fusion of a vector-search list and a BM25 list.

    Each input row is (id, title, body, sentiment, score).
    RRF formula: score(d) = Σ  1 / (k + rank_i(d))

    k=60 is the standard default from the original RRF paper (Cormack 2009).
    A document that appears in only one list still contributes its single-list
    term — it is not penalised for being absent from the other.

    Returns rows sorted by descending RRF score, limited to top_k if given.
    Each returned row has the same shape: (id, title, body, sentiment, rrf_score).
    """
    scores: dict[int, dict[str, Any]] = {}

    for rank, (doc_id, title, body, sentiment, _) in enumerate(vec_rows, 1):
        scores[doc_id] = {"title": title, "body": body, "sentiment": sentiment, "rrf": 0.0}
        scores[doc_id]["rrf"] += 1.0 / (k + rank)

    for rank, (doc_id, title, body, sentiment, _) in enumerate(bm25_rows, 1):
        if doc_id not in scores:
            scores[doc_id] = {"title": title, "body": body, "sentiment": sentiment, "rrf": 0.0}
        scores[doc_id]["rrf"] += 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1]["rrf"], reverse=True)
    if top_k is not None:
        fused = fused[:top_k]
    return [
        (doc_id, d["title"], d["body"], d["sentiment"], round(d["rrf"], 4)) for doc_id, d in fused
    ]


def rag_fetch(
    conn: psycopg.Connection,
    query_template: str,
    ticker: str,
    company_name: str = "",
    threshold: float = 0.55,
    top_k_positive: int = 20,
    top_k_neutral: int = 20,
    top_k_negative: int = 20,
    retrieval_mode: str = "hybrid",
    rrf_k: int = 60,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Retrieve top-k chunks per sentiment bucket and return them as a formatted
    prompt string plus a list of structured dicts for dashboard / metadata.

    retrieval_mode='hybrid'  — BM25 keyword search + vector cosine search,
                               re-ranked with Reciprocal Rank Fusion (k=rrf_k).
    retrieval_mode='vector'  — pure cosine similarity (legacy behaviour).

    Why sentiment buckets instead of a single top-k query?
    A stock in a bull run may have 90 % positive articles.  A naïve global
    top-k would fill the context window with bullish news and bury the few
    negative risk signals that matter most for downside analysis.  Running
    three independent queries — one per sentiment — guarantees that minority
    sentiment always reaches the model, regardless of corpus composition.

    Why threshold=0.55 as the default?
    Empirically calibrated against nomic-embed-text 768-dim vectors on
    financial news: below 0.55 the retrieved chunks are mostly off-topic macro
    articles that share a handful of keywords with the query but carry no
    ticker-specific signal.  The sweet spot for on-topic context is 0.55–0.85.
    Strategies that need broader macro coverage can lower it; tight-signal
    strategies should raise it to 0.65+.
    """
    from sky_finance.storage.embedder import embed_single

    query = query_template.replace("{ticker}", ticker)
    if company_name:
        query = f"{query} {company_name}"
    vector = embed_single(query)

    from pgvector.psycopg import register_vector

    register_vector(conn)

    buckets = [
        ("positive", top_k_positive),
        ("neutral", top_k_neutral),
        ("negative", top_k_negative),
    ]

    raw: list[tuple[Any, ...]] = []
    with conn.cursor() as cur:
        for sentiment, top_k in buckets:
            if top_k <= 0:
                continue
            cur.execute(
                _RAG_QUERY,
                {
                    "vec": vector,
                    "ticker": ticker,
                    "sentiment": sentiment,
                    "threshold": threshold,
                    "top_k": top_k,
                },
            )
            vec_rows = cur.fetchall()  # (id, title, body, sentiment, score)

            if retrieval_mode == "hybrid":
                cur.execute(
                    _BM25_QUERY,
                    {
                        "query_text": query,
                        "ticker": ticker,
                        "sentiment": sentiment,
                        "top_k": top_k,
                    },
                )
                bm25_rows = cur.fetchall()
                bucket_rows = _rrf_fuse(vec_rows, bm25_rows, k=rrf_k, top_k=top_k)
            else:
                bucket_rows = vec_rows

            raw.extend(bucket_rows)

    if not raw:
        return f"[No relevant documents found for {ticker}]", []

    # Sort merged results by score descending (index 4: sim or rrf)
    raw.sort(key=lambda r: r[4], reverse=True)

    score_label = "rrf" if retrieval_mode == "hybrid" else "sim"
    structured = [
        {
            "title": title,
            "body": body,
            "sentiment": sentiment,
            "sim": round(float(score), 3),
            "score_type": score_label,
        }
        for _, title, body, sentiment, score in raw
    ]

    chunks = []
    for item in structured:
        sentiment_tag = f" [{item['sentiment']}]" if item["sentiment"] else ""
        chunks.append(
            f"### {item['title']}{sentiment_tag} ({score_label}={item['sim']:.4f})\n{item['body']}"
        )

    return "\n\n".join(chunks), structured


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def build_prompt(
    prompt_template: str,
    ticker_contexts: dict[str, str],
    tickers: list[str],
    ticker_names: dict[str, str] | None = None,
) -> str:
    """
    Fill placeholders in the prompt template.

    Available placeholders:
        {tickers}          — comma-separated ticker list (with company names)
        {rag_context}      — all retrieved chunks concatenated
        {ticker}           — only for single-ticker strategies
    """
    names = ticker_names or {}

    def label(t: str) -> str:
        n = names.get(t, "")
        return f"{t} ({n})" if n else t

    rag_context = "\n\n---\n\n".join(
        f"## {label(ticker)}\n{ctx}" for ticker, ctx in ticker_contexts.items()
    )
    ticker_str = ", ".join(label(t) for t in tickers)
    filled = prompt_template.replace("{tickers}", ticker_str).replace("{rag_context}", rag_context)
    if len(tickers) == 1:
        filled = filled.replace("{ticker}", label(tickers[0]))
    return filled


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------


def run_with_model(tier: str, system_prompt: str, user_content: str) -> tuple[str, str, UsageStats]:
    """
    Route to the correct model based on tier.

    Returns (report_text, model_id_used, usage_stats).
    """
    cfg = _load_model_cfg(tier)
    provider = cfg["provider"]
    model = cfg["model"]

    logger.info("Calling model tier=%r provider=%r model=%r", tier, provider, model)

    if provider == "ollama":
        text, usage = _call_ollama(cfg, system_prompt, user_content)
    elif provider == "openai":
        text, usage = _call_openai(cfg, system_prompt, user_content)
    elif provider == "claude":
        text, usage = _call_claude(cfg, system_prompt, user_content)
    else:
        raise ValueError(f"Unknown provider {provider!r}")
    return text, model, usage


def _call_ollama(
    cfg: dict[str, Any], system_prompt: str, user_content: str
) -> tuple[str, UsageStats]:
    import httpx

    base_url = cfg.get("base_url", "http://localhost:11434")
    resp = httpx.post(
        f"{base_url}/api/chat",
        json={
            "model": cfg["model"],
            "stream": False,
            "options": {"num_predict": cfg.get("max_tokens", 2048)},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    input_tok = data.get("prompt_eval_count", 0)
    output_tok = data.get("eval_count", 0)
    usage = compute_cost(
        model=cfg["model"], provider="ollama", input_tokens=input_tok, output_tokens=output_tok
    )
    logger.info(
        "Ollama usage | model=%s | input=%d | output=%d | cost=$0 (local)",
        cfg["model"],
        input_tok,
        output_tok,
    )
    return data["message"]["content"], usage


def _call_openai(
    cfg: dict[str, Any], system_prompt: str, user_content: str
) -> tuple[str, UsageStats]:
    from openai import OpenAI

    client = OpenAI(timeout=120)
    response = client.chat.completions.create(
        model=cfg["model"],
        max_completion_tokens=cfg.get("max_tokens", 4096),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    # OpenAI automatically caches prompts longer than 1024 tokens (50% cost
    # reduction on cache hits).  Log hit count so savings are visible in logs.
    if usage := response.usage:
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        stats = compute_cost(
            model=cfg["model"],
            provider="openai",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cached_tokens=cached,
        )
        logger.info(
            "OpenAI usage | model=%s | input=%d | output=%d | cached=%d | cost=$%.4f",
            cfg["model"],
            usage.prompt_tokens,
            usage.completion_tokens,
            cached,
            stats.cost_usd if stats.cost_usd is not None else 0,
        )
    else:
        stats = compute_cost(model=cfg["model"], provider="openai", input_tokens=0, output_tokens=0)
    return response.choices[0].message.content or "", stats


def _call_claude(
    cfg: dict[str, Any], system_prompt: str, user_content: str
) -> tuple[str, UsageStats]:
    import anthropic

    client = anthropic.Anthropic(timeout=180)

    # The system prompt (strategy instructions) is identical across every ticker
    # in a single strategy run.  Marking it ephemeral lets the Anthropic API
    # serve subsequent calls from the prompt cache — same 5-min TTL window —
    # reducing input-token cost by ~90% for the repeated portion.
    response = client.messages.create(
        model=cfg["model"],
        max_tokens=cfg.get("max_tokens", 16000),
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    )

    u = response.usage
    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
    cache_created = getattr(u, "cache_creation_input_tokens", 0) or 0
    # Normalise: input_tokens in our UsageStats is the *total* prompt size so
    # the dashboard can display a meaningful context-window figure.
    total_input = u.input_tokens + cache_read + cache_created
    usage = compute_cost(
        model=cfg["model"],
        provider="claude",
        input_tokens=total_input,
        output_tokens=u.output_tokens,
        cached_tokens=cache_read,
        cache_creation_tokens=cache_created,
    )
    logger.info(
        "Claude usage | model=%s | input=%d | output=%d"
        " | cache_read=%d | cache_created=%d | cost=$%.4f",
        cfg["model"],
        total_input,
        u.output_tokens,
        cache_read,
        cache_created,
        usage.cost_usd if usage.cost_usd is not None else 0,
    )

    block = response.content[0]
    if block.type != "text":
        raise ValueError(f"Unexpected Claude response block type: {block.type!r}")
    return block.text, usage


# ---------------------------------------------------------------------------
# Streaming model calls — async, for the SSE dashboard endpoint
#
# Using async clients (AsyncAnthropic, AsyncOpenAI, httpx.AsyncClient) means
# each yielded token is a real asyncio yield point: the event loop can flush
# the SSE chunk to the browser before fetching the next token, giving true
# real-time output without relying on Starlette's iterate_in_threadpool.
# ---------------------------------------------------------------------------


async def astream_with_model(
    tier: str,
    system_prompt: str,
    user_content: str,
    *,
    usage_out: list[UsageStats] | None = None,
) -> AsyncGenerator[str]:
    """
    Async generator — yields text chunks as they arrive from the model.

    Same provider routing as run_with_model.  Pass an empty list as
    ``usage_out`` and it will contain a single UsageStats after the generator
    is exhausted — useful for persisting cost data from streaming endpoints.
    """
    cfg = _load_model_cfg(tier)
    provider = cfg["provider"]
    logger.info("Streaming model tier=%r provider=%r model=%r", tier, provider, cfg["model"])

    if provider == "ollama":
        async for chunk in _astream_ollama(cfg, system_prompt, user_content, usage_out=usage_out):
            yield chunk
    elif provider == "openai":
        async for chunk in _astream_openai(cfg, system_prompt, user_content, usage_out=usage_out):
            yield chunk
    elif provider == "claude":
        async for chunk in _astream_claude(cfg, system_prompt, user_content, usage_out=usage_out):
            yield chunk
    else:
        raise ValueError(f"Unknown provider {provider!r}")


async def _astream_ollama(
    cfg: dict[str, Any],
    system_prompt: str,
    user_content: str,
    *,
    usage_out: list[UsageStats] | None = None,
) -> AsyncGenerator[str]:
    import httpx

    base_url = cfg.get("base_url", "http://localhost:11434")
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST",
            f"{base_url}/api/chat",
            json={
                "model": cfg["model"],
                "stream": True,
                "options": {"num_predict": cfg.get("max_tokens", 2048)},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    input_tok = data.get("prompt_eval_count", 0)
                    output_tok = data.get("eval_count", 0)
                    logger.info(
                        "Ollama stream done | model=%s | input=%d | output=%d | cost=$0 (local)",
                        cfg["model"],
                        input_tok,
                        output_tok,
                    )
                    if usage_out is not None:
                        usage_out.append(
                            compute_cost(
                                model=cfg["model"],
                                provider="ollama",
                                input_tokens=input_tok,
                                output_tokens=output_tok,
                            )
                        )


async def _astream_openai(
    cfg: dict[str, Any],
    system_prompt: str,
    user_content: str,
    *,
    usage_out: list[UsageStats] | None = None,
) -> AsyncGenerator[str]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(timeout=180)
    # stream_options=include_usage delivers a final usage chunk after the last
    # content chunk so we can log OpenAI's automatic prompt-cache hit count.
    stream = await client.chat.completions.create(
        model=cfg["model"],
        max_completion_tokens=cfg.get("max_tokens", 4096),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        stream=True,
        stream_options={"include_usage": True},
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta
        if chunk.usage:
            u = chunk.usage
            cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            stats = compute_cost(
                model=cfg["model"],
                provider="openai",
                input_tokens=u.prompt_tokens,
                output_tokens=u.completion_tokens,
                cached_tokens=cached,
            )
            logger.info(
                "OpenAI stream done | model=%s | input=%d | output=%d | cached=%d | cost=$%.4f",
                cfg["model"],
                u.prompt_tokens,
                u.completion_tokens,
                cached,
                stats.cost_usd if stats.cost_usd is not None else 0,
            )
            if usage_out is not None:
                usage_out.append(stats)


async def _astream_claude(
    cfg: dict[str, Any],
    system_prompt: str,
    user_content: str,
    *,
    usage_out: list[UsageStats] | None = None,
) -> AsyncGenerator[str]:
    import anthropic

    client = anthropic.AsyncAnthropic(timeout=300)
    # System prompt cached — same saving as _call_claude: ~90% cost reduction
    # for the fixed portion when the same strategy runs multiple tickers.
    async with client.messages.stream(
        model=cfg["model"],
        max_tokens=cfg.get("max_tokens", 16000),
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
        msg = await stream.get_final_message()
        u = msg.usage
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_created = getattr(u, "cache_creation_input_tokens", 0) or 0
        total_input = u.input_tokens + cache_read + cache_created
        stats = compute_cost(
            model=cfg["model"],
            provider="claude",
            input_tokens=total_input,
            output_tokens=u.output_tokens,
            cached_tokens=cache_read,
            cache_creation_tokens=cache_created,
        )
        logger.info(
            "Claude stream done | model=%s | input=%d | output=%d"
            " | cache_read=%d | cache_created=%d | cost=$%.4f",
            cfg["model"],
            total_input,
            u.output_tokens,
            cache_read,
            cache_created,
            stats.cost_usd if stats.cost_usd is not None else 0,
        )
        if usage_out is not None:
            usage_out.append(stats)


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


RunResult = tuple[
    str,
    str,
    list[str],
    dict[str, str],
    dict[str, list[dict[str, Any]]],
    datetime,
    float,
    UsageStats,
]


def run_strategy(strategy: dict[str, Any], conn: psycopg.Connection) -> RunResult:
    """
    Execute one strategy end-to-end.

    Returns (report, model_id, tickers_used, ticker_names, ticker_news,
    started_at, duration_seconds, usage).
    """
    tickers = resolve_tickers(strategy)
    if not tickers:
        logger.warning("Strategy %r resolved zero tickers — skipping", strategy["name"])
        empty_usage = compute_cost(model="", provider="", input_tokens=0, output_tokens=0)
        return "", "", [], {}, {}, datetime.now(UTC), 0.0, empty_usage

    logger.info(
        "Strategy %r | scope=%s | tickers=%s | model=%s",
        strategy["name"],
        strategy["scope"],
        tickers,
        strategy["model_tier"],
    )

    started_at = datetime.now(UTC)
    t0 = time.monotonic()

    # Load company names for all tickers (best-effort — empty string on miss)
    ticker_names: dict[str, str] = {}
    for ticker in tickers:
        cfg = load_stock_config(ticker)
        ticker_names[ticker] = cfg.get("name", "") if cfg else ""

    retrieval_mode = strategy.get("retrieval_mode", "hybrid")
    rrf_k = strategy.get("rrf_k", 60)

    # RAG fetch per ticker — include company name in query for better embedding match
    ticker_contexts: dict[str, str] = {}
    ticker_news: dict[str, list[dict[str, Any]]] = {}
    for ticker in tickers:
        text, news_rows = rag_fetch(
            conn,
            strategy["rag_query_template"],
            ticker,
            company_name=ticker_names.get(ticker, ""),
            threshold=strategy.get("rag_threshold", 0.55),
            top_k_positive=strategy.get("rag_top_k_positive", 20),
            top_k_neutral=strategy.get("rag_top_k_neutral", 20),
            top_k_negative=strategy.get("rag_top_k_negative", 20),
            retrieval_mode=retrieval_mode,
            rrf_k=rrf_k,
        )
        ticker_contexts[ticker] = text
        ticker_news[ticker] = news_rows

    user_content = build_prompt(
        strategy["prompt_template"],
        ticker_contexts,
        tickers,
        ticker_names=ticker_names,
    )

    report, model_id, usage = run_with_model(
        strategy["model_tier"],
        strategy["prompt_template"],
        user_content,
    )

    duration = round(time.monotonic() - t0, 1)
    return report, model_id, tickers, ticker_names, ticker_news, started_at, duration, usage
