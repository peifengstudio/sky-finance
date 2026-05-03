# Architecture & Design Decisions

This document explains the key architectural choices in sky-finance — not what
the system does, but *why* it is built the way it is. The target reader is
someone evaluating the codebase who wants to understand the reasoning behind
the technology stack.

---

## 1. Why Celery + Redis Instead of a Simple Cron

The most obvious alternative to Celery is a standard cron job that calls a
Python script on a schedule. For a small project this would work, but it has
two problems that become painful quickly.

**Problem 1: No retry.**
yfinance and Google RSS are external services. They return HTTP 429s, time out,
and occasionally go down. A cron job that fails just fails — there is no
built-in way to retry after 60 seconds. Celery's `autoretry_for` with
exponential backoff handles this transparently.

**Problem 2: No parallelism.**
The ingestion dispatch pattern fans out one task per ticker:
```
dispatch_ingest_news → group([ingest_news("AAPL"), ingest_news("NVDA"), ...])
```
With cron + a sequential script, fetching news for 10 tickers takes 10× as
long as fetching for 1. With Celery, each ticker task is independent and
multiple workers consume them concurrently. Adding tickers does not increase
wall-clock time, only task queue depth.

**Why Redis as the broker (not RabbitMQ)?**
Redis is already present as the result backend and is sufficient for this
workload. The queue sizes here are small (tens to hundreds of tasks per cycle,
not millions). RabbitMQ would add operational complexity (separate service,
different protocol, management UI) for no tangible benefit at this scale.

**Beat vs worker separation.**
Beat's only job is to put task names into Redis at the right time. It never
executes business logic. This means:
- Beat can crash and restart without losing in-flight tasks (they are already
  in Redis waiting for a worker).
- Workers can be scaled horizontally (run 2, 4, N workers) without touching
  Beat at all.
- The system degrades gracefully: if Beat is down, scheduled tasks stop
  arriving but any already-queued tasks still complete.

---

## 2. Why PostgreSQL + pgvector Instead of a Dedicated Vector DB

The short answer: **one less service to run**.

Dedicated vector databases (Pinecone, Weaviate, Qdrant, Chroma) solve a real
problem at scale — when you have hundreds of millions of vectors, you need
specialised storage and distributed ANN search. At the scale of this project
(tens of thousands of vectors per ticker, a few tickers), pgvector inside the
existing PostgreSQL instance is sufficient.

The practical advantages:

**Joins.** The most common retrieval query joins `embeddings` with `documents`
to get the text alongside the vector:
```sql
SELECT d.title, d.body, d.sentiment,
       1 - (e.embedding <=> $1) AS similarity
FROM embeddings e
JOIN documents d ON d.id = e.document_id
WHERE e.ticker = $2
  AND d.sentiment = $3
  AND 1 - (e.embedding <=> $1) >= $4
ORDER BY e.embedding <=> $1
LIMIT $5
```
This is a standard SQL join. In a dedicated vector DB, you would retrieve IDs
from the vector index and then make a second call to fetch metadata. pgvector
does both in one query.

**Transactions.** Writing a document and its embedding atomically (both succeed
or both roll back) is trivial with PostgreSQL transactions. With a separate
vector DB, you need to implement your own two-phase consistency logic or accept
the possibility of orphaned vectors.

**HNSW index.** pgvector's HNSW index (`vector_cosine_ops`, m=16,
ef_construction=64) provides sub-millisecond approximate nearest-neighbour
search for the corpus sizes here. The parameters are chosen conservatively —
higher `m` and `ef_construction` improve recall at the cost of more memory and
slower inserts, which is not a tradeoff worth making for this workload.

**When to reconsider.** If the corpus grows to millions of vectors per ticker
or if multi-tenancy/isolation become requirements, a dedicated vector DB would
be the right migration. The `embedder.py` backend abstraction and the
repository pattern in `storage/repository.py` mean that migration would be
isolated to the storage layer.

---

## 3. Per-Ticker TOML Configuration

Each stock has its own file at `config/stocks/<TICKER>.toml`. The filename is
the ticker symbol. There is no shared list of tickers.

```
config/stocks/
    AAPL.toml
    NVDA.toml
    7203.T.toml
    6758.T.toml
    local/           ← gitignored; private overrides (real buy prices, share counts)
        AAPL.toml
```

**Why file-per-ticker instead of a shared list?**

*Git history is per-stock.* Adding a ticker, changing its stop-loss, or
updating its L2 keywords all produce focused diffs that are easy to review and
easy to revert. With a shared `stocks.toml`, a single file accumulates all
changes for all tickers and becomes noisy.

*Isolation.* Deleting a ticker is `rm config/stocks/NVDA.toml`. There is no
risk of accidentally editing another stock's entry. The loader scans the
directory at startup — no index to keep in sync.

*Private overrides without secrets in git.* The `local/` subdirectory is
gitignored. Real buy prices and share counts go there:
```toml
# config/stocks/local/AAPL.toml  ← not committed
[price]
buy_price = 172.50

[position]
shares = 25
```
These are deep-merged on top of the shared file at load time. The shared file
can be published openly; sensitive portfolio data stays local.

**Three-tier keyword model.**
Each stock config declares keywords at three specificity levels:

```toml
[ingestion]
l1_keywords = ["Apple earnings", "iPhone sales"]   # L1: company-specific
l2_topics   = ["AI smartphone", "China Apple ban"] # L2: sector/theme context
l3_macro    = ["Fed rate cut", "USD CNY rate"]     # L3: macro factors
```

- L1 and L2 are used to build Google RSS queries (two separate queries per
  ticker to avoid dilution).
- L3 is intentionally excluded from RSS fetching — macro terms like "Fed rate
  cut" would return thousands of irrelevant articles. L3 is used only by the
  strategy engine for RAG context expansion.
- Japan tickers include Japanese-language keywords for L1 and L2, which trigger
  additional JA-locale RSS queries.

---

## 4. Queue-Per-Stage Architecture

Each pipeline stage has its own Celery queue:

```
ingestion   → news and stock data fetching
pipeline    → cleaning, LLM summarisation, embedding
storage     → (reserved for bulk re-indexing operations)
strategies  → RAG retrieval + model calls
notifications → Slack delivery
```

**Why not one queue?**
Worker concurrency requirements differ significantly by stage:

| Stage | Bottleneck | Ideal concurrency |
|-------|-----------|-------------------|
| Ingestion | Network I/O | 4–8 |
| Pipeline | Local Ollama (CPU) | 2 |
| Strategies | API rate limits | 2–4 |
| Notifications | Network I/O | 4 |

With a single queue, you either starve the fast stages (low concurrency) or
overwhelm Ollama (high concurrency). With separate queues, each worker pool
is sized independently:
```bash
celery worker --queues=ingestion  --concurrency=6
celery worker --queues=pipeline   --concurrency=2
celery worker --queues=strategies --concurrency=3
```

---

## 5. Reliability Configuration

Three `celery_app.py` settings that matter for production:

```python
"task_acks_late": True,
"task_reject_on_worker_lost": True,
```

**`task_acks_late=True`**: A task is acknowledged (removed from the queue)
only after it completes successfully, not when it is picked up. If a worker
dies mid-execution (OOM, host reboot), the task is returned to the queue and
another worker retries it. The alternative (`task_acks_late=False`, the default)
acknowledges at pickup time — a crash mid-task means the work is silently lost.

**`task_reject_on_worker_lost=True`**: Paired with `task_acks_late`, this
ensures that when a worker process is killed (e.g. by the OS OOM killer), the
task is explicitly rejected rather than left in an ambiguous state. Without
this, Celery may wait for the ack indefinitely.

**`task_track_started=True`**: Exposes the `STARTED` state in the result
backend. Without this, a task goes directly from `PENDING` to `SUCCESS` or
`FAILURE` — there is no way to distinguish "not yet picked up" from "currently
running" from Flower or the dashboard.

---

## 6. Local-First Design

The project is designed to run entirely on a single laptop without any paid
external services:

| Component | Local default | Cloud alternative |
|-----------|--------------|------------------|
| LLM (summarisation) | Ollama qwen2.5:3b | OpenAI gpt-4o-mini |
| Embeddings | Ollama nomic-embed-text | OpenAI text-embedding-3-small |
| Vector store | PostgreSQL + pgvector | Pinecone, Weaviate |
| Task queue | Redis (Docker) | AWS SQS, CloudAMQP |
| Analysis LLM | Ollama (local tier) | OpenAI / Claude (nano / advanced tier) |

Every external service dependency is behind an abstraction that can be swapped
via environment variables or `config/settings.toml`. No code changes are
required to move from local to cloud for any individual component.

This matters for a financial tool where data privacy is a concern — raw
portfolio data, buy prices, and news article content never leave the machine
unless the user explicitly configures a cloud provider.
