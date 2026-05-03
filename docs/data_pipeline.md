# Data Pipeline Design

This document covers the cleaning and enrichment pipeline that transforms raw
ingested data into vector-searchable documents. The pipeline lives in
`src/sky_finance/pipeline/` and runs every 30 minutes via Celery Beat.

---

## 1. Overview

```
news_raw / raw_data          (PostgreSQL — unprocessed rows)
        │
        │  SELECT … FOR UPDATE SKIP LOCKED  (batch claim, avoids double-processing)
        ▼
  dispatch_pipeline  ──fan-out──▶  group([process_news_article × N,
                                          process_stock_record × M])

  process_news_article                    process_stock_record
  ┌────────────────────────────┐          ┌────────────────────────────┐
  │ 1. fetch from DB           │          │ 1. fetch payload from DB   │
  │ 2. clean text              │          │ 2. ohlcv_to_text()         │
  │    (HTML strip, WS norm,   │          │    (no LLM needed)         │
  │     truncate 2 000 chars)  │          │ 3. insert document         │
  │ 3. LLM summarise (Ollama)  │          │ 4. embed (nomic)           │
  │ 4. insert document         │          │ 5. insert embedding        │
  │ 5. embed (nomic)           │          │ 6. mark processed          │
  │ 6. insert embedding        │          └────────────────────────────┘
  │ 7. mark processed          │
  └────────────────────────────┘
        │                                         │
        ▼                                         ▼
   documents table                         embeddings table
   (title, body,                           (VECTOR(768),
    sentiment, key_facts)                   ticker, source_type)
```

Two document types flow through the same pipeline but take different paths:
news articles go through the local LLM; OHLCV records are converted to text
deterministically and embedded directly.

---

## 2. Batch Claiming with `FOR UPDATE SKIP LOCKED`

```sql
SELECT id FROM news_raw
WHERE processed = false
ORDER BY created_at
LIMIT 50
FOR UPDATE SKIP LOCKED
```

`SKIP LOCKED` is the key detail: when multiple workers run `dispatch_pipeline`
concurrently (or when a previous batch is still in flight), rows that are
already locked by another connection are silently skipped. This gives
at-most-once processing per dispatch cycle without a separate "claimed" flag or
an external coordination layer.

The `LIMIT` per batch (`_NEWS_BATCH = 50`, `_STOCK_BATCH = 20`) is deliberately
conservative — OHLCV is smaller because it skips the LLM step and completes
faster, so fewer in-flight tasks are needed to keep workers busy.

---

## 3. Text Cleaning (`pipeline/cleaner.py`)

### News articles

```python
def clean_news_article(article: dict) -> dict:
    title   = normalise_ws(strip_html(article["title"]))
    content = normalise_ws(strip_html(article["content"] or article["summary"]))
    if len(content) > 2000:
        content = content[:2000] + "…"
    return {**article, "title": title, "content": content}
```

**Why 2 000 character truncation?**
The local LLM (qwen2.5:3b-instruct) has a context window of ~32k tokens but
runs on CPU. At 3B parameters, quality degrades noticeably when the input
exceeds ~500 tokens, which corresponds to roughly 2 000 characters of English
text. Truncating here keeps the LLM in its reliable operating range. The
summary it produces (the part that actually gets embedded and retrieved) is
typically 2–3 sentences, so the information loss from truncating long articles
is acceptable — the most relevant information in a financial news article is
almost always in the first 2 000 characters.

**Why strip HTML?**
Google RSS feeds embed HTML tags in the `<summary>` field (`<b>`, `<i>`, `<a
href="...">`, etc.). These survive into the raw DB record. HTML tags fragment
tokenisation — `<b>Apple</b>` produces different token boundaries than `Apple`
— which degrades both the LLM's understanding and the quality of the resulting
embedding.

### OHLCV records

```python
def ohlcv_to_text(record: dict) -> str:
    # Returns e.g.:
    # "AAPL (US) — 2024-01-03
    #  Open: 185.0  Close: 187.2  High: 188.0  Low: 184.5  Volume: 52_000_000
    #  Daily change: +1.19%
    #  Market cap: 2,900,000,000,000 USD
    #  Trailing P/E: 28.4
    #  52-week range: 142.0 – 199.6
    #  Sector: Technology  Industry: Consumer Electronics"
```

**Why convert OHLCV to text at all?**
pgvector operates on dense vectors. Price data in its native form (floats in a
JSON array) cannot be embedded semantically — a cosine similarity search on
price vectors would find "similar-looking candlestick shapes", not "stocks
with similar fundamental characteristics and recent performance". Converting to
a human-readable text string lets the same embedding model (`nomic-embed-text`)
that handles news also handle price data, so RAG queries like "high P/E growth
stocks" or "recent volume spike" can retrieve both relevant news articles and
relevant OHLCV records from the same vector index.

**No LLM for OHLCV — why?**
Price data is already structured and factual. An LLM summary would add latency
(~2–5 s per record on CPU) without adding information — `ohlcv_to_text` is
deterministic and produces the same information density as an LLM would. The
LLM step is reserved for unstructured text where it adds real value: extracting
sentiment, key facts, and topics from the noise of a news article.

---

## 4. Local LLM Summarisation (`pipeline/llm_summariser.py`)

### Model and parameters

```python
client.chat(
    model="qwen2.5:3b-instruct",  # configurable via settings.toml
    messages=[system_msg, user_msg],
    format="json",                 # Ollama structured output
    options={"num_predict": 512, "temperature": 0.1},
)
```

**Why `format="json"`?**
Ollama's `format="json"` mode enables constrained decoding — the model's
sampling is biased to produce valid JSON tokens at each step. This is not
post-processing; it changes what the model generates. Without it, small models
(≤7B) frequently wrap their output in markdown code fences, add explanatory
prose, or produce trailing commas. `format="json"` eliminates all of these
failure modes and makes `json.loads()` on the raw output reliable.

**Why `temperature=0.1`?**
Financial data extraction is a retrieval task, not a creative task. The correct
sentiment for "Apple beats earnings estimates by 15%" is unambiguously
"positive". A higher temperature introduces randomness that can flip sentiment
or add hallucinated key facts. `0.1` keeps the output deterministic for
unambiguous articles while allowing the model to choose naturally between
synonymous phrasings.

**Why `num_predict=512`?**
The output schema is:
```json
{
  "summary": "2-3 sentences (~100 tokens)",
  "sentiment": "positive | neutral | negative",
  "key_facts": ["up to 5 items (~150 tokens total)"],
  "topics": ["3-5 tags (~30 tokens)"],
  "relevance_score": 0.85
}
```
512 tokens comfortably covers the maximum expected output with room to spare.
Larger values waste time on forced-stop padding; smaller values risk truncating
mid-JSON and triggering the fallback path.

### Fallback on invalid JSON

```python
try:
    result = json.loads(response.message.content)
except json.JSONDecodeError:
    return _fallback(ticker)  # neutral sentiment, empty fields, relevance=0.0
```

The fallback returns a safe zero-signal record rather than raising. The article
is still inserted into `documents` with empty metadata so it can be embedded
and retrieved — it just won't contribute structured signal to the strategy
engine. This matters for pipeline reliability: a single misbehaving article
should not block the remaining 49 in the batch.

### What the LLM produces

| Field | Purpose downstream |
|-------|-------------------|
| `summary` | Stored as `documents.body` — retrieved as RAG context |
| `sentiment` | Used as a filter in the pgvector query (positive/neutral/negative bucket) |
| `key_facts` | Appended to the body for richer retrieval context |
| `topics` | Available in metadata; not yet used in retrieval (future: topic filtering) |
| `relevance_score` | Available in metadata; not yet used in retrieval (future: relevance pre-filter) |

---

## 5. Embedding (`storage/embedder.py`)

### What gets embedded

For news:
```python
embed_text = f"{ticker}\n{cleaned_title}\n{llm_summary}\n{key_facts_joined}"
```

The ticker is prepended to the embedding input so that the resulting vector is
anchored in the ticker's semantic neighbourhood. Without this, an article about
"Apple's supply chain" and an article about "Samsung's supply chain" would
produce nearly identical vectors — indistinguishable in a cosine similarity
search. With the ticker prepended, the model encodes the company identity into
the vector.

For OHLCV:
```python
embed_text = ohlcv_to_text(payload)  # the full human-readable string
```

### Batch processing

```python
for i in range(0, len(texts), BATCH_SIZE):  # BATCH_SIZE = 32
    batch = texts[i : i + BATCH_SIZE]
    vectors.extend(_embed_ollama(batch))
```

`dispatch_pipeline` fans out one Celery task per article, so in practice each
task calls `embed_single()` (batch size = 1). The `embed_texts()` batch API
exists for bulk re-indexing scenarios (e.g. switching embedding models) where
processing one-at-a-time would be prohibitively slow.

### Backend switch

```bash
EMBEDDING_BACKEND=openai  # switches to text-embedding-3-small (1536 dims)
```

Switching backends requires a DB migration (`VECTOR(768)` → `VECTOR(1536)`) and
a full re-embedding run. The backend abstraction makes this feasible without
changing any pipeline code — only the config and the migration.

---

## 6. Worker Concurrency Constraint

```bash
# Pipeline workers must be started with low concurrency
celery worker --queues=pipeline --concurrency=2
```

The local Ollama instance is the bottleneck. A single `qwen2.5:3b-instruct`
call takes 2–8 seconds on CPU. With `concurrency=2`, two articles are
summarised in parallel, which is approximately the maximum throughput a single
Ollama instance can sustain without request queuing. The ingestion worker
(`--queues=ingestion`) can run at higher concurrency (`4–8`) because it only
makes network calls.

Keeping the queues separate (ingestion / pipeline / strategies) means each
worker pool can be sized independently — a useful scaling lever if the article
volume grows significantly.
