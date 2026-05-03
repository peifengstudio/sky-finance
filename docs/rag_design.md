# RAG Design — sky-finance

This document explains the design decisions behind the RAG (Retrieval-Augmented
Generation) pipeline in `src/sky_finance/strategies/engine.py`. Each section
covers one design choice, the reasoning behind it, and the tradeoffs accepted.

---

## 1. Pipeline Overview

```
[documents table]         [embeddings table]
  title, body,      ←→     VECTOR(768), ticker,
  sentiment,                source_type
  key_facts

         ↑                         ↑
         │     pipeline/           │
         └──── llm_summariser ─────┘
               (Ollama qwen2.5:3b)
               → summary, sentiment,
                 key_facts, topics,
                 relevance_score

                         ↓ at strategy run time

              embed(rag_query_template + ticker)
                         ↓
              cosine similarity search per
              sentiment bucket (positive /
              neutral / negative)
                         ↓
              build_prompt() → LLM → report
```

The two tables (`documents`, `embeddings`) are kept separate so the same
embedding can serve multiple query types without duplicating the document body.
The `VECTOR(768)` column is backed by an **HNSW index**
(`vector_cosine_ops`, m=16, ef_construction=64) for sub-millisecond ANN search
at the corpus sizes expected here (tens of thousands of chunks).

---

## 2. Embedding Model Choice

| | nomic-embed-text (default) | text-embedding-3-small (alt) |
|-|---------------------------|------------------------------|
| Dims | 768 | 1536 |
| Cost | Free (local Ollama) | ~$0.02 / 1M tokens |
| Throughput | ~200 texts/min on CPU | API rate-limited |
| Quality on financial text | Good — trained on diverse EN corpus | Slightly better on domain-specific queries |

**Default is nomic-embed-text** because the project is local-first: no API
cost, no rate limits, and embeddings can be regenerated offline. The backend
is swappable via `EMBEDDING_BACKEND=openai` without code changes — the DB
schema (`VECTOR(768)`) would need a migration to `VECTOR(1536)` for the OpenAI
model, which is intentionally deferred until the quality difference justifies it.

The same model is used for both **indexing** (pipeline) and **querying**
(strategy engine). Using different models for the two sides is a common source
of subtle quality regressions and is explicitly avoided here.

---

## 3. Sentiment-Bucketed Retrieval

### The problem with naïve top-k

A standard top-k similarity search returns the N most similar documents
globally. For stocks in a sustained rally, the corpus might be 85% positive
articles. A top-20 query would return 17 positive and 3 neutral articles,
leaving the model with no negative signal even if strong risk factors exist.

This matters because **downside risks are often the most actionable signal**
for a strategy engine. An analysis that only sees bullish news will consistently
produce overconfident buy recommendations.

### The solution: three independent queries

```python
buckets = [
    ("positive", top_k_positive),   # bullish news
    ("neutral",  top_k_neutral),    # macro / sector context
    ("negative", top_k_negative),   # risk signals
]
for sentiment, top_k in buckets:
    cur.execute(_RAG_QUERY, {"sentiment": sentiment, "top_k": top_k, ...})
```

Each bucket runs an independent cosine similarity search filtered by
`d.sentiment = %s`. Results from all three are merged and re-ranked by
similarity before being passed to the prompt. This guarantees that **minority
sentiment is always represented**, regardless of corpus composition.

### Bucket sizing

Defaults are `top_k = 20` per bucket (60 total chunks). Each chunk is
truncated to 400 characters in the SQL query (`LEFT(d.body, 400)`), so the
worst-case context is ~24 000 characters — well within gpt-4o-mini's 128k
window and acceptable for local models with a 32k context.

Individual strategies can override per-bucket sizes:

```
# Strategy tuned for downside risk scanning
rag_top_k_positive = 5    # minimal bull context
rag_top_k_neutral  = 15   # macro backdrop
rag_top_k_negative = 30   # max risk signal coverage
```

Setting a bucket to `0` skips that sentiment entirely — useful for
pure-sentiment strategies (e.g. a "positive momentum screener" that explicitly
ignores negative news).

---

## 4. Query Construction

```python
query = rag_query_template.replace("{ticker}", ticker)
if company_name:
    query = f"{query} {company_name}"
vector = embed_single(query)
```

Two design choices here:

**Company name appended to query.** Ticker symbols (`AAPL`, `7203.T`) appear
infrequently in financial news text — articles typically use the company name.
Appending the company name (`Apple Inc.`) to the query vector anchors it closer
to the documents' semantic neighbourhood and meaningfully improves recall,
especially for Japan tickers where the ticker format (`7203.T`) never appears
in Japanese-language articles.

**Template over free-form query.** The `rag_query_template` is stored in the
database and editable per-strategy without code changes. This allows analysts
to tune retrieval intent (e.g. "earnings guidance and revenue outlook for
{ticker}" vs "regulatory and legal risks for {ticker}") independently of the
analysis prompt. The template is embedded at run time, not at ingestion time,
so changing it immediately affects the next run without re-indexing.

---

## 5. Similarity Threshold

```python
AND 1 - (e.embedding <=> %(vec)s::vector) >= %(threshold)s
```

Default threshold: **0.55** (cosine similarity, nomic-embed-text 768d).

This floor filters chunks that happen to share a few keywords with the query
but are semantically off-topic. Without it, a query about "iPhone supply chain"
would retrieve macro articles about "global supply chain disruptions" that
contain no Apple-specific signal.

At 0.55 for nomic-embed-text:
- **Below**: mostly noise — general macro articles with incidental ticker mentions
- **0.55–0.70**: relevant sector/theme context (neutral bucket territory)
- **0.70–0.85**: directly on-topic news for the ticker
- **Above 0.85**: near-duplicate articles or very tight matches

Strategies that need broader macro context can lower this (e.g. `rag_threshold
= 0.45`). Strategies that need tight signal can raise it (`0.65+`). The
threshold applies equally across all three sentiment buckets.

---

## 6. Prompt Assembly

```python
def build_prompt(prompt_template, ticker_contexts, tickers, ticker_names):
    rag_context = "\n\n---\n\n".join(
        f"## {label(ticker)}\n{ctx}" for ticker, ctx in ticker_contexts.items()
    )
    filled = prompt_template
        .replace("{tickers}", ticker_str)
        .replace("{rag_context}", rag_context)
```

**Structure over concatenation.** Retrieved chunks are grouped by ticker under
a Markdown `##` heading, separated by `---`. This gives the LLM explicit
boundaries between companies, reducing cross-ticker attribution errors (e.g.
attributing Apple news to Toyota in a multi-ticker global strategy).

**Sentiment tag in chunk headers.** Each chunk is prefixed with its sentiment
label and similarity score:

```
### Apple iPhone sales beat estimates [positive] (sim=0.81)
Apple reported iPhone unit sales of 52M in Q4, beating estimates by 8%...
```

The explicit sentiment label reduces hallucination risk — the model knows
whether the retrieved context is bearish or bullish without having to infer it
from the body text, which is truncated to 400 characters.

**System prompt = `prompt_template`.** The same field serves as both the RAG
query template and the system prompt. This is intentional: the vocabulary used
in the query should be consistent with the vocabulary in the system instruction,
which keeps query intent and generation intent aligned. A strategy about
"earnings risk" retrieves earnings-risk documents and also instructs the model
to reason about earnings risk.

---

## 7. Multi-Provider Model Routing

```python
# config/settings.toml
[models.local]
provider = "ollama"
model    = "qwen2.5:14b-instruct"

[models.nano]
provider = "openai"
model    = "gpt-4o-mini"

[models.advanced]
provider = "claude"
model    = "claude-opus-4-7"
```

Strategies reference a **tier name**, not a model ID. This decouples strategy
definitions from model availability — upgrading from `claude-opus-4-5` to
`claude-opus-4-7` is a one-line config change with no database migration and
no strategy edits.

Each provider uses its official SDK (`ollama`, `openai`, `anthropic`) with
SDK-managed auth and error handling. The Ollama path uses a raw `httpx` POST
because the Ollama Python SDK does not support a `stream=False` chat call with
a `num_predict` cap in the same API surface as the others.

**Tier selection guidance:**

| Tier | When to use |
|------|-------------|
| `local` | Single-ticker daily summaries; latency-insensitive, zero cost |
| `nano` | Multi-ticker group reports; good quality/cost balance |
| `advanced` | Global portfolio analysis, complex cross-market reasoning |

---

## 8. Hybrid Retrieval — BM25 + Vector + RRF

### Why pure vector search is not enough

Cosine similarity excels at semantic recall — it finds articles that are
*about* the same topic even when they use different words. But it can miss
exact-match signals. A strategy querying "earnings guidance outlook for AAPL"
may rank a general AI-sector article above a filing that contains the exact
phrase "Apple raised FY2026 EPS guidance" if the embedding space is crowded
near that query vector.

BM25 (approximated here via PostgreSQL `ts_rank_cd`) solves the opposite
problem: it rewards term frequency and document frequency with no semantic
reasoning. It catches exact keyword matches ("EPS guidance", "profit warning",
"class action") but will miss semantically related articles that use synonyms.

### The solution: fuse both with RRF

```
query text
    ├──▶  embed_single()       ──▶  _RAG_QUERY (cosine, HNSW)  ──▶  vec_rows
    └──▶  plainto_tsquery()    ──▶  _BM25_QUERY (ts_rank_cd)   ──▶  bm25_rows
                                                   │                    │
                                                   └──── _rrf_fuse() ───┘
                                                               ↓
                                                       bucket_rows (top-k)
```

Both queries run inside the same cursor, inside the same sentiment bucket loop
that already existed. The only new round-trip is the BM25 SELECT — no extra
network calls, no external services.

### Reciprocal Rank Fusion (RRF)

RRF (Cormack, Clarke & Buettcher, 2009) is the standard way to merge ranked
lists without needing normalised scores:

```
RRF(d) = Σᵢ  1 / (k + rankᵢ(d))
```

`k = 60` is the original paper's recommendation. The formula has two useful
properties for this use case:

1. **Outlier-resistant** — a document ranked #1 in one list but absent from
   the other scores 1/61 ≈ 0.0164. A document ranked #2 in both lists scores
   1/62 + 1/62 ≈ 0.0323. Cross-list agreement consistently beats single-list
   dominance.

2. **Score-free** — only rank positions matter, so the cosine [0, 1] and
   ts_rank_cd [0, 1] scales never need normalisation.

After fusion, rows are sorted by RRF score descending and sliced to `top_k`.
The `sim` field in the returned structured dicts contains the RRF score;
`score_type` is set to `"rrf"` so the dashboard can label it correctly.

### Schema changes (migration 007)

```sql
-- Generated tsvector — auto-updated on every INSERT/UPDATE
ALTER TABLE documents
ADD COLUMN body_tsv tsvector
GENERATED ALWAYS AS (
    to_tsvector('english', coalesce(title, '') || ' ' || body)
) STORED;

CREATE INDEX documents_body_tsv_gin ON documents USING gin(body_tsv);

-- Per-strategy mode switch
ALTER TABLE strategies
ADD COLUMN retrieval_mode TEXT NOT NULL DEFAULT 'hybrid';
```

The GIN index makes the BM25 `@@` operator fast — the same index type used
by full-text-search in production PostgreSQL installations.

### Configuration

Global default in `config/settings.toml`:

```toml
[rag]
retrieval_mode = "hybrid"   # "hybrid" | "vector"
rrf_k          = 60         # RRF k constant
```

Per-strategy override: set `retrieval_mode = "vector"` in the strategy row
(editable via the dashboard) to fall back to pure cosine similarity for a
specific strategy without affecting others.

---

## 9. Known Limitations

**No chunk overlap.** Documents are stored as single units (article body,
truncated to 2 000 chars at pipeline time, then further to 400 chars in the
SQL query). Long articles lose their tail. Implementing sliding-window chunking
with overlap would improve recall for long-form earnings call transcripts, at
the cost of a larger embedding table and more complex deduplication.

**Embedding staleness.** Embeddings are written once at pipeline time and never
updated. If the embedding model changes (e.g. switching from nomic-embed-text
to a newer model), existing vectors are incompatible with new query vectors.
A migration that re-embeds all documents would be required — the Alembic
migration pattern makes this feasible but it is not automated.

**No reranker.** The final ranking uses RRF scores. A cross-encoder reranker
(e.g. `ms-marco-MiniLM`) applied after the hybrid retrieval step would improve
precision further, especially for the neutral bucket where similarity scores
cluster tightly. Not implemented because the 60-chunk context is already within
model limits and the marginal quality gain does not justify the latency cost for
a local-first tool.
