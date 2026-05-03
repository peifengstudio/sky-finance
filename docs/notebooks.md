# Notebooks — AI Exploration Guide

Three Jupyter notebooks in `notebooks/` provide hands-on intuition for the AI
techniques that power sky-finance.  They are the recommended starting point for
any full-stack developer who wants to understand how the strategy engine works
before reading the source code.

## Notebooks at a glance

| Notebook | Concept | Needs DB? | Needs API key? |
|----------|---------|-----------|----------------|
| [`01_rag_exploration.ipynb`](../notebooks/01_rag_exploration.ipynb) | Embedding similarity, sentiment bucketing, threshold tuning | Yes | No |
| [`02_prompt_engineering.ipynb`](../notebooks/02_prompt_engineering.ipynb) | Prompt iteration — minimal → production-grade | No | No (local only) |
| [`03_model_comparison.ipynb`](../notebooks/03_model_comparison.ipynb) | Cost, latency, and quality across all four model tiers | No | Optional |

---

## Quick start

### 1. Install notebook dependencies

```bash
uv sync --extra dev   # adds jupyterlab + matplotlib (already in pyproject.toml)
```

### 2. Start the required services

**Notebook 01** requires a running PostgreSQL instance with data:

```bash
# Start PostgreSQL + pgvector
docker compose -f docker/docker-compose.yml up -d

# Pull the embedding model
ollama pull nomic-embed-text

# Run at least one ingest + pipeline cycle so the DB has embeddings
uv run celery -A sky_finance.scheduler.celery_app call \
  sky_finance.ingestion.tasks.dispatch_ingest_us_stocks
uv run celery -A sky_finance.scheduler.celery_app call \
  sky_finance.pipeline.tasks.dispatch_pipeline
```

**Notebook 02** only needs Ollama:

```bash
ollama pull qwen2.5:14b-instruct
```

**Notebook 03** needs Ollama for the local tier; API keys for paid tiers:

```bash
# Local tier
ollama pull qwen2.5:14b-instruct

# Paid tiers — add to .env (skip any you don't have)
echo 'OPENAI_API_KEY=sk-...'      >> .env
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env
```

### 3. Launch JupyterLab

Run from the **project root** so that `sky_finance` is importable (it's
installed as an editable package via `uv sync`):

```bash
uv run jupyter lab
```

JupyterLab opens at `http://localhost:8888`.  Navigate to `notebooks/` and
open any notebook.

---

## Notebook details

### 01 — RAG Exploration

**Concept**: embedding-based similarity search and why sentiment bucketing
matters.

**What you'll do**:

1. Embed a natural-language query (e.g. `"Apple iPhone supply chain risk"`)
   with `nomic-embed-text` and inspect the resulting 768-dimensional vector.
2. Run a plain cosine-similarity top-k search against your local pgvector
   store and see which news chunks the engine retrieves.
3. Switch to sentiment-bucketed retrieval — three separate queries, one per
   sentiment — and compare: does the plain search bury negative signals?
4. Plot similarity scores by sentiment as a bar chart.
5. Vary the similarity threshold and watch how precision / recall changes.

**Key insight**: for a stock in a bull run, the news corpus may be 80 %
positive.  Plain top-k fills the LLM's context with bullish articles and
buries the few negative signals.  Running separate per-bucket queries
guarantees that negative evidence always reaches the model, even when it is a
tiny minority of the corpus.

---

### 02 — Prompt Engineering

**Concept**: how prompt structure affects output quality, independent of the
model or retrieved context.

**What you'll do**:

1. Start with a fixed, hardcoded news context (no DB needed) so the only
   variable is the prompt.
2. Run **v1** — a minimal one-line prompt — and note the hallucination risk
   (model draws on training data, not current news).
3. Run **v2** — inject the context and request structure — output is now
   grounded but still tends to summarise rather than analyse.
4. Run **v3** — add a persona, a chain-of-thought instruction, and a strict
   output format — and see the qualitative jump in specificity and
   actionability.
5. Compare all three side-by-side.

**Key insight**: the three additions in v3 — persona, chain-of-thought, and
format — each independently improve output quality.  Together they turn a weak
local model into something portfolio-manager-readable.  This is exactly the
pattern used in production `strategies.prompt_template` entries.

**Experiment**: swap `TIER = 'local'` for `'claude'` (with API key) and re-run
to see the same prompt improvement pattern on a stronger model.

---

### 03 — Model Comparison

**Concept**: cost, latency, and quality trade-offs across all four sky-finance
model tiers.

**What you'll do**:

1. Use a fixed prompt (from notebook 02's v3) so the only variable is the
   model.
2. Run `local` (Ollama, free), `nano` (OpenAI gpt-5.4-nano), `advanced`
   (OpenAI gpt-5), and `claude` (Anthropic claude-sonnet-4-6) in sequence.
3. Inspect each output and the printed token counts and cost.
4. Read the comparison table: latency, tokens, cost per call.
5. Extrapolate to a 20-ticker strategy run to see projected daily costs.

**Key insight**: Claude's prompt-caching advantage is most visible at scale.
When the same strategy runs across 20 tickers the system prompt is cached
after call 1 — calls 2–20 pay only the cache-read rate (~10 %).  For long
system prompts this can make Claude cheaper per-ticker than `nano`.

---

## Tips

**Changing the ticker**: update `TICKER = 'AAPL'` at the top of notebook 01 to
any symbol you have configured and ingested (e.g. `'7203.T'` for Toyota).

**Iterating on prompts**: copy the v3 prompt from notebook 02 into the
dashboard at `/strategies` → New Strategy.  You can iterate on the template
there and run live strategy calls with real retrieved context.

**Running without a DB**: notebooks 02 and 03 use a hardcoded sample context
and work without PostgreSQL.  They are designed to be runnable on a laptop
with only Ollama (or API keys) available.

**Saving outputs**: Jupyter saves cell outputs in the notebook file.  Run
`uv run jupyter nbconvert --to html notebooks/03_model_comparison.ipynb` to
export a static comparison report.
