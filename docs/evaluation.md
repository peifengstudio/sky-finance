# RAG Evaluation Design

This document explains the evaluation module's design, the LLM-as-a-judge approach,
what each scoring dimension means, and how to run and interpret results.

## Why evaluate RAG quality?

sky-finance uses **sentiment-bucketed retrieval**: the corpus is split into positive,
neutral, and negative sentiment buckets and top-k chunks are retrieved from each bucket
separately before being merged. The intuition is that plain cosine similarity would
over-sample the majority sentiment (often positive) and bury minority-signal chunks —
exactly the risk signals that matter most.

But this is an *assumption*, not a measurement. The evaluation module turns it into
a testable, reproducible claim.

## What is compared

| Method | Description |
|--------|-------------|
| **Bucketed** | `top_k_positive + top_k_neutral + top_k_negative` separate queries, then merged and re-ranked by similarity |
| **Plain** | Flat cosine similarity, same total chunk budget (`top_k_positive + top_k_neutral + top_k_negative`) |

Both methods use the **same model tier**, the **same system prompt**, and the **same
`rag_query_template`** from the strategy. The only variable is how chunks are selected.

## LLM-as-a-judge

The judge is an Anthropic Claude model (default: `claude-sonnet-4-6`). It receives:

- The ticker symbol and company name
- The strategy query
- Report A (bucketed) and Report B (plain), labelled neutrally
- The chunk counts for each

It returns a structured JSON verdict:

```json
{
  "A": {"faithfulness": 8, "coverage": 7, "actionability": 6},
  "B": {"faithfulness": 6, "coverage": 5, "actionability": 7},
  "winner": "A",
  "reasoning": "Report A more thoroughly addresses downside risk signals..."
}
```

### Scoring dimensions

| Dimension | What it measures | Score range |
|-----------|-----------------|-------------|
| **Faithfulness** | Is the report grounded in the retrieved evidence? Are claims traceable to specific chunks? Low scores indicate hallucination or unsupported assertions. | 0–10 |
| **Coverage** | How broadly does the report address the relevant signals in the context? Does it capture both bull and bear cases? | 0–10 |
| **Actionability** | Is the output useful for making investment decisions? Does it give concrete signals, price levels, or risk factors rather than vague summaries? | 0–10 |

The **aggregate score** is the unweighted average of these three dimensions.

### Winner determination

The judge declares a winner based on its holistic assessment, which may differ from
the dimension averages if one report is strongly superior on the most important dimension
for that particular query. The `winner` field is the ground truth for win-rate statistics.

## Schema

Results are stored in the `eval_results` table (migration `006_eval_results.py`):

```
id                SERIAL PRIMARY KEY
strategy_id       INTEGER REFERENCES strategies(id) ON DELETE SET NULL
strategy_name     TEXT
ticker            VARCHAR(20)
query             TEXT
bucketed_n_chunks INTEGER
plain_n_chunks    INTEGER
bucketed_report   TEXT
plain_report      TEXT
bucketed_score    FLOAT          -- mean(faithfulness, coverage, actionability)
plain_score       FLOAT
bucketed_scores   JSONB          -- {"faithfulness": N, "coverage": N, "actionability": N}
plain_scores      JSONB
winner            VARCHAR(20)    -- "bucketed" | "plain" | "tie"
judge_reasoning   TEXT
judge_model       VARCHAR(100)
ran_at            TIMESTAMPTZ
```

## Running evaluations

```bash
# Evaluate all tickers in strategy 1 (default judge: claude-sonnet-4-6)
uv run sky-eval --strategy-id 1

# Evaluate a single ticker only
uv run sky-eval --strategy-id 1 --ticker AAPL

# Use a different judge model
uv run sky-eval --strategy-id 1 --judge-model claude-opus-4-7

# Show help
uv run sky-eval --help
```

The CLI prints a formatted results table and summary:

```
────────────────────────────────────────────────────────────────────────
Ticker     Bucketed   Plain  Winner     Judge reasoning
────────────────────────────────────────────────────────────────────────
AAPL       7.7/10     6.0/10 bucketed   Bucketed report better addresses downside…
TSLA       6.3/10     6.7/10 plain      Plain report captured a broader set of mac…
────────────────────────────────────────────────────────────────────────

Summary  (2 tickers evaluated)
  Bucketed wins : 1
  Plain wins    : 1
  Ties          : 0
  Avg score     : bucketed 7.0  plain 6.3  Δ +0.7
```

## Dashboard

Results are visible at `/eval` in the web dashboard. The list view shows aggregate
statistics (total evals, win rate, average scores, average delta) and a sortable table
of all runs.

Clicking "Detail →" opens the detail page (`/eval/<id>`) which shows:

- Per-dimension score breakdown table with delta column
- Judge reasoning text
- Side-by-side rendered Markdown reports (bucketed left, plain right)

## Interpreting results

**Win rate above 60%** — strong evidence that sentiment bucketing adds value for
this strategy's query profile. The improvement is consistent enough to rely on.

**Win rate 40–60%** — mixed results. Could be a marginal benefit, or the query
may not benefit from sentiment separation (e.g. purely macro queries where sentiment
doesn't differentiate quality of retrieved chunks).

**Win rate below 40%** — plain retrieval is producing better reports. Common causes:
- Strategy queries are not sentiment-sensitive
- The chunk budget is too large (both methods retrieve enough context to saturate the model)
- The `rag_threshold` is too permissive, causing both methods to pull in noise

**Score delta** — even when win rates are close, a consistent positive average delta
(+0.5 or more) confirms bucketed retrieval is contributing additional quality.

## Limitations

- The judge model can be inconsistent on borderline cases; run multiple evaluations
  per strategy to get a stable win rate estimate.
- Scores are relative to the quality of the retrieved chunks — a low score for both
  methods may indicate a corpus quality problem, not a retrieval strategy problem.
- The evaluation uses the same model that might be used in production, so very long
  reports are truncated at 3,500 characters for the judge to keep costs predictable.
