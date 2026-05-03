# LLM Integration Patterns

This document covers the four production-grade LLM patterns implemented in
sky-finance.  Each pattern is independently useful — they compose in the
strategy engine to deliver reliable, cost-observable AI analysis.

---

## 1. Multi-Provider Model Routing

`src/sky_finance/strategies/engine.py`

Every strategy references a **tier name** (`local`, `nano`, `advanced`,
`claude`), never a model ID.  `run_with_model` resolves the tier to a provider
and dispatches accordingly:

```python
def run_with_model(tier, system_prompt, user_content):
    cfg = _load_model_cfg(tier)      # reads config/settings.toml
    provider = cfg["provider"]       # "ollama" | "openai" | "claude"

    if provider == "ollama":
        text, usage = _call_ollama(cfg, system_prompt, user_content)
    elif provider == "openai":
        text, usage = _call_openai(cfg, system_prompt, user_content)
    elif provider == "claude":
        text, usage = _call_claude(cfg, system_prompt, user_content)

    return text, model_id, usage
```

The same routing logic applies to the async streaming path
(`astream_with_model`).  Swapping a model requires a one-line edit in
`config/settings.toml` — no code changes.

---

## 2. Claude API + Prompt Caching

`src/sky_finance/strategies/engine.py` → `_call_claude` / `_astream_claude`

### Why prompt caching

A strategy run calls the model once per ticker.  The system prompt (strategy
instructions + RAG query guidance) is identical across all tickers — only the
user message (retrieved news chunks) changes.  Marking the system prompt
`ephemeral` lets Anthropic serve it from cache on every call after the first:

```
Call 1 (AAPL):  system prompt → paid at full input rate; written to cache
Call 2 (MSFT):  system prompt → served from cache at ~10% of full input rate
Call 3 (NVDA):  system prompt → from cache (same 5-min TTL window)
...
```

For a 10-ticker strategy run, ~90% of system-prompt tokens are served from
cache.  At `claude-sonnet-4-6` rates ($3.00/MTok input → $0.30/MTok cached),
this is a ~$0.027 saving per 10k system-prompt tokens per run — meaningful at
scale.

### Implementation

```python
response = client.messages.create(
    model=cfg["model"],
    max_tokens=cfg.get("max_tokens", 16000),
    system=[
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},  # mark for caching
        }
    ],
    messages=[{"role": "user", "content": user_content}],
)

usage = response.usage
cache_read    = getattr(usage, "cache_read_input_tokens", 0) or 0
cache_created = getattr(usage, "cache_creation_input_tokens", 0) or 0
```

- `cache_read_input_tokens > 0` → cache hit (cheap)
- `cache_creation_input_tokens > 0` → cache miss on this call, but written for
  the next one

The same `cache_control` block appears in `_astream_claude` (the SSE streaming
path) — usage is logged after `stream.get_final_message()`.

### OpenAI automatic caching

OpenAI automatically caches prompts longer than 1024 tokens (50% cost
reduction, ~1-hour TTL).  No API change is required — just read the stats:

```python
u = response.usage
cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
```

For streaming, pass `stream_options={"include_usage": True}` so the final
chunk includes usage:

```python
stream = await client.chat.completions.create(
    ...,
    stream=True,
    stream_options={"include_usage": True},
)
async for chunk in stream:
    if chunk.usage:           # final chunk
        cached = chunk.usage.prompt_tokens_details.cached_tokens
```

### Comparison

| Provider | Mechanism | Cost reduction | TTL | How to activate |
|----------|-----------|---------------|-----|-----------------|
| Claude | `cache_control: ephemeral` | ~90% on cached tokens | 5 min | Explicit header per block |
| OpenAI | Automatic on prompts ≥ 1024 tok | 50% on cached tokens | ~1 hour | None — read stats only |

---

## 3. Structured Output / Tool Use

Prompt-engineering the model to "return JSON" is fragile — models add markdown
fences, extra keys, or malformed strings.  Each provider now enforces structure
at the generation level.

### Anthropic — `tool_use` (forced)

`src/sky_finance/evaluation/judge.py` → `_call_anthropic`

```python
_SUBMIT_VERDICT_TOOL = {
    "name": "submit_verdict",
    "description": "Submit the evaluation verdict.",
    "input_schema": _VERDICT_SCHEMA,   # JSON Schema dict
}

response = client.messages.create(
    model=model,
    max_tokens=512,
    system=[{"type": "text", "text": _SYSTEM,
             "cache_control": {"type": "ephemeral"}}],
    messages=[{"role": "user", "content": user_msg}],
    tools=[_SUBMIT_VERDICT_TOOL],
    tool_choice={"type": "tool", "name": "submit_verdict"},  # force this tool
)

for block in response.content:
    if block.type == "tool_use":
        return block.input   # already a parsed dict — no json.loads needed
```

`tool_choice={"type": "tool", "name": "..."}` forces the model to call exactly
that tool.  `block.input` is already a Python dict — no JSON parsing step.

### OpenAI — `response_format: json_schema`

`src/sky_finance/evaluation/judge.py` → `_call_openai`

```python
response = client.chat.completions.create(
    model=model,
    messages=[...],
    temperature=0,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "judge_verdict",
            "strict": True,          # no extra keys; required == all props
            "schema": _VERDICT_SCHEMA,
        },
    },
)
result = json.loads(response.choices[0].message.content)
# json.loads will not raise — schema is enforced at generation time
```

`strict: True` enables constrained decoding — the sampler can only produce
tokens that extend a valid schema-conformant JSON string.

### Ollama — `format: <schema>`

`src/sky_finance/evaluation/judge.py` → `_call_ollama`  
`src/sky_finance/pipeline/llm_summariser.py`

```python
# judge.py
payload = {"model": model, "messages": [...], "format": _VERDICT_SCHEMA}

# llm_summariser.py (Ollama SDK)
response = client.chat(
    model=model,
    messages=[...],
    format=_FORMAT_SCHEMA,    # dict instead of the old "json" string
)
```

Passing a JSON Schema dict instead of the string `"json"` activates
schema-constrained generation (Ollama ≥ 0.5).  The enum constraint on
`sentiment` (`["positive","neutral","negative"]`) means the model cannot
output an invalid value regardless of prompt phrasing.

### One schema, three providers

The judge uses a single `_VERDICT_SCHEMA` dict shared across all three
`_call_*` functions:

```python
_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "A": _SCORES_SCHEMA,
        "B": _SCORES_SCHEMA,
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "reasoning": {"type": "string"},
    },
    "required": ["A", "B", "winner", "reasoning"],
    "additionalProperties": False,
}
```

---

## 4. Cost Tracking

`src/sky_finance/strategies/costs.py`

Every LLM call produces a `UsageStats` object that is written to
`strategy_results.metadata["usage"]` and surfaced in the dashboard detail page.

### Pricing table

```python
_PRICING = {
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00,
                          "cache_read": 0.30, "cache_write": 3.75},
    "gpt-5":             {"in": 10.00, "out": 30.00,
                          "cache_read": 5.00,  "cache_write": 0.0},
    "gpt-5.4-nano":      {"in": 0.30,  "out": 1.20,
                          "cache_read": 0.15,  "cache_write": 0.0},
    ...
}
```

Prices are per 1 million tokens (USD).  Update this table when you change
models or when providers adjust their rates — no other code needs to change.

### Cost formula

`input_tokens` is always the **total** prompt-side token count (cached +
non-cached + cache-creation).  The formula bills each segment at its own rate:

```
uncached = input_tokens − cached_tokens − cache_creation_tokens
cost     = uncached            × in_rate
         + cached_tokens       × cache_read_rate
         + cache_creation_tokens × cache_write_rate   ← Anthropic only
         + output_tokens       × out_rate
```

This is provider-neutral: for OpenAI, `cache_creation_tokens` is always 0.

### UsageStats dataclass

```python
@dataclass
class UsageStats:
    model: str
    provider: str               # "openai" | "claude" | "ollama"
    input_tokens: int           # total prompt tokens (cached + non-cached)
    output_tokens: int
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float | None = None   # None if model not in pricing table

    def to_dict(self) -> dict: ...  # stored in JSONB metadata column
```

Ollama always receives `cost_usd=0.0` (not `None`) — the distinction matters in
the template (`$0` vs `?`).

### Data flow

```
_call_claude / _call_openai / _call_ollama
        │
        └─▶ compute_cost(model, provider, input, output, cached, cache_creation)
                │
                └─▶ UsageStats
                        │
        run_with_model ─┘ (returned as 3rd element)
                │
        run_strategy ──▶  metadata["usage"] = usage.to_dict()
                │
        save_strategy_result ──▶ strategy_results.metadata (JSONB)
                │
        dashboard /strategies/results/<id> ──▶ cost card
```

For the streaming path, `astream_with_model` accepts a `usage_out: list`
out-parameter that is populated after the generator exhausts:

```python
usage_out: list = []
async for chunk in astream_with_model(tier, sys, user, usage_out=usage_out):
    yield chunk
usage_dict = usage_out[0].to_dict() if usage_out else None
```

### Dashboard display

The report detail page shows a four-box cost card:

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  2,840       │  312         │  1,024       │  $0.0043     │
│ input tokens │ output tokens│ cache written│  est. cost   │
│ 1,024 from   │              │              │              │
│  cache       │              │              │              │
└──────────────┴──────────────┴──────────────┴──────────────┘
  estimated from published claude rates · claude-sonnet-4-6
```

- Green sub-text on input tokens when `cached_tokens > 0`
- Sky-blue "cache written" for Anthropic cache-creation events
- Amber for cloud costs; slate `$0` for Ollama; `?` for unknown models

### Updating prices

Edit `_PRICING` in `src/sky_finance/strategies/costs.py`.  The key must match
the model ID string exactly (e.g. `"claude-sonnet-4-6"`, `"gpt-5.4-nano"`).
Models not in the table get `cost_usd=None` — they still track tokens.
