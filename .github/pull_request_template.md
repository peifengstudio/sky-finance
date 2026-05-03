## Summary

<!-- What does this PR do? 1–3 bullet points. -->

- 
- 

## Motivation

<!-- Why is this change needed? Link the related issue: "Closes #123" -->

Closes #

## Changes

<!-- List the files / modules changed and what was done to each. -->

| File / Module | Change |
|---|---|
| | |

## Testing

<!-- How did you verify this works? Tick all that apply. -->

- [ ] Added / updated unit tests
- [ ] All existing tests pass (`uv run pytest`)
- [ ] Manually tested in the dashboard (`uv run honcho start`)
- [ ] Manually triggered a Celery task and verified output

## Checklist

- [ ] `uv run ruff check src tests` passes
- [ ] `uv run ruff format src tests` passes
- [ ] `uv run mypy src` passes
- [ ] No secrets committed (API keys, tokens, passwords)
- [ ] `README.md` updated if any run command changed
- [ ] Alembic migration included if the database schema changed
- [ ] New stock configs use a per-file TOML (`config/stocks/<TICKER>.toml`), not a shared list
