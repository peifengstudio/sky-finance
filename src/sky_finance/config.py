"""
Stock config loader with local override support.

Layout:
    config/stocks/<TICKER>.toml        — committed, safe-to-publish defaults
    config/stocks/local/<TICKER>.toml  — gitignored, personal overrides

Local files are deep-merged on top of shared files, so you only need to
specify the keys you want to change (e.g. real buy_price, shares, notes).
A ticker that only exists in local/ is also fully supported.
"""

import tomllib
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parents[2]
_STOCKS_DIR = _ROOT / "config" / "stocks"
_LOCAL_DIR = _STOCKS_DIR / "local"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_stock_config(ticker: str) -> dict[str, Any] | None:
    """Load config for one ticker, applying local override if present."""
    shared = _STOCKS_DIR / f"{ticker}.toml"
    local = _LOCAL_DIR / f"{ticker}.toml"
    if not shared.exists() and not local.exists():
        return None
    cfg: dict[str, Any] = {}
    if shared.exists():
        with shared.open("rb") as f:
            cfg = tomllib.load(f)
    if local.exists():
        with local.open("rb") as f:
            cfg = _deep_merge(cfg, tomllib.load(f))
    return cfg


def list_stock_configs(
    market: str | None = None,
    enabled_only: bool = True,
) -> list[dict[str, Any]]:
    """Return all stock configs (shared + local-only), with local overrides applied."""
    tickers: set[str] = {f.stem for f in _STOCKS_DIR.glob("*.toml")}
    if _LOCAL_DIR.exists():
        tickers |= {f.stem for f in _LOCAL_DIR.glob("*.toml")}

    configs = []
    for ticker in sorted(tickers):
        cfg = load_stock_config(ticker)
        if cfg is None:
            continue
        if enabled_only and not cfg.get("enabled", True):
            continue
        if market and cfg.get("market") != market:
            continue
        configs.append(cfg)
    return configs
