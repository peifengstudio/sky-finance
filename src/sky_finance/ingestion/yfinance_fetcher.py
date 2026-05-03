"""
yfinance fetcher — fetch and persist raw stock data.

Raw files are stored under:
    data/raw/{market}/{ticker}/{YYYY-MM-DD}.json   (date = UTC fetch date)

Two fetch profiles:
  - US stocks   : OHLCV (5 days, 1d interval) + key fundamentals from yfinance info
  - Japan stocks: OHLCV (5 days, 1d interval) + basic info
                  (yfinance fundamental coverage for .T tickers is limited)
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Key fundamentals to extract for US stocks.
# Keeping an explicit list avoids pulling in hundreds of noisy / unstable fields.
# ---------------------------------------------------------------------------
_US_FUNDAMENTAL_KEYS = (
    "shortName",
    "longName",
    "sector",
    "industry",
    "currency",
    "exchange",
    "marketCap",
    "enterpriseValue",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "trailingEps",
    "forwardEps",
    "revenuePerShare",
    "returnOnEquity",
    "returnOnAssets",
    "operatingMargins",
    "profitMargins",
    "dividendYield",
    "payoutRatio",
    "beta",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "averageVolume",
    "sharesOutstanding",
)

_JAPAN_FUNDAMENTAL_KEYS = (
    "shortName",
    "longName",
    "currency",
    "exchange",
    "marketCap",
    "trailingPE",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "averageVolume",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _serialize_value(val: Any) -> Any:
    """Convert numpy / pandas scalars to plain Python types for JSON."""
    if val is None or val != val:  # NaN check
        return None
    if isinstance(val, (bool, int, float, str)):
        return val
    try:
        return float(val)
    except TypeError, ValueError:
        return str(val)


def _extract_info(info: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: _serialize_value(info.get(k)) for k in keys}


def _serialize_ohlcv(hist: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert yfinance history DataFrame to a JSON-serializable list of dicts."""
    records = []
    for ts, row in hist.iterrows():
        records.append(
            {
                "date": str(ts)[:10],
                "open": round(float(row["Open"]), 6) if pd.notna(row.get("Open")) else None,
                "high": round(float(row["High"]), 6) if pd.notna(row.get("High")) else None,
                "low": round(float(row["Low"]), 6) if pd.notna(row.get("Low")) else None,
                "close": round(float(row["Close"]), 6) if pd.notna(row.get("Close")) else None,
                "volume": int(row["Volume"]) if pd.notna(row.get("Volume")) else None,
            }
        )
    return records


def _raw_path(market: str, ticker: str, date_utc: str) -> Path:
    """
    Returns the file path for a raw payload, creating parent dirs if needed.
    e.g. data/raw/us/AAPL/2025-04-17.json
         data/raw/japan/7203.T/2025-04-17.json
    """
    # Resolve relative to project root (3 levels up from this file in src/)
    base = Path(__file__).parents[3] / "data" / "raw" / market / ticker
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{date_utc}.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_us_stock(ticker: str) -> dict[str, Any]:
    """
    Fetch OHLCV (5-day, 1d bars) and key fundamentals for a US equity.

    Args:
        ticker: plain ticker symbol, e.g. "AAPL".

    Returns:
        dict with keys: ticker, market, fetched_at, ohlcv, fundamentals.

    Raises:
        ValueError: if yfinance returns an empty history (ticker may be delisted).
        RuntimeError: on unexpected yfinance errors.
    """
    logger.info("yfinance fetch [us]: %s", ticker)
    yf_ticker = yf.Ticker(ticker)

    hist = yf_ticker.history(period="5d", interval="1d", auto_adjust=True)
    if hist.empty:
        raise ValueError(f"yfinance returned empty history for {ticker}")

    info = yf_ticker.info or {}

    return {
        "ticker": ticker,
        "market": "us",
        "fetched_at": datetime.now(UTC).isoformat(),
        "ohlcv": _serialize_ohlcv(hist),
        "fundamentals": _extract_info(info, _US_FUNDAMENTAL_KEYS),
    }


def fetch_japan_stock(ticker: str) -> dict[str, Any]:
    """
    Fetch OHLCV (5-day, 1d bars) and basic info for a Tokyo-listed equity.

    Args:
        ticker: ticker with .T suffix, e.g. "7203.T".

    Returns:
        dict with keys: ticker, market, fetched_at, ohlcv, fundamentals.

    Raises:
        ValueError: if yfinance returns an empty history.
        RuntimeError: on unexpected yfinance errors.
    """
    logger.info("yfinance fetch [japan]: %s", ticker)
    yf_ticker = yf.Ticker(ticker)

    hist = yf_ticker.history(period="5d", interval="1d", auto_adjust=True)
    if hist.empty:
        raise ValueError(f"yfinance returned empty history for {ticker}")

    info = yf_ticker.info or {}

    return {
        "ticker": ticker,
        "market": "japan",
        "fetched_at": datetime.now(UTC).isoformat(),
        "ohlcv": _serialize_ohlcv(hist),
        "fundamentals": _extract_info(info, _JAPAN_FUNDAMENTAL_KEYS),
    }


def save_raw(payload: dict[str, Any]) -> Path:
    """
    Persist a raw fetch payload to disk.

    File path: data/raw/{market}/{ticker}/{YYYY-MM-DD}.json
    Date is derived from fetched_at (UTC) in the payload.

    Args:
        payload: dict returned by fetch_us_stock() or fetch_japan_stock().

    Returns:
        Path of the written file.
    """
    market = payload["market"]
    ticker = payload["ticker"]
    date_utc = datetime.fromisoformat(payload["fetched_at"]).strftime("%Y-%m-%d")

    path = _raw_path(market, ticker, date_utc)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Saved raw data → %s", path)
    return path
