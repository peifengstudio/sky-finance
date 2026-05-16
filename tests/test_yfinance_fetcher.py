"""Unit tests for sky_finance.ingestion.yfinance_fetcher."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sky_finance.ingestion.yfinance_fetcher import (
    _JAPAN_FUNDAMENTAL_KEYS,
    _US_FUNDAMENTAL_KEYS,
    _serialize_ohlcv,
    _serialize_value,
    fetch_japan_stock,
    fetch_us_stock,
    save_raw,
)

# ---------------------------------------------------------------------------
# _serialize_value — pure function tests
# ---------------------------------------------------------------------------


def test_serialize_value_none():
    assert _serialize_value(None) is None


def test_serialize_value_nan():
    assert _serialize_value(float("nan")) is None


def test_serialize_value_bool():
    assert _serialize_value(True) is True
    assert _serialize_value(False) is False


def test_serialize_value_int():
    assert _serialize_value(42) == 42


def test_serialize_value_float():
    assert _serialize_value(3.14) == 3.14


def test_serialize_value_str():
    assert _serialize_value("hello") == "hello"


def test_serialize_value_numpy_int64():
    val = np.int64(42)
    result = _serialize_value(val)
    assert result == 42.0
    assert isinstance(result, float)


def test_serialize_value_numpy_float64():
    val = np.float64(3.14)
    result = _serialize_value(val)
    assert abs(result - 3.14) < 1e-9


def test_serialize_value_unconvertible_falls_back_to_str():
    class Weird:
        def __float__(self):
            raise TypeError("no float")

        def __str__(self):
            return "weird_repr"

    result = _serialize_value(Weird())
    assert result == "weird_repr"


# ---------------------------------------------------------------------------
# _serialize_ohlcv — pure function tests
# ---------------------------------------------------------------------------


def _make_hist_df(rows=None):
    if rows is None:
        rows = [
            {"Open": 150.0, "High": 155.0, "Low": 149.0, "Close": 153.0, "Volume": 1_000_000},
            {"Open": 151.0, "High": 156.0, "Low": 150.0, "Close": 154.0, "Volume": 1_200_000},
        ]
    idx = pd.to_datetime([f"2024-01-0{i + 2}" for i in range(len(rows))])
    return pd.DataFrame(rows, index=idx)


def test_serialize_ohlcv_returns_correct_shape():
    df = _make_hist_df()
    result = _serialize_ohlcv(df)
    assert len(result) == 2
    for row in result:
        assert set(row.keys()) == {"date", "open", "high", "low", "close", "volume"}


def test_serialize_ohlcv_date_format():
    df = _make_hist_df()
    result = _serialize_ohlcv(df)
    assert result[0]["date"] == "2024-01-02"


def test_serialize_ohlcv_nan_close_becomes_none():
    df = _make_hist_df(
        [
            {"Open": 100.0, "High": 105.0, "Low": 99.0, "Close": float("nan"), "Volume": 500},
        ]
    )
    result = _serialize_ohlcv(df)
    assert result[0]["close"] is None


def test_serialize_ohlcv_values_rounded():
    df = _make_hist_df(
        [
            {"Open": 150.123456789, "High": 155.0, "Low": 149.0, "Close": 153.0, "Volume": 1000},
        ]
    )
    result = _serialize_ohlcv(df)
    assert result[0]["open"] == round(150.123456789, 6)


# ---------------------------------------------------------------------------
# fetch_us_stock — mocked yfinance
# ---------------------------------------------------------------------------


def _make_mock_ticker(hist_df, info_dict=None):
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = hist_df
    mock_ticker.info = info_dict or {}
    return mock_ticker


def test_fetch_us_stock_happy_path():
    hist = _make_hist_df()
    mock_ticker = _make_mock_ticker(hist, {"shortName": "Apple Inc.", "marketCap": 3e12})
    with patch("sky_finance.ingestion.yfinance_fetcher.yf.Ticker", return_value=mock_ticker):
        result = fetch_us_stock("AAPL")

    assert result["ticker"] == "AAPL"
    assert result["market"] == "us"
    assert "fetched_at" in result
    assert isinstance(result["ohlcv"], list)
    assert len(result["ohlcv"]) == 2
    assert isinstance(result["fundamentals"], dict)


def test_fetch_us_stock_all_fundamental_keys_present():
    hist = _make_hist_df()
    info = {k: 1.0 for k in _US_FUNDAMENTAL_KEYS}
    mock_ticker = _make_mock_ticker(hist, info)
    with patch("sky_finance.ingestion.yfinance_fetcher.yf.Ticker", return_value=mock_ticker):
        result = fetch_us_stock("AAPL")

    for key in _US_FUNDAMENTAL_KEYS:
        assert key in result["fundamentals"]


def test_fetch_us_stock_empty_history_raises():
    mock_ticker = _make_mock_ticker(pd.DataFrame())
    with patch("sky_finance.ingestion.yfinance_fetcher.yf.Ticker", return_value=mock_ticker):
        with pytest.raises(ValueError, match="empty history"):
            fetch_us_stock("INVALID")


def test_fetch_us_stock_none_info_uses_empty_dict():
    hist = _make_hist_df()
    mock_ticker = _make_mock_ticker(hist, None)
    mock_ticker.info = None
    with patch("sky_finance.ingestion.yfinance_fetcher.yf.Ticker", return_value=mock_ticker):
        result = fetch_us_stock("AAPL")
    assert result["fundamentals"] == {k: None for k in _US_FUNDAMENTAL_KEYS}


# ---------------------------------------------------------------------------
# fetch_japan_stock — mocked yfinance
# ---------------------------------------------------------------------------


def test_fetch_japan_stock_happy_path():
    hist = _make_hist_df()
    mock_ticker = _make_mock_ticker(hist, {"shortName": "Toyota"})
    with patch("sky_finance.ingestion.yfinance_fetcher.yf.Ticker", return_value=mock_ticker):
        result = fetch_japan_stock("7203.T")

    assert result["ticker"] == "7203.T"
    assert result["market"] == "japan"


def test_fetch_japan_stock_fundamental_keys():
    hist = _make_hist_df()
    info = {k: 1.0 for k in _JAPAN_FUNDAMENTAL_KEYS}
    mock_ticker = _make_mock_ticker(hist, info)
    with patch("sky_finance.ingestion.yfinance_fetcher.yf.Ticker", return_value=mock_ticker):
        result = fetch_japan_stock("7203.T")

    for key in _JAPAN_FUNDAMENTAL_KEYS:
        assert key in result["fundamentals"]


def test_fetch_japan_stock_empty_history_raises():
    mock_ticker = _make_mock_ticker(pd.DataFrame())
    with patch("sky_finance.ingestion.yfinance_fetcher.yf.Ticker", return_value=mock_ticker):
        with pytest.raises(ValueError, match="empty history"):
            fetch_japan_stock("9999.T")


# ---------------------------------------------------------------------------
# save_raw — filesystem mocked
# ---------------------------------------------------------------------------


def test_save_raw_writes_json(tmp_path, monkeypatch):
    out_file = tmp_path / "out.json"
    monkeypatch.setattr(
        "sky_finance.ingestion.yfinance_fetcher._raw_path",
        lambda market, ticker, date: out_file,
    )
    payload = {
        "ticker": "AAPL",
        "market": "us",
        "fetched_at": "2024-01-02T10:00:00+00:00",
        "ohlcv": [],
        "fundamentals": {},
    }
    result = save_raw(payload)
    assert result == out_file
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["ticker"] == "AAPL"
