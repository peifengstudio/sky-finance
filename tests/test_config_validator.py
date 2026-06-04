"""Unit tests for sky_finance.config_validator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sky_finance.config_validator as _cv_module
from sky_finance.config_validator import (
    FileResult,
    Issue,
    _render_report,
    _validate_cfg,
    validate_all,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_CFG: dict = {
    "ticker": "AAPL",
    "name": "Apple Inc.",
    "market": "us",
    "enabled": True,
    "price": {
        "buy_price": 150.0,
        "stop_loss": 130.0,
        "targets": [170.0, 190.0, 210.0],
        "alerts": [145.0, 140.0, 135.0],
    },
    "position": {
        "shares": 10,
        "max_weight": 0.08,
        "notes": "Watchlist.",
    },
    "ingestion": {
        "l1_keywords": ["Apple earnings"],
        "l2_topics": ["AI smartphone"],
        "l3_macro": ["Fed rate"],
    },
    "analysis": {"signal_chain": "demand → revenue → stock"},
    "strategies": {"enabled": ["momentum_crosscheck"]},
}

_FAKE_PATH = Path("/project/config/stocks/AAPL.toml")


def _errors(issues: list[Issue]) -> list[str]:
    return [i.field for i in issues if i.level == "error"]


def _warnings(issues: list[Issue]) -> list[str]:
    return [i.field for i in issues if i.level == "warning"]


# ---------------------------------------------------------------------------
# _validate_cfg — valid config produces no issues
# ---------------------------------------------------------------------------


def test_valid_config_no_issues():
    issues = _validate_cfg(_VALID_CFG, _FAKE_PATH)
    assert issues == []


# ---------------------------------------------------------------------------
# Required top-level fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing_key", ["ticker", "name", "market", "enabled"])
def test_missing_required_field_is_error(missing_key):
    cfg = {k: v for k, v in _VALID_CFG.items() if k != missing_key}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert missing_key in _errors(issues)


def test_empty_ticker_is_error():
    cfg = {**_VALID_CFG, "ticker": ""}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert "ticker" in _errors(issues)


def test_ticker_mismatch_filename_is_error():
    cfg = {**_VALID_CFG, "ticker": "NVDA"}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert "ticker" in _errors(issues)


def test_empty_name_is_error():
    cfg = {**_VALID_CFG, "name": ""}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert "name" in _errors(issues)


# ---------------------------------------------------------------------------
# market enum
# ---------------------------------------------------------------------------


def test_invalid_market_is_error():
    cfg = {**_VALID_CFG, "market": "eu"}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert "market" in _errors(issues)


@pytest.mark.parametrize("valid_market", ["us", "japan"])
def test_valid_markets_accepted(valid_market):
    cfg = {**_VALID_CFG, "market": valid_market}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert "market" not in _errors(issues)


# ---------------------------------------------------------------------------
# enabled type
# ---------------------------------------------------------------------------


def test_enabled_non_bool_is_error():
    cfg = {**_VALID_CFG, "enabled": "yes"}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert "enabled" in _errors(issues)


# ---------------------------------------------------------------------------
# [price] section
# ---------------------------------------------------------------------------


def test_missing_price_section_is_warning():
    cfg = {k: v for k, v in _VALID_CFG.items() if k != "price"}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert "price" in _warnings(issues)


@pytest.mark.parametrize("arr_key", ["targets", "alerts"])
def test_missing_price_array_is_error(arr_key):
    cfg = {**_VALID_CFG, "price": {k: v for k, v in _VALID_CFG["price"].items() if k != arr_key}}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert f"price.{arr_key}" in _errors(issues)


@pytest.mark.parametrize("arr_key", ["targets", "alerts"])
def test_wrong_length_price_array_is_error(arr_key):
    cfg = {**_VALID_CFG, "price": {**_VALID_CFG["price"], arr_key: [1.0, 2.0]}}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert f"price.{arr_key}" in _errors(issues)


@pytest.mark.parametrize("arr_key", ["targets", "alerts"])
def test_non_numeric_price_array_element_is_error(arr_key):
    cfg = {**_VALID_CFG, "price": {**_VALID_CFG["price"], arr_key: [1.0, "bad", 3.0]}}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert any(i.field.startswith(f"price.{arr_key}[") for i in issues if i.level == "error")


# ---------------------------------------------------------------------------
# [position] section
# ---------------------------------------------------------------------------


def test_missing_position_section_is_warning():
    cfg = {k: v for k, v in _VALID_CFG.items() if k != "position"}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert "position" in _warnings(issues)


def test_max_weight_above_one_is_warning():
    cfg = {**_VALID_CFG, "position": {**_VALID_CFG["position"], "max_weight": 1.5}}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert "position.max_weight" in _warnings(issues)


def test_max_weight_zero_is_warning():
    cfg = {**_VALID_CFG, "position": {**_VALID_CFG["position"], "max_weight": 0.0}}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert "position.max_weight" in _warnings(issues)


# ---------------------------------------------------------------------------
# [ingestion] section
# ---------------------------------------------------------------------------


def test_missing_ingestion_section_is_warning():
    cfg = {k: v for k, v in _VALID_CFG.items() if k != "ingestion"}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert "ingestion" in _warnings(issues)


@pytest.mark.parametrize("kw_key", ["l1_keywords", "l2_topics", "l3_macro"])
def test_missing_keyword_list_is_warning(kw_key):
    ingestion = {k: v for k, v in _VALID_CFG["ingestion"].items() if k != kw_key}
    cfg = {**_VALID_CFG, "ingestion": ingestion}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert f"ingestion.{kw_key}" in _warnings(issues)


@pytest.mark.parametrize("kw_key", ["l1_keywords", "l2_topics", "l3_macro"])
def test_empty_keyword_list_is_warning(kw_key):
    ingestion = {**_VALID_CFG["ingestion"], kw_key: []}
    cfg = {**_VALID_CFG, "ingestion": ingestion}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert f"ingestion.{kw_key}" in _warnings(issues)


def test_oversized_keyword_list_is_warning():
    ingestion = {**_VALID_CFG["ingestion"], "l1_keywords": [f"kw{i}" for i in range(25)]}
    cfg = {**_VALID_CFG, "ingestion": ingestion}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert "ingestion.l1_keywords" in _warnings(issues)


# ---------------------------------------------------------------------------
# [strategies] section
# ---------------------------------------------------------------------------


def test_missing_strategies_section_is_warning():
    cfg = {k: v for k, v in _VALID_CFG.items() if k != "strategies"}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert "strategies" in _warnings(issues)


def test_strategies_enabled_not_list_is_error():
    cfg = {**_VALID_CFG, "strategies": {"enabled": "momentum_crosscheck"}}
    issues = _validate_cfg(cfg, _FAKE_PATH)
    assert "strategies.enabled" in _errors(issues)


# ---------------------------------------------------------------------------
# validate_all — filesystem-level tests
# ---------------------------------------------------------------------------


def _make_toml_path(stem: str) -> MagicMock:
    p = MagicMock(spec=Path)
    p.stem = stem
    p.name = f"{stem}.toml"
    return p


def test_validate_all_parse_error_recorded(tmp_path):
    bad_toml = tmp_path / "BAD.toml"
    bad_toml.write_bytes(b"ticker = [unclosed")

    with (
        patch("sky_finance.config_validator._STOCKS_DIR", tmp_path),
        patch("sky_finance.config_validator._LOCAL_DIR", tmp_path / "local"),
    ):
        results = validate_all()

    assert len(results) == 1
    assert results[0].ticker == "BAD"
    assert any(i.level == "error" for i in results[0].issues)


def test_validate_all_valid_file_passes(tmp_path):

    toml_content = (
        'ticker   = "AAPL"\n'
        'name     = "Apple Inc."\n'
        'market   = "us"\n'
        "enabled  = true\n"
        "[price]\n"
        "buy_price = 0.0\n"
        "stop_loss = 0.0\n"
        "targets   = [0.0, 0.0, 0.0]\n"
        "alerts    = [0.0, 0.0, 0.0]\n"
        "[position]\n"
        "shares     = 0\n"
        "max_weight = 0.08\n"
        'notes      = ""\n'
        "[ingestion]\n"
        'l1_keywords = ["Apple earnings"]\n'
        'l2_topics   = ["AI smartphone"]\n'
        'l3_macro    = ["Fed rate"]\n'
        "[strategies]\n"
        'enabled = ["momentum_crosscheck"]\n'
    )
    (tmp_path / "AAPL.toml").write_text(toml_content)

    with (
        patch("sky_finance.config_validator._STOCKS_DIR", tmp_path),
        patch("sky_finance.config_validator._LOCAL_DIR", tmp_path / "local"),
    ):
        results = validate_all()

    assert len(results) == 1
    assert results[0].ok
    assert results[0].issues == []


# ---------------------------------------------------------------------------
# _render_report — exit codes and output
# ---------------------------------------------------------------------------


def _make_result(
    ticker: str, issues: list[Issue] | None = None, root: Path | None = None
) -> FileResult:
    base = root or Path("/tmp")
    r = FileResult(path=base / f"config/stocks/{ticker}.toml", ticker=ticker)
    if issues:
        r.issues = issues
    return r


def test_render_report_all_ok_returns_zero(capsys, tmp_path):
    with patch.object(_cv_module, "_ROOT", tmp_path):
        results = [_make_result("AAPL", root=tmp_path), _make_result("NVDA", root=tmp_path)]
        code = _render_report(results)
    assert code == 0


def test_render_report_with_error_returns_one(capsys, tmp_path):
    with patch.object(_cv_module, "_ROOT", tmp_path):
        results = [
            _make_result("AAPL", root=tmp_path),
            _make_result("BAD", [Issue("error", "ticker", "missing")], root=tmp_path),
        ]
        code = _render_report(results)
    assert code == 1


def test_render_report_warning_only_returns_zero(capsys, tmp_path):
    with patch.object(_cv_module, "_ROOT", tmp_path):
        results = [
            _make_result("AAPL", [Issue("warning", "price", "missing section")], root=tmp_path)
        ]
        code = _render_report(results)
    assert code == 0


def test_render_report_output_contains_ticker(capsys, tmp_path):
    with patch.object(_cv_module, "_ROOT", tmp_path):
        results = [_make_result("AAPL", root=tmp_path)]
        _render_report(results)
    out = capsys.readouterr().out
    assert "AAPL" in out


def test_render_report_fail_line_shown_for_error(capsys, tmp_path):
    with patch.object(_cv_module, "_ROOT", tmp_path):
        results = [_make_result("BAD", [Issue("error", "market", "bad value")], root=tmp_path)]
        _render_report(results)
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "market" in out
