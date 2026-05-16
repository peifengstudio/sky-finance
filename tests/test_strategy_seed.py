"""Unit tests for sky_finance.strategies.seed."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from sky_finance.strategies.seed import _flatten, seed_strategies

# ---------------------------------------------------------------------------
# _flatten — pure function tests
# ---------------------------------------------------------------------------


def test_flatten_full_config():
    data = {
        "name": "momentum",
        "description": "Momentum strategy",
        "scope": "group",
        "scope_value": "us",
        "rag": {
            "query_template": "{ticker} momentum",
            "threshold": 0.6,
            "top_k_positive": 15,
            "top_k_neutral": 10,
            "top_k_negative": 5,
            "retrieval_mode": "vector",
        },
        "prompt": {"template": "Analyse {ticker}"},
        "model_tier": "cloud",
        "schedule": "0 8 * * 1-5",
        "enabled": False,
    }
    result = _flatten(data)
    assert result["name"] == "momentum"
    assert result["description"] == "Momentum strategy"
    assert result["scope"] == "group"
    assert result["scope_value"] == "us"
    assert result["rag_query_template"] == "{ticker} momentum"
    assert result["prompt_template"] == "Analyse {ticker}"
    assert result["model_tier"] == "cloud"
    assert result["schedule"] == "0 8 * * 1-5"
    assert result["enabled"] is False
    assert result["rag_threshold"] == 0.6
    assert result["rag_top_k_positive"] == 15
    assert result["rag_top_k_neutral"] == 10
    assert result["rag_top_k_negative"] == 5
    assert result["retrieval_mode"] == "vector"


def test_flatten_defaults_for_missing_optional_keys():
    result = _flatten({"name": "minimal"})
    assert result["description"] == ""
    assert result["scope"] == "global"
    assert result["scope_value"] is None
    assert result["rag_query_template"] == ""
    assert result["prompt_template"] == ""
    assert result["model_tier"] == "local"
    assert result["schedule"] is None
    assert result["enabled"] is True
    assert result["rag_threshold"] == 0.55
    assert result["rag_top_k_positive"] == 20
    assert result["rag_top_k_neutral"] == 20
    assert result["rag_top_k_negative"] == 20
    assert result["retrieval_mode"] == "hybrid"


def test_flatten_empty_scope_value_becomes_none():
    result = _flatten({"name": "x", "scope_value": ""})
    assert result["scope_value"] is None


# ---------------------------------------------------------------------------
# seed_strategies — filesystem + DB mocked
# ---------------------------------------------------------------------------


def _make_mock_conn():
    """Build a mock psycopg connection that works as a context manager."""
    mock_cur = MagicMock()
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur

    return mock_conn, mock_cur


def _make_path_mock(name: str, toml_data: dict) -> MagicMock:
    mock_file = MagicMock()
    mock_file.__enter__ = MagicMock(return_value=mock_file)
    mock_file.__exit__ = MagicMock(return_value=False)

    mock_path = MagicMock(spec=Path)
    mock_path.name = name
    mock_path.open.return_value = mock_file
    mock_path.__lt__ = lambda self, other: self.name < other.name
    return mock_path, toml_data


def test_seed_strategies_empty_dir_returns_zero():
    with patch("sky_finance.strategies.seed._STRATEGIES_DIR") as mock_dir:
        mock_dir.glob.return_value = []
        result = seed_strategies()
    assert result == 0


def test_seed_strategies_empty_dir_no_db_call():
    with (
        patch("sky_finance.strategies.seed._STRATEGIES_DIR") as mock_dir,
        patch("sky_finance.strategies.seed.get_connection") as mock_gc,
    ):
        mock_dir.glob.return_value = []
        seed_strategies()
    mock_gc.assert_not_called()


def test_seed_strategies_one_file_returns_one():
    mock_path, toml_data = _make_path_mock("test.toml", {"name": "test_strategy"})
    mock_conn, mock_cur = _make_mock_conn()

    @contextmanager
    def fake_connection():
        yield mock_conn

    with (
        patch("sky_finance.strategies.seed._STRATEGIES_DIR") as mock_dir,
        patch("sky_finance.strategies.seed.get_connection", side_effect=fake_connection),
        patch("tomllib.load", return_value=toml_data),
    ):
        mock_dir.glob.return_value = [mock_path]
        result = seed_strategies()

    assert result == 1


def test_seed_strategies_one_file_executes_and_commits():
    mock_path, toml_data = _make_path_mock("test.toml", {"name": "test_strategy"})
    mock_conn, mock_cur = _make_mock_conn()

    @contextmanager
    def fake_connection():
        yield mock_conn

    with (
        patch("sky_finance.strategies.seed._STRATEGIES_DIR") as mock_dir,
        patch("sky_finance.strategies.seed.get_connection", side_effect=fake_connection),
        patch("tomllib.load", return_value=toml_data),
    ):
        mock_dir.glob.return_value = [mock_path]
        seed_strategies()

    mock_cur.execute.assert_called_once()
    mock_conn.commit.assert_called_once()


def test_seed_strategies_execute_params_order():
    """Verify the 14-tuple argument order matches the INSERT column list."""
    toml_data = {
        "name": "alpha",
        "description": "Alpha desc",
        "scope": "ticker",
        "scope_value": "AAPL",
        "rag": {
            "query_template": "{ticker} alpha",
            "threshold": 0.7,
            "top_k_positive": 10,
            "top_k_neutral": 10,
            "top_k_negative": 10,
            "retrieval_mode": "hybrid",
        },
        "prompt": {"template": "prompt here"},
        "model_tier": "cloud",
        "schedule": "0 9 * * *",
        "enabled": True,
    }
    mock_path, _ = _make_path_mock("alpha.toml", toml_data)
    mock_conn, mock_cur = _make_mock_conn()

    @contextmanager
    def fake_connection():
        yield mock_conn

    with (
        patch("sky_finance.strategies.seed._STRATEGIES_DIR") as mock_dir,
        patch("sky_finance.strategies.seed.get_connection", side_effect=fake_connection),
        patch("tomllib.load", return_value=toml_data),
    ):
        mock_dir.glob.return_value = [mock_path]
        seed_strategies()

    args = mock_cur.execute.call_args[0][1]
    assert args == (
        "alpha",
        "Alpha desc",
        "ticker",
        "AAPL",
        "{ticker} alpha",
        "prompt here",
        "cloud",
        "0 9 * * *",
        True,
        0.7,
        10,
        10,
        10,
        "hybrid",
    )


def test_seed_strategies_two_files_returns_two():
    paths_and_data = [
        _make_path_mock("a.toml", {"name": "strat_a"}),
        _make_path_mock("b.toml", {"name": "strat_b"}),
    ]
    mock_conn, mock_cur = _make_mock_conn()
    toml_sequence = [d for _, d in paths_and_data]

    @contextmanager
    def fake_connection():
        yield mock_conn

    with (
        patch("sky_finance.strategies.seed._STRATEGIES_DIR") as mock_dir,
        patch("sky_finance.strategies.seed.get_connection", side_effect=fake_connection),
        patch("tomllib.load", side_effect=toml_sequence),
    ):
        mock_dir.glob.return_value = [p for p, _ in paths_and_data]
        result = seed_strategies()

    assert result == 2
    assert mock_cur.execute.call_count == 2
    assert mock_conn.commit.call_count == 2
