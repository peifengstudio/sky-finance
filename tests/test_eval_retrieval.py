"""Unit tests for sky_finance.evaluation.retrieval."""

from unittest.mock import MagicMock, patch

from sky_finance.evaluation.retrieval import plain_rag_fetch


def _make_mock_conn(rows):
    """Build a mock psycopg connection with a cursor that returns given rows."""
    mock_cur = MagicMock()
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_cur.fetchall.return_value = rows

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    return mock_conn


def _call_plain_rag(conn, query_template="{ticker} outlook", ticker="AAPL", company_name="Apple"):
    # embed_single and register_vector are locally imported inside plain_rag_fetch,
    # so patch them at their source module locations.
    with (
        patch("sky_finance.storage.embedder.embed_single", return_value=[0.1] * 768),
        patch("pgvector.psycopg.register_vector"),
    ):
        return plain_rag_fetch(conn, query_template, ticker, company_name)


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_plain_rag_fetch_two_rows_returns_two_structured():
    rows = [
        ("Title A", "Body A", "positive", 0.85),
        ("Title B", "Body B", "negative", 0.72),
    ]
    conn = _make_mock_conn(rows)
    text, structured = _call_plain_rag(conn)

    assert len(structured) == 2
    assert structured[0]["title"] == "Title A"
    assert structured[1]["sentiment"] == "negative"


def test_plain_rag_fetch_text_contains_markdown_headers():
    rows = [("Title A", "Body A", "positive", 0.85)]
    conn = _make_mock_conn(rows)
    text, _ = _call_plain_rag(conn)
    assert "###" in text
    assert "Title A" in text


def test_plain_rag_fetch_empty_returns_placeholder():
    conn = _make_mock_conn([])
    text, structured = _call_plain_rag(conn)
    assert "No relevant documents" in text
    assert structured == []


# ---------------------------------------------------------------------------
# Query template substitution
# ---------------------------------------------------------------------------


def test_plain_rag_fetch_ticker_substituted_in_query():
    rows = [("T", "B", "neutral", 0.7)]
    conn = _make_mock_conn(rows)
    with (
        patch("sky_finance.storage.embedder.embed_single", return_value=[0.1] * 768) as mock_embed,
        patch("pgvector.psycopg.register_vector"),
    ):
        plain_rag_fetch(conn, "{ticker} earnings outlook", "TSLA", "Tesla")

    call_arg = mock_embed.call_args[0][0]
    assert "TSLA" in call_arg
    assert "{ticker}" not in call_arg


def test_plain_rag_fetch_company_name_appended_to_query():
    rows = [("T", "B", "neutral", 0.7)]
    conn = _make_mock_conn(rows)
    with (
        patch("sky_finance.storage.embedder.embed_single", return_value=[0.1] * 768) as mock_embed,
        patch("pgvector.psycopg.register_vector"),
    ):
        plain_rag_fetch(conn, "{ticker} outlook", "AAPL", "Apple Inc.")

    call_arg = mock_embed.call_args[0][0]
    assert "Apple Inc." in call_arg


# ---------------------------------------------------------------------------
# Sentiment tag in output
# ---------------------------------------------------------------------------


def test_plain_rag_fetch_sentiment_tag_present():
    rows = [("Title", "Body", "bearish", 0.8)]
    conn = _make_mock_conn(rows)
    text, _ = _call_plain_rag(conn)
    assert "[bearish]" in text


def test_plain_rag_fetch_empty_sentiment_no_bracket_tag():
    rows = [("Title", "Body", "", 0.8)]
    conn = _make_mock_conn(rows)
    text, _ = _call_plain_rag(conn)
    # No empty brackets like "[]" should appear
    assert "[]" not in text


def test_plain_rag_fetch_sim_rounded_in_structured():
    rows = [("T", "B", "positive", 0.856789)]
    conn = _make_mock_conn(rows)
    _, structured = _call_plain_rag(conn)
    assert structured[0]["sim"] == round(0.856789, 3)
