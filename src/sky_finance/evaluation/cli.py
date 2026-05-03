"""
CLI entry point for the evaluation module.

Usage
-----
    # Evaluate all tickers in strategy 1 (uses strategy's own model tier)
    uv run sky-eval --strategy-id 1

    # Evaluate a single ticker
    uv run sky-eval --strategy-id 1 --ticker AAPL

    # Override the report-generation model (e.g. use local Ollama instead of nano/claude)
    uv run sky-eval --strategy-id 1 --model-tier local

    # Use a different judge model
    uv run sky-eval --strategy-id 1 --judge-model claude-opus-4-7

Results are stored in the eval_results table and visible at /eval in the dashboard.
"""

import argparse
import sys

from sky_finance.logging_config import setup_logging


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        prog="sky-eval",
        description=(
            "Evaluate RAG quality: sentiment-bucketed retrieval vs plain retrieval.\n"
            "Uses LLM-as-a-judge to score faithfulness, coverage, and actionability."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--strategy-id",
        type=int,
        required=True,
        metavar="ID",
        help="Strategy ID to evaluate (see /strategies in the dashboard)",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        metavar="SYMBOL",
        help="Evaluate a single ticker symbol (default: all tickers in strategy scope)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only evaluate the first N tickers — useful for quick smoke tests.",
    )
    parser.add_argument(
        "--model-tier",
        type=str,
        default=None,
        metavar="TIER",
        help=(
            "Model tier used to generate reports (default: strategy's own tier). "
            "Override with 'local' to use Ollama when the strategy tier requires a "
            "paid API key (nano / advanced / claude)."
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="claude-sonnet-4-6",
        metavar="MODEL",
        help="Anthropic model used as judge (default: claude-sonnet-4-6)",
    )
    args = parser.parse_args()

    from sky_finance.evaluation.runner import run_eval

    print(
        f"\nEvaluating strategy {args.strategy_id}"
        + (f" · ticker {args.ticker}" if args.ticker else " · all tickers")
        + (f" · tier override {args.model_tier}" if args.model_tier else "")
        + f" · judge {args.judge_model}\n"
    )

    try:
        results = run_eval(
            strategy_id=args.strategy_id,
            ticker=args.ticker,
            judge_model=args.judge_model,
            model_tier=args.model_tier,
            limit=args.limit,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("No results — check that the strategy has tickers in scope.")
        sys.exit(0)

    # ── Results table ──────────────────────────────────────────────────────
    col = "{:<10} {:>9} {:>7} {:<10} {}"
    sep = "─" * 72
    print(sep)
    print(col.format("Ticker", "Bucketed", "Plain", "Winner", "Judge reasoning"))
    print(sep)
    for r in results:
        snippet = (r["judge_reasoning"] or "")[:55]
        if len(r["judge_reasoning"]) > 55:
            snippet += "…"
        print(
            col.format(
                r["ticker"],
                f"{r['bucketed_score']:.1f}/10",
                f"{r['plain_score']:.1f}/10",
                r["winner"],
                snippet,
            )
        )
    print(sep)

    # ── Summary ────────────────────────────────────────────────────────────
    bucketed_wins = sum(1 for r in results if r["winner"] == "bucketed")
    plain_wins = sum(1 for r in results if r["winner"] == "plain")
    ties = len(results) - bucketed_wins - plain_wins

    avg_b = sum(r["bucketed_score"] for r in results) / len(results)
    avg_p = sum(r["plain_score"] for r in results) / len(results)
    delta = avg_b - avg_p

    print(f"\nSummary  ({len(results)} tickers evaluated)")
    print(f"  Bucketed wins : {bucketed_wins}")
    print(f"  Plain wins    : {plain_wins}")
    print(f"  Ties          : {ties}")
    print(f"  Avg score     : bucketed {avg_b:.1f}  plain {avg_p:.1f}  Δ {delta:+.1f}")
    print("\nResults saved to eval_results table. View at /eval in the dashboard.\n")


if __name__ == "__main__":
    main()
