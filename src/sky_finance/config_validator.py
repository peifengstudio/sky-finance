"""
Stock config validator.

Validates all TOML files under config/stocks/ against the expected schema and
emits a structured report.  Used by `make validate-stocks`.
"""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parents[2]
_STOCKS_DIR = _ROOT / "config" / "stocks"
_LOCAL_DIR = _STOCKS_DIR / "local"

VALID_MARKETS = {"us", "japan"}

# Maximum sensible list lengths (warn above these)
_MAX_KEYWORDS = 20
_MAX_TOPICS = 20
_MAX_MACRO = 20

# Expected targets/alerts tuple length
_PRICE_ARRAY_LEN = 3


@dataclass
class Issue:
    level: str  # "error" | "warning"
    field: str
    message: str


@dataclass
class FileResult:
    path: Path
    ticker: str
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


def _err(issues: list[Issue], field: str, msg: str) -> None:
    issues.append(Issue("error", field, msg))


def _warn(issues: list[Issue], field: str, msg: str) -> None:
    issues.append(Issue("warning", field, msg))


def _validate_cfg(cfg: dict[str, Any], path: Path) -> list[Issue]:
    issues: list[Issue] = []

    # ── Required top-level fields ──────────────────────────────────────────
    for key in ("ticker", "name", "market", "enabled"):
        if key not in cfg:
            _err(issues, key, f"Required field '{key}' is missing")

    # ── ticker matches filename ────────────────────────────────────────────
    if "ticker" in cfg:
        if not isinstance(cfg["ticker"], str) or not cfg["ticker"].strip():
            _err(issues, "ticker", "Must be a non-empty string")
        elif cfg["ticker"] != path.stem:
            _err(
                issues,
                "ticker",
                f"ticker value '{cfg['ticker']}' does not match filename '{path.stem}.toml'",
            )

    # ── name ───────────────────────────────────────────────────────────────
    if "name" in cfg:
        if not isinstance(cfg["name"], str) or not cfg["name"].strip():
            _err(issues, "name", "Must be a non-empty string")

    # ── market enum ────────────────────────────────────────────────────────
    if "market" in cfg:
        if cfg["market"] not in VALID_MARKETS:
            _err(
                issues,
                "market",
                f"Invalid market '{cfg['market']}'; must be one of {sorted(VALID_MARKETS)}",
            )

    # ── enabled type ───────────────────────────────────────────────────────
    if "enabled" in cfg and not isinstance(cfg["enabled"], bool):
        _err(issues, "enabled", "Must be a boolean (true / false)")

    # ── [price] section ────────────────────────────────────────────────────
    price = cfg.get("price")
    if price is None:
        _warn(issues, "price", "[price] section is missing")
    else:
        for arr_key in ("targets", "alerts"):
            val = price.get(arr_key)
            if val is None:
                _err(issues, f"price.{arr_key}", f"Required array 'price.{arr_key}' is missing")
            elif not isinstance(val, list):
                _err(issues, f"price.{arr_key}", f"'price.{arr_key}' must be an array")
            elif len(val) != _PRICE_ARRAY_LEN:
                _err(
                    issues,
                    f"price.{arr_key}",
                    f"'price.{arr_key}' must have exactly {_PRICE_ARRAY_LEN} elements"
                    f" (got {len(val)})",
                )
            else:
                for i, v in enumerate(val):
                    if not isinstance(v, (int, float)):
                        _err(
                            issues,
                            f"price.{arr_key}[{i}]",
                            f"All elements must be numeric (got {type(v).__name__!r})",
                        )

        for num_key in ("buy_price", "stop_loss"):
            val = price.get(num_key)
            if val is not None and not isinstance(val, (int, float)):
                _err(issues, f"price.{num_key}", "Must be numeric")

    # ── [position] section ─────────────────────────────────────────────────
    position = cfg.get("position")
    if position is None:
        _warn(issues, "position", "[position] section is missing")
    else:
        mw = position.get("max_weight")
        if mw is not None:
            if not isinstance(mw, (int, float)):
                _err(issues, "position.max_weight", "Must be numeric")
            elif not (0 < mw <= 1):
                _warn(
                    issues,
                    "position.max_weight",
                    f"max_weight={mw} is outside (0, 1] — looks suspicious",
                )

    # ── [ingestion] section ────────────────────────────────────────────────
    ingestion = cfg.get("ingestion")
    if ingestion is None:
        _warn(issues, "ingestion", "[ingestion] section is missing")
    else:
        for kw_key, limit in (
            ("l1_keywords", _MAX_KEYWORDS),
            ("l2_topics", _MAX_TOPICS),
            ("l3_macro", _MAX_MACRO),
        ):
            val = ingestion.get(kw_key)
            if val is None:
                _warn(issues, f"ingestion.{kw_key}", f"'{kw_key}' is missing")
            elif not isinstance(val, list):
                _err(issues, f"ingestion.{kw_key}", f"'{kw_key}' must be an array")
            else:
                if len(val) == 0:
                    _warn(issues, f"ingestion.{kw_key}", f"'{kw_key}' is empty")
                if len(val) > limit:
                    _warn(
                        issues,
                        f"ingestion.{kw_key}",
                        f"'{kw_key}' has {len(val)} entries (>{limit}) — consider trimming",
                    )

    # ── [strategies] section ───────────────────────────────────────────────
    strategies = cfg.get("strategies")
    if strategies is None:
        _warn(issues, "strategies", "[strategies] section is missing")
    else:
        enabled = strategies.get("enabled")
        if enabled is None:
            _warn(issues, "strategies.enabled", "'strategies.enabled' is missing")
        elif not isinstance(enabled, list):
            _err(issues, "strategies.enabled", "'strategies.enabled' must be an array")

    return issues


def _load_raw(path: Path) -> dict[str, Any] | str:
    """Return parsed TOML dict, or an error string on parse failure."""
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        return f"TOML parse error: {exc}"


def validate_all(include_local: bool = False) -> list[FileResult]:
    paths: list[Path] = sorted(_STOCKS_DIR.glob("*.toml"))
    if include_local and _LOCAL_DIR.exists():
        paths += sorted(_LOCAL_DIR.glob("*.toml"))

    results: list[FileResult] = []
    for path in paths:
        raw = _load_raw(path)
        if isinstance(raw, str):
            result = FileResult(path=path, ticker=path.stem)
            result.issues.append(Issue("error", "<file>", raw))
            results.append(result)
            continue

        issues = _validate_cfg(raw, path)
        results.append(FileResult(path=path, ticker=path.stem, issues=issues))

    return results


# ── CLI / report rendering ──────────────────────────────────────────────────


def _render_report(results: list[FileResult]) -> int:
    """Print a structured validation report.  Returns exit code (0 = all OK)."""
    total = len(results)
    n_ok = sum(1 for r in results if r.ok)
    n_fail = total - n_ok

    width = 60
    print("=" * width)
    print(f"  Stock config validation — {total} file(s)")
    print("=" * width)

    for result in results:
        rel = result.path.relative_to(_ROOT)
        if result.ok and not result.warnings:
            print(f"  OK   {rel}")
        else:
            status = "FAIL" if result.errors else "WARN"
            print(f"  {status} {rel}")
            for issue in result.issues:
                icon = "✗" if issue.level == "error" else "⚠"
                print(f"         {icon} [{issue.field}] {issue.message}")

    print("-" * width)
    summary_parts = [f"{n_ok}/{total} passed"]
    if n_fail:
        summary_parts.append(f"{n_fail} with errors")
    n_warn_files = sum(1 for r in results if r.warnings and r.ok)
    if n_warn_files:
        summary_parts.append(f"{n_warn_files} with warnings only")
    print(f"  {', '.join(summary_parts)}")
    print("=" * width)

    return 0 if n_fail == 0 else 1


def main() -> None:
    results = validate_all()
    sys.exit(_render_report(results))


if __name__ == "__main__":
    main()
