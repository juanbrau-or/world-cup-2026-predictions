"""GitHub Step Summary for the operational prediction workflow."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class OperationalSummaryError(RuntimeError):
    """Raised when the operational summary cannot be written."""


@dataclass(frozen=True)
class OperationalSummaryResult:
    """Summary file write result."""

    summary_path: Path
    fixtures_received: int | None
    fixtures_tbd: int | None
    predictable_fixtures: int
    official_selected: int | None
    official_evaluated: int | None
    publication_ready: bool


def write_operational_step_summary(
    *,
    summary_path: Path | None = None,
    predictions_root: Path = Path("predictions"),
    simulations_root: Path = Path("simulations"),
    interim_root: Path = Path("data/interim"),
    publication_root: Path = Path("dist/predictions-data"),
    logs_root: Path = Path("logs"),
) -> OperationalSummaryResult:
    """Write a compact operational status summary for GitHub Actions."""

    target = summary_path or _summary_path_from_env()
    ingest_report = _read_json(interim_root / "world_cup_2026_ingest_report.json")
    scorecard = _read_json(predictions_root / "prospective_scorecard.json")
    shadow_scorecard = _read_json(predictions_root / "shadow" / "contextual_scorecard.json")
    manifest = _read_json(publication_root / "manifest.json")
    simulation_manifest = _read_json(simulations_root / "latest" / "manifest.json")
    latest_rows = _read_csv(predictions_root / "latest.csv")
    shadow_latest_rows = _read_csv(predictions_root / "shadow" / "contextual_latest.csv")
    simulation_team_rows = _read_csv(simulations_root / "latest" / "team_probabilities.csv")
    fixtures_received = _optional_int(ingest_report.get("provider_fixtures_received"))
    fixtures_tbd = _optional_int(ingest_report.get("fixtures_with_tbd_participants"))
    next_kickoff = _nested_value(ingest_report, ("freshness", "next_kickoff_utc"))
    official_selected = _optional_int(scorecard.get("official_predictions_selected"))
    official_evaluated = _optional_int(scorecard.get("official_predictions_evaluated"))
    shadow_evaluated = _optional_int(shadow_scorecard.get("official_predictions_evaluated"))
    metrics = scorecard.get("metrics") if isinstance(scorecard.get("metrics"), Mapping) else {}
    shadow_metrics = (
        shadow_scorecard.get("metrics")
        if isinstance(shadow_scorecard.get("metrics"), Mapping)
        else {}
    )
    publication_ready = bool(manifest)
    shadow_publication_ready = isinstance(manifest.get("shadow"), Mapping)
    simulation_publication_ready = isinstance(manifest.get("simulation"), Mapping)
    simulation_summary = _simulation_summary(
        manifest=simulation_manifest,
        team_rows=simulation_team_rows,
    )
    status_lines = _status_lines(
        ingest_report=ingest_report,
        scorecard=scorecard,
        logs_root=logs_root,
        predictable_fixtures=len(latest_rows),
        official_evaluated=official_evaluated,
    )

    lines = [
        "# Operational Predictions",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Fixtures received | {_value(fixtures_received)} |",
        f"| Fixtures TBD | {_value(fixtures_tbd)} |",
        f"| Predictable fixtures | {len(latest_rows)} |",
        f"| New predictions | {len(latest_rows)} |",
        f"| Shadow predictions | {len(shadow_latest_rows)} |",
        f"| Official predictions selected | {_value(official_selected)} |",
        f"| Official matches evaluable | {_value(official_evaluated)} |",
        f"| Accumulated log loss | {_metric(metrics, 'log_loss')} |",
        f"| Accumulated Brier | {_metric(metrics, 'brier_score')} |",
        f"| Accumulated RPS | {_metric(metrics, 'ranked_probability_score')} |",
        f"| Accumulated accuracy | {_metric(metrics, 'accuracy')} |",
        f"| Shadow matches evaluable | {_value(shadow_evaluated)} |",
        f"| Shadow log loss | {_metric(shadow_metrics, 'log_loss')} |",
        f"| Next kickoff UTC | {_value(next_kickoff)} |",
        f"| Publication ready | {'yes' if publication_ready else 'no'} |",
        f"| Shadow publication ready | {'yes' if shadow_publication_ready else 'no'} |",
        f"| Simulation publication ready | {'yes' if simulation_publication_ready else 'no'} |",
        "",
        "## Simulation",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Cutoff UTC | {_value(simulation_summary.get('cutoff'))} |",
        f"| Finished fixtures | {_value(simulation_summary.get('observed'))} |",
        f"| Future fixtures | {_value(simulation_summary.get('future'))} |",
        f"| TBD fixtures | {_value(simulation_summary.get('tbd'))} |",
        f"| Simulations | {_value(simulation_summary.get('runs'))} |",
        f"| Seed | {_value(simulation_summary.get('seed'))} |",
        f"| Title favorite | {_value(simulation_summary.get('favorite'))} |",
        f"| Mexico | {_value(simulation_summary.get('mexico'))} |",
        f"| Tie fallback count | {_value(simulation_summary.get('fallback_count'))} |",
        f"| Bracket status | {_value(simulation_summary.get('bracket_status'))} |",
        "",
        "### Top Champion Probabilities",
        "",
        *simulation_summary.get("top_lines", ["- n/a"]),
        "",
        "### Published Simulation Paths",
        "",
        *simulation_summary.get("published_paths", ["- n/a"]),
        "",
        "## Status",
        "",
        *status_lines,
        "",
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return OperationalSummaryResult(
        summary_path=target,
        fixtures_received=fixtures_received,
        fixtures_tbd=fixtures_tbd,
        predictable_fixtures=len(latest_rows),
        official_selected=official_selected,
        official_evaluated=official_evaluated,
        publication_ready=publication_ready,
    )


def _summary_path_from_env() -> Path:
    raw = os.environ.get("GITHUB_STEP_SUMMARY")
    if raw:
        return Path(raw)
    return Path("dist/operational_step_summary.md")


def _status_lines(
    *,
    ingest_report: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    logs_root: Path,
    predictable_fixtures: int,
    official_evaluated: int | None,
) -> list[str]:
    lines: list[str] = []
    pending = _optional_int(ingest_report.get("pending_fixtures"))
    if pending == 0 and predictable_fixtures == 0:
        lines.append("- No future fixtures are currently available.")
    if official_evaluated == 0:
        lines.append("- No newly evaluable official predictions at this results cutoff.")
    discrepancies = ingest_report.get("validation_discrepancies")
    if isinstance(discrepancies, list):
        for item in discrepancies:
            if isinstance(item, Mapping) and item.get("kind") == "secondary_provider_unavailable":
                lines.append("- Secondary provider unavailable; primary source output was used.")
                break
    if not ingest_report:
        lines.append("- Primary source did not produce an ingest report.")
    if _log_contains(logs_root, ("World Cup provider request failed", "Provider request failed")):
        lines.append("- Primary source failed; see operational logs.")
    ledger = scorecard.get("ledger") if isinstance(scorecard.get("ledger"), Mapping) else {}
    invalidity = ledger.get("invalidity_counts") if isinstance(ledger, Mapping) else {}
    if isinstance(invalidity, Mapping) and invalidity:
        lines.append("- Leakage or invalid predictions detected; workflow should fail.")
    if _log_contains(logs_root, ("invalid prospective predictions", "duplicate prediction_id")):
        lines.append("- Prediction leakage or data corruption detected in logs.")
    if _log_contains(logs_root, ("shadow-contextual", "ShadowContextualError")):
        lines.append("- Shadow contextual path reported a degraded or failed status.")
    if _log_contains(logs_root, ("Simulation failed", "TournamentSimulationError")):
        lines.append("- Tournament simulation failed; baseline prediction publication continued.")
    if not lines:
        lines.append("- Operational pipeline completed without reportable warnings.")
    return lines


def _simulation_summary(
    *,
    manifest: Mapping[str, Any],
    team_rows: list[dict[str, str]],
) -> Mapping[str, Any]:
    if not manifest:
        return {
            "bracket_status": "not available",
            "top_lines": ["- n/a"],
            "published_paths": ["- n/a"],
        }
    raw_fixture_summary = manifest.get("fixtures")
    fixture_summary = raw_fixture_summary if isinstance(raw_fixture_summary, Mapping) else {}
    observed = (
        _optional_int(fixture_summary.get("observed"))
        if isinstance(fixture_summary, Mapping)
        else None
    )
    future_known = (
        _optional_int(fixture_summary.get("future_known"))
        if isinstance(fixture_summary, Mapping)
        else None
    )
    future_tbd = (
        _optional_int(fixture_summary.get("future_tbd"))
        if isinstance(fixture_summary, Mapping)
        else None
    )
    future_partial = (
        _optional_int(fixture_summary.get("future_partially_known"))
        if isinstance(fixture_summary, Mapping)
        else None
    )
    future = _sum_optional(future_known, future_tbd, future_partial)
    top = sorted(team_rows, key=lambda row: _float(row.get("champion")), reverse=True)[:10]
    top_lines = [
        f"- {row.get('team_id', 'unknown')}: {_probability(row.get('champion'))}"
        for row in top
    ]
    if not top_lines:
        top_lines = ["- n/a"]
    favorite = top_lines[0].removeprefix("- ") if top_lines else None
    mexico = next((row for row in team_rows if row.get("team_id") == "mexico"), None)
    fallback_counts = manifest.get("fallback_counts")
    fallback_count = None
    if isinstance(fallback_counts, Mapping):
        fallback_count = sum(value for value in fallback_counts.values() if isinstance(value, int))
    outputs = manifest.get("outputs")
    published_paths: list[str] = []
    if isinstance(outputs, Mapping):
        for key in ("team_csv", "team_json", "champions", "rounds", "group_summary", "bracket"):
            value = outputs.get(key)
            if isinstance(value, str):
                published_paths.append(f"- {value}")
    return {
        "cutoff": manifest.get("data_cutoff_utc"),
        "observed": observed,
        "future": future,
        "tbd": future_tbd,
        "runs": manifest.get("runs"),
        "seed": manifest.get("seed"),
        "favorite": favorite,
        "mexico": _mexico_summary(mexico),
        "fallback_count": fallback_count,
        "bracket_status": "valid" if team_rows else "missing team probabilities",
        "top_lines": top_lines,
        "published_paths": published_paths or ["- n/a"],
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as file:
            return [dict(row) for row in csv.DictReader(file)]
    except OSError:
        return []


def _log_contains(root: Path, needles: tuple[str, ...]) -> bool:
    if not root.is_dir():
        return False
    for path in root.glob("*.log"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(needle in text for needle in needles):
            return True
    return False


def _nested_value(payload: Mapping[str, Any], keys: tuple[str, ...]) -> object:
    current: object = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _optional_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _metric(metrics: object, key: str) -> str:
    if not isinstance(metrics, Mapping):
        return "n/a"
    value = metrics.get(key)
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return "n/a"


def _sum_optional(*values: int | None) -> int | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _float(value: str | None) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _probability(value: str | None) -> str:
    return f"{_float(value):.4f}"


def _mexico_summary(row: Mapping[str, str] | None) -> str:
    if row is None:
        return "n/a"
    return (
        f"champion={_probability(row.get('champion'))}, "
        f"final={_probability(row.get('final'))}, "
        f"round_of_16={_probability(row.get('round_of_16'))}"
    )


def _value(value: object) -> str:
    if value is None:
        return "n/a"
    return str(value)
