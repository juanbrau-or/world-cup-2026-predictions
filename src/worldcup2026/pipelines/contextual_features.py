"""Pipeline helpers for contextual feature artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from worldcup2026.features.contextual import (
    ContextualFeatureError,
    build_contextual_features,
    validate_contextual_feature_rows,
    write_contextual_feature_outputs,
)


class ContextualFeaturePipelineError(RuntimeError):
    """Raised when contextual feature preparation or auditing fails."""


@dataclass(frozen=True)
class ContextualFeaturePipelineResult:
    """Written contextual feature artifacts."""

    team_fixture_rows: int
    match_rows: int
    team_fixture_parquet: Path
    match_parquet: Path
    manifest_path: Path
    coverage_json_path: Path
    coverage_markdown_path: Path
    missing_data_report_path: Path
    leakage_audit_path: Path
    descriptive_report_path: Path
    leakage_audit_passed: bool


@dataclass(frozen=True)
class ContextualFeatureAuditResult:
    """Audit result for existing contextual feature artifacts."""

    team_fixture_rows: int
    match_rows: int
    leakage_audit_passed: bool
    violations: int


def run_contextual_feature_pipeline(
    *,
    historical_matches_path: Path | None = Path("data/processed/international_matches.parquet"),
    live_matches_path: Path | None = Path("data/processed/world_cup_2026/matches.parquet"),
    venue_catalog_path: Path = Path("data/static/venues.csv"),
    output_root: Path = Path("data/processed/contextual_features"),
    interim_root: Path = Path("data/interim/contextual_features"),
    feature_generated_at_utc: datetime | None = None,
    data_cutoff_utc: datetime | None = None,
    include_historical: bool = True,
    include_live: bool = True,
) -> ContextualFeaturePipelineResult:
    """Build and write contextual feature datasets and reports."""

    generated_at = _utc_now() if feature_generated_at_utc is None else feature_generated_at_utc
    try:
        result = build_contextual_features(
            historical_matches_path=historical_matches_path,
            live_matches_path=live_matches_path,
            venue_catalog_path=venue_catalog_path,
            feature_generated_at_utc=generated_at,
            data_cutoff_utc=data_cutoff_utc,
            include_historical=include_historical,
            include_live=include_live,
        )
        paths = write_contextual_feature_outputs(
            result,
            output_root=output_root,
            interim_root=interim_root,
        )
    except ContextualFeatureError as exc:
        raise ContextualFeaturePipelineError(str(exc)) from exc
    audit_passed = bool(result.leakage_audit.get("passed"))
    if not audit_passed:
        raise ContextualFeaturePipelineError(
            f"contextual feature leakage audit failed: {paths['leakage_audit']}"
        )
    return ContextualFeaturePipelineResult(
        team_fixture_rows=len(result.team_rows),
        match_rows=len(result.match_rows),
        team_fixture_parquet=paths["team_fixture_parquet"],
        match_parquet=paths["match_parquet"],
        manifest_path=paths["manifest"],
        coverage_json_path=paths["coverage_json"],
        coverage_markdown_path=paths["coverage_markdown"],
        missing_data_report_path=paths["missing_data_report"],
        leakage_audit_path=paths["leakage_audit"],
        descriptive_report_path=paths["descriptive_report"],
        leakage_audit_passed=audit_passed,
    )


def audit_contextual_feature_outputs(
    *,
    team_fixture_path: Path = Path(
        "data/processed/contextual_features/team_fixture_contextual_features.parquet"
    ),
    match_path: Path = Path("data/processed/contextual_features/match_contextual_features.parquet"),
) -> ContextualFeatureAuditResult:
    """Audit existing contextual feature Parquet outputs for leakage and corrupt schemas."""

    try:
        team_rows = _read_parquet_rows(team_fixture_path)
        match_rows = _read_parquet_rows(match_path)
        audit = validate_contextual_feature_rows(team_rows, match_rows)
    except ContextualFeatureError as exc:
        raise ContextualFeaturePipelineError(str(exc)) from exc
    if not audit.passed:
        raise ContextualFeaturePipelineError(
            f"contextual feature audit failed with {len(audit.violations)} violation(s)"
        )
    return ContextualFeatureAuditResult(
        team_fixture_rows=len(team_rows),
        match_rows=len(match_rows),
        leakage_audit_passed=audit.passed,
        violations=len(audit.violations),
    )


def contextual_feature_report_summary(
    *,
    coverage_report_path: Path = Path(
        "data/interim/contextual_features/contextual_features_coverage.json"
    ),
    leakage_audit_path: Path = Path(
        "data/interim/contextual_features/contextual_features_leakage_audit.json"
    ),
) -> dict[str, Any]:
    """Read small JSON reports for CLI display."""

    try:
        coverage = _read_json(coverage_report_path)
        leakage = _read_json(leakage_audit_path)
    except OSError as exc:
        raise ContextualFeaturePipelineError(str(exc)) from exc
    return {
        "rows_total": coverage.get("rows_total", 0),
        "feature_classification": coverage.get("feature_classification", {}),
        "leakage_audit_passed": leakage.get("passed", False),
        "leakage_violations": len(leakage.get("violations", []))
        if isinstance(leakage.get("violations"), list)
        else 0,
    }


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ContextualFeatureError(f"contextual feature Parquet is missing: {path}")
    try:
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
    except (OSError, pa.ArrowInvalid) as exc:
        msg = f"failed to read contextual feature Parquet {path}: {exc}"
        raise ContextualFeatureError(msg) from exc
    return [dict(row) for row in table.to_pylist()]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OSError(f"report must contain a JSON object: {path}")
    return payload


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)
