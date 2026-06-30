"""Operational World Cup 2026 predictions from the frozen goal-model selection."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from worldcup2026.evaluation.dixon_coles_backtest import (
    DixonColesModelConfig,
    load_dixon_coles_config,
)
from worldcup2026.features.elo import EloRatingsConfig, _rate_matches, load_elo_ratings_config
from worldcup2026.models.dixon_coles import DixonColesGoalModel

PREDICTION_CONTEXT = "early_v1"
PREDICTION_STATUS = "prospective"
PREDICTION_SCHEMA_VERSION = "world_cup_prediction_v1"
OPERATIONAL_DATASET_VERSION = "operational_dataset_v1"
MODEL_FAMILY = "poisson"


class OperationalPredictionError(RuntimeError):
    """Raised when operational predictions cannot be generated safely."""


@dataclass(frozen=True)
class OperationalPredictionResult:
    """Summary returned by the upcoming-prediction pipeline."""

    predictions: tuple[dict[str, Any], ...]
    latest_csv_path: Path
    latest_parquet_path: Path
    history_path: Path
    report_path: Path
    model_family: str
    model_version: str
    half_life_days: float | None
    data_cutoff_utc: datetime
    training_matches: int
    live_finished_2026_matches: int
    dataset_revision: str
    live_snapshot_checksum: str
    excluded_fixtures: Mapping[str, int]


def run_predict_upcoming(
    *,
    model_config_path: Path = Path("configs/model.yaml"),
    modeling_matches_path: Path = Path("data/processed/modeling_matches.parquet"),
    live_matches_path: Path = Path("data/processed/world_cup_2026/matches.parquet"),
    ingest_report_path: Path = Path("data/interim/world_cup_2026_ingest_report.json"),
    predictions_root: Path = Path("predictions"),
    created_at: datetime | None = None,
) -> OperationalPredictionResult:
    """Generate traceable predictions for future known-participant World Cup fixtures."""

    prediction_created_at = _utc_now() if created_at is None else _require_utc(created_at)
    goal_config = load_dixon_coles_config(model_config_path)
    _assert_frozen_goal_selection(goal_config)
    elo_config = load_elo_ratings_config(model_config_path)

    historical_rows = _read_parquet_rows(modeling_matches_path)
    live_rows = _read_parquet_rows(live_matches_path)
    cutoff = _live_data_cutoff(live_rows, ingest_report_path=ingest_report_path)
    live_snapshot_checksum = _live_snapshot_checksum(
        live_matches_path,
        ingest_report_path=ingest_report_path,
    )
    training_rows, live_finished_rows = _operational_training_rows(
        historical_rows,
        live_rows,
        cutoff=cutoff,
    )
    dataset_checksum = _rows_checksum(training_rows)
    dataset_revision = f"{OPERATIONAL_DATASET_VERSION}:{dataset_checksum[:16]}"

    goal_model = _fit_goal_model(training_rows, goal_config=goal_config, cutoff=cutoff)
    current_elo = _current_elo_by_team(training_rows, elo_config=elo_config)
    upcoming, excluded = _upcoming_fixtures(
        live_rows,
        cutoff=cutoff,
        prediction_created_at=prediction_created_at,
    )
    config_payload = _selected_config_payload(goal_config)
    config_checksum = _json_checksum(config_payload)
    prediction_run_id = _prediction_run_id(
        created_at=prediction_created_at,
        cutoff=cutoff,
        live_snapshot_checksum=live_snapshot_checksum,
        dataset_revision=dataset_revision,
        config_checksum=config_checksum,
    )
    predictions = tuple(
        _prediction_row(
            fixture,
            goal_model=goal_model,
            elo_by_team=current_elo,
            elo_config=elo_config,
            prediction_created_at=prediction_created_at,
            data_cutoff_utc=cutoff,
            prediction_run_id=prediction_run_id,
            model_version=goal_config.model_version,
            selected_config=config_payload,
            config_checksum=config_checksum,
            dataset_revision=dataset_revision,
            dataset_checksum=dataset_checksum,
            live_snapshot_checksum=live_snapshot_checksum,
            training_matches=len(training_rows),
            live_finished_2026_matches=len(live_finished_rows),
        )
        for fixture in upcoming
    )
    _assert_prediction_probabilities(predictions)

    latest_csv = predictions_root / "latest.csv"
    latest_parquet = predictions_root / "latest.parquet"
    report_path = predictions_root / "upcoming.md"
    history_path = _history_path(
        predictions_root,
        created_at=prediction_created_at,
        predictions=predictions,
        live_snapshot_checksum=live_snapshot_checksum,
    )
    _write_prediction_outputs(
        predictions,
        latest_csv_path=latest_csv,
        latest_parquet_path=latest_parquet,
        history_path=history_path,
        report_path=report_path,
        cutoff=cutoff,
        model_config=goal_config,
        dataset_revision=dataset_revision,
        live_snapshot_checksum=live_snapshot_checksum,
        training_matches=len(training_rows),
        live_finished_2026_matches=len(live_finished_rows),
        excluded=excluded,
    )
    return OperationalPredictionResult(
        predictions=predictions,
        latest_csv_path=latest_csv,
        latest_parquet_path=latest_parquet,
        history_path=history_path,
        report_path=report_path,
        model_family=MODEL_FAMILY,
        model_version=goal_config.model_version,
        half_life_days=goal_config.time_decay_half_life_days,
        data_cutoff_utc=cutoff,
        training_matches=len(training_rows),
        live_finished_2026_matches=len(live_finished_rows),
        dataset_revision=dataset_revision,
        live_snapshot_checksum=live_snapshot_checksum,
        excluded_fixtures=excluded,
    )


def _assert_frozen_goal_selection(config: DixonColesModelConfig) -> None:
    if config.model_type != "poisson":
        msg = "operational MVP requires the frozen selected goal model to be poisson"
        raise OperationalPredictionError(msg)
    if config.time_decay_half_life_days != 730:
        msg = "operational MVP requires the frozen selected half-life of 730 days"
        raise OperationalPredictionError(msg)


def _operational_training_rows(
    historical_rows: Sequence[Mapping[str, Any]],
    live_rows: Sequence[Mapping[str, Any]],
    *,
    cutoff: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    historical = [
        dict(row)
        for row in historical_rows
        if row.get("model_eligible") is True
        and str(row.get("match_status") or "played") == "played"
        and _has_result(row)
        and _match_date(row) < cutoff.date()
    ]
    historical_keys = {_dedupe_key(row) for row in historical}
    live_finished: list[dict[str, Any]] = []
    for row in live_rows:
        if not _is_live_finished_training_row(row, cutoff=cutoff):
            continue
        prepared = _live_row_to_modeling_row(row)
        key = _dedupe_key(prepared)
        if key in historical_keys:
            continue
        historical_keys.add(key)
        live_finished.append(prepared)
    rows = [*historical, *live_finished]
    rows.sort(key=lambda row: (_match_date(row), _require_str(row, "match_id")))
    return rows, live_finished


def _is_live_finished_training_row(row: Mapping[str, Any], *, cutoff: datetime) -> bool:
    if str(row.get("match_status")) != "played":
        return False
    kickoff = _optional_datetime(row.get("kickoff_utc"))
    if kickoff is None or kickoff > cutoff:
        return False
    data_cutoff = _optional_datetime(row.get("data_cutoff_utc"))
    if data_cutoff is not None and data_cutoff > cutoff:
        return False
    return _has_result(row) and _require_str(row, "source").startswith("world_cup_2026_")


def _live_row_to_modeling_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output.update(
        {
            "model_eligible": True,
            "exclusion_reason": None,
            "competition_category": "world_cup",
            "competition_importance": 5,
            "home_advantage_eligible": False,
            "neutral_site": True,
            "home_advantage_status": "neutral",
            "modeling_period": "evaluation",
            "same_date_batch_id": _match_date(row).isoformat(),
        }
    )
    return output


def _fit_goal_model(
    training_rows: Sequence[Mapping[str, Any]],
    *,
    goal_config: DixonColesModelConfig,
    cutoff: datetime,
) -> DixonColesGoalModel:
    if not training_rows:
        msg = "operational model cannot train without eligible historical rows"
        raise OperationalPredictionError(msg)
    model = DixonColesGoalModel(
        model_type=goal_config.model_type,
        half_life_days=goal_config.time_decay_half_life_days,
        max_goals=goal_config.max_goals,
        regularization_strength=goal_config.regularization_strength,
    )
    fit_cutoff = cutoff.date() + timedelta(days=1)
    try:
        model.fit(training_rows, cutoff=fit_cutoff)
    except Exception as exc:
        msg = f"failed to fit operational goal model: {exc}"
        raise OperationalPredictionError(msg) from exc
    return model


def _current_elo_by_team(
    training_rows: Sequence[Mapping[str, Any]],
    *,
    elo_config: EloRatingsConfig,
) -> dict[str, float]:
    try:
        _, current_rows = _rate_matches(training_rows, config=elo_config)
    except Exception as exc:
        msg = f"failed to compute operational Elo diagnostics: {exc}"
        raise OperationalPredictionError(msg) from exc
    return {str(row["canonical_team_id"]): float(row["elo_rating"]) for row in current_rows}


def _upcoming_fixtures(
    live_rows: Sequence[Mapping[str, Any]],
    *,
    cutoff: datetime,
    prediction_created_at: datetime,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for row in live_rows:
        reason = _fixture_exclusion_reason(
            row,
            cutoff=cutoff,
            prediction_created_at=prediction_created_at,
        )
        if reason is not None:
            excluded[reason] += 1
            continue
        rows.append(dict(row))
    rows.sort(
        key=lambda row: (_require_datetime(row, "kickoff_utc"), _require_str(row, "match_id"))
    )
    return rows, dict(sorted(excluded.items()))


def _fixture_exclusion_reason(
    row: Mapping[str, Any],
    *,
    cutoff: datetime,
    prediction_created_at: datetime,
) -> str | None:
    status = str(row.get("match_status") or "")
    if status == "played":
        return "finished"
    if status == "in_progress":
        return "in_progress"
    if status in {"cancelled", "suspended", "postponed", "abandoned"}:
        return status
    if status != "scheduled":
        return "unsupported_status"
    kickoff = _optional_datetime(row.get("kickoff_utc"))
    if kickoff is None:
        return "missing_kickoff"
    if kickoff <= cutoff:
        return "kickoff_not_after_cutoff"
    if kickoff <= prediction_created_at:
        return "kickoff_not_after_prediction_creation"
    if (
        _optional_str(row.get("home_team_id")) is None
        or _optional_str(row.get("away_team_id")) is None
    ):
        return "unresolved_team"
    if _has_result(row):
        return "result_already_available"
    return None


def _prediction_row(
    fixture: Mapping[str, Any],
    *,
    goal_model: DixonColesGoalModel,
    elo_by_team: Mapping[str, float],
    elo_config: EloRatingsConfig,
    prediction_created_at: datetime,
    data_cutoff_utc: datetime,
    prediction_run_id: str,
    model_version: str,
    selected_config: Mapping[str, Any],
    config_checksum: str,
    dataset_revision: str,
    dataset_checksum: str,
    live_snapshot_checksum: str,
    training_matches: int,
    live_finished_2026_matches: int,
) -> dict[str, Any]:
    row = {
        "home_team_id": _require_str(fixture, "home_team_id"),
        "away_team_id": _require_str(fixture, "away_team_id"),
        "competition_category": "world_cup",
        "home_advantage_eligible": False,
    }
    distribution = goal_model.predict_match(row)
    kickoff = _require_datetime(fixture, "kickoff_utc")
    home_team_id = _require_str(fixture, "home_team_id")
    away_team_id = _require_str(fixture, "away_team_id")
    score_probabilities = json.dumps(
        distribution.score_probabilities,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "prediction_id": _stable_prediction_id(
            source_fixture_id=_require_str(fixture, "source_match_id"),
            prediction_created_at_utc=prediction_created_at,
            data_cutoff_utc=data_cutoff_utc,
            model_version=model_version,
            dataset_revision=dataset_revision,
            config_checksum=config_checksum,
        ),
        "prediction_run_id": prediction_run_id,
        "source_fixture_id": _require_str(fixture, "source_match_id"),
        "match_id": _require_str(fixture, "match_id"),
        "source": _require_str(fixture, "source"),
        "prediction_created_at_utc": prediction_created_at,
        "data_cutoff_utc": data_cutoff_utc,
        "kickoff_utc": kickoff,
        "hours_before_kickoff": (kickoff - prediction_created_at).total_seconds() / 3600,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_team_name": _require_str(fixture, "home_team_name_original"),
        "away_team_name": _require_str(fixture, "away_team_name_original"),
        "home_elo_pre": float(elo_by_team.get(home_team_id, elo_config.initial_rating)),
        "away_elo_pre": float(elo_by_team.get(away_team_id, elo_config.initial_rating)),
        "expected_home_goals": distribution.expected_home_goals,
        "expected_away_goals": distribution.expected_away_goals,
        "probability_home_win": distribution.prob_home_win,
        "probability_draw": distribution.prob_draw,
        "probability_away_win": distribution.prob_away_win,
        "modal_score": distribution.modal_score,
        "score_probabilities_json": score_probabilities,
        "score_probability_mass": distribution.score_probability_mass,
        "residual_probability": distribution.residual_probability,
        "model_family": MODEL_FAMILY,
        "model_version": model_version,
        "selected_config_json": json.dumps(
            selected_config,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "selected_config_checksum": config_checksum,
        "dataset_revision": dataset_revision,
        "dataset_checksum": dataset_checksum,
        "live_snapshot_checksum": live_snapshot_checksum,
        "prediction_context": PREDICTION_CONTEXT,
        "prediction_status": PREDICTION_STATUS,
        "competition": _require_str(fixture, "competition"),
        "stage": fixture.get("stage"),
        "training_matches": training_matches,
        "live_finished_2026_matches": live_finished_2026_matches,
    }


def _assert_prediction_probabilities(predictions: Sequence[Mapping[str, Any]]) -> None:
    for row in predictions:
        probabilities = [
            float(row["probability_home_win"]),
            float(row["probability_draw"]),
            float(row["probability_away_win"]),
        ]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities):
            msg = f"prediction probabilities are invalid for {row['source_fixture_id']}"
            raise OperationalPredictionError(msg)
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
            msg = f"prediction probabilities do not sum to 1 for {row['source_fixture_id']}"
            raise OperationalPredictionError(msg)
        created_at = _require_datetime(row, "prediction_created_at_utc")
        kickoff = _require_datetime(row, "kickoff_utc")
        cutoff = _require_datetime(row, "data_cutoff_utc")
        if row["prediction_status"] == PREDICTION_STATUS:
            if not created_at < kickoff:
                raise OperationalPredictionError("prospective prediction created after kickoff")
            if cutoff > created_at:
                raise OperationalPredictionError("prediction cutoff is after prediction creation")


def _write_prediction_outputs(
    predictions: Sequence[Mapping[str, Any]],
    *,
    latest_csv_path: Path,
    latest_parquet_path: Path,
    history_path: Path,
    report_path: Path,
    cutoff: datetime,
    model_config: DixonColesModelConfig,
    dataset_revision: str,
    live_snapshot_checksum: str,
    training_matches: int,
    live_finished_2026_matches: int,
    excluded: Mapping[str, int],
) -> None:
    _write_predictions_parquet(predictions, latest_parquet_path)
    _write_predictions_csv(predictions, latest_csv_path)
    _write_immutable_predictions_parquet(predictions, history_path)
    _write_upcoming_report(
        predictions,
        report_path,
        cutoff=cutoff,
        model_config=model_config,
        dataset_revision=dataset_revision,
        live_snapshot_checksum=live_snapshot_checksum,
        training_matches=training_matches,
        live_finished_2026_matches=live_finished_2026_matches,
        excluded=excluded,
        history_path=history_path,
        latest_csv_path=latest_csv_path,
        latest_parquet_path=latest_parquet_path,
    )


def _write_predictions_parquet(predictions: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in predictions]
    table = pa.Table.from_pylist(rows, schema=_prediction_schema())
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


def _write_immutable_predictions_parquet(
    predictions: Sequence[Mapping[str, Any]], path: Path
) -> None:
    if path.exists():
        existing = path.read_bytes()
        table = pa.Table.from_pylist(
            [dict(row) for row in predictions],
            schema=_prediction_schema(),
        )
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink)  # type: ignore[no-untyped-call]
        if existing != sink.getvalue().to_pybytes():
            raise OperationalPredictionError(f"immutable prediction collision at {path}")
        return
    _write_predictions_parquet(predictions, path)


def _write_predictions_csv(predictions: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _prediction_csv_fields()
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in predictions:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _write_upcoming_report(
    predictions: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    cutoff: datetime,
    model_config: DixonColesModelConfig,
    dataset_revision: str,
    live_snapshot_checksum: str,
    training_matches: int,
    live_finished_2026_matches: int,
    excluded: Mapping[str, int],
    history_path: Path,
    latest_csv_path: Path,
    latest_parquet_path: Path,
) -> None:
    lines = [
        "# Upcoming World Cup 2026 Predictions",
        "",
        f"Data cutoff UTC: {cutoff.isoformat()}",
        f"Model: {model_config.model_type} ({model_config.model_version})",
        f"Half-life days: {model_config.time_decay_half_life_days}",
        f"Training matches: {training_matches}",
        f"World Cup 2026 finished matches incorporated: {live_finished_2026_matches}",
        f"Dataset revision: {dataset_revision}",
        f"Live snapshot checksum: {live_snapshot_checksum}",
        "",
        "## Predictions",
        "",
        _markdown_predictions(predictions),
        "",
        "## Exclusions",
        "",
        _markdown_counts(excluded),
        "",
        "## Files",
        "",
        f"- Latest CSV: `{latest_csv_path}`",
        f"- Latest Parquet: `{latest_parquet_path}`",
        f"- Historical snapshot: `{history_path}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_predictions(predictions: Sequence[Mapping[str, Any]]) -> str:
    if not predictions:
        return "No eligible future known-participant fixtures at this cutoff."
    fields = (
        "kickoff_utc",
        "home_team_name",
        "away_team_name",
        "probability_home_win",
        "probability_draw",
        "probability_away_win",
        "expected_home_goals",
        "expected_away_goals",
        "modal_score",
    )
    lines = [
        "| Kickoff UTC | Home | Away | P(home) | P(draw) | P(away) | xG home | xG away | Modal |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in predictions:
        values = [_markdown_value(row[field]) for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _markdown_counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return "No excluded canonical fixtures."
    lines = ["| Reason | Fixtures |", "| --- | ---: |"]
    for key, value in sorted(counts.items()):
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def _markdown_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _history_path(
    predictions_root: Path,
    *,
    created_at: datetime,
    predictions: Sequence[Mapping[str, Any]],
    live_snapshot_checksum: str,
) -> Path:
    payload_checksum = _rows_checksum(predictions) if predictions else live_snapshot_checksum
    token = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    return predictions_root / "history" / f"{token}_{payload_checksum[:12]}.parquet"


def _prediction_schema() -> pa.Schema:
    return pa.schema(
        [
            ("schema_version", pa.string()),
            ("prediction_id", pa.string()),
            ("prediction_run_id", pa.string()),
            ("source_fixture_id", pa.string()),
            ("match_id", pa.string()),
            ("source", pa.string()),
            ("prediction_created_at_utc", pa.timestamp("us", tz="UTC")),
            ("data_cutoff_utc", pa.timestamp("us", tz="UTC")),
            ("kickoff_utc", pa.timestamp("us", tz="UTC")),
            ("hours_before_kickoff", pa.float64()),
            ("home_team_id", pa.string()),
            ("away_team_id", pa.string()),
            ("home_team_name", pa.string()),
            ("away_team_name", pa.string()),
            ("home_elo_pre", pa.float64()),
            ("away_elo_pre", pa.float64()),
            ("expected_home_goals", pa.float64()),
            ("expected_away_goals", pa.float64()),
            ("probability_home_win", pa.float64()),
            ("probability_draw", pa.float64()),
            ("probability_away_win", pa.float64()),
            ("modal_score", pa.string()),
            ("score_probabilities_json", pa.string()),
            ("score_probability_mass", pa.float64()),
            ("residual_probability", pa.float64()),
            ("model_family", pa.string()),
            ("model_version", pa.string()),
            ("selected_config_json", pa.string()),
            ("selected_config_checksum", pa.string()),
            ("dataset_revision", pa.string()),
            ("dataset_checksum", pa.string()),
            ("live_snapshot_checksum", pa.string()),
            ("prediction_context", pa.string()),
            ("prediction_status", pa.string()),
            ("competition", pa.string()),
            ("stage", pa.string()),
            ("training_matches", pa.int64()),
            ("live_finished_2026_matches", pa.int64()),
        ]
    )


def _prediction_csv_fields() -> list[str]:
    return [
        "schema_version",
        "prediction_id",
        "prediction_run_id",
        "source_fixture_id",
        "match_id",
        "source",
        "prediction_created_at_utc",
        "data_cutoff_utc",
        "kickoff_utc",
        "hours_before_kickoff",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
        "home_elo_pre",
        "away_elo_pre",
        "expected_home_goals",
        "expected_away_goals",
        "probability_home_win",
        "probability_draw",
        "probability_away_win",
        "modal_score",
        "score_probabilities_json",
        "score_probability_mass",
        "residual_probability",
        "model_family",
        "model_version",
        "selected_config_json",
        "selected_config_checksum",
        "dataset_revision",
        "dataset_checksum",
        "live_snapshot_checksum",
        "prediction_context",
        "prediction_status",
        "competition",
        "stage",
        "training_matches",
        "live_finished_2026_matches",
    ]


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise OperationalPredictionError(f"required Parquet input is missing: {path}")
    try:
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
    except (OSError, pa.ArrowInvalid) as exc:
        msg = f"failed to read Parquet input {path}: {exc}"
        raise OperationalPredictionError(msg) from exc
    return [dict(row) for row in table.to_pylist()]


def _live_data_cutoff(
    live_rows: Sequence[Mapping[str, Any]],
    *,
    ingest_report_path: Path,
) -> datetime:
    report_cutoff = _cutoff_from_report(ingest_report_path)
    row_cutoffs = [
        value
        for value in (_optional_datetime(row.get("data_cutoff_utc")) for row in live_rows)
        if value is not None
    ]
    if row_cutoffs:
        return max(row_cutoffs)
    if report_cutoff is not None:
        return report_cutoff
    msg = "cannot determine live data_cutoff_utc; run `uv run wc2026 ingest world-cup` first"
    raise OperationalPredictionError(msg)


def _cutoff_from_report(path: Path) -> datetime | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    raw = payload.get("data_cutoff_utc") or payload.get("fetched_at")
    return _optional_datetime(raw)


def _live_snapshot_checksum(live_matches_path: Path, *, ingest_report_path: Path) -> str:
    report_checksum = _snapshot_checksum_from_report(ingest_report_path)
    if report_checksum is not None:
        return report_checksum
    if live_matches_path.is_file():
        return hashlib.sha256(live_matches_path.read_bytes()).hexdigest()
    raise OperationalPredictionError(f"live matches input is missing: {live_matches_path}")


def _snapshot_checksum_from_report(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, Mapping) and isinstance(payload.get("snapshot_checksum"), str):
        return str(payload["snapshot_checksum"])
    manifests = payload.get("snapshot_manifests") if isinstance(payload, Mapping) else None
    if isinstance(manifests, list):
        for manifest in manifests:
            if (
                isinstance(manifest, Mapping)
                and manifest.get("source_fixture_id") == "__collection__"
                and isinstance(manifest.get("checksum"), str)
            ):
                return str(manifest["checksum"])
    return None


def _selected_config_payload(config: DixonColesModelConfig) -> dict[str, Any]:
    return {
        "model_type": config.model_type,
        "time_decay_half_life_days": config.time_decay_half_life_days,
        "regularization_strength": config.regularization_strength,
        "max_goals": config.max_goals,
        "selection_source": "configs/model.yaml:dixon_coles",
    }


def _prediction_run_id(
    *,
    created_at: datetime,
    cutoff: datetime,
    live_snapshot_checksum: str,
    dataset_revision: str,
    config_checksum: str,
) -> str:
    payload = "|".join(
        (
            created_at.isoformat(),
            cutoff.isoformat(),
            live_snapshot_checksum,
            dataset_revision,
            config_checksum,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _stable_prediction_id(
    *,
    source_fixture_id: str,
    prediction_created_at_utc: datetime,
    data_cutoff_utc: datetime,
    model_version: str,
    dataset_revision: str,
    config_checksum: str,
) -> str:
    payload = "|".join(
        (
            source_fixture_id,
            prediction_created_at_utc.isoformat(),
            data_cutoff_utc.isoformat(),
            model_version,
            dataset_revision,
            config_checksum,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _json_checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_value)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _rows_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        [dict(row) for row in rows],
        sort_keys=True,
        separators=(",", ":"),
        default=_json_value,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _dedupe_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        _match_date(row).isoformat(),
        _require_str(row, "home_team_id"),
        _require_str(row, "away_team_id"),
        _require_str(row, "competition"),
    )


def _has_result(row: Mapping[str, Any]) -> bool:
    home = row.get("home_goals_90")
    away = row.get("away_goals_90")
    return _is_int(home) and _is_int(away)


def _is_int(value: object) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool)


def _match_date(row: Mapping[str, Any]) -> date:
    value = row.get("match_date")
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value)
    msg = f"expected match_date for match {row.get('match_id')!r}, got {value!r}"
    raise OperationalPredictionError(msg)


def _require_str(row: Mapping[str, Any], field_name: str) -> str:
    value = row.get(field_name)
    if isinstance(value, str) and value:
        return value
    msg = f"required field {field_name} is missing or blank for match {row.get('match_id')!r}"
    raise OperationalPredictionError(msg)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _require_datetime(row: Mapping[str, Any], field_name: str) -> datetime:
    value = _optional_datetime(row.get(field_name))
    if value is None:
        msg = f"required datetime field {field_name} is missing for match {row.get('match_id')!r}"
        raise OperationalPredictionError(msg)
    return value


def _optional_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return _require_utc(value)
    if isinstance(value, str) and value:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return _require_utc(parsed)
    return None


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise OperationalPredictionError("datetime values must be timezone-aware UTC")
    return value.astimezone(UTC)


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _csv_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        return f"{value:.10f}"
    return value


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
