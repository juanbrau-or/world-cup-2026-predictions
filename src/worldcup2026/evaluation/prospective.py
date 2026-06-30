"""Prospective ledger and scorecard for operational World Cup predictions."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml  # type: ignore[import-untyped]

EPSILON = 1e-15
HOME_CLASS = 0
DRAW_CLASS = 1
AWAY_CLASS = 2
CLASS_LABELS = (HOME_CLASS, DRAW_CLASS, AWAY_CLASS)
CLASS_NAMES = ("home_win", "draw", "away_win")
RESULT_METRIC_BASIS = "result_90"
LEDGER_SCHEMA_VERSION = "prediction_ledger_v1"
SCORECARD_SCHEMA_VERSION = "prospective_scorecard_v1"
DEFAULT_CONFIG_PATH = Path("configs/prospective_evaluation.yaml")
DEFAULT_RESULTS_CUTOFF_PATH = Path("data/interim/world_cup_2026_ingest_report.json")
LEGACY_ID_REPAIR_REASON = "legacy_prediction_id_omitted_created_at"


class ProspectiveEvaluationError(RuntimeError):
    """Raised when saved prospective predictions cannot be evaluated safely."""


@dataclass(frozen=True)
class HorizonBucket:
    """Configured prediction lead-time bucket."""

    bucket_id: str
    label: str
    min_hours: float | None
    min_inclusive: bool
    max_hours: float | None
    max_inclusive: bool

    def contains(self, value: float) -> bool:
        """Return whether ``value`` belongs to this bucket."""

        if self.min_hours is not None:
            if self.min_inclusive and value < self.min_hours:
                return False
            if not self.min_inclusive and value <= self.min_hours:
                return False
        if self.max_hours is not None:
            if self.max_inclusive and value > self.max_hours:
                return False
            if not self.max_inclusive and value >= self.max_hours:
                return False
        return True


@dataclass(frozen=True)
class OfficialSelectionConfig:
    """Versioned deterministic policy for one official prediction per fixture."""

    policy_id: str
    policy_version: str
    prediction_context: str
    min_hours_before_kickoff: float
    primary_rule_id: str
    fallback_rule_id: str


@dataclass(frozen=True)
class HistoricalFrequencyBaselineConfig:
    """Frozen historical baseline configuration."""

    enabled: bool
    input_matches_path: Path
    cutoff_utc: datetime


@dataclass(frozen=True)
class BaselinesConfig:
    """Baseline model configuration."""

    uniform_enabled: bool
    historical_frequency: HistoricalFrequencyBaselineConfig
    elo_enabled: bool


@dataclass(frozen=True)
class ProspectiveEvaluationConfig:
    """Configuration for prospective ledger, policy and scorecard generation."""

    schema_version: str
    result_metric_basis: str
    minimum_calibration_matches: int
    small_sample_warning_threshold: int
    horizon_version: str
    horizon_buckets: tuple[HorizonBucket, ...]
    official_selection: OfficialSelectionConfig
    baselines: BaselinesConfig


@dataclass(frozen=True)
class ProspectiveEvaluationResult:
    """Summary returned by the prospective evaluation pipeline."""

    evaluable_predictions: int
    log_loss: float | None
    brier_score: float | None
    ranked_probability_score: float | None
    accuracy: float | None
    kickoff_range: tuple[str | None, str | None]
    average_hours_before_kickoff: float | None
    report_path: Path
    json_path: Path
    matches_path: Path
    ledger_path: Path
    ledger_predictions: int
    official_predictions_selected: int
    results_cutoff_utc: str | None


def load_prospective_evaluation_config(
    path: Path = DEFAULT_CONFIG_PATH,
) -> ProspectiveEvaluationConfig:
    """Load the versioned prospective evaluation configuration."""

    if not path.is_file():
        raise ProspectiveEvaluationError(f"prospective evaluation config is missing: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProspectiveEvaluationError("prospective evaluation config must contain an object")

    buckets_payload = _require_mapping(payload, "horizons")
    bucket_rows = _require_sequence(buckets_payload, "buckets")
    buckets = tuple(_parse_horizon_bucket(item) for item in bucket_rows)
    if not buckets:
        raise ProspectiveEvaluationError("at least one horizon bucket is required")

    policy_payload = _require_mapping(payload, "official_selection")
    primary = _require_mapping(policy_payload, "primary_rule")
    fallback = _require_mapping(policy_payload, "fallback_rule")
    baselines_payload = _require_mapping(payload, "baselines")
    historical_payload = _require_mapping(baselines_payload, "historical_frequency")

    return ProspectiveEvaluationConfig(
        schema_version=_require_str(payload, "schema_version"),
        result_metric_basis=_require_str(payload, "result_metric_basis"),
        minimum_calibration_matches=_require_int(payload, "minimum_calibration_matches"),
        small_sample_warning_threshold=_require_int(payload, "small_sample_warning_threshold"),
        horizon_version=_require_str(buckets_payload, "version"),
        horizon_buckets=buckets,
        official_selection=OfficialSelectionConfig(
            policy_id=_require_str(policy_payload, "policy_id"),
            policy_version=_require_str(policy_payload, "policy_version"),
            prediction_context=_require_str(policy_payload, "prediction_context"),
            min_hours_before_kickoff=_require_float(primary, "min_hours_before_kickoff"),
            primary_rule_id=_require_str(primary, "id"),
            fallback_rule_id=_require_str(fallback, "id"),
        ),
        baselines=BaselinesConfig(
            uniform_enabled=_require_bool(
                _require_mapping(baselines_payload, "uniform_1x2"), "enabled"
            ),
            historical_frequency=HistoricalFrequencyBaselineConfig(
                enabled=_require_bool(historical_payload, "enabled"),
                input_matches_path=Path(_require_str(historical_payload, "input_matches_path")),
                cutoff_utc=_parse_utc(_require_str(historical_payload, "cutoff_utc")),
            ),
            elo_enabled=_require_bool(
                _require_mapping(baselines_payload, "elo_operational"), "enabled"
            ),
        ),
    )


def run_prospective_evaluation(
    *,
    predictions_history_root: Path = Path("predictions/history"),
    live_matches_path: Path = Path("data/processed/world_cup_2026/matches.parquet"),
    report_path: Path = Path("predictions/prospective_scorecard.md"),
    json_path: Path = Path("predictions/prospective_scorecard.json"),
    matches_csv_path: Path = Path("predictions/prospective_matches.csv"),
    ledger_path: Path = Path("predictions/prediction_ledger.parquet"),
    config_path: Path = DEFAULT_CONFIG_PATH,
    results_cutoff_path: Path = DEFAULT_RESULTS_CUTOFF_PATH,
    generated_at: datetime | None = None,
) -> ProspectiveEvaluationResult:
    """Build the prediction ledger and evaluate the official prospective policy."""

    config = load_prospective_evaluation_config(config_path)
    if config.result_metric_basis != RESULT_METRIC_BASIS:
        raise ProspectiveEvaluationError("only result_90 is currently supported for 1X2 metrics")
    created_at = _utc_now() if generated_at is None else _require_utc(generated_at)
    prediction_rows = _read_prediction_history(predictions_history_root)
    existing_ledger_rows = _read_existing_ledger(ledger_path)
    ledger_rows = build_prediction_ledger_rows(
        prediction_rows,
        existing_ledger_rows=existing_ledger_rows,
        config=config,
    )
    _write_ledger(ledger_rows, ledger_path)

    live_rows = _read_live_matches(live_matches_path)
    result_contracts = _canonical_result_contracts(live_rows)
    results_cutoff = _results_cutoff(live_rows, results_cutoff_path)
    official_candidates = select_official_predictions(ledger_rows, config=config)
    official_rows = _join_official_predictions_with_results(
        official_candidates,
        result_contracts=result_contracts,
    )
    descriptive_rows = _join_all_valid_predictions_with_results(
        ledger_rows,
        result_contracts=result_contracts,
        prediction_context=config.official_selection.prediction_context,
    )
    official_evaluable = [row for row in official_rows if row["evaluation_status"] == "evaluated"]
    descriptive_evaluable = [
        row for row in descriptive_rows if row["evaluation_status"] == "evaluated"
    ]
    baseline_probabilities = _baseline_probabilities(config, official_evaluable)
    official_metrics = metrics_for_rows(
        official_evaluable,
        minimum_calibration_matches=config.minimum_calibration_matches,
    )
    scorecard = _scorecard_payload(
        config=config,
        generated_at=created_at,
        ledger_rows=ledger_rows,
        official_rows=official_rows,
        official_evaluable=official_evaluable,
        descriptive_evaluable=descriptive_evaluable,
        official_metrics=official_metrics,
        baseline_probabilities=baseline_probabilities,
        results_cutoff=results_cutoff,
    )
    _write_scorecard_json(scorecard, json_path)
    _write_scorecard_markdown(scorecard, report_path)
    _write_matches_csv(official_evaluable, matches_csv_path, baseline_probabilities)

    return ProspectiveEvaluationResult(
        evaluable_predictions=int(official_metrics["matches"]),
        log_loss=_optional_float(official_metrics["log_loss"]),
        brier_score=_optional_float(official_metrics["brier_score"]),
        ranked_probability_score=_optional_float(official_metrics["ranked_probability_score"]),
        accuracy=_optional_float(official_metrics["accuracy"]),
        kickoff_range=(
            _optional_str(official_metrics["kickoff_start"]),
            _optional_str(official_metrics["kickoff_end"]),
        ),
        average_hours_before_kickoff=_optional_float(
            official_metrics["average_hours_before_kickoff"]
        ),
        report_path=report_path,
        json_path=json_path,
        matches_path=matches_csv_path,
        ledger_path=ledger_path,
        ledger_predictions=len(ledger_rows),
        official_predictions_selected=len(official_rows),
        results_cutoff_utc=_format_utc(results_cutoff) if results_cutoff is not None else None,
    )


def build_prediction_ledger_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    existing_ledger_rows: Sequence[Mapping[str, Any]] = (),
    config: ProspectiveEvaluationConfig,
) -> list[dict[str, Any]]:
    """Validate and merge source prediction rows into an append-only ledger view."""

    collision_repair_ids = _legacy_prediction_ids_requiring_repair(prediction_rows)
    source_ledger_rows = [
        _ledger_row_from_prediction(row, config=config, collision_repair_ids=collision_repair_ids)
        for row in prediction_rows
    ]
    existing_rows = [_normalize_existing_ledger_row(row) for row in existing_ledger_rows]
    merged = _merge_ledger_rows([*existing_rows, *source_ledger_rows])
    invalid = [row for row in merged if row["prospective_validity_status"] != "valid"]
    if invalid:
        reasons = Counter(str(row["invalidity_reason"]) for row in invalid)
        detail = ", ".join(f"{key}={value}" for key, value in sorted(reasons.items()))
        raise ProspectiveEvaluationError(f"invalid prospective predictions detected: {detail}")
    return merged


def select_official_predictions(
    ledger_rows: Sequence[Mapping[str, Any]],
    *,
    config: ProspectiveEvaluationConfig,
) -> list[dict[str, Any]]:
    """Select one official prediction per fixture/context without looking at results."""

    policy = config.official_selection
    by_fixture: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in ledger_rows:
        if row.get("prospective_validity_status") != "valid":
            continue
        if row.get("prediction_context") != policy.prediction_context:
            continue
        by_fixture[(str(row["source_fixture_id"]), str(row["prediction_context"]))].append(row)

    selected: list[dict[str, Any]] = []
    for (fixture_id, context), rows in sorted(by_fixture.items()):
        primary_candidates = [
            row
            for row in rows
            if float(row["hours_before_kickoff"]) >= policy.min_hours_before_kickoff
        ]
        if primary_candidates:
            chosen = max(
                primary_candidates,
                key=lambda row: (
                    _require_utc_datetime(row["prediction_created_at_utc"]),
                    str(row["prediction_id"]),
                ),
            )
            rule = policy.primary_rule_id
        else:
            chosen = min(
                rows,
                key=lambda row: (
                    _require_utc_datetime(row["prediction_created_at_utc"]),
                    str(row["prediction_id"]),
                ),
            )
            rule = policy.fallback_rule_id
        output = dict(chosen)
        output["official_policy_id"] = policy.policy_id
        output["official_policy_version"] = policy.policy_version
        output["official_selection_rule"] = rule
        output["official_selection_context"] = context
        output["official_selection_fixture_id"] = fixture_id
        selected.append(output)
    selected.sort(
        key=lambda row: (
            _require_utc_datetime(row["kickoff_utc"]),
            str(row["source_fixture_id"]),
            str(row["prediction_id"]),
        )
    )
    return selected


def metrics_for_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_calibration_matches: int,
) -> dict[str, Any]:
    """Compute probabilistic multiclass metrics for evaluated rows."""

    if not rows:
        return _empty_metrics(minimum_calibration_matches=minimum_calibration_matches)

    probabilities = _probability_matrix(rows)
    targets = np.asarray([_class_from_name(str(row["metric_result_1x2"])) for row in rows])
    true_probabilities = probabilities[np.arange(len(targets)), targets]
    one_hot = np.eye(len(CLASS_LABELS))[targets]
    kickoffs = [_format_utc(_require_utc_datetime(row["kickoff_utc"])) for row in rows]
    hours = np.asarray([float(row["hours_before_kickoff"]) for row in rows], dtype=float)
    calibration_error = (
        _calibration_error(probabilities, targets, bins=10)
        if len(rows) >= minimum_calibration_matches
        else None
    )
    return {
        "matches": len(rows),
        "log_loss": float(-np.mean(np.log(np.clip(true_probabilities, EPSILON, 1.0)))),
        "brier_score": float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1))),
        "ranked_probability_score": _ranked_probability_score(probabilities, targets),
        "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == targets)),
        "calibration_error": calibration_error,
        "calibration_status": "computed"
        if calibration_error is not None
        else "insufficient_sample",
        "calibration_minimum_matches": minimum_calibration_matches,
        "kickoff_start": min(kickoffs),
        "kickoff_end": max(kickoffs),
        "average_hours_before_kickoff": float(np.mean(hours)),
        "median_hours_before_kickoff": float(np.median(hours)),
    }


def _read_prediction_history(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.parquet")):
        try:
            table = pq.read_table(path)  # type: ignore[no-untyped-call]
        except (OSError, pa.ArrowInvalid) as exc:
            msg = f"failed to read prediction history {path}: {exc}"
            raise ProspectiveEvaluationError(msg) from exc
        snapshot_checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in table.to_pylist():
            item = dict(row)
            item["_history_path"] = path.as_posix()
            item["_prediction_snapshot_checksum"] = snapshot_checksum
            rows.append(item)
    return rows


def _read_existing_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
    except (OSError, pa.ArrowInvalid) as exc:
        msg = f"failed to read prediction ledger {path}: {exc}"
        raise ProspectiveEvaluationError(msg) from exc
    return [dict(row) for row in table.to_pylist()]


def _read_live_matches(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
    except (OSError, pa.ArrowInvalid) as exc:
        msg = f"failed to read live matches {path}: {exc}"
        raise ProspectiveEvaluationError(msg) from exc
    return [dict(row) for row in table.to_pylist()]


def _legacy_prediction_ids_requiring_repair(
    prediction_rows: Sequence[Mapping[str, Any]],
) -> set[str]:
    by_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        prediction_id = _optional_nonempty_str(row.get("prediction_id"))
        if prediction_id is None:
            continue
        by_id[prediction_id].append(row)

    repair_ids: set[str] = set()
    for prediction_id, rows in by_id.items():
        normalized = {_row_fingerprint(row, exclude_source_metadata=True) for row in rows}
        if len(normalized) <= 1:
            continue
        if _is_repairable_legacy_id_collision(rows):
            repair_ids.add(prediction_id)
            continue
        raise ProspectiveEvaluationError(
            f"duplicate prediction_id with different content: {prediction_id}"
        )
    return repair_ids


def _is_repairable_legacy_id_collision(rows: Sequence[Mapping[str, Any]]) -> bool:
    stable_fields = (
        "source_fixture_id",
        "match_id",
        "source",
        "data_cutoff_utc",
        "kickoff_utc",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
        "expected_home_goals",
        "expected_away_goals",
        "probability_home_win",
        "probability_draw",
        "probability_away_win",
        "modal_score",
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
    )
    fingerprints = {
        json.dumps({field: _json_value(row.get(field)) for field in stable_fields}, sort_keys=True)
        for row in rows
    }
    created_values = {_json_value(row.get("prediction_created_at_utc")) for row in rows}
    return len(fingerprints) == 1 and len(created_values) == len(rows)


def _ledger_row_from_prediction(
    row: Mapping[str, Any],
    *,
    config: ProspectiveEvaluationConfig,
    collision_repair_ids: set[str],
) -> dict[str, Any]:
    source_prediction_id = _require_nonempty_str(row, "prediction_id")
    repair_reason = (
        LEGACY_ID_REPAIR_REASON if source_prediction_id in collision_repair_ids else None
    )
    prediction_created_at = _require_utc_datetime(row.get("prediction_created_at_utc"))
    data_cutoff = _require_utc_datetime(row.get("data_cutoff_utc"))
    kickoff = _require_utc_datetime(row.get("kickoff_utc"))
    hours_before_kickoff = (kickoff - prediction_created_at).total_seconds() / 3600
    probabilities = (
        _require_probability(row, "probability_home_win"),
        _require_probability(row, "probability_draw"),
        _require_probability(row, "probability_away_win"),
    )
    model_name = _require_nonempty_str(row, "model_family")
    model_version = _require_nonempty_str(row, "model_version")
    source_snapshot_checksum = _require_source_snapshot_checksum(row)
    dataset_revision = _require_nonempty_str(row, "dataset_revision")
    model_configuration = _optional_nonempty_str(row.get("selected_config_json"))
    if model_configuration is None:
        raise ProspectiveEvaluationError("model configuration is required in prediction ledger")
    prediction_id = (
        _derived_repaired_prediction_id(row, source_prediction_id=source_prediction_id)
        if repair_reason is not None
        else source_prediction_id
    )
    validity_status, invalidity_reason = _prospective_validity(
        row,
        prediction_created_at=prediction_created_at,
        data_cutoff=data_cutoff,
        kickoff=kickoff,
        probabilities=probabilities,
        model_version=model_version,
        source_snapshot_checksum=source_snapshot_checksum,
    )
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "prediction_id": prediction_id,
        "source_prediction_id": source_prediction_id,
        "prediction_id_repair_reason": repair_reason,
        "prediction_run_id": _optional_nonempty_str(row.get("prediction_run_id")),
        "source_fixture_id": _require_nonempty_str(row, "source_fixture_id"),
        "match_id": _optional_nonempty_str(row.get("match_id")),
        "source": _optional_nonempty_str(row.get("source")),
        "prediction_created_at_utc": prediction_created_at,
        "data_cutoff_utc": data_cutoff,
        "kickoff_utc": kickoff,
        "hours_before_kickoff": hours_before_kickoff,
        "horizon_bucket": _horizon_bucket(hours_before_kickoff, config.horizon_buckets),
        "home_team_id": _require_nonempty_str(row, "home_team_id"),
        "away_team_id": _require_nonempty_str(row, "away_team_id"),
        "home_team_name": _require_nonempty_str(row, "home_team_name"),
        "away_team_name": _require_nonempty_str(row, "away_team_name"),
        "probability_home_win": probabilities[0],
        "probability_draw": probabilities[1],
        "probability_away_win": probabilities[2],
        "expected_home_goals": _require_finite_float(row, "expected_home_goals"),
        "expected_away_goals": _require_finite_float(row, "expected_away_goals"),
        "modal_score": _require_nonempty_str(row, "modal_score"),
        "model_name": model_name,
        "model_version": model_version,
        "model_configuration": model_configuration,
        "model_configuration_checksum": _optional_nonempty_str(
            row.get("selected_config_checksum")
        ),
        "dataset_revision": dataset_revision,
        "dataset_checksum": _optional_nonempty_str(row.get("dataset_checksum")),
        "source_snapshot_checksum": source_snapshot_checksum,
        "prediction_snapshot_checksum": _optional_nonempty_str(
            row.get("_prediction_snapshot_checksum")
        ),
        "prediction_context": _require_nonempty_str(row, "prediction_context"),
        "prediction_status": _optional_nonempty_str(row.get("prediction_status")),
        "prospective_validity_status": validity_status,
        "invalidity_reason": invalidity_reason,
        "competition": _optional_nonempty_str(row.get("competition")),
        "stage": _optional_nonempty_str(row.get("stage")),
        "training_matches": _optional_int(row.get("training_matches")),
        "live_finished_2026_matches": _optional_int(row.get("live_finished_2026_matches")),
        "source_history_path": _optional_nonempty_str(row.get("_history_path")),
    }


def _normalize_existing_ledger_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = {field.name: row.get(field.name) for field in _ledger_schema()}
    output["prediction_created_at_utc"] = _require_utc_datetime(
        output["prediction_created_at_utc"]
    )
    output["data_cutoff_utc"] = _require_utc_datetime(output["data_cutoff_utc"])
    output["kickoff_utc"] = _require_utc_datetime(output["kickoff_utc"])
    return output


def _merge_ledger_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, str] = {}
    for row in rows:
        prediction_id = _require_nonempty_str(row, "prediction_id")
        normalized = {field.name: row.get(field.name) for field in _ledger_schema()}
        fingerprint = _row_fingerprint(normalized, exclude_source_metadata=False)
        if prediction_id in by_id:
            if fingerprints[prediction_id] != fingerprint:
                raise ProspectiveEvaluationError(
                    f"duplicate ledger prediction_id with different content: {prediction_id}"
                )
            continue
        by_id[prediction_id] = dict(normalized)
        fingerprints[prediction_id] = fingerprint
    output = list(by_id.values())
    output.sort(
        key=lambda row: (
            _require_utc_datetime(row["prediction_created_at_utc"]),
            _require_utc_datetime(row["kickoff_utc"]),
            str(row["source_fixture_id"]),
            str(row["prediction_id"]),
        )
    )
    return output


def _prospective_validity(
    row: Mapping[str, Any],
    *,
    prediction_created_at: datetime,
    data_cutoff: datetime,
    kickoff: datetime,
    probabilities: tuple[float, float, float],
    model_version: str,
    source_snapshot_checksum: str,
) -> tuple[str, str | None]:
    if prediction_created_at >= kickoff:
        return "invalid", "prediction_created_at_not_before_kickoff"
    if data_cutoff > prediction_created_at:
        return "invalid", "data_cutoff_after_prediction_creation"
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
        return "invalid", "probabilities_do_not_sum_to_one"
    if not model_version:
        return "invalid", "missing_model_version"
    if not source_snapshot_checksum:
        return "invalid", "missing_source_snapshot_checksum"
    if row.get("prediction_status") != "prospective":
        return "invalid", "prediction_status_not_prospective"
    if _prediction_row_contains_final_result(row):
        return "invalid", "final_result_available_in_prediction_snapshot"
    return "valid", None


def _prediction_row_contains_final_result(row: Mapping[str, Any]) -> bool:
    fields = (
        "actual_result",
        "metric_result_1x2",
        "home_goals_90",
        "away_goals_90",
        "result_90",
        "home_goals_after_extra_time",
        "away_goals_after_extra_time",
        "home_penalty_goals",
        "away_penalty_goals",
    )
    return any(row.get(field) is not None for field in fields)


def _canonical_result_contracts(
    live_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_fixture: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, str] = {}
    for row in live_rows:
        fixture_id = _optional_nonempty_str(row.get("source_match_id"))
        if fixture_id is None:
            continue
        contract = _canonical_result_contract(row, fixture_id=fixture_id)
        fingerprint = _row_fingerprint(contract, exclude_source_metadata=False)
        if fixture_id in by_fixture:
            if fingerprints[fixture_id] != fingerprint:
                raise ProspectiveEvaluationError(
                    f"ambiguous result contract for source_fixture_id={fixture_id}"
                )
            continue
        by_fixture[fixture_id] = contract
        fingerprints[fixture_id] = fingerprint
    return by_fixture


def _canonical_result_contract(row: Mapping[str, Any], *, fixture_id: str) -> dict[str, Any]:
    status = _optional_nonempty_str(row.get("match_status"))
    kickoff = _optional_utc_datetime(row.get("kickoff_utc"))
    base = {
        "source_fixture_id": fixture_id,
        "match_status": status,
        "kickoff_utc": kickoff,
        "home_team_id": _optional_nonempty_str(row.get("home_team_id")),
        "away_team_id": _optional_nonempty_str(row.get("away_team_id")),
        "home_team_name": _optional_nonempty_str(row.get("home_team_name_original")),
        "away_team_name": _optional_nonempty_str(row.get("away_team_name_original")),
        "stage": _optional_nonempty_str(row.get("stage")),
        "competition": _optional_nonempty_str(row.get("competition")),
        "result_metric_basis": RESULT_METRIC_BASIS,
        "extra_time_played": _optional_bool(row.get("extra_time_played")),
        "home_goals_after_extra_time": _optional_int(row.get("home_goals_after_extra_time")),
        "away_goals_after_extra_time": _optional_int(row.get("away_goals_after_extra_time")),
        "penalty_shootout": _optional_bool(row.get("penalty_shootout")),
        "home_penalty_goals": _optional_int(row.get("home_penalty_goals")),
        "away_penalty_goals": _optional_int(row.get("away_penalty_goals")),
        "data_cutoff_utc": _optional_utc_datetime(row.get("data_cutoff_utc")),
    }
    if status in {"cancelled", "postponed", "suspended", "abandoned"}:
        return {**base, "result_status": "not_evaluable", "result_status_reason": status}
    if status != "played":
        return {
            **base,
            "result_status": "not_evaluable",
            "result_status_reason": "result_not_final",
        }
    home_goals = _optional_int(row.get("home_goals_90"))
    away_goals = _optional_int(row.get("away_goals_90"))
    result_90 = _optional_nonempty_str(row.get("result_90"))
    if home_goals is None or away_goals is None or result_90 is None:
        return {
            **base,
            "result_status": "not_evaluable",
            "result_status_reason": "incomplete_result",
        }
    computed = _actual_result_from_goals(home_goals, away_goals)
    if result_90 != computed:
        raise ProspectiveEvaluationError(
            f"result_90 does not match 90-minute score for fixture {fixture_id}"
        )
    return {
        **base,
        "result_status": "evaluable",
        "result_status_reason": None,
        "home_goals_90": home_goals,
        "away_goals_90": away_goals,
        "actual_result_90": result_90,
        "metric_result_1x2": result_90,
        "qualification_winner": _qualification_winner(row, result_90=result_90),
    }


def _join_official_predictions_with_results(
    official_rows: Sequence[Mapping[str, Any]],
    *,
    result_contracts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in official_rows:
        fixture_id = str(row["source_fixture_id"])
        result = result_contracts.get(fixture_id)
        if result is None:
            output.append(
                {
                    **dict(row),
                    "evaluation_status": "not_evaluable",
                    "evaluation_status_reason": "missing_result_contract",
                }
            )
            continue
        status = str(result["result_status"])
        output.append(
            {
                **dict(row),
                **dict(result),
                "evaluation_status": "evaluated" if status == "evaluable" else "not_evaluable",
                "evaluation_status_reason": result.get("result_status_reason"),
            }
        )
    return output


def _join_all_valid_predictions_with_results(
    ledger_rows: Sequence[Mapping[str, Any]],
    *,
    result_contracts: Mapping[str, Mapping[str, Any]],
    prediction_context: str,
) -> list[dict[str, Any]]:
    valid_rows = [
        row
        for row in ledger_rows
        if row.get("prospective_validity_status") == "valid"
        and row.get("prediction_context") == prediction_context
    ]
    return _join_official_predictions_with_results(valid_rows, result_contracts=result_contracts)


def _scorecard_payload(
    *,
    config: ProspectiveEvaluationConfig,
    generated_at: datetime,
    ledger_rows: Sequence[Mapping[str, Any]],
    official_rows: Sequence[Mapping[str, Any]],
    official_evaluable: Sequence[Mapping[str, Any]],
    descriptive_evaluable: Sequence[Mapping[str, Any]],
    official_metrics: Mapping[str, Any],
    baseline_probabilities: Mapping[str, tuple[float, float, float] | None],
    results_cutoff: datetime | None,
) -> dict[str, Any]:
    baselines = _baseline_metrics(
        official_evaluable,
        baseline_probabilities,
        minimum_calibration_matches=config.minimum_calibration_matches,
    )
    policy = config.official_selection
    official_by_horizon = _group_metrics(
        official_evaluable,
        "horizon_bucket",
        minimum_calibration_matches=config.minimum_calibration_matches,
    )
    descriptive_by_horizon = _group_metrics(
        descriptive_evaluable,
        "horizon_bucket",
        minimum_calibration_matches=config.minimum_calibration_matches,
    )
    by_stage = _group_metrics(
        official_evaluable,
        "stage",
        minimum_calibration_matches=config.minimum_calibration_matches,
        missing_label="unknown",
    )
    by_favorite = _favorite_segment_metrics(
        official_evaluable,
        minimum_calibration_matches=config.minimum_calibration_matches,
    )
    small_sample = len(official_evaluable) < config.small_sample_warning_threshold
    validity_counts = Counter(str(row["prospective_validity_status"]) for row in ledger_rows)
    invalidity_counts = Counter(
        str(row["invalidity_reason"])
        for row in ledger_rows
        if row.get("invalidity_reason") is not None
    )
    snapshot_count = len(
        {
            str(row["source_history_path"])
            for row in ledger_rows
            if row.get("source_history_path") is not None
        }
    )
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "generated_at_utc": _format_utc(generated_at),
        "result_metric_basis": RESULT_METRIC_BASIS,
        "result_metric_basis_description": "1X2 metrics use the 90-minute result only.",
        "results_cutoff_utc": _format_utc(results_cutoff) if results_cutoff else None,
        "ledger": {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "predictions": len(ledger_rows),
            "unique_fixtures": len({str(row["source_fixture_id"]) for row in ledger_rows}),
            "snapshots": snapshot_count,
            "validity_counts": dict(sorted(validity_counts.items())),
            "invalidity_counts": dict(sorted(invalidity_counts.items())),
        },
        "horizons": {
            "version": config.horizon_version,
            "buckets": [
                {
                    "id": bucket.bucket_id,
                    "label": bucket.label,
                    "min_hours": bucket.min_hours,
                    "min_inclusive": bucket.min_inclusive,
                    "max_hours": bucket.max_hours,
                    "max_inclusive": bucket.max_inclusive,
                }
                for bucket in config.horizon_buckets
            ],
        },
        "official_selection_policy": {
            "policy_id": policy.policy_id,
            "policy_version": policy.policy_version,
            "prediction_context": policy.prediction_context,
            "primary_rule": {
                "id": policy.primary_rule_id,
                "min_hours_before_kickoff": policy.min_hours_before_kickoff,
                "description": (
                    "latest valid prediction generated at least the configured hours before kickoff"
                ),
            },
            "fallback_rule": {
                "id": policy.fallback_rule_id,
                "description": (
                    "earliest valid prediction before kickoff when the primary rule has no "
                    "candidate"
                ),
            },
        },
        "official_predictions_selected": len(official_rows),
        "official_predictions_evaluated": len(official_evaluable),
        "metrics": dict(official_metrics),
        "baselines": baselines,
        "metrics_by_horizon": official_by_horizon,
        "descriptive_metrics_by_horizon_all_valid_predictions": descriptive_by_horizon,
        "metrics_by_stage": by_stage,
        "metrics_by_favorite_segment": by_favorite,
        "matches": [_match_payload(row, baseline_probabilities) for row in official_evaluable],
        "small_sample_warning": {
            "applies": small_sample,
            "threshold": config.small_sample_warning_threshold,
            "message": (
                "Sample is too small for firm statistical conclusions."
                if small_sample
                else None
            ),
        },
    }


def _baseline_probabilities(
    config: ProspectiveEvaluationConfig,
    official_rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[float, float, float] | None]:
    probabilities: dict[str, tuple[float, float, float] | None] = {}
    if config.baselines.uniform_enabled:
        probabilities["uniform_1x2"] = (1 / 3, 1 / 3, 1 / 3)
    if config.baselines.historical_frequency.enabled:
        probabilities["historical_frequency_frozen"] = _historical_frequency_probabilities(
            config.baselines.historical_frequency
        )
    if config.baselines.elo_enabled or official_rows:
        probabilities["elo_operational"] = None
    return probabilities


def _historical_frequency_probabilities(
    baseline_config: HistoricalFrequencyBaselineConfig,
) -> tuple[float, float, float] | None:
    if not baseline_config.input_matches_path.is_file():
        return None
    rows = _read_live_matches(baseline_config.input_matches_path)
    counts = Counter[str]()
    for row in rows:
        if row.get("model_eligible") is not True:
            continue
        if str(row.get("match_status") or "played") != "played":
            continue
        kickoff = _optional_utc_datetime(row.get("kickoff_utc"))
        match_date = _optional_date(row.get("match_date"))
        if kickoff is not None:
            if kickoff >= baseline_config.cutoff_utc:
                continue
        elif match_date is not None and match_date >= baseline_config.cutoff_utc.date():
            continue
        result = _result_from_row(row)
        if result is not None:
            counts[result] += 1
    total = sum(counts.values())
    if total == 0:
        return None
    return (
        counts["home_win"] / total,
        counts["draw"] / total,
        counts["away_win"] / total,
    )


def _baseline_metrics(
    rows: Sequence[Mapping[str, Any]],
    baseline_probabilities: Mapping[str, tuple[float, float, float] | None],
    *,
    minimum_calibration_matches: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, probabilities in baseline_probabilities.items():
        if probabilities is None:
            output[name] = {"status": "not_available", "matches": len(rows)}
            continue
        baseline_rows = [
            {
                **dict(row),
                "probability_home_win": probabilities[0],
                "probability_draw": probabilities[1],
                "probability_away_win": probabilities[2],
            }
            for row in rows
        ]
        output[name] = {
            "status": "computed",
            "probabilities": {
                "home_win": probabilities[0],
                "draw": probabilities[1],
                "away_win": probabilities[2],
            },
            "metrics": metrics_for_rows(
                baseline_rows,
                minimum_calibration_matches=minimum_calibration_matches,
            ),
        }
    return output


def _group_metrics(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    minimum_calibration_matches: int,
    missing_label: str = "unbucketed",
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        label = _optional_nonempty_str(row.get(field)) or missing_label
        grouped[label].append(row)
    return {
        key: metrics_for_rows(values, minimum_calibration_matches=minimum_calibration_matches)
        for key, values in sorted(grouped.items())
    }


def _favorite_segment_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_calibration_matches: int,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        favorite = _predicted_class_name(row)
        actual = str(row["metric_result_1x2"])
        grouped["favorite" if favorite == actual else "non_favorite"].append(row)
    return {
        key: metrics_for_rows(values, minimum_calibration_matches=minimum_calibration_matches)
        for key, values in sorted(grouped.items())
    }


def _match_payload(
    row: Mapping[str, Any],
    baseline_probabilities: Mapping[str, tuple[float, float, float] | None],
) -> dict[str, Any]:
    payload = {
        "source_fixture_id": row["source_fixture_id"],
        "prediction_id": row["prediction_id"],
        "prediction_created_at_utc": _format_utc(
            _require_utc_datetime(row["prediction_created_at_utc"])
        ),
        "data_cutoff_utc": _format_utc(_require_utc_datetime(row["data_cutoff_utc"])),
        "kickoff_utc": _format_utc(_require_utc_datetime(row["kickoff_utc"])),
        "hours_before_kickoff": row["hours_before_kickoff"],
        "horizon_bucket": row["horizon_bucket"],
        "home_team_name": row["home_team_name"],
        "away_team_name": row["away_team_name"],
        "probability_home_win": row["probability_home_win"],
        "probability_draw": row["probability_draw"],
        "probability_away_win": row["probability_away_win"],
        "predicted_result": _predicted_class_name(row),
        "actual_result_90": row["actual_result_90"],
        "metric_result_1x2": row["metric_result_1x2"],
        "home_goals_90": row["home_goals_90"],
        "away_goals_90": row["away_goals_90"],
        "extra_time_played": row.get("extra_time_played"),
        "home_goals_after_extra_time": row.get("home_goals_after_extra_time"),
        "away_goals_after_extra_time": row.get("away_goals_after_extra_time"),
        "penalty_shootout": row.get("penalty_shootout"),
        "home_penalty_goals": row.get("home_penalty_goals"),
        "away_penalty_goals": row.get("away_penalty_goals"),
        "qualification_winner": row.get("qualification_winner"),
        "stage": row.get("stage"),
        "official_selection_rule": row["official_selection_rule"],
        "model_name": row["model_name"],
        "model_version": row["model_version"],
        "dataset_revision": row["dataset_revision"],
        "source_snapshot_checksum": row["source_snapshot_checksum"],
    }
    for name, probabilities in baseline_probabilities.items():
        if probabilities is None:
            continue
        payload[f"{name}_probability_home_win"] = probabilities[0]
        payload[f"{name}_probability_draw"] = probabilities[1]
        payload[f"{name}_probability_away_win"] = probabilities[2]
    return payload


def _write_ledger(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(row) for row in rows], schema=_ledger_schema())
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


def _write_scorecard_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_value) + "\n")


def _write_scorecard_markdown(payload: Mapping[str, Any], path: Path) -> None:
    lines = _scorecard_markdown(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_matches_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    baseline_probabilities: Mapping[str, tuple[float, float, float] | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _matches_csv_fields(baseline_probabilities)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = _match_payload(row, baseline_probabilities)
            writer.writerow({field: _csv_value(payload.get(field)) for field in fieldnames})


def _scorecard_markdown(payload: Mapping[str, Any]) -> list[str]:
    metrics = _require_mapping(payload, "metrics")
    policy = _require_mapping(payload, "official_selection_policy")
    primary = _require_mapping(policy, "primary_rule")
    fallback = _require_mapping(policy, "fallback_rule")
    small_sample = _require_mapping(payload, "small_sample_warning")
    baselines = _require_mapping(payload, "baselines")
    matches = payload.get("matches")
    match_rows = matches if isinstance(matches, list) else []
    lines = [
        "# Prospective Scorecard",
        "",
        f"Generated UTC: {payload.get('generated_at_utc')}",
        f"Results cutoff UTC: {payload.get('results_cutoff_utc') or 'n/a'}",
        "1X2 metric basis: 90-minute result (`result_90`). Extra time and penalties are reported "
        "separately and are not mixed into the 1X2 metric.",
        "",
        "## Official Policy",
        "",
        f"Policy: {policy.get('policy_id')} ({policy.get('policy_version')})",
        f"Context: {policy.get('prediction_context')}",
        f"Primary rule: {primary.get('id')} at >= {primary.get('min_hours_before_kickoff')} hours",
        f"Fallback rule: {fallback.get('id')}",
        "",
    ]
    if small_sample.get("applies") is True:
        lines.extend(
            [
                "> Sample is too small for firm statistical conclusions. Reported aggregates are "
                "monitoring diagnostics, not evidence of model improvement.",
                "",
            ]
        )
    lines.extend(
        [
            "## Metrics",
            "",
            f"Official matches evaluated: {payload.get('official_predictions_evaluated')}",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| log loss | {_metric_value(metrics.get('log_loss'))} |",
            f"| Brier score | {_metric_value(metrics.get('brier_score'))} |",
            f"| RPS | {_metric_value(metrics.get('ranked_probability_score'))} |",
            f"| accuracy | {_metric_value(metrics.get('accuracy'))} |",
            f"| calibration error | {_metric_value(metrics.get('calibration_error'))} |",
            f"| mean hours before kickoff | "
            f"{_metric_value(metrics.get('average_hours_before_kickoff'))} |",
            f"| median hours before kickoff | "
            f"{_metric_value(metrics.get('median_hours_before_kickoff'))} |",
            "",
            "## Baselines",
            "",
            "| Baseline | Status | Matches | Log loss | Brier | RPS | Accuracy |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, values in baselines.items():
        if not isinstance(values, Mapping):
            continue
        status = str(values.get("status"))
        baseline_metrics = (
            values.get("metrics") if isinstance(values.get("metrics"), Mapping) else {}
        )
        metrics_mapping = baseline_metrics if isinstance(baseline_metrics, Mapping) else {}
        lines.append(
            "| "
            + " | ".join(
                (
                    str(name),
                    status,
                    str(metrics_mapping.get("matches", values.get("matches", "n/a"))),
                    _metric_value(metrics_mapping.get("log_loss")),
                    _metric_value(metrics_mapping.get("brier_score")),
                    _metric_value(metrics_mapping.get("ranked_probability_score")),
                    _metric_value(metrics_mapping.get("accuracy")),
                )
            )
            + " |"
        )
    lines.extend(["", "## Matches", ""])
    if not match_rows:
        lines.append("No evaluable official predictions at this results cutoff.")
        return lines
    lines.extend(
        [
            "| Kickoff UTC | Match | Pick | Actual 90 | Rule | Log-loss input |",
            "| --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for row in match_rows:
        if not isinstance(row, Mapping):
            continue
        actual = str(row.get("metric_result_1x2"))
        probability = _probability_for_actual(row, actual)
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row.get("kickoff_utc")),
                    f"{row.get('home_team_name')} vs {row.get('away_team_name')}",
                    str(row.get("predicted_result")),
                    actual,
                    str(row.get("official_selection_rule")),
                    _metric_value(probability),
                )
            )
            + " |"
        )
    return lines


def _matches_csv_fields(
    baseline_probabilities: Mapping[str, tuple[float, float, float] | None],
) -> list[str]:
    fields = [
        "source_fixture_id",
        "prediction_id",
        "prediction_created_at_utc",
        "data_cutoff_utc",
        "kickoff_utc",
        "hours_before_kickoff",
        "horizon_bucket",
        "home_team_name",
        "away_team_name",
        "probability_home_win",
        "probability_draw",
        "probability_away_win",
        "predicted_result",
        "actual_result_90",
        "metric_result_1x2",
        "home_goals_90",
        "away_goals_90",
        "extra_time_played",
        "home_goals_after_extra_time",
        "away_goals_after_extra_time",
        "penalty_shootout",
        "home_penalty_goals",
        "away_penalty_goals",
        "qualification_winner",
        "stage",
        "official_selection_rule",
        "model_name",
        "model_version",
        "dataset_revision",
        "source_snapshot_checksum",
    ]
    for name, probabilities in sorted(baseline_probabilities.items()):
        if probabilities is None:
            continue
        fields.extend(
            [
                f"{name}_probability_home_win",
                f"{name}_probability_draw",
                f"{name}_probability_away_win",
            ]
        )
    return fields


def _ledger_schema() -> pa.Schema:
    return pa.schema(
        [
            ("schema_version", pa.string()),
            ("prediction_id", pa.string()),
            ("source_prediction_id", pa.string()),
            ("prediction_id_repair_reason", pa.string()),
            ("prediction_run_id", pa.string()),
            ("source_fixture_id", pa.string()),
            ("match_id", pa.string()),
            ("source", pa.string()),
            ("prediction_created_at_utc", pa.timestamp("us", tz="UTC")),
            ("data_cutoff_utc", pa.timestamp("us", tz="UTC")),
            ("kickoff_utc", pa.timestamp("us", tz="UTC")),
            ("hours_before_kickoff", pa.float64()),
            ("horizon_bucket", pa.string()),
            ("home_team_id", pa.string()),
            ("away_team_id", pa.string()),
            ("home_team_name", pa.string()),
            ("away_team_name", pa.string()),
            ("probability_home_win", pa.float64()),
            ("probability_draw", pa.float64()),
            ("probability_away_win", pa.float64()),
            ("expected_home_goals", pa.float64()),
            ("expected_away_goals", pa.float64()),
            ("modal_score", pa.string()),
            ("model_name", pa.string()),
            ("model_version", pa.string()),
            ("model_configuration", pa.string()),
            ("model_configuration_checksum", pa.string()),
            ("dataset_revision", pa.string()),
            ("dataset_checksum", pa.string()),
            ("source_snapshot_checksum", pa.string()),
            ("prediction_snapshot_checksum", pa.string()),
            ("prediction_context", pa.string()),
            ("prediction_status", pa.string()),
            ("prospective_validity_status", pa.string()),
            ("invalidity_reason", pa.string()),
            ("competition", pa.string()),
            ("stage", pa.string()),
            ("training_matches", pa.int64()),
            ("live_finished_2026_matches", pa.int64()),
            ("source_history_path", pa.string()),
        ]
    )


def _probability_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                row["probability_home_win"],
                row["probability_draw"],
                row["probability_away_win"],
            ]
            for row in rows
        ],
        dtype=float,
    )


def _ranked_probability_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    true = np.eye(len(CLASS_LABELS))[targets]
    return float(
        np.mean(
            np.sum(
                np.square(np.cumsum(probabilities, axis=1) - np.cumsum(true, axis=1)),
                axis=1,
            )
            / (len(CLASS_LABELS) - 1)
        )
    )


def _calibration_error(probabilities: np.ndarray, targets: np.ndarray, *, bins: int) -> float:
    confidences = np.max(probabilities, axis=1)
    predicted = np.argmax(probabilities, axis=1)
    correct = (predicted == targets).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = len(confidences)
    error = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        error += float(
            np.sum(mask)
            / total
            * abs(np.mean(confidences[mask]) - np.mean(correct[mask]))
        )
    return error


def _empty_metrics(*, minimum_calibration_matches: int) -> dict[str, Any]:
    return {
        "matches": 0,
        "log_loss": None,
        "brier_score": None,
        "ranked_probability_score": None,
        "accuracy": None,
        "calibration_error": None,
        "calibration_status": "insufficient_sample",
        "calibration_minimum_matches": minimum_calibration_matches,
        "kickoff_start": None,
        "kickoff_end": None,
        "average_hours_before_kickoff": None,
        "median_hours_before_kickoff": None,
    }


def _results_cutoff(live_rows: Sequence[Mapping[str, Any]], path: Path) -> datetime | None:
    cutoffs = [
        value
        for value in (_optional_utc_datetime(row.get("data_cutoff_utc")) for row in live_rows)
        if value is not None
    ]
    if cutoffs:
        return max(cutoffs)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return _optional_utc_datetime(payload.get("data_cutoff_utc") or payload.get("fetched_at"))


def _horizon_bucket(hours_before_kickoff: float, buckets: Sequence[HorizonBucket]) -> str:
    for bucket in buckets:
        if bucket.contains(hours_before_kickoff):
            return bucket.bucket_id
    return "unbucketed"


def _derived_repaired_prediction_id(
    row: Mapping[str, Any],
    *,
    source_prediction_id: str,
) -> str:
    fields = (
        source_prediction_id,
        _format_utc(_require_utc_datetime(row.get("prediction_created_at_utc"))),
        str(row.get("prediction_run_id") or ""),
        str(row.get("_history_path") or ""),
    )
    return hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()[:24]


def _require_source_snapshot_checksum(row: Mapping[str, Any]) -> str:
    value = _optional_nonempty_str(row.get("source_snapshot_checksum"))
    if value is not None:
        return value
    value = _optional_nonempty_str(row.get("live_snapshot_checksum"))
    if value is not None:
        return value
    raise ProspectiveEvaluationError("source snapshot checksum is required")


def _require_probability(row: Mapping[str, Any], field_name: str) -> float:
    value = _require_finite_float(row, field_name)
    if value < 0 or value > 1:
        raise ProspectiveEvaluationError(f"{field_name} must be between 0 and 1")
    return value


def _require_finite_float(row: Mapping[str, Any], field_name: str) -> float:
    value = row.get(field_name)
    if isinstance(value, int | float | np.integer | np.floating) and not isinstance(value, bool):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    raise ProspectiveEvaluationError(f"{field_name} must be a finite number")


def _require_nonempty_str(row: Mapping[str, Any], field_name: str) -> str:
    value = _optional_nonempty_str(row.get(field_name))
    if value is None:
        raise ProspectiveEvaluationError(f"{field_name} is required")
    return value


def _optional_nonempty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProspectiveEvaluationError(f"timestamp is not valid ISO 8601 UTC: {value}") from exc
    return _require_utc(parsed)


def _require_utc_datetime(value: object) -> datetime:
    parsed = _optional_utc_datetime(value)
    if parsed is None:
        raise ProspectiveEvaluationError("timestamps must be timezone-aware UTC")
    return parsed


def _optional_utc_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return _require_utc(value)
    if isinstance(value, str) and value:
        return _parse_utc(value)
    return None


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ProspectiveEvaluationError("timestamps must be timezone-aware UTC")
    return value.astimezone(UTC)


def _format_utc(value: datetime) -> str:
    return _require_utc(value).isoformat().replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _class_from_name(name: str) -> int:
    if name == "home_win":
        return HOME_CLASS
    if name == "draw":
        return DRAW_CLASS
    if name == "away_win":
        return AWAY_CLASS
    raise ProspectiveEvaluationError(f"unknown result class: {name}")


def _predicted_class_name(row: Mapping[str, Any]) -> str:
    values = [
        float(row["probability_home_win"]),
        float(row["probability_draw"]),
        float(row["probability_away_win"]),
    ]
    return CLASS_NAMES[int(np.argmax(values))]


def _actual_result_from_goals(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def _result_from_row(row: Mapping[str, Any]) -> str | None:
    result = _optional_nonempty_str(row.get("result_90"))
    if result in set(CLASS_NAMES):
        return result
    home = _optional_int(row.get("home_goals_90"))
    away = _optional_int(row.get("away_goals_90"))
    if home is None or away is None:
        return None
    return _actual_result_from_goals(home, away)


def _qualification_winner(row: Mapping[str, Any], *, result_90: str) -> str | None:
    if result_90 == "home_win":
        return "home"
    if result_90 == "away_win":
        return "away"
    home_penalties = _optional_int(row.get("home_penalty_goals"))
    away_penalties = _optional_int(row.get("away_penalty_goals"))
    if (
        home_penalties is not None
        and away_penalties is not None
        and home_penalties != away_penalties
    ):
        return "home" if home_penalties > away_penalties else "away"
    home_after_extra = _optional_int(row.get("home_goals_after_extra_time"))
    away_after_extra = _optional_int(row.get("away_goals_after_extra_time"))
    if (
        home_after_extra is not None
        and away_after_extra is not None
        and home_after_extra != away_after_extra
    ):
        return "home" if home_after_extra > away_after_extra else "away"
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, int | np.integer) and not isinstance(value, bool):
        return int(value)
    return None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_date(value: object) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str) and value:
        return date.fromisoformat(value)
    return None


def _parse_horizon_bucket(payload: object) -> HorizonBucket:
    if not isinstance(payload, Mapping):
        raise ProspectiveEvaluationError("horizon bucket entries must be objects")
    return HorizonBucket(
        bucket_id=_require_str(payload, "id"),
        label=_require_str(payload, "label"),
        min_hours=_optional_float(payload.get("min_hours")),
        min_inclusive=_require_bool(payload, "min_inclusive"),
        max_hours=_optional_float(payload.get("max_hours")),
        max_inclusive=_require_bool(payload, "max_inclusive"),
    )


def _require_mapping(payload: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    value = payload.get(field_name)
    if not isinstance(value, Mapping):
        raise ProspectiveEvaluationError(f"{field_name} must be an object")
    return value


def _require_sequence(payload: Mapping[str, Any], field_name: str) -> Sequence[Any]:
    value = payload.get(field_name)
    if not isinstance(value, list):
        raise ProspectiveEvaluationError(f"{field_name} must be a list")
    return value


def _require_str(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if isinstance(value, str) and value:
        return value
    raise ProspectiveEvaluationError(f"{field_name} must be a non-empty string")


def _require_int(payload: Mapping[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ProspectiveEvaluationError(f"{field_name} must be an integer")


def _require_float(payload: Mapping[str, Any], field_name: str) -> float:
    value = payload.get(field_name)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise ProspectiveEvaluationError(f"{field_name} must be numeric")


def _require_bool(payload: Mapping[str, Any], field_name: str) -> bool:
    value = payload.get(field_name)
    if isinstance(value, bool):
        return value
    raise ProspectiveEvaluationError(f"{field_name} must be boolean")


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float | np.integer | np.floating) and not isinstance(value, bool):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    return None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _metric_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _probability_for_actual(row: Mapping[str, Any], actual: str) -> float | None:
    if actual == "home_win":
        return _optional_float(row.get("probability_home_win"))
    if actual == "draw":
        return _optional_float(row.get("probability_draw"))
    if actual == "away_win":
        return _optional_float(row.get("probability_away_win"))
    return None


def _row_fingerprint(row: Mapping[str, Any], *, exclude_source_metadata: bool) -> str:
    excluded = (
        {"_history_path", "_prediction_snapshot_checksum"}
        if exclude_source_metadata
        else set()
    )
    payload = {key: _json_value(value) for key, value in row.items() if key not in excluded}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _csv_value(value: object) -> object:
    if isinstance(value, datetime):
        return _format_utc(value)
    if isinstance(value, float):
        return f"{value:.10f}"
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return _format_utc(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
