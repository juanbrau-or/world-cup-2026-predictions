"""Shadow-mode contextual challenger predictions and prospective evaluation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from worldcup2026.evaluation.contextual_challenger import (
    ContextualChallengerConfig,
    FeatureWhitelistConfig,
    _contextual_rows_by_match_id,
    _feature_frame,
    _logits,
    fit_selected_shadow_estimator,
    load_contextual_challenger_config,
    load_feature_whitelist,
)
from worldcup2026.evaluation.prospective import (
    ProspectiveEvaluationError,
    run_prospective_evaluation,
)
from worldcup2026.pipelines.operational_predictions import (
    PREDICTION_SCHEMA_VERSION,
    PREDICTION_STATUS,
    _assert_prediction_probabilities,
    _json_checksum,
    _live_data_cutoff,
    _live_snapshot_checksum,
    _operational_training_rows,
    _prediction_run_id,
    _read_parquet_rows,
    _rows_checksum,
    _stable_prediction_id,
    _write_immutable_predictions_parquet,
    _write_predictions_csv,
    _write_predictions_parquet,
)

SHADOW_MODEL_FAMILY = "contextual_challenger"
SHADOW_CONTEXT = "shadow_contextual_v1"
SHADOW_OUTPUT_VERSION = "shadow_contextual_outputs_v1"


class ShadowContextualError(RuntimeError):
    """Raised when shadow contextual predictions cannot be produced safely."""


@dataclass(frozen=True)
class ShadowContextualPredictionResult:
    """Summary returned by shadow prediction generation."""

    predictions: tuple[dict[str, Any], ...]
    latest_csv_path: Path
    latest_parquet_path: Path
    history_path: Path
    report_path: Path
    model_family: str
    model_version: str
    data_cutoff_utc: datetime
    baseline_fixture_count: int
    training_matches: int
    live_finished_2026_matches: int


@dataclass(frozen=True)
class ShadowContextualEvaluationResult:
    """Summary returned by shadow prospective evaluation."""

    scorecard_json_path: Path
    scorecard_report_path: Path
    ledger_path: Path
    comparison_path: Path
    evaluable_predictions: int
    paired_matches: int


def run_predict_shadow_contextual(
    *,
    model_config_path: Path = Path("configs/model.yaml"),
    modeling_matches_path: Path = Path("data/processed/modeling_matches.parquet"),
    live_matches_path: Path = Path("data/processed/world_cup_2026/matches.parquet"),
    contextual_match_features_path: Path = Path(
        "data/processed/contextual_features/match_contextual_features.parquet"
    ),
    ingest_report_path: Path = Path("data/interim/world_cup_2026_ingest_report.json"),
    predictions_root: Path = Path("predictions"),
    created_at: datetime | None = None,
) -> ShadowContextualPredictionResult:
    """Generate contextual challenger predictions for the exact official fixtures."""

    prediction_created_at = _utc_now() if created_at is None else _require_utc(created_at)
    goal_config, elo_config, _, challenger_config = load_contextual_challenger_config(
        model_config_path
    )
    whitelist = load_feature_whitelist(challenger_config.feature_whitelist_path)
    baseline_rows = _read_baseline_latest(predictions_root / "latest.parquet")
    historical_rows = _read_parquet_rows(modeling_matches_path)
    live_rows = _read_parquet_rows(live_matches_path)
    contextual_rows = _read_parquet_rows(contextual_match_features_path)
    contextual_by_id = _contextual_rows_by_match_id(contextual_rows)
    live_cutoff = _live_data_cutoff(live_rows, ingest_report_path=ingest_report_path)
    cutoff = _baseline_cutoff(baseline_rows) or live_cutoff
    if cutoff != live_cutoff and baseline_rows:
        raise ShadowContextualError("baseline cutoff does not match current live cutoff")
    training_rows, live_finished_rows = _operational_training_rows(
        historical_rows,
        live_rows,
        cutoff=cutoff,
    )
    estimator, _, _, scaler = fit_selected_shadow_estimator(
        rows=training_rows,
        contextual_by_id=contextual_by_id,
        goal_config=goal_config,
        elo_config=elo_config,
        config=challenger_config,
        whitelist=whitelist,
    )
    fixture_rows = _official_fixture_rows(live_rows, baseline_rows)
    _assert_same_fixtures(baseline_rows, fixture_rows)
    feature_frame = _shadow_feature_frame(
        fixture_rows,
        baseline_rows=baseline_rows,
        contextual_by_id=contextual_by_id,
        whitelist=whitelist,
        challenger_config=challenger_config,
    )
    probabilities = scaler.transform(estimator.predict_proba(feature_frame))
    live_snapshot_checksum = _live_snapshot_checksum(
        live_matches_path,
        ingest_report_path=ingest_report_path,
    )
    dataset_checksum = _rows_checksum(training_rows)
    dataset_revision = f"shadow_contextual_dataset_v1:{dataset_checksum[:16]}"
    selected_config = _shadow_selected_config(challenger_config, whitelist)
    config_checksum = _json_checksum(selected_config)
    prediction_run_id = _prediction_run_id(
        created_at=prediction_created_at,
        cutoff=cutoff,
        live_snapshot_checksum=live_snapshot_checksum,
        dataset_revision=dataset_revision,
        config_checksum=config_checksum,
    )
    predictions = tuple(
        _shadow_prediction_row(
            fixture,
            baseline=baseline_rows[index],
            probabilities=probabilities[index],
            prediction_created_at=prediction_created_at,
            data_cutoff_utc=cutoff,
            prediction_run_id=prediction_run_id,
            selected_config=selected_config,
            config_checksum=config_checksum,
            dataset_revision=dataset_revision,
            dataset_checksum=dataset_checksum,
            live_snapshot_checksum=live_snapshot_checksum,
            training_matches=len(training_rows),
            live_finished_2026_matches=len(live_finished_rows),
            challenger_config=challenger_config,
        )
        for index, fixture in enumerate(fixture_rows)
    )
    _assert_prediction_probabilities(predictions)
    shadow_root = predictions_root / "shadow"
    latest_csv = shadow_root / "contextual_latest.csv"
    latest_parquet = shadow_root / "contextual_latest.parquet"
    report_path = shadow_root / "contextual_upcoming.md"
    history_path = _shadow_history_path(
        shadow_root,
        created_at=prediction_created_at,
        predictions=predictions,
        live_snapshot_checksum=live_snapshot_checksum,
    )
    _write_predictions_parquet(predictions, latest_parquet)
    _write_predictions_csv(predictions, latest_csv)
    _write_immutable_predictions_parquet(predictions, history_path)
    _write_shadow_report(
        predictions,
        report_path,
        cutoff=cutoff,
        baseline_fixture_count=len(baseline_rows),
        training_matches=len(training_rows),
        live_finished_2026_matches=len(live_finished_rows),
        challenger_config=challenger_config,
    )
    return ShadowContextualPredictionResult(
        predictions=predictions,
        latest_csv_path=latest_csv,
        latest_parquet_path=latest_parquet,
        history_path=history_path,
        report_path=report_path,
        model_family=SHADOW_MODEL_FAMILY,
        model_version=challenger_config.shadow_selection.selected_model_name,
        data_cutoff_utc=cutoff,
        baseline_fixture_count=len(baseline_rows),
        training_matches=len(training_rows),
        live_finished_2026_matches=len(live_finished_rows),
    )


def run_evaluate_shadow_contextual(
    *,
    config_path: Path = Path("configs/shadow_contextual_evaluation.yaml"),
    official_scorecard_path: Path = Path("predictions/prospective_scorecard.json"),
    shadow_history_root: Path = Path("predictions/shadow/history"),
    live_matches_path: Path = Path("data/processed/world_cup_2026/matches.parquet"),
    predictions_root: Path = Path("predictions"),
    published_history_root: Path | None = None,
) -> ShadowContextualEvaluationResult:
    """Evaluate shadow predictions in a separate prospective ledger."""

    shadow_root = predictions_root / "shadow"
    matches_csv_path = shadow_root / "contextual_matches.csv"
    try:
        result = run_prospective_evaluation(
            config_path=config_path,
            predictions_history_root=shadow_history_root,
            live_matches_path=live_matches_path,
            report_path=shadow_root / "contextual_scorecard.md",
            json_path=shadow_root / "contextual_scorecard.json",
            matches_csv_path=matches_csv_path,
            ledger_path=shadow_root / "contextual_ledger.parquet",
            published_history_root=published_history_root,
        )
    except ProspectiveEvaluationError as exc:
        raise ShadowContextualError(str(exc)) from exc
    paired_matches = _write_shadow_comparison(
        official_scorecard_path=official_scorecard_path,
        shadow_scorecard_path=result.json_path,
        output_path=shadow_root / "contextual_comparison.md",
    )
    return ShadowContextualEvaluationResult(
        scorecard_json_path=result.json_path,
        scorecard_report_path=result.report_path,
        ledger_path=result.ledger_path,
        comparison_path=shadow_root / "contextual_comparison.md",
        evaluable_predictions=result.evaluable_predictions,
        paired_matches=paired_matches,
    )


def _shadow_feature_frame(
    fixture_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    contextual_by_id: Mapping[str, Mapping[str, Any]],
    whitelist: FeatureWhitelistConfig,
    challenger_config: ContextualChallengerConfig,
) -> Any:
    poisson_by_id = {
        _require_str(row, "match_id"): _poisson_from_baseline(row) for row in baseline_rows
    }
    elo_by_id = {
        _require_str(row, "match_id"): {
            "home_elo_pre": float(row["home_elo_pre"]),
            "away_elo_pre": float(row["away_elo_pre"]),
            "elo_difference_pre": float(row["home_elo_pre"]) - float(row["away_elo_pre"]),
        }
        for row in baseline_rows
    }
    include_contextual = challenger_config.shadow_selection.selected_ablation.endswith(
        "contextual"
    )
    prepared_rows = [_fixture_model_row(row) for row in fixture_rows]
    return _feature_frame(
        prepared_rows,
        whitelist=whitelist,
        contextual_by_id=contextual_by_id,
        elo_by_id=elo_by_id,
        poisson_by_id=poisson_by_id,
        include_contextual=include_contextual,
    )


def _poisson_from_baseline(row: Mapping[str, Any]) -> Mapping[str, float]:
    probabilities = np.asarray(
        [
            float(row["probability_home_win"]),
            float(row["probability_draw"]),
            float(row["probability_away_win"]),
        ],
        dtype=float,
    )
    logits = _logits(probabilities)
    return {
        "base_poisson_prob_home_win": float(probabilities[0]),
        "base_poisson_prob_draw": float(probabilities[1]),
        "base_poisson_prob_away_win": float(probabilities[2]),
        "base_poisson_expected_home_goals": float(row["expected_home_goals"]),
        "base_poisson_expected_away_goals": float(row["expected_away_goals"]),
        "base_poisson_logit_home_win": float(logits[0]),
        "base_poisson_logit_draw": float(logits[1]),
        "base_poisson_logit_away_win": float(logits[2]),
        "base_poisson_log_home_away_ratio": float(
            math.log(max(probabilities[0], 1e-15) / max(probabilities[2], 1e-15))
        ),
    }


def _shadow_prediction_row(
    fixture: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any],
    probabilities: np.ndarray,
    prediction_created_at: datetime,
    data_cutoff_utc: datetime,
    prediction_run_id: str,
    selected_config: Mapping[str, Any],
    config_checksum: str,
    dataset_revision: str,
    dataset_checksum: str,
    live_snapshot_checksum: str,
    training_matches: int,
    live_finished_2026_matches: int,
    challenger_config: ContextualChallengerConfig,
) -> dict[str, Any]:
    source_fixture_id = _require_str(fixture, "source_match_id")
    kickoff = _require_datetime(fixture, "kickoff_utc")
    model_version = challenger_config.shadow_selection.selected_model_name
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "prediction_id": _stable_prediction_id(
            source_fixture_id=source_fixture_id,
            prediction_created_at_utc=prediction_created_at,
            data_cutoff_utc=data_cutoff_utc,
            model_version=model_version,
            dataset_revision=dataset_revision,
            config_checksum=config_checksum,
        ),
        "prediction_run_id": prediction_run_id,
        "source_fixture_id": source_fixture_id,
        "match_id": _require_str(fixture, "match_id"),
        "source": _require_str(fixture, "source"),
        "prediction_created_at_utc": prediction_created_at,
        "data_cutoff_utc": data_cutoff_utc,
        "kickoff_utc": kickoff,
        "hours_before_kickoff": (kickoff - prediction_created_at).total_seconds() / 3600,
        "home_team_id": _require_str(fixture, "home_team_id"),
        "away_team_id": _require_str(fixture, "away_team_id"),
        "home_team_name": _require_str(fixture, "home_team_name_original"),
        "away_team_name": _require_str(fixture, "away_team_name_original"),
        "home_elo_pre": float(baseline["home_elo_pre"]),
        "away_elo_pre": float(baseline["away_elo_pre"]),
        "expected_home_goals": float(baseline["expected_home_goals"]),
        "expected_away_goals": float(baseline["expected_away_goals"]),
        "probability_home_win": float(probabilities[0]),
        "probability_draw": float(probabilities[1]),
        "probability_away_win": float(probabilities[2]),
        "modal_score": _require_str(baseline, "modal_score"),
        "score_probabilities_json": baseline.get("score_probabilities_json"),
        "score_probability_mass": float(baseline["score_probability_mass"]),
        "residual_probability": float(baseline["residual_probability"]),
        "model_family": SHADOW_MODEL_FAMILY,
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
        "prediction_context": challenger_config.shadow_prediction_context,
        "prediction_status": PREDICTION_STATUS,
        "competition": _require_str(fixture, "competition"),
        "stage": fixture.get("stage"),
        "training_matches": training_matches,
        "live_finished_2026_matches": live_finished_2026_matches,
    }


def _shadow_selected_config(
    config: ContextualChallengerConfig,
    whitelist: FeatureWhitelistConfig,
) -> Mapping[str, Any]:
    return {
        "schema_version": "shadow_contextual_selected_config_v1",
        "model_name": config.shadow_selection.selected_model_name,
        "selected_ablation": config.shadow_selection.selected_ablation,
        "feature_set_version": whitelist.feature_set_version,
        "features": {
            "base": list(whitelist.base_features),
            "contextual": list(whitelist.contextual_features)
            if config.shadow_selection.selected_ablation.endswith("contextual")
            else [],
        },
        "hyperparameters": dict(config.shadow_selection.selected_hyperparameters),
        "calibration_method": config.shadow_selection.calibration_method,
        "folds_version": config.folds_version,
        "random_seed": config.random_seed,
        "promotion_status": config.shadow_selection.promotion_status,
    }


def _official_fixture_rows(
    live_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_fixture = {str(row.get("source_match_id")): dict(row) for row in live_rows}
    output = []
    for baseline in baseline_rows:
        fixture_id = _require_str(baseline, "source_fixture_id")
        fixture = by_fixture.get(fixture_id)
        if fixture is None:
            raise ShadowContextualError(f"official fixture is missing from live rows: {fixture_id}")
        output.append(_fixture_model_row(fixture))
    return output


def _fixture_model_row(row: Mapping[str, Any]) -> dict[str, Any]:
    prepared = dict(row)
    prepared.update(
        {
            "competition_category": "world_cup",
            "home_advantage_eligible": False,
            "neutral_site": True,
        }
    )
    return prepared


def _assert_same_fixtures(
    baseline_rows: Sequence[Mapping[str, Any]],
    fixture_rows: Sequence[Mapping[str, Any]],
) -> None:
    baseline_ids = [_require_str(row, "source_fixture_id") for row in baseline_rows]
    fixture_ids = [_require_str(row, "source_match_id") for row in fixture_rows]
    if baseline_ids != fixture_ids:
        raise ShadowContextualError("shadow fixtures do not match official baseline fixtures")


def _baseline_cutoff(rows: Sequence[Mapping[str, Any]]) -> datetime | None:
    cutoffs = {_require_datetime(row, "data_cutoff_utc") for row in rows}
    if not cutoffs:
        return None
    if len(cutoffs) != 1:
        raise ShadowContextualError("official baseline contains multiple cutoffs")
    return next(iter(cutoffs))


def _read_baseline_latest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ShadowContextualError(f"official baseline latest Parquet is missing: {path}")
    try:
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
    except (OSError, pa.ArrowInvalid) as exc:
        raise ShadowContextualError(f"failed to read official latest Parquet: {exc}") from exc
    return [dict(row) for row in table.to_pylist()]


def _shadow_history_path(
    shadow_root: Path,
    *,
    created_at: datetime,
    predictions: Sequence[Mapping[str, Any]],
    live_snapshot_checksum: str,
) -> Path:
    checksum = _rows_checksum(predictions) if predictions else live_snapshot_checksum
    token = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    return shadow_root / "history" / f"{token}_{checksum[:12]}.parquet"


def _write_shadow_report(
    predictions: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    cutoff: datetime,
    baseline_fixture_count: int,
    training_matches: int,
    live_finished_2026_matches: int,
    challenger_config: ContextualChallengerConfig,
) -> None:
    lines = [
        "# Shadow Contextual Challenger Predictions",
        "",
        f"Data cutoff UTC: {cutoff.isoformat()}",
        f"Model: {SHADOW_MODEL_FAMILY} ({challenger_config.shadow_selection.selected_model_name})",
        f"Prediction context: {challenger_config.shadow_prediction_context}",
        f"Official baseline fixtures: {baseline_fixture_count}",
        f"Shadow predictions: {len(predictions)}",
        f"Training matches: {training_matches}",
        f"World Cup 2026 finished matches incorporated: {live_finished_2026_matches}",
        "",
        "## Predictions",
        "",
        _markdown_predictions(predictions),
        "",
        "poisson_goal_v1 remains the official model; these predictions are shadow only.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_predictions(predictions: Sequence[Mapping[str, Any]]) -> str:
    if not predictions:
        return "No official baseline fixtures were eligible at this cutoff."
    lines = [
        "| Kickoff UTC | Home | Away | P(home) | P(draw) | P(away) |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in predictions:
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_value(row["kickoff_utc"]),
                    str(row["home_team_name"]),
                    str(row["away_team_name"]),
                    _format_value(row["probability_home_win"]),
                    _format_value(row["probability_draw"]),
                    _format_value(row["probability_away_win"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _write_shadow_comparison(
    *,
    official_scorecard_path: Path,
    shadow_scorecard_path: Path,
    output_path: Path,
) -> int:
    official = _read_json(official_scorecard_path)
    shadow = _read_json(shadow_scorecard_path)
    official_matches = {
        str(row["source_fixture_id"]): row
        for row in official.get("matches", [])
        if isinstance(row, Mapping)
    }
    shadow_matches = {
        str(row["source_fixture_id"]): row
        for row in shadow.get("matches", [])
        if isinstance(row, Mapping)
    }
    shared_ids = sorted(set(official_matches) & set(shadow_matches))
    lines = [
        "# Shadow Contextual Comparison",
        "",
        f"Shared evaluated fixtures: {len(shared_ids)}",
        "",
    ]
    if not shared_ids:
        lines.extend(
            [
                "No paired prospective observations are available yet.",
                "",
            ]
        )
    else:
        official_losses = [_log_loss_for_match(official_matches[item]) for item in shared_ids]
        shadow_losses = [_log_loss_for_match(shadow_matches[item]) for item in shared_ids]
        deltas = [
            shadow - official
            for shadow, official in zip(shadow_losses, official_losses, strict=True)
        ]
        lines.extend(
            [
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Official log loss | {np.mean(official_losses):.6f} |",
                f"| Shadow log loss | {np.mean(shadow_losses):.6f} |",
                f"| Shadow minus official | {np.mean(deltas):.6f} |",
                "",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return len(shared_ids)


def _log_loss_for_match(row: Mapping[str, Any]) -> float:
    result = _require_str(row, "metric_result_1x2")
    probabilities = {
        "home_win": float(row["probability_home_win"]),
        "draw": float(row["probability_draw"]),
        "away_win": float(row["probability_away_win"]),
    }
    return -math.log(max(probabilities[result], 1e-15))


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, Mapping) else {}


def _require_datetime(row: Mapping[str, Any], field_name: str) -> datetime:
    value = row.get(field_name)
    if isinstance(value, datetime):
        return _require_utc(value)
    if isinstance(value, str) and value:
        return _require_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    raise ShadowContextualError(f"{field_name} is required")


def _require_str(row: Mapping[str, Any], field_name: str) -> str:
    value = row.get(field_name)
    if isinstance(value, str) and value:
        return value
    raise ShadowContextualError(f"{field_name} is required")


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != datetime.min.replace(tzinfo=UTC).utcoffset():
        raise ShadowContextualError("timestamps must be timezone-aware UTC")
    return value.astimezone(UTC)


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _format_value(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
