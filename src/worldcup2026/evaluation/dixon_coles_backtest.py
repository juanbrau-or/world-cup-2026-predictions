"""Walk-forward evaluation for Poisson and Dixon-Coles goal models."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from worldcup2026.evaluation.elo_backtest import (
    AWAY_CLASS,
    DRAW_CLASS,
    HOME_CLASS,
    EloEvaluationConfig,
    build_walk_forward_folds,
)
from worldcup2026.models.dixon_coles import (
    DixonColesGoalModel,
    GoalModelType,
)

CLASS_LABELS = (HOME_CLASS, DRAW_CLASS, AWAY_CLASS)
EPSILON = 1e-15


class DixonColesBacktestError(RuntimeError):
    """Raised when Dixon-Coles evaluation cannot be completed."""


class DixonColesModelConfig(BaseModel):
    """Declarative configuration for fitting the selected goal model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_version: str = "dixon_coles_v1"
    input_matches_path: Path = Path("data/processed/modeling_matches.parquet")
    output_model_path: Path = Path("artifacts/models/dixon_coles/model.json")
    report_output: Path = Path("artifacts/models/dixon_coles/report.json")
    final_holdout_start: date = date(2026, 1, 1)
    model_type: GoalModelType = "dixon_coles"
    time_decay_half_life_days: float | None = 365.0
    regularization_strength: float = 0.05
    max_goals: int = 10

    @model_validator(mode="after")
    def validate_model_config(self) -> Self:
        """Require finite model settings."""

        if not self.model_version.strip():
            msg = "model_version cannot be blank"
            raise ValueError(msg)
        if self.time_decay_half_life_days is not None and self.time_decay_half_life_days <= 0:
            msg = "time_decay_half_life_days must be positive or null"
            raise ValueError(msg)
        if self.regularization_strength < 0:
            msg = "regularization_strength cannot be negative"
            raise ValueError(msg)
        if self.max_goals < 1:
            msg = "max_goals must be at least 1"
            raise ValueError(msg)
        return self


class DixonColesSearchConfig(BaseModel):
    """Search grid for goal-model validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_types: tuple[GoalModelType, ...] = ("poisson", "dixon_coles")
    half_life_days: tuple[float | None, ...] = (180.0, 365.0, 730.0, None)
    regularization_strength: tuple[float, ...] = (0.05,)
    max_goals: int = 10

    @model_validator(mode="after")
    def validate_search(self) -> Self:
        """Require non-empty finite search dimensions."""

        if not self.model_types:
            msg = "model_types cannot be empty"
            raise ValueError(msg)
        if not self.half_life_days:
            msg = "half_life_days cannot be empty"
            raise ValueError(msg)
        for half_life in self.half_life_days:
            if half_life is not None and half_life <= 0:
                msg = "half_life_days values must be positive or null"
                raise ValueError(msg)
        if not self.regularization_strength:
            msg = "regularization_strength cannot be empty"
            raise ValueError(msg)
        if any(value < 0 for value in self.regularization_strength):
            msg = "regularization_strength values cannot be negative"
            raise ValueError(msg)
        if self.max_goals < 1:
            msg = "max_goals must be at least 1"
            raise ValueError(msg)
        return self


class DixonColesEvaluationOutputConfig(BaseModel):
    """Output paths for Dixon-Coles evaluation artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: Path = Path("artifacts/evaluation/dixon_coles")
    selected_config_path: Path = Path("artifacts/evaluation/dixon_coles/selected_config.json")
    search_metrics_path: Path = Path("artifacts/evaluation/dixon_coles/search_metrics.csv")
    metrics_by_fold_path: Path = Path("artifacts/evaluation/dixon_coles/metrics_by_fold.csv")
    comparison_with_elo_path: Path = Path(
        "artifacts/evaluation/dixon_coles/comparison_with_elo.csv"
    )
    out_of_fold_predictions_path: Path = Path(
        "artifacts/evaluation/dixon_coles/predictions_out_of_fold.parquet"
    )
    prospective_2026_predictions_path: Path = Path(
        "artifacts/evaluation/dixon_coles/predictions_2026_prospective.parquet"
    )
    report_path: Path = Path("artifacts/evaluation/dixon_coles/report.md")


class DixonColesEvaluationConfig(BaseModel):
    """Evaluation settings for Poisson and Dixon-Coles models."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_matches_path: Path = Path("data/processed/modeling_matches.parquet")
    random_seed: int = 2026
    search: DixonColesSearchConfig = Field(default_factory=DixonColesSearchConfig)
    outputs: DixonColesEvaluationOutputConfig = Field(
        default_factory=DixonColesEvaluationOutputConfig
    )


class DixonColesEvaluationResult(BaseModel):
    """Return value for programmatic Dixon-Coles evaluation runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_config_path: Path
    search_metrics_path: Path
    metrics_by_fold_path: Path
    comparison_with_elo_path: Path
    out_of_fold_predictions_path: Path
    prospective_2026_predictions_path: Path
    report_path: Path
    selected_model_type: str
    selected_half_life_days: float | None
    validation_log_loss: float
    validation_matches: int
    prospective_2026_matches: int


class DixonColesModelResult(BaseModel):
    """Return value for fitting the selected full goal model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_path: Path
    report_path: Path
    model_type: str
    half_life_days: float | None
    training_matches: int
    teams: int


def load_dixon_coles_config(
    config_path: Path = Path("configs/model.yaml"),
) -> DixonColesModelConfig:
    """Load model-fitting configuration from the shared model config file."""

    config = _read_yaml_mapping(config_path, label="Dixon-Coles model")
    try:
        return DixonColesModelConfig.model_validate(config["dixon_coles"])
    except KeyError as exc:
        msg = f"{config_path} is missing the dixon_coles section"
        raise DixonColesBacktestError(msg) from exc
    except ValidationError as exc:
        msg = f"Dixon-Coles model config {config_path} is invalid: {exc}"
        raise DixonColesBacktestError(msg) from exc


def load_dixon_coles_evaluation_config(
    config_path: Path = Path("configs/model.yaml"),
) -> tuple[DixonColesModelConfig, EloEvaluationConfig, DixonColesEvaluationConfig]:
    """Load Dixon-Coles config and the Elo fold definition it must reuse."""

    config = _read_yaml_mapping(config_path, label="Dixon-Coles evaluation")
    try:
        model_config = DixonColesModelConfig.model_validate(config["dixon_coles"])
        elo_evaluation_config = EloEvaluationConfig.model_validate(config["elo_evaluation"])
        evaluation_config = DixonColesEvaluationConfig.model_validate(
            config["dixon_coles_evaluation"]
        )
    except KeyError as exc:
        msg = f"{config_path} is missing required Dixon-Coles evaluation section: {exc}"
        raise DixonColesBacktestError(msg) from exc
    except ValidationError as exc:
        msg = f"Dixon-Coles evaluation config {config_path} is invalid: {exc}"
        raise DixonColesBacktestError(msg) from exc
    return model_config, elo_evaluation_config, evaluation_config


def run_dixon_coles_model(config: DixonColesModelConfig) -> DixonColesModelResult:
    """Fit the configured model on pre-2026 eligible matches and write parameters."""

    rows = _eligible_rows(_read_modeling_matches(config.input_matches_path))
    train_rows = [row for row in rows if _match_date(row) < config.final_holdout_start]
    model = DixonColesGoalModel(
        model_type=config.model_type,
        half_life_days=config.time_decay_half_life_days,
        max_goals=config.max_goals,
        regularization_strength=config.regularization_strength,
    )
    model.fit(train_rows, cutoff=config.final_holdout_start)
    if model.parameters is None:
        msg = "Dixon-Coles model did not produce fitted parameters"
        raise DixonColesBacktestError(msg)
    payload = {
        "model_version": config.model_version,
        "final_holdout_start": config.final_holdout_start.isoformat(),
        "parameters": model.parameters.to_json_dict(),
        "notes": [
            "The full model excludes 2026 rows from fitting.",
            "Attack and defense effects are constrained to sum to zero.",
        ],
    }
    report = {
        "model_version": config.model_version,
        "model_type": config.model_type,
        "half_life_days": config.time_decay_half_life_days,
        "training_matches": len(train_rows),
        "teams": len(model.parameters.teams),
        "categories": len(model.parameters.categories),
        "attack_sum": sum(model.parameters.attack.values()),
        "defense_sum": sum(model.parameters.defense.values()),
    }
    _write_json(payload, config.output_model_path)
    _write_json(report, config.report_output)
    return DixonColesModelResult(
        model_path=config.output_model_path,
        report_path=config.report_output,
        model_type=config.model_type,
        half_life_days=config.time_decay_half_life_days,
        training_matches=len(train_rows),
        teams=len(model.parameters.teams),
    )


def run_dixon_coles_evaluation(
    model_config: DixonColesModelConfig,
    elo_evaluation_config: EloEvaluationConfig,
    evaluation_config: DixonColesEvaluationConfig,
) -> DixonColesEvaluationResult:
    """Run walk-forward validation using the exact Elo validation folds."""

    del model_config
    rows = _eligible_rows(_read_modeling_matches(evaluation_config.input_matches_path))
    folds = build_walk_forward_folds(rows, elo_evaluation_config.folds)
    parameter_grid = list(_iter_parameter_grid(evaluation_config.search))
    if not parameter_grid:
        msg = "Dixon-Coles parameter grid is empty"
        raise DixonColesBacktestError(msg)

    search_rows: list[dict[str, Any]] = []
    best_predictions: list[dict[str, Any]] = []
    best_key: tuple[float, str, str, float] | None = None
    best_parameters: dict[str, Any] | None = None
    best_score = math.inf

    for parameter_set in parameter_grid:
        fold_predictions: list[dict[str, Any]] = []
        for fold in folds:
            predictions = _predict_fold(rows, fold=fold, parameter_set=parameter_set)
            metrics = _metrics_for_predictions(predictions)
            search_rows.append(
                {
                    "fold": fold["name"],
                    **_parameter_summary(parameter_set),
                    **metrics,
                }
            )
            fold_predictions.extend(predictions)
        fold_metrics = _metrics_by_fold(fold_predictions)
        mean_log_loss = float(np.mean([row["log_loss"] for row in fold_metrics]))
        selection_key = (
            mean_log_loss,
            str(parameter_set["model_type"]),
            _half_life_label(parameter_set["half_life_days"]),
            float(parameter_set["regularization_strength"]),
        )
        if selection_key < (best_key or (math.inf, "", "", math.inf)):
            best_key = selection_key
            best_score = mean_log_loss
            best_predictions = fold_predictions
            best_parameters = parameter_set

    if best_parameters is None:
        msg = "no Dixon-Coles candidate produced predictions"
        raise DixonColesBacktestError(msg)

    selected_fold_metrics = _metrics_by_fold(best_predictions)
    prospective_predictions = _predict_2026(
        rows,
        holdout_start=elo_evaluation_config.final_holdout_start,
        parameter_set=best_parameters,
    )
    comparison_rows = _comparison_with_elo(
        selected_fold_metrics,
        elo_metrics_path=elo_evaluation_config.outputs.metrics_by_fold_path,
    )
    selected_config = {
        "selected_by": "mean_validation_log_loss",
        "selected_model_type": best_parameters["model_type"],
        "selected_half_life_days": best_parameters["half_life_days"],
        "regularization_strength": best_parameters["regularization_strength"],
        "max_goals": best_parameters["max_goals"],
        "validation_log_loss": best_score,
        "validation_folds": [fold["name"] for fold in folds],
        "final_holdout_start": elo_evaluation_config.final_holdout_start.isoformat(),
        "notes": [
            "The exact Elo folds are reused for 2018, 2022, and 2024 validation.",
            "No 2026 result is used to select half-life, model type, or regularization.",
            "score_probabilities_json stores the truncated score matrix; residual_probability "
            "stores mass outside max_goals.",
        ],
    }

    outputs = evaluation_config.outputs
    _ensure_output_dirs(outputs)
    _write_json(selected_config, outputs.selected_config_path)
    _write_csv(search_rows, outputs.search_metrics_path)
    _write_csv(selected_fold_metrics, outputs.metrics_by_fold_path)
    _write_csv(comparison_rows, outputs.comparison_with_elo_path)
    _write_predictions(best_predictions, outputs.out_of_fold_predictions_path)
    _write_predictions(prospective_predictions, outputs.prospective_2026_predictions_path)
    _write_report(
        outputs.report_path,
        selected_config=selected_config,
        selected_fold_metrics=selected_fold_metrics,
        comparison_rows=comparison_rows,
        validation_matches=len(best_predictions),
        prospective_matches=len(prospective_predictions),
    )
    return DixonColesEvaluationResult(
        selected_config_path=outputs.selected_config_path,
        search_metrics_path=outputs.search_metrics_path,
        metrics_by_fold_path=outputs.metrics_by_fold_path,
        comparison_with_elo_path=outputs.comparison_with_elo_path,
        out_of_fold_predictions_path=outputs.out_of_fold_predictions_path,
        prospective_2026_predictions_path=outputs.prospective_2026_predictions_path,
        report_path=outputs.report_path,
        selected_model_type=str(best_parameters["model_type"]),
        selected_half_life_days=best_parameters["half_life_days"],
        validation_log_loss=best_score,
        validation_matches=len(best_predictions),
        prospective_2026_matches=len(prospective_predictions),
    )


def _predict_fold(
    rows: Sequence[Mapping[str, Any]],
    *,
    fold: Mapping[str, Any],
    parameter_set: Mapping[str, Any],
) -> list[dict[str, Any]]:
    train_rows = [row for row in rows if _require_str(row, "match_id") in fold["train_ids"]]
    test_rows = [row for row in rows if _require_str(row, "match_id") in fold["test_ids"]]
    _assert_no_future_training(train_rows, fold_name=str(fold["name"]), cutoff=fold["start"])
    model = _new_model(parameter_set)
    model.fit(train_rows, cutoff=fold["start"])
    return _prediction_rows(
        model,
        test_rows,
        fold_name=str(fold["name"]),
        parameter_set=parameter_set,
    )


def _predict_2026(
    rows: Sequence[Mapping[str, Any]],
    *,
    holdout_start: date,
    parameter_set: Mapping[str, Any],
) -> list[dict[str, Any]]:
    train_rows = [row for row in rows if _match_date(row) < holdout_start]
    test_rows = [row for row in rows if _match_date(row) >= holdout_start]
    if not test_rows:
        return []
    _assert_no_future_training(train_rows, fold_name="prospective_2026", cutoff=holdout_start)
    model = _new_model(parameter_set)
    model.fit(train_rows, cutoff=holdout_start)
    return _prediction_rows(
        model,
        test_rows,
        fold_name="prospective_2026",
        parameter_set=parameter_set,
    )


def _prediction_rows(
    model: DixonColesGoalModel,
    rows: Sequence[Mapping[str, Any]],
    *,
    fold_name: str,
    parameter_set: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        distribution = model.predict_match(row)
        actual_result = _target_class(row)
        probabilities = np.array(
            [
                distribution.prob_home_win,
                distribution.prob_draw,
                distribution.prob_away_win,
            ],
            dtype=float,
        )
        predicted_class = int(np.argmax(probabilities))
        kickoff_iso = _isoformat_or_none(row.get("kickoff_utc"))
        output.append(
            {
                "fold": fold_name,
                "match_id": _require_str(row, "match_id"),
                "match_date": _match_date(row).isoformat(),
                "kickoff_utc": kickoff_iso,
                "data_cutoff_utc": kickoff_iso,
                "home_team_id": _require_str(row, "home_team_id"),
                "away_team_id": _require_str(row, "away_team_id"),
                "competition": _require_str(row, "competition"),
                "competition_category": _require_str(row, "competition_category"),
                "neutral": bool(row.get("neutral_site")),
                "home_advantage_eligible": bool(row.get("home_advantage_eligible")),
                "model_type": parameter_set["model_type"],
                "half_life_days": parameter_set["half_life_days"],
                "regularization_strength": parameter_set["regularization_strength"],
                "max_goals": parameter_set["max_goals"],
                "actual_home_goals": _require_int(row, "home_goals_90"),
                "actual_away_goals": _require_int(row, "away_goals_90"),
                "actual_result": _class_name(actual_result),
                "predicted_result": _class_name(predicted_class),
                "expected_home_goals": distribution.expected_home_goals,
                "expected_away_goals": distribution.expected_away_goals,
                "prob_home_win": distribution.prob_home_win,
                "prob_draw": distribution.prob_draw,
                "prob_away_win": distribution.prob_away_win,
                "modal_score": distribution.modal_score,
                "score_probabilities_json": json.dumps(
                    distribution.score_probabilities,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "score_probability_mass": distribution.score_probability_mass,
                "residual_probability": distribution.residual_probability,
                "goal_log_likelihood": model.score_log_probability(row),
            }
        )
    _assert_valid_prediction_rows(output)
    return output


def _metrics_by_fold(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault(str(row["fold"]), []).append(row)
    return [
        {"fold": fold, **_metrics_for_predictions(rows)}
        for fold, rows in sorted(grouped.items())
    ]


def _metrics_for_predictions(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not predictions:
        msg = "cannot compute metrics for an empty prediction set"
        raise DixonColesBacktestError(msg)
    probabilities = _probability_array(predictions)
    targets = np.asarray([_class_from_name(str(row["actual_result"])) for row in predictions])
    true_probabilities = probabilities[np.arange(len(targets)), targets]
    one_hot = np.eye(len(CLASS_LABELS))[targets]
    goal_log_likelihood = np.asarray(
        [float(row["goal_log_likelihood"]) for row in predictions],
        dtype=float,
    )
    return {
        "matches": len(predictions),
        "log_loss": float(-np.mean(np.log(np.clip(true_probabilities, EPSILON, 1.0)))),
        "brier_score": float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1))),
        "ranked_probability_score": _ranked_probability_score(probabilities, targets),
        "goals_log_likelihood": float(np.mean(goal_log_likelihood)),
        "goals_negative_log_likelihood": float(-np.mean(goal_log_likelihood)),
        "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == targets)),
    }


def _ranked_probability_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    order = [AWAY_CLASS, DRAW_CLASS, HOME_CLASS]
    ordered_probabilities = probabilities[:, order]
    mapping = {AWAY_CLASS: 0, DRAW_CLASS: 1, HOME_CLASS: 2}
    ordered_targets = np.asarray([mapping[int(value)] for value in targets], dtype=int)
    true = np.eye(len(CLASS_LABELS))[ordered_targets]
    return float(
        np.mean(
            np.sum(
                np.square(np.cumsum(ordered_probabilities, axis=1) - np.cumsum(true, axis=1)),
                axis=1,
            )
            / 2
        )
    )


def _comparison_with_elo(
    goal_metrics: Sequence[Mapping[str, Any]],
    *,
    elo_metrics_path: Path,
) -> list[dict[str, Any]]:
    elo_rows = _read_csv_by_fold(elo_metrics_path)
    output: list[dict[str, Any]] = []
    for row in goal_metrics:
        fold = str(row["fold"])
        elo_row = elo_rows.get(fold)
        comparison = {
            "fold": fold,
            "dixon_coles_log_loss": row["log_loss"],
            "dixon_coles_brier_score": row["brier_score"],
            "dixon_coles_ranked_probability_score": row["ranked_probability_score"],
            "dixon_coles_goals_log_likelihood": row["goals_log_likelihood"],
        }
        if elo_row is not None:
            comparison.update(
                {
                    "elo_log_loss": float(elo_row["log_loss"]),
                    "elo_brier_score": float(elo_row["brier_score"]),
                    "elo_ranked_probability_score": float(elo_row["ranked_probability_score"]),
                    "log_loss_delta_vs_elo": row["log_loss"] - float(elo_row["log_loss"]),
                    "brier_delta_vs_elo": row["brier_score"] - float(elo_row["brier_score"]),
                    "rps_delta_vs_elo": row["ranked_probability_score"]
                    - float(elo_row["ranked_probability_score"]),
                }
            )
        output.append(comparison)
    return output


def _read_csv_by_fold(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as file:
        return {str(row["fold"]): dict(row) for row in csv.DictReader(file)}


def _iter_parameter_grid(config: DixonColesSearchConfig) -> Iterable[dict[str, Any]]:
    for model_type in config.model_types:
        for half_life in config.half_life_days:
            for regularization_strength in config.regularization_strength:
                yield {
                    "model_type": model_type,
                    "half_life_days": half_life,
                    "regularization_strength": regularization_strength,
                    "max_goals": config.max_goals,
                }


def _new_model(parameter_set: Mapping[str, Any]) -> DixonColesGoalModel:
    return DixonColesGoalModel(
        model_type=_goal_model_type(parameter_set["model_type"]),
        half_life_days=(
            None
            if parameter_set["half_life_days"] is None
            else float(parameter_set["half_life_days"])
        ),
        max_goals=int(parameter_set["max_goals"]),
        regularization_strength=float(parameter_set["regularization_strength"]),
    )


def _goal_model_type(value: object) -> GoalModelType:
    if value == "poisson":
        return "poisson"
    if value == "dixon_coles":
        return "dixon_coles"
    msg = f"unknown goal model type: {value}"
    raise DixonColesBacktestError(msg)


def _parameter_summary(parameter_set: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_type": parameter_set["model_type"],
        "half_life_days": _half_life_label(parameter_set["half_life_days"]),
        "regularization_strength": parameter_set["regularization_strength"],
        "max_goals": parameter_set["max_goals"],
    }


def _half_life_label(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, int | float):
        return str(float(value))
    return str(value)


def _probability_array(predictions: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [row["prob_home_win"], row["prob_draw"], row["prob_away_win"]]
            for row in predictions
        ],
        dtype=float,
    )


def _assert_valid_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    probabilities = _probability_array(rows)
    if not np.all(np.isfinite(probabilities)):
        msg = "Dixon-Coles probabilities must be finite"
        raise DixonColesBacktestError(msg)
    if np.any(probabilities < -1e-12) or np.any(probabilities > 1 + 1e-12):
        msg = "Dixon-Coles probabilities must be between 0 and 1"
        raise DixonColesBacktestError(msg)
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        msg = "Dixon-Coles probability rows must sum to 1"
        raise DixonColesBacktestError(msg)
    for row in rows:
        mass = float(row["score_probability_mass"])
        residual = float(row["residual_probability"])
        if mass < -1e-12 or residual < -1e-12 or not math.isclose(
            mass + residual,
            1.0,
            abs_tol=1e-9,
        ):
            msg = "score_probability_mass plus residual_probability must equal 1"
            raise DixonColesBacktestError(msg)


def _assert_no_future_training(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    fold_name: str,
    cutoff: date,
) -> None:
    future_rows = [
        _require_str(row, "match_id")
        for row in train_rows
        if _match_date(row) >= cutoff
    ]
    if future_rows:
        msg = (
            f"fold {fold_name} uses training matches on or after {cutoff.isoformat()}: "
            + ", ".join(sorted(future_rows)[:5])
        )
        raise DixonColesBacktestError(msg)


def _read_modeling_matches(path: Path) -> list[dict[str, Any]]:
    try:
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
    except (OSError, pa.ArrowInvalid) as exc:
        msg = f"failed to read modeling matches parquet {path}: {exc}"
        raise DixonColesBacktestError(msg) from exc
    return [dict(row) for row in table.to_pylist()]


def _eligible_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    eligible = [dict(row) for row in rows if row.get("model_eligible") is True]
    eligible.sort(key=lambda row: (_match_date(row), _require_str(row, "match_id")))
    return eligible


def _write_predictions(predictions: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([dict(row) for row in predictions]), path)  # type: ignore[no-untyped-call]


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    *,
    selected_config: Mapping[str, Any],
    selected_fold_metrics: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    validation_matches: int,
    prospective_matches: int,
) -> None:
    lines = [
        "# Dixon-Coles Backtest Report",
        "",
        f"Selected model: `{selected_config['selected_model_type']}`",
        f"Selected half-life days: `{selected_config['selected_half_life_days']}`",
        f"Validation matches: {validation_matches}",
        f"Prospective 2026 matches: {prospective_matches}",
        "",
        "## Fold Metrics",
        "",
        _markdown_table(selected_fold_metrics),
        "",
        "## Comparison With Elo",
        "",
        _markdown_table(comparison_rows),
        "",
        (
            "Validation reuses the Elo temporal folds. The 2026 holdout is not used "
            "for model selection."
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    fields = list(rows[0].keys())
    output = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = [_format_markdown_value(row.get(field)) for field in fields]
        output.append("| " + " | ".join(values) + " |")
    return "\n".join(output)


def _format_markdown_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _ensure_output_dirs(outputs: DixonColesEvaluationOutputConfig) -> None:
    outputs.root.mkdir(parents=True, exist_ok=True)
    for path in (
        outputs.selected_config_path,
        outputs.search_metrics_path,
        outputs.metrics_by_fold_path,
        outputs.comparison_with_elo_path,
        outputs.out_of_fold_predictions_path,
        outputs.prospective_2026_predictions_path,
        outputs.report_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)


def _read_yaml_mapping(config_path: Path, *, label: str) -> dict[str, Any]:
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"failed to read {label} config {config_path}: {exc}"
        raise DixonColesBacktestError(msg) from exc
    except yaml.YAMLError as exc:
        msg = f"failed to parse {label} config {config_path}: {exc}"
        raise DixonColesBacktestError(msg) from exc
    if not isinstance(config, dict):
        msg = f"{label} config {config_path} must contain a YAML mapping"
        raise DixonColesBacktestError(msg)
    return config


def _target_class(row: Mapping[str, Any]) -> int:
    home_goals = _require_int(row, "home_goals_90")
    away_goals = _require_int(row, "away_goals_90")
    if home_goals > away_goals:
        return HOME_CLASS
    if home_goals == away_goals:
        return DRAW_CLASS
    return AWAY_CLASS


def _class_name(class_value: int) -> str:
    if class_value == HOME_CLASS:
        return "home_win"
    if class_value == DRAW_CLASS:
        return "draw"
    if class_value == AWAY_CLASS:
        return "away_win"
    msg = f"unknown class value: {class_value}"
    raise DixonColesBacktestError(msg)


def _class_from_name(name: str) -> int:
    if name == "home_win":
        return HOME_CLASS
    if name == "draw":
        return DRAW_CLASS
    if name == "away_win":
        return AWAY_CLASS
    msg = f"unknown class name: {name}"
    raise DixonColesBacktestError(msg)


def _match_date(row: Mapping[str, Any]) -> date:
    value = row.get("match_date")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    msg = f"expected match_date for match {row.get('match_id')!r}, got {value!r}"
    raise DixonColesBacktestError(msg)


def _isoformat_or_none(value: object) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


def _require_str(row: Mapping[str, Any], field_name: str) -> str:
    value = row.get(field_name)
    if isinstance(value, str) and value:
        return value
    msg = f"required field {field_name} is missing or blank for match {row.get('match_id')!r}"
    raise DixonColesBacktestError(msg)


def _require_int(row: Mapping[str, Any], field_name: str) -> int:
    value = row.get(field_name)
    if isinstance(value, int):
        return value
    if isinstance(value, np.integer):
        return int(value)
    msg = f"required integer field {field_name} is missing for match {row.get('match_id')!r}"
    raise DixonColesBacktestError(msg)
