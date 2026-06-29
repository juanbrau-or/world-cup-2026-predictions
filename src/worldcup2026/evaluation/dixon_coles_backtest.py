"""Walk-forward evaluation for Poisson and Dixon-Coles goal models."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
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
OUT_OF_FOLD_STATUS = "out_of_fold"
HOLDOUT_2026_STATUS = "holdout_2026"
METRIC_NAMES = ("log_loss", "brier_score", "ranked_probability_score")


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


class GoalModelSelectionConfig(BaseModel):
    """Declarative record of the currently selected goal-model configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_model_type: GoalModelType = "poisson"
    selected_half_life_days: float | None = 730.0
    regularization_strength: float = 0.05
    max_goals: int = 10
    folds_version: str = "tournament_validation_v1"
    selected_by: str = "mean_validation_log_loss"
    metrics: tuple[str, ...] = METRIC_NAMES

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """Require a usable declarative selection record."""

        if self.selected_half_life_days is not None and self.selected_half_life_days <= 0:
            msg = "selected_half_life_days must be positive or null"
            raise ValueError(msg)
        if self.regularization_strength < 0:
            msg = "regularization_strength cannot be negative"
            raise ValueError(msg)
        if self.max_goals < 1:
            msg = "max_goals must be at least 1"
            raise ValueError(msg)
        if not self.folds_version.strip():
            msg = "folds_version cannot be blank"
            raise ValueError(msg)
        if not self.metrics:
            msg = "metrics cannot be empty"
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
    fold_report_path: Path = Path("artifacts/evaluation/dixon_coles/fold_report.csv")
    paired_match_comparisons_path: Path = Path(
        "artifacts/evaluation/dixon_coles/paired_match_comparisons.csv"
    )
    paired_comparison_summary_path: Path = Path(
        "artifacts/evaluation/dixon_coles/paired_comparison_summary.csv"
    )
    evaluation_summary_path: Path = Path("artifacts/evaluation/dixon_coles/evaluation_summary.json")
    out_of_fold_predictions_path: Path = Path(
        "artifacts/evaluation/dixon_coles/predictions_out_of_fold.parquet"
    )
    holdout_2026_predictions_path: Path = Path(
        "artifacts/evaluation/dixon_coles/predictions_2026_holdout.parquet"
    )
    report_path: Path = Path("artifacts/evaluation/dixon_coles/report.md")


class DixonColesEvaluationConfig(BaseModel):
    """Evaluation settings for Poisson and Dixon-Coles models."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_matches_path: Path = Path("data/processed/modeling_matches.parquet")
    random_seed: int = 2026
    bootstrap_iterations: int = 10000
    search: DixonColesSearchConfig = Field(default_factory=DixonColesSearchConfig)
    selection: GoalModelSelectionConfig = Field(default_factory=GoalModelSelectionConfig)
    outputs: DixonColesEvaluationOutputConfig = Field(
        default_factory=DixonColesEvaluationOutputConfig
    )

    @model_validator(mode="after")
    def validate_evaluation(self) -> Self:
        """Require a positive bootstrap budget."""

        if self.bootstrap_iterations < 1:
            msg = "bootstrap_iterations must be at least 1"
            raise ValueError(msg)
        return self


class DixonColesEvaluationResult(BaseModel):
    """Return value for programmatic Dixon-Coles evaluation runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_config_path: Path
    search_metrics_path: Path
    metrics_by_fold_path: Path
    comparison_with_elo_path: Path
    fold_report_path: Path
    paired_match_comparisons_path: Path
    paired_comparison_summary_path: Path
    evaluation_summary_path: Path
    out_of_fold_predictions_path: Path
    holdout_2026_predictions_path: Path
    report_path: Path
    selected_model_type: str
    selected_half_life_days: float | None
    validation_log_loss: float
    validation_matches: int
    holdout_2026_matches: int


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
    fold_inventory = _fold_inventory(rows, folds)
    parameter_grid = list(_iter_parameter_grid(evaluation_config.search))
    if not parameter_grid:
        msg = "Dixon-Coles parameter grid is empty"
        raise DixonColesBacktestError(msg)

    search_rows: list[dict[str, Any]] = []
    best_predictions: list[dict[str, Any]] = []
    best_key: tuple[float, str, str, float] | None = None
    best_parameters: dict[str, Any] | None = None
    best_score = math.inf
    best_by_model_type: dict[str, dict[str, Any]] = {}
    generated_at = _generated_at()

    for parameter_set in parameter_grid:
        fold_predictions: list[dict[str, Any]] = []
        for fold in folds:
            predictions = _predict_fold(
                rows,
                fold=fold,
                parameter_set=parameter_set,
                generated_at=generated_at,
            )
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
        model_type = str(parameter_set["model_type"])
        current_for_type = best_by_model_type.get(model_type)
        if current_for_type is None or selection_key < current_for_type["selection_key"]:
            best_by_model_type[model_type] = {
                "selection_key": selection_key,
                "validation_log_loss": mean_log_loss,
                "parameters": parameter_set,
                "predictions": fold_predictions,
                "fold_metrics": fold_metrics,
            }

    if best_parameters is None:
        msg = "no Dixon-Coles candidate produced predictions"
        raise DixonColesBacktestError(msg)
    _assert_required_model_types(best_by_model_type)

    selected_fold_metrics = _metrics_by_fold(best_predictions, fold_inventory=fold_inventory)
    holdout_predictions = _predict_2026(
        rows,
        holdout_start=elo_evaluation_config.final_holdout_start,
        parameter_set=best_parameters,
        generated_at=generated_at,
    )
    elo_predictions = _read_predictions(
        elo_evaluation_config.outputs.out_of_fold_predictions_path,
        model_name="elo",
    )
    model_predictions = {
        "poisson": best_by_model_type["poisson"]["predictions"],
        "dixon_coles": best_by_model_type["dixon_coles"]["predictions"],
        "elo": elo_predictions,
    }
    _assert_same_prediction_matches(model_predictions)
    comparison_rows = _comparison_with_elo(
        model_predictions["poisson"],
        elo_predictions=model_predictions["elo"],
        fold_inventory=fold_inventory,
    )
    fold_report = _fold_report(
        model_predictions=model_predictions,
        fold_inventory=fold_inventory,
    )
    paired_rows = _paired_match_comparisons(model_predictions)
    paired_summary = _paired_comparison_summary(
        paired_rows,
        bootstrap_iterations=evaluation_config.bootstrap_iterations,
        random_seed=evaluation_config.random_seed,
    )
    evaluation_summary = _evaluation_summary(
        selection=evaluation_config.selection,
        selected_config={
            "selected_by": "mean_validation_log_loss",
            "selected_model_type": best_parameters["model_type"],
            "selected_half_life_days": best_parameters["half_life_days"],
            "regularization_strength": best_parameters["regularization_strength"],
            "max_goals": best_parameters["max_goals"],
            "validation_log_loss": best_score,
        },
        fold_inventory=fold_inventory,
        paired_summary=paired_summary,
        validation_matches=len(best_predictions),
        holdout_matches=len(holdout_predictions),
    )
    selected_config = {
        "selected_by": "mean_validation_log_loss",
        "selected_model_type": best_parameters["model_type"],
        "selected_half_life_days": best_parameters["half_life_days"],
        "regularization_strength": best_parameters["regularization_strength"],
        "max_goals": best_parameters["max_goals"],
        "validation_log_loss": best_score,
        "folds_version": evaluation_config.selection.folds_version,
        "validation_folds": [fold["name"] for fold in folds],
        "final_holdout_start": elo_evaluation_config.final_holdout_start.isoformat(),
        "holdout_2026_status": HOLDOUT_2026_STATUS,
        "bootstrap_iterations": evaluation_config.bootstrap_iterations,
        "random_seed": evaluation_config.random_seed,
        "notes": [
            "The exact Elo folds are reused for 2018, 2022, and 2024 validation.",
            "No 2026 result is used to select half-life, model type, or regularization.",
            "2026 rows are a retrospective holdout because observed results are present.",
            "score_probabilities_json stores the truncated score matrix; residual_probability "
            "stores mass outside max_goals.",
        ],
    }

    outputs = evaluation_config.outputs
    _ensure_output_dirs(outputs)
    _write_json(selected_config, outputs.selected_config_path)
    _write_json(evaluation_summary, outputs.evaluation_summary_path)
    _write_csv(search_rows, outputs.search_metrics_path)
    _write_csv(selected_fold_metrics, outputs.metrics_by_fold_path)
    _write_csv(comparison_rows, outputs.comparison_with_elo_path)
    _write_csv(fold_report, outputs.fold_report_path)
    _write_csv(paired_rows, outputs.paired_match_comparisons_path)
    _write_csv(paired_summary, outputs.paired_comparison_summary_path)
    _write_predictions(best_predictions, outputs.out_of_fold_predictions_path)
    _write_predictions(holdout_predictions, outputs.holdout_2026_predictions_path)
    _write_report(
        outputs.report_path,
        selected_config=selected_config,
        fold_report=fold_report,
        comparison_rows=comparison_rows,
        paired_summary=paired_summary,
        validation_matches=len(best_predictions),
        holdout_matches=len(holdout_predictions),
    )
    return DixonColesEvaluationResult(
        selected_config_path=outputs.selected_config_path,
        search_metrics_path=outputs.search_metrics_path,
        metrics_by_fold_path=outputs.metrics_by_fold_path,
        comparison_with_elo_path=outputs.comparison_with_elo_path,
        fold_report_path=outputs.fold_report_path,
        paired_match_comparisons_path=outputs.paired_match_comparisons_path,
        paired_comparison_summary_path=outputs.paired_comparison_summary_path,
        evaluation_summary_path=outputs.evaluation_summary_path,
        out_of_fold_predictions_path=outputs.out_of_fold_predictions_path,
        holdout_2026_predictions_path=outputs.holdout_2026_predictions_path,
        report_path=outputs.report_path,
        selected_model_type=str(best_parameters["model_type"]),
        selected_half_life_days=best_parameters["half_life_days"],
        validation_log_loss=best_score,
        validation_matches=len(best_predictions),
        holdout_2026_matches=len(holdout_predictions),
    )


def _predict_fold(
    rows: Sequence[Mapping[str, Any]],
    *,
    fold: Mapping[str, Any],
    parameter_set: Mapping[str, Any],
    generated_at: str,
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
        data_cutoff=_require_date_mapping(fold, "start").isoformat(),
        generated_at=generated_at,
        prediction_status=OUT_OF_FOLD_STATUS,
    )


def _predict_2026(
    rows: Sequence[Mapping[str, Any]],
    *,
    holdout_start: date,
    parameter_set: Mapping[str, Any],
    generated_at: str,
) -> list[dict[str, Any]]:
    train_rows = [row for row in rows if _match_date(row) < holdout_start]
    test_rows = [row for row in rows if _match_date(row) >= holdout_start]
    if not test_rows:
        return []
    _assert_no_future_training(train_rows, fold_name=HOLDOUT_2026_STATUS, cutoff=holdout_start)
    model = _new_model(parameter_set)
    model.fit(train_rows, cutoff=holdout_start)
    return _prediction_rows(
        model,
        test_rows,
        fold_name=HOLDOUT_2026_STATUS,
        parameter_set=parameter_set,
        data_cutoff=holdout_start.isoformat(),
        generated_at=generated_at,
        prediction_status=HOLDOUT_2026_STATUS,
    )


def _prediction_rows(
    model: DixonColesGoalModel,
    rows: Sequence[Mapping[str, Any]],
    *,
    fold_name: str,
    parameter_set: Mapping[str, Any],
    data_cutoff: str,
    generated_at: str,
    prediction_status: str,
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
                "data_cutoff": data_cutoff,
                "data_cutoff_utc": data_cutoff,
                "generated_at": generated_at,
                "prediction_status": prediction_status,
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


def _metrics_by_fold(
    predictions: Sequence[Mapping[str, Any]],
    *,
    fold_inventory: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault(str(row["fold"]), []).append(row)
    output = []
    for fold, rows in sorted(grouped.items()):
        inventory = dict(fold_inventory.get(fold, {})) if fold_inventory is not None else {}
        output.append({"fold": fold, **inventory, **_metrics_for_predictions(rows)})
    return output


def _metrics_for_predictions(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not predictions:
        msg = "cannot compute metrics for an empty prediction set"
        raise DixonColesBacktestError(msg)
    probabilities = _probability_array(predictions)
    targets = np.asarray([_class_from_name(str(row["actual_result"])) for row in predictions])
    true_probabilities = probabilities[np.arange(len(targets)), targets]
    one_hot = np.eye(len(CLASS_LABELS))[targets]
    metrics = {
        "matches": len(predictions),
        "log_loss": float(-np.mean(np.log(np.clip(true_probabilities, EPSILON, 1.0)))),
        "brier_score": float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1))),
        "ranked_probability_score": _ranked_probability_score(probabilities, targets),
        "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == targets)),
    }
    if all("goal_log_likelihood" in row for row in predictions):
        goal_log_likelihood = np.asarray(
            [float(row["goal_log_likelihood"]) for row in predictions],
            dtype=float,
        )
        metrics.update(
            {
                "goals_log_likelihood": float(np.mean(goal_log_likelihood)),
                "goals_negative_log_likelihood": float(-np.mean(goal_log_likelihood)),
            }
        )
    return metrics


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


def _fold_inventory(
    rows: Sequence[Mapping[str, Any]],
    folds: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows_by_id = {_require_str(row, "match_id"): row for row in rows}
    output: dict[str, dict[str, Any]] = {}
    for fold in folds:
        fold_rows = [rows_by_id[match_id] for match_id in sorted(fold["test_ids"])]
        if not fold_rows:
            msg = f"fold {fold['name']} has no rows for fold inventory"
            raise DixonColesBacktestError(msg)
        competitions = sorted({_require_str(row, "competition") for row in fold_rows})
        output[str(fold["name"])] = {
            "matches": len(fold_rows),
            "date_start": min(_match_date(row) for row in fold_rows).isoformat(),
            "date_end": max(_match_date(row) for row in fold_rows).isoformat(),
            "competitions": "; ".join(competitions),
        }
    return output


def _fold_report(
    *,
    model_predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    fold_inventory: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model_name, predictions in sorted(model_predictions.items()):
        rows = _metrics_by_fold(predictions, fold_inventory=fold_inventory)
        for row in rows:
            output.append(
                {
                    "model": model_name,
                    "fold": row["fold"],
                    "matches": row["matches"],
                    "date_start": row["date_start"],
                    "date_end": row["date_end"],
                    "competitions": row["competitions"],
                    "log_loss": row["log_loss"],
                    "brier_score": row["brier_score"],
                    "ranked_probability_score": row["ranked_probability_score"],
                    "goals_log_likelihood": row.get("goals_log_likelihood", ""),
                }
            )
    return output


def _assert_required_model_types(best_by_model_type: Mapping[str, Mapping[str, Any]]) -> None:
    missing = {"poisson", "dixon_coles"} - set(best_by_model_type)
    if missing:
        msg = "goal-model search did not evaluate required model types: " + ", ".join(
            sorted(missing)
        )
        raise DixonColesBacktestError(msg)


def _read_predictions(path: Path, *, model_name: str) -> list[dict[str, Any]]:
    if not path.is_file():
        msg = f"{model_name} prediction artifact is missing: {path}"
        raise DixonColesBacktestError(msg)
    try:
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
    except (OSError, pa.ArrowInvalid) as exc:
        msg = f"failed to read {model_name} predictions {path}: {exc}"
        raise DixonColesBacktestError(msg) from exc
    rows = [dict(row) for row in table.to_pylist()]
    if not rows:
        msg = f"{model_name} prediction artifact is empty: {path}"
        raise DixonColesBacktestError(msg)
    return rows


def _assert_same_prediction_matches(
    model_predictions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    ids_by_model: dict[str, list[str]] = {}
    fold_by_model: dict[str, dict[str, str]] = {}
    for model_name, predictions in model_predictions.items():
        ids = [_require_str(row, "match_id") for row in predictions]
        duplicate_ids = sorted({match_id for match_id in ids if ids.count(match_id) > 1})
        if duplicate_ids:
            msg = f"{model_name} predictions contain duplicate match_id values: {duplicate_ids[:5]}"
            raise DixonColesBacktestError(msg)
        ids_by_model[model_name] = sorted(ids)
        fold_by_model[model_name] = {
            _require_str(row, "match_id"): str(row["fold"]) for row in predictions
        }
    reference_model = sorted(ids_by_model)[0]
    reference_ids = ids_by_model[reference_model]
    reference_folds = fold_by_model[reference_model]
    for model_name, ids in sorted(ids_by_model.items()):
        if ids != reference_ids:
            missing = sorted(set(reference_ids) - set(ids))[:5]
            extra = sorted(set(ids) - set(reference_ids))[:5]
            msg = (
                f"{model_name} predictions do not match {reference_model} match_id set; "
                f"missing={missing}, extra={extra}"
            )
            raise DixonColesBacktestError(msg)
        if fold_by_model[model_name] != reference_folds:
            msg = f"{model_name} fold assignment differs from {reference_model}"
            raise DixonColesBacktestError(msg)


def _prediction_losses(predictions: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for row in predictions:
        probabilities = np.asarray(
            [row["prob_home_win"], row["prob_draw"], row["prob_away_win"]],
            dtype=float,
        )
        target = _class_from_name(str(row["actual_result"]))
        one_hot = np.eye(len(CLASS_LABELS))[target]
        rps = _ranked_probability_score(probabilities.reshape(1, -1), np.asarray([target]))
        output[_require_str(row, "match_id")] = {
            "log_loss": float(-math.log(max(float(probabilities[target]), EPSILON))),
            "brier_score": float(np.sum(np.square(probabilities - one_hot))),
            "ranked_probability_score": rps,
        }
    return output


def _paired_match_comparisons(
    model_predictions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    pairs = (
        ("poisson", "dixon_coles"),
        ("poisson", "elo"),
        ("dixon_coles", "elo"),
    )
    rows_by_model = {
        model_name: {_require_str(row, "match_id"): row for row in predictions}
        for model_name, predictions in model_predictions.items()
    }
    losses_by_model = {
        model_name: _prediction_losses(predictions)
        for model_name, predictions in model_predictions.items()
    }
    output: list[dict[str, Any]] = []
    for model_a, model_b in pairs:
        for match_id in sorted(rows_by_model[model_a]):
            row = rows_by_model[model_a][match_id]
            base = {
                "pair": f"{model_a}_minus_{model_b}",
                "model_a": model_a,
                "model_b": model_b,
                "match_id": match_id,
                "fold": row["fold"],
                "match_date": row["match_date"],
                "competition": row["competition"],
            }
            for metric in METRIC_NAMES:
                base[f"{metric}_delta"] = (
                    losses_by_model[model_a][match_id][metric]
                    - losses_by_model[model_b][match_id][metric]
                )
            output.append(base)
    return output


def _paired_comparison_summary(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(random_seed)
    output: list[dict[str, Any]] = []
    for pair in sorted({str(row["pair"]) for row in paired_rows}):
        pair_rows = [row for row in paired_rows if row["pair"] == pair]
        for metric in METRIC_NAMES:
            deltas = np.asarray([float(row[f"{metric}_delta"]) for row in pair_rows], dtype=float)
            ci_low, ci_high = _bootstrap_mean_ci(
                deltas,
                iterations=bootstrap_iterations,
                rng=rng,
            )
            output.append(
                {
                    "pair": pair,
                    "metric": metric,
                    "matches": len(deltas),
                    "mean_delta": float(np.mean(deltas)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "bootstrap_iterations": bootstrap_iterations,
                    "random_seed": random_seed,
                }
            )
    return output


def _bootstrap_mean_ci(
    deltas: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if len(deltas) == 0:
        msg = "cannot bootstrap an empty paired comparison"
        raise DixonColesBacktestError(msg)
    sample_indices = rng.integers(0, len(deltas), size=(iterations, len(deltas)))
    means = np.mean(deltas[sample_indices], axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def _comparison_with_elo(
    goal_predictions: Sequence[Mapping[str, Any]],
    *,
    elo_predictions: Sequence[Mapping[str, Any]],
    fold_inventory: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    goal_metrics = _metrics_by_fold(goal_predictions, fold_inventory=fold_inventory)
    elo_metrics = {
        str(row["fold"]): row
        for row in _metrics_by_fold(elo_predictions, fold_inventory=fold_inventory)
    }
    output: list[dict[str, Any]] = []
    for row in goal_metrics:
        fold = str(row["fold"])
        elo_row = elo_metrics[fold]
        output.append(
            {
                "fold": fold,
                "matches": row["matches"],
                "date_start": row["date_start"],
                "date_end": row["date_end"],
                "competitions": row["competitions"],
                "poisson_log_loss": row["log_loss"],
                "poisson_brier_score": row["brier_score"],
                "poisson_ranked_probability_score": row["ranked_probability_score"],
                "poisson_goals_log_likelihood": row["goals_log_likelihood"],
                "elo_log_loss": elo_row["log_loss"],
                "elo_brier_score": elo_row["brier_score"],
                "elo_ranked_probability_score": elo_row["ranked_probability_score"],
                "log_loss_delta_vs_elo": row["log_loss"] - elo_row["log_loss"],
                "brier_delta_vs_elo": row["brier_score"] - elo_row["brier_score"],
                "rps_delta_vs_elo": row["ranked_probability_score"]
                - elo_row["ranked_probability_score"],
            }
        )
    return output


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


def _evaluation_summary(
    *,
    selection: GoalModelSelectionConfig,
    selected_config: Mapping[str, Any],
    fold_inventory: Mapping[str, Mapping[str, Any]],
    paired_summary: Sequence[Mapping[str, Any]],
    validation_matches: int,
    holdout_matches: int,
) -> dict[str, Any]:
    return {
        "artifact_schema": "evaluation_summary_v1",
        "evaluation_sets": {
            OUT_OF_FOLD_STATUS: {
                "matches": validation_matches,
                "definition": "walk-forward tournament validation folds before 2026",
            },
            HOLDOUT_2026_STATUS: {
                "matches": holdout_matches,
                "definition": (
                    "retrospective 2026 holdout with observed results present; "
                    "not used for hyperparameter selection"
                ),
            },
        },
        "declared_selection": selection.model_dump(mode="json"),
        "observed_selection": dict(selected_config),
        "folds": [
            {"fold": fold, **dict(values)}
            for fold, values in sorted(fold_inventory.items())
        ],
        "paired_comparison_summary": [dict(row) for row in paired_summary],
        "artifact_policy": {
            "versionable": [
                "configs/model.yaml",
                "artifacts/evaluation/dixon_coles/evaluation_summary.json",
                "artifacts/evaluation/dixon_coles/paired_comparison_summary.csv",
                "artifacts/evaluation/dixon_coles/fold_report.csv",
                "artifacts/evaluation/dixon_coles/report.md",
            ],
            "regenerable_ignored": [
                "artifacts/evaluation/**/predictions_*.parquet",
                "artifacts/models/**",
            ],
            "regenerate_with": [
                "uv run wc2026 model dixon-coles",
                "uv run wc2026 evaluate dixon-coles",
            ],
        },
    }


def _write_report(
    path: Path,
    *,
    selected_config: Mapping[str, Any],
    fold_report: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    paired_summary: Sequence[Mapping[str, Any]],
    validation_matches: int,
    holdout_matches: int,
) -> None:
    lines = [
        "# Dixon-Coles Backtest Report",
        "",
        f"Selected model: `{selected_config['selected_model_type']}`",
        f"Selected half-life days: `{selected_config['selected_half_life_days']}`",
        f"Validation matches: {validation_matches}",
        f"Holdout 2026 matches: {holdout_matches}",
        "",
        "## Fold Metrics",
        "",
        _markdown_table(fold_report),
        "",
        "## Poisson Comparison With Elo",
        "",
        _markdown_table(comparison_rows),
        "",
        "## Paired Comparison Summary",
        "",
        _markdown_table(paired_summary),
        "",
        (
            "Validation reuses the Elo temporal folds. The 2026 rows are a retrospective "
            "holdout, not prospective predictions, and are not used for model selection."
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
        outputs.fold_report_path,
        outputs.paired_match_comparisons_path,
        outputs.paired_comparison_summary_path,
        outputs.evaluation_summary_path,
        outputs.out_of_fold_predictions_path,
        outputs.holdout_2026_predictions_path,
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


def _generated_at() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _require_date_mapping(row: Mapping[str, Any], field_name: str) -> date:
    value = row.get(field_name)
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    msg = f"required date field {field_name} is missing, got {value!r}"
    raise DixonColesBacktestError(msg)


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
