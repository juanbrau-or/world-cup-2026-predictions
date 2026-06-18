"""Walk-forward backtesting and probability calibration for the Elo engine."""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Protocol, Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa
import pyarrow.parquet as pq
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]

from worldcup2026.features.elo import (
    EloRatingsConfig,
    EloRatingsError,
    MarginOfVictoryConfig,
    RatingRegressionAfterInactivityConfig,
    _rate_matches,
)

HOME_CLASS = 0
DRAW_CLASS = 1
AWAY_CLASS = 2
CLASS_LABELS = (HOME_CLASS, DRAW_CLASS, AWAY_CLASS)
PROBABILITY_COLUMNS = ("prob_home_win", "prob_draw", "prob_away_win")
FEATURE_COLUMNS = (
    "elo_difference_pre",
    "home_advantage_eligible",
    "neutral",
    "competition_category",
)
EPSILON = 1e-15


class BacktestFoldConfig(BaseModel):
    """Tournament-centered validation window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    start: date
    end: date
    competitions: tuple[str, ...]

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        """Require an ordered non-empty fold definition."""

        if not self.name.strip():
            msg = "fold name cannot be blank"
            raise ValueError(msg)
        if self.end < self.start:
            msg = f"fold {self.name} end cannot be earlier than start"
            raise ValueError(msg)
        if not self.competitions:
            msg = f"fold {self.name} must include at least one competition"
            raise ValueError(msg)
        return self


class CompetitionWeightsProfile(BaseModel):
    """Named competition-importance search option."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    weights: Mapping[str, float]


class MarginSearchOption(BaseModel):
    """Named margin-of-victory search option."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    enabled: bool
    goal_difference_weight: float = 0.0


class InactivitySearchOption(BaseModel):
    """Named inactivity-regression search option."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    enabled: bool
    inactivity_days: int = 365
    regression_fraction: float = 0.0


class EloParameterSearchConfig(BaseModel):
    """Small explicit grid for Elo rating parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    k_base: tuple[float, ...] = (15.0, 20.0, 25.0)
    home_advantage: tuple[float, ...] = (50.0, 75.0)
    competition_weight_profiles: tuple[CompetitionWeightsProfile, ...]
    margin_of_victory: tuple[MarginSearchOption, ...] = (
        MarginSearchOption(name="off", enabled=False),
        MarginSearchOption(name="modest", enabled=True, goal_difference_weight=0.15),
    )
    rating_regression_after_inactivity: tuple[InactivitySearchOption, ...] = (
        InactivitySearchOption(name="off", enabled=False),
        InactivitySearchOption(
            name="two_years_10pct",
            enabled=True,
            inactivity_days=730,
            regression_fraction=0.10,
        ),
    )

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        """Require non-empty finite search dimensions."""

        for field_name, values in (
            ("k_base", self.k_base),
            ("home_advantage", self.home_advantage),
        ):
            if not values or any(not math.isfinite(value) for value in values):
                msg = f"{field_name} must contain finite values"
                raise ValueError(msg)
        if not self.competition_weight_profiles:
            msg = "competition_weight_profiles cannot be empty"
            raise ValueError(msg)
        return self


class EloEvaluationOutputConfig(BaseModel):
    """Output paths for Elo evaluation artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: Path = Path("artifacts/evaluation/elo")
    selected_config_path: Path = Path("artifacts/evaluation/elo/selected_config.json")
    search_metrics_path: Path = Path("artifacts/evaluation/elo/search_metrics.csv")
    metrics_by_fold_path: Path = Path("artifacts/evaluation/elo/metrics_by_fold.csv")
    segment_metrics_path: Path = Path("artifacts/evaluation/elo/metrics_by_segment.csv")
    out_of_fold_predictions_path: Path = Path(
        "artifacts/evaluation/elo/predictions_out_of_fold.parquet"
    )
    prospective_2026_predictions_path: Path = Path(
        "artifacts/evaluation/elo/predictions_2026_prospective.parquet"
    )
    calibration_curves_path: Path = Path("artifacts/evaluation/elo/calibration_curves.csv")
    report_path: Path = Path("artifacts/evaluation/elo/report.md")


class EloEvaluationConfig(BaseModel):
    """Declarative configuration for Elo probability backtesting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_matches_path: Path = Path("data/processed/modeling_matches.parquet")
    random_seed: int = 2026
    calibration_bins: int = 10
    final_holdout_start: date = date(2026, 1, 1)
    folds: tuple[BacktestFoldConfig, ...]
    search: EloParameterSearchConfig
    outputs: EloEvaluationOutputConfig = Field(default_factory=EloEvaluationOutputConfig)

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """Require usable folds and calibration bins."""

        if self.calibration_bins < 2:
            msg = "calibration_bins must be at least 2"
            raise ValueError(msg)
        if not self.folds:
            msg = "at least one walk-forward fold is required"
            raise ValueError(msg)
        if any(fold.start >= self.final_holdout_start for fold in self.folds):
            msg = "retrospective validation folds must start before final_holdout_start"
            raise ValueError(msg)
        return self


class EloEvaluationResult(BaseModel):
    """Return value for programmatic Elo evaluation runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_config_path: Path
    search_metrics_path: Path
    metrics_by_fold_path: Path
    segment_metrics_path: Path
    out_of_fold_predictions_path: Path
    prospective_2026_predictions_path: Path
    calibration_curves_path: Path
    report_path: Path
    selected_method: str
    validation_log_loss: float
    validation_matches: int
    prospective_2026_matches: int


class EloEvaluationError(RuntimeError):
    """Raised when Elo backtesting cannot be completed."""


class ProbabilityModel(Protocol):
    """Minimal interface for 1X2 probability estimators."""

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> None:
        """Fit model parameters from pre-match features and historical outcomes."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return probabilities in home/draw/away order."""


class ConstantBinaryModel:
    """Fallback for one-class cumulative targets in small test fixtures."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(features), self.probability)
        return np.column_stack([1 - positive, positive])


class SimpleEloProbabilityModel:
    """Convert Elo expected score to 1X2 probabilities with train-fold draw rate."""

    def __init__(self) -> None:
        self.draw_probability = 0.25

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> None:
        draw_rate = float(np.mean(target == DRAW_CLASS)) if len(target) else self.draw_probability
        self.draw_probability = min(max(draw_rate, 0.05), 0.45)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        elo_diff = features["elo_difference_pre"].to_numpy(dtype=float)
        expected_home_score = 1 / (1 + np.power(10, -elo_diff / 400))
        non_draw = 1 - self.draw_probability
        probabilities = np.column_stack(
            [
                non_draw * expected_home_score,
                np.full(len(features), self.draw_probability),
                non_draw * (1 - expected_home_score),
            ]
        )
        return _normalize_probabilities(probabilities)


class OrdinalLogisticProbabilityModel:
    """Cumulative-logit 1X2 model using pre-match Elo features."""

    def __init__(self, *, random_seed: int) -> None:
        self.random_seed = random_seed
        self.away_or_lower_model: Pipeline | ConstantBinaryModel | None = None
        self.draw_or_lower_model: Pipeline | ConstantBinaryModel | None = None

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> None:
        ordered_target = _home_draw_away_to_ordered(target)
        self.away_or_lower_model = _fit_binary_logistic(
            features,
            (ordered_target <= 0).astype(int),
            random_seed=self.random_seed,
        )
        self.draw_or_lower_model = _fit_binary_logistic(
            features,
            (ordered_target <= 1).astype(int),
            random_seed=self.random_seed,
        )

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.away_or_lower_model is None or self.draw_or_lower_model is None:
            msg = "ordinal logistic model must be fit before prediction"
            raise EloEvaluationError(msg)
        away_cdf = _positive_probability(self.away_or_lower_model, features)
        draw_cdf = _positive_probability(self.draw_or_lower_model, features)
        lower = np.minimum(away_cdf, draw_cdf)
        upper = np.maximum(away_cdf, draw_cdf)
        away = lower
        draw = upper - lower
        home = 1 - upper
        return _normalize_probabilities(np.column_stack([home, draw, away]))


class MultinomialLogisticProbabilityModel:
    """Multinomial logistic 1X2 model using pre-match Elo features."""

    def __init__(self, *, random_seed: int) -> None:
        self.random_seed = random_seed
        self.model: Pipeline | None = None
        self.constant_class: int | None = None

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> None:
        unique_classes = sorted(int(value) for value in np.unique(target))
        if len(unique_classes) == 1:
            self.constant_class = unique_classes[0]
            self.model = None
            return
        self.constant_class = None
        self.model = Pipeline(
            steps=[
                ("preprocess", _feature_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=self.random_seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        self.model.fit(features, target)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.constant_class is not None:
            probabilities = np.zeros((len(features), len(CLASS_LABELS)))
            probabilities[:, self.constant_class] = 1.0
            return probabilities
        if self.model is None:
            msg = "multinomial logistic model must be fit before prediction"
            raise EloEvaluationError(msg)
        raw_probabilities = self.model.predict_proba(features)
        class_values = [int(value) for value in self.model.named_steps["model"].classes_]
        probabilities = np.zeros((len(features), len(CLASS_LABELS)))
        for index, class_value in enumerate(class_values):
            probabilities[:, class_value] = raw_probabilities[:, index]
        return _normalize_probabilities(probabilities)


def load_elo_evaluation_config(
    config_path: Path = Path("configs/model.yaml"),
) -> tuple[EloRatingsConfig, EloEvaluationConfig]:
    """Load base Elo and evaluation configuration from the model config file."""

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"failed to read Elo evaluation config {config_path}: {exc}"
        raise EloEvaluationError(msg) from exc
    except yaml.YAMLError as exc:
        msg = f"failed to parse Elo evaluation config {config_path}: {exc}"
        raise EloEvaluationError(msg) from exc
    if not isinstance(config, dict):
        msg = f"Elo evaluation config {config_path} must contain a YAML mapping"
        raise EloEvaluationError(msg)
    try:
        base_elo = EloRatingsConfig.model_validate(config["elo"])
        evaluation = EloEvaluationConfig.model_validate(config["elo_evaluation"])
    except KeyError as exc:
        msg = f"{config_path} is missing required Elo evaluation section: {exc}"
        raise EloEvaluationError(msg) from exc
    except ValidationError as exc:
        msg = f"Elo evaluation config {config_path} is invalid: {exc}"
        raise EloEvaluationError(msg) from exc
    return base_elo, evaluation


def run_elo_evaluation(
    base_elo_config: EloRatingsConfig,
    evaluation_config: EloEvaluationConfig,
) -> EloEvaluationResult:
    """Run walk-forward validation, select Elo probability config, and write artifacts."""

    rows = _read_modeling_matches(evaluation_config.input_matches_path)
    eligible_rows = _eligible_rows(rows)
    folds = build_walk_forward_folds(eligible_rows, evaluation_config.folds)
    parameter_grid = list(_iter_parameter_grid(evaluation_config.search))
    if not parameter_grid:
        msg = "Elo parameter grid is empty"
        raise EloEvaluationError(msg)

    search_rows: list[dict[str, Any]] = []
    best_predictions: list[dict[str, Any]] = []
    best_score = math.inf
    best_key: tuple[float, str, str] | None = None
    best_parameter_set: dict[str, Any] | None = None
    best_method_name: str | None = None

    for parameter_set in parameter_grid:
        rating_rows_by_id = _rating_rows_by_id(eligible_rows, base_elo_config, parameter_set)
        for method_name in _method_names():
            fold_predictions: list[dict[str, Any]] = []
            fold_metrics: list[dict[str, Any]] = []
            for fold in folds:
                prediction_rows = _predict_fold(
                    eligible_rows,
                    rating_rows_by_id,
                    fold=fold,
                    method_name=method_name,
                    random_seed=evaluation_config.random_seed,
                )
                metrics = _metrics_for_predictions(prediction_rows)
                metrics.update(
                        {
                            "fold": fold["name"],
                            "method": method_name,
                            **_parameter_summary(parameter_set),
                        }
                )
                fold_metrics.append(metrics)
                fold_predictions.extend(prediction_rows)
            mean_log_loss = float(np.mean([row["log_loss"] for row in fold_metrics]))
            mean_brier = float(np.mean([row["brier_score"] for row in fold_metrics]))
            search_rows.extend(fold_metrics)
            selection_key = (mean_log_loss, method_name, json.dumps(parameter_set, sort_keys=True))
            if selection_key < (best_key or (math.inf, "", "")):
                best_score = mean_log_loss
                best_key = selection_key
                best_predictions = fold_predictions
                best_parameter_set = parameter_set
                best_method_name = method_name
                best_mean_brier = mean_brier

    if best_parameter_set is None or best_method_name is None:
        msg = "no Elo evaluation candidate produced predictions"
        raise EloEvaluationError(msg)

    selected_fold_metrics = _metrics_by_fold(best_predictions)
    calibration_curves = _calibration_curves(
        best_predictions,
        bins=evaluation_config.calibration_bins,
        group_fields=("fold",),
    )
    prospective_predictions = _predict_2026(
        eligible_rows,
        base_elo_config,
        best_parameter_set,
        method_name=best_method_name,
        holdout_start=evaluation_config.final_holdout_start,
        random_seed=evaluation_config.random_seed,
    )
    prospective_metrics = (
        _metrics_for_predictions(prospective_predictions) if prospective_predictions else {}
    )
    segment_metrics = _segment_metrics(best_predictions, evaluation_set="validation")
    if prospective_predictions:
        segment_metrics.extend(
            _segment_metrics(prospective_predictions, evaluation_set="prospective_2026")
        )

    selected_config = {
        "selected_by": "mean_validation_log_loss",
        "selected_method": best_method_name,
        "validation_log_loss": best_score,
        "validation_brier_score": best_mean_brier,
        "rating_parameters": _serializable_parameter_set(best_parameter_set),
        "validation_folds": [fold["name"] for fold in folds],
        "final_holdout_start": evaluation_config.final_holdout_start.isoformat(),
        "features": list(FEATURE_COLUMNS),
        "probability_columns": list(PROBABILITY_COLUMNS),
        "notes": [
            "2026 matches are excluded from retrospective parameter selection.",
            "Each fold trains on matches strictly before the fold start date.",
        ],
    }

    outputs = evaluation_config.outputs
    _ensure_output_dirs(outputs)
    _write_json(selected_config, outputs.selected_config_path)
    _write_csv(search_rows, outputs.search_metrics_path)
    _write_csv(selected_fold_metrics, outputs.metrics_by_fold_path)
    _write_csv(segment_metrics, outputs.segment_metrics_path)
    _write_predictions(best_predictions, outputs.out_of_fold_predictions_path)
    _write_predictions(prospective_predictions, outputs.prospective_2026_predictions_path)
    _write_csv(calibration_curves, outputs.calibration_curves_path)
    _write_report(
        outputs.report_path,
        selected_config=selected_config,
        selected_fold_metrics=selected_fold_metrics,
        segment_metrics=segment_metrics,
        prospective_metrics=prospective_metrics,
        retrospective_matches=len(best_predictions),
        prospective_matches=len(prospective_predictions),
    )

    return EloEvaluationResult(
        selected_config_path=outputs.selected_config_path,
        search_metrics_path=outputs.search_metrics_path,
        metrics_by_fold_path=outputs.metrics_by_fold_path,
        segment_metrics_path=outputs.segment_metrics_path,
        out_of_fold_predictions_path=outputs.out_of_fold_predictions_path,
        prospective_2026_predictions_path=outputs.prospective_2026_predictions_path,
        calibration_curves_path=outputs.calibration_curves_path,
        report_path=outputs.report_path,
        selected_method=best_method_name,
        validation_log_loss=best_score,
        validation_matches=len(best_predictions),
        prospective_2026_matches=len(prospective_predictions),
    )


def build_walk_forward_folds(
    rows: Sequence[Mapping[str, Any]],
    fold_configs: Sequence[BacktestFoldConfig],
) -> tuple[dict[str, Any], ...]:
    """Build walk-forward folds and fail if any fold can train on future matches."""

    folds: list[dict[str, Any]] = []
    for config in fold_configs:
        test_ids = {
            _require_str(row, "match_id")
            for row in rows
            if _row_in_fold(row, config)
        }
        train_ids = {
            _require_str(row, "match_id")
            for row in rows
            if _match_date(row) < config.start
        }
        if not test_ids:
            msg = f"fold {config.name} has no validation matches"
            raise EloEvaluationError(msg)
        if not train_ids:
            msg = f"fold {config.name} has no training matches before {config.start.isoformat()}"
            raise EloEvaluationError(msg)
        train_max_date = max(_match_date(row) for row in rows if row["match_id"] in train_ids)
        if train_max_date >= config.start:
            msg = f"fold {config.name} includes future training matches"
            raise EloEvaluationError(msg)
        folds.append(
            {
                "name": config.name,
                "start": config.start,
                "end": config.end,
                "competitions": config.competitions,
                "train_ids": train_ids,
                "test_ids": test_ids,
            }
        )
    return tuple(folds)


def _read_modeling_matches(path: Path) -> list[dict[str, Any]]:
    try:
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
    except (OSError, pa.ArrowInvalid) as exc:
        msg = f"failed to read modeling matches parquet {path}: {exc}"
        raise EloEvaluationError(msg) from exc
    return [dict(row) for row in table.to_pylist()]


def _eligible_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    eligible = [dict(row) for row in rows if row.get("model_eligible") is True]
    eligible.sort(key=lambda row: (_match_date(row), _require_str(row, "match_id")))
    return eligible


def _iter_parameter_grid(config: EloParameterSearchConfig) -> Iterable[dict[str, Any]]:
    for k_base, home_advantage, weights, margin, inactivity in itertools.product(
        config.k_base,
        config.home_advantage,
        config.competition_weight_profiles,
        config.margin_of_victory,
        config.rating_regression_after_inactivity,
    ):
        yield {
            "k_base": k_base,
            "home_advantage": home_advantage,
            "competition_weight_profile": weights.name,
            "competition_importance": dict(weights.weights),
            "margin_of_victory": margin.model_dump(mode="python"),
            "rating_regression_after_inactivity": inactivity.model_dump(mode="python"),
        }


def _rating_rows_by_id(
    rows: Sequence[Mapping[str, Any]],
    base_config: EloRatingsConfig,
    parameter_set: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    config = _elo_config_for_parameters(base_config, parameter_set)
    try:
        rating_rows, _ = _rate_matches(rows, config=config)
    except EloRatingsError as exc:
        msg = f"failed to build Elo ratings for evaluation: {exc}"
        raise EloEvaluationError(msg) from exc
    return {str(row["match_id"]): dict(row) for row in rating_rows}


def _elo_config_for_parameters(
    base_config: EloRatingsConfig,
    parameter_set: Mapping[str, Any],
) -> EloRatingsConfig:
    margin_config = dict(parameter_set["margin_of_victory"])
    margin_config.pop("name", None)
    inactivity_config = dict(parameter_set["rating_regression_after_inactivity"])
    inactivity_config.pop("name", None)
    update = {
        "k_base": parameter_set["k_base"],
        "home_advantage": parameter_set["home_advantage"],
        "competition_importance": parameter_set["competition_importance"],
        "margin_of_victory": MarginOfVictoryConfig.model_validate(margin_config),
        "rating_regression_after_inactivity": (
            RatingRegressionAfterInactivityConfig.model_validate(inactivity_config)
        ),
    }
    return EloRatingsConfig.model_validate(base_config.model_copy(update=update).model_dump())


def _method_names() -> tuple[str, ...]:
    return ("elo_simple", "ordinal_logistic", "multinomial_logistic")


def _predict_fold(
    rows: Sequence[Mapping[str, Any]],
    rating_rows_by_id: Mapping[str, Mapping[str, Any]],
    *,
    fold: Mapping[str, Any],
    method_name: str,
    random_seed: int,
) -> list[dict[str, Any]]:
    train_rows = [
        row
        for row in rows
        if _require_str(row, "match_id") in fold["train_ids"]
        and _require_str(row, "match_id") in rating_rows_by_id
    ]
    test_rows = [
        row
        for row in rows
        if _require_str(row, "match_id") in fold["test_ids"]
        and _require_str(row, "match_id") in rating_rows_by_id
    ]
    _assert_no_future_training(train_rows, fold_name=str(fold["name"]), cutoff=fold["start"])
    model = _new_model(method_name, random_seed=random_seed)
    train_features = _feature_frame(train_rows, rating_rows_by_id)
    train_target = _target_array(train_rows)
    test_features = _feature_frame(test_rows, rating_rows_by_id)
    model.fit(train_features, train_target)
    probabilities = model.predict_proba(test_features)
    _assert_valid_probabilities(probabilities)
    return _prediction_rows(
        test_rows,
        rating_rows_by_id,
        probabilities,
        fold_name=str(fold["name"]),
    )


def _predict_2026(
    rows: Sequence[Mapping[str, Any]],
    base_config: EloRatingsConfig,
    parameter_set: Mapping[str, Any],
    *,
    method_name: str,
    holdout_start: date,
    random_seed: int,
) -> list[dict[str, Any]]:
    rating_rows_by_id = _rating_rows_by_id(rows, base_config, parameter_set)
    train_rows = [
        row
        for row in rows
        if _match_date(row) < holdout_start and _require_str(row, "match_id") in rating_rows_by_id
    ]
    test_rows = [
        row
        for row in rows
        if _match_date(row) >= holdout_start and _require_str(row, "match_id") in rating_rows_by_id
    ]
    if not test_rows:
        return []
    _assert_no_future_training(train_rows, fold_name="prospective_2026", cutoff=holdout_start)
    model = _new_model(method_name, random_seed=random_seed)
    model.fit(_feature_frame(train_rows, rating_rows_by_id), _target_array(train_rows))
    probabilities = model.predict_proba(_feature_frame(test_rows, rating_rows_by_id))
    _assert_valid_probabilities(probabilities)
    return _prediction_rows(
        test_rows,
        rating_rows_by_id,
        probabilities,
        fold_name="prospective_2026",
    )


def _new_model(method_name: str, *, random_seed: int) -> ProbabilityModel:
    if method_name == "elo_simple":
        return SimpleEloProbabilityModel()
    if method_name == "ordinal_logistic":
        return OrdinalLogisticProbabilityModel(random_seed=random_seed)
    if method_name == "multinomial_logistic":
        return MultinomialLogisticProbabilityModel(random_seed=random_seed)
    msg = f"unknown Elo probability method: {method_name}"
    raise EloEvaluationError(msg)


def _feature_frame(
    rows: Sequence[Mapping[str, Any]],
    rating_rows_by_id: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    records = []
    for row in rows:
        match_id = _require_str(row, "match_id")
        rating_row = rating_rows_by_id[match_id]
        records.append(
            {
                "elo_difference_pre": float(rating_row["elo_difference_pre"]),
                "home_advantage_eligible": bool(row.get("home_advantage_eligible")),
                "neutral": bool(row.get("neutral_site")),
                "competition_category": _require_str(row, "competition_category"),
            }
        )
    return pd.DataFrame.from_records(records, columns=list(FEATURE_COLUMNS))


def _target_array(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.array([_target_class(row) for row in rows], dtype=int)


def _target_class(row: Mapping[str, Any]) -> int:
    home_goals = _require_int(row, "home_goals_90")
    away_goals = _require_int(row, "away_goals_90")
    if home_goals > away_goals:
        return HOME_CLASS
    if home_goals == away_goals:
        return DRAW_CLASS
    return AWAY_CLASS


def _prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    rating_rows_by_id: Mapping[str, Mapping[str, Any]],
    probabilities: np.ndarray,
    *,
    fold_name: str,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        match_id = _require_str(row, "match_id")
        rating_row = rating_rows_by_id[match_id]
        target = _target_class(row)
        predicted_class = int(np.argmax(probabilities[index]))
        kickoff = row.get("kickoff_utc")
        kickoff_iso = _isoformat_or_none(kickoff)
        output_rows.append(
            {
                "fold": fold_name,
                "match_id": match_id,
                "match_date": _match_date(row).isoformat(),
                "kickoff_utc": kickoff_iso,
                "data_cutoff_utc": kickoff_iso,
                "home_team_id": _require_str(row, "home_team_id"),
                "away_team_id": _require_str(row, "away_team_id"),
                "competition": _require_str(row, "competition"),
                "competition_category": _require_str(row, "competition_category"),
                "official_status": (
                    "friendly" if row.get("competition_category") == "friendly" else "official"
                ),
                "neutral": bool(row.get("neutral_site")),
                "home_advantage_eligible": bool(row.get("home_advantage_eligible")),
                "year": _match_date(row).year,
                "elo_difference_pre": float(rating_row["elo_difference_pre"]),
                "elo_difference_bucket": _elo_difference_bucket(
                    float(rating_row["elo_difference_pre"])
                ),
                "actual_result": _class_name(target),
                "predicted_result": _class_name(predicted_class),
                "prob_home_win": float(probabilities[index, HOME_CLASS]),
                "prob_draw": float(probabilities[index, DRAW_CLASS]),
                "prob_away_win": float(probabilities[index, AWAY_CLASS]),
            }
        )
    return output_rows


def _metrics_by_fold(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault(str(row["fold"]), []).append(row)
    return [
        {"fold": fold, **_metrics_for_predictions(rows)}
        for fold, rows in sorted(grouped.items())
    ]


def _segment_metrics(
    predictions: Sequence[Mapping[str, Any]],
    *,
    evaluation_set: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for segment_name, field_name in (
        ("friendly_official", "official_status"),
        ("neutral", "neutral"),
        ("year", "year"),
        ("elo_difference", "elo_difference_bucket"),
        ("tournament", "competition"),
    ):
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in predictions:
            grouped.setdefault(str(row[field_name]), []).append(row)
        for segment_value, rows in sorted(grouped.items()):
            metrics = _metrics_for_predictions(rows)
            output.append(
                {
                    "evaluation_set": evaluation_set,
                    "segment": segment_name,
                    "segment_value": segment_value,
                    **metrics,
                }
            )
    return output


def _metrics_for_predictions(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not predictions:
        msg = "cannot compute metrics for an empty prediction set"
        raise EloEvaluationError(msg)
    probabilities = _probability_array(predictions)
    targets = np.array([_class_from_name(str(row["actual_result"])) for row in predictions])
    true_probabilities = probabilities[np.arange(len(targets)), targets]
    clipped = np.clip(true_probabilities, EPSILON, 1)
    one_hot = np.eye(len(CLASS_LABELS))[targets]
    predicted = np.argmax(probabilities, axis=1)
    return {
        "matches": len(predictions),
        "log_loss": float(-np.mean(np.log(clipped))),
        "brier_score": float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1))),
        "ranked_probability_score": _ranked_probability_score(probabilities, targets),
        "calibration_error": _calibration_error(probabilities, targets, bins=10),
        "accuracy": float(np.mean(predicted == targets)),
    }


def _probability_array(predictions: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.array(
        [
            [row["prob_home_win"], row["prob_draw"], row["prob_away_win"]]
            for row in predictions
        ],
        dtype=float,
    )


def _ranked_probability_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    order = [AWAY_CLASS, DRAW_CLASS, HOME_CLASS]
    ordered_probabilities = probabilities[:, order]
    ordered_targets = _home_draw_away_to_ordered(targets)
    true = np.eye(len(CLASS_LABELS))[ordered_targets]
    cumulative_probability = np.cumsum(ordered_probabilities, axis=1)
    cumulative_true = np.cumsum(true, axis=1)
    return float(np.mean(np.sum(np.square(cumulative_probability - cumulative_true), axis=1) / 2))


def _calibration_error(probabilities: np.ndarray, targets: np.ndarray, *, bins: int) -> float:
    errors = []
    for class_label in CLASS_LABELS:
        class_probabilities = probabilities[:, class_label]
        class_observed = (targets == class_label).astype(float)
        errors.append(_binary_calibration_error(class_probabilities, class_observed, bins=bins))
    return float(np.mean(errors))


def _binary_calibration_error(
    probabilities: np.ndarray,
    observed: np.ndarray,
    *,
    bins: int,
) -> float:
    total = len(probabilities)
    error = 0.0
    for lower, upper in _bin_edges(bins):
        if upper == 1.0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        bin_gap = abs(np.mean(probabilities[mask]) - np.mean(observed[mask]))
        error += float(np.sum(mask) / total * bin_gap)
    return error


def _calibration_curves(
    predictions: Sequence[Mapping[str, Any]],
    *,
    bins: int,
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in predictions:
        key = tuple(str(row[field]) for field in group_fields)
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        probabilities = _probability_array(rows)
        targets = np.array([_class_from_name(str(row["actual_result"])) for row in rows])
        for class_label, probability_column in zip(
            CLASS_LABELS, PROBABILITY_COLUMNS, strict=True
        ):
            class_probabilities = probabilities[:, class_label]
            observed = (targets == class_label).astype(float)
            for bin_index, (lower, upper) in enumerate(_bin_edges(bins), start=1):
                if upper == 1.0:
                    mask = (class_probabilities >= lower) & (class_probabilities <= upper)
                else:
                    mask = (class_probabilities >= lower) & (class_probabilities < upper)
                if not np.any(mask):
                    continue
                output.append(
                    {
                        **dict(zip(group_fields, key, strict=True)),
                        "class": probability_column.removeprefix("prob_"),
                        "bin": bin_index,
                        "bin_lower": lower,
                        "bin_upper": upper,
                        "matches": int(np.sum(mask)),
                        "mean_predicted_probability": float(np.mean(class_probabilities[mask])),
                        "observed_frequency": float(np.mean(observed[mask])),
                    }
                )
    return output


def _bin_edges(bins: int) -> list[tuple[float, float]]:
    edges = np.linspace(0, 1, bins + 1)
    return [(float(edges[index]), float(edges[index + 1])) for index in range(bins)]


def _feature_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), ["elo_difference_pre"]),
            ("boolean", "passthrough", ["home_advantage_eligible", "neutral"]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["competition_category"],
            ),
        ]
    )


def _fit_binary_logistic(
    features: pd.DataFrame,
    target: np.ndarray,
    *,
    random_seed: int,
) -> Pipeline | ConstantBinaryModel:
    unique = np.unique(target)
    if len(unique) == 1:
        return ConstantBinaryModel(float(unique[0]))
    model = Pipeline(
        steps=[
            ("preprocess", _feature_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(features, target)
    return model


def _positive_probability(
    model: Pipeline | ConstantBinaryModel,
    features: pd.DataFrame,
) -> np.ndarray:
    probabilities = model.predict_proba(features)
    return probabilities[:, 1]


def _home_draw_away_to_ordered(target: np.ndarray) -> np.ndarray:
    mapping = {
        AWAY_CLASS: 0,
        DRAW_CLASS: 1,
        HOME_CLASS: 2,
    }
    return np.array([mapping[int(value)] for value in target], dtype=int)


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 0, 1)
    totals = clipped.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        msg = "probabilities must have positive row sums"
        raise EloEvaluationError(msg)
    normalized: np.ndarray = clipped / totals
    return normalized


def _assert_valid_probabilities(probabilities: np.ndarray) -> None:
    if probabilities.shape[1] != len(CLASS_LABELS):
        msg = "probability matrix must have home/draw/away columns"
        raise EloEvaluationError(msg)
    if not np.all(np.isfinite(probabilities)):
        msg = "probabilities must be finite"
        raise EloEvaluationError(msg)
    if np.any(probabilities < -1e-12) or np.any(probabilities > 1 + 1e-12):
        msg = "probabilities must be between 0 and 1"
        raise EloEvaluationError(msg)
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        msg = "probability rows must sum to 1"
        raise EloEvaluationError(msg)


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
        raise EloEvaluationError(msg)


def _row_in_fold(row: Mapping[str, Any], fold: BacktestFoldConfig) -> bool:
    match_date = _match_date(row)
    return (
        fold.start <= match_date <= fold.end
        and str(row.get("competition")) in set(fold.competitions)
    )


def _parameter_summary(parameter_set: Mapping[str, Any]) -> dict[str, Any]:
    margin = parameter_set["margin_of_victory"]
    inactivity = parameter_set["rating_regression_after_inactivity"]
    return {
        "k_base": parameter_set["k_base"],
        "home_advantage": parameter_set["home_advantage"],
        "competition_weight_profile": parameter_set["competition_weight_profile"],
        "margin_of_victory": margin["name"],
        "inactivity_regression": inactivity["name"],
    }


def _serializable_parameter_set(parameter_set: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "k_base": parameter_set["k_base"],
        "home_advantage": parameter_set["home_advantage"],
        "competition_weight_profile": parameter_set["competition_weight_profile"],
        "competition_importance": dict(parameter_set["competition_importance"]),
        "margin_of_victory": dict(parameter_set["margin_of_victory"]),
        "rating_regression_after_inactivity": dict(
            parameter_set["rating_regression_after_inactivity"]
        ),
    }


def _write_predictions(predictions: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(row) for row in predictions])
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


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
    segment_metrics: Sequence[Mapping[str, Any]],
    prospective_metrics: Mapping[str, Any],
    retrospective_matches: int,
    prospective_matches: int,
) -> None:
    lines = [
        "# Elo Backtest Report",
        "",
        f"Selected method: `{selected_config['selected_method']}`",
        f"Validation matches: {retrospective_matches}",
        f"Prospective 2026 matches: {prospective_matches}",
        "",
        "## Selected Rating Parameters",
        "",
        "```json",
        json.dumps(selected_config["rating_parameters"], indent=2, sort_keys=True),
        "```",
        "",
        "## Fold Metrics",
        "",
        _markdown_table(selected_fold_metrics),
        "",
        "## Prospective 2026 Metrics",
        "",
        _markdown_table([prospective_metrics]) if prospective_metrics else "No 2026 matches found.",
        "",
        "## Segment Metrics",
        "",
        _markdown_table(segment_metrics[:80]),
        "",
        (
            "Validation uses only matches strictly before each fold start. "
            "The 2026 holdout is excluded from parameter selection."
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


def _ensure_output_dirs(outputs: EloEvaluationOutputConfig) -> None:
    outputs.root.mkdir(parents=True, exist_ok=True)
    for path in (
        outputs.selected_config_path,
        outputs.search_metrics_path,
        outputs.metrics_by_fold_path,
        outputs.segment_metrics_path,
        outputs.out_of_fold_predictions_path,
        outputs.prospective_2026_predictions_path,
        outputs.calibration_curves_path,
        outputs.report_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)


def _elo_difference_bucket(value: float) -> str:
    if value < -300:
        return "<-300"
    if value < -150:
        return "-300_to_-150"
    if value < -50:
        return "-150_to_-50"
    if value <= 50:
        return "-50_to_50"
    if value <= 150:
        return "50_to_150"
    if value <= 300:
        return "150_to_300"
    return ">300"


def _class_name(class_value: int) -> str:
    if class_value == HOME_CLASS:
        return "home_win"
    if class_value == DRAW_CLASS:
        return "draw"
    if class_value == AWAY_CLASS:
        return "away_win"
    msg = f"unknown class value: {class_value}"
    raise EloEvaluationError(msg)


def _class_from_name(name: str) -> int:
    if name == "home_win":
        return HOME_CLASS
    if name == "draw":
        return DRAW_CLASS
    if name == "away_win":
        return AWAY_CLASS
    msg = f"unknown class name: {name}"
    raise EloEvaluationError(msg)


def _match_date(row: Mapping[str, Any]) -> date:
    value = row.get("match_date")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    msg = f"expected match_date for match {row.get('match_id')!r}, got {value!r}"
    raise EloEvaluationError(msg)


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
    raise EloEvaluationError(msg)


def _require_int(row: Mapping[str, Any], field_name: str) -> int:
    value = row.get(field_name)
    if isinstance(value, int):
        return value
    msg = f"required integer field {field_name} is missing for match {row.get('match_id')!r}"
    raise EloEvaluationError(msg)
