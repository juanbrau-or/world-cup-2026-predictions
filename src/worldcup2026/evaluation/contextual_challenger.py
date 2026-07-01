"""Temporal contextual challenger evaluation for 1X2 World Cup predictions."""

from __future__ import annotations

import csv
import json
import math
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol, Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa
import pyarrow.parquet as pq
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]

from worldcup2026.evaluation.elo_backtest import (
    AWAY_CLASS,
    DRAW_CLASS,
    HOME_CLASS,
    EloEvaluationConfig,
    build_walk_forward_folds,
)
from worldcup2026.features.elo import EloRatingsConfig, _rate_matches
from worldcup2026.models.dixon_coles import DixonColesGoalModel

CLASS_LABELS = (HOME_CLASS, DRAW_CLASS, AWAY_CLASS)
CLASS_NAMES = ("home_win", "draw", "away_win")
PROBABILITY_COLUMNS = ("prob_home_win", "prob_draw", "prob_away_win")
EPSILON = 1e-15
OUT_OF_FOLD_STATUS = "out_of_fold"
HOLDOUT_2026_STATUS = "holdout_2026"
BASELINE_MODEL_NAME = "poisson_goal_v1"
SANITY_MODEL_NAME = "contextual_logit_v1"
PRIMARY_MODEL_NAME = "contextual_lgbm_v1"
METRIC_NAMES = ("log_loss", "brier_score", "ranked_probability_score")

PROHIBITED_FEATURE_TOKENS = (
    "venue",
    "latitude",
    "longitude",
    "timezone",
    "elevation",
    "travel",
    "weather",
    "lineup",
    "player",
    "injury",
    "odds",
    "home_team_id",
    "away_team_id",
    "home_goals",
    "away_goals",
    "result_90",
    "rest_hours",
    "hours_since_previous_match",
    "minutes_equivalent",
    "previous_match_extra_time",
)


class ContextualChallengerError(RuntimeError):
    """Raised when contextual challenger evaluation cannot be completed safely."""


class TemporalSplitConfig(BaseModel):
    """Nested temporal split settings inside each outer validation fold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tuning_window_days: int = 730
    calibration_window_days: int = 365
    minimum_train_matches: int = 500
    base_oof_blocks: int = 5

    @model_validator(mode="after")
    def validate_split(self) -> Self:
        """Require finite positive split controls."""

        if self.tuning_window_days < 1:
            raise ValueError("tuning_window_days must be positive")
        if self.calibration_window_days < 1:
            raise ValueError("calibration_window_days must be positive")
        if self.minimum_train_matches < 1:
            raise ValueError("minimum_train_matches must be positive")
        if self.base_oof_blocks < 1:
            raise ValueError("base_oof_blocks must be positive")
        return self


class TemperatureCalibrationConfig(BaseModel):
    """Simple multiclass temperature scaling config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["temperature_scaling"] = "temperature_scaling"
    candidate_temperatures: tuple[float, ...] = (1.0,)

    @model_validator(mode="after")
    def validate_temperatures(self) -> Self:
        """Require a non-empty positive deterministic temperature grid."""

        if not self.candidate_temperatures:
            raise ValueError("candidate_temperatures cannot be empty")
        if any(value <= 0 or not math.isfinite(value) for value in self.candidate_temperatures):
            raise ValueError("candidate_temperatures must be positive finite values")
        if 1.0 not in self.candidate_temperatures:
            raise ValueError("candidate_temperatures must include identity temperature 1.0")
        return self


class LogisticGridOption(BaseModel):
    """Regularized multinomial logistic candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    c: float = Field(alias="c")
    max_iter: int = 1000

    @model_validator(mode="after")
    def validate_option(self) -> Self:
        """Require usable logistic hyperparameters."""

        if not self.name.strip():
            raise ValueError("logistic option name cannot be blank")
        if self.c <= 0 or not math.isfinite(self.c):
            raise ValueError("logistic c must be positive and finite")
        if self.max_iter < 100:
            raise ValueError("max_iter must be at least 100")
        return self


class LightGBMGridOption(BaseModel):
    """Small LightGBM multiclass grid option."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    learning_rate: float
    n_estimators: int
    num_leaves: int
    max_depth: int
    min_child_samples: int
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    reg_alpha: float
    reg_lambda: float

    @model_validator(mode="after")
    def validate_option(self) -> Self:
        """Require bounded LightGBM complexity controls."""

        if not self.name.strip():
            raise ValueError("LightGBM option name cannot be blank")
        if self.learning_rate <= 0 or self.learning_rate > 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if self.num_leaves < 2:
            raise ValueError("num_leaves must be at least 2")
        if self.min_child_samples < 1:
            raise ValueError("min_child_samples must be positive")
        if not 0 < self.feature_fraction <= 1:
            raise ValueError("feature_fraction must be in (0, 1]")
        if not 0 < self.bagging_fraction <= 1:
            raise ValueError("bagging_fraction must be in (0, 1]")
        if self.bagging_freq < 0:
            raise ValueError("bagging_freq cannot be negative")
        if self.reg_alpha < 0 or self.reg_lambda < 0:
            raise ValueError("LightGBM regularization cannot be negative")
        return self


class ShadowSelectionConfig(BaseModel):
    """Frozen shadow challenger selection used by operational predictions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_model_name: str
    selected_ablation: str
    selected_feature_set: str
    selected_hyperparameters: Mapping[str, Any]
    calibration_method: str
    training_cutoff: date
    promotion_status: Literal["not_eligible", "shadow_monitoring", "candidate_for_review"]


class ContextualChallengerOutputConfig(BaseModel):
    """Output paths for contextual challenger artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: Path = Path("artifacts/evaluation/contextual_challenger")
    selected_config_path: Path = Path(
        "artifacts/evaluation/contextual_challenger/selected_config.json"
    )
    model_manifest_path: Path = Path(
        "artifacts/evaluation/contextual_challenger/model_manifest.json"
    )
    fold_metrics_path: Path = Path("artifacts/evaluation/contextual_challenger/fold_metrics.csv")
    segment_metrics_path: Path = Path(
        "artifacts/evaluation/contextual_challenger/segment_metrics.csv"
    )
    paired_comparison_path: Path = Path(
        "artifacts/evaluation/contextual_challenger/paired_comparison.csv"
    )
    bootstrap_report_path: Path = Path(
        "artifacts/evaluation/contextual_challenger/bootstrap_report.json"
    )
    calibration_report_path: Path = Path(
        "artifacts/evaluation/contextual_challenger/calibration_report.json"
    )
    ablation_report_path: Path = Path(
        "artifacts/evaluation/contextual_challenger/ablation_report.csv"
    )
    feature_importance_path: Path = Path(
        "artifacts/evaluation/contextual_challenger/feature_importance.csv"
    )
    out_of_fold_predictions_path: Path = Path(
        "artifacts/evaluation/contextual_challenger/predictions_out_of_fold.parquet"
    )
    holdout_2026_predictions_path: Path = Path(
        "artifacts/evaluation/contextual_challenger/predictions_2026_holdout.parquet"
    )
    report_path: Path = Path("artifacts/evaluation/contextual_challenger/report.md")


class ContextualChallengerConfig(BaseModel):
    """Declarative config for temporal contextual challengers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["contextual_challenger_config_v1"]
    official_baseline_model_version: str = BASELINE_MODEL_NAME
    sanity_model_name: str = SANITY_MODEL_NAME
    primary_model_name: str = PRIMARY_MODEL_NAME
    shadow_prediction_context: str = "shadow_contextual_v1"
    input_matches_path: Path = Path("data/processed/modeling_matches.parquet")
    contextual_match_features_path: Path = Path(
        "data/processed/contextual_features/match_contextual_features.parquet"
    )
    feature_whitelist_path: Path = Path("configs/contextual_challenger_features.yaml")
    final_holdout_start: date = date(2026, 1, 1)
    folds_version: str = "tournament_validation_v1"
    random_seed: int = 2026
    bootstrap_iterations: int = 5000
    max_threads: int = 2
    split: TemporalSplitConfig = Field(default_factory=TemporalSplitConfig)
    calibration: TemperatureCalibrationConfig = Field(default_factory=TemperatureCalibrationConfig)
    logistic_grid: tuple[LogisticGridOption, ...]
    lightgbm_grid: tuple[LightGBMGridOption, ...]
    shadow_selection: ShadowSelectionConfig
    outputs: ContextualChallengerOutputConfig = Field(
        default_factory=ContextualChallengerOutputConfig
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """Require explicit small search grids and stable identifiers."""

        if self.bootstrap_iterations < 1:
            raise ValueError("bootstrap_iterations must be positive")
        if self.max_threads < 1:
            raise ValueError("max_threads must be positive")
        if not self.logistic_grid:
            raise ValueError("logistic_grid cannot be empty")
        if not self.lightgbm_grid:
            raise ValueError("lightgbm_grid cannot be empty")
        if not self.folds_version.strip():
            raise ValueError("folds_version cannot be blank")
        if self.official_baseline_model_version != BASELINE_MODEL_NAME:
            raise ValueError("contextual challenger must compare against poisson_goal_v1")
        return self


class FeatureWhitelistConfig(BaseModel):
    """Versioned whitelist of allowed model inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["contextual_challenger_feature_whitelist_v1"]
    feature_set_version: str
    description: str
    base_features: tuple[str, ...]
    contextual_features: tuple[str, ...]
    categorical_features: tuple[str, ...] = ("competition_category",)
    excluded_features: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_whitelist(self) -> Self:
        """Reject prohibited leakage-prone feature names."""

        if not self.feature_set_version.strip():
            raise ValueError("feature_set_version cannot be blank")
        selected = [*self.base_features, *self.contextual_features]
        duplicates = sorted(name for name, count in Counter(selected).items() if count > 1)
        if duplicates:
            raise ValueError("feature whitelist contains duplicates: " + ", ".join(duplicates))
        excluded = set(self.excluded_features)
        leaked = sorted(set(selected) & excluded)
        if leaked:
            raise ValueError("feature whitelist includes excluded features: " + ", ".join(leaked))
        prohibited = sorted(
            feature
            for feature in selected
            for token in PROHIBITED_FEATURE_TOKENS
            if token in feature and not _allowed_prohibited_token_feature(feature)
        )
        if prohibited:
            raise ValueError(
                "feature whitelist includes prohibited names: " + ", ".join(prohibited)
            )
        unknown_categorical = sorted(set(self.categorical_features) - set(selected))
        if unknown_categorical:
            raise ValueError(
                "categorical features must be selected features: " + ", ".join(unknown_categorical)
            )
        return self


def _allowed_prohibited_token_feature(feature: str) -> bool:
    return feature in {"neutral"} or feature.startswith("base_poisson_expected_")


class ContextualChallengerEvaluationResult(BaseModel):
    """Return value for challenger evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_config_path: Path
    model_manifest_path: Path
    fold_metrics_path: Path
    segment_metrics_path: Path
    paired_comparison_path: Path
    bootstrap_report_path: Path
    calibration_report_path: Path
    ablation_report_path: Path
    feature_importance_path: Path
    out_of_fold_predictions_path: Path
    holdout_2026_predictions_path: Path
    report_path: Path
    selected_model_name: str
    selected_ablation: str
    validation_matches: int
    holdout_2026_matches: int
    promotion_status: str


class ContextualChallengerModelResult(BaseModel):
    """Return value for writing the frozen shadow manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_path: Path
    selected_model_name: str
    selected_ablation: str
    feature_set_version: str
    training_cutoff: date


class ProbabilityEstimator(Protocol):
    """Minimal multiclass probability estimator."""

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> None:
        """Fit model parameters."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return probabilities in home/draw/away order."""


@dataclass(frozen=True)
class AblationSpec:
    """One challenger ablation configuration."""

    ablation: str
    model_name: str
    model_kind: Literal["poisson", "logistic", "lightgbm"]
    include_contextual: bool


@dataclass(frozen=True)
class FoldSplit:
    """Nested temporal split for one outer fold."""

    train_rows: tuple[dict[str, Any], ...]
    tune_rows: tuple[dict[str, Any], ...]
    calibration_rows: tuple[dict[str, Any], ...]
    test_rows: tuple[dict[str, Any], ...]
    tune_start: date
    calibration_start: date


@dataclass(frozen=True)
class TemperatureScaler:
    """Fitted temperature scaling transform."""

    method: Literal["identity", "temperature_scaling"]
    temperature: float
    log_loss_before: float
    log_loss_after: float

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply the selected temperature transform."""

        if self.method == "identity" or self.temperature == 1.0:
            return _normalize_probabilities(probabilities)
        clipped = np.clip(probabilities, EPSILON, 1.0)
        scaled = np.power(clipped, 1.0 / self.temperature)
        return _normalize_probabilities(scaled)


class ConstantProbabilityEstimator:
    """Fallback estimator for folds with a single observed class."""

    def __init__(self) -> None:
        self.probabilities: np.ndarray | None = None

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> None:
        del features
        counts = np.bincount(target, minlength=len(CLASS_LABELS)).astype(float)
        if counts.sum() == 0:
            self.probabilities = np.full(len(CLASS_LABELS), 1 / len(CLASS_LABELS))
        else:
            self.probabilities = counts / counts.sum()

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.probabilities is None:
            raise ContextualChallengerError("constant estimator must be fit before prediction")
        return np.tile(self.probabilities, (len(features), 1))


class PipelineProbabilityEstimator:
    """Sklearn pipeline wrapper that aligns missing class columns."""

    def __init__(self, *, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> None:
        self.pipeline.fit(features, target)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "X does not have valid feature names, "
                    "but LGBMClassifier was fitted with feature names"
                ),
                category=UserWarning,
            )
            raw = self.pipeline.predict_proba(features)
        model = self.pipeline.named_steps["model"]
        class_values = [int(value) for value in model.classes_]
        probabilities = np.zeros((len(features), len(CLASS_LABELS)), dtype=float)
        for index, class_value in enumerate(class_values):
            probabilities[:, class_value] = raw[:, index]
        return _normalize_probabilities(probabilities)

    def transformed_feature_names(self) -> list[str]:
        """Return transformed feature names when the preprocessor exposes them."""

        preprocessor = self.pipeline.named_steps["preprocess"]
        names = preprocessor.get_feature_names_out()
        return [str(name) for name in names]

    def lightgbm_importances(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return LightGBM gain and split importances if this is an LGBM pipeline."""

        model = self.pipeline.named_steps["model"]
        booster = getattr(model, "booster_", None)
        if booster is None:
            return None
        gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
        split = np.asarray(booster.feature_importance(importance_type="split"), dtype=float)
        return gain, split


def load_contextual_challenger_config(
    config_path: Path = Path("configs/model.yaml"),
) -> tuple[Any, EloRatingsConfig, EloEvaluationConfig, ContextualChallengerConfig]:
    """Load goal, Elo, fold and contextual challenger configuration."""

    payload = _read_yaml_mapping(config_path, label="contextual challenger")
    try:
        from worldcup2026.evaluation.dixon_coles_backtest import DixonColesModelConfig

        goal_config = DixonColesModelConfig.model_validate(payload["dixon_coles"])
        elo_config = EloRatingsConfig.model_validate(payload["elo"])
        elo_evaluation_config = EloEvaluationConfig.model_validate(payload["elo_evaluation"])
        challenger_config = ContextualChallengerConfig.model_validate(
            payload["contextual_challenger"]
        )
    except KeyError as exc:
        msg = f"{config_path} is missing required contextual challenger section: {exc}"
        raise ContextualChallengerError(msg) from exc
    except ValidationError as exc:
        msg = f"contextual challenger config {config_path} is invalid: {exc}"
        raise ContextualChallengerError(msg) from exc
    return goal_config, elo_config, elo_evaluation_config, challenger_config


def load_feature_whitelist(path: Path) -> FeatureWhitelistConfig:
    """Load the versioned feature whitelist."""

    payload = _read_yaml_mapping(path, label="contextual challenger feature whitelist")
    try:
        return FeatureWhitelistConfig.model_validate(payload)
    except ValidationError as exc:
        msg = f"feature whitelist {path} is invalid: {exc}"
        raise ContextualChallengerError(msg) from exc


def run_contextual_challenger_model(
    *,
    config_path: Path = Path("configs/model.yaml"),
) -> ContextualChallengerModelResult:
    """Write a small manifest for the frozen shadow challenger selection."""

    _, _, elo_evaluation_config, config = load_contextual_challenger_config(config_path)
    whitelist = load_feature_whitelist(config.feature_whitelist_path)
    if config.shadow_selection.training_cutoff >= config.final_holdout_start:
        training_cutoff = config.shadow_selection.training_cutoff
    else:
        training_cutoff = config.final_holdout_start
    manifest = {
        "schema_version": "contextual_challenger_model_manifest_v1",
        "model_name": config.shadow_selection.selected_model_name,
        "model_version": config.shadow_selection.selected_model_name,
        "selected_ablation": config.shadow_selection.selected_ablation,
        "official_baseline_model_version": config.official_baseline_model_version,
        "feature_set_version": whitelist.feature_set_version,
        "base_features": list(whitelist.base_features),
        "contextual_features": list(whitelist.contextual_features),
        "excluded_features": list(whitelist.excluded_features),
        "hyperparameters": dict(config.shadow_selection.selected_hyperparameters),
        "calibration_method": config.shadow_selection.calibration_method,
        "folds_version": elo_evaluation_config.folds_version,
        "training_cutoff": training_cutoff.isoformat(),
        "random_seed": config.random_seed,
        "promotion_status": config.shadow_selection.promotion_status,
        "notes": [
            "This manifest freezes a shadow challenger only; poisson_goal_v1 remains official.",
            "Operational prediction fits as-of at each cutoff and does not retune on 2026 results.",
        ],
    }
    _write_json(manifest, config.outputs.model_manifest_path)
    return ContextualChallengerModelResult(
        manifest_path=config.outputs.model_manifest_path,
        selected_model_name=config.shadow_selection.selected_model_name,
        selected_ablation=config.shadow_selection.selected_ablation,
        feature_set_version=whitelist.feature_set_version,
        training_cutoff=training_cutoff,
    )


def run_contextual_challenger_evaluation(
    *,
    config_path: Path = Path("configs/model.yaml"),
) -> ContextualChallengerEvaluationResult:
    """Run nested temporal evaluation for contextual challengers."""

    goal_config, elo_config, elo_evaluation_config, config = load_contextual_challenger_config(
        config_path
    )
    whitelist = load_feature_whitelist(config.feature_whitelist_path)
    rows = _eligible_rows(_read_parquet_rows(config.input_matches_path, label="modeling matches"))
    contextual_rows = _read_parquet_rows(
        config.contextual_match_features_path,
        label="contextual match features",
    )
    contextual_by_id = _contextual_rows_by_match_id(contextual_rows)
    _assert_contextual_rows_available(rows, contextual_by_id)
    folds = build_walk_forward_folds(rows, elo_evaluation_config.folds)
    _assert_fold_version(config, elo_evaluation_config)
    elo_by_id = _elo_rows_by_id(rows, elo_config=elo_config)
    generated_at = _generated_at()

    all_predictions: list[dict[str, Any]] = []
    calibration_reports: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []

    for fold in folds:
        split = _fold_split(rows, fold=fold, split_config=config.split)
        poisson_by_id = _fold_poisson_features(
            rows,
            split=split,
            goal_config=goal_config,
            split_config=config.split,
        )
        split = FoldSplit(
            train_rows=tuple(
                row for row in split.train_rows if _require_str(row, "match_id") in poisson_by_id
            ),
            tune_rows=split.tune_rows,
            calibration_rows=split.calibration_rows,
            test_rows=tuple(_rows_with_contextual_segments(split.test_rows, contextual_by_id)),
            tune_start=split.tune_start,
            calibration_start=split.calibration_start,
        )
        feature_frames = {
            "train_base": _feature_frame(
                split.train_rows,
                whitelist=whitelist,
                contextual_by_id=contextual_by_id,
                elo_by_id=elo_by_id,
                poisson_by_id=poisson_by_id,
                include_contextual=False,
            ),
            "train_contextual": _feature_frame(
                split.train_rows,
                whitelist=whitelist,
                contextual_by_id=contextual_by_id,
                elo_by_id=elo_by_id,
                poisson_by_id=poisson_by_id,
                include_contextual=True,
            ),
            "tune_base": _feature_frame(
                split.tune_rows,
                whitelist=whitelist,
                contextual_by_id=contextual_by_id,
                elo_by_id=elo_by_id,
                poisson_by_id=poisson_by_id,
                include_contextual=False,
            ),
            "tune_contextual": _feature_frame(
                split.tune_rows,
                whitelist=whitelist,
                contextual_by_id=contextual_by_id,
                elo_by_id=elo_by_id,
                poisson_by_id=poisson_by_id,
                include_contextual=True,
            ),
            "calibration_base": _feature_frame(
                split.calibration_rows,
                whitelist=whitelist,
                contextual_by_id=contextual_by_id,
                elo_by_id=elo_by_id,
                poisson_by_id=poisson_by_id,
                include_contextual=False,
            ),
            "calibration_contextual": _feature_frame(
                split.calibration_rows,
                whitelist=whitelist,
                contextual_by_id=contextual_by_id,
                elo_by_id=elo_by_id,
                poisson_by_id=poisson_by_id,
                include_contextual=True,
            ),
            "test_base": _feature_frame(
                split.test_rows,
                whitelist=whitelist,
                contextual_by_id=contextual_by_id,
                elo_by_id=elo_by_id,
                poisson_by_id=poisson_by_id,
                include_contextual=False,
            ),
            "test_contextual": _feature_frame(
                split.test_rows,
                whitelist=whitelist,
                contextual_by_id=contextual_by_id,
                elo_by_id=elo_by_id,
                poisson_by_id=poisson_by_id,
                include_contextual=True,
            ),
        }
        all_predictions.extend(
            _baseline_prediction_rows(
                split.test_rows,
                poisson_by_id=poisson_by_id,
                fold_name=str(fold["name"]),
                generated_at=generated_at,
            )
        )
        for spec in _ablation_specs(config):
            if spec.model_kind == "poisson":
                continue
            result = _evaluate_ablation_for_fold(
                spec,
                config=config,
                whitelist=whitelist,
                split=split,
                feature_frames=feature_frames,
                fold_name=str(fold["name"]),
                generated_at=generated_at,
            )
            all_predictions.extend(result["predictions"])
            calibration_reports.append(result["calibration_report"])
            search_rows.extend(result["search_rows"])
            importance_rows.extend(result["importance_rows"])

    _assert_same_prediction_matches_by_model(all_predictions)
    fold_metrics = _metrics_by_group(
        all_predictions,
        group_fields=("model_name", "ablation", "fold"),
    )
    aggregate_metrics = _metrics_by_group(all_predictions, group_fields=("model_name", "ablation"))
    segment_metrics = _segment_metrics(all_predictions)
    paired_rows = _paired_match_comparisons(all_predictions)
    bootstrap_report = _bootstrap_report(
        paired_rows,
        iterations=config.bootstrap_iterations,
        random_seed=config.random_seed,
    )
    ablation_report = _ablation_report(search_rows, fold_metrics, aggregate_metrics)
    selected = _selected_shadow_candidate(ablation_report, search_rows, config)
    selected_config = _selected_config_payload(
        config=config,
        whitelist=whitelist,
        selected=selected,
        folds=folds,
        fold_metrics=fold_metrics,
        bootstrap_report=bootstrap_report,
    )
    holdout_predictions = _holdout_2026_predictions(
        rows,
        goal_config=goal_config,
        elo_config=elo_config,
        contextual_by_id=contextual_by_id,
        whitelist=whitelist,
        config=config,
        selected=selected,
        generated_at=generated_at,
    )
    model_manifest = {
        **selected_config,
        "schema_version": "contextual_challenger_model_manifest_v1",
        "promotion_status": selected["promotion_status"],
    }

    outputs = config.outputs
    _ensure_output_dirs(outputs)
    _write_json(selected_config, outputs.selected_config_path)
    _write_json(model_manifest, outputs.model_manifest_path)
    _write_csv(fold_metrics, outputs.fold_metrics_path)
    _write_csv(segment_metrics, outputs.segment_metrics_path)
    _write_csv(paired_rows, outputs.paired_comparison_path)
    _write_json(bootstrap_report, outputs.bootstrap_report_path)
    _write_json({"folds": calibration_reports}, outputs.calibration_report_path)
    _write_csv(ablation_report, outputs.ablation_report_path)
    _write_csv(importance_rows, outputs.feature_importance_path)
    _write_predictions(all_predictions, outputs.out_of_fold_predictions_path)
    _write_predictions(holdout_predictions, outputs.holdout_2026_predictions_path)
    _write_report(
        outputs.report_path,
        selected_config=selected_config,
        aggregate_metrics=aggregate_metrics,
        fold_metrics=fold_metrics,
        bootstrap_report=bootstrap_report,
        ablation_report=ablation_report,
        holdout_matches=len(holdout_predictions),
    )
    return ContextualChallengerEvaluationResult(
        selected_config_path=outputs.selected_config_path,
        model_manifest_path=outputs.model_manifest_path,
        fold_metrics_path=outputs.fold_metrics_path,
        segment_metrics_path=outputs.segment_metrics_path,
        paired_comparison_path=outputs.paired_comparison_path,
        bootstrap_report_path=outputs.bootstrap_report_path,
        calibration_report_path=outputs.calibration_report_path,
        ablation_report_path=outputs.ablation_report_path,
        feature_importance_path=outputs.feature_importance_path,
        out_of_fold_predictions_path=outputs.out_of_fold_predictions_path,
        holdout_2026_predictions_path=outputs.holdout_2026_predictions_path,
        report_path=outputs.report_path,
        selected_model_name=str(selected["model_name"]),
        selected_ablation=str(selected["ablation"]),
        validation_matches=_unique_matches(all_predictions),
        holdout_2026_matches=len(holdout_predictions),
        promotion_status=str(selected["promotion_status"]),
    )


def fit_selected_shadow_estimator(
    *,
    rows: Sequence[Mapping[str, Any]],
    contextual_by_id: Mapping[str, Mapping[str, Any]],
    goal_config: Any,
    elo_config: EloRatingsConfig,
    config: ContextualChallengerConfig,
    whitelist: FeatureWhitelistConfig,
) -> tuple[
    ProbabilityEstimator,
    pd.DataFrame,
    Mapping[str, Mapping[str, float]],
    TemperatureScaler,
]:
    """Fit the frozen shadow estimator with leakage-safe base features."""

    selected = config.shadow_selection
    model_kind: Literal["logistic", "lightgbm"] = (
        "lightgbm" if selected.selected_model_name == config.primary_model_name else "logistic"
    )
    return _fit_shadow_estimator(
        rows=rows,
        contextual_by_id=contextual_by_id,
        goal_config=goal_config,
        elo_config=elo_config,
        config=config,
        whitelist=whitelist,
        selected_ablation=selected.selected_ablation,
        model_kind=model_kind,
        hyperparameters=selected.selected_hyperparameters,
    )


def _fit_shadow_estimator(
    *,
    rows: Sequence[Mapping[str, Any]],
    contextual_by_id: Mapping[str, Mapping[str, Any]],
    goal_config: Any,
    elo_config: EloRatingsConfig,
    config: ContextualChallengerConfig,
    whitelist: FeatureWhitelistConfig,
    selected_ablation: str,
    model_kind: Literal["logistic", "lightgbm"],
    hyperparameters: Mapping[str, Any],
) -> tuple[
    ProbabilityEstimator,
    pd.DataFrame,
    Mapping[str, Mapping[str, float]],
    TemperatureScaler,
]:
    eligible_rows = _eligible_rows(rows)
    include_contextual = selected_ablation.endswith("contextual")
    elo_by_id = _elo_rows_by_id(eligible_rows, elo_config=elo_config)
    poisson_by_id = _poisson_oof_training_features(
        eligible_rows,
        goal_config=goal_config,
        split_config=config.split,
    )
    feature_rows = [
        row for row in eligible_rows if _require_str(row, "match_id") in poisson_by_id
    ]
    if len(feature_rows) < config.split.minimum_train_matches:
        msg = "not enough OOF base-feature rows to fit selected shadow challenger"
        raise ContextualChallengerError(msg)
    fit_rows, calibration_rows = _shadow_fit_calibration_split(
        feature_rows,
        split_config=config.split,
    )
    fit_features = _feature_frame(
        fit_rows,
        whitelist=whitelist,
        contextual_by_id=contextual_by_id,
        elo_by_id=elo_by_id,
        poisson_by_id=poisson_by_id,
        include_contextual=include_contextual,
    )
    calibration_features = _feature_frame(
        calibration_rows,
        whitelist=whitelist,
        contextual_by_id=contextual_by_id,
        elo_by_id=elo_by_id,
        poisson_by_id=poisson_by_id,
        include_contextual=include_contextual,
    )
    target = _target_array(fit_rows)
    estimator = _fit_estimator(
        model_kind,
        hyperparameters,
        feature_columns=list(fit_features.columns),
        categorical_features=whitelist.categorical_features,
        random_seed=config.random_seed,
        max_threads=config.max_threads,
        features=fit_features,
        target=target,
    )
    scaler = _fit_temperature_scaler(
        estimator.predict_proba(calibration_features),
        _target_array(calibration_rows),
        config=config.calibration,
    )
    return estimator, fit_features, poisson_by_id, scaler


def _evaluate_ablation_for_fold(
    spec: AblationSpec,
    *,
    config: ContextualChallengerConfig,
    whitelist: FeatureWhitelistConfig,
    split: FoldSplit,
    feature_frames: Mapping[str, pd.DataFrame],
    fold_name: str,
    generated_at: str,
) -> dict[str, Any]:
    prefix = "contextual" if spec.include_contextual else "base"
    train_features = feature_frames[f"train_{prefix}"]
    tune_features = feature_frames[f"tune_{prefix}"]
    calibration_features = feature_frames[f"calibration_{prefix}"]
    test_features = feature_frames[f"test_{prefix}"]
    train_target = _target_array(split.train_rows)
    tune_target = _target_array(split.tune_rows)
    calibration_target = _target_array(split.calibration_rows)
    search_options = _search_options(spec, config)
    search_rows: list[dict[str, Any]] = []
    best_key: tuple[float, str] | None = None
    best_option: Mapping[str, Any] | None = None
    model_kind = _challenger_model_kind(spec)

    for option in search_options:
        estimator = _fit_estimator(
            model_kind,
            option,
            feature_columns=list(train_features.columns),
            categorical_features=whitelist.categorical_features,
            random_seed=config.random_seed,
            max_threads=config.max_threads,
            features=train_features,
            target=train_target,
        )
        tune_probabilities = estimator.predict_proba(tune_features)
        metrics = _metrics_from_arrays(tune_probabilities, tune_target)
        row = {
            "fold": fold_name,
            "ablation": spec.ablation,
            "model_name": spec.model_name,
            "model_kind": spec.model_kind,
            "include_contextual": spec.include_contextual,
            "hyperparameter_name": str(option["name"]),
            "selection_set": "tune",
            **metrics,
        }
        search_rows.append(row)
        key = (float(metrics["log_loss"]), str(option["name"]))
        if best_key is None or key < best_key:
            best_key = key
            best_option = dict(option)

    if best_option is None:
        raise ContextualChallengerError(f"no hyperparameter option selected for {spec.ablation}")

    fit_features = pd.concat([train_features, tune_features], ignore_index=True)
    fit_target = np.concatenate([train_target, tune_target])
    final_estimator = _fit_estimator(
        model_kind,
        best_option,
        feature_columns=list(fit_features.columns),
        categorical_features=whitelist.categorical_features,
        random_seed=config.random_seed,
        max_threads=config.max_threads,
        features=fit_features,
        target=fit_target,
    )
    calibration_raw = final_estimator.predict_proba(calibration_features)
    scaler = _fit_temperature_scaler(
        calibration_raw,
        calibration_target,
        config=config.calibration,
    )
    test_raw = final_estimator.predict_proba(test_features)
    test_probabilities = scaler.transform(test_raw)
    predictions = _model_prediction_rows(
        split.test_rows,
        probabilities=test_probabilities,
        fold_name=fold_name,
        generated_at=generated_at,
        spec=spec,
        hyperparameters=best_option,
        calibration=scaler,
    )
    calibration_report = {
        "fold": fold_name,
        "ablation": spec.ablation,
        "model_name": spec.model_name,
        "method": scaler.method,
        "temperature": scaler.temperature,
        "matches": len(split.calibration_rows),
        "log_loss_before": scaler.log_loss_before,
        "log_loss_after": scaler.log_loss_after,
        "calibration_start": split.calibration_start.isoformat(),
        "calibration_end": (min(_match_date(row) for row in split.test_rows) - timedelta(days=1))
        .isoformat(),
        "hyperparameter_name": str(best_option["name"]),
    }
    importance_rows = _importance_rows(
        final_estimator,
        test_features,
        _target_array(split.test_rows),
        fold_name=fold_name,
        spec=spec,
        random_seed=config.random_seed,
    )
    return {
        "predictions": predictions,
        "calibration_report": calibration_report,
        "search_rows": search_rows,
        "importance_rows": importance_rows,
    }


def _ablation_specs(config: ContextualChallengerConfig) -> tuple[AblationSpec, ...]:
    return (
        AblationSpec("poisson_official", BASELINE_MODEL_NAME, "poisson", False),
        AblationSpec("logistic_stack", config.sanity_model_name, "logistic", False),
        AblationSpec("contextual_logistic", config.sanity_model_name, "logistic", True),
        AblationSpec("lgbm_stack", config.primary_model_name, "lightgbm", False),
        AblationSpec("lgbm_contextual", config.primary_model_name, "lightgbm", True),
    )


def _search_options(
    spec: AblationSpec,
    config: ContextualChallengerConfig,
) -> list[Mapping[str, Any]]:
    if spec.model_kind == "logistic":
        return [option.model_dump(by_alias=True) for option in config.logistic_grid]
    if spec.model_kind == "lightgbm":
        return [option.model_dump() for option in config.lightgbm_grid]
    raise ContextualChallengerError(f"unsupported search model kind: {spec.model_kind}")


def _challenger_model_kind(spec: AblationSpec) -> Literal["logistic", "lightgbm"]:
    if spec.model_kind == "logistic":
        return "logistic"
    if spec.model_kind == "lightgbm":
        return "lightgbm"
    raise ContextualChallengerError(f"{spec.ablation} is not a fitted challenger")


def _fit_estimator(
    model_kind: Literal["logistic", "lightgbm"],
    hyperparameters: Mapping[str, Any],
    *,
    feature_columns: Sequence[str],
    categorical_features: Sequence[str],
    random_seed: int,
    max_threads: int,
    features: pd.DataFrame,
    target: np.ndarray,
) -> ProbabilityEstimator:
    if len(set(int(value) for value in target)) < 2:
        constant_estimator = ConstantProbabilityEstimator()
        constant_estimator.fit(features, target)
        return constant_estimator
    numeric_features = [name for name in feature_columns if name not in set(categorical_features)]
    active_categorical = [name for name in categorical_features if name in feature_columns]
    preprocessor = _preprocessor(numeric_features, active_categorical)
    if model_kind == "logistic":
        model = LogisticRegression(
            C=float(hyperparameters["c"]),
            max_iter=int(hyperparameters.get("max_iter", 1000)),
            random_state=random_seed,
            solver="lbfgs",
        )
    else:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ContextualChallengerError("LightGBM is required for contextual_lgbm_v1") from exc
        model = LGBMClassifier(
            objective="multiclass",
            num_class=len(CLASS_LABELS),
            random_state=random_seed,
            n_jobs=max_threads,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            learning_rate=float(hyperparameters["learning_rate"]),
            n_estimators=int(hyperparameters["n_estimators"]),
            num_leaves=int(hyperparameters["num_leaves"]),
            max_depth=int(hyperparameters["max_depth"]),
            min_child_samples=int(hyperparameters["min_child_samples"]),
            feature_fraction=float(hyperparameters["feature_fraction"]),
            bagging_fraction=float(hyperparameters["bagging_fraction"]),
            bagging_freq=int(hyperparameters["bagging_freq"]),
            reg_alpha=float(hyperparameters["reg_alpha"]),
            reg_lambda=float(hyperparameters["reg_lambda"]),
        )
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    estimator: ProbabilityEstimator = PipelineProbabilityEstimator(pipeline=pipeline)
    estimator.fit(features, target)
    return estimator


def _preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_features),
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(categorical_features),
            )
        )
    if not transformers:
        raise ContextualChallengerError("feature set cannot be empty")
    return ColumnTransformer(transformers=transformers, verbose_feature_names_out=False)


def _fold_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    fold: Mapping[str, Any],
    split_config: TemporalSplitConfig,
) -> FoldSplit:
    fold_start = _require_date_mapping(fold, "start")
    fold_test_ids = set(str(value) for value in fold["test_ids"])
    test_rows = tuple(
        dict(row) for row in rows if _require_str(row, "match_id") in fold_test_ids
    )
    before_fold = [dict(row) for row in rows if _match_date(row) < fold_start]
    calibration_start = fold_start - timedelta(days=split_config.calibration_window_days)
    tune_start = calibration_start - timedelta(days=split_config.tuning_window_days)
    train_rows = [row for row in before_fold if _match_date(row) < tune_start]
    tune_rows = [
        row for row in before_fold if tune_start <= _match_date(row) < calibration_start
    ]
    calibration_rows = [
        row for row in before_fold if calibration_start <= _match_date(row) < fold_start
    ]
    if (
        len(train_rows) < split_config.minimum_train_matches
        or not tune_rows
        or not calibration_rows
    ):
        train_rows, tune_rows, calibration_rows, tune_start, calibration_start = (
            _fallback_temporal_split(before_fold, split_config=split_config)
        )
    if not test_rows:
        raise ContextualChallengerError(f"fold {fold['name']} has no test rows")
    _assert_ordered_split(train_rows, tune_rows, calibration_rows, test_rows, fold_start=fold_start)
    return FoldSplit(
        train_rows=tuple(train_rows),
        tune_rows=tuple(tune_rows),
        calibration_rows=tuple(calibration_rows),
        test_rows=test_rows,
        tune_start=tune_start,
        calibration_start=calibration_start,
    )


def _fallback_temporal_split(
    before_fold: Sequence[dict[str, Any]],
    *,
    split_config: TemporalSplitConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], date, date]:
    if len(before_fold) < max(6, split_config.minimum_train_matches + 2):
        raise ContextualChallengerError("not enough historical rows for nested temporal split")
    ordered = sorted(before_fold, key=lambda row: (_match_date(row), _require_str(row, "match_id")))
    train_end = max(split_config.minimum_train_matches, int(len(ordered) * 0.6))
    tune_end = max(train_end + 1, int(len(ordered) * 0.8))
    tune_end = min(tune_end, len(ordered) - 1)
    train_rows = ordered[:train_end]
    tune_rows = ordered[train_end:tune_end]
    calibration_rows = ordered[tune_end:]
    if not tune_rows or not calibration_rows:
        raise ContextualChallengerError("fallback temporal split could not create tune/calibration")
    tune_start = min(_match_date(row) for row in tune_rows)
    calibration_start = min(_match_date(row) for row in calibration_rows)
    return train_rows, tune_rows, calibration_rows, tune_start, calibration_start


def _shadow_fit_calibration_split(
    rows: Sequence[dict[str, Any]],
    *,
    split_config: TemporalSplitConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (_match_date(row), _require_str(row, "match_id")))
    latest_date = max(_match_date(row) for row in ordered)
    calibration_start = latest_date - timedelta(days=split_config.calibration_window_days)
    fit_rows = [row for row in ordered if _match_date(row) < calibration_start]
    calibration_rows = [row for row in ordered if _match_date(row) >= calibration_start]
    if len(fit_rows) >= split_config.minimum_train_matches and calibration_rows:
        return fit_rows, calibration_rows
    split_index = max(split_config.minimum_train_matches, int(len(ordered) * 0.8))
    split_index = min(split_index, len(ordered) - 1)
    fit_rows = ordered[:split_index]
    calibration_rows = ordered[split_index:]
    if not fit_rows or not calibration_rows:
        raise ContextualChallengerError("shadow calibration split is empty")
    return fit_rows, calibration_rows


def _assert_ordered_split(
    train_rows: Sequence[Mapping[str, Any]],
    tune_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    *,
    fold_start: date,
) -> None:
    if not train_rows or not tune_rows or not calibration_rows:
        raise ContextualChallengerError(
            "nested temporal split has an empty train/tune/calibration block"
        )
    if max(_match_date(row) for row in train_rows) >= min(_match_date(row) for row in tune_rows):
        raise ContextualChallengerError("training block overlaps tune block")
    if max(_match_date(row) for row in tune_rows) >= min(
        _match_date(row) for row in calibration_rows
    ):
        raise ContextualChallengerError("tune block overlaps calibration block")
    if max(_match_date(row) for row in calibration_rows) >= fold_start:
        raise ContextualChallengerError("calibration block overlaps outer test fold")
    if min(_match_date(row) for row in test_rows) < fold_start:
        raise ContextualChallengerError("outer test fold starts before declared fold cutoff")


def _fold_poisson_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: FoldSplit,
    goal_config: Any,
    split_config: TemporalSplitConfig,
) -> dict[str, Mapping[str, float]]:
    features: dict[str, Mapping[str, float]] = {}
    features.update(
        _poisson_oof_training_features(
            split.train_rows,
            goal_config=goal_config,
            split_config=split_config,
        )
    )
    tune_start = min(_match_date(row) for row in split.tune_rows)
    calibration_start = min(_match_date(row) for row in split.calibration_rows)
    test_start = min(_match_date(row) for row in split.test_rows)
    for predict_rows, cutoff in (
        (split.tune_rows, tune_start),
        (split.calibration_rows, calibration_start),
        (split.test_rows, test_start),
    ):
        train_rows = [row for row in rows if _match_date(row) < cutoff]
        features.update(
            _poisson_features_for_prediction_rows(
                train_rows,
                predict_rows,
                cutoff=cutoff,
                goal_config=goal_config,
            )
        )
    train_rows_with_features = [
        row for row in split.train_rows if _require_str(row, "match_id") in features
    ]
    if len(train_rows_with_features) < split_config.minimum_train_matches:
        msg = "not enough OOF Poisson rows remain in training block after leakage-safe filtering"
        raise ContextualChallengerError(msg)
    return features


def _poisson_oof_training_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    goal_config: Any,
    split_config: TemporalSplitConfig,
) -> dict[str, Mapping[str, float]]:
    ordered = sorted(rows, key=lambda row: (_match_date(row), _require_str(row, "match_id")))
    if len(ordered) <= split_config.minimum_train_matches:
        return {}
    remainder = ordered[split_config.minimum_train_matches :]
    chunks = [
        [dict(item) for item in chunk if len(chunk)]
        for chunk in np.array_split(
            np.asarray(remainder, dtype=object),
            split_config.base_oof_blocks,
        )
    ]
    features: dict[str, Mapping[str, float]] = {}
    for chunk in chunks:
        if not chunk:
            continue
        cutoff = min(_match_date(row) for row in chunk)
        train_rows = [row for row in ordered if _match_date(row) < cutoff]
        if len(train_rows) < split_config.minimum_train_matches:
            continue
        features.update(
            _poisson_features_for_prediction_rows(
                train_rows,
                chunk,
                cutoff=cutoff,
                goal_config=goal_config,
            )
        )
    return features


def _poisson_features_for_prediction_rows(
    train_rows: Sequence[Mapping[str, Any]],
    predict_rows: Sequence[Mapping[str, Any]],
    *,
    cutoff: date,
    goal_config: Any,
) -> dict[str, Mapping[str, float]]:
    if not predict_rows:
        return {}
    if not train_rows:
        raise ContextualChallengerError("cannot build Poisson base features without past rows")
    model = DixonColesGoalModel(
        model_type=goal_config.model_type,
        half_life_days=goal_config.time_decay_half_life_days,
        max_goals=goal_config.max_goals,
        regularization_strength=goal_config.regularization_strength,
    )
    model.fit(train_rows, cutoff=cutoff)
    output: dict[str, Mapping[str, float]] = {}
    for row in predict_rows:
        distribution = model.predict_match(row)
        probabilities = np.asarray(
            [distribution.prob_home_win, distribution.prob_draw, distribution.prob_away_win],
            dtype=float,
        )
        logits = _logits(probabilities)
        output[_require_str(row, "match_id")] = {
            "base_poisson_prob_home_win": float(probabilities[HOME_CLASS]),
            "base_poisson_prob_draw": float(probabilities[DRAW_CLASS]),
            "base_poisson_prob_away_win": float(probabilities[AWAY_CLASS]),
            "base_poisson_expected_home_goals": distribution.expected_home_goals,
            "base_poisson_expected_away_goals": distribution.expected_away_goals,
            "base_poisson_logit_home_win": float(logits[HOME_CLASS]),
            "base_poisson_logit_draw": float(logits[DRAW_CLASS]),
            "base_poisson_logit_away_win": float(logits[AWAY_CLASS]),
            "base_poisson_log_home_away_ratio": float(
                math.log(
                    max(probabilities[HOME_CLASS], EPSILON)
                    / max(probabilities[AWAY_CLASS], EPSILON)
                )
            ),
        }
    return output


def _feature_frame(
    rows: Sequence[Mapping[str, Any]],
    *,
    whitelist: FeatureWhitelistConfig,
    contextual_by_id: Mapping[str, Mapping[str, Any]],
    elo_by_id: Mapping[str, Mapping[str, Any]],
    poisson_by_id: Mapping[str, Mapping[str, float]],
    include_contextual: bool,
) -> pd.DataFrame:
    feature_names = _feature_names(whitelist, include_contextual=include_contextual)
    records: list[dict[str, Any]] = []
    usable_rows: list[Mapping[str, Any]] = []
    for row in rows:
        match_id = _require_str(row, "match_id")
        if match_id not in poisson_by_id:
            continue
        if match_id not in contextual_by_id:
            raise ContextualChallengerError(f"contextual features missing for match_id={match_id}")
        if match_id not in elo_by_id:
            raise ContextualChallengerError(f"Elo features missing for match_id={match_id}")
        contextual = contextual_by_id[match_id]
        elo = elo_by_id[match_id]
        record: dict[str, Any] = {
            **dict(poisson_by_id[match_id]),
            "home_elo_pre": float(elo["home_elo_pre"]),
            "away_elo_pre": float(elo["away_elo_pre"]),
            "elo_difference_pre": float(elo["elo_difference_pre"]),
            "neutral": bool(row.get("neutral_site") or contextual.get("is_neutral_venue")),
            "competition_category": _require_str(row, "competition_category"),
        }
        if include_contextual:
            record.update(_contextual_feature_values(contextual))
        records.append({name: _feature_value(record.get(name)) for name in feature_names})
        usable_rows.append(row)
    if len(records) != len(rows):
        missing = len(rows) - len(records)
        if missing and len(records) == 0:
            raise ContextualChallengerError("no rows have complete OOF base features")
    frame = pd.DataFrame.from_records(records, columns=feature_names)
    if len(frame) != len(usable_rows):
        raise ContextualChallengerError("feature frame row count mismatch")
    return frame


def _contextual_feature_values(row: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        "home_rest_days": row.get("home_rest_days"),
        "away_rest_days": row.get("away_rest_days"),
        "home_matches_last_7d": row.get("home_matches_last_7d"),
        "away_matches_last_7d": row.get("away_matches_last_7d"),
        "home_matches_last_14d": row.get("home_matches_last_14d"),
        "away_matches_last_14d": row.get("away_matches_last_14d"),
        "home_matches_last_30d": row.get("home_matches_last_30d"),
        "away_matches_last_30d": row.get("away_matches_last_30d"),
        "home_previous_match_penalty_shootout": row.get("home_previous_match_penalty_shootout"),
        "away_previous_match_penalty_shootout": row.get("away_previous_match_penalty_shootout"),
        "home_consecutive_matches_without_7d_rest": row.get(
            "home_consecutive_matches_without_7d_rest"
        ),
        "away_consecutive_matches_without_7d_rest": row.get(
            "away_consecutive_matches_without_7d_rest"
        ),
        "home_tournament_match_number": row.get("home_tournament_match_number"),
        "away_tournament_match_number": row.get("away_tournament_match_number"),
        "home_is_first_tournament_match": row.get("home_is_first_tournament_match"),
        "away_is_first_tournament_match": row.get("away_is_first_tournament_match"),
    }
    values["rest_days_diff"] = _difference(values["home_rest_days"], values["away_rest_days"])
    values["matches_last_7d_diff"] = _difference(
        values["home_matches_last_7d"],
        values["away_matches_last_7d"],
    )
    values["matches_last_14d_diff"] = _difference(
        values["home_matches_last_14d"],
        values["away_matches_last_14d"],
    )
    values["matches_last_30d_diff"] = _difference(
        values["home_matches_last_30d"],
        values["away_matches_last_30d"],
    )
    values["consecutive_matches_without_7d_rest_diff"] = _difference(
        values["home_consecutive_matches_without_7d_rest"],
        values["away_consecutive_matches_without_7d_rest"],
    )
    return values


def _feature_names(
    whitelist: FeatureWhitelistConfig,
    *,
    include_contextual: bool,
) -> list[str]:
    names = list(whitelist.base_features)
    if include_contextual:
        names.extend(whitelist.contextual_features)
    return names


def _target_array(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([_target_class(row) for row in rows], dtype=int)


def _target_class(row: Mapping[str, Any]) -> int:
    home_goals = _require_int(row, "home_goals_90")
    away_goals = _require_int(row, "away_goals_90")
    if home_goals > away_goals:
        return HOME_CLASS
    if home_goals == away_goals:
        return DRAW_CLASS
    return AWAY_CLASS


def _baseline_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    poisson_by_id: Mapping[str, Mapping[str, float]],
    fold_name: str,
    generated_at: str,
) -> list[dict[str, Any]]:
    probabilities = np.asarray(
        [
            [
                poisson_by_id[_require_str(row, "match_id")]["base_poisson_prob_home_win"],
                poisson_by_id[_require_str(row, "match_id")]["base_poisson_prob_draw"],
                poisson_by_id[_require_str(row, "match_id")]["base_poisson_prob_away_win"],
            ]
            for row in rows
        ],
        dtype=float,
    )
    return _prediction_rows(
        rows,
        probabilities=probabilities,
        fold_name=fold_name,
        generated_at=generated_at,
        model_name=BASELINE_MODEL_NAME,
        ablation="poisson_official",
        model_kind="poisson",
        hyperparameters={"name": "official_poisson_goal_v1"},
        calibration=TemperatureScaler("identity", 1.0, 0.0, 0.0),
    )


def _model_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    probabilities: np.ndarray,
    fold_name: str,
    generated_at: str,
    spec: AblationSpec,
    hyperparameters: Mapping[str, Any],
    calibration: TemperatureScaler,
) -> list[dict[str, Any]]:
    return _prediction_rows(
        rows,
        probabilities=probabilities,
        fold_name=fold_name,
        generated_at=generated_at,
        model_name=spec.model_name,
        ablation=spec.ablation,
        model_kind=spec.model_kind,
        hyperparameters=hyperparameters,
        calibration=calibration,
    )


def _prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    probabilities: np.ndarray,
    fold_name: str,
    generated_at: str,
    model_name: str,
    ablation: str,
    model_kind: str,
    hyperparameters: Mapping[str, Any],
    calibration: TemperatureScaler,
) -> list[dict[str, Any]]:
    _assert_valid_probabilities(probabilities)
    output: list[dict[str, Any]] = []
    data_cutoff = min(_match_date(row) for row in rows).isoformat() if rows else fold_name
    for index, row in enumerate(rows):
        target = _target_class(row)
        predicted_class = int(np.argmax(probabilities[index]))
        match_date = _match_date(row)
        output.append(
            {
                "fold": fold_name,
                "match_id": _require_str(row, "match_id"),
                "match_date": match_date.isoformat(),
                "kickoff_utc": _isoformat_or_none(row.get("kickoff_utc")),
                "data_cutoff_utc": data_cutoff,
                "generated_at": generated_at,
                "prediction_status": OUT_OF_FOLD_STATUS,
                "model_name": model_name,
                "ablation": ablation,
                "model_kind": model_kind,
                "hyperparameter_name": str(hyperparameters.get("name", "")),
                "calibration_method": calibration.method,
                "calibration_temperature": calibration.temperature,
                "home_team_id": _require_str(row, "home_team_id"),
                "away_team_id": _require_str(row, "away_team_id"),
                "competition": _require_str(row, "competition"),
                "competition_category": _require_str(row, "competition_category"),
                "stage": _optional_str(row.get("stage")) or "unknown",
                "year": match_date.year,
                "neutral": bool(row.get("neutral_site")),
                "actual_result": _class_name(target),
                "predicted_result": _class_name(predicted_class),
                "prob_home_win": float(probabilities[index, HOME_CLASS]),
                "prob_draw": float(probabilities[index, DRAW_CLASS]),
                "prob_away_win": float(probabilities[index, AWAY_CLASS]),
                "favorite_segment": "favorite"
                if predicted_class == target
                else "non_favorite",
                "congestion_level": _congestion_level(row),
                "contextual_missingness": _missingness_level(row),
            }
        )
    return output


def _rows_with_contextual_segments(
    rows: Sequence[Mapping[str, Any]],
    contextual_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        match_id = _require_str(row, "match_id")
        contextual = contextual_by_id[match_id]
        output.append({**dict(row), **_contextual_feature_values(contextual)})
    return output


def _holdout_2026_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    goal_config: Any,
    elo_config: EloRatingsConfig,
    contextual_by_id: Mapping[str, Mapping[str, Any]],
    whitelist: FeatureWhitelistConfig,
    config: ContextualChallengerConfig,
    selected: Mapping[str, Any],
    generated_at: str,
) -> list[dict[str, Any]]:
    holdout_rows = [dict(row) for row in rows if _match_date(row) >= config.final_holdout_start]
    if not holdout_rows:
        return []
    training_rows = [dict(row) for row in rows if _match_date(row) < config.final_holdout_start]
    fit_rows = _eligible_rows(training_rows)
    if len(fit_rows) < config.split.minimum_train_matches:
        return []
    model_kind: Literal["logistic", "lightgbm"] = (
        "lightgbm" if selected["model_kind"] == "lightgbm" else "logistic"
    )
    estimator, _, _, scaler = _fit_shadow_estimator(
        rows=fit_rows,
        contextual_by_id=contextual_by_id,
        goal_config=goal_config,
        elo_config=elo_config,
        config=config,
        whitelist=whitelist,
        selected_ablation=str(selected["ablation"]),
        model_kind=model_kind,
        hyperparameters=selected["hyperparameters"],
    )
    poisson_by_id = _poisson_features_for_prediction_rows(
        fit_rows,
        holdout_rows,
        cutoff=config.final_holdout_start,
        goal_config=goal_config,
    )
    elo_by_id = _elo_rows_by_id([*fit_rows, *holdout_rows], elo_config=elo_config)
    include_contextual = str(selected["ablation"]).endswith("contextual")
    features = _feature_frame(
        holdout_rows,
        whitelist=whitelist,
        contextual_by_id=contextual_by_id,
        elo_by_id=elo_by_id,
        poisson_by_id=poisson_by_id,
        include_contextual=include_contextual,
    )
    probabilities = scaler.transform(estimator.predict_proba(features))
    prediction_rows = _prediction_rows(
        holdout_rows,
        probabilities=probabilities,
        fold_name=HOLDOUT_2026_STATUS,
        generated_at=generated_at,
        model_name=str(selected["model_name"]),
        ablation=str(selected["ablation"]),
        model_kind=str(selected["model_kind"]),
        hyperparameters=selected["hyperparameters"],
        calibration=scaler,
    )
    for row in prediction_rows:
        row["prediction_status"] = HOLDOUT_2026_STATUS
    return prediction_rows


def _fit_temperature_scaler(
    probabilities: np.ndarray,
    target: np.ndarray,
    *,
    config: TemperatureCalibrationConfig,
) -> TemperatureScaler:
    identity = _log_loss(probabilities, target)
    best_temperature = 1.0
    best_loss = identity
    for temperature in sorted(config.candidate_temperatures):
        scaled = TemperatureScaler(
            method="temperature_scaling",
            temperature=temperature,
            log_loss_before=identity,
            log_loss_after=identity,
        ).transform(probabilities)
        loss = _log_loss(scaled, target)
        if (loss, temperature) < (best_loss, best_temperature):
            best_loss = loss
            best_temperature = temperature
    if best_loss < identity - 1e-12 and best_temperature != 1.0:
        return TemperatureScaler("temperature_scaling", best_temperature, identity, best_loss)
    return TemperatureScaler("identity", 1.0, identity, identity)


def _metrics_by_group(
    predictions: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        key = tuple(str(row[field]) for field in group_fields)
        grouped[key].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        payload = {field: value for field, value in zip(group_fields, key, strict=True)}
        output.append({**payload, **_metrics_for_predictions(rows)})
    return output


def _metrics_for_predictions(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    probabilities = _probability_array(predictions)
    targets = np.asarray([_class_from_name(str(row["actual_result"])) for row in predictions])
    return _metrics_from_arrays(probabilities, targets)


def _metrics_from_arrays(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    if len(targets) == 0:
        raise ContextualChallengerError("cannot compute metrics for empty target array")
    _assert_valid_probabilities(probabilities)
    true_probabilities = probabilities[np.arange(len(targets)), targets]
    one_hot = np.eye(len(CLASS_LABELS))[targets]
    return {
        "matches": len(targets),
        "log_loss": _log_loss(probabilities, targets),
        "brier_score": float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1))),
        "ranked_probability_score": _ranked_probability_score(probabilities, targets),
        "calibration_error": _calibration_error(probabilities, targets, bins=10),
        "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == targets)),
        "mean_true_probability": float(np.mean(true_probabilities)),
    }


def _segment_metrics(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for segment, field_name in (
        ("fold", "fold"),
        ("year", "year"),
        ("tournament", "competition"),
        ("stage", "stage"),
        ("favorite", "favorite_segment"),
        ("congestion", "congestion_level"),
        ("missingness", "contextual_missingness"),
    ):
        grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in predictions:
            key = (str(row["model_name"]), str(row["ablation"]), str(row[field_name]))
            grouped[key].append(row)
        for (model_name, ablation, value), rows in sorted(grouped.items()):
            output.append(
                {
                    "model_name": model_name,
                    "ablation": ablation,
                    "segment": segment,
                    "segment_value": value,
                    **_metrics_for_predictions(rows),
                }
            )
    return output


def _paired_match_comparisons(
    predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in predictions:
        key = f"{row['model_name']}::{row['ablation']}"
        by_model[str(key)][_require_str(row, "match_id")] = row
    baseline_key = f"{BASELINE_MODEL_NAME}::poisson_official"
    if baseline_key not in by_model:
        raise ContextualChallengerError("poisson baseline predictions are missing")
    baseline_ids = set(by_model[baseline_key])
    output: list[dict[str, Any]] = []
    for model_name, rows_by_id in sorted(by_model.items()):
        if model_name == baseline_key:
            continue
        if set(rows_by_id) != baseline_ids:
            missing = sorted(baseline_ids - set(rows_by_id))[:5]
            extra = sorted(set(rows_by_id) - baseline_ids)[:5]
            msg = (
                f"{model_name} predictions do not match poisson ids; "
                f"missing={missing}, extra={extra}"
            )
            raise ContextualChallengerError(msg)
        for match_id in sorted(baseline_ids):
            base = by_model[baseline_key][match_id]
            challenger = rows_by_id[match_id]
            base_losses = _losses_for_row(base)
            challenger_losses = _losses_for_row(challenger)
            row = {
                "pair": f"{model_name}_minus_{baseline_key}",
                "model_a": challenger["model_name"],
                "ablation_a": challenger["ablation"],
                "model_b": BASELINE_MODEL_NAME,
                "match_id": match_id,
                "fold": base["fold"],
                "match_date": base["match_date"],
                "competition": base["competition"],
            }
            for metric in METRIC_NAMES:
                row[f"{metric}_delta"] = challenger_losses[metric] - base_losses[metric]
            output.append(row)
    return output


def _bootstrap_report(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    for pair in sorted({str(row["pair"]) for row in paired_rows}):
        pair_rows = [row for row in paired_rows if row["pair"] == pair]
        for metric in METRIC_NAMES:
            deltas = np.asarray([float(row[f"{metric}_delta"]) for row in pair_rows], dtype=float)
            if len(deltas) == 0:
                rows.append(
                    {
                        "pair": pair,
                        "metric": metric,
                        "matches": 0,
                        "mean_delta": None,
                        "ci_low": None,
                        "ci_high": None,
                        "proportion_favorable": None,
                    }
                )
                continue
            sampled = rng.integers(0, len(deltas), size=(iterations, len(deltas)))
            means = np.mean(deltas[sampled], axis=1)
            rows.append(
                {
                    "pair": pair,
                    "metric": metric,
                    "matches": len(deltas),
                    "mean_delta": float(np.mean(deltas)),
                    "ci_low": float(np.quantile(means, 0.025)),
                    "ci_high": float(np.quantile(means, 0.975)),
                    "proportion_favorable": float(np.mean(means < 0)),
                    "bootstrap_iterations": iterations,
                    "random_seed": random_seed,
                }
            )
    return {
        "schema_version": "paired_bootstrap_report_v1",
        "baseline_model": BASELINE_MODEL_NAME,
        "favorable_direction": "negative_delta",
        "rows": rows,
    }


def _ablation_report(
    search_rows: Sequence[Mapping[str, Any]],
    fold_metrics: Sequence[Mapping[str, Any]],
    aggregate_metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    best_tune: dict[tuple[str, str], float] = {}
    for row in search_rows:
        key = (str(row["model_name"]), str(row["ablation"]))
        current = best_tune.get(key, math.inf)
        best_tune[key] = min(current, float(row["log_loss"]))
    aggregate_by_model = {
        (str(row["model_name"]), str(row["ablation"])): row for row in aggregate_metrics
    }
    fold_counts: dict[tuple[str, str], int] = Counter()
    fold_worse_than_poisson: Counter[tuple[str, str]] = Counter()
    poisson_by_fold = {
        str(row["fold"]): float(row["log_loss"])
        for row in fold_metrics
        if row["model_name"] == BASELINE_MODEL_NAME
    }
    for row in fold_metrics:
        key = (str(row["model_name"]), str(row.get("ablation", "")))
        fold_counts[key] += 1
        if row["model_name"] != BASELINE_MODEL_NAME:
            fold = str(row["fold"])
            if float(row["log_loss"]) >= poisson_by_fold[fold]:
                fold_worse_than_poisson[key] += 1
    rows: list[dict[str, Any]] = [
        {
            "ablation": "poisson_official",
            "model_name": BASELINE_MODEL_NAME,
            "model_kind": "poisson",
            "include_contextual": False,
            "mean_tune_log_loss": "",
            **aggregate_by_model[(BASELINE_MODEL_NAME, "poisson_official")],
        }
    ]
    for spec in _ablation_specs_for_report():
        aggregate = aggregate_by_model.get((spec.model_name, spec.ablation))
        key = (spec.model_name, spec.ablation)
        spec_metrics = [
            row
            for row in fold_metrics
            if row["model_name"] == spec.model_name and row["ablation"] == spec.ablation
        ]
        if spec_metrics:
            metrics = _mean_metric_rows(spec_metrics)
        elif aggregate is not None:
            metrics = dict(aggregate)
        else:
            continue
        rows.append(
            {
                "ablation": spec.ablation,
                "model_name": spec.model_name,
                "model_kind": spec.model_kind,
                "include_contextual": spec.include_contextual,
                "mean_tune_log_loss": best_tune.get(key, math.inf),
                "folds_evaluated": fold_counts.get(key, 0),
                "folds_not_better_than_poisson": fold_worse_than_poisson.get(key, 0),
                **metrics,
            }
        )
    return rows


def _ablation_specs_for_report() -> tuple[AblationSpec, ...]:
    return (
        AblationSpec("logistic_stack", SANITY_MODEL_NAME, "logistic", False),
        AblationSpec("contextual_logistic", SANITY_MODEL_NAME, "logistic", True),
        AblationSpec("lgbm_stack", PRIMARY_MODEL_NAME, "lightgbm", False),
        AblationSpec("lgbm_contextual", PRIMARY_MODEL_NAME, "lightgbm", True),
    )


def _selected_shadow_candidate(
    ablation_report: Sequence[Mapping[str, Any]],
    search_rows: Sequence[Mapping[str, Any]],
    config: ContextualChallengerConfig,
) -> dict[str, Any]:
    by_ablation = {str(row["ablation"]): row for row in ablation_report}
    logistic = by_ablation.get("contextual_logistic")
    lgbm = by_ablation.get("lgbm_contextual")
    poisson = by_ablation.get("poisson_official")
    if logistic is None or lgbm is None or poisson is None:
        raise ContextualChallengerError("ablation report is missing required rows")
    logistic_tune = _optional_float(logistic.get("mean_tune_log_loss")) or math.inf
    lgbm_tune = _optional_float(lgbm.get("mean_tune_log_loss")) or math.inf
    selected_row = lgbm if lgbm_tune < logistic_tune else logistic
    selected_model_name = str(selected_row["model_name"])
    selected_ablation = str(selected_row["ablation"])
    selected_kind = str(selected_row["model_kind"])
    selected_log_loss = float(selected_row["log_loss"])
    poisson_log_loss = float(poisson["log_loss"])
    promotion_status = (
        "shadow_monitoring" if selected_log_loss < poisson_log_loss else "not_eligible"
    )
    hyperparameters = _best_hyperparameters_for_ablation(
        selected_ablation,
        search_rows,
        config,
    )
    return {
        "model_name": selected_model_name,
        "ablation": selected_ablation,
        "model_kind": selected_kind,
        "hyperparameters": hyperparameters,
        "promotion_status": promotion_status,
    }


def _best_hyperparameters_for_ablation(
    ablation: str,
    search_rows: Sequence[Mapping[str, Any]],
    config: ContextualChallengerConfig,
) -> dict[str, Any]:
    spec_by_ablation = {spec.ablation: spec for spec in _ablation_specs(config)}
    spec = spec_by_ablation.get(ablation)
    if spec is None or spec.model_kind == "poisson":
        return {"name": "poisson_official"}
    losses_by_name: dict[str, list[float]] = defaultdict(list)
    for row in search_rows:
        if str(row.get("ablation")) != ablation:
            continue
        if str(row.get("selection_set")) != "tune":
            continue
        losses_by_name[str(row["hyperparameter_name"])].append(float(row["log_loss"]))
    if not losses_by_name:
        raise ContextualChallengerError(f"no tune search rows available for {ablation}")
    best_name = min(
        losses_by_name,
        key=lambda name: (float(np.mean(losses_by_name[name])), name),
    )
    options_by_name = {
        str(option["name"]): dict(option) for option in _search_options(spec, config)
    }
    option = options_by_name.get(best_name)
    if option is None:
        raise ContextualChallengerError(f"selected hyperparameter {best_name} is not configured")
    option["selection_source"] = "mean_inner_tune_log_loss"
    option["selection_mean_tune_log_loss"] = float(np.mean(losses_by_name[best_name]))
    option["selection_folds"] = len(losses_by_name[best_name])
    return option


def _selected_config_payload(
    *,
    config: ContextualChallengerConfig,
    whitelist: FeatureWhitelistConfig,
    selected: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    fold_metrics: Sequence[Mapping[str, Any]],
    bootstrap_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "contextual_challenger_selected_config_v1",
        "model_name": selected["model_name"],
        "model_version": selected["model_name"],
        "selected_ablation": selected["ablation"],
        "official_baseline_model_version": config.official_baseline_model_version,
        "feature_set_version": whitelist.feature_set_version,
        "base_features": list(whitelist.base_features),
        "contextual_features": list(whitelist.contextual_features),
        "excluded_features": list(whitelist.excluded_features),
        "hyperparameters": selected["hyperparameters"],
        "calibration": config.calibration.model_dump(mode="json"),
        "folds_version": config.folds_version,
        "validation_folds": [str(fold["name"]) for fold in folds],
        "training_cutoff": config.final_holdout_start.isoformat(),
        "random_seed": config.random_seed,
        "metrics_historical": list(fold_metrics),
        "bootstrap_summary": bootstrap_report,
        "promotion_status": selected["promotion_status"],
        "notes": [
            "Selection excludes 2026 results.",
            "poisson_goal_v1 remains the official model.",
            "Shadow status cannot be promoted by this task.",
        ],
    }


def _importance_rows(
    estimator: ProbabilityEstimator,
    test_features: pd.DataFrame,
    target: np.ndarray,
    *,
    fold_name: str,
    spec: AblationSpec,
    random_seed: int,
) -> list[dict[str, Any]]:
    if spec.model_kind != "lightgbm" or not isinstance(estimator, PipelineProbabilityEstimator):
        return []
    importances = estimator.lightgbm_importances()
    if importances is None:
        return []
    feature_names = estimator.transformed_feature_names()
    gain, split = importances
    rows = []
    for index, feature in enumerate(feature_names):
        rows.append(
            {
                "fold": fold_name,
                "model_name": spec.model_name,
                "ablation": spec.ablation,
                "importance_type": "gain",
                "feature": feature,
                "importance": float(gain[index]) if index < len(gain) else 0.0,
            }
        )
        rows.append(
            {
                "fold": fold_name,
                "model_name": spec.model_name,
                "ablation": spec.ablation,
                "importance_type": "split",
                "feature": feature,
                "importance": float(split[index]) if index < len(split) else 0.0,
            }
        )
    rows.extend(
        _permutation_importance_rows(
            estimator,
            test_features,
            target,
            fold_name=fold_name,
            spec=spec,
            random_seed=random_seed,
        )
    )
    return rows


def _permutation_importance_rows(
    estimator: ProbabilityEstimator,
    test_features: pd.DataFrame,
    target: np.ndarray,
    *,
    fold_name: str,
    spec: AblationSpec,
    random_seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(random_seed)
    baseline = _log_loss(estimator.predict_proba(test_features), target)
    rows = []
    for feature in test_features.columns:
        permuted = test_features.copy()
        values = permuted[feature].to_numpy(copy=True)
        rng.shuffle(values)
        permuted[feature] = values
        score = _log_loss(estimator.predict_proba(permuted), target)
        rows.append(
            {
                "fold": fold_name,
                "model_name": spec.model_name,
                "ablation": spec.ablation,
                "importance_type": "permutation_log_loss_delta",
                "feature": str(feature),
                "importance": float(score - baseline),
            }
        )
    return rows


def _elo_rows_by_id(
    rows: Sequence[Mapping[str, Any]],
    *,
    elo_config: EloRatingsConfig,
) -> dict[str, Mapping[str, Any]]:
    try:
        rating_rows, _ = _rate_matches(rows, config=elo_config)
    except Exception as exc:
        raise ContextualChallengerError(f"failed to compute Elo base features: {exc}") from exc
    return {_require_str(row, "match_id"): dict(row) for row in rating_rows}


def _contextual_rows_by_match_id(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        match_id = _optional_str(row.get("match_id"))
        if match_id is None:
            continue
        existing = output.get(match_id)
        if existing is not None and _row_fingerprint(existing) != _row_fingerprint(row):
            raise ContextualChallengerError(f"duplicate contextual rows for match_id={match_id}")
        output[match_id] = dict(row)
    return output


def _assert_contextual_rows_available(
    rows: Sequence[Mapping[str, Any]],
    contextual_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    missing = sorted(
        _require_str(row, "match_id")
        for row in rows
        if _require_str(row, "match_id") not in contextual_by_id
    )
    if missing:
        raise ContextualChallengerError(
            "contextual features are missing for modeling rows: " + ", ".join(missing[:5])
        )


def _assert_fold_version(
    config: ContextualChallengerConfig,
    elo_evaluation_config: EloEvaluationConfig,
) -> None:
    if config.folds_version != elo_evaluation_config.folds_version:
        msg = (
            "contextual challenger folds_version must match Elo folds_version: "
            f"{config.folds_version} != {elo_evaluation_config.folds_version}"
        )
        raise ContextualChallengerError(msg)


def _assert_same_prediction_matches_by_model(predictions: Sequence[Mapping[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in predictions:
        key = (str(row["model_name"]), str(row["ablation"]))
        grouped[key].append(_require_str(row, "match_id"))
    reference_key = (BASELINE_MODEL_NAME, "poisson_official")
    reference = sorted(grouped[reference_key])
    for key, ids in sorted(grouped.items()):
        duplicates = sorted(match_id for match_id, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise ContextualChallengerError(f"{key} has duplicate match ids: {duplicates[:5]}")
        if sorted(ids) != reference:
            missing = sorted(set(reference) - set(ids))[:5]
            extra = sorted(set(ids) - set(reference))[:5]
            raise ContextualChallengerError(
                f"{key} match ids differ from poisson; missing={missing}, extra={extra}"
            )


def _losses_for_row(row: Mapping[str, Any]) -> dict[str, float]:
    probabilities = np.asarray(
        [row["prob_home_win"], row["prob_draw"], row["prob_away_win"]],
        dtype=float,
    ).reshape(1, -1)
    target = np.asarray([_class_from_name(str(row["actual_result"]))], dtype=int)
    one_hot = np.eye(len(CLASS_LABELS))[target[0]]
    return {
        "log_loss": _log_loss(probabilities, target),
        "brier_score": float(np.sum(np.square(probabilities[0] - one_hot))),
        "ranked_probability_score": _ranked_probability_score(probabilities, target),
    }


def _probability_array(predictions: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [row["prob_home_win"], row["prob_draw"], row["prob_away_win"]]
            for row in predictions
        ],
        dtype=float,
    )


def _log_loss(probabilities: np.ndarray, targets: np.ndarray) -> float:
    true_probabilities = probabilities[np.arange(len(targets)), targets]
    return float(-np.mean(np.log(np.clip(true_probabilities, EPSILON, 1.0))))


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


def _calibration_error(probabilities: np.ndarray, targets: np.ndarray, *, bins: int) -> float:
    errors = []
    for class_label in CLASS_LABELS:
        class_probabilities = probabilities[:, class_label]
        observed = (targets == class_label).astype(float)
        errors.append(_binary_calibration_error(class_probabilities, observed, bins=bins))
    return float(np.mean(errors))


def _binary_calibration_error(
    probabilities: np.ndarray,
    observed: np.ndarray,
    *,
    bins: int,
) -> float:
    total = len(probabilities)
    if total == 0:
        return 0.0
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        gap = abs(np.mean(probabilities[mask]) - np.mean(observed[mask]))
        error += float(np.sum(mask) / total * gap)
    return error


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.asarray(np.clip(probabilities, EPSILON, 1.0), dtype=float)
    totals = clipped.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ContextualChallengerError("probability rows must have positive mass")
    return np.asarray(clipped / totals, dtype=float)


def _assert_valid_probabilities(probabilities: np.ndarray) -> None:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_LABELS):
        raise ContextualChallengerError("probabilities must be an n x 3 matrix")
    if not np.all(np.isfinite(probabilities)):
        raise ContextualChallengerError("probabilities contain non-finite values")
    if np.any(probabilities < -1e-12) or np.any(probabilities > 1 + 1e-12):
        raise ContextualChallengerError("probabilities are outside [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ContextualChallengerError("probabilities do not sum to one")


def _logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.asarray(np.clip(probabilities, EPSILON, 1 - EPSILON), dtype=float)
    return np.asarray(np.log(clipped / (1 - clipped)), dtype=float)


def _mean_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in (
            "log_loss",
            "brier_score",
            "ranked_probability_score",
            "calibration_error",
            "accuracy",
            "mean_true_probability",
        )
        if all(key in row for row in rows)
    }
    return {"matches": int(sum(int(row["matches"]) for row in rows)), **metrics}


def _read_yaml_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContextualChallengerError(f"failed to read {label} config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ContextualChallengerError(f"failed to parse {label} config {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ContextualChallengerError(f"{label} config must contain a mapping: {path}")
    return payload


def _read_parquet_rows(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ContextualChallengerError(f"{label} Parquet is missing: {path}")
    try:
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
    except (OSError, pa.ArrowInvalid) as exc:
        raise ContextualChallengerError(f"failed to read {label} Parquet {path}: {exc}") from exc
    return [dict(row) for row in table.to_pylist()]


def _eligible_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        dict(row)
        for row in rows
        if row.get("model_eligible") is True
        and str(row.get("match_status") or "played") == "played"
        and _is_int(row.get("home_goals_90"))
        and _is_int(row.get("away_goals_90"))
    ]
    eligible.sort(key=lambda row: (_match_date(row), _require_str(row, "match_id")))
    return eligible


def _ensure_output_dirs(outputs: ContextualChallengerOutputConfig) -> None:
    for value in outputs.model_dump(mode="python").values():
        if isinstance(value, Path):
            value.parent.mkdir(parents=True, exist_ok=True)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _write_predictions(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(row) for row in rows])
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


def _write_report(
    path: Path,
    *,
    selected_config: Mapping[str, Any],
    aggregate_metrics: Sequence[Mapping[str, Any]],
    fold_metrics: Sequence[Mapping[str, Any]],
    bootstrap_report: Mapping[str, Any],
    ablation_report: Sequence[Mapping[str, Any]],
    holdout_matches: int,
) -> None:
    lines = [
        "# Contextual Challenger Evaluation",
        "",
        f"Official baseline remains: `{BASELINE_MODEL_NAME}`",
        f"Selected shadow model: `{selected_config['model_name']}`",
        f"Promotion status: `{selected_config['promotion_status']}`",
        f"Holdout 2026 matches scored: {holdout_matches}",
        "",
        "## Aggregate Metrics",
        "",
        _markdown_table(
            aggregate_metrics,
            fields=(
                "model_name",
                "ablation",
                "matches",
                "log_loss",
                "brier_score",
                "ranked_probability_score",
                "calibration_error",
                "accuracy",
            ),
        ),
        "",
        "## Ablations",
        "",
        _markdown_table(
            ablation_report,
            fields=(
                "ablation",
                "model_name",
                "include_contextual",
                "matches",
                "log_loss",
                "brier_score",
                "ranked_probability_score",
            ),
        ),
        "",
        "## Fold Metrics",
        "",
        _markdown_table(
            fold_metrics,
            fields=(
                "model_name",
                "ablation",
                "fold",
                "matches",
                "log_loss",
                "brier_score",
                "ranked_probability_score",
            ),
        ),
        "",
        "## Paired Bootstrap",
        "",
        _markdown_table(
            bootstrap_report.get("rows", []),
            fields=(
                "pair",
                "metric",
                "matches",
                "mean_delta",
                "ci_low",
                "ci_high",
                "proportion_favorable",
            ),
        ),
        "",
        "Feature importance is predictive diagnostics only; it is not causal evidence.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(rows: Sequence[Mapping[str, Any]], *, fields: Sequence[str]) -> str:
    if not rows:
        return "No rows."
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_value(row.get(field)) for field in fields) + " |")
    return "\n".join(lines)


def _markdown_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return ""
    return str(value)


def _unique_matches(predictions: Sequence[Mapping[str, Any]]) -> int:
    return len({_require_str(row, "match_id") for row in predictions})


def _class_name(value: int) -> str:
    return CLASS_NAMES[value]


def _class_from_name(value: str) -> int:
    mapping = {"home_win": HOME_CLASS, "draw": DRAW_CLASS, "away_win": AWAY_CLASS}
    try:
        return mapping[value]
    except KeyError as exc:
        raise ContextualChallengerError(f"unknown class label: {value}") from exc


def _congestion_level(row: Mapping[str, Any]) -> str:
    home = _optional_float(row.get("home_matches_last_7d"))
    away = _optional_float(row.get("away_matches_last_7d"))
    value = max(home or 0.0, away or 0.0)
    if value <= 0:
        return "none"
    if value == 1:
        return "one_match_last_7d"
    return "two_plus_matches_last_7d"


def _missingness_level(row: Mapping[str, Any]) -> str:
    fields = (
        "home_rest_days",
        "away_rest_days",
        "home_previous_match_penalty_shootout",
        "away_previous_match_penalty_shootout",
    )
    missing = sum(1 for field in fields if row.get(field) is None)
    return "complete" if missing == 0 else "has_missing"


def _difference(left: object, right: object) -> float | None:
    left_value = _optional_float(left)
    right_value = _optional_float(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _feature_value(value: object) -> object:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, np.bool_):
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _match_date(row: Mapping[str, Any]) -> date:
    value = row.get("match_date")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ContextualChallengerError(f"match_date is required for match {row.get('match_id')!r}")


def _require_date_mapping(row: Mapping[str, Any], field_name: str) -> date:
    value = row.get(field_name)
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ContextualChallengerError(f"{field_name} must be a date")


def _require_str(row: Mapping[str, Any], field_name: str) -> str:
    value = row.get(field_name)
    if isinstance(value, str) and value:
        return value
    raise ContextualChallengerError(
        f"required string field {field_name} is missing for match {row.get('match_id')!r}"
    )


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _require_int(row: Mapping[str, Any], field_name: str) -> int:
    value = row.get(field_name)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ContextualChallengerError(f"{field_name} must be an integer for {row.get('match_id')!r}")


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float | np.integer | np.floating):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if not isinstance(value, str):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _is_int(value: object) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool)


def _isoformat_or_none(value: object) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return None


def _row_fingerprint(row: Mapping[str, Any]) -> str:
    return json.dumps(dict(row), sort_keys=True, default=_json_value)


def _generated_at() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _csv_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.10f}"
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True, default=_json_value)
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
    if isinstance(value, np.bool_):
        return bool(value)
    return value
