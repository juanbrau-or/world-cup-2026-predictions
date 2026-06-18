"""Poisson and Dixon-Coles goal models for international football."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, Self

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.special import gammaln  # type: ignore[import-untyped]

GoalModelType = Literal["poisson", "dixon_coles"]

EPSILON = 1e-15


class DixonColesModelError(RuntimeError):
    """Raised when a Poisson or Dixon-Coles model cannot be fit or scored."""


@dataclass(frozen=True)
class ScoreDistribution:
    """Predicted goal distribution and derived 1X2 probabilities."""

    expected_home_goals: float
    expected_away_goals: float
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    modal_score: str
    score_probabilities: dict[str, float]
    score_probability_mass: float
    residual_probability: float


@dataclass(frozen=True)
class FittedGoalParameters:
    """Serializable fitted goal-model parameters."""

    model_type: GoalModelType
    half_life_days: float | None
    max_goals: int
    regularization_strength: float
    teams: tuple[str, ...]
    categories: tuple[str, ...]
    intercept: float
    home_advantage: float
    rho: float
    attack: dict[str, float]
    defense: dict[str, float]
    category_effects: dict[str, float]

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible parameter representation."""

        return {
            "model_type": self.model_type,
            "half_life_days": self.half_life_days,
            "max_goals": self.max_goals,
            "regularization_strength": self.regularization_strength,
            "teams": list(self.teams),
            "categories": list(self.categories),
            "intercept": self.intercept,
            "home_advantage": self.home_advantage,
            "rho": self.rho,
            "attack": self.attack,
            "defense": self.defense,
            "category_effects": self.category_effects,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Create parameters from a JSON-compatible mapping."""

        raw_model_type = str(payload["model_type"])
        if raw_model_type == "poisson":
            model_type: GoalModelType = "poisson"
        elif raw_model_type == "dixon_coles":
            model_type = "dixon_coles"
        else:
            msg = f"unknown goal model type: {raw_model_type}"
            raise DixonColesModelError(msg)
        half_life = payload.get("half_life_days")
        return cls(
            model_type=model_type,
            half_life_days=None if half_life is None else float(half_life),
            max_goals=int(payload["max_goals"]),
            regularization_strength=float(payload["regularization_strength"]),
            teams=tuple(str(value) for value in payload["teams"]),
            categories=tuple(str(value) for value in payload["categories"]),
            intercept=float(payload["intercept"]),
            home_advantage=float(payload["home_advantage"]),
            rho=float(payload["rho"]),
            attack={str(key): float(value) for key, value in dict(payload["attack"]).items()},
            defense={str(key): float(value) for key, value in dict(payload["defense"]).items()},
            category_effects={
                str(key): float(value) for key, value in dict(payload["category_effects"]).items()
            },
        )


class DixonColesGoalModel:
    """Regularized Poisson goal model with optional Dixon-Coles correction."""

    def __init__(
        self,
        *,
        model_type: GoalModelType,
        half_life_days: float | None,
        max_goals: int,
        regularization_strength: float,
    ) -> None:
        if max_goals < 1:
            msg = "max_goals must be at least 1"
            raise DixonColesModelError(msg)
        if half_life_days is not None and half_life_days <= 0:
            msg = "half_life_days must be positive or None"
            raise DixonColesModelError(msg)
        if regularization_strength < 0:
            msg = "regularization_strength cannot be negative"
            raise DixonColesModelError(msg)
        self.model_type = model_type
        self.half_life_days = half_life_days
        self.max_goals = max_goals
        self.regularization_strength = regularization_strength
        self.parameters: FittedGoalParameters | None = None

    def fit(self, rows: Sequence[Mapping[str, Any]], *, cutoff: date) -> None:
        """Fit the model using only rows before the supplied cutoff date."""

        train_rows = [row for row in rows if _match_date(row) < cutoff]
        if not train_rows:
            msg = f"cannot fit goal model without matches before {cutoff.isoformat()}"
            raise DixonColesModelError(msg)
        if any(_match_date(row) >= cutoff for row in train_rows):
            msg = f"training rows include matches on or after {cutoff.isoformat()}"
            raise DixonColesModelError(msg)

        teams = _sorted_teams(train_rows)
        categories = _sorted_categories(train_rows)
        if len(teams) < 2:
            msg = "goal model requires at least two teams"
            raise DixonColesModelError(msg)

        design = _build_design(
            train_rows,
            cutoff=cutoff,
            half_life_days=self.half_life_days,
            teams=teams,
            categories=categories,
        )
        result = minimize(
            lambda params: _objective_and_gradient(
                params,
                design=design,
                teams=teams,
                categories=categories,
                model_type=self.model_type,
                regularization_strength=self.regularization_strength,
            ),
            _initial_parameters(design, teams=teams, categories=categories),
            method="L-BFGS-B",
            jac=True,
            bounds=_parameter_bounds(
                teams=teams,
                categories=categories,
                model_type=self.model_type,
            ),
            options={"maxiter": 5000, "maxfun": 100000, "ftol": 1e-10},
        )
        if not result.success:
            msg = f"goal model optimization failed: {result.message}"
            raise DixonColesModelError(msg)

        unpacked = _unpack_parameters(
            np.asarray(result.x, dtype=float),
            teams=teams,
            categories=categories,
            model_type=self.model_type,
        )
        self.parameters = FittedGoalParameters(
            model_type=self.model_type,
            half_life_days=self.half_life_days,
            max_goals=self.max_goals,
            regularization_strength=self.regularization_strength,
            teams=teams,
            categories=categories,
            intercept=unpacked.intercept,
            home_advantage=unpacked.home_advantage,
            rho=unpacked.rho,
            attack=dict(zip(teams, unpacked.attack, strict=True)),
            defense=dict(zip(teams, unpacked.defense, strict=True)),
            category_effects=dict(zip(categories, unpacked.category_effects, strict=True)),
        )

    def predict_match(self, row: Mapping[str, Any]) -> ScoreDistribution:
        """Return the score distribution and 1X2 probabilities for one match row."""

        if self.parameters is None:
            msg = "goal model must be fit before prediction"
            raise DixonColesModelError(msg)
        home_team = _require_str(row, "home_team_id")
        away_team = _require_str(row, "away_team_id")
        category = _require_str(row, "competition_category")
        home_advantage_eligible = bool(row.get("home_advantage_eligible"))
        return score_distribution(
            self.parameters,
            home_team=home_team,
            away_team=away_team,
            competition_category=category,
            home_advantage_eligible=home_advantage_eligible,
        )

    def score_log_probability(self, row: Mapping[str, Any]) -> float:
        """Return the full, untruncated log probability of the observed score."""

        if self.parameters is None:
            msg = "goal model must be fit before scoring"
            raise DixonColesModelError(msg)
        home_mean, away_mean = expected_goals(
            self.parameters,
            home_team=_require_str(row, "home_team_id"),
            away_team=_require_str(row, "away_team_id"),
            competition_category=_require_str(row, "competition_category"),
            home_advantage_eligible=bool(row.get("home_advantage_eligible")),
        )
        home_goals = _require_int(row, "home_goals_90")
        away_goals = _require_int(row, "away_goals_90")
        log_probability = _poisson_log_pmf(home_goals, home_mean) + _poisson_log_pmf(
            away_goals,
            away_mean,
        )
        if self.parameters.model_type == "dixon_coles":
            tau = _dixon_coles_tau(
                home_goals,
                away_goals,
                home_mean,
                away_mean,
                self.parameters.rho,
            )
            log_probability += math.log(max(tau, EPSILON))
        return float(log_probability)

    def to_json(self) -> str:
        """Serialize fitted parameters to a deterministic JSON string."""

        if self.parameters is None:
            msg = "goal model must be fit before serialization"
            raise DixonColesModelError(msg)
        return json.dumps(self.parameters.to_json_dict(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class _DesignMatrix:
    home_team_index: np.ndarray
    away_team_index: np.ndarray
    category_index: np.ndarray
    home_goals: np.ndarray
    away_goals: np.ndarray
    home_advantage_eligible: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class _UnpackedParameters:
    intercept: float
    home_advantage: float
    attack: np.ndarray
    defense: np.ndarray
    category_effects: np.ndarray
    rho: float


def score_distribution(
    parameters: FittedGoalParameters,
    *,
    home_team: str,
    away_team: str,
    competition_category: str,
    home_advantage_eligible: bool,
) -> ScoreDistribution:
    """Build a truncated score matrix plus residual probability."""

    home_mean, away_mean = expected_goals(
        parameters,
        home_team=home_team,
        away_team=away_team,
        competition_category=competition_category,
        home_advantage_eligible=home_advantage_eligible,
    )
    grid = _score_matrix(
        home_mean,
        away_mean,
        max_goals=parameters.max_goals,
        model_type=parameters.model_type,
        rho=parameters.rho,
    )
    mass = float(np.sum(grid))
    residual = max(0.0, min(1.0, 1.0 - mass))
    score_probabilities: dict[str, float] = {}
    for home_goals in range(parameters.max_goals + 1):
        for away_goals in range(parameters.max_goals + 1):
            score_probabilities[f"{home_goals}-{away_goals}"] = float(
                grid[home_goals, away_goals]
            )
    modal_index = np.unravel_index(int(np.argmax(grid)), grid.shape)

    outcome_matrix = _score_matrix(
        home_mean,
        away_mean,
        max_goals=max(parameters.max_goals, 40),
        model_type=parameters.model_type,
        rho=parameters.rho,
    )
    home_probability = float(np.tril(outcome_matrix, k=-1).sum())
    draw_probability = float(np.trace(outcome_matrix))
    away_probability = float(np.triu(outcome_matrix, k=1).sum())
    total = home_probability + draw_probability + away_probability
    if total <= 0:
        msg = "score distribution produced zero outcome probability"
        raise DixonColesModelError(msg)
    home_probability /= total
    draw_probability /= total
    away_probability /= total
    return ScoreDistribution(
        expected_home_goals=home_mean,
        expected_away_goals=away_mean,
        prob_home_win=home_probability,
        prob_draw=draw_probability,
        prob_away_win=away_probability,
        modal_score=f"{modal_index[0]}-{modal_index[1]}",
        score_probabilities=score_probabilities,
        score_probability_mass=mass,
        residual_probability=residual,
    )


def expected_goals(
    parameters: FittedGoalParameters,
    *,
    home_team: str,
    away_team: str,
    competition_category: str,
    home_advantage_eligible: bool,
) -> tuple[float, float]:
    """Return expected home and away goals for a match."""

    home_attack = parameters.attack.get(home_team, 0.0)
    away_attack = parameters.attack.get(away_team, 0.0)
    home_defense = parameters.defense.get(home_team, 0.0)
    away_defense = parameters.defense.get(away_team, 0.0)
    category_effect = parameters.category_effects.get(competition_category, 0.0)
    home_eta = (
        parameters.intercept
        + home_attack
        + away_defense
        + category_effect
        + (parameters.home_advantage if home_advantage_eligible else 0.0)
    )
    away_eta = parameters.intercept + away_attack + home_defense + category_effect
    return float(math.exp(_clip_eta(home_eta))), float(math.exp(_clip_eta(away_eta)))


def temporal_weights(
    rows: Sequence[Mapping[str, Any]],
    *,
    cutoff: date,
    half_life_days: float | None,
) -> np.ndarray:
    """Return exponential temporal weights relative to the fold cutoff."""

    if half_life_days is None:
        return np.ones(len(rows), dtype=float)
    if half_life_days <= 0:
        msg = "half_life_days must be positive or None"
        raise DixonColesModelError(msg)
    weights = [
        math.exp(-math.log(2.0) * max((cutoff - _match_date(row)).days, 0) / half_life_days)
        for row in rows
    ]
    return np.asarray(weights, dtype=float)


def _objective_and_gradient(
    params: np.ndarray,
    *,
    design: _DesignMatrix,
    teams: tuple[str, ...],
    categories: tuple[str, ...],
    model_type: GoalModelType,
    regularization_strength: float,
) -> tuple[float, np.ndarray]:
    unpacked = _unpack_parameters(
        params,
        teams=teams,
        categories=categories,
        model_type=model_type,
    )
    home_eta = (
        unpacked.intercept
        + unpacked.attack[design.home_team_index]
        + unpacked.defense[design.away_team_index]
        + unpacked.category_effects[design.category_index]
        + unpacked.home_advantage * design.home_advantage_eligible
    )
    away_eta = (
        unpacked.intercept
        + unpacked.attack[design.away_team_index]
        + unpacked.defense[design.home_team_index]
        + unpacked.category_effects[design.category_index]
    )
    home_mean = np.exp(np.clip(home_eta, -8.0, 4.0))
    away_mean = np.exp(np.clip(away_eta, -8.0, 4.0))
    log_likelihood = (
        design.home_goals * np.log(home_mean)
        - home_mean
        - gammaln(design.home_goals + 1)
        + design.away_goals * np.log(away_mean)
        - away_mean
        - gammaln(design.away_goals + 1)
    )
    home_eta_score = design.home_goals - home_mean
    away_eta_score = design.away_goals - away_mean
    rho_score = np.zeros_like(home_mean)
    if model_type == "dixon_coles":
        tau = _dixon_coles_tau_array(
            design.home_goals,
            design.away_goals,
            home_mean,
            away_mean,
            unpacked.rho,
        )
        if np.any(tau <= 0) or not np.all(np.isfinite(tau)):
            return 1e9, np.zeros_like(params)
        log_likelihood = log_likelihood + np.log(tau)
        dc_home_score, dc_away_score, rho_score = _dixon_coles_log_tau_scores(
            design.home_goals,
            design.away_goals,
            home_mean,
            away_mean,
            unpacked.rho,
            tau,
        )
        home_eta_score = home_eta_score + dc_home_score
        away_eta_score = away_eta_score + dc_away_score
    mean_negative_log_likelihood = -float(
        np.sum(design.weights * log_likelihood) / np.sum(design.weights)
    )
    penalty, penalty_gradient = _regularization_penalty_and_gradient(
        unpacked,
        teams=teams,
        categories=categories,
        model_type=model_type,
        regularization_strength=regularization_strength,
    )
    gradient = _negative_log_likelihood_gradient(
        design,
        teams=teams,
        categories=categories,
        home_eta_score=home_eta_score,
        away_eta_score=away_eta_score,
        rho_score=rho_score,
    )
    return mean_negative_log_likelihood + penalty, gradient + penalty_gradient


def _regularization_penalty_and_gradient(
    unpacked: _UnpackedParameters,
    *,
    teams: tuple[str, ...],
    categories: tuple[str, ...],
    model_type: GoalModelType,
    regularization_strength: float,
) -> tuple[float, np.ndarray]:
    penalty = regularization_strength * float(
        np.mean(np.square(unpacked.attack))
        + np.mean(np.square(unpacked.defense))
        + np.mean(np.square(unpacked.category_effects))
        + 0.25 * unpacked.home_advantage**2
        + (10.0 * unpacked.rho**2 if model_type == "dixon_coles" else 0.0)
    )
    gradient = np.zeros(2 + (len(teams) - 1) * 2 + max(len(categories) - 1, 0) + 1)
    gradient[1] = regularization_strength * 0.5 * unpacked.home_advantage
    index = 2
    if len(teams) > 1:
        attack_last = unpacked.attack[-1]
        gradient[index : index + len(teams) - 1] = (
            regularization_strength
            * (2.0 / len(teams))
            * (unpacked.attack[:-1] - attack_last)
        )
        index += len(teams) - 1
        defense_last = unpacked.defense[-1]
        gradient[index : index + len(teams) - 1] = (
            regularization_strength
            * (2.0 / len(teams))
            * (unpacked.defense[:-1] - defense_last)
        )
        index += len(teams) - 1
    if len(categories) > 1:
        gradient[index : index + len(categories) - 1] = (
            regularization_strength
            * (2.0 / len(categories))
            * unpacked.category_effects[1:]
        )
        index += len(categories) - 1
    if model_type == "dixon_coles":
        gradient[index] = regularization_strength * 20.0 * unpacked.rho
    return penalty, gradient


def _negative_log_likelihood_gradient(
    design: _DesignMatrix,
    *,
    teams: tuple[str, ...],
    categories: tuple[str, ...],
    home_eta_score: np.ndarray,
    away_eta_score: np.ndarray,
    rho_score: np.ndarray,
) -> np.ndarray:
    weights = design.weights
    weight_total = float(np.sum(weights))
    weighted_home_score = weights * home_eta_score / weight_total
    weighted_away_score = weights * away_eta_score / weight_total
    gradient = np.zeros(2 + (len(teams) - 1) * 2 + max(len(categories) - 1, 0) + 1)
    gradient[0] = -float(np.sum(weighted_home_score + weighted_away_score))
    gradient[1] = -float(np.sum(weighted_home_score * design.home_advantage_eligible))

    attack_scores = np.bincount(
        design.home_team_index,
        weights=weighted_home_score,
        minlength=len(teams),
    ) + np.bincount(
        design.away_team_index,
        weights=weighted_away_score,
        minlength=len(teams),
    )
    defense_scores = np.bincount(
        design.away_team_index,
        weights=weighted_home_score,
        minlength=len(teams),
    ) + np.bincount(
        design.home_team_index,
        weights=weighted_away_score,
        minlength=len(teams),
    )
    category_scores = np.bincount(
        design.category_index,
        weights=weighted_home_score + weighted_away_score,
        minlength=len(categories),
    )
    index = 2
    gradient[index : index + len(teams) - 1] = -(
        attack_scores[: len(teams) - 1] - attack_scores[-1]
    )
    index += len(teams) - 1
    gradient[index : index + len(teams) - 1] = -(
        defense_scores[: len(teams) - 1] - defense_scores[-1]
    )
    index += len(teams) - 1
    if len(categories) > 1:
        gradient[index : index + len(categories) - 1] = -category_scores[1:]
        index += len(categories) - 1
    gradient[index] = -float(np.sum(weights * rho_score) / weight_total)
    return gradient


def _build_design(
    rows: Sequence[Mapping[str, Any]],
    *,
    cutoff: date,
    half_life_days: float | None,
    teams: tuple[str, ...],
    categories: tuple[str, ...],
) -> _DesignMatrix:
    team_index = {team: index for index, team in enumerate(teams)}
    category_index = {category: index for index, category in enumerate(categories)}
    return _DesignMatrix(
        home_team_index=np.asarray(
            [team_index[_require_str(row, "home_team_id")] for row in rows],
            dtype=int,
        ),
        away_team_index=np.asarray(
            [team_index[_require_str(row, "away_team_id")] for row in rows],
            dtype=int,
        ),
        category_index=np.asarray(
            [category_index[_require_str(row, "competition_category")] for row in rows],
            dtype=int,
        ),
        home_goals=np.asarray([_require_int(row, "home_goals_90") for row in rows], dtype=float),
        away_goals=np.asarray([_require_int(row, "away_goals_90") for row in rows], dtype=float),
        home_advantage_eligible=np.asarray(
            [1.0 if row.get("home_advantage_eligible") else 0.0 for row in rows],
            dtype=float,
        ),
        weights=temporal_weights(rows, cutoff=cutoff, half_life_days=half_life_days),
    )


def _initial_parameters(
    design: _DesignMatrix,
    *,
    teams: tuple[str, ...],
    categories: tuple[str, ...],
) -> np.ndarray:
    mean_goals = float(np.mean(np.concatenate([design.home_goals, design.away_goals])))
    intercept = math.log(max(mean_goals, 0.1))
    parameter_count = 2 + (len(teams) - 1) * 2 + max(len(categories) - 1, 0) + 1
    initial = np.zeros(parameter_count, dtype=float)
    initial[0] = intercept
    initial[1] = 0.1
    return initial


def _parameter_bounds(
    *,
    teams: tuple[str, ...],
    categories: tuple[str, ...],
    model_type: GoalModelType,
) -> list[tuple[float | None, float | None]]:
    parameter_count = 2 + (len(teams) - 1) * 2 + max(len(categories) - 1, 0)
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * parameter_count
    if model_type == "dixon_coles":
        bounds.append((-0.2, 0.2))
    else:
        bounds.append((0.0, 0.0))
    return bounds


def _unpack_parameters(
    params: np.ndarray,
    *,
    teams: tuple[str, ...],
    categories: tuple[str, ...],
    model_type: GoalModelType,
) -> _UnpackedParameters:
    del model_type
    index = 0
    intercept = float(params[index])
    index += 1
    home_advantage = float(params[index])
    index += 1
    attack_free = params[index : index + len(teams) - 1]
    index += len(teams) - 1
    defense_free = params[index : index + len(teams) - 1]
    index += len(teams) - 1
    category_free = params[index : index + max(len(categories) - 1, 0)]
    index += max(len(categories) - 1, 0)
    rho = float(params[index])
    attack = np.concatenate([attack_free, np.array([-float(np.sum(attack_free))])])
    defense = np.concatenate([defense_free, np.array([-float(np.sum(defense_free))])])
    category_effects = np.concatenate([np.array([0.0]), category_free])
    return _UnpackedParameters(
        intercept=intercept,
        home_advantage=home_advantage,
        attack=attack,
        defense=defense,
        category_effects=category_effects,
        rho=rho,
    )


def _score_matrix(
    home_mean: float,
    away_mean: float,
    *,
    max_goals: int,
    model_type: GoalModelType,
    rho: float,
) -> np.ndarray:
    home_probabilities = _poisson_pmf_vector(home_mean, max_goals)
    away_probabilities = _poisson_pmf_vector(away_mean, max_goals)
    matrix = np.outer(home_probabilities, away_probabilities)
    if model_type == "dixon_coles":
        for home_goals in (0, 1):
            for away_goals in (0, 1):
                matrix[home_goals, away_goals] *= _dixon_coles_tau(
                    home_goals,
                    away_goals,
                    home_mean,
                    away_mean,
                    rho,
                )
    return np.maximum(matrix, 0.0)


def _poisson_pmf_vector(mean: float, max_goals: int) -> np.ndarray:
    values = [math.exp(-mean)]
    for goals in range(1, max_goals + 1):
        values.append(values[-1] * mean / goals)
    return np.asarray(values, dtype=float)


def _poisson_log_pmf(goals: int, mean: float) -> float:
    return float(goals * math.log(mean) - mean - math.lgamma(goals + 1))


def _dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    home_mean: float,
    away_mean: float,
    rho: float,
) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - home_mean * away_mean * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + home_mean * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + away_mean * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def _dixon_coles_tau_array(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_mean: np.ndarray,
    away_mean: np.ndarray,
    rho: float,
) -> np.ndarray:
    tau = np.ones_like(home_mean)
    mask_00 = (home_goals == 0) & (away_goals == 0)
    mask_01 = (home_goals == 0) & (away_goals == 1)
    mask_10 = (home_goals == 1) & (away_goals == 0)
    mask_11 = (home_goals == 1) & (away_goals == 1)
    tau[mask_00] = 1.0 - home_mean[mask_00] * away_mean[mask_00] * rho
    tau[mask_01] = 1.0 + home_mean[mask_01] * rho
    tau[mask_10] = 1.0 + away_mean[mask_10] * rho
    tau[mask_11] = 1.0 - rho
    return tau


def _dixon_coles_log_tau_scores(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_mean: np.ndarray,
    away_mean: np.ndarray,
    rho: float,
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    home_score = np.zeros_like(home_mean)
    away_score = np.zeros_like(away_mean)
    rho_score = np.zeros_like(home_mean)
    mask_00 = (home_goals == 0) & (away_goals == 0)
    mask_01 = (home_goals == 0) & (away_goals == 1)
    mask_10 = (home_goals == 1) & (away_goals == 0)
    mask_11 = (home_goals == 1) & (away_goals == 1)

    home_score[mask_00] = (
        -home_mean[mask_00] * away_mean[mask_00] * rho / tau[mask_00]
    )
    away_score[mask_00] = home_score[mask_00]
    rho_score[mask_00] = -home_mean[mask_00] * away_mean[mask_00] / tau[mask_00]

    home_score[mask_01] = home_mean[mask_01] * rho / tau[mask_01]
    rho_score[mask_01] = home_mean[mask_01] / tau[mask_01]

    away_score[mask_10] = away_mean[mask_10] * rho / tau[mask_10]
    rho_score[mask_10] = away_mean[mask_10] / tau[mask_10]

    rho_score[mask_11] = -1.0 / tau[mask_11]
    return home_score, away_score, rho_score


def _clip_eta(value: float) -> float:
    return min(max(value, -8.0), 4.0)


def _sorted_teams(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    teams = {
        _require_str(row, "home_team_id")
        for row in rows
    } | {_require_str(row, "away_team_id") for row in rows}
    return tuple(sorted(teams))


def _sorted_categories(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted({_require_str(row, "competition_category") for row in rows}))


def _match_date(row: Mapping[str, Any]) -> date:
    value = row.get("match_date")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    msg = f"expected match_date for match {row.get('match_id')!r}, got {value!r}"
    raise DixonColesModelError(msg)


def _require_str(row: Mapping[str, Any], field_name: str) -> str:
    value = row.get(field_name)
    if isinstance(value, str) and value:
        return value
    msg = f"required field {field_name} is missing or blank for match {row.get('match_id')!r}"
    raise DixonColesModelError(msg)


def _require_int(row: Mapping[str, Any], field_name: str) -> int:
    value = row.get(field_name)
    if isinstance(value, int):
        return value
    if isinstance(value, np.integer):
        return int(value)
    msg = f"required integer field {field_name} is missing for match {row.get('match_id')!r}"
    raise DixonColesModelError(msg)
