"""Match-level simulation primitives for group and knockout matches."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

MatchDecisionMethod = Literal["regulation", "extra_time", "penalties"]


class MatchSimulationError(RuntimeError):
    """Raised when a match cannot be simulated or audited safely."""


@dataclass(frozen=True)
class ScoreMatrix:
    """A normalized 90-minute score matrix from the official goal model."""

    probabilities: np.ndarray
    expected_home_goals: float
    expected_away_goals: float
    source: str
    original_mass: float


@dataclass(frozen=True)
class MatchSimulation:
    """One simulated or observed match result."""

    match_number: int | None
    source_fixture_id: str | None
    stage: str
    home_team_id: str
    away_team_id: str
    home_goals_90: int
    away_goals_90: int
    home_goals_after_extra_time: int | None
    away_goals_after_extra_time: int | None
    penalty_winner_team_id: str | None
    advancing_team_id: str
    losing_team_id: str
    decision_method: MatchDecisionMethod
    observed: bool


def score_matrix_from_prediction(row: Mapping[str, object]) -> ScoreMatrix:
    """Build a score matrix from one persisted prediction row."""

    raw = row.get("score_probabilities_json")
    if not isinstance(raw, str) or not raw:
        raise MatchSimulationError("prediction row is missing score_probabilities_json")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MatchSimulationError("score_probabilities_json is invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise MatchSimulationError("score_probabilities_json must contain an object")
    return score_matrix_from_mapping(
        payload,
        expected_home_goals=_finite_float(row.get("expected_home_goals")),
        expected_away_goals=_finite_float(row.get("expected_away_goals")),
        source="persisted_prediction",
    )


def score_matrix_from_mapping(
    score_probabilities: Mapping[str, object],
    *,
    expected_home_goals: float,
    expected_away_goals: float,
    source: str,
) -> ScoreMatrix:
    """Normalize score probabilities keyed as ``H-A`` strings."""

    parsed: dict[tuple[int, int], float] = {}
    max_goal = 0
    for raw_score, raw_probability in score_probabilities.items():
        home_goals, away_goals = _parse_score_key(str(raw_score))
        probability = _finite_float(raw_probability)
        if probability < 0:
            raise MatchSimulationError("score probabilities cannot be negative")
        parsed[(home_goals, away_goals)] = probability
        max_goal = max(max_goal, home_goals, away_goals)
    if not parsed:
        raise MatchSimulationError("score matrix cannot be empty")
    matrix = np.zeros((max_goal + 1, max_goal + 1), dtype=float)
    for (home_goals, away_goals), probability in parsed.items():
        matrix[home_goals, away_goals] = probability
    mass = float(matrix.sum())
    if not math.isfinite(mass) or mass <= 0:
        raise MatchSimulationError("score matrix has zero or non-finite probability mass")
    matrix = matrix / mass
    return ScoreMatrix(
        probabilities=matrix,
        expected_home_goals=expected_home_goals,
        expected_away_goals=expected_away_goals,
        source=source,
        original_mass=mass,
    )


def simulate_group_score(matrix: ScoreMatrix, rng: np.random.Generator) -> tuple[int, int]:
    """Sample a 90-minute score for a group-stage match."""

    return _sample_score(matrix.probabilities, rng)


def simulate_knockout_match(
    *,
    match_number: int | None,
    source_fixture_id: str | None,
    stage: str,
    home_team_id: str,
    away_team_id: str,
    matrix: ScoreMatrix,
    rng: np.random.Generator,
    extra_time_goal_scale: float,
) -> MatchSimulation:
    """Simulate a knockout match, including extra time and penalties after a 90-minute draw."""

    if extra_time_goal_scale < 0:
        raise MatchSimulationError("extra_time_goal_scale cannot be negative")
    home_90, away_90 = _sample_score(matrix.probabilities, rng)
    if home_90 > away_90:
        return _knockout_result(
            match_number=match_number,
            source_fixture_id=source_fixture_id,
            stage=stage,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_goals_90=home_90,
            away_goals_90=away_90,
            home_goals_after_extra_time=None,
            away_goals_after_extra_time=None,
            penalty_winner_team_id=None,
            advancing_team_id=home_team_id,
            decision_method="regulation",
            observed=False,
        )
    if away_90 > home_90:
        return _knockout_result(
            match_number=match_number,
            source_fixture_id=source_fixture_id,
            stage=stage,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_goals_90=home_90,
            away_goals_90=away_90,
            home_goals_after_extra_time=None,
            away_goals_after_extra_time=None,
            penalty_winner_team_id=None,
            advancing_team_id=away_team_id,
            decision_method="regulation",
            observed=False,
        )

    home_extra_goals = int(rng.poisson(matrix.expected_home_goals * extra_time_goal_scale))
    away_extra_goals = int(rng.poisson(matrix.expected_away_goals * extra_time_goal_scale))
    home_after_extra = home_90 + home_extra_goals
    away_after_extra = away_90 + away_extra_goals
    if home_after_extra > away_after_extra:
        advancing = home_team_id
        method: MatchDecisionMethod = "extra_time"
        penalty_winner = None
    elif away_after_extra > home_after_extra:
        advancing = away_team_id
        method = "extra_time"
        penalty_winner = None
    else:
        advancing = home_team_id if float(rng.random()) < 0.5 else away_team_id
        method = "penalties"
        penalty_winner = advancing
    return _knockout_result(
        match_number=match_number,
        source_fixture_id=source_fixture_id,
        stage=stage,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_goals_90=home_90,
        away_goals_90=away_90,
        home_goals_after_extra_time=home_after_extra,
        away_goals_after_extra_time=away_after_extra,
        penalty_winner_team_id=penalty_winner,
        advancing_team_id=advancing,
        decision_method=method,
        observed=False,
    )


def observed_knockout_match(
    *,
    match_number: int | None,
    source_fixture_id: str | None,
    stage: str,
    home_team_id: str,
    away_team_id: str,
    home_goals_90: int,
    away_goals_90: int,
    home_goals_after_extra_time: int | None,
    away_goals_after_extra_time: int | None,
    home_penalty_goals: int | None,
    away_penalty_goals: int | None,
) -> MatchSimulation:
    """Return the advancement result for an observed knockout fixture."""

    if home_goals_90 > away_goals_90:
        advancing = home_team_id
        method: MatchDecisionMethod = "regulation"
        penalty_winner = None
    elif away_goals_90 > home_goals_90:
        advancing = away_team_id
        method = "regulation"
        penalty_winner = None
    elif (
        home_goals_after_extra_time is not None
        and away_goals_after_extra_time is not None
        and home_goals_after_extra_time != away_goals_after_extra_time
    ):
        advancing = (
            home_team_id
            if home_goals_after_extra_time > away_goals_after_extra_time
            else away_team_id
        )
        method = "extra_time"
        penalty_winner = None
    elif home_penalty_goals is not None and away_penalty_goals is not None:
        if home_penalty_goals == away_penalty_goals:
            raise MatchSimulationError("observed penalties must identify a winner")
        advancing = home_team_id if home_penalty_goals > away_penalty_goals else away_team_id
        method = "penalties"
        penalty_winner = advancing
    else:
        raise MatchSimulationError("observed knockout draw is missing extra-time or penalty winner")
    return _knockout_result(
        match_number=match_number,
        source_fixture_id=source_fixture_id,
        stage=stage,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_goals_90=home_goals_90,
        away_goals_90=away_goals_90,
        home_goals_after_extra_time=home_goals_after_extra_time,
        away_goals_after_extra_time=away_goals_after_extra_time,
        penalty_winner_team_id=penalty_winner,
        advancing_team_id=advancing,
        decision_method=method,
        observed=True,
    )


def _knockout_result(
    *,
    match_number: int | None,
    source_fixture_id: str | None,
    stage: str,
    home_team_id: str,
    away_team_id: str,
    home_goals_90: int,
    away_goals_90: int,
    home_goals_after_extra_time: int | None,
    away_goals_after_extra_time: int | None,
    penalty_winner_team_id: str | None,
    advancing_team_id: str,
    decision_method: MatchDecisionMethod,
    observed: bool,
) -> MatchSimulation:
    losing = away_team_id if advancing_team_id == home_team_id else home_team_id
    return MatchSimulation(
        match_number=match_number,
        source_fixture_id=source_fixture_id,
        stage=stage,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_goals_90=home_goals_90,
        away_goals_90=away_goals_90,
        home_goals_after_extra_time=home_goals_after_extra_time,
        away_goals_after_extra_time=away_goals_after_extra_time,
        penalty_winner_team_id=penalty_winner_team_id,
        advancing_team_id=advancing_team_id,
        losing_team_id=losing,
        decision_method=decision_method,
        observed=observed,
    )


def _sample_score(matrix: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    flat = matrix.ravel()
    index = int(rng.choice(len(flat), p=flat))
    home_goals, away_goals = np.unravel_index(index, matrix.shape)
    return int(home_goals), int(away_goals)


def _parse_score_key(value: str) -> tuple[int, int]:
    raw_home, separator, raw_away = value.partition("-")
    if separator != "-":
        raise MatchSimulationError(f"invalid score key: {value}")
    try:
        home_goals = int(raw_home)
        away_goals = int(raw_away)
    except ValueError as exc:
        raise MatchSimulationError(f"invalid score key: {value}") from exc
    if home_goals < 0 or away_goals < 0:
        raise MatchSimulationError(f"invalid negative score key: {value}")
    return home_goals, away_goals


def _finite_float(value: object) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise MatchSimulationError("expected a finite floating point value") from exc
    if not math.isfinite(parsed):
        raise MatchSimulationError("expected a finite floating point value")
    return parsed
