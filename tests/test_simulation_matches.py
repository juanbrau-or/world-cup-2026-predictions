from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from worldcup2026.simulation.matches import (
    ScoreMatrix,
    observed_knockout_match,
    score_matrix_from_mapping,
    simulate_group_score,
    simulate_knockout_match,
)


class FixedRng:
    def __init__(
        self,
        *,
        choice_index: int,
        poisson_values: tuple[int, int] = (0, 0),
        random_value: float = 0.0,
    ) -> None:
        self.choice_index = choice_index
        self.poisson_values = list(poisson_values)
        self.random_value = random_value

    def choice(self, *_args: Any, **_kwargs: Any) -> int:
        return self.choice_index

    def poisson(self, _lam: float) -> int:
        return self.poisson_values.pop(0)

    def random(self) -> float:
        return self.random_value


def test_score_matrix_is_normalized() -> None:
    matrix = score_matrix_from_mapping(
        {"0-0": 2.0, "1-0": 1.0},
        expected_home_goals=0.5,
        expected_away_goals=0.2,
        source="test",
    )

    assert matrix.original_mass == pytest.approx(3.0)
    assert float(matrix.probabilities.sum()) == pytest.approx(1.0)


def test_group_score_samples_from_matrix() -> None:
    matrix = score_matrix_from_mapping(
        {"2-1": 1.0},
        expected_home_goals=2.0,
        expected_away_goals=1.0,
        source="test",
    )

    assert simulate_group_score(matrix, np.random.default_rng(1)) == (2, 1)


def test_knockout_regulation_winner() -> None:
    matrix = score_matrix_from_mapping(
        {"1-0": 1.0},
        expected_home_goals=1.0,
        expected_away_goals=0.0,
        source="test",
    )

    result = simulate_knockout_match(
        match_number=73,
        source_fixture_id="fixture-73",
        stage="round_of_32",
        home_team_id="home",
        away_team_id="away",
        matrix=matrix,
        rng=np.random.default_rng(1),
        extra_time_goal_scale=1 / 3,
    )

    assert result.decision_method == "regulation"
    assert result.advancing_team_id == "home"
    assert result.home_goals_after_extra_time is None


def test_knockout_extra_time_winner() -> None:
    matrix = ScoreMatrix(
        probabilities=np.array([[1.0]]),
        expected_home_goals=1.0,
        expected_away_goals=1.0,
        source="test",
        original_mass=1.0,
    )
    rng = FixedRng(choice_index=0, poisson_values=(1, 0))

    result = simulate_knockout_match(
        match_number=73,
        source_fixture_id=None,
        stage="round_of_32",
        home_team_id="home",
        away_team_id="away",
        matrix=matrix,
        rng=rng,  # type: ignore[arg-type]
        extra_time_goal_scale=1 / 3,
    )

    assert result.decision_method == "extra_time"
    assert result.home_goals_after_extra_time == 1
    assert result.away_goals_after_extra_time == 0
    assert result.advancing_team_id == "home"


def test_knockout_penalty_winner_is_not_regulation_win() -> None:
    matrix = ScoreMatrix(
        probabilities=np.array([[1.0]]),
        expected_home_goals=0.0,
        expected_away_goals=0.0,
        source="test",
        original_mass=1.0,
    )
    rng = FixedRng(choice_index=0, poisson_values=(0, 0), random_value=0.75)

    result = simulate_knockout_match(
        match_number=73,
        source_fixture_id=None,
        stage="round_of_32",
        home_team_id="home",
        away_team_id="away",
        matrix=matrix,
        rng=rng,  # type: ignore[arg-type]
        extra_time_goal_scale=1 / 3,
    )

    assert result.home_goals_90 == 0
    assert result.away_goals_90 == 0
    assert result.decision_method == "penalties"
    assert result.penalty_winner_team_id == "away"
    assert result.advancing_team_id == "away"


def test_observed_knockout_preserves_penalty_result() -> None:
    result = observed_knockout_match(
        match_number=73,
        source_fixture_id="fixture-73",
        stage="round_of_32",
        home_team_id="home",
        away_team_id="away",
        home_goals_90=1,
        away_goals_90=1,
        home_goals_after_extra_time=1,
        away_goals_after_extra_time=1,
        home_penalty_goals=4,
        away_penalty_goals=5,
    )

    assert result.observed is True
    assert result.decision_method == "penalties"
    assert result.advancing_team_id == "away"
    assert result.home_goals_90 == 1

