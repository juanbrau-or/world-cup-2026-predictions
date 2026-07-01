from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from worldcup2026.simulation.matches import ScoreMatrix
from worldcup2026.simulation.rules import GROUPS
from worldcup2026.simulation.tournament import (
    PredictionStore,
    TournamentFixture,
    TournamentState,
    _simulate_many,
)


class DrawScoreProvider:
    def score_matrix(self, _home_team_id: str, _away_team_id: str) -> ScoreMatrix:
        return ScoreMatrix(
            probabilities=np.array([[1.0]]),
            expected_home_goals=0.0,
            expected_away_goals=0.0,
            source="test",
            original_mass=1.0,
        )


def test_monte_carlo_reproducible_across_same_seed_and_chunking() -> None:
    state = _synthetic_state()
    first = _simulate_many(
        state=state,
        prediction_store=PredictionStore([]),
        score_provider=DrawScoreProvider(),  # type: ignore[arg-type]
        runs=5,
        seed=2026,
        chunk_size=1,
        extra_time_goal_scale=1 / 3,
        cutoff=state.data_cutoff_utc,
    )
    second = _simulate_many(
        state=state,
        prediction_store=PredictionStore([]),
        score_provider=DrawScoreProvider(),  # type: ignore[arg-type]
        runs=5,
        seed=2026,
        chunk_size=3,
        extra_time_goal_scale=1 / 3,
        cutoff=state.data_cutoff_utc,
    )

    assert first.team_rows == second.team_rows
    assert first.simulation_rows == second.simulation_rows
    assert len(first.simulation_rows) == 5


def test_monte_carlo_different_seed_can_change_paths() -> None:
    state = _synthetic_state()
    first = _simulate_many(
        state=state,
        prediction_store=PredictionStore([]),
        score_provider=DrawScoreProvider(),  # type: ignore[arg-type]
        runs=10,
        seed=1,
        chunk_size=4,
        extra_time_goal_scale=1 / 3,
        cutoff=state.data_cutoff_utc,
    )
    second = _simulate_many(
        state=state,
        prediction_store=PredictionStore([]),
        score_provider=DrawScoreProvider(),  # type: ignore[arg-type]
        runs=10,
        seed=2,
        chunk_size=4,
        extra_time_goal_scale=1 / 3,
        cutoff=state.data_cutoff_utc,
    )

    assert first.simulation_rows != second.simulation_rows


def test_monte_carlo_zero_and_one_run_outputs_are_valid() -> None:
    state = _synthetic_state()
    zero = _simulate_many(
        state=state,
        prediction_store=PredictionStore([]),
        score_provider=DrawScoreProvider(),  # type: ignore[arg-type]
        runs=0,
        seed=2026,
        chunk_size=1,
        extra_time_goal_scale=1 / 3,
        cutoff=state.data_cutoff_utc,
    )
    one = _simulate_many(
        state=state,
        prediction_store=PredictionStore([]),
        score_provider=DrawScoreProvider(),  # type: ignore[arg-type]
        runs=1,
        seed=2026,
        chunk_size=1,
        extra_time_goal_scale=1 / 3,
        cutoff=state.data_cutoff_utc,
    )

    assert len(zero.team_rows) == 48
    assert zero.simulation_rows == ()
    assert len(one.simulation_rows) == 1
    assert sum(float(row["champion"]) for row in one.team_rows) == 1.0
    for row in one.team_rows:
        assert float(row["champion"]) <= float(row["final"]) <= float(row["semi_final"])
        assert float(row["semi_final"]) <= float(row["quarter_final"])
        assert float(row["quarter_final"]) <= float(row["round_of_16"])
        assert float(row["round_of_16"]) <= float(row["round_of_32"])


def _synthetic_state() -> TournamentState:
    fixtures: list[TournamentFixture] = []
    kickoff = datetime(2026, 6, 11, 19, tzinfo=UTC)
    source_id = 1
    for group in GROUPS:
        teams = [f"{group.lower()}{index}" for index in range(1, 5)]
        pairings = ((0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2))
        for home_index, away_index in pairings:
            fixtures.append(
                TournamentFixture(
                    source_fixture_id=f"g{source_id}",
                    stage="group_stage",
                    group=group,
                    kickoff_utc=kickoff,
                    status="scheduled",
                    source_status="TIMED",
                    home_team_id=teams[home_index],
                    away_team_id=teams[away_index],
                    home_team_name=None,
                    away_team_name=None,
                    home_goals_90=None,
                    away_goals_90=None,
                    home_goals_after_extra_time=None,
                    away_goals_after_extra_time=None,
                    home_penalty_goals=None,
                    away_penalty_goals=None,
                )
            )
            source_id += 1
    return TournamentState(
        fixtures=tuple(fixtures),
        data_cutoff_utc=datetime(2026, 6, 1, tzinfo=UTC),
        snapshot_checksum="a" * 64,
        snapshot_reference="synthetic",
        raw_snapshot_path=None,
        source_fixture_count=len(fixtures),
    )

