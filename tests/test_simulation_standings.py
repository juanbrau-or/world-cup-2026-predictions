from __future__ import annotations

import numpy as np

from worldcup2026.simulation.standings import (
    GroupMatchResult,
    StandingRow,
    assign_group_classification,
    rank_best_thirds,
    rank_group,
)


def test_group_standings_points_goals_and_difference() -> None:
    result = rank_group(
        group="A",
        teams=["a", "b", "c", "d"],
        matches=[
            _match("a", "b", 2, 0),
            _match("c", "d", 1, 1),
            _match("a", "c", 0, 0),
            _match("b", "d", 1, 0),
            _match("a", "d", 3, 1),
            _match("b", "c", 2, 2),
        ],
        rng=np.random.default_rng(1),
    )

    rows = {row.team_id: row for row in result.rows}
    assert rows["a"].points == 7
    assert rows["a"].goals_for == 5
    assert rows["a"].goals_against == 1
    assert rows["a"].goal_difference == 4
    assert rows["a"].position == 1


def test_two_team_tie_uses_head_to_head_before_global_goal_difference() -> None:
    result = rank_group(
        group="A",
        teams=["a", "b", "c", "d"],
        matches=[
            _match("a", "b", 1, 0),
            _match("a", "c", 0, 3),
            _match("a", "d", 1, 0),
            _match("b", "c", 3, 0),
            _match("b", "d", 3, 0),
            _match("c", "d", 0, 0),
        ],
        rng=np.random.default_rng(1),
    )

    assert [row.team_id for row in result.rows[:2]] == ["a", "b"]
    assert result.rows[0].tie_break_resolution == "head_to_head_points"


def test_three_team_tie_uses_head_to_head_table() -> None:
    result = rank_group(
        group="A",
        teams=["a", "b", "c", "d"],
        matches=[
            _match("a", "b", 1, 0),
            _match("b", "c", 2, 0),
            _match("c", "a", 1, 0),
            _match("a", "d", 1, 0),
            _match("b", "d", 1, 0),
            _match("c", "d", 4, 0),
        ],
        rng=np.random.default_rng(1),
    )

    assert [row.team_id for row in result.rows[:3]] == ["b", "a", "c"]
    assert result.rows[0].tie_break_resolution == "head_to_head_goal_difference"


def test_unmodelled_fair_play_reaches_deterministic_random_lot_fallback() -> None:
    matches = [
        _match("a", "b", 0, 0),
        _match("a", "c", 0, 0),
        _match("a", "d", 0, 0),
        _match("b", "c", 0, 0),
        _match("b", "d", 0, 0),
        _match("c", "d", 0, 0),
    ]

    first = rank_group(
        group="A",
        teams=["a", "b", "c", "d"],
        matches=matches,
        rng=np.random.default_rng(123),
    )
    second = rank_group(
        group="A",
        teams=["a", "b", "c", "d"],
        matches=matches,
        rng=np.random.default_rng(123),
    )

    assert first.fallback_counts == {"random_lot_proxy": 1}
    assert [row.team_id for row in first.rows] == [row.team_id for row in second.rows]
    assert {row.tie_break_resolution for row in first.rows} == {"random_lot_proxy"}


def test_best_thirds_selects_exact_qualifier_count_and_ranks() -> None:
    rows = [
        _row("a", "A", points=5, gd=1, gf=3),
        _row("b", "B", points=4, gd=3, gf=4),
        _row("c", "C", points=4, gd=2, gf=5),
        _row("d", "D", points=2, gd=0, gf=2),
    ]

    result = rank_best_thirds(rows, qualifiers=2, rng=np.random.default_rng(1))
    classified = assign_group_classification(result.rows, ["a", "b"])

    assert [row.team_id for row in result.rows] == ["a", "b", "c", "d"]
    assert [row.classification_status for row in result.rows] == [
        "best_third",
        "best_third",
        "group_eliminated",
        "group_eliminated",
    ]
    assert [row.third_place_rank for row in result.rows] == [1, 2, 3, 4]
    assert {row.team_id: row.classification_status for row in classified}["a"] == "best_third"


def _match(home: str, away: str, home_goals: int, away_goals: int) -> GroupMatchResult:
    return GroupMatchResult(
        group="A",
        home_team_id=home,
        away_team_id=away,
        home_goals=home_goals,
        away_goals=away_goals,
    )


def _row(team: str, group: str, *, points: int, gd: int, gf: int) -> StandingRow:
    return StandingRow(
        team_id=team,
        group=group,
        played=3,
        wins=0,
        draws=0,
        losses=0,
        goals_for=gf,
        goals_against=gf - gd,
        goal_difference=gd,
        points=points,
        position=3,
    )
