from __future__ import annotations

import pytest

from worldcup2026.simulation.rules import GROUPS
from worldcup2026.simulation.standings import StandingRow
from worldcup2026.simulation.tournament import (
    TournamentSimulationError,
    _build_round_of_32,
    _resolve_dependency,
    _validate_round_of_32,
)


def test_round_of_32_builds_16_fixtures_and_32_unique_teams() -> None:
    standings_by_group, third_ranking, classified = _classified_rows()

    matches = _build_round_of_32(standings_by_group, third_ranking)

    assert len(matches) == 16
    assert matches[0][0] == 73
    assert matches[-1][0] == 88
    teams = [team for _, home, away in matches for team in (home, away)]
    assert len(teams) == 32
    assert len(set(teams)) == 32
    _validate_round_of_32(matches, classified)


def test_round_of_32_rejects_duplicate_team() -> None:
    standings_by_group, third_ranking, classified = _classified_rows()
    matches = _build_round_of_32(standings_by_group, third_ranking)
    duplicate = [(matches[0][0], matches[0][1], matches[0][1]), *matches[1:]]

    with pytest.raises(TournamentSimulationError, match="32 unique"):
        _validate_round_of_32(duplicate, classified)


def test_round_of_32_rejects_eliminated_team() -> None:
    standings_by_group, third_ranking, classified = _classified_rows()
    matches = _build_round_of_32(standings_by_group, third_ranking)
    first_match, *_ = matches
    eliminated = next(
        row.team_id
        for row in classified.values()
        if row.classification_status == "group_eliminated"
    )
    invalid = [(first_match[0], eliminated, first_match[2]), *matches[1:]]

    with pytest.raises(TournamentSimulationError, match="do not match group qualifiers"):
        _validate_round_of_32(invalid, classified)


def test_next_round_dependencies_use_winners_and_losers() -> None:
    winners = {101: "semi_one_winner"}
    losers = {101: "semi_one_loser"}

    assert _resolve_dependency("W101", winners, losers) == "semi_one_winner"
    assert _resolve_dependency("L101", winners, losers) == "semi_one_loser"


def _classified_rows() -> tuple[
    dict[str, tuple[StandingRow, ...]],
    tuple[StandingRow, ...],
    dict[str, StandingRow],
]:
    standings_by_group: dict[str, tuple[StandingRow, ...]] = {}
    classified: dict[str, StandingRow] = {}
    third_rows: list[StandingRow] = []
    best_third_groups = set("EFGHIJKL")
    for group in GROUPS:
        rows: list[StandingRow] = []
        for position in range(1, 5):
            status = "direct" if position <= 2 else "group_eliminated"
            if position == 3 and group in best_third_groups:
                status = "best_third"
            row = StandingRow(
                team_id=f"{group.lower()}{position}",
                group=group,
                played=3,
                wins=0,
                draws=0,
                losses=0,
                goals_for=0,
                goals_against=0,
                goal_difference=0,
                points=0,
                position=position,
                classification_status=status,
            )
            rows.append(row)
            classified[row.team_id] = row
            if position == 3:
                third_rows.append(row)
        standings_by_group[group] = tuple(rows)
    third_ranking = tuple(
        row for group in "EFGHIJKL" for row in third_rows if row.group == group
    ) + tuple(row for group in "ABCD" for row in third_rows if row.group == group)
    return standings_by_group, third_ranking, classified
