"""Group standings and best-third rankings for World Cup tournament simulations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np


class StandingsError(RuntimeError):
    """Raised when group standings cannot be computed safely."""


@dataclass(frozen=True)
class GroupMatchResult:
    """One group-stage result at 90 minutes."""

    group: str
    home_team_id: str
    away_team_id: str
    home_goals: int
    away_goals: int
    source_fixture_id: str | None = None
    observed: bool = False


@dataclass(frozen=True)
class StandingRow:
    """Deterministic team standing row."""

    team_id: str
    group: str
    played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    goal_difference: int
    points: int
    position: int
    classification_status: str = "pending"
    tie_break_resolution: str | None = None
    third_place_rank: int | None = None


@dataclass(frozen=True)
class StandingsResult:
    """Resolved standings plus audit counters."""

    rows: tuple[StandingRow, ...]
    fallback_counts: Mapping[str, int]


def rank_group(
    *,
    group: str,
    teams: Sequence[str],
    matches: Sequence[GroupMatchResult],
    rng: np.random.Generator,
) -> StandingsResult:
    """Rank one four-team group using the FIFA World Cup 2026 tiebreaker order."""

    if len(set(teams)) != len(teams):
        raise StandingsError(f"group {group} contains duplicate teams")
    if len(teams) < 2:
        raise StandingsError(f"group {group} must contain at least two teams")
    stats = _initial_stats(group=group, teams=teams)
    for match in matches:
        if match.group != group:
            continue
        if match.home_team_id not in stats or match.away_team_id not in stats:
            raise StandingsError(f"group {group} match includes a team outside the group")
        _apply_match(stats, match)

    fallback_counts: Counter[str] = Counter()
    points_buckets = _split_by_metric(
        list(teams),
        lambda team: stats[team].points,
        descending=True,
    )
    ordered: list[str] = []
    tie_resolution_by_team: dict[str, str | None] = {team: None for team in teams}
    for bucket in points_buckets:
        if len(bucket) == 1:
            ordered.extend(bucket)
            continue
        bucket_order, resolutions = _rank_equal_points_bucket(
            bucket,
            matches=matches,
            stats=stats,
            rng=rng,
            fallback_counts=fallback_counts,
            global_metric_index=0,
        )
        ordered.extend(bucket_order)
        tie_resolution_by_team.update(resolutions)

    rows = tuple(
        replace(
            stats[team],
            position=index + 1,
            tie_break_resolution=tie_resolution_by_team.get(team),
        )
        for index, team in enumerate(ordered)
    )
    return StandingsResult(rows=rows, fallback_counts=dict(fallback_counts))


def rank_best_thirds(
    third_rows: Sequence[StandingRow],
    *,
    qualifiers: int,
    rng: np.random.Generator,
) -> StandingsResult:
    """Rank third-placed teams globally and mark the teams selected as best thirds."""

    if qualifiers < 0:
        raise StandingsError("qualifiers cannot be negative")
    if qualifiers > len(third_rows):
        raise StandingsError("qualifiers cannot exceed the number of third-placed teams")
    fallback_counts: Counter[str] = Counter()
    by_team = {row.team_id: row for row in third_rows}
    ordered, resolutions = _rank_global_rows(
        list(by_team),
        rows=by_team,
        rng=rng,
        fallback_counts=fallback_counts,
        metric_index=0,
    )
    rows: list[StandingRow] = []
    for index, team_id in enumerate(ordered):
        status = "best_third" if index < qualifiers else "group_eliminated"
        rows.append(
            replace(
                by_team[team_id],
                classification_status=status,
                third_place_rank=index + 1,
                tie_break_resolution=resolutions.get(team_id),
            )
        )
    return StandingsResult(rows=tuple(rows), fallback_counts=dict(fallback_counts))


def assign_group_classification(
    rows: Sequence[StandingRow],
    best_third_team_ids: Iterable[str],
) -> tuple[StandingRow, ...]:
    """Mark direct qualifiers, best thirds and group-eliminated teams."""

    best_thirds = set(best_third_team_ids)
    output: list[StandingRow] = []
    for row in rows:
        if row.position <= 2:
            status = "direct"
        elif row.position == 3 and row.team_id in best_thirds:
            status = "best_third"
        else:
            status = "group_eliminated"
        output.append(replace(row, classification_status=status))
    return tuple(output)


def _initial_stats(*, group: str, teams: Sequence[str]) -> dict[str, StandingRow]:
    return {
        team: StandingRow(
            team_id=team,
            group=group,
            played=0,
            wins=0,
            draws=0,
            losses=0,
            goals_for=0,
            goals_against=0,
            goal_difference=0,
            points=0,
            position=0,
        )
        for team in teams
    }


def _apply_match(stats: dict[str, StandingRow], match: GroupMatchResult) -> None:
    home_points, away_points = _points(match.home_goals, match.away_goals)
    home_wdl = _wdl(match.home_goals, match.away_goals)
    away_wdl = _wdl(match.away_goals, match.home_goals)
    stats[match.home_team_id] = _updated_row(
        stats[match.home_team_id],
        goals_for=match.home_goals,
        goals_against=match.away_goals,
        points=home_points,
        wdl=home_wdl,
    )
    stats[match.away_team_id] = _updated_row(
        stats[match.away_team_id],
        goals_for=match.away_goals,
        goals_against=match.home_goals,
        points=away_points,
        wdl=away_wdl,
    )


def _updated_row(
    row: StandingRow,
    *,
    goals_for: int,
    goals_against: int,
    points: int,
    wdl: tuple[int, int, int],
) -> StandingRow:
    wins, draws, losses = wdl
    new_goals_for = row.goals_for + goals_for
    new_goals_against = row.goals_against + goals_against
    return replace(
        row,
        played=row.played + 1,
        wins=row.wins + wins,
        draws=row.draws + draws,
        losses=row.losses + losses,
        goals_for=new_goals_for,
        goals_against=new_goals_against,
        goal_difference=new_goals_for - new_goals_against,
        points=row.points + points,
    )


def _points(goals_for: int, goals_against: int) -> tuple[int, int]:
    if goals_for > goals_against:
        return 3, 0
    if goals_for < goals_against:
        return 0, 3
    return 1, 1


def _wdl(goals_for: int, goals_against: int) -> tuple[int, int, int]:
    if goals_for > goals_against:
        return 1, 0, 0
    if goals_for < goals_against:
        return 0, 0, 1
    return 0, 1, 0


def _rank_equal_points_bucket(
    teams: Sequence[str],
    *,
    matches: Sequence[GroupMatchResult],
    stats: Mapping[str, StandingRow],
    rng: np.random.Generator,
    fallback_counts: Counter[str],
    global_metric_index: int,
) -> tuple[list[str], dict[str, str | None]]:
    h2h_split = _first_head_to_head_split(teams, matches)
    if h2h_split is not None:
        ordered: list[str] = []
        resolutions: dict[str, str | None] = {}
        for bucket, criterion in h2h_split:
            if len(bucket) == 1:
                ordered.extend(bucket)
                resolutions[bucket[0]] = criterion
            else:
                sub_order, sub_resolutions = _rank_equal_points_bucket(
                    bucket,
                    matches=matches,
                    stats=stats,
                    rng=rng,
                    fallback_counts=fallback_counts,
                    global_metric_index=0,
                )
                ordered.extend(sub_order)
                resolutions.update(sub_resolutions)
        return ordered, resolutions
    return _rank_global_rows(
        list(teams),
        rows=stats,
        rng=rng,
        fallback_counts=fallback_counts,
        metric_index=global_metric_index,
    )


def _first_head_to_head_split(
    teams: Sequence[str],
    matches: Sequence[GroupMatchResult],
) -> list[tuple[list[str], str]] | None:
    h2h_stats = _head_to_head_stats(teams, matches)
    criteria: tuple[tuple[str, Callable[[str], int]], ...] = (
        ("head_to_head_points", lambda team: h2h_stats[team].points),
        ("head_to_head_goal_difference", lambda team: h2h_stats[team].goal_difference),
        ("head_to_head_goals_for", lambda team: h2h_stats[team].goals_for),
    )
    for criterion, metric in criteria:
        buckets = _split_by_metric(list(teams), metric, descending=True)
        if len(buckets) > 1:
            return [(bucket, criterion) for bucket in buckets]
    return None


def _head_to_head_stats(
    teams: Sequence[str],
    matches: Sequence[GroupMatchResult],
) -> dict[str, StandingRow]:
    team_set = set(teams)
    stats = _initial_stats(group="head_to_head", teams=teams)
    for match in matches:
        if match.home_team_id in team_set and match.away_team_id in team_set:
            _apply_match(stats, match)
    return stats


def _rank_global_rows(
    teams: list[str],
    *,
    rows: Mapping[str, StandingRow],
    rng: np.random.Generator,
    fallback_counts: Counter[str],
    metric_index: int,
) -> tuple[list[str], dict[str, str | None]]:
    metrics: tuple[tuple[str, Callable[[StandingRow], int]], ...] = (
        ("points", lambda row: row.points),
        ("goal_difference", lambda row: row.goal_difference),
        ("goals_for", lambda row: row.goals_for),
    )
    for index in range(metric_index, len(metrics)):
        criterion, metric = metrics[index]
        current_metric = metric

        def row_metric(
            team: str,
            selected_metric: Callable[[StandingRow], int] = current_metric,
        ) -> int:
            return selected_metric(rows[team])

        buckets = _split_by_metric(
            teams,
            row_metric,
            descending=True,
        )
        if len(buckets) == 1:
            continue
        ordered: list[str] = []
        resolutions: dict[str, str | None] = {}
        for bucket in buckets:
            if len(bucket) == 1:
                ordered.extend(bucket)
                resolutions[bucket[0]] = criterion
            else:
                sub_order, sub_resolutions = _rank_global_rows(
                    bucket,
                    rows=rows,
                    rng=rng,
                    fallback_counts=fallback_counts,
                    metric_index=index + 1,
                )
                ordered.extend(sub_order)
                resolutions.update(sub_resolutions)
        return ordered, resolutions
    fallback_counts["random_lot_proxy"] += 1
    shuffled = list(sorted(teams))
    rng.shuffle(shuffled)
    return shuffled, {team: "random_lot_proxy" for team in shuffled}


def _split_by_metric(
    teams: list[str],
    metric: Callable[[str], int],
    *,
    descending: bool,
) -> list[list[str]]:
    values: dict[int, list[str]] = {}
    for team in teams:
        values.setdefault(metric(team), []).append(team)
    ordered_values = sorted(values, reverse=descending)
    return [sorted(values[value]) for value in ordered_values]
