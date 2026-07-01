from __future__ import annotations

from datetime import UTC, datetime

from worldcup2026.simulation.tournament import TournamentFixture, TournamentState, _state_as_of


def test_as_of_view_clears_future_results_without_mutating_raw_state() -> None:
    cutoff = datetime(2026, 6, 12, 12, tzinfo=UTC)
    state = TournamentState(
        fixtures=(
            _fixture(
                source_fixture_id="group-future",
                stage="group_stage",
                group="A",
                kickoff=datetime(2026, 6, 13, 19, tzinfo=UTC),
                status="played",
                home="a1",
                away="a2",
                home_goals=2,
                away_goals=0,
            ),
            _fixture(
                source_fixture_id="knockout-future",
                stage="round_of_32",
                group=None,
                kickoff=datetime(2026, 6, 29, 19, tzinfo=UTC),
                status="scheduled",
                home="winner_a",
                away="third_e",
                home_goals=None,
                away_goals=None,
            ),
        ),
        data_cutoff_utc=datetime(2026, 7, 1, tzinfo=UTC),
        snapshot_checksum="a" * 64,
        snapshot_reference="raw",
        raw_snapshot_path=None,
        source_fixture_count=2,
    )

    adjusted = _state_as_of(state, cutoff=cutoff)

    group_fixture = adjusted.fixtures[0]
    knockout_fixture = adjusted.fixtures[1]
    assert group_fixture.status == "scheduled"
    assert group_fixture.home_team_id == "a1"
    assert group_fixture.home_goals_90 is None
    assert knockout_fixture.status == "scheduled"
    assert knockout_fixture.home_team_id is None
    assert knockout_fixture.away_team_id is None
    assert state.fixtures[0].status == "played"
    assert state.fixtures[0].home_goals_90 == 2


def test_as_of_view_treats_recent_observed_match_as_in_progress() -> None:
    cutoff = datetime(2026, 6, 12, 20, tzinfo=UTC)
    state = TournamentState(
        fixtures=(
            _fixture(
                source_fixture_id="recent",
                stage="group_stage",
                group="A",
                kickoff=datetime(2026, 6, 12, 19, tzinfo=UTC),
                status="played",
                home="a1",
                away="a2",
                home_goals=1,
                away_goals=0,
            ),
        ),
        data_cutoff_utc=datetime(2026, 7, 1, tzinfo=UTC),
        snapshot_checksum="a" * 64,
        snapshot_reference="raw",
        raw_snapshot_path=None,
        source_fixture_count=1,
    )

    adjusted = _state_as_of(state, cutoff=cutoff)

    assert adjusted.fixtures[0].status == "in_progress"
    assert adjusted.fixtures[0].home_goals_90 is None
    assert adjusted.fixtures[0].home_team_id == "a1"


def _fixture(
    *,
    source_fixture_id: str,
    stage: str,
    group: str | None,
    kickoff: datetime,
    status: str,
    home: str | None,
    away: str | None,
    home_goals: int | None,
    away_goals: int | None,
) -> TournamentFixture:
    return TournamentFixture(
        source_fixture_id=source_fixture_id,
        stage=stage,
        group=group,
        kickoff_utc=kickoff,
        status=status,
        source_status=status.upper(),
        home_team_id=home,
        away_team_id=away,
        home_team_name=None,
        away_team_name=None,
        home_goals_90=home_goals,
        away_goals_90=away_goals,
        home_goals_after_extra_time=None,
        away_goals_after_extra_time=None,
        home_penalty_goals=None,
        away_penalty_goals=None,
    )

