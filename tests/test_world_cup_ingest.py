"""Offline tests for the append-only World Cup 2026 live ingestion boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import pytest

from worldcup2026.data.contracts import resolve_team_alias
from worldcup2026.data.historical_ingest import load_team_aliases, load_team_catalog
from worldcup2026.data.world_cup_ingest import (
    ApiFootballProvider,
    FootballDataProvider,
    MissingApiKeyError,
    ProviderRequestError,
    RateLimitedHttpClient,
    WorldCupIngestResult,
    offline_provider,
    run_world_cup_ingest,
)

FIXTURE = Path("tests/fixtures/world_cup_2026/football_data_matches.json")


class FakeTransport:
    """A deterministic transport response queue."""

    def __init__(self, outcomes: list[httpx.Response | Exception]) -> None:
        self.outcomes = outcomes
        self.calls = 0
        self.headers: list[dict[str, str]] = []

    def get(self, url: str, *, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        del url, timeout
        self.headers.append(headers)
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def run_offline(tmp_path: Path, *, fetched_at: datetime) -> WorldCupIngestResult:
    return run_world_cup_ingest(
        offline_provider(FIXTURE),
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        interim_root=tmp_path / "interim",
        fetched_at=fetched_at,
    )


def test_offline_ingest_snapshots_normalizes_and_partitions(tmp_path: Path) -> None:
    result = run_offline(tmp_path, fetched_at=datetime(2026, 6, 20, 18, tzinfo=UTC))

    assert len(result.matches) == 2
    assert result.report.provider_fixtures_received == 3
    assert result.report.fixtures_with_known_participants == 2
    assert result.report.fixtures_with_tbd_participants == 1
    assert result.report.invalid_fixtures == 0
    assert result.report.pending_fixtures == 1
    assert result.report.in_progress_matches == 0
    assert result.report.finished_matches == 1
    assert result.report.unresolved_teams == ()
    assert result.report.freshness.next_kickoff_utc == datetime(2026, 6, 21, 18, tzinfo=UTC)
    assert result.report.snapshot_manifests[0].checksum
    assert result.report.snapshot_manifests[0].raw_path.is_file()
    assert (tmp_path / "processed" / "pending.parquet").is_file()
    assert (tmp_path / "processed" / "in_progress.parquet").is_file()
    assert (tmp_path / "processed" / "finished.parquet").is_file()

    assert all(item.source_match_id != "1002" for item in result.matches)
    assert result.tbd_fixtures[0].source_fixture_id == "1002"
    assert result.tbd_fixtures[0].participants_status == "tbd"
    assert result.tbd_fixtures[0].score["home_goals_90"] is None
    assert result.tbd_fixtures[0].raw_snapshot_path.is_file()
    assert json.loads((tmp_path / "interim" / "world_cup_2026_tbd_fixtures.json").read_text())


def test_snapshot_is_immutable_and_repeat_is_idempotent(tmp_path: Path) -> None:
    fetched_at = datetime(2026, 6, 20, 18, tzinfo=UTC)
    first = run_offline(tmp_path, fetched_at=fetched_at)
    raw_path = first.report.snapshot_manifests[0].raw_path
    original = raw_path.read_bytes()

    second = run_offline(tmp_path, fetched_at=fetched_at)

    assert raw_path.read_bytes() == original
    assert second.report.freshness.differences_from_previous_snapshot == ()
    assert len(list((tmp_path / "processed" / "snapshots").glob("*/matches.parquet"))) == 1


def test_transition_scheduled_to_finished_preserves_previous_snapshot(tmp_path: Path) -> None:
    initial = json.loads(FIXTURE.read_text(encoding="utf-8"))
    updated = json.loads(FIXTURE.read_text(encoding="utf-8"))
    match = next(item for item in updated["matches"] if item["id"] == 1001)
    match["status"] = "FINISHED"
    match["lastUpdated"] = "2026-06-21T20:00:00Z"
    match["score"] = {
        "duration": "REGULAR",
        "regularTime": {"home": 2, "away": 0},
        "fullTime": {"home": 2, "away": 0},
    }
    initial_path = tmp_path / "initial.json"
    updated_path = tmp_path / "updated.json"
    initial_path.write_text(json.dumps(initial), encoding="utf-8")
    updated_path.write_text(json.dumps(updated), encoding="utf-8")
    common = {
        "raw_root": tmp_path / "raw",
        "processed_root": tmp_path / "processed",
        "interim_root": tmp_path / "interim",
    }
    run_world_cup_ingest(
        offline_provider(initial_path),
        fetched_at=datetime(2026, 6, 20, 18, tzinfo=UTC),
        **common,
    )
    result = run_world_cup_ingest(
        offline_provider(updated_path),
        fetched_at=datetime(2026, 6, 21, 21, tzinfo=UTC),
        **common,
    )

    assert result.report.finished_matches == 2
    assert any(
        change.source_fixture_id == "1001" and change.field == "match_status"
        for change in result.report.freshness.differences_from_previous_snapshot
    )
    assert (
        len(list((tmp_path / "raw" / "world_cup_2026" / "football_data").glob("*/response.json")))
        == 2
    )


def test_dry_run_writes_no_snapshot_or_operational_view(tmp_path: Path) -> None:
    result = run_world_cup_ingest(
        offline_provider(FIXTURE),
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        interim_root=tmp_path / "interim",
        fetched_at=datetime(2026, 6, 20, 18, tzinfo=UTC),
        dry_run=True,
    )

    assert result.report.operational_table is None
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "processed").exists()


def test_http_retries_timeout_and_hides_api_key(tmp_path: Path) -> None:
    transport = FakeTransport([httpx.ReadTimeout("timeout"), httpx.ReadTimeout("timeout")])
    client = RateLimitedHttpClient(
        cache_root=tmp_path / "cache",
        timeout_seconds=1,
        retries=1,
        min_interval_seconds=0,
        cache_ttl_seconds=0,
        transport=transport,
        sleeper=lambda _: None,
    )
    provider = FootballDataProvider(
        api_key="not-for-logs",
        base_url="https://provider.invalid",
        competition_code="WC",
        client=client,
    )

    with pytest.raises(ProviderRequestError) as exc_info:
        provider.fetch()

    assert transport.calls == 2
    assert "not-for-logs" not in str(exc_info.value)


def test_http_success_and_rate_limit(tmp_path: Path) -> None:
    transport = FakeTransport(
        [
            httpx.Response(
                200,
                content=b'{"matches": []}',
                request=httpx.Request("GET", "https://provider.invalid/a"),
            ),
            httpx.Response(
                200,
                content=b'{"matches": []}',
                request=httpx.Request("GET", "https://provider.invalid/b"),
            ),
        ]
    )
    sleeps: list[float] = []
    client = RateLimitedHttpClient(
        cache_root=tmp_path / "cache",
        timeout_seconds=1,
        retries=0,
        min_interval_seconds=2,
        cache_ttl_seconds=0,
        transport=transport,
        clock=lambda: 10,
        sleeper=sleeps.append,
    )

    assert client.get_json(
        url="https://provider.invalid/a", headers={"x": "y"}, cache_namespace="x"
    )
    assert client.get_json(
        url="https://provider.invalid/b", headers={"x": "y"}, cache_namespace="x"
    )
    assert sleeps == [2]


def test_football_data_provider_downloads_and_parses_fixture(tmp_path: Path) -> None:
    transport = FakeTransport(
        [
            httpx.Response(
                200,
                content=FIXTURE.read_bytes(),
                request=httpx.Request("GET", "https://provider.invalid/competitions/WC/matches"),
            )
        ]
    )
    client = RateLimitedHttpClient(
        cache_root=tmp_path / "cache",
        timeout_seconds=1,
        retries=0,
        min_interval_seconds=0,
        cache_ttl_seconds=0,
        transport=transport,
    )
    provider = FootballDataProvider(
        api_key="test-key",
        base_url="https://provider.invalid",
        competition_code="WC",
        client=client,
    )

    response = provider.fetch()

    assert response.provider == "football_data"
    assert len(response.fixtures) == 3
    assert response.fixture_payloads["1001"]
    assert transport.headers == [{"X-Auth-Token": "test-key"}]
    assert response.endpoint.endswith("/matches?season=2026")


def test_api_football_regular_result_does_not_imply_extra_time(tmp_path: Path) -> None:
    payload = {
        "response": [
            {
                "fixture": {
                    "id": 5,
                    "date": "2026-06-20T18:00:00+00:00",
                    "timezone": "UTC",
                    "status": {"short": "FT"},
                    "venue": {"name": "Venue", "city": "City"},
                },
                "league": {"round": "Group Stage"},
                "teams": {"home": {"name": "Spain"}, "away": {"name": "France"}},
                "score": {
                    "fulltime": {"home": 2, "away": 1},
                    "extratime": {"home": None, "away": None},
                    "penalty": {"home": None, "away": None},
                },
            }
        ]
    }
    transport = FakeTransport(
        [
            httpx.Response(
                200,
                content=json.dumps(payload).encode("utf-8"),
                request=httpx.Request(
                    "GET", "https://provider.invalid/fixtures?league=1&season=2026"
                ),
            )
        ]
    )
    client = RateLimitedHttpClient(
        cache_root=tmp_path / "cache",
        timeout_seconds=1,
        retries=0,
        min_interval_seconds=0,
        cache_ttl_seconds=0,
        transport=transport,
    )
    response = ApiFootballProvider(
        api_key="test-key",
        base_url="https://provider.invalid",
        league_id=1,
        client=client,
    ).fetch()

    assert response.fixtures[0].extra_time_played is False
    assert response.fixtures[0].penalty_shootout is False


def test_missing_api_key_is_actionable_without_serializing_secret(tmp_path: Path) -> None:
    client = RateLimitedHttpClient(
        cache_root=tmp_path / "cache",
        timeout_seconds=1,
        retries=0,
        min_interval_seconds=0,
        cache_ttl_seconds=0,
    )
    provider = FootballDataProvider(
        api_key=None,
        base_url="https://provider.invalid",
        competition_code="WC",
        client=client,
    )

    with pytest.raises(MissingApiKeyError, match="FOOTBALL_DATA_API_KEY"):
        provider.fetch()


def test_invalid_score_contract_is_reported(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    match = next(item for item in payload["matches"] if item["id"] == 1003)
    match["score"] = {
        "duration": "EXTRA_TIME",
        "regularTime": {"home": 2, "away": 1},
        "fullTime": {"home": 2, "away": 1},
        "extraTime": {"home": 2, "away": 1},
    }
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_world_cup_ingest(
        offline_provider(path),
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        interim_root=tmp_path / "interim",
        fetched_at=datetime(2026, 6, 20, 18, tzinfo=UTC),
    )

    assert result.report.invalid_fixtures == 1
    assert result.report.invalid_fixture_records[0].source_fixture_id == "1003"
    assert len(result.matches) == 1


def test_partially_determined_fixture_is_reported_separately(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    tbd = next(item for item in payload["matches"] if item["id"] == 1002)
    tbd["homeTeam"]["name"] = "Mexico"
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_world_cup_ingest(
        offline_provider(path),
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        interim_root=tmp_path / "interim",
        fetched_at=datetime(2026, 6, 20, 18, tzinfo=UTC),
    )

    assert result.report.fixtures_with_tbd_participants == 0
    assert result.report.fixtures_with_partially_known_participants == 1
    assert result.report.invalid_provider_fixtures == 0
    assert result.partially_known_fixtures[0].source_fixture_id == "1002"
    assert result.partially_known_fixtures[0].home_name == "Mexico"
    assert result.partially_known_fixtures[0].away_name is None
    assert all(item.source_match_id != "1002" for item in result.matches)


def test_tbd_transition_to_known_preserves_raw_snapshots_and_updates_operational_view(
    tmp_path: Path,
) -> None:
    initial = json.loads(FIXTURE.read_text(encoding="utf-8"))
    updated = json.loads(FIXTURE.read_text(encoding="utf-8"))
    tbd = next(item for item in updated["matches"] if item["id"] == 1002)
    tbd["homeTeam"]["name"] = "Germany"
    tbd["awayTeam"]["name"] = "Portugal"
    initial_path = tmp_path / "initial.json"
    updated_path = tmp_path / "updated.json"
    initial_path.write_text(json.dumps(initial), encoding="utf-8")
    updated_path.write_text(json.dumps(updated), encoding="utf-8")
    common = {
        "raw_root": tmp_path / "raw",
        "processed_root": tmp_path / "processed",
        "interim_root": tmp_path / "interim",
    }
    run_world_cup_ingest(
        offline_provider(initial_path),
        fetched_at=datetime(2026, 6, 20, 18, tzinfo=UTC),
        **common,
    )
    result = run_world_cup_ingest(
        offline_provider(updated_path),
        fetched_at=datetime(2026, 6, 21, 18, tzinfo=UTC),
        **common,
    )

    assert result.report.fixtures_with_tbd_participants == 0
    assert any(item.source_match_id == "1002" for item in result.matches)
    assert any(
        item.source_fixture_id == "1002" and item.field == "participants_status"
        for item in result.report.freshness.differences_from_previous_snapshot
    )
    raw_snapshots = list(
        (tmp_path / "raw" / "world_cup_2026" / "football_data").glob("*/response.json")
    )
    assert len(raw_snapshots) == 2
    snapshot_tbd_views = list((tmp_path / "processed" / "snapshots").glob("*/tbd_fixtures.json"))
    assert len(snapshot_tbd_views) == 2
    assert any(json.loads(path.read_text()) for path in snapshot_tbd_views)
    rows = pq.read_table(tmp_path / "processed" / "matches.parquet").to_pylist()
    assert sum(row["source_match_id"] == "1002" for row in rows) == 1


def test_tbd_is_neither_unresolved_alias_nor_prediction_candidate(tmp_path: Path) -> None:
    result = run_offline(tmp_path, fetched_at=datetime(2026, 6, 20, 18, tzinfo=UTC))

    assert all(item.source_fixture_id != "1002" for item in result.report.unresolved_teams)
    assert all(item.source_match_id != "1002" for item in result.matches)
    assert result.tbd_fixtures[0].home_name is None
    assert result.tbd_fixtures[0].away_name is None


def test_mixed_collection_and_empty_collection_are_processed_individually(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    invalid = {
        "id": 1004,
        "utcDate": "2026-06-29T19:00:00Z",
        "status": "TIMED",
        "stage": "LAST_32",
        "homeTeam": {"name": "Mexico"},
        "awayTeam": {"name": None},
        "score": {"duration": "REGULAR", "fullTime": {"home": None, "away": None}},
    }
    payload["matches"] = [payload["matches"][0], payload["matches"][1], invalid]
    mixed_path = tmp_path / "mixed.json"
    mixed_path.write_text(json.dumps(payload), encoding="utf-8")
    mixed = run_world_cup_ingest(
        offline_provider(mixed_path),
        raw_root=tmp_path / "mixed_raw",
        processed_root=tmp_path / "mixed_processed",
        interim_root=tmp_path / "mixed_interim",
        fetched_at=datetime(2026, 6, 20, 18, tzinfo=UTC),
    )
    empty_path = tmp_path / "empty.json"
    empty_path.write_text('{"matches": []}', encoding="utf-8")
    empty = run_world_cup_ingest(
        offline_provider(empty_path),
        raw_root=tmp_path / "empty_raw",
        processed_root=tmp_path / "empty_processed",
        interim_root=tmp_path / "empty_interim",
        fetched_at=datetime(2026, 6, 20, 18, tzinfo=UTC),
    )

    assert (mixed.report.provider_fixtures_received, mixed.report.canonical_matches) == (3, 1)
    assert mixed.report.fixtures_with_tbd_participants == 1
    assert mixed.report.fixtures_with_partially_known_participants == 1
    assert mixed.report.invalid_provider_fixtures == 0
    assert empty.report.provider_fixtures_received == 0
    assert empty.matches == ()


def test_real_diagnostic_collection_can_be_processed_without_copying_it(tmp_path: Path) -> None:
    diagnostic = Path("/tmp/football-data-diagnostic/matches_2026.json")
    if not diagnostic.is_file():
        pytest.skip("external Football-Data diagnostic fixture is unavailable")

    result = run_world_cup_ingest(
        offline_provider(diagnostic),
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        interim_root=tmp_path / "interim",
        fetched_at=datetime(2026, 6, 20, 18, tzinfo=UTC),
        dry_run=True,
    )

    assert result.report.provider_fixtures_received == 104
    assert result.report.fixtures_with_tbd_participants == 32
    assert result.report.fixtures_with_known_participants == 72
    assert result.report.invalid_fixtures == 0


def test_real_football_data_alias_variants_are_explicit() -> None:
    teams = load_team_catalog(Path("data/static/teams.csv"))
    aliases = load_team_aliases(Path("data/static/team_aliases.csv"), teams=teams)
    observed = {
        "Bosnia-Herzegovina": "bosnia_and_herzegovina",
        "Cape Verde Islands": "cape_verde",
        "Congo DR": "dr_congo",
        "Czechia": "czech_republic",
    }

    for source_name, expected_id in observed.items():
        assert (
            resolve_team_alias(
                source="world_cup_2026_football_data",
                source_name=source_name,
                match_date=datetime(2026, 6, 20, tzinfo=UTC).date(),
                aliases=aliases,
            )
            == expected_id
        )
