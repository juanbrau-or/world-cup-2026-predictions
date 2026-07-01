from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from worldcup2026.cli import app
from worldcup2026.features.contextual import (
    ContextualFeatureError,
    build_contextual_features,
    elevation_change_m,
    haversine_distance_km,
    load_venue_catalog,
    timezone_delta_hours,
    validate_contextual_feature_rows,
    write_contextual_feature_outputs,
)

runner = CliRunner()


def test_rest_load_extra_time_penalties_same_kickoff_and_ordering(tmp_path: Path) -> None:
    rows = [
        _match_row(
            "m3",
            kickoff=datetime(2026, 1, 20, 18, tzinfo=UTC),
            home="alpha",
            away="beta",
            venue="BC Place",
        ),
        _match_row(
            "m1",
            kickoff=datetime(2026, 1, 10, 18, tzinfo=UTC),
            home="alpha",
            away="beta",
            venue="BC Place",
        ),
        _match_row(
            "m4",
            kickoff=datetime(2026, 1, 25, 18, tzinfo=UTC),
            home="alpha",
            away="gamma",
            venue="Estadio Azteca",
        ),
        _match_row(
            "m2",
            kickoff=datetime(2026, 1, 15, 18, tzinfo=UTC),
            home="gamma",
            away="alpha",
            venue="Estadio Azteca",
            extra_time=True,
            penalties=True,
        ),
        _match_row(
            "m5",
            kickoff=datetime(2026, 1, 25, 18, tzinfo=UTC),
            home="delta",
            away="alpha",
            venue="Estadio Azteca",
        ),
    ]

    result = _build(tmp_path, rows)
    alpha_m3 = _team_row(result.team_rows, "m3", "alpha")
    alpha_m4 = _team_row(result.team_rows, "m4", "alpha")
    alpha_m5 = _team_row(result.team_rows, "m5", "alpha")

    assert alpha_m3["rest_hours"] == pytest.approx(120)
    assert alpha_m3["rest_days"] == pytest.approx(5)
    assert alpha_m3["matches_last_7d"] == 1
    assert alpha_m3["matches_last_14d"] == 2
    assert alpha_m3["minutes_equivalent_last_7d"] == 120
    assert alpha_m3["previous_match_extra_time"] is True
    assert alpha_m3["previous_match_penalty_shootout"] is True
    assert alpha_m3["consecutive_matches_without_7d_rest"] == 2
    assert alpha_m3["tournament_match_number"] == 3
    assert alpha_m3["is_first_tournament_match"] is False

    assert alpha_m4["previous_match_id"] == "world_cup_2026_football_data:m3"
    assert alpha_m5["previous_match_id"] == "world_cup_2026_football_data:m3"
    assert alpha_m4["previous_match_kickoff_utc"] < alpha_m4["kickoff_utc"]
    assert alpha_m5["previous_match_kickoff_utc"] < alpha_m5["kickoff_utc"]


def test_cutoff_excludes_previous_result_not_known_as_of_current_fixture(tmp_path: Path) -> None:
    rows = [
        _match_row(
            "known-late",
            kickoff=datetime(2026, 1, 15, 18, tzinfo=UTC),
            home="alpha",
            away="beta",
            data_cutoff=datetime(2026, 1, 16, 0, tzinfo=UTC),
        ),
        _match_row(
            "current",
            kickoff=datetime(2026, 1, 20, 18, tzinfo=UTC),
            home="alpha",
            away="gamma",
            data_cutoff=datetime(2026, 1, 14, 0, tzinfo=UTC),
        ),
        _match_row(
            "future",
            kickoff=datetime(2026, 1, 25, 18, tzinfo=UTC),
            home="alpha",
            away="delta",
            data_cutoff=datetime(2026, 1, 14, 0, tzinfo=UTC),
        ),
    ]

    result = _build(tmp_path, rows, generated_at=datetime(2026, 1, 16, 1, tzinfo=UTC))
    current = _team_row(result.team_rows, "current", "alpha")

    assert current["previous_match_id"] is None
    assert current["matches_last_7d"] == 0
    assert current["rest_hours"] is None
    assert result.leakage_audit["passed"] is True


def test_travel_timezone_altitude_host_and_neutrality(tmp_path: Path) -> None:
    rows = [
        _match_row(
            "canada-home",
            kickoff=datetime(2026, 6, 18, 20, tzinfo=UTC),
            home="canada",
            away="japan",
            venue="BC Place",
        ),
        _match_row(
            "canada-away",
            kickoff=datetime(2026, 6, 23, 20, tzinfo=UTC),
            home="mexico",
            away="canada",
            venue="Estadio Azteca",
        ),
    ]

    result = _build(tmp_path, rows)
    canada_away = _team_row(result.team_rows, "canada-away", "canada")
    mexico_home = _team_row(result.team_rows, "canada-away", "mexico")

    assert canada_away["venue_id"] == "mexico_city_stadium"
    assert canada_away["previous_venue_id"] == "bc_place_vancouver"
    assert canada_away["travel_distance_km"] == pytest.approx(
        haversine_distance_km(49.276666666, -123.111944444, 19.303055555, -99.150555555)
    )
    assert canada_away["cumulative_travel_km_7d"] == pytest.approx(
        canada_away["travel_distance_km"]
    )
    assert canada_away["timezone_delta_hours"] == pytest.approx(1)
    assert canada_away["elevation_change_m"] == pytest.approx(2227)
    assert canada_away["absolute_elevation_change_m"] == pytest.approx(2227)
    assert canada_away["cross_border_travel"] is True
    assert canada_away["host_country_match"] is False
    assert canada_away["is_neutral_venue"] is True
    assert mexico_home["host_country_match"] is True
    assert mexico_home["is_neutral_venue"] is False


def test_timezone_uses_real_offsets_and_dst() -> None:
    before_london_dst = timezone_delta_hours(
        "Europe/London",
        "America/New_York",
        datetime(2026, 3, 20, 12, tzinfo=UTC),
    )
    after_london_dst = timezone_delta_hours(
        "Europe/London",
        "America/New_York",
        datetime(2026, 4, 10, 12, tzinfo=UTC),
    )

    assert before_london_dst == pytest.approx(4)
    assert after_london_dst == pytest.approx(5)
    assert timezone_delta_hours(None, "America/New_York", datetime(2026, 1, 1, tzinfo=UTC)) is None
    with pytest.raises(ContextualFeatureError, match="invalid IANA timezone"):
        timezone_delta_hours("UTC-05", "America/New_York", datetime(2026, 1, 1, tzinfo=UTC))


def test_geography_and_altitude_validation() -> None:
    assert haversine_distance_km(1, 2, 1, 2) == 0
    assert haversine_distance_km(None, 2, 1, 2) is None
    with pytest.raises(ContextualFeatureError, match="latitude out of range"):
        haversine_distance_km(91, 2, 1, 2)
    with pytest.raises(ContextualFeatureError, match="longitude out of range"):
        haversine_distance_km(1, 181, 1, 2)

    assert elevation_change_m(120, 50) == 70
    assert elevation_change_m(50, 120) == -70
    assert elevation_change_m(None, 120) is None
    with pytest.raises(ContextualFeatureError, match="elevation_m out of reasonable range"):
        elevation_change_m(7000, 100)


def test_dataset_outputs_match_view_checksums_and_idempotence(tmp_path: Path) -> None:
    rows = [
        _match_row("m1", kickoff=datetime(2026, 1, 1, 18, tzinfo=UTC), home="alpha", away="beta"),
        _match_row("m2", kickoff=datetime(2026, 1, 8, 18, tzinfo=UTC), home="alpha", away="beta"),
    ]

    first = _build(tmp_path, rows)
    second = _build(tmp_path, rows)

    assert len(first.team_rows) == 4
    assert len(first.match_rows) == 2
    assert first.team_rows == second.team_rows
    assert first.match_rows == second.match_rows
    assert first.manifest["source_dataset_revision"] == second.manifest["source_dataset_revision"]
    match = next(row for row in first.match_rows if row["fixture_id"] == "m2")
    assert match["home_team_id"] == "alpha"
    assert match["away_team_id"] == "beta"
    assert match["matches_last_14d_diff"] == 0
    assert match["venue_catalog_checksum"]

    paths = write_contextual_feature_outputs(
        first,
        output_root=tmp_path / "processed",
        interim_root=tmp_path / "interim",
    )
    assert paths["team_fixture_parquet"].is_file()
    assert paths["match_parquet"].is_file()
    assert json.loads(paths["leakage_audit"].read_text(encoding="utf-8"))["passed"] is True


def test_validator_fails_leakage_and_duplicate_team_fixture(tmp_path: Path) -> None:
    rows = [
        _match_row(
            "future",
            kickoff=datetime(2026, 1, 20, 18, tzinfo=UTC),
            home="alpha",
            away="beta",
            status="scheduled",
        )
    ]
    result = _build(tmp_path, rows, generated_at=datetime(2026, 1, 21, 0, tzinfo=UTC))

    audit = validate_contextual_feature_rows(
        [*result.team_rows, result.team_rows[0]],
        result.match_rows,
    )

    assert audit.passed is False
    reasons = {str(item["reason"]) for item in audit.violations}
    assert "duplicate_team_fixture" in reasons
    assert "prospective_feature_generated_at_or_after_kickoff" in reasons


def test_venue_catalog_rejects_bad_timezone_and_bad_coordinates(tmp_path: Path) -> None:
    path = _venue_catalog_path(tmp_path)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("America/Vancouver", "UTC-08", 1), encoding="utf-8")

    with pytest.raises(ContextualFeatureError, match="invalid IANA timezone"):
        load_venue_catalog(path)

    path = _venue_catalog_path(tmp_path / "coords")
    path.write_text(text.replace("49.276666666", "100", 1), encoding="utf-8")
    with pytest.raises(ContextualFeatureError, match="latitude out of range"):
        load_venue_catalog(path)


def test_cli_offline_contextual_features_does_not_publish_parquet(tmp_path: Path) -> None:
    output_root = tmp_path / "processed"
    interim_root = tmp_path / "interim"

    result = runner.invoke(
        app,
        [
            "prepare",
            "contextual-features",
            "--no-historical",
            "--offline-fixture",
            "--as-of",
            "2026-06-20T18:00:00Z",
            "--data-cutoff",
            "2026-06-20T18:00:00Z",
            "--output-root",
            str(output_root),
            "--interim-root",
            str(interim_root),
        ],
    )

    assert result.exit_code == 0
    assert (output_root / "team_fixture_contextual_features.parquet").is_file()
    assert (output_root / "match_contextual_features.parquet").is_file()
    assert (interim_root / "contextual_features_coverage.md").is_file()
    assert not (tmp_path / "predictions-data" / "latest.csv").exists()


def _build(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    generated_at: datetime = datetime(2026, 1, 30, 12, tzinfo=UTC),
):
    input_path = tmp_path / "matches.parquet"
    _write_rows(input_path, rows)
    return build_contextual_features(
        historical_matches_path=None,
        live_matches_path=input_path,
        venue_catalog_path=_venue_catalog_path(tmp_path),
        feature_generated_at_utc=generated_at,
        include_historical=False,
        include_live=True,
    )


def _team_row(
    rows: tuple[dict[str, object], ...], fixture_id: str, team_id: str
) -> dict[str, object]:
    for row in rows:
        if row["fixture_id"] == fixture_id and row["team_id"] == team_id:
            return row
    raise AssertionError(f"missing team row {fixture_id}/{team_id}")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


def _match_row(
    source_match_id: str,
    *,
    kickoff: datetime = datetime(2026, 1, 1, 18, tzinfo=UTC),
    home: str = "alpha",
    away: str = "beta",
    venue: str = "BC Place",
    status: str = "played",
    extra_time: bool = False,
    penalties: bool = False,
    data_cutoff: datetime = datetime(2026, 1, 1, 12, tzinfo=UTC),
) -> dict[str, object]:
    return {
        "match_id": f"world_cup_2026_football_data:{source_match_id}",
        "match_status": status,
        "match_date": date(kickoff.year, kickoff.month, kickoff.day),
        "kickoff_utc": kickoff,
        "home_team_id": home,
        "away_team_id": away,
        "home_team_name_original": home.title(),
        "away_team_name_original": away.title(),
        "competition": "FIFA World Cup",
        "stage": "GROUP_STAGE",
        "source": "world_cup_2026_football_data",
        "source_match_id": source_match_id,
        "retrieved_at_utc": data_cutoff,
        "source_updated_at_utc": data_cutoff,
        "data_cutoff_utc": data_cutoff,
        "neutral_site": None,
        "venue_name_original": venue,
        "city": None,
        "host_country": None,
        "extra_time_played": extra_time,
        "penalty_shootout": penalties,
    }


def _venue_catalog_path(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "venues.csv"
    path.write_text(
        "\n".join(
            [
                "venue_id,canonical_name,provider_aliases,city,country,latitude,longitude,"
                "elevation_m,timezone,source_name,source_url,source_retrieved_at,source_version",
                "bc_place_vancouver,BC Place Vancouver,BC Place|BC Place Vancouver,Vancouver,"
                "Canada,49.276666666,-123.111944444,18,America/Vancouver,test,test,2026-07-01,v1",
                "mexico_city_stadium,Mexico City Stadium,Estadio Azteca|Mexico City Stadium,"
                "Mexico City,Mexico,19.303055555,-99.150555555,2245,America/Mexico_City,"
                "test,test,2026-07-01,v1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path
