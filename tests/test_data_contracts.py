import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from worldcup2026.data import (
    CanonicalMatch,
    KickoffTimeStatus,
    MatchStatus,
    RawSnapshotManifest,
    Result90,
    TeamAlias,
    resolve_team_alias,
    sha256_bytes,
    validate_match_records,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_matches.json"


def load_fixture_records() -> list[dict[str, Any]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def valid_played_record() -> dict[str, Any]:
    return load_fixture_records()[1].copy()


def fixture_aliases() -> list[dict[str, Any]]:
    return [
        {
            "canonical_team_id": "arg",
            "canonical_name": "Argentina",
            "source": "synthetic_fixture",
            "source_name": "Argentina",
            "valid_from": "1902-07-20",
            "valid_to": None,
        },
        {
            "canonical_team_id": "deu",
            "canonical_name": "Germany",
            "source": "synthetic_fixture",
            "source_name": "Germany",
            "valid_from": "1990-10-03",
            "valid_to": None,
        },
        {
            "canonical_team_id": "deu",
            "canonical_name": "Germany",
            "source": "synthetic_fixture",
            "source_name": "West Germany",
            "valid_from": "1949-05-23",
            "valid_to": "1990-10-02",
        },
        {
            "canonical_team_id": "civ",
            "canonical_name": "Cote d'Ivoire",
            "source": "synthetic_fixture",
            "source_name": "Ivory Coast",
            "valid_from": "1960-08-07",
            "valid_to": None,
        },
        {
            "canonical_team_id": "usa",
            "canonical_name": "United States",
            "source": "synthetic_fixture",
            "source_name": "USA",
            "valid_from": "1916-08-20",
            "valid_to": None,
        },
    ]


def test_fixture_records_validate_as_collection() -> None:
    matches = validate_match_records(load_fixture_records(), require_temporal_order=True)

    assert len(matches) == 3
    assert matches[0].kickoff_time_status is KickoffTimeStatus.DATE_ONLY
    assert matches[1].result_90 is Result90.DRAW
    assert matches[1].penalty_shootout is True
    assert matches[2].match_status is MatchStatus.CANCELLED


def test_fixture_records_validate_against_team_aliases() -> None:
    matches = validate_match_records(load_fixture_records(), team_aliases=fixture_aliases())

    assert len(matches) == 3


def test_required_fact_flags_must_be_explicit() -> None:
    record = valid_played_record()
    del record["extra_time_played"]

    with pytest.raises(ValidationError, match="extra_time_played"):
        CanonicalMatch.model_validate(record)


def test_required_fact_flags_must_be_strict_booleans() -> None:
    record = valid_played_record()
    record["extra_time_played"] = "true"

    with pytest.raises(ValidationError, match="extra_time_played"):
        CanonicalMatch.model_validate(record)


def test_played_match_requires_90_minute_scores() -> None:
    record = valid_played_record()
    record["home_goals_90"] = None

    with pytest.raises(ValidationError, match="played matches require both 90-minute goal fields"):
        CanonicalMatch.model_validate(record)


def test_result_90_must_match_scoreline() -> None:
    record = valid_played_record()
    record["result_90"] = "away_win"

    with pytest.raises(ValidationError, match="result_90 must match"):
        CanonicalMatch.model_validate(record)


def test_cancelled_match_must_not_include_scores() -> None:
    record = load_fixture_records()[2].copy()
    record["home_goals_90"] = 1

    with pytest.raises(ValidationError, match="matches that were not played"):
        CanonicalMatch.model_validate(record)


def test_extra_time_scores_cannot_be_lower_than_90_minute_scores() -> None:
    record = valid_played_record()
    record["home_goals_after_extra_time"] = 0

    with pytest.raises(ValidationError, match="extra-time goals cannot be lower"):
        CanonicalMatch.model_validate(record)


def test_extra_time_requires_tied_90_minute_scores() -> None:
    record = valid_played_record()
    record["home_goals_90"] = 2
    record["away_goals_90"] = 1
    record["result_90"] = "home_win"
    record["home_goals_after_extra_time"] = 2
    record["away_goals_after_extra_time"] = 2
    record["penalty_shootout"] = False
    record["home_penalty_goals"] = None
    record["away_penalty_goals"] = None

    with pytest.raises(ValidationError, match="requires tied 90-minute goals"):
        CanonicalMatch.model_validate(record)


def test_penalty_shootout_requires_decisive_penalty_score() -> None:
    record = valid_played_record()
    record["home_penalty_goals"] = 4
    record["away_penalty_goals"] = 4

    with pytest.raises(ValidationError, match="penalty shootout goals must identify a winner"):
        CanonicalMatch.model_validate(record)


def test_penalty_shootout_requires_tied_prior_score() -> None:
    record = valid_played_record()
    record["away_goals_after_extra_time"] = 2

    with pytest.raises(ValidationError, match="require tied extra-time goals"):
        CanonicalMatch.model_validate(record)


def test_exact_utc_records_require_utc_timestamp() -> None:
    record = valid_played_record()
    record["kickoff_utc"] = None

    with pytest.raises(ValidationError, match="kickoff_utc is required"):
        CanonicalMatch.model_validate(record)


def test_exact_utc_allows_local_match_date_to_differ_from_utc_date() -> None:
    record = valid_played_record()
    record["match_id"] = "synthetic-utc-date-crossing"
    record["match_date"] = "2026-06-11"
    record["kickoff_utc"] = "2026-06-12T01:00:00Z"
    record["kickoff_local_time"] = "20:00"
    record["kickoff_timezone"] = "America/Mexico_City"

    match = CanonicalMatch.model_validate(record)

    assert match.match_date.isoformat() == "2026-06-11"
    assert match.kickoff_utc is not None
    assert match.kickoff_utc.date().isoformat() == "2026-06-12"


def test_exact_utc_records_reject_invalid_timezone_names() -> None:
    record = valid_played_record()
    record["kickoff_timezone"] = "not/a-real-zone"

    with pytest.raises(ValidationError, match="IANA timezone or UTC offset"):
        CanonicalMatch.model_validate(record)


def test_exact_utc_records_accept_utc_offsets_as_timezone() -> None:
    record = valid_played_record()
    record["kickoff_timezone"] = "-06:00"

    match = CanonicalMatch.model_validate(record)

    assert match.kickoff_timezone == "-06:00"


def test_date_only_records_reject_kickoff_time() -> None:
    record = load_fixture_records()[0].copy()
    record["kickoff_local_time"] = "21:00"

    with pytest.raises(ValidationError, match="date_only records must not include"):
        CanonicalMatch.model_validate(record)


def test_date_only_records_reject_timezone() -> None:
    record = load_fixture_records()[0].copy()
    record["kickoff_timezone"] = "UTC"

    with pytest.raises(ValidationError, match="date_only records must not include"):
        CanonicalMatch.model_validate(record)


def test_local_time_without_timezone_rejects_timezone() -> None:
    record = load_fixture_records()[2].copy()
    record["kickoff_timezone"] = "America/New_York"

    with pytest.raises(ValidationError, match="must not include kickoff_timezone"):
        CanonicalMatch.model_validate(record)


def test_home_and_away_team_ids_must_differ() -> None:
    record = valid_played_record()
    record["away_team_id"] = record["home_team_id"]

    with pytest.raises(ValidationError, match="must be different"):
        CanonicalMatch.model_validate(record)


def test_neutral_site_rejects_single_team_home_advantage() -> None:
    record = valid_played_record()
    record["neutral_site"] = True

    with pytest.raises(ValidationError, match="neutral_site=true requires"):
        CanonicalMatch.model_validate(record)


def test_non_neutral_site_rejects_neutral_home_advantage() -> None:
    record = valid_played_record()
    record["neutral_site"] = False
    record["home_advantage_status"] = "neutral"

    with pytest.raises(ValidationError, match="neutral_site=false is incompatible"):
        CanonicalMatch.model_validate(record)


def test_team_ids_must_use_canonical_format() -> None:
    record = valid_played_record()
    record["home_team_id"] = "Germany"

    with pytest.raises(ValidationError, match="home_team_id"):
        CanonicalMatch.model_validate(record)


def test_team_aliases_resolve_by_source_name_and_date() -> None:
    aliases = [TeamAlias.model_validate(alias) for alias in fixture_aliases()]

    assert (
        resolve_team_alias(
            source="synthetic_fixture",
            source_name="West Germany",
            match_date=CanonicalMatch.model_validate(valid_played_record()).match_date.replace(
                year=1986
            ),
            aliases=aliases,
        )
        == "deu"
    )


def test_collection_rejects_team_id_that_conflicts_with_alias() -> None:
    records = load_fixture_records()[:1]
    records[0] = records[0].copy()
    records[0]["home_team_id"] = "usa"

    with pytest.raises(ValueError, match="home_team_id usa does not match alias"):
        validate_match_records(records, team_aliases=fixture_aliases())


def test_collection_rejects_duplicate_match_ids() -> None:
    records = load_fixture_records()[:2]
    records[1] = records[1].copy()
    records[1]["match_id"] = records[0]["match_id"]

    with pytest.raises(ValueError, match="duplicate match_id"):
        validate_match_records(records)


def test_collection_rejects_duplicate_source_keys() -> None:
    records = load_fixture_records()[:2]
    records[1] = records[1].copy()
    records[1]["source_match_id"] = records[0]["source_match_id"]

    with pytest.raises(ValueError, match="duplicate source/source_match_id"):
        validate_match_records(records)


def test_collection_can_require_temporal_order() -> None:
    records = [load_fixture_records()[1], load_fixture_records()[0]]

    with pytest.raises(ValueError, match="matches must be ordered"):
        validate_match_records(records, require_temporal_order=True)


def test_collection_temporal_order_allows_same_date_mixed_precision() -> None:
    records = [valid_played_record(), load_fixture_records()[0].copy()]
    records[1]["match_id"] = "synthetic-date-only-same-day"
    records[1]["match_date"] = records[0]["match_date"]
    records[1]["source_match_id"] = "fixture-date-only-same-day"

    matches = validate_match_records(records, require_temporal_order=True)

    assert [match.match_id for match in matches] == [
        "synthetic-2006-deu-arg-world-cup",
        "synthetic-date-only-same-day",
    ]


def test_raw_snapshot_manifest_uses_sha256_and_utc_timestamp() -> None:
    manifest = RawSnapshotManifest.model_validate(
        {
            "source": "international_results_csv",
            "logical_uri": "local-fixture://results.csv",
            "source_revision": "c44451d1a07f736502f364a62b6fbc947a544809",
            "retrieved_at_utc": "2026-01-01T00:00:00Z",
            "content_sha256": sha256_bytes(b"fixture"),
            "cache_key": "international_results_csv:fixture",
            "raw_path": "data/raw/international_results_csv/2026-01-01/results.csv",
            "input_uri": "file:///tmp/results.csv",
        }
    )

    assert manifest.content_sha256 == (
        "f16d05ec6b29248d2c61adb1e9263f78"
        "e4f7bace1b955014a2d17872cfe4064d"
    )
    assert manifest.source_revision == "c44451d1a07f736502f364a62b6fbc947a544809"
