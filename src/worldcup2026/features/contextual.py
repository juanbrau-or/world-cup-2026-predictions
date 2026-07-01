"""As-of contextual features for international fixtures.

The functions in this module deliberately compute diagnostics only.  They do not feed the frozen
operational Poisson model and they never infer future bracket information.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pyarrow as pa
import pyarrow.parquet as pq

FEATURE_SET_VERSION = "contextual_asof_v1"
CONTEXTUAL_SCHEMA_VERSION = "contextual_features_v1"
SOURCE_DATASET_REVISION_PREFIX = "contextual_source_v1"
VENUE_CATALOG_SCHEMA_VERSION = "world_cup_2026_venues_v1"
HOST_COUNTRY_TEAMS = {
    "Canada": "canada",
    "Mexico": "mexico",
    "United States": "united_states",
}
EARTH_RADIUS_KM = 6371.0088
WINDOW_DAYS = (7, 14, 30)
TRAINABLE_COVERAGE_THRESHOLD = 0.80
MISSING_INDICATOR_THRESHOLD = 0.20
REASONABLE_ELEVATION_RANGE_M = (-500.0, 6000.0)

LEVEL_A_FEATURES = (
    "rest_hours",
    "rest_days",
    "matches_last_7d",
    "matches_last_14d",
    "matches_last_30d",
    "minutes_equivalent_last_7d",
    "minutes_equivalent_last_14d",
    "minutes_equivalent_last_30d",
    "previous_match_extra_time",
    "previous_match_penalty_shootout",
    "hours_since_previous_match",
    "consecutive_matches_without_7d_rest",
    "tournament_match_number",
    "is_first_tournament_match",
)
LEVEL_B_FEATURES = (
    "venue_id",
    "venue_city",
    "venue_country",
    "venue_latitude",
    "venue_longitude",
    "venue_elevation_m",
    "venue_timezone",
    "previous_venue_id",
    "travel_distance_km",
    "cumulative_travel_km_7d",
    "cumulative_travel_km_14d",
    "cumulative_travel_km_30d",
    "timezone_delta_hours",
    "elevation_change_m",
    "absolute_elevation_change_m",
    "cross_border_travel",
    "host_country_match",
)
NUMERIC_DIFF_FEATURES = (
    "rest_hours",
    "travel_distance_km",
    "elevation_change_m",
    "matches_last_14d",
)
TEAM_ROW_ORDER = (
    "fixture_id",
    "match_id",
    "source",
    "source_match_id",
    "match_status",
    "match_date",
    "kickoff_utc",
    "data_cutoff_utc",
    "feature_generated_at_utc",
    "feature_set_version",
    "source_dataset_revision",
    "source_row_checksum",
    "venue_catalog_checksum",
    "team_id",
    "opponent_team_id",
    "side",
    "competition",
    "stage",
    "is_neutral_venue",
    *LEVEL_A_FEATURES,
    "previous_match_id",
    "previous_match_date",
    "previous_match_kickoff_utc",
    "extra_time_data_quality",
    "minutes_equivalent_missing_reason",
    *LEVEL_B_FEATURES,
)
PREFIXED_FEATURES = (
    "team_id",
    "opponent_team_id",
    "rest_hours",
    "rest_days",
    "matches_last_7d",
    "matches_last_14d",
    "matches_last_30d",
    "minutes_equivalent_last_7d",
    "minutes_equivalent_last_14d",
    "minutes_equivalent_last_30d",
    "previous_match_extra_time",
    "previous_match_penalty_shootout",
    "hours_since_previous_match",
    "consecutive_matches_without_7d_rest",
    "tournament_match_number",
    "is_first_tournament_match",
    "venue_id",
    "venue_city",
    "venue_country",
    "venue_latitude",
    "venue_longitude",
    "venue_elevation_m",
    "venue_timezone",
    "previous_venue_id",
    "travel_distance_km",
    "cumulative_travel_km_7d",
    "cumulative_travel_km_14d",
    "cumulative_travel_km_30d",
    "timezone_delta_hours",
    "elevation_change_m",
    "absolute_elevation_change_m",
    "cross_border_travel",
    "host_country_match",
)


class ContextualFeatureError(RuntimeError):
    """Raised when contextual features cannot be produced safely."""


@dataclass(frozen=True)
class Venue:
    """Auditable World Cup 2026 venue metadata."""

    venue_id: str
    canonical_name: str
    provider_aliases: tuple[str, ...]
    city: str
    country: str
    latitude: float
    longitude: float
    elevation_m: float
    timezone: str
    source_name: str
    source_url: str
    source_retrieved_at: date
    source_version: str


@dataclass(frozen=True)
class TeamFixture:
    """One team's view of a canonical fixture before feature calculation."""

    fixture_id: str
    match_id: str
    source: str
    source_match_id: str
    match_status: str
    match_date: date
    kickoff_utc: datetime | None
    data_cutoff_utc: datetime
    source_updated_at_utc: datetime | None
    team_id: str
    opponent_team_id: str
    side: str
    competition: str
    stage: str | None
    neutral_site: bool | None
    venue_name_original: str | None
    city: str | None
    host_country: str | None
    extra_time_played: bool | None
    penalty_shootout: bool | None
    source_row_checksum: str


@dataclass(frozen=True)
class LeakageAudit:
    """Leakage and schema validation result."""

    passed: bool
    violations: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ContextualFeatureResult:
    """In-memory contextual feature build result."""

    team_rows: tuple[dict[str, Any], ...]
    match_rows: tuple[dict[str, Any], ...]
    manifest: Mapping[str, Any]
    coverage_report: Mapping[str, Any]
    missing_data_report: Mapping[str, Any]
    leakage_audit: Mapping[str, Any]
    descriptive_report: Mapping[str, Any]


def load_venue_catalog(path: Path) -> tuple[Venue, ...]:
    """Load and validate a small versioned venue catalog."""

    if not path.is_file():
        raise ContextualFeatureError(f"venue catalog is missing: {path}")
    try:
        with path.open(encoding="utf-8", newline="") as file:
            rows = [dict(row) for row in csv.DictReader(file)]
    except OSError as exc:
        raise ContextualFeatureError(f"failed to read venue catalog {path}: {exc}") from exc
    venues = tuple(_venue_from_row(row, row_number=index + 2) for index, row in enumerate(rows))
    ids = [venue.venue_id for venue in venues]
    duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ContextualFeatureError(
            "venue catalog has duplicate venue_id values: " + ", ".join(duplicates)
        )
    alias_index: dict[str, str] = {}
    for venue in venues:
        for alias in _venue_aliases(venue):
            normalized = _normalize_alias(alias)
            existing = alias_index.get(normalized)
            if existing is not None and existing != venue.venue_id:
                msg = f"venue alias {alias!r} maps to both {existing!r} and {venue.venue_id!r}"
                raise ContextualFeatureError(msg)
            alias_index[normalized] = venue.venue_id
    return venues


def venue_catalog_checksum(path: Path) -> str:
    """Return the SHA-256 checksum for the catalog bytes."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ContextualFeatureError(f"failed to checksum venue catalog {path}: {exc}") from exc


def haversine_distance_km(
    lat_a: float | None,
    lon_a: float | None,
    lat_b: float | None,
    lon_b: float | None,
) -> float | None:
    """Return Haversine distance in kilometers, or ``None`` when any coordinate is missing."""

    if lat_a is None or lon_a is None or lat_b is None or lon_b is None:
        return None
    _validate_latitude(lat_a)
    _validate_latitude(lat_b)
    _validate_longitude(lon_a)
    _validate_longitude(lon_b)
    if lat_a == lat_b and lon_a == lon_b:
        return 0.0
    phi_a = math.radians(lat_a)
    phi_b = math.radians(lat_b)
    delta_phi = math.radians(lat_b - lat_a)
    delta_lambda = math.radians(lon_b - lon_a)
    haversine = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(haversine))


def timezone_delta_hours(
    current_timezone: str | None,
    previous_timezone: str | None,
    kickoff_utc: datetime | None,
) -> float | None:
    """Return current minus previous UTC offset hours at the actual kickoff date."""

    if current_timezone is None or previous_timezone is None or kickoff_utc is None:
        return None
    current_offset = _timezone_offset_hours(current_timezone, kickoff_utc)
    previous_offset = _timezone_offset_hours(previous_timezone, kickoff_utc)
    return current_offset - previous_offset


def elevation_change_m(current: float | None, previous: float | None) -> float | None:
    """Return current minus previous venue elevation in meters."""

    if current is None or previous is None:
        return None
    _validate_elevation(current)
    _validate_elevation(previous)
    return current - previous


def build_contextual_features(
    *,
    historical_matches_path: Path | None,
    live_matches_path: Path | None,
    venue_catalog_path: Path,
    feature_generated_at_utc: datetime,
    data_cutoff_utc: datetime | None = None,
    include_historical: bool = True,
    include_live: bool = True,
) -> ContextualFeatureResult:
    """Build team-fixture and match-level contextual feature rows."""

    generated_at = _require_utc(feature_generated_at_utc, field_name="feature_generated_at_utc")
    cutoff = (
        _require_utc(data_cutoff_utc, field_name="data_cutoff_utc")
        if data_cutoff_utc is not None
        else None
    )
    venues = load_venue_catalog(venue_catalog_path)
    venue_by_alias = _venue_alias_index(venues)
    venue_checksum = venue_catalog_checksum(venue_catalog_path)
    input_rows = _load_input_match_rows(
        historical_matches_path=historical_matches_path,
        live_matches_path=live_matches_path,
        include_historical=include_historical,
        include_live=include_live,
        data_cutoff_utc=cutoff,
    )
    source_revision = _source_dataset_revision(input_rows, venue_checksum=venue_checksum)
    fixtures = _expand_team_fixtures(input_rows, explicit_cutoff=cutoff)
    team_rows = _calculate_team_rows(
        fixtures,
        venue_by_alias=venue_by_alias,
        feature_generated_at_utc=generated_at,
        feature_set_version=FEATURE_SET_VERSION,
        source_dataset_revision=source_revision,
        venue_catalog_checksum=venue_checksum,
    )
    match_rows = _match_level_rows(team_rows)
    audit = validate_contextual_feature_rows(team_rows, match_rows)
    coverage = coverage_report(team_rows)
    missing = missing_data_report(team_rows)
    descriptive = descriptive_quality_report(team_rows)
    manifest = _manifest(
        input_rows=input_rows,
        team_rows=team_rows,
        match_rows=match_rows,
        generated_at=generated_at,
        data_cutoff_utc=cutoff,
        source_dataset_revision=source_revision,
        venue_catalog_path=venue_catalog_path,
        venue_catalog_checksum=venue_checksum,
        audit=audit,
    )
    return ContextualFeatureResult(
        team_rows=team_rows,
        match_rows=match_rows,
        manifest=manifest,
        coverage_report=coverage,
        missing_data_report=missing,
        leakage_audit={
            "schema_version": CONTEXTUAL_SCHEMA_VERSION,
            "passed": audit.passed,
            "violations": list(audit.violations),
        },
        descriptive_report=descriptive,
    )


def validate_contextual_feature_rows(
    team_rows: Sequence[Mapping[str, Any]],
    match_rows: Sequence[Mapping[str, Any]],
) -> LeakageAudit:
    """Validate leakage, row shape and geographic ranges for contextual feature outputs."""

    violations: list[Mapping[str, Any]] = []
    seen_team_fixtures: set[tuple[str, str]] = set()
    team_counts: Counter[str] = Counter()
    for row in team_rows:
        fixture_id = _required_str(row, "fixture_id")
        team_id = _required_str(row, "team_id")
        team_key = (fixture_id, team_id)
        if team_key in seen_team_fixtures:
            violations.append(_violation(row, "duplicate_team_fixture"))
        seen_team_fixtures.add(team_key)
        team_counts[fixture_id] += 1
        data_cutoff = _required_datetime(row, "data_cutoff_utc")
        generated = _required_datetime(row, "feature_generated_at_utc")
        kickoff = _optional_datetime(row.get("kickoff_utc"))
        if data_cutoff > generated:
            violations.append(_violation(row, "data_cutoff_after_feature_generation"))
        if _is_prospective_row(row) and kickoff is not None and generated >= kickoff:
            violations.append(_violation(row, "prospective_feature_generated_at_or_after_kickoff"))
        previous_kickoff = _optional_datetime(row.get("previous_match_kickoff_utc"))
        if kickoff is not None and previous_kickoff is not None and previous_kickoff >= kickoff:
            violations.append(_violation(row, "previous_match_not_before_current_kickoff"))
        previous_date = _optional_date(row.get("previous_match_date"))
        match_date = _required_date(row, "match_date")
        if kickoff is None and previous_date is not None and previous_date >= match_date:
            violations.append(_violation(row, "previous_match_not_before_current_date"))
        _append_geography_violations(row, violations)
        _append_negative_value_violations(row, violations)
    for fixture_id, count in team_counts.items():
        if count != 2:
            violations.append({"fixture_id": fixture_id, "reason": "fixture_without_two_team_rows"})
    for row in match_rows:
        kickoff = _optional_datetime(row.get("kickoff_utc"))
        generated = _required_datetime(row, "feature_generated_at_utc")
        if _is_match_prospective_row(row) and kickoff is not None and generated >= kickoff:
            violations.append(
                {"fixture_id": row.get("fixture_id"), "reason": "match_row_generated_after_kickoff"}
            )
    return LeakageAudit(passed=not violations, violations=tuple(violations))


def coverage_report(team_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Build deterministic coverage and feature classification diagnostics."""

    total = len(team_rows)
    feature_names = (*LEVEL_A_FEATURES, *LEVEL_B_FEATURES)
    by_feature = {
        feature: _feature_coverage(team_rows, feature)
        for feature in feature_names
    }
    historical_rows = [row for row in team_rows if _is_historical_source(row)]
    world_cup_rows = [row for row in team_rows if _is_world_cup_2026_source(row)]
    report = {
        "schema_version": CONTEXTUAL_SCHEMA_VERSION,
        "rows_total": total,
        "team_fixture_rows": total,
        "fixture_rows": len({str(row.get("fixture_id")) for row in team_rows}),
        "coverage_by_feature": by_feature,
        "coverage_by_year": _coverage_by_dimension(team_rows, "match_year", feature_names),
        "coverage_by_tournament": _coverage_by_dimension(team_rows, "competition", feature_names),
        "coverage_by_team": _coverage_by_dimension(team_rows, "team_id", feature_names),
        "coverage_world_cup_2026": {
            feature: _feature_coverage(world_cup_rows, feature) for feature in feature_names
        },
        "coverage_historical": {
            feature: _feature_coverage(historical_rows, feature) for feature in feature_names
        },
        "missing_percent_by_feature": {
            feature: by_feature[feature]["missing_percent"] for feature in feature_names
        },
        "impossible_values": _impossible_value_counts(team_rows),
        "duplicates": _duplicate_team_fixture_count(team_rows),
        "records_discarded": 0,
        "thresholds": {
            "trainable_min_non_null_coverage": TRAINABLE_COVERAGE_THRESHOLD,
            "missing_indicator_min_missingness": MISSING_INDICATOR_THRESHOLD,
            "elevation_reasonable_min_m": REASONABLE_ELEVATION_RANGE_M[0],
            "elevation_reasonable_max_m": REASONABLE_ELEVATION_RANGE_M[1],
        },
        "feature_classification": _feature_classification(team_rows),
    }
    return report


def missing_data_report(team_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Summarize feature missingness and documented source limitations."""

    feature_names = (*LEVEL_A_FEATURES, *LEVEL_B_FEATURES)
    missing_by_feature = {
        feature: {
            "missing_rows": sum(1 for row in team_rows if row.get(feature) is None),
            "rows": len(team_rows),
            "missing_percent": _safe_ratio(
                sum(1 for row in team_rows if row.get(feature) is None), len(team_rows)
            ),
        }
        for feature in feature_names
    }
    reasons = Counter(
        str(row.get("minutes_equivalent_missing_reason"))
        for row in team_rows
        if row.get("minutes_equivalent_missing_reason") is not None
    )
    unresolved_venues = Counter(
        str(row.get("source"))
        for row in team_rows
        if row.get("venue_id") is None and _is_world_cup_2026_source(row)
    )
    return {
        "schema_version": CONTEXTUAL_SCHEMA_VERSION,
        "rows": len(team_rows),
        "missing_by_feature": missing_by_feature,
        "minutes_equivalent_missing_reasons": dict(sorted(reasons.items())),
        "unresolved_world_cup_venues_by_source": dict(sorted(unresolved_venues.items())),
        "documented_limitations": [
            "international_results_csv provides dates but no kickoff UTC/timezone, so hour-based "
            "rest features are missing for historical rows.",
            "international_results_csv does not safely identify all extra-time matches; "
            "minutes-equivalent rolling loads are missing for rows whose prior matches are from "
            "that source.",
            "Football-Data venue coverage is source-dependent; World Cup venue diagnostics are "
            "only populated when provider venue aliases match the audited static catalog.",
        ],
    }


def descriptive_quality_report(team_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Return descriptive, non-modeling diagnostics for contextual features."""

    return {
        "schema_version": CONTEXTUAL_SCHEMA_VERSION,
        "rows": len(team_rows),
        "distributions": {
            "rest_hours": _distribution(team_rows, "rest_hours"),
            "travel_distance_km": _distribution(team_rows, "travel_distance_km"),
            "timezone_delta_hours": _distribution(team_rows, "timezone_delta_hours"),
            "venue_elevation_m": _distribution(team_rows, "venue_elevation_m"),
            "elevation_change_m": _distribution(team_rows, "elevation_change_m"),
        },
        "missingness": {
            feature: _feature_coverage(team_rows, feature)["missing_percent"]
            for feature in (*LEVEL_A_FEATURES, *LEVEL_B_FEATURES)
        },
        "outliers": {
            "rest_hours_negative": [
                _row_ref(row) for row in team_rows if _is_negative(row.get("rest_hours"))
            ],
            "travel_distance_over_6000_km": [
                _row_ref(row)
                for row in team_rows
                if _optional_float(row.get("travel_distance_km")) is not None
                and float(row["travel_distance_km"]) > 6000
            ],
            "absolute_timezone_delta_over_4h": [
                _row_ref(row)
                for row in team_rows
                if _optional_float(row.get("timezone_delta_hours")) is not None
                and abs(float(row["timezone_delta_hours"])) > 4
            ],
            "absolute_elevation_change_over_2500m": [
                _row_ref(row)
                for row in team_rows
                if _optional_float(row.get("absolute_elevation_change_m")) is not None
                and float(row["absolute_elevation_change_m"]) > 2500
            ],
        },
    }


def write_contextual_feature_outputs(
    result: ContextualFeatureResult,
    *,
    output_root: Path,
    interim_root: Path,
) -> Mapping[str, Path]:
    """Write Parquet datasets plus JSON/Markdown reports."""

    output_root.mkdir(parents=True, exist_ok=True)
    interim_root.mkdir(parents=True, exist_ok=True)
    team_path = output_root / "team_fixture_contextual_features.parquet"
    match_path = output_root / "match_contextual_features.parquet"
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(list(result.team_rows), schema=_team_feature_schema()),
        team_path,
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(list(result.match_rows), schema=_match_feature_schema()),
        match_path,
    )
    manifest_path = interim_root / "contextual_features_manifest.json"
    coverage_json_path = interim_root / "contextual_features_coverage.json"
    coverage_md_path = interim_root / "contextual_features_coverage.md"
    missing_path = interim_root / "contextual_features_missing_data.json"
    leakage_path = interim_root / "contextual_features_leakage_audit.json"
    descriptive_path = interim_root / "contextual_features_descriptive_quality.json"
    _write_json(manifest_path, result.manifest)
    _write_json(coverage_json_path, result.coverage_report)
    coverage_md_path.write_text(_coverage_markdown(result.coverage_report), encoding="utf-8")
    _write_json(missing_path, result.missing_data_report)
    _write_json(leakage_path, result.leakage_audit)
    _write_json(descriptive_path, result.descriptive_report)
    return {
        "team_fixture_parquet": team_path,
        "match_parquet": match_path,
        "manifest": manifest_path,
        "coverage_json": coverage_json_path,
        "coverage_markdown": coverage_md_path,
        "missing_data_report": missing_path,
        "leakage_audit": leakage_path,
        "descriptive_report": descriptive_path,
    }


def _venue_from_row(row: Mapping[str, str], *, row_number: int) -> Venue:
    required = (
        "venue_id",
        "canonical_name",
        "provider_aliases",
        "city",
        "country",
        "latitude",
        "longitude",
        "elevation_m",
        "timezone",
        "source_name",
        "source_url",
        "source_retrieved_at",
        "source_version",
    )
    missing = [field for field in required if not row.get(field)]
    if missing:
        msg = f"venue catalog row {row_number} is missing fields: {', '.join(missing)}"
        raise ContextualFeatureError(msg)
    latitude = _parse_float(row["latitude"], field_name="latitude", row_number=row_number)
    longitude = _parse_float(row["longitude"], field_name="longitude", row_number=row_number)
    elevation = _parse_float(row["elevation_m"], field_name="elevation_m", row_number=row_number)
    _validate_latitude(latitude)
    _validate_longitude(longitude)
    _validate_elevation(elevation)
    timezone = row["timezone"].strip()
    _zoneinfo(timezone)
    try:
        retrieved_at = date.fromisoformat(row["source_retrieved_at"].strip())
    except ValueError as exc:
        raise ContextualFeatureError(
            f"venue catalog row {row_number} has invalid source_retrieved_at"
        ) from exc
    aliases = tuple(
        alias.strip() for alias in row["provider_aliases"].split("|") if alias.strip()
    )
    return Venue(
        venue_id=row["venue_id"].strip(),
        canonical_name=row["canonical_name"].strip(),
        provider_aliases=aliases,
        city=row["city"].strip(),
        country=row["country"].strip(),
        latitude=latitude,
        longitude=longitude,
        elevation_m=elevation,
        timezone=timezone,
        source_name=row["source_name"].strip(),
        source_url=row["source_url"].strip(),
        source_retrieved_at=retrieved_at,
        source_version=row["source_version"].strip(),
    )


def _parse_float(value: str, *, field_name: str, row_number: int) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ContextualFeatureError(
            f"venue catalog row {row_number} has invalid {field_name}"
        ) from exc


def _validate_latitude(value: float) -> None:
    if not math.isfinite(value) or value < -90 or value > 90:
        raise ContextualFeatureError(f"latitude out of range: {value}")


def _validate_longitude(value: float) -> None:
    if not math.isfinite(value) or value < -180 or value > 180:
        raise ContextualFeatureError(f"longitude out of range: {value}")


def _validate_elevation(value: float) -> None:
    if (
        not math.isfinite(value)
        or value < REASONABLE_ELEVATION_RANGE_M[0]
        or value > REASONABLE_ELEVATION_RANGE_M[1]
    ):
        raise ContextualFeatureError(f"elevation_m out of reasonable range: {value}")


def _zoneinfo(value: str) -> ZoneInfo:
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        raise ContextualFeatureError(f"invalid IANA timezone: {value}") from exc


def _timezone_offset_hours(timezone: str, kickoff_utc: datetime) -> float:
    utc = _require_utc(kickoff_utc, field_name="kickoff_utc")
    offset = utc.astimezone(_zoneinfo(timezone)).utcoffset()
    if offset is None:
        raise ContextualFeatureError(f"timezone has no UTC offset at kickoff: {timezone}")
    return offset.total_seconds() / 3600


def _venue_aliases(venue: Venue) -> tuple[str, ...]:
    return (venue.venue_id, venue.canonical_name, *venue.provider_aliases)


def _normalize_alias(value: str) -> str:
    return " ".join(value.casefold().replace("-", " ").replace("_", " ").split())


def _venue_alias_index(venues: Iterable[Venue]) -> Mapping[str, Venue]:
    index: dict[str, Venue] = {}
    for venue in venues:
        for alias in _venue_aliases(venue):
            index[_normalize_alias(alias)] = venue
    return index


def _match_venue(row: TeamFixture, venue_by_alias: Mapping[str, Venue]) -> Venue | None:
    candidates = [row.venue_name_original, row.city]
    for candidate in candidates:
        if not candidate:
            continue
        venue = venue_by_alias.get(_normalize_alias(candidate))
        if venue is not None:
            return venue
    return None


def _load_input_match_rows(
    *,
    historical_matches_path: Path | None,
    live_matches_path: Path | None,
    include_historical: bool,
    include_live: bool,
    data_cutoff_utc: datetime | None,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    if (
        include_historical
        and historical_matches_path is not None
        and historical_matches_path.is_file()
    ):
        rows.extend(_read_parquet_rows(historical_matches_path))
    if include_live and live_matches_path is not None and live_matches_path.is_file():
        rows.extend(_read_parquet_rows(live_matches_path))
    filtered = []
    for row in rows:
        row_cutoff = _row_cutoff(row, explicit_cutoff=None)
        if data_cutoff_utc is not None and row_cutoff > data_cutoff_utc:
            continue
        filtered.append(row)
    filtered.sort(key=lambda row: (_match_sort_date(row), str(row.get("match_id") or "")))
    return tuple(filtered)


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
    except (OSError, pa.ArrowInvalid) as exc:
        raise ContextualFeatureError(f"failed to read Parquet input {path}: {exc}") from exc
    return [dict(row) for row in table.to_pylist()]


def _expand_team_fixtures(
    rows: Sequence[Mapping[str, Any]], *, explicit_cutoff: datetime | None
) -> tuple[TeamFixture, ...]:
    fixtures: list[TeamFixture] = []
    for row in rows:
        home_id = _optional_str(row.get("home_team_id"))
        away_id = _optional_str(row.get("away_team_id"))
        if home_id is None or away_id is None:
            continue
        source_row_checksum = _row_checksum(row)
        fixture_id = _fixture_id(row)
        match_id = _required_str(row, "match_id")
        source = _required_str(row, "source")
        source_match_id = _required_str(row, "source_match_id")
        match_status = _required_str(row, "match_status")
        match_date = _required_date(row, "match_date")
        kickoff_utc = _optional_datetime(row.get("kickoff_utc"))
        row_cutoff = _row_cutoff(row, explicit_cutoff=explicit_cutoff)
        source_updated_at = _optional_datetime(row.get("source_updated_at_utc"))
        competition = _required_str(row, "competition")
        stage = _optional_str(row.get("stage"))
        neutral_site = _optional_bool(row.get("neutral_site"))
        venue_name = _optional_str(row.get("venue_name_original"))
        city = _optional_str(row.get("city"))
        host_country = _optional_str(row.get("host_country"))
        extra_time_played = _optional_bool(row.get("extra_time_played"))
        penalty_shootout = _optional_bool(row.get("penalty_shootout"))
        fixtures.append(
            TeamFixture(
                fixture_id=fixture_id,
                match_id=match_id,
                source=source,
                source_match_id=source_match_id,
                match_status=match_status,
                match_date=match_date,
                kickoff_utc=kickoff_utc,
                data_cutoff_utc=row_cutoff,
                source_updated_at_utc=source_updated_at,
                team_id=home_id,
                opponent_team_id=away_id,
                side="home",
                competition=competition,
                stage=stage,
                neutral_site=neutral_site,
                venue_name_original=venue_name,
                city=city,
                host_country=host_country,
                extra_time_played=extra_time_played,
                penalty_shootout=penalty_shootout,
                source_row_checksum=source_row_checksum,
            )
        )
        fixtures.append(
            TeamFixture(
                fixture_id=fixture_id,
                match_id=match_id,
                source=source,
                source_match_id=source_match_id,
                match_status=match_status,
                match_date=match_date,
                kickoff_utc=kickoff_utc,
                data_cutoff_utc=row_cutoff,
                source_updated_at_utc=source_updated_at,
                team_id=away_id,
                opponent_team_id=home_id,
                side="away",
                competition=competition,
                stage=stage,
                neutral_site=neutral_site,
                venue_name_original=venue_name,
                city=city,
                host_country=host_country,
                extra_time_played=extra_time_played,
                penalty_shootout=penalty_shootout,
                source_row_checksum=source_row_checksum,
            )
        )
    fixtures.sort(key=lambda row: (_team_sort_key(row), row.fixture_id, row.team_id, row.side))
    return tuple(fixtures)


def _calculate_team_rows(
    fixtures: Sequence[TeamFixture],
    *,
    venue_by_alias: Mapping[str, Venue],
    feature_generated_at_utc: datetime,
    feature_set_version: str,
    source_dataset_revision: str,
    venue_catalog_checksum: str,
) -> tuple[dict[str, Any], ...]:
    by_team: dict[str, list[TeamFixture]] = defaultdict(list)
    for fixture in fixtures:
        by_team[fixture.team_id].append(fixture)
    rows: list[dict[str, Any]] = []
    rows_by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for team_id in sorted(by_team):
        prior: list[TeamFixture] = []
        prior_feature_rows: list[dict[str, Any]] = []
        streak = 0
        sorted_fixtures = sorted(
            by_team[team_id],
            key=lambda item: (_team_sort_key(item), item.fixture_id),
        )
        for fixture in sorted_fixtures:
            eligible_prior = [
                item
                for item in prior
                if _is_strictly_before(item, fixture)
                and item.data_cutoff_utc <= fixture.data_cutoff_utc
            ]
            previous = eligible_prior[-1] if eligible_prior else None
            previous_feature = prior_feature_rows[-1] if previous is not None else None
            row = _base_team_row(
                fixture,
                feature_generated_at_utc=feature_generated_at_utc,
                feature_set_version=feature_set_version,
                source_dataset_revision=source_dataset_revision,
                venue_catalog_checksum=venue_catalog_checksum,
            )
            current_venue = _match_venue(fixture, venue_by_alias)
            previous_venue = (
                _match_venue(previous, venue_by_alias) if previous is not None else None
            )
            _add_sequence_features(row, fixture, eligible_prior, previous)
            streak = _update_streak(row, previous_feature, previous_streak=streak)
            row["consecutive_matches_without_7d_rest"] = streak
            _add_venue_features(row, fixture, previous, current_venue, previous_venue)
            rows.append(row)
            rows_by_team[team_id].append(row)
            prior.append(fixture)
            prior_feature_rows.append(row)
        _add_cumulative_travel(rows_by_team[team_id])
    return tuple(_ordered_row(row, TEAM_ROW_ORDER) for row in sorted(
        rows,
        key=lambda item: (
            _optional_datetime(item.get("kickoff_utc")) or datetime.combine(
                _required_date(item, "match_date"), datetime.min.time(), tzinfo=UTC
            ),
            str(item.get("fixture_id")),
            str(item.get("side")),
        ),
    ))


def _base_team_row(
    fixture: TeamFixture,
    *,
    feature_generated_at_utc: datetime,
    feature_set_version: str,
    source_dataset_revision: str,
    venue_catalog_checksum: str,
) -> dict[str, Any]:
    return {
        "fixture_id": fixture.fixture_id,
        "match_id": fixture.match_id,
        "source": fixture.source,
        "source_match_id": fixture.source_match_id,
        "match_status": fixture.match_status,
        "match_date": fixture.match_date,
        "kickoff_utc": fixture.kickoff_utc,
        "data_cutoff_utc": fixture.data_cutoff_utc,
        "feature_generated_at_utc": feature_generated_at_utc,
        "feature_set_version": feature_set_version,
        "source_dataset_revision": source_dataset_revision,
        "source_row_checksum": fixture.source_row_checksum,
        "venue_catalog_checksum": venue_catalog_checksum,
        "team_id": fixture.team_id,
        "opponent_team_id": fixture.opponent_team_id,
        "side": fixture.side,
        "competition": fixture.competition,
        "stage": fixture.stage,
        "is_neutral_venue": _neutrality_from_fixture(fixture, None),
    }


def _add_sequence_features(
    row: dict[str, Any],
    fixture: TeamFixture,
    prior: Sequence[TeamFixture],
    previous: TeamFixture | None,
) -> None:
    previous_rest_hours = _rest_hours(previous, fixture)
    previous_rest_days = _rest_days(previous, fixture)
    row["rest_hours"] = previous_rest_hours
    row["rest_days"] = previous_rest_days
    row["hours_since_previous_match"] = previous_rest_hours
    for days in WINDOW_DAYS:
        rows_in_window = [item for item in prior if _within_window(item, fixture, days=days)]
        row[f"matches_last_{days}d"] = len(rows_in_window)
        minutes = _minutes_equivalent_sum(rows_in_window)
        row[f"minutes_equivalent_last_{days}d"] = minutes
    row["previous_match_extra_time"] = (
        _trusted_extra_time(previous) if previous is not None else None
    )
    row["previous_match_penalty_shootout"] = (
        previous.penalty_shootout if previous is not None else None
    )
    same_tournament_prior = [item for item in prior if item.competition == fixture.competition]
    row["tournament_match_number"] = len(same_tournament_prior) + 1
    row["is_first_tournament_match"] = len(same_tournament_prior) == 0
    row["previous_match_id"] = previous.match_id if previous is not None else None
    row["previous_match_date"] = previous.match_date if previous is not None else None
    row["previous_match_kickoff_utc"] = previous.kickoff_utc if previous is not None else None
    row["extra_time_data_quality"] = _extra_time_data_quality(previous)
    row["minutes_equivalent_missing_reason"] = _minutes_missing_reason(prior)


def _update_streak(
    row: Mapping[str, Any],
    previous_feature: Mapping[str, Any] | None,
    *,
    previous_streak: int,
) -> int:
    if previous_feature is None:
        return 0
    rest_days = _optional_float(row.get("rest_days"))
    if rest_days is None:
        return 0
    if rest_days < 7:
        return previous_streak + 1
    return 0


def _add_venue_features(
    row: dict[str, Any],
    fixture: TeamFixture,
    previous: TeamFixture | None,
    current_venue: Venue | None,
    previous_venue: Venue | None,
) -> None:
    if current_venue is not None:
        row.update(
            {
                "venue_id": current_venue.venue_id,
                "venue_city": current_venue.city,
                "venue_country": current_venue.country,
                "venue_latitude": current_venue.latitude,
                "venue_longitude": current_venue.longitude,
                "venue_elevation_m": current_venue.elevation_m,
                "venue_timezone": current_venue.timezone,
            }
        )
    else:
        row.update(
            {
                "venue_id": None,
                "venue_city": None,
                "venue_country": None,
                "venue_latitude": None,
                "venue_longitude": None,
                "venue_elevation_m": None,
                "venue_timezone": None,
            }
        )
    row["is_neutral_venue"] = _neutrality_from_fixture(fixture, current_venue)
    row["host_country_match"] = _host_country_match(fixture.team_id, current_venue)
    row["previous_venue_id"] = previous_venue.venue_id if previous_venue is not None else None
    row["travel_distance_km"] = (
        haversine_distance_km(
            previous_venue.latitude,
            previous_venue.longitude,
            current_venue.latitude,
            current_venue.longitude,
        )
        if previous is not None and previous_venue is not None and current_venue is not None
        else None
    )
    row["timezone_delta_hours"] = (
        timezone_delta_hours(
            current_venue.timezone if current_venue is not None else None,
            previous_venue.timezone if previous_venue is not None else None,
            fixture.kickoff_utc,
        )
        if previous is not None
        else None
    )
    change = (
        elevation_change_m(current_venue.elevation_m, previous_venue.elevation_m)
        if current_venue is not None and previous_venue is not None
        else None
    )
    row["elevation_change_m"] = change
    row["absolute_elevation_change_m"] = abs(change) if change is not None else None
    row["cross_border_travel"] = (
        current_venue.country != previous_venue.country
        if current_venue is not None and previous_venue is not None
        else None
    )
    for days in WINDOW_DAYS:
        row[f"cumulative_travel_km_{days}d"] = None


def _add_cumulative_travel(rows: Sequence[dict[str, Any]]) -> None:
    for index, current in enumerate(rows):
        for days in WINDOW_DAYS:
            distances: list[float] = []
            for candidate in rows[: index + 1]:
                distance = _optional_float(candidate.get("travel_distance_km"))
                if distance is None:
                    continue
                if _team_row_within_window(candidate, current, days=days):
                    distances.append(distance)
            current[f"cumulative_travel_km_{days}d"] = sum(distances) if distances else None


def _match_level_rows(team_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    by_fixture: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in team_rows:
        side = _required_str(row, "side")
        by_fixture[_required_str(row, "fixture_id")][side] = row
    rows: list[dict[str, Any]] = []
    for fixture_id in sorted(by_fixture):
        sides = by_fixture[fixture_id]
        if "home" not in sides or "away" not in sides:
            continue
        home = sides["home"]
        away = sides["away"]
        output: dict[str, Any] = {
            "fixture_id": fixture_id,
            "match_id": home["match_id"],
            "source": home["source"],
            "source_match_id": home["source_match_id"],
            "match_status": home["match_status"],
            "match_date": home["match_date"],
            "kickoff_utc": home["kickoff_utc"],
            "data_cutoff_utc": home["data_cutoff_utc"],
            "feature_generated_at_utc": home["feature_generated_at_utc"],
            "feature_set_version": home["feature_set_version"],
            "source_dataset_revision": home["source_dataset_revision"],
            "venue_catalog_checksum": home["venue_catalog_checksum"],
            "competition": home["competition"],
            "stage": home["stage"],
            "is_neutral_venue": home["is_neutral_venue"],
        }
        for prefix, source_row in (("home", home), ("away", away)):
            for feature in PREFIXED_FEATURES:
                output[f"{prefix}_{feature}"] = source_row.get(feature)
        for feature in NUMERIC_DIFF_FEATURES:
            home_value = _optional_float(home.get(feature))
            away_value = _optional_float(away.get(feature))
            output[f"{feature}_diff"] = (
                home_value - away_value
                if home_value is not None and away_value is not None
                else None
            )
        rows.append(output)
    return tuple(rows)


def _rest_hours(previous: TeamFixture | None, current: TeamFixture) -> float | None:
    if previous is None or previous.kickoff_utc is None or current.kickoff_utc is None:
        return None
    return (current.kickoff_utc - previous.kickoff_utc).total_seconds() / 3600


def _rest_days(previous: TeamFixture | None, current: TeamFixture) -> float | None:
    hours = _rest_hours(previous, current)
    if hours is not None:
        return hours / 24
    if previous is None:
        return None
    delta = (current.match_date - previous.match_date).days
    return float(delta) if delta > 0 else None


def _within_window(previous: TeamFixture, current: TeamFixture, *, days: int) -> bool:
    if not _is_strictly_before(previous, current):
        return False
    if previous.kickoff_utc is not None and current.kickoff_utc is not None:
        delta = current.kickoff_utc - previous.kickoff_utc
        return timedelta(0) < delta <= timedelta(days=days)
    delta_days = (current.match_date - previous.match_date).days
    return 0 < delta_days <= days


def _team_row_within_window(
    previous: Mapping[str, Any], current: Mapping[str, Any], *, days: int
) -> bool:
    previous_kickoff = _optional_datetime(previous.get("kickoff_utc"))
    current_kickoff = _optional_datetime(current.get("kickoff_utc"))
    if previous_kickoff is not None and current_kickoff is not None:
        delta = current_kickoff - previous_kickoff
        return timedelta(0) <= delta <= timedelta(days=days)
    previous_date = _required_date(previous, "match_date")
    current_date = _required_date(current, "match_date")
    delta_days = (current_date - previous_date).days
    return 0 <= delta_days <= days


def _minutes_equivalent_sum(rows: Sequence[TeamFixture]) -> float | None:
    if not rows:
        return 0.0
    minutes = 0.0
    for row in rows:
        equivalent = _minutes_equivalent(row)
        if equivalent is None:
            return None
        minutes += equivalent
    return minutes


def _minutes_equivalent(row: TeamFixture) -> float | None:
    extra_time = _trusted_extra_time(row)
    if extra_time is None:
        return None
    return 120.0 if extra_time else 90.0


def _trusted_extra_time(row: TeamFixture | None) -> bool | None:
    if row is None:
        return None
    if row.source == "international_results_csv":
        return None
    return row.extra_time_played


def _extra_time_data_quality(row: TeamFixture | None) -> str | None:
    if row is None:
        return None
    if row.source == "international_results_csv":
        return "unknown_historical_extra_time"
    return "trusted_source_contract"


def _minutes_missing_reason(rows: Sequence[TeamFixture]) -> str | None:
    if any(row.source == "international_results_csv" for row in rows):
        return "historical_extra_time_not_safely_distinguishable"
    return None


def _is_strictly_before(previous: TeamFixture, current: TeamFixture) -> bool:
    if previous.kickoff_utc is not None and current.kickoff_utc is not None:
        return previous.kickoff_utc < current.kickoff_utc
    return previous.match_date < current.match_date


def _team_sort_key(row: TeamFixture) -> tuple[datetime, str]:
    instant = row.kickoff_utc or datetime.combine(row.match_date, datetime.min.time(), tzinfo=UTC)
    return instant, row.match_id


def _match_sort_date(row: Mapping[str, Any]) -> date:
    return _required_date(row, "match_date")


def _row_cutoff(row: Mapping[str, Any], *, explicit_cutoff: datetime | None) -> datetime:
    if explicit_cutoff is not None:
        return explicit_cutoff
    for field in ("data_cutoff_utc", "retrieved_at_utc", "source_updated_at_utc"):
        value = _optional_datetime(row.get(field))
        if value is not None:
            return value
    raise ContextualFeatureError(f"row has no usable data cutoff: {row.get('match_id')}")


def _fixture_id(row: Mapping[str, Any]) -> str:
    source_match_id = _optional_str(row.get("source_match_id"))
    if source_match_id is not None:
        return source_match_id
    return _required_str(row, "match_id")


def _neutrality_from_fixture(fixture: TeamFixture, venue: Venue | None) -> bool | None:
    if fixture.neutral_site is not None:
        return fixture.neutral_site
    host_match = _host_country_match(fixture.team_id, venue)
    if host_match is not None:
        return not host_match
    return None


def _host_country_match(team_id: str, venue: Venue | None) -> bool | None:
    if venue is None:
        return None
    host_team_id = HOST_COUNTRY_TEAMS.get(venue.country)
    if host_team_id is None:
        return False
    return team_id == host_team_id


def _source_dataset_revision(
    rows: Sequence[Mapping[str, Any]], *, venue_checksum: str
) -> str:
    payload = {
        "rows": [_stable_json(row) for row in rows],
        "venue_catalog_checksum": venue_checksum,
        "feature_set_version": FEATURE_SET_VERSION,
    }
    checksum = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{SOURCE_DATASET_REVISION_PREFIX}:{checksum[:16]}"


def _row_checksum(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(_stable_json(row).encode("utf-8")).hexdigest()


def _stable_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: pair[0])
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return _format_utc(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _manifest(
    *,
    input_rows: Sequence[Mapping[str, Any]],
    team_rows: Sequence[Mapping[str, Any]],
    match_rows: Sequence[Mapping[str, Any]],
    generated_at: datetime,
    data_cutoff_utc: datetime | None,
    source_dataset_revision: str,
    venue_catalog_path: Path,
    venue_catalog_checksum: str,
    audit: LeakageAudit,
) -> Mapping[str, Any]:
    return {
        "schema_version": CONTEXTUAL_SCHEMA_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "generated_at_utc": _format_utc(generated_at),
        "data_cutoff_utc": _format_utc(data_cutoff_utc) if data_cutoff_utc is not None else None,
        "source_dataset_revision": source_dataset_revision,
        "source_rows": len(input_rows),
        "team_fixture_rows": len(team_rows),
        "match_rows": len(match_rows),
        "venue_catalog": {
            "path": venue_catalog_path.as_posix(),
            "schema_version": VENUE_CATALOG_SCHEMA_VERSION,
            "checksum": venue_catalog_checksum,
        },
        "checksums": {
            "source_dataset_revision": source_dataset_revision,
            "venue_catalog_checksum": venue_catalog_checksum,
        },
        "leakage_audit_passed": audit.passed,
        "leakage_violations": len(audit.violations),
        "operational_model_changed": False,
        "model_version": "poisson_goal_v1 unchanged",
    }


def _feature_coverage(
    rows: Sequence[Mapping[str, Any]], feature: str
) -> Mapping[str, int | float]:
    total = len(rows)
    non_null = sum(1 for row in rows if row.get(feature) is not None)
    missing = total - non_null
    return {
        "rows": total,
        "non_null": non_null,
        "missing": missing,
        "coverage": _safe_ratio(non_null, total),
        "missing_percent": _safe_ratio(missing, total),
    }


def _coverage_by_dimension(
    rows: Sequence[Mapping[str, Any]], dimension: str, features: Sequence[str]
) -> Mapping[str, Mapping[str, Mapping[str, int | float]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if dimension == "match_year":
            key = str(_required_date(row, "match_date").year)
        else:
            key = str(row.get(dimension) or "missing")
        grouped[key].append(row)
    return {
        key: {feature: _feature_coverage(group, feature) for feature in features}
        for key, group in sorted(grouped.items())
    }


def _feature_classification(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Mapping[str, Any]]:
    historical_rows = [row for row in rows if _is_historical_source(row)]
    output: dict[str, Mapping[str, Any]] = {}
    for feature in LEVEL_A_FEATURES:
        historical_coverage = float(_feature_coverage(historical_rows, feature)["coverage"])
        missingness = float(_feature_coverage(historical_rows, feature)["missing_percent"])
        if historical_coverage >= TRAINABLE_COVERAGE_THRESHOLD:
            category = "historically_trainable"
        elif missingness >= MISSING_INDICATOR_THRESHOLD:
            category = "usable_only_as_missing_indicator_or_exclude"
        else:
            category = "not_available_with_sufficient_quality"
        output[feature] = {
            "tier": "A",
            "category": category,
            "historical_coverage": historical_coverage,
            "depends_on_external_source": False,
        }
    for feature in LEVEL_B_FEATURES:
        output[feature] = {
            "tier": "B",
            "category": "world_cup_2026_operational_only",
            "historical_coverage": 0.0,
            "depends_on_external_source": feature.startswith("venue")
            or feature
            in {
                "previous_venue_id",
                "travel_distance_km",
                "cumulative_travel_km_7d",
                "cumulative_travel_km_14d",
                "cumulative_travel_km_30d",
                "timezone_delta_hours",
                "elevation_change_m",
                "absolute_elevation_change_m",
                "cross_border_travel",
                "host_country_match",
            },
        }
    return output


def _impossible_value_counts(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    return {
        "negative_rest_hours": sum(1 for row in rows if _is_negative(row.get("rest_hours"))),
        "negative_travel_distance_km": sum(
            1 for row in rows if _is_negative(row.get("travel_distance_km"))
        ),
        "invalid_latitude": sum(
            1
            for row in rows
            if row.get("venue_latitude") is not None
            and not -90 <= float(row["venue_latitude"]) <= 90
        ),
        "invalid_longitude": sum(
            1
            for row in rows
            if row.get("venue_longitude") is not None
            and not -180 <= float(row["venue_longitude"]) <= 180
        ),
        "invalid_elevation": sum(
            1
            for row in rows
            if row.get("venue_elevation_m") is not None
            and not (
                REASONABLE_ELEVATION_RANGE_M[0]
                <= float(row["venue_elevation_m"])
                <= REASONABLE_ELEVATION_RANGE_M[1]
            )
        ),
    }


def _duplicate_team_fixture_count(rows: Sequence[Mapping[str, Any]]) -> int:
    keys = [(str(row.get("fixture_id")), str(row.get("team_id"))) for row in rows]
    return sum(count - 1 for count in Counter(keys).values() if count > 1)


def _distribution(
    rows: Sequence[Mapping[str, Any]], feature: str
) -> Mapping[str, float | int | None]:
    values = sorted(
        float(row[feature])
        for row in rows
        if row.get(feature) is not None and math.isfinite(float(row[feature]))
    )
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    return {
        "count": len(values),
        "min": values[0],
        "p25": _quantile(values, 0.25),
        "median": _quantile(values, 0.5),
        "p75": _quantile(values, 0.75),
        "max": values[-1],
    }


def _quantile(values: Sequence[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[int(position)]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _append_geography_violations(
    row: Mapping[str, Any], violations: list[Mapping[str, Any]]
) -> None:
    try:
        if row.get("venue_latitude") is not None:
            _validate_latitude(float(row["venue_latitude"]))
        if row.get("venue_longitude") is not None:
            _validate_longitude(float(row["venue_longitude"]))
        if row.get("venue_elevation_m") is not None:
            _validate_elevation(float(row["venue_elevation_m"]))
        if row.get("venue_timezone") is not None:
            _zoneinfo(str(row["venue_timezone"]))
    except ContextualFeatureError as exc:
        violations.append(_violation(row, f"invalid_geography: {exc}"))


def _append_negative_value_violations(
    row: Mapping[str, Any], violations: list[Mapping[str, Any]]
) -> None:
    for field in (
        "rest_hours",
        "rest_days",
        "hours_since_previous_match",
        "travel_distance_km",
        "cumulative_travel_km_7d",
        "cumulative_travel_km_14d",
        "cumulative_travel_km_30d",
    ):
        if _is_negative(row.get(field)):
            violations.append(_violation(row, f"negative_{field}"))


def _violation(row: Mapping[str, Any], reason: str) -> Mapping[str, Any]:
    return {
        "fixture_id": row.get("fixture_id"),
        "team_id": row.get("team_id"),
        "reason": reason,
    }


def _is_prospective_row(row: Mapping[str, Any]) -> bool:
    kickoff = _optional_datetime(row.get("kickoff_utc"))
    data_cutoff = _optional_datetime(row.get("data_cutoff_utc"))
    return str(row.get("match_status")) == "scheduled" and kickoff is not None and (
        data_cutoff is None or data_cutoff < kickoff
    )


def _is_match_prospective_row(row: Mapping[str, Any]) -> bool:
    kickoff = _optional_datetime(row.get("kickoff_utc"))
    data_cutoff = _optional_datetime(row.get("data_cutoff_utc"))
    return str(row.get("match_status")) == "scheduled" and kickoff is not None and (
        data_cutoff is None or data_cutoff < kickoff
    )


def _is_historical_source(row: Mapping[str, Any]) -> bool:
    return str(row.get("source")) == "international_results_csv"


def _is_world_cup_2026_source(row: Mapping[str, Any]) -> bool:
    return str(row.get("source")).startswith("world_cup_2026_")


def _is_negative(value: object) -> bool:
    number = _optional_float(value)
    return number is not None and number < 0


def _row_ref(row: Mapping[str, Any]) -> Mapping[str, str]:
    return {
        "fixture_id": str(row.get("fixture_id")),
        "team_id": str(row.get("team_id")),
    }


def _ordered_row(row: Mapping[str, Any], order: Sequence[str]) -> dict[str, Any]:
    return {field: row.get(field) for field in order}


def _coverage_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Contextual Feature Coverage",
        "",
        f"Rows total: {report.get('rows_total', 0)}",
        "",
        "| Feature | Coverage | Missing | Classification |",
        "| --- | ---: | ---: | --- |",
    ]
    coverage = report.get("coverage_by_feature")
    classification = report.get("feature_classification")
    if isinstance(coverage, Mapping) and isinstance(classification, Mapping):
        for feature in sorted(coverage):
            item = coverage[feature]
            class_item = classification.get(feature)
            if isinstance(item, Mapping) and isinstance(class_item, Mapping):
                lines.append(
                    f"| {feature} | {float(item['coverage']):.2%} | "
                    f"{float(item['missing_percent']):.2%} | {class_item['category']} |"
                )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Historical source rows are date-only, so hour-based rest is missing.",
            "- Historical extra time cannot be safely distinguished for all matches.",
            "- Venue diagnostics are operational-only until historical venue coverage improves.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_value) + "\n")


def _team_feature_schema() -> pa.Schema:
    return pa.schema(
        [
            ("fixture_id", pa.string()),
            ("match_id", pa.string()),
            ("source", pa.string()),
            ("source_match_id", pa.string()),
            ("match_status", pa.string()),
            ("match_date", pa.date32()),
            ("kickoff_utc", pa.timestamp("us", tz="UTC")),
            ("data_cutoff_utc", pa.timestamp("us", tz="UTC")),
            ("feature_generated_at_utc", pa.timestamp("us", tz="UTC")),
            ("feature_set_version", pa.string()),
            ("source_dataset_revision", pa.string()),
            ("source_row_checksum", pa.string()),
            ("venue_catalog_checksum", pa.string()),
            ("team_id", pa.string()),
            ("opponent_team_id", pa.string()),
            ("side", pa.string()),
            ("competition", pa.string()),
            ("stage", pa.string()),
            ("is_neutral_venue", pa.bool_()),
            ("rest_hours", pa.float64()),
            ("rest_days", pa.float64()),
            ("matches_last_7d", pa.int64()),
            ("matches_last_14d", pa.int64()),
            ("matches_last_30d", pa.int64()),
            ("minutes_equivalent_last_7d", pa.float64()),
            ("minutes_equivalent_last_14d", pa.float64()),
            ("minutes_equivalent_last_30d", pa.float64()),
            ("previous_match_extra_time", pa.bool_()),
            ("previous_match_penalty_shootout", pa.bool_()),
            ("hours_since_previous_match", pa.float64()),
            ("consecutive_matches_without_7d_rest", pa.int64()),
            ("tournament_match_number", pa.int64()),
            ("is_first_tournament_match", pa.bool_()),
            ("previous_match_id", pa.string()),
            ("previous_match_date", pa.date32()),
            ("previous_match_kickoff_utc", pa.timestamp("us", tz="UTC")),
            ("extra_time_data_quality", pa.string()),
            ("minutes_equivalent_missing_reason", pa.string()),
            ("venue_id", pa.string()),
            ("venue_city", pa.string()),
            ("venue_country", pa.string()),
            ("venue_latitude", pa.float64()),
            ("venue_longitude", pa.float64()),
            ("venue_elevation_m", pa.float64()),
            ("venue_timezone", pa.string()),
            ("previous_venue_id", pa.string()),
            ("travel_distance_km", pa.float64()),
            ("cumulative_travel_km_7d", pa.float64()),
            ("cumulative_travel_km_14d", pa.float64()),
            ("cumulative_travel_km_30d", pa.float64()),
            ("timezone_delta_hours", pa.float64()),
            ("elevation_change_m", pa.float64()),
            ("absolute_elevation_change_m", pa.float64()),
            ("cross_border_travel", pa.bool_()),
            ("host_country_match", pa.bool_()),
        ]
    )


def _match_feature_schema() -> pa.Schema:
    fields = [
        ("fixture_id", pa.string()),
        ("match_id", pa.string()),
        ("source", pa.string()),
        ("source_match_id", pa.string()),
        ("match_status", pa.string()),
        ("match_date", pa.date32()),
        ("kickoff_utc", pa.timestamp("us", tz="UTC")),
        ("data_cutoff_utc", pa.timestamp("us", tz="UTC")),
        ("feature_generated_at_utc", pa.timestamp("us", tz="UTC")),
        ("feature_set_version", pa.string()),
        ("source_dataset_revision", pa.string()),
        ("venue_catalog_checksum", pa.string()),
        ("competition", pa.string()),
        ("stage", pa.string()),
        ("is_neutral_venue", pa.bool_()),
    ]
    for prefix in ("home", "away"):
        for feature in PREFIXED_FEATURES:
            fields.append((f"{prefix}_{feature}", _prefixed_feature_type(feature)))
    for feature in NUMERIC_DIFF_FEATURES:
        fields.append((f"{feature}_diff", pa.float64()))
    return pa.schema(fields)


def _prefixed_feature_type(feature: str) -> pa.DataType:
    if feature in {
        "team_id",
        "opponent_team_id",
        "venue_id",
        "venue_city",
        "venue_country",
        "venue_timezone",
        "previous_venue_id",
    }:
        return pa.string()
    if feature in {
        "matches_last_7d",
        "matches_last_14d",
        "matches_last_30d",
        "consecutive_matches_without_7d_rest",
        "tournament_match_number",
    }:
        return pa.int64()
    if feature in {
        "previous_match_extra_time",
        "previous_match_penalty_shootout",
        "is_first_tournament_match",
        "cross_border_travel",
        "host_country_match",
    }:
        return pa.bool_()
    return pa.float64()


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _format_utc(value: datetime) -> str:
    return _require_utc(value, field_name="datetime").isoformat().replace("+00:00", "Z")


def _required_str(row: Mapping[str, Any], field_name: str) -> str:
    value = _optional_str(row.get(field_name))
    if value is None:
        raise ContextualFeatureError(f"missing required string field {field_name}")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_date(row: Mapping[str, Any], field_name: str) -> date:
    value = _optional_date(row.get(field_name))
    if value is None:
        raise ContextualFeatureError(f"missing required date field {field_name}")
    return value


def _optional_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ContextualFeatureError(f"expected date value, got {value!r}")


def _required_datetime(row: Mapping[str, Any], field_name: str) -> datetime:
    value = _optional_datetime(row.get(field_name))
    if value is None:
        raise ContextualFeatureError(f"missing required datetime field {field_name}")
    return value


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _require_utc(value, field_name="datetime")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return _require_utc(parsed, field_name="datetime")
    raise ContextualFeatureError(f"expected datetime value, got {value!r}")


def _require_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ContextualFeatureError(f"{field_name} must be timezone-aware UTC")
    return value.astimezone(UTC)


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ContextualFeatureError(f"expected bool or None, got {value!r}")


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None
