"""Traceable live World Cup 2026 fixture and result ingestion.

Raw provider responses are append-only.  The operational table is a convenience view; every
version of that table is also written beside the raw snapshot so a past data cutoff can be
reconstructed without asking a provider to replay history.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from worldcup2026.config import Settings
from worldcup2026.data.contracts import (
    CanonicalMatch,
    CanonicalTeam,
    HomeAdvantageStatus,
    KickoffTimeStatus,
    MatchStatus,
    MatchType,
    Result90,
    TeamAlias,
    resolve_team_alias,
)
from worldcup2026.data.historical_ingest import load_team_aliases, load_team_catalog
from worldcup2026.data.sources import sha256_bytes

SNAPSHOT_SCHEMA_REVISION = "world_cup_raw_snapshot_v1"
PRIMARY_SOURCE_PREFIX = "world_cup_2026"


class WorldCupIngestError(RuntimeError):
    """Raised when live World Cup ingestion cannot proceed safely."""


class MissingApiKeyError(WorldCupIngestError):
    """Raised without exposing the missing or supplied secret value."""


class ProviderRequestError(WorldCupIngestError):
    """Raised after bounded network retries are exhausted."""


class LiveSnapshotManifest(BaseModel):
    """Metadata for one immutable raw provider response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_revision: str = SNAPSHOT_SCHEMA_REVISION
    provider: str
    endpoint: str
    source_fixture_id: str
    fetched_at: datetime
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_path: Path
    response_scope: str


class UnresolvedLiveTeam(BaseModel):
    """A source name that did not have an exact catalog resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    source_fixture_id: str
    side: str
    source_name: str
    match_date: str


class ParticipantStatus(StrEnum):
    """Whether a provider has determined both participants for a fixture."""

    KNOWN = "known"
    PARTIALLY_KNOWN = "partially_known"
    TBD = "tbd"


class InvalidProviderFixture(BaseModel):
    """One malformed provider record, retained as an ingestion finding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_fixture_id: str | None
    reason: str
    field: str | None = None
    source_status: str | None = None


class PendingParticipantFixture(BaseModel):
    """A fixture whose participants are not yet fully determined by the provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_fixture_id: str
    participants_status: ParticipantStatus
    home_name: str | None
    away_name: str | None
    kickoff_utc: datetime
    source_status: str
    stage: str | None
    source_updated_at_utc: datetime | None
    score: dict[str, int | bool | None]
    raw_snapshot_path: Path


class SnapshotDifference(BaseModel):
    """A semantic change observed between two canonical live snapshots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_fixture_id: str
    field: str
    previous: object
    current: object


class ResultDiscrepancy(BaseModel):
    """A difference reported by the optional validation source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    primary_fixture_id: str | None
    secondary_fixture_id: str | None
    detail: str


class FreshnessReport(BaseModel):
    """Current temporal state of the World Cup feed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    last_fetch_utc: datetime
    next_kickoff_utc: datetime | None
    fixtures_without_recent_update: tuple[str, ...]
    differences_from_previous_snapshot: tuple[SnapshotDifference, ...]


class WorldCupIngestReport(BaseModel):
    """Serializable result of one live ingestion attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    fetched_at: datetime
    dry_run: bool
    snapshot_manifests: tuple[LiveSnapshotManifest, ...]
    provider_fixtures_received: int
    fixtures_with_known_participants: int
    fixtures_with_partially_known_participants: int
    fixtures_with_tbd_participants: int
    invalid_provider_fixtures: int
    invalid_fixtures: int
    invalid_fixture_records: tuple[InvalidProviderFixture, ...]
    canonical_matches: int
    unresolved_named_teams: int
    pending_fixtures: int
    in_progress_matches: int
    finished_matches: int
    interrupted_matches: int
    source_status_counts: Mapping[str, int]
    canonical_status_counts: Mapping[str, int]
    data_cutoff_utc: datetime
    snapshot_checksum: str
    snapshot_reference: str
    unresolved_teams: tuple[UnresolvedLiveTeam, ...]
    freshness: FreshnessReport
    validation_discrepancies: tuple[ResultDiscrepancy, ...]
    operational_table: Path | None


class WorldCupIngestResult(BaseModel):
    """Programmatic return value for live ingestion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    matches: tuple[CanonicalMatch, ...]
    tbd_fixtures: tuple[PendingParticipantFixture, ...]
    partially_known_fixtures: tuple[PendingParticipantFixture, ...]
    report: WorldCupIngestReport


@dataclass(frozen=True)
class ProviderFixture:
    """Provider-neutral observed match facts before catalog normalization."""

    source_fixture_id: str
    participants_status: ParticipantStatus
    home_name: str | None
    away_name: str | None
    kickoff_utc: datetime
    original_timezone: str
    venue_name: str | None
    city: str | None
    source_status: str
    stage: str | None
    source_updated_at_utc: datetime | None
    home_goals_90: int | None
    away_goals_90: int | None
    extra_time_played: bool
    home_goals_after_extra_time: int | None
    away_goals_after_extra_time: int | None
    penalty_shootout: bool
    home_penalty_goals: int | None
    away_penalty_goals: int | None


@dataclass(frozen=True)
class ProviderResponse:
    """The original response and parsed fixtures returned by one provider endpoint."""

    provider: str
    endpoint: str
    payload: bytes
    fixtures: tuple[ProviderFixture, ...]
    fixture_payloads: Mapping[str, bytes]
    invalid_fixtures: tuple[InvalidProviderFixture, ...] = ()


class WorldCupProvider(Protocol):
    """Provider interface deliberately separate from pipeline orchestration."""

    name: str

    def fetch(self) -> ProviderResponse:
        """Fetch the current tournament fixture collection."""


class HttpTransport(Protocol):
    """Minimal transport boundary that keeps retry behavior independently testable."""

    def get(self, url: str, *, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        """Perform one HTTP GET."""


class _HttpxTransport:
    def get(self, url: str, *, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        with httpx.Client() as client:
            return client.get(url, headers=dict(headers), timeout=timeout)


class RateLimitedHttpClient:
    """Bounded retrying client with local cache and no secret-bearing diagnostics."""

    def __init__(
        self,
        *,
        cache_root: Path,
        timeout_seconds: float,
        retries: int,
        min_interval_seconds: float,
        cache_ttl_seconds: int,
        transport: HttpTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache_root = cache_root
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.min_interval_seconds = min_interval_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self.transport = transport or _HttpxTransport()
        self.clock = clock
        self.sleeper = sleeper
        self._last_request_at: float | None = None

    def get_json(self, *, url: str, headers: Mapping[str, str], cache_namespace: str) -> bytes:
        """Return JSON bytes, using a short-lived cache before contacting the provider."""

        cache_path = self._cache_path(url, cache_namespace)
        if self._cache_is_fresh(cache_path):
            return cache_path.read_bytes()

        last_error: ProviderRequestError | Exception | None = None
        for attempt in range(self.retries + 1):
            self._wait_for_rate_limit()
            try:
                response = self.transport.get(url, headers=headers, timeout=self.timeout_seconds)
                if response.status_code == 429 or response.status_code >= 500:
                    raise ProviderRequestError(
                        _provider_http_error(cache_namespace, response)
                    )
                if response.status_code >= 400:
                    raise ProviderRequestError(
                        _provider_http_error(cache_namespace, response)
                    )
                content = response.content
                self._validate_json(content)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(content)
                return content
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                ProviderRequestError,
            ) as exc:
                last_error = exc
                retryable = not isinstance(exc, ProviderRequestError) or _is_retryable(exc)
                if attempt < self.retries and retryable:
                    self.sleeper(0.25 * (2**attempt))
                    continue
                break
        detail = type(last_error).__name__ if last_error is not None else "unknown error"
        if isinstance(last_error, ProviderRequestError):
            detail = str(last_error)
        raise ProviderRequestError(
            f"World Cup provider request failed after {self.retries + 1} attempts ({detail})"
        )

    def _wait_for_rate_limit(self) -> None:
        now = self.clock()
        if self._last_request_at is not None:
            remaining = self.min_interval_seconds - (now - self._last_request_at)
            if remaining > 0:
                self.sleeper(remaining)
        self._last_request_at = self.clock()

    def _cache_path(self, url: str, namespace: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_root / namespace / f"{digest}.json"

    def _cache_is_fresh(self, path: Path) -> bool:
        if not path.is_file():
            return False
        age = time.time() - path.stat().st_mtime
        return age <= self.cache_ttl_seconds

    @staticmethod
    def _validate_json(content: bytes) -> None:
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            raise ProviderRequestError("provider returned invalid JSON") from exc


def _provider_http_error(provider: str, response: httpx.Response) -> str:
    message = _safe_provider_message(response)
    return f"{provider} HTTP {response.status_code}: {message}"


def _safe_provider_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, Mapping):
        for key in ("message", "error", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return _redact_sensitive_text(value.strip())
    text = response.text.strip()[:200]
    if text:
        return _redact_sensitive_text(text)
    return response.reason_phrase or "HTTP error"


def _redact_sensitive_text(value: str) -> str:
    redacted = value
    for marker in ("X-Auth-Token", "x-apisports-key", "Authorization", "token", "key"):
        redacted = redacted.replace(marker, "[redacted]")
    return redacted


def _is_retryable(exc: ProviderRequestError) -> bool:
    text = str(exc)
    return " HTTP 429:" in text or any(f" HTTP {status}:" in text for status in range(500, 600))


def _safe_provider_error(value: str) -> str:
    return _redact_sensitive_text(value)[:300]


class FootballDataProvider:
    """Primary adapter for football-data.org's World Cup competition endpoint."""

    name = "football_data"

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        competition_code: str,
        client: RateLimitedHttpClient,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.competition_code = competition_code
        self.client = client

    def fetch(self) -> ProviderResponse:
        if not self.api_key:
            raise MissingApiKeyError("FOOTBALL_DATA_API_KEY is required for provider football_data")
        endpoint = f"{self.base_url}/competitions/{self.competition_code}/matches?season=2026"
        payload = self.client.get_json(
            url=endpoint,
            headers={"X-Auth-Token": self.api_key},
            cache_namespace=self.name,
        )
        try:
            raw = json.loads(payload)
            matches = raw["matches"] if isinstance(raw, dict) else raw
            fixtures, invalid, items = _parse_fixture_collection(matches, _football_data_fixture)
        except (KeyError, TypeError, ValueError) as exc:
            raise WorldCupIngestError(
                "football_data response does not match the documented match schema"
            ) from exc
        return ProviderResponse(
            self.name,
            endpoint,
            payload,
            fixtures,
            _fixture_payloads(fixtures, items),
            invalid,
        )


class ApiFootballProvider:
    """Optional result-validation adapter for API-Football, not a second pipeline."""

    name = "api_football"

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        league_id: int,
        client: RateLimitedHttpClient,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.league_id = league_id
        self.client = client

    def fetch(self) -> ProviderResponse:
        if not self.api_key:
            raise MissingApiKeyError("API_FOOTBALL_KEY is required for provider api_football")
        endpoint = f"{self.base_url}/fixtures?league={self.league_id}&season=2026"
        payload = self.client.get_json(
            url=endpoint,
            headers={"x-apisports-key": self.api_key},
            cache_namespace=self.name,
        )
        try:
            raw = json.loads(payload)
            fixtures, invalid, items = _parse_fixture_collection(
                raw["response"], _api_football_fixture
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WorldCupIngestError(
                "api_football response does not match the documented fixture schema"
            ) from exc
        return ProviderResponse(
            self.name,
            endpoint,
            payload,
            fixtures,
            _fixture_payloads(fixtures, items),
            invalid,
        )


def provider_from_settings(settings: Settings, *, cache_root: Path) -> WorldCupProvider:
    """Construct only the configured primary provider from environment-backed settings."""

    client = RateLimitedHttpClient(
        cache_root=cache_root,
        timeout_seconds=settings.world_cup_http_timeout_seconds,
        retries=settings.world_cup_http_retries,
        min_interval_seconds=settings.world_cup_rate_limit_seconds,
        cache_ttl_seconds=settings.world_cup_cache_ttl_seconds,
    )
    if settings.world_cup_provider == FootballDataProvider.name:
        return FootballDataProvider(
            api_key=settings.football_data_api_key,
            base_url=settings.football_data_base_url,
            competition_code=settings.football_data_competition_code,
            client=client,
        )
    if settings.world_cup_provider == ApiFootballProvider.name:
        return ApiFootballProvider(
            api_key=settings.api_football_key,
            base_url=settings.api_football_base_url,
            league_id=settings.api_football_league_id,
            client=client,
        )
    raise WorldCupIngestError("WORLD_CUP_PROVIDER must be football_data or api_football")


def secondary_provider_from_settings(
    settings: Settings, *, cache_root: Path
) -> WorldCupProvider | None:
    """Return a configured secondary source solely for result validation."""

    if settings.world_cup_provider == FootballDataProvider.name and settings.api_football_key:
        client = RateLimitedHttpClient(
            cache_root=cache_root,
            timeout_seconds=settings.world_cup_http_timeout_seconds,
            retries=settings.world_cup_http_retries,
            min_interval_seconds=settings.world_cup_rate_limit_seconds,
            cache_ttl_seconds=settings.world_cup_cache_ttl_seconds,
        )
        return ApiFootballProvider(
            api_key=settings.api_football_key,
            base_url=settings.api_football_base_url,
            league_id=settings.api_football_league_id,
            client=client,
        )
    if settings.world_cup_provider == ApiFootballProvider.name and settings.football_data_api_key:
        client = RateLimitedHttpClient(
            cache_root=cache_root,
            timeout_seconds=settings.world_cup_http_timeout_seconds,
            retries=settings.world_cup_http_retries,
            min_interval_seconds=settings.world_cup_rate_limit_seconds,
            cache_ttl_seconds=settings.world_cup_cache_ttl_seconds,
        )
        return FootballDataProvider(
            api_key=settings.football_data_api_key,
            base_url=settings.football_data_base_url,
            competition_code=settings.football_data_competition_code,
            client=client,
        )
    return None


def run_world_cup_ingest(
    provider: WorldCupProvider,
    *,
    raw_root: Path = Path("data/raw"),
    processed_root: Path = Path("data/processed/world_cup_2026"),
    interim_root: Path = Path("data/interim"),
    teams_path: Path = Path("data/static/teams.csv"),
    aliases_path: Path = Path("data/static/team_aliases.csv"),
    secondary_provider: WorldCupProvider | None = None,
    dry_run: bool = False,
    fetched_at: datetime | None = None,
    stale_after: timedelta = timedelta(hours=6),
) -> WorldCupIngestResult:
    """Fetch, snapshot, normalize, diff, and publish the current operational view."""

    current_fetched_at = _utc_now() if fetched_at is None else _require_utc(fetched_at)
    response = provider.fetch()
    teams = load_team_catalog(teams_path)
    aliases = load_team_aliases(aliases_path, teams=teams)
    matches, unresolved, normalization_invalid = _normalize_fixtures(
        response.fixtures,
        provider=response.provider,
        fetched_at=current_fetched_at,
        teams=teams,
        aliases=aliases,
    )
    sorted_matches = tuple(sorted(matches, key=lambda match: (match.kickoff_utc, match.match_id)))
    manifests = _build_manifests(
        response,
        fetched_at=current_fetched_at,
        raw_root=raw_root,
    )
    pending_participants = _pending_participant_fixtures(response.fixtures, manifests)
    tbd_fixtures = tuple(
        item for item in pending_participants if item.participants_status is ParticipantStatus.TBD
    )
    partially_known_fixtures = tuple(
        item
        for item in pending_participants
        if item.participants_status is ParticipantStatus.PARTIALLY_KNOWN
    )
    invalid_fixtures = [*response.invalid_fixtures, *normalization_invalid]
    primary_invalid_provider_fixtures = len(invalid_fixtures)
    previous_matches = _read_latest_snapshot_matches(processed_root)
    previous_pending = _read_latest_snapshot_pending_participant_fixtures(processed_root)
    differences = _snapshot_differences(previous_matches, sorted_matches)
    differences = (
        *differences,
        *_participant_status_differences(previous_pending, pending_participants, sorted_matches),
    )
    freshness = _freshness_report(
        sorted_matches,
        fetched_at=current_fetched_at,
        stale_after=stale_after,
        differences=differences,
    )
    discrepancies: tuple[ResultDiscrepancy, ...] = ()
    if secondary_provider is not None:
        try:
            secondary = secondary_provider.fetch()
            secondary_matches, secondary_unresolved, secondary_invalid = _normalize_fixtures(
                secondary.fixtures,
                provider=secondary.provider,
                fetched_at=current_fetched_at,
                teams=teams,
                aliases=aliases,
            )
            unresolved = [*unresolved, *secondary_unresolved]
            invalid_fixtures.extend([*secondary.invalid_fixtures, *secondary_invalid])
            discrepancies = _validate_secondary_results(sorted_matches, secondary_matches)
        except WorldCupIngestError as exc:
            discrepancies = (
                ResultDiscrepancy(
                    kind="secondary_provider_unavailable",
                    primary_fixture_id=None,
                    secondary_fixture_id=None,
                    detail=_safe_provider_error(str(exc)),
                ),
            )

    operational_table = processed_root / "matches.parquet"
    if not dry_run:
        _write_raw_snapshots(response, manifests)
        _write_snapshot_tables(
            sorted_matches,
            pending_participants,
            processed_root=processed_root,
            fetched_at=current_fetched_at,
        )
        _write_operational_tables(sorted_matches, processed_root=processed_root)
        _write_json(
            interim_root / "world_cup_2026_unresolved_teams.json",
            [item.model_dump() for item in unresolved],
        )
        _write_json(
            interim_root / "world_cup_2026_tbd_fixtures.json",
            [item.model_dump(mode="json") for item in tbd_fixtures],
        )
        _write_json(
            interim_root / "world_cup_2026_partially_known_fixtures.json",
            [item.model_dump(mode="json") for item in partially_known_fixtures],
        )
        _write_json(
            interim_root / "world_cup_2026_invalid_fixtures.json",
            [item.model_dump() for item in invalid_fixtures],
        )
        _write_json(
            interim_root / "world_cup_2026_freshness_report.json",
            freshness.model_dump(mode="json"),
        )
        _write_json(
            interim_root / "world_cup_2026_result_discrepancies.json",
            [item.model_dump() for item in discrepancies],
        )

    counts = _status_counts(sorted_matches)
    invalid_ids = {
        item.source_fixture_id for item in normalization_invalid if item.source_fixture_id
    }
    partition_fixtures = [
        fixture for fixture in response.fixtures if fixture.source_fixture_id not in invalid_ids
    ]
    collection_checksum = sha256_bytes(response.payload)
    snapshot_reference = _snapshot_token(current_fetched_at, collection_checksum)
    report = WorldCupIngestReport(
        provider=response.provider,
        fetched_at=current_fetched_at,
        dry_run=dry_run,
        snapshot_manifests=tuple(manifests),
        provider_fixtures_received=len(response.fixtures) + len(response.invalid_fixtures),
        fixtures_with_known_participants=sum(
            item.participants_status is ParticipantStatus.KNOWN for item in partition_fixtures
        ),
        fixtures_with_partially_known_participants=sum(
            item.participants_status is ParticipantStatus.PARTIALLY_KNOWN
            for item in partition_fixtures
        ),
        fixtures_with_tbd_participants=len(tbd_fixtures),
        invalid_provider_fixtures=primary_invalid_provider_fixtures,
        invalid_fixtures=len(invalid_fixtures),
        invalid_fixture_records=tuple(invalid_fixtures),
        canonical_matches=len(sorted_matches),
        unresolved_named_teams=len(unresolved),
        pending_fixtures=counts["pending"],
        in_progress_matches=counts["in_progress"],
        finished_matches=counts["finished"],
        interrupted_matches=counts["interrupted"],
        source_status_counts=_source_status_counts(response.fixtures, response.invalid_fixtures),
        canonical_status_counts=counts,
        data_cutoff_utc=current_fetched_at,
        snapshot_checksum=collection_checksum,
        snapshot_reference=snapshot_reference,
        unresolved_teams=tuple(unresolved),
        freshness=freshness,
        validation_discrepancies=discrepancies,
        operational_table=None if dry_run else operational_table,
    )
    _assert_ingest_report_partition(report)
    if not dry_run:
        _write_json(
            interim_root / "world_cup_2026_ingest_report.json", report.model_dump(mode="json")
        )
    return WorldCupIngestResult(
        matches=sorted_matches,
        tbd_fixtures=tbd_fixtures,
        partially_known_fixtures=partially_known_fixtures,
        report=report,
    )


def offline_provider(path: Path) -> WorldCupProvider:
    """Load a Football-Data-compatible test fixture without network access or secrets."""

    return _OfflineFootballDataProvider(path)


class _OfflineFootballDataProvider:
    name = FootballDataProvider.name

    def __init__(self, path: Path) -> None:
        self.path = path

    def fetch(self) -> ProviderResponse:
        try:
            payload = self.path.read_bytes()
            raw = json.loads(payload)
            matches = raw["matches"] if isinstance(raw, dict) else raw
            fixtures, invalid, items = _parse_fixture_collection(matches, _football_data_fixture)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise WorldCupIngestError(f"offline fixture cannot be read: {self.path}") from exc
        return ProviderResponse(
            self.name,
            self.path.resolve().as_uri(),
            payload,
            fixtures,
            _fixture_payloads(fixtures, items),
            invalid,
        )


def _football_data_fixture(value: Mapping[str, object]) -> ProviderFixture:
    score = _mapping(value.get("score"))
    home = _required_mapping(value, "homeTeam")
    away = _required_mapping(value, "awayTeam")
    utc = _parse_datetime(_required_str(value, "utcDate"))
    source_status = _required_str(value, "status")
    home_name = _nullable_team_name(home, "name")
    away_name = _nullable_team_name(away, "name")
    duration = str(score.get("duration") or "REGULAR")
    full_time = _mapping(score.get("fullTime"))
    regular_time = _mapping(score.get("regularTime")) or full_time
    extra_time = _mapping(score.get("extraTime"))
    penalties = _mapping(score.get("penalties"))
    return ProviderFixture(
        source_fixture_id=_required_fixture_id(value),
        participants_status=_participant_status(home_name, away_name, source_status),
        home_name=home_name,
        away_name=away_name,
        kickoff_utc=utc,
        original_timezone="UTC",
        venue_name=_optional_str(value.get("venue")),
        city=_optional_str(value.get("city")),
        source_status=source_status,
        stage=_optional_str(value.get("stage")) or _optional_str(value.get("group")),
        source_updated_at_utc=_optional_datetime(value.get("lastUpdated")),
        home_goals_90=_optional_int(regular_time.get("home")),
        away_goals_90=_optional_int(regular_time.get("away")),
        extra_time_played=duration in {"EXTRA_TIME", "PENALTY_SHOOTOUT"},
        home_goals_after_extra_time=_optional_int(extra_time.get("home")),
        away_goals_after_extra_time=_optional_int(extra_time.get("away")),
        penalty_shootout=duration == "PENALTY_SHOOTOUT",
        home_penalty_goals=_optional_int(penalties.get("home")),
        away_penalty_goals=_optional_int(penalties.get("away")),
    )


def _api_football_fixture(value: Mapping[str, object]) -> ProviderFixture:
    fixture = _mapping(value.get("fixture"))
    teams = _mapping(value.get("teams"))
    score = _mapping(value.get("score"))
    league = _mapping(value.get("league"))
    home = _required_mapping(teams, "home")
    away = _required_mapping(teams, "away")
    extra_time = _mapping(score.get("extratime"))
    penalty = _mapping(score.get("penalty"))
    home_extra_time = _optional_int(extra_time.get("home"))
    away_extra_time = _optional_int(extra_time.get("away"))
    home_penalties = _optional_int(penalty.get("home"))
    away_penalties = _optional_int(penalty.get("away"))
    source_status = _required_str(_required_mapping(fixture, "status"), "short")
    home_name = _nullable_team_name(home, "name")
    away_name = _nullable_team_name(away, "name")
    return ProviderFixture(
        source_fixture_id=_required_fixture_id(fixture),
        participants_status=_participant_status(home_name, away_name, source_status),
        home_name=home_name,
        away_name=away_name,
        kickoff_utc=_parse_datetime(_required_str(fixture, "date")),
        original_timezone=str(fixture.get("timezone") or "UTC"),
        venue_name=_optional_str(_mapping(fixture.get("venue")).get("name")),
        city=_optional_str(_mapping(fixture.get("venue")).get("city")),
        source_status=source_status,
        stage=_optional_str(league.get("round")),
        source_updated_at_utc=None,
        home_goals_90=_optional_int(_mapping(score.get("fulltime")).get("home")),
        away_goals_90=_optional_int(_mapping(score.get("fulltime")).get("away")),
        extra_time_played=home_extra_time is not None and away_extra_time is not None,
        home_goals_after_extra_time=home_extra_time,
        away_goals_after_extra_time=away_extra_time,
        penalty_shootout=home_penalties is not None and away_penalties is not None,
        home_penalty_goals=home_penalties,
        away_penalty_goals=away_penalties,
    )


def _fixture_payloads(
    fixtures: Sequence[ProviderFixture], items: Sequence[Mapping[str, object]]
) -> dict[str, bytes]:
    """Preserve each exact fixture member as an immutable raw JSON fragment."""

    return {
        fixture.source_fixture_id: json.dumps(
            item,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        for fixture, item in zip(fixtures, items, strict=True)
    }


def _parse_fixture_collection(
    value: object, parser: Callable[[Mapping[str, object]], ProviderFixture]
) -> tuple[
    tuple[ProviderFixture, ...], tuple[InvalidProviderFixture, ...], list[Mapping[str, object]]
]:
    """Parse every collection member independently so one bad record cannot abort ingestion."""

    if not isinstance(value, list):
        raise ValueError("expected list of objects")
    fixtures: list[ProviderFixture] = []
    invalid: list[InvalidProviderFixture] = []
    items: list[Mapping[str, object]] = []
    seen_ids: set[str] = set()
    for value_item in value:
        if not isinstance(value_item, Mapping):
            invalid.append(
                InvalidProviderFixture(
                    source_fixture_id=None,
                    reason="fixture is not an object",
                )
            )
            continue
        item = value_item
        try:
            fixture = parser(item)
            if fixture.source_fixture_id in seen_ids:
                raise ValueError("duplicate source_fixture_id in provider collection")
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append(
                _invalid_fixture_record(item, reason=str(exc) or type(exc).__name__)
            )
            continue
        seen_ids.add(fixture.source_fixture_id)
        fixtures.append(fixture)
        items.append(item)
    return tuple(fixtures), tuple(invalid), items


def _invalid_fixture_record(item: Mapping[str, object], *, reason: str) -> InvalidProviderFixture:
    return InvalidProviderFixture(
        source_fixture_id=_optional_fixture_id(item),
        reason=reason,
        field=_invalid_field_from_reason(reason),
        source_status=_source_status_from_item(item),
    )


def _invalid_field_from_reason(reason: str) -> str | None:
    for prefix in ("missing ", "invalid "):
        if reason.startswith(prefix):
            return reason.removeprefix(prefix).split()[0]
    if "score" in reason:
        return "score"
    if "participants" in reason:
        return "homeTeam.name/awayTeam.name"
    return None


def _source_status_from_item(item: Mapping[str, object]) -> str | None:
    direct = _optional_str(item.get("status"))
    if direct is not None:
        return direct
    fixture = _mapping(item.get("fixture"))
    status = _mapping(fixture.get("status"))
    return _optional_str(status.get("short"))


def _validation_error_field(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "canonical_match"
    loc = errors[0].get("loc")
    if isinstance(loc, tuple) and loc:
        return str(loc[0])
    return "canonical_match"


def _normalize_fixtures(
    fixtures: Iterable[ProviderFixture],
    *,
    provider: str,
    fetched_at: datetime,
    teams: Sequence[CanonicalTeam],
    aliases: Sequence[TeamAlias],
) -> tuple[list[CanonicalMatch], list[UnresolvedLiveTeam], list[InvalidProviderFixture]]:
    canonical_names = {team.canonical_name: team.canonical_team_id for team in teams}
    matches: list[CanonicalMatch] = []
    unresolved: list[UnresolvedLiveTeam] = []
    invalid: list[InvalidProviderFixture] = []
    source = f"{PRIMARY_SOURCE_PREFIX}_{provider}"
    for fixture in fixtures:
        if fixture.participants_status is ParticipantStatus.TBD:
            continue
        if fixture.participants_status is ParticipantStatus.PARTIALLY_KNOWN:
            continue
        if fixture.home_name is None or fixture.away_name is None:
            invalid.append(
                InvalidProviderFixture(
                    source_fixture_id=fixture.source_fixture_id,
                    reason="known-participant fixture is missing a team name",
                    field="homeTeam.name/awayTeam.name",
                    source_status=fixture.source_status,
                )
            )
            continue
        match_date = fixture.kickoff_utc.date()
        home_id = _resolve_live_team(
            fixture.home_name,
            source=source,
            match_date=match_date,
            aliases=aliases,
            canonical_names=canonical_names,
        )
        away_id = _resolve_live_team(
            fixture.away_name,
            source=source,
            match_date=match_date,
            aliases=aliases,
            canonical_names=canonical_names,
        )
        if home_id is None:
            unresolved.append(
                UnresolvedLiveTeam(
                    provider=provider,
                    source_fixture_id=fixture.source_fixture_id,
                    side="home",
                    source_name=fixture.home_name,
                    match_date=match_date.isoformat(),
                )
            )
        if away_id is None:
            unresolved.append(
                UnresolvedLiveTeam(
                    provider=provider,
                    source_fixture_id=fixture.source_fixture_id,
                    side="away",
                    source_name=fixture.away_name,
                    match_date=match_date.isoformat(),
                )
            )
        if home_id is None or away_id is None:
            continue
        status = _canonical_status(fixture.source_status)
        try:
            matches.append(
                CanonicalMatch(
                    match_id=f"{source}:{fixture.source_fixture_id}",
                    match_status=status,
                    match_date=match_date,
                    kickoff_utc=fixture.kickoff_utc,
                    kickoff_local_time=None,
                    kickoff_timezone=fixture.original_timezone,
                    kickoff_time_status=KickoffTimeStatus.EXACT_UTC,
                    home_team_name_original=fixture.home_name,
                    away_team_name_original=fixture.away_name,
                    home_team_id=home_id,
                    away_team_id=away_id,
                    home_goals_90=_score_if_present(status, fixture.home_goals_90),
                    away_goals_90=_score_if_present(status, fixture.away_goals_90),
                    result_90=_result_90(status, fixture.home_goals_90, fixture.away_goals_90),
                    extra_time_played=status is MatchStatus.PLAYED and fixture.extra_time_played,
                    home_goals_after_extra_time=_extra_score(
                        status, fixture.home_goals_after_extra_time
                    ),
                    away_goals_after_extra_time=_extra_score(
                        status, fixture.away_goals_after_extra_time
                    ),
                    penalty_shootout=status is MatchStatus.PLAYED and fixture.penalty_shootout,
                    home_penalty_goals=_extra_score(status, fixture.home_penalty_goals),
                    away_penalty_goals=_extra_score(status, fixture.away_penalty_goals),
                    competition="FIFA World Cup",
                    stage=fixture.stage,
                    match_type=MatchType.WORLD_CUP,
                    city=fixture.city,
                    host_country=None,
                    venue_name_original=fixture.venue_name,
                    neutral_site=None,
                    home_advantage_status=HomeAdvantageStatus.UNKNOWN,
                    source=source,
                    source_match_id=fixture.source_fixture_id,
                    retrieved_at_utc=fetched_at,
                    source_updated_at_utc=fixture.source_updated_at_utc,
                    data_cutoff_utc=fetched_at,
                )
            )
        except ValidationError as exc:
            invalid.append(
                InvalidProviderFixture(
                    source_fixture_id=fixture.source_fixture_id,
                    reason=f"canonical score contract: {exc.errors()[0]['msg']}",
                    field=_validation_error_field(exc),
                    source_status=fixture.source_status,
                )
            )
    return matches, unresolved, invalid


def _resolve_live_team(
    source_name: str,
    *,
    source: str,
    match_date: date,
    aliases: Sequence[TeamAlias],
    canonical_names: Mapping[str, str],
) -> str | None:
    try:
        resolved = resolve_team_alias(
            source=source,
            source_name=source_name,
            match_date=match_date,
            aliases=aliases,
        )
    except ValueError:
        resolved = None
    if resolved is not None:
        return resolved
    return canonical_names.get(source_name)


def _canonical_status(source_status: str) -> MatchStatus:
    normalized = source_status.upper().replace("-", "_")
    if normalized in {"FINISHED", "FT", "AET", "PEN", "AWD", "WO"}:
        return MatchStatus.PLAYED
    if normalized == "AWARDED":
        return MatchStatus.PLAYED
    if normalized in {"IN_PLAY", "PAUSED", "LIVE", "1H", "2H", "HT", "ET", "P", "BT"}:
        return MatchStatus.IN_PROGRESS
    if normalized in {"POSTPONED", "PST"}:
        return MatchStatus.POSTPONED
    if normalized in {"CANCELLED", "CANCELED", "CAN"}:
        return MatchStatus.CANCELLED
    if normalized in {"SUSPENDED", "INT"}:
        return MatchStatus.SUSPENDED
    if normalized in {"ABANDONED", "ABD"}:
        return MatchStatus.ABANDONED
    return MatchStatus.SCHEDULED


def _score_if_present(status: MatchStatus, score: int | None) -> int | None:
    return score if status in {MatchStatus.PLAYED, MatchStatus.IN_PROGRESS} else None


def _extra_score(status: MatchStatus, score: int | None) -> int | None:
    return score if status is MatchStatus.PLAYED else None


def _result_90(status: MatchStatus, home: int | None, away: int | None) -> Result90 | None:
    if status is not MatchStatus.PLAYED or home is None or away is None:
        return None
    if home > away:
        return Result90.HOME_WIN
    if home < away:
        return Result90.AWAY_WIN
    return Result90.DRAW


def _build_manifests(
    response: ProviderResponse, *, fetched_at: datetime, raw_root: Path
) -> list[LiveSnapshotManifest]:
    token = _snapshot_token(fetched_at, sha256_bytes(response.payload))
    directory = raw_root / PRIMARY_SOURCE_PREFIX / response.provider / token
    manifests = [
        LiveSnapshotManifest(
            provider=response.provider,
            endpoint=response.endpoint,
            source_fixture_id="__collection__",
            fetched_at=fetched_at,
            checksum=sha256_bytes(response.payload),
            raw_path=directory / "response.json",
            response_scope="collection",
        )
    ]
    for fixture in response.fixtures:
        fixture_payload = response.fixture_payloads[fixture.source_fixture_id]
        manifests.append(
            LiveSnapshotManifest(
                provider=response.provider,
                endpoint=response.endpoint,
                source_fixture_id=fixture.source_fixture_id,
                fetched_at=fetched_at,
                checksum=sha256_bytes(fixture_payload),
                raw_path=directory / "fixtures" / f"{fixture.source_fixture_id}.json",
                response_scope="fixture_member",
            )
        )
    return manifests


def _pending_participant_fixtures(
    fixtures: Iterable[ProviderFixture], manifests: Iterable[LiveSnapshotManifest]
) -> list[PendingParticipantFixture]:
    """Build explicit participant-pending views without attempting alias resolution."""

    raw_paths = {
        manifest.source_fixture_id: manifest.raw_path
        for manifest in manifests
        if manifest.response_scope == "fixture_member"
    }
    pending: list[PendingParticipantFixture] = []
    for fixture in fixtures:
        if fixture.participants_status not in {
            ParticipantStatus.TBD,
            ParticipantStatus.PARTIALLY_KNOWN,
        }:
            continue
        pending.append(
            PendingParticipantFixture(
                source_fixture_id=fixture.source_fixture_id,
                participants_status=fixture.participants_status,
                home_name=fixture.home_name,
                away_name=fixture.away_name,
                kickoff_utc=fixture.kickoff_utc,
                source_status=fixture.source_status,
                stage=fixture.stage,
                source_updated_at_utc=fixture.source_updated_at_utc,
                score=_pending_score(fixture),
                raw_snapshot_path=raw_paths[fixture.source_fixture_id],
            )
        )
    return pending


def _write_raw_snapshots(
    response: ProviderResponse, manifests: Sequence[LiveSnapshotManifest]
) -> None:
    for manifest in manifests:
        content = (
            response.payload
            if manifest.response_scope == "collection"
            else response.fixture_payloads[manifest.source_fixture_id]
        )
        _write_immutable_bytes(manifest.raw_path, content)
        manifest_path = manifest.raw_path.with_suffix(".manifest.json")
        _write_immutable_bytes(
            manifest_path,
            (json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )


def _write_immutable_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise WorldCupIngestError(f"immutable snapshot collision at {path}")
        return
    path.write_bytes(content)


def _write_snapshot_tables(
    matches: Sequence[CanonicalMatch],
    pending_participants: Sequence[PendingParticipantFixture],
    *,
    processed_root: Path,
    fetched_at: datetime,
) -> None:
    checksum = sha256_bytes(
        (_canonical_json(matches) + _pending_json(pending_participants)).encode("utf-8")
    )
    directory = processed_root / "snapshots" / _snapshot_token(fetched_at, checksum)
    path = directory / "matches.parquet"
    _write_immutable_parquet(matches, path)
    _write_immutable_json(
        directory / "pending_participant_fixtures.json",
        [item.model_dump(mode="json") for item in pending_participants],
    )
    _write_immutable_json(
        directory / "tbd_fixtures.json",
        [
            item.model_dump(mode="json")
            for item in pending_participants
            if item.participants_status is ParticipantStatus.TBD
        ],
    )
    _write_immutable_json(
        directory / "partially_known_fixtures.json",
        [
            item.model_dump(mode="json")
            for item in pending_participants
            if item.participants_status is ParticipantStatus.PARTIALLY_KNOWN
        ],
    )


def _write_operational_tables(matches: Sequence[CanonicalMatch], *, processed_root: Path) -> None:
    source_ids = [item.source_match_id for item in matches]
    duplicates = sorted({source_id for source_id in source_ids if source_ids.count(source_id) > 1})
    if duplicates:
        msg = "operational World Cup view would contain duplicate source_fixture_id values: "
        raise WorldCupIngestError(msg + ", ".join(duplicates[:5]))
    _write_parquet(matches, processed_root / "matches.parquet")
    groups = {
        "pending": [item for item in matches if item.match_status is MatchStatus.SCHEDULED],
        "in_progress": [item for item in matches if item.match_status is MatchStatus.IN_PROGRESS],
        "finished": [item for item in matches if item.match_status is MatchStatus.PLAYED],
        "interrupted": [
            item
            for item in matches
            if item.match_status
            in {
                MatchStatus.POSTPONED,
                MatchStatus.CANCELLED,
                MatchStatus.SUSPENDED,
                MatchStatus.ABANDONED,
            }
        ],
    }
    for name, rows in groups.items():
        _write_parquet(rows, processed_root / f"{name}.parquet")


def _write_immutable_parquet(matches: Sequence[CanonicalMatch], path: Path) -> None:
    if path.exists():
        return
    _write_parquet(matches, path)


def _write_immutable_json(path: Path, content: object) -> None:
    encoded = (
        json.dumps(content, indent=2, sort_keys=True, default=_json_value) + "\n"
    ).encode("utf-8")
    _write_immutable_bytes(path, encoded)


def _write_parquet(matches: Sequence[CanonicalMatch], path: Path) -> None:
    rows = [item.model_dump(mode="python") for item in matches]
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        table = pa.Table.from_pylist(rows)
    else:
        table = pa.table({"match_id": pa.array([], type=pa.string())})
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


def _read_latest_snapshot_matches(processed_root: Path) -> tuple[CanonicalMatch, ...]:
    snapshots = processed_root / "snapshots"
    if not snapshots.is_dir():
        return ()
    candidates = sorted(snapshots.glob("*/matches.parquet"))
    if not candidates:
        return ()
    try:
        rows = pq.read_table(candidates[-1]).to_pylist()  # type: ignore[no-untyped-call]
        return tuple(CanonicalMatch.model_validate(row) for row in rows)
    except (OSError, pa.ArrowInvalid, ValidationError) as exc:
        raise WorldCupIngestError("cannot read prior World Cup canonical snapshot") from exc


def _read_latest_snapshot_pending_participant_fixtures(
    processed_root: Path,
) -> tuple[PendingParticipantFixture, ...]:
    snapshots = processed_root / "snapshots"
    if not snapshots.is_dir():
        return ()
    candidates = sorted(snapshots.glob("*/pending_participant_fixtures.json"))
    if not candidates:
        candidates = sorted(snapshots.glob("*/tbd_fixtures.json"))
    if not candidates:
        return ()
    try:
        rows = json.loads(candidates[-1].read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("participant-pending snapshot must be a list")
        return tuple(PendingParticipantFixture.model_validate(row) for row in rows)
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise WorldCupIngestError("cannot read prior World Cup participant snapshot") from exc


def _snapshot_differences(
    previous: Sequence[CanonicalMatch], current: Sequence[CanonicalMatch]
) -> tuple[SnapshotDifference, ...]:
    previous_by_source = {item.source_match_id: item for item in previous}
    differences: list[SnapshotDifference] = []
    fields = (
        "match_status",
        "kickoff_utc",
        "home_team_id",
        "away_team_id",
        "home_goals_90",
        "away_goals_90",
        "home_goals_after_extra_time",
        "away_goals_after_extra_time",
        "home_penalty_goals",
        "away_penalty_goals",
    )
    for item in current:
        old = previous_by_source.get(item.source_match_id)
        if old is None:
            differences.append(
                SnapshotDifference(
                    source_fixture_id=item.source_match_id,
                    field="fixture",
                    previous=None,
                    current="new",
                )
            )
            continue
        for field in fields:
            before = getattr(old, field)
            after = getattr(item, field)
            if before != after:
                differences.append(
                    SnapshotDifference(
                        source_fixture_id=item.source_match_id,
                        field=field,
                        previous=_json_value(before),
                        current=_json_value(after),
                    )
                )
    return tuple(differences)


def _participant_status_differences(
    previous_pending: Sequence[PendingParticipantFixture],
    current_pending: Sequence[PendingParticipantFixture],
    current_matches: Sequence[CanonicalMatch],
) -> tuple[SnapshotDifference, ...]:
    """Record participant-status transitions while canonical snapshots stay team-complete."""

    current_pending_by_id = {item.source_fixture_id: item for item in current_pending}
    current_known_ids = {item.source_match_id for item in current_matches}
    differences: list[SnapshotDifference] = []
    for item in previous_pending:
        current_item = current_pending_by_id.get(item.source_fixture_id)
        if current_item is not None:
            if current_item.participants_status != item.participants_status:
                differences.append(
                    SnapshotDifference(
                        source_fixture_id=item.source_fixture_id,
                        field="participants_status",
                        previous=item.participants_status.value,
                        current=current_item.participants_status.value,
                    )
                )
            continue
        if item.source_fixture_id in current_known_ids:
            differences.append(
                SnapshotDifference(
                    source_fixture_id=item.source_fixture_id,
                    field="participants_status",
                    previous=item.participants_status.value,
                    current=ParticipantStatus.KNOWN.value,
                )
            )
    return tuple(differences)


def _freshness_report(
    matches: Sequence[CanonicalMatch],
    *,
    fetched_at: datetime,
    stale_after: timedelta,
    differences: tuple[SnapshotDifference, ...],
) -> FreshnessReport:
    future = [
        item.kickoff_utc
        for item in matches
        if item.match_status is MatchStatus.SCHEDULED
        and item.kickoff_utc is not None
        and item.kickoff_utc > fetched_at
    ]
    stale = tuple(
        item.source_match_id
        for item in matches
        if item.match_status in {MatchStatus.SCHEDULED, MatchStatus.IN_PROGRESS}
        and (
            item.source_updated_at_utc is None
            or fetched_at - item.source_updated_at_utc > stale_after
        )
    )
    return FreshnessReport(
        last_fetch_utc=fetched_at,
        next_kickoff_utc=min(future) if future else None,
        fixtures_without_recent_update=stale,
        differences_from_previous_snapshot=differences,
    )


def _validate_secondary_results(
    primary: Sequence[CanonicalMatch], secondary: Sequence[CanonicalMatch]
) -> tuple[ResultDiscrepancy, ...]:
    secondary_by_key = {_fixture_identity(item): item for item in secondary}
    findings: list[ResultDiscrepancy] = []
    for first in primary:
        second = secondary_by_key.get(_fixture_identity(first))
        if second is None:
            findings.append(
                ResultDiscrepancy(
                    kind="missing_secondary_fixture",
                    primary_fixture_id=first.source_match_id,
                    secondary_fixture_id=None,
                    detail="No exact canonical-team and kickoff-date match in validation source",
                )
            )
            continue
        for field in (
            "home_team_id",
            "away_team_id",
            "kickoff_utc",
            "home_goals_90",
            "away_goals_90",
            "home_goals_after_extra_time",
            "away_goals_after_extra_time",
            "home_penalty_goals",
            "away_penalty_goals",
        ):
            if getattr(first, field) != getattr(second, field):
                detail = (
                    f"primary={_json_value(getattr(first, field))!r}; "
                    f"secondary={_json_value(getattr(second, field))!r}"
                )
                findings.append(
                    ResultDiscrepancy(
                        kind=field,
                        primary_fixture_id=first.source_match_id,
                        secondary_fixture_id=second.source_match_id,
                        detail=detail,
                    )
                )
    return tuple(findings)


def _fixture_identity(match: CanonicalMatch) -> tuple[str, str, str]:
    if match.kickoff_utc is None:
        raise WorldCupIngestError("live canonical fixtures require kickoff_utc")
    return (match.home_team_id, match.away_team_id, match.kickoff_utc.date().isoformat())


def _status_counts(matches: Sequence[CanonicalMatch]) -> dict[str, int]:
    return {
        "pending": sum(item.match_status is MatchStatus.SCHEDULED for item in matches),
        "in_progress": sum(item.match_status is MatchStatus.IN_PROGRESS for item in matches),
        "finished": sum(item.match_status is MatchStatus.PLAYED for item in matches),
        "interrupted": sum(
            item.match_status
            in {
                MatchStatus.POSTPONED,
                MatchStatus.CANCELLED,
                MatchStatus.SUSPENDED,
                MatchStatus.ABANDONED,
            }
            for item in matches
        ),
    }


def _source_status_counts(
    fixtures: Sequence[ProviderFixture],
    invalid_fixtures: Sequence[InvalidProviderFixture],
) -> dict[str, int]:
    counts: Counter[str] = Counter(fixture.source_status for fixture in fixtures)
    for item in invalid_fixtures:
        counts[item.source_status or "invalid_unknown_status"] += 1
    return dict(sorted(counts.items()))


def _assert_ingest_report_partition(report: WorldCupIngestReport) -> None:
    observed = (
        report.fixtures_with_known_participants
        + report.fixtures_with_partially_known_participants
        + report.fixtures_with_tbd_participants
        + report.invalid_provider_fixtures
    )
    if observed != report.provider_fixtures_received:
        msg = (
            "World Cup ingest report partition is inconsistent: "
            f"{observed} != {report.provider_fixtures_received}"
        )
        raise WorldCupIngestError(msg)


def _canonical_json(matches: Sequence[CanonicalMatch]) -> str:
    return json.dumps(
        [item.model_dump(mode="json") for item in matches], sort_keys=True, separators=(",", ":")
    )


def _pending_json(fixtures: Sequence[PendingParticipantFixture]) -> str:
    return json.dumps(
        [item.model_dump(mode="json") for item in fixtures], sort_keys=True, separators=(",", ":")
    )


def _write_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content, indent=2, sort_keys=True, default=_json_value) + "\n", encoding="utf-8"
    )


def _snapshot_token(fetched_at: datetime, checksum: str) -> str:
    return f"{fetched_at.strftime('%Y%m%dT%H%M%SZ')}_{checksum[:12]}"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise WorldCupIngestError("fetched_at must be timezone-aware UTC")
    return value.astimezone(UTC)


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _require_utc(parsed.astimezone(UTC))


def _optional_datetime(value: object) -> datetime | None:
    return _parse_datetime(value) if isinstance(value, str) and value else None


def _as_mapping_sequence(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError("expected list of objects")
    return list(value)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _required_mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"missing {key}")
    return result


def _required_str(value: Mapping[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"missing {key}")
    return result


def _nullable_team_name(value: Mapping[str, object], key: str) -> str | None:
    """Return an explicitly null provider team name without accepting malformed names."""

    if key not in value:
        raise ValueError(f"missing {key}")
    result = value[key]
    if result is None:
        return None
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"missing {key}")
    return result.strip()


def _required_fixture_id(value: Mapping[str, object]) -> str:
    result = value.get("id")
    if isinstance(result, bool) or result is None:
        raise ValueError("missing id")
    if isinstance(result, (int, str)) and str(result).strip():
        return str(result)
    raise ValueError("missing id")


def _optional_fixture_id(value: Mapping[str, object]) -> str | None:
    try:
        return _required_fixture_id(value)
    except ValueError:
        return None


def _participant_status(
    home_name: str | None, away_name: str | None, source_status: str
) -> ParticipantStatus:
    if home_name is not None and away_name is not None:
        return ParticipantStatus.KNOWN
    canonical_status = _canonical_status(source_status)
    if canonical_status not in {
        MatchStatus.SCHEDULED,
        MatchStatus.POSTPONED,
        MatchStatus.CANCELLED,
        MatchStatus.SUSPENDED,
    }:
        raise ValueError("participants are required once a fixture is active or final")
    if home_name is None and away_name is None:
        return ParticipantStatus.TBD
    return ParticipantStatus.PARTIALLY_KNOWN


def _pending_score(fixture: ProviderFixture) -> dict[str, int | bool | None]:
    """Preserve provider score fields for participant-pending fixtures."""

    return {
        "home_goals_90": fixture.home_goals_90,
        "away_goals_90": fixture.away_goals_90,
        "extra_time_played": fixture.extra_time_played,
        "home_goals_after_extra_time": fixture.home_goals_after_extra_time,
        "away_goals_after_extra_time": fixture.away_goals_after_extra_time,
        "penalty_shootout": fixture.penalty_shootout,
        "home_penalty_goals": fixture.home_penalty_goals,
        "away_penalty_goals": fixture.away_penalty_goals,
    }


def _optional_str(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean score is invalid")
    if not isinstance(value, (int, str)):
        raise ValueError("score must be an integer")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("score must be an integer") from exc


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    return value.value if hasattr(value, "value") else value
