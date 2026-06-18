"""Data contracts, validation, and normalization helpers."""

from worldcup2026.data.contracts import (
    MATCH_SCHEMA_VERSION,
    CanonicalMatch,
    HomeAdvantageStatus,
    KickoffTimeStatus,
    MatchStatus,
    MatchType,
    Result90,
    TeamAlias,
    resolve_team_alias,
    validate_match_records,
)
from worldcup2026.data.sources import (
    RAW_SNAPSHOT_MANIFEST_VERSION,
    HistoricalDataClient,
    HistoricalFetchRequest,
    HistoricalMatchAdapter,
    HistoricalSourceCandidate,
    HistoricalSourceRecord,
    RawSnapshot,
    RawSnapshotManifest,
    sha256_bytes,
)

__all__ = [
    "MATCH_SCHEMA_VERSION",
    "RAW_SNAPSHOT_MANIFEST_VERSION",
    "CanonicalMatch",
    "HistoricalDataClient",
    "HistoricalFetchRequest",
    "HistoricalMatchAdapter",
    "HistoricalSourceCandidate",
    "HistoricalSourceRecord",
    "HomeAdvantageStatus",
    "KickoffTimeStatus",
    "MatchStatus",
    "MatchType",
    "RawSnapshot",
    "RawSnapshotManifest",
    "Result90",
    "TeamAlias",
    "resolve_team_alias",
    "sha256_bytes",
    "validate_match_records",
]
