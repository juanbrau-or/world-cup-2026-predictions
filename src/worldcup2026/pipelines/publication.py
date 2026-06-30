"""Prepare small prediction outputs for the public data branch."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

PUBLICATION_SCHEMA_VERSION = "prediction_publication_manifest_v1"
LATEST_JSON_SCHEMA_VERSION = "predictions_latest_v1"
PUBLICATION_VERSION = "publication_v1"
MAX_BRANCH_FILE_BYTES = 2_000_000
SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(football_data_api_key|api_football_key|api[-_ ]?key|authorization|bearer|token)"
    r"\b\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{16,}"
)
DATA_CUTOFF_PATTERN = re.compile(r"^Data cutoff UTC:\s*(?P<value>\S+)\s*$", re.MULTILINE)
MODEL_PATTERN = re.compile(
    r"^Model:\s*(?P<family>[^(\n]+)\((?P<version>[^)\n]+)\)\s*$",
    re.MULTILINE,
)


class PublicationError(RuntimeError):
    """Raised when prediction outputs are unsafe to publish."""


@dataclass(frozen=True)
class PublicationResult:
    """Summary of prepared publication files."""

    output_root: Path
    changed: bool
    manifest_path: Path
    latest_csv_path: Path
    latest_json_path: Path
    upcoming_report_path: Path
    prospective_evaluation_json_path: Path
    prospective_evaluation_report_path: Path
    history_path: Path | None
    prediction_count: int
    prospective_evaluation_observations: int
    data_cutoff: str
    checksum: str


def prepare_predictions_publication(
    *,
    predictions_root: Path = Path("predictions"),
    output_root: Path = Path("dist/predictions-data"),
    generated_at: datetime | None = None,
    secret_values: Sequence[str] = (),
) -> PublicationResult:
    """Copy only branch-safe prediction outputs into ``output_root``."""

    generated = _utc_now() if generated_at is None else _require_utc(generated_at)
    source = _read_source_outputs(predictions_root)
    rows = _read_prediction_rows(source.latest_csv_bytes)
    metadata = _publication_metadata(rows, source.upcoming_report)
    evaluation_count = _prospective_evaluation_count(source.prospective_evaluation_json)
    latest_checksum = _sha256(source.latest_csv_bytes)
    source_fingerprint = _source_fingerprint(source)
    history_path = _history_path(
        output_root,
        data_cutoff=metadata["data_cutoff"],
        checksum=latest_checksum,
        prediction_count=len(rows),
    )
    history_bytes = _history_bytes(source.latest_csv_bytes) if history_path is not None else None

    output_root.mkdir(parents=True, exist_ok=True)
    assert_allowed_publication_tree(output_root)
    if history_path is not None and history_path.exists():
        assert history_bytes is not None
        if history_path.read_bytes() != history_bytes:
            raise PublicationError(f"immutable history collision at {history_path}")

    previous_manifest = _read_existing_manifest(output_root / "manifest.json")
    if (
        previous_manifest is not None
        and previous_manifest.get("publication_fingerprint") == source_fingerprint
        and _required_outputs_match(
            output_root,
            history_path=history_path,
            manifest=previous_manifest,
        )
    ):
        assert_no_secrets(output_root, secret_values=secret_values)
        return PublicationResult(
            output_root=output_root,
            changed=False,
            manifest_path=output_root / "manifest.json",
            latest_csv_path=output_root / "latest.csv",
            latest_json_path=output_root / "latest.json",
            upcoming_report_path=output_root / "upcoming.md",
            prospective_evaluation_json_path=output_root / "prospective_evaluation.json",
            prospective_evaluation_report_path=output_root / "prospective_evaluation.md",
            history_path=history_path,
            prediction_count=len(rows),
            prospective_evaluation_observations=evaluation_count,
            data_cutoff=str(metadata["data_cutoff"]),
            checksum=latest_checksum,
        )

    latest_json_bytes = _latest_json_bytes(
        rows,
        generated_at=generated,
        data_cutoff=str(metadata["data_cutoff"]),
        model_family=str(metadata["model_family"]),
        model_version=str(metadata["model_version"]),
        checksum=latest_checksum,
    )
    files: dict[str, bytes] = {
        "latest.csv": source.latest_csv_bytes,
        "latest.json": latest_json_bytes,
        "upcoming.md": source.upcoming_report_bytes,
        "prospective_evaluation.json": source.prospective_evaluation_json_bytes,
        "prospective_evaluation.md": source.prospective_evaluation_report_bytes,
    }
    if history_path is not None:
        assert history_bytes is not None
        files[str(history_path.relative_to(output_root))] = history_bytes

    checksums = {relative_path: _sha256(content) for relative_path, content in files.items()}
    manifest = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "version": PUBLICATION_VERSION,
        "generated_at": _format_utc(generated),
        "data_cutoff": str(metadata["data_cutoff"]),
        "model": {
            "family": str(metadata["model_family"]),
            "version": str(metadata["model_version"]),
        },
        "checksum": latest_checksum,
        "checksums": checksums,
        "prediction_count": len(rows),
        "prospective_evaluation_observations": evaluation_count,
        "history_path": str(history_path.relative_to(output_root)) if history_path else None,
        "publication_fingerprint": source_fingerprint,
        "published_files": sorted(files),
    }
    files["manifest.json"] = _json_bytes(manifest)

    _assert_no_secret_text(files, secret_values=_combined_secret_values(secret_values))
    for relative_path, content in files.items():
        assert_allowed_publication_path(Path(relative_path), size_bytes=len(content))
        path = output_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() == content:
            continue
        path.write_bytes(content)

    assert_allowed_publication_tree(output_root)
    assert_no_secrets(output_root, secret_values=secret_values)
    changed = _outputs_differ(previous_manifest, manifest)
    return PublicationResult(
        output_root=output_root,
        changed=changed,
        manifest_path=output_root / "manifest.json",
        latest_csv_path=output_root / "latest.csv",
        latest_json_path=output_root / "latest.json",
        upcoming_report_path=output_root / "upcoming.md",
        prospective_evaluation_json_path=output_root / "prospective_evaluation.json",
        prospective_evaluation_report_path=output_root / "prospective_evaluation.md",
        history_path=history_path,
        prediction_count=len(rows),
        prospective_evaluation_observations=evaluation_count,
        data_cutoff=str(metadata["data_cutoff"]),
        checksum=latest_checksum,
    )


def assert_allowed_publication_tree(root: Path) -> None:
    """Validate that a data-branch worktree contains only allowed small files."""

    if not root.exists():
        return
    for path in root.rglob("*"):
        if ".git" in path.relative_to(root).parts:
            continue
        if not path.is_file():
            continue
        relative_path = path.relative_to(root)
        assert_allowed_publication_path(relative_path, size_bytes=path.stat().st_size)


def assert_allowed_publication_path(relative_path: Path, *, size_bytes: int) -> None:
    """Validate one relative path intended for the public data branch."""

    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise PublicationError(f"publication path must be relative and contained: {relative_path}")
    normalized = relative_path.as_posix()
    if size_bytes > MAX_BRANCH_FILE_BYTES:
        raise PublicationError(f"publication file is too large for data branch: {relative_path}")
    if any(part in {"raw", "models"} for part in relative_path.parts):
        raise PublicationError(
            f"raw snapshots and models are not allowed on data branch: {relative_path}"
        )
    if relative_path.suffix == ".parquet" or ".parquet" in relative_path.suffixes:
        raise PublicationError(f"Parquet is not allowed on data branch: {relative_path}")

    exact = {
        "latest.csv",
        "latest.json",
        "upcoming.md",
        "prospective_evaluation.json",
        "prospective_evaluation.md",
        "manifest.json",
    }
    if normalized in exact:
        return
    if (
        len(relative_path.parts) == 2
        and relative_path.parts[0] == "history"
        and relative_path.name.endswith(".csv.gz")
    ):
        return
    raise PublicationError(f"path is not allowed on data branch: {relative_path}")


def assert_no_secrets(root: Path, *, secret_values: Sequence[str] = ()) -> None:
    """Scan publishable files for configured secret values and key-like assignments."""

    values = _combined_secret_values(secret_values)
    if not root.exists():
        return
    for path in root.rglob("*"):
        if ".git" in path.relative_to(root).parts:
            continue
        if not path.is_file():
            continue
        relative_path = path.relative_to(root)
        assert_allowed_publication_path(relative_path, size_bytes=path.stat().st_size)
        text = _file_text_for_secret_scan(path)
        _assert_no_secret_string(relative_path.as_posix(), text, secret_values=values)


@dataclass(frozen=True)
class _SourceOutputs:
    latest_csv_bytes: bytes
    upcoming_report_bytes: bytes
    prospective_evaluation_json_bytes: bytes
    prospective_evaluation_report_bytes: bytes

    @property
    def upcoming_report(self) -> str:
        return self.upcoming_report_bytes.decode("utf-8")

    @property
    def prospective_evaluation_json(self) -> Mapping[str, Any]:
        payload = json.loads(self.prospective_evaluation_json_bytes.decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise PublicationError("prospective evaluation JSON must contain an object")
        return payload


def _read_source_outputs(predictions_root: Path) -> _SourceOutputs:
    files = {
        "latest_csv_bytes": predictions_root / "latest.csv",
        "upcoming_report_bytes": predictions_root / "upcoming.md",
        "prospective_evaluation_json_bytes": predictions_root / "prospective_evaluation.json",
        "prospective_evaluation_report_bytes": predictions_root / "prospective_evaluation.md",
    }
    values: dict[str, bytes] = {}
    for key, path in files.items():
        if not path.is_file():
            raise PublicationError(f"required prediction output is missing: {path}")
        _reject_source_path(path)
        values[key] = path.read_bytes()
    return _SourceOutputs(**values)


def _reject_source_path(path: Path) -> None:
    parts = set(path.parts)
    if "raw" in parts:
        raise PublicationError(f"raw provider snapshots cannot be published: {path}")
    if path.suffix == ".parquet" or ".parquet" in path.suffixes:
        raise PublicationError(
            f"Parquet outputs must use Actions artifacts, not data branch: {path}"
        )


def _read_prediction_rows(content: bytes) -> list[dict[str, str]]:
    text = content.decode("utf-8")
    reader = csv.DictReader(StringIO(text))
    if reader.fieldnames is None:
        raise PublicationError("latest.csv must include a header")
    required = {
        "prediction_id",
        "source_fixture_id",
        "data_cutoff_utc",
        "kickoff_utc",
        "probability_home_win",
        "probability_draw",
        "probability_away_win",
        "model_family",
        "model_version",
    }
    missing = sorted(required.difference(reader.fieldnames))
    if missing:
        raise PublicationError(f"latest.csv is missing required columns: {', '.join(missing)}")
    rows = [dict(row) for row in reader]
    _assert_probabilities(rows)
    return rows


def _assert_probabilities(rows: Sequence[Mapping[str, str]]) -> None:
    for row in rows:
        fixture_id = row.get("source_fixture_id") or "<unknown>"
        values = [
            _finite_probability(row.get("probability_home_win"), fixture_id=fixture_id),
            _finite_probability(row.get("probability_draw"), fixture_id=fixture_id),
            _finite_probability(row.get("probability_away_win"), fixture_id=fixture_id),
        ]
        if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise PublicationError(f"prediction probabilities do not sum to 1 for {fixture_id}")


def _finite_probability(value: str | None, *, fixture_id: str) -> float:
    try:
        parsed = float(value or "")
    except ValueError as exc:
        raise PublicationError(f"invalid probability for {fixture_id}") from exc
    if not math.isfinite(parsed) or parsed < 0 or parsed > 1:
        raise PublicationError(f"invalid probability for {fixture_id}")
    return parsed


def _publication_metadata(
    rows: Sequence[Mapping[str, str]],
    upcoming_report: str,
) -> dict[str, str]:
    if rows:
        data_cutoffs = {_format_utc(_parse_utc(row["data_cutoff_utc"])) for row in rows}
        model_families = {row["model_family"] for row in rows}
        model_versions = {row["model_version"] for row in rows}
        for row in rows:
            _parse_utc(row["kickoff_utc"])
        if len(data_cutoffs) != 1:
            raise PublicationError("latest.csv contains multiple data cutoffs")
        if len(model_families) != 1 or len(model_versions) != 1:
            raise PublicationError("latest.csv contains multiple model identifiers")
        return {
            "data_cutoff": next(iter(data_cutoffs)),
            "model_family": next(iter(model_families)),
            "model_version": next(iter(model_versions)),
        }

    cutoff_match = DATA_CUTOFF_PATTERN.search(upcoming_report)
    model_match = MODEL_PATTERN.search(upcoming_report)
    if cutoff_match is None:
        raise PublicationError("upcoming.md must include Data cutoff UTC when there are no rows")
    if model_match is None:
        raise PublicationError("upcoming.md must include Model when there are no rows")
    return {
        "data_cutoff": _format_utc(_parse_utc(cutoff_match.group("value"))),
        "model_family": model_match.group("family").strip(),
        "model_version": model_match.group("version").strip(),
    }


def _prospective_evaluation_count(payload: Mapping[str, Any]) -> int:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise PublicationError("prospective evaluation JSON must include metrics")
    predictions = metrics.get("predictions")
    if not isinstance(predictions, int) or isinstance(predictions, bool) or predictions < 0:
        raise PublicationError(
            "prospective evaluation predictions count must be a non-negative integer"
        )
    return predictions


def _history_path(
    output_root: Path,
    *,
    data_cutoff: str,
    checksum: str,
    prediction_count: int,
) -> Path | None:
    if prediction_count == 0:
        return None
    cutoff = _parse_utc(data_cutoff)
    token = cutoff.strftime("%Y%m%dT%H%M%SZ")
    return output_root / "history" / f"{token}_{checksum[:12]}.csv.gz"


def _history_bytes(content: bytes) -> bytes:
    return gzip.compress(content, compresslevel=9, mtime=0)


def _latest_json_bytes(
    rows: Sequence[Mapping[str, str]],
    *,
    generated_at: datetime,
    data_cutoff: str,
    model_family: str,
    model_version: str,
    checksum: str,
) -> bytes:
    payload = {
        "schema_version": LATEST_JSON_SCHEMA_VERSION,
        "generated_at": _format_utc(generated_at),
        "data_cutoff": data_cutoff,
        "model": {"family": model_family, "version": model_version},
        "checksum": checksum,
        "prediction_count": len(rows),
        "predictions": [dict(row) for row in rows],
    }
    return _json_bytes(payload)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _source_fingerprint(source: _SourceOutputs) -> str:
    digest = hashlib.sha256()
    for content in (
        source.latest_csv_bytes,
        source.upcoming_report_bytes,
        source.prospective_evaluation_json_bytes,
        source.prospective_evaluation_report_bytes,
    ):
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _read_existing_manifest(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PublicationError(f"existing manifest is invalid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise PublicationError(f"existing manifest must contain an object: {path}")
    return payload


def _required_outputs_match(
    output_root: Path,
    *,
    history_path: Path | None,
    manifest: Mapping[str, Any],
) -> bool:
    required = [
        output_root / "latest.csv",
        output_root / "latest.json",
        output_root / "upcoming.md",
        output_root / "prospective_evaluation.json",
        output_root / "prospective_evaluation.md",
        output_root / "manifest.json",
    ]
    if history_path is not None:
        required.append(history_path)
    checksums = manifest.get("checksums")
    if not isinstance(checksums, Mapping):
        return False
    for path in required:
        if not path.is_file():
            return False
        relative_path = path.relative_to(output_root).as_posix()
        if relative_path == "manifest.json":
            continue
        checksum = checksums.get(relative_path)
        if not isinstance(checksum, str) or _sha256(path.read_bytes()) != checksum:
            return False
    return True


def _outputs_differ(
    previous_manifest: Mapping[str, Any] | None,
    next_manifest: Mapping[str, Any],
) -> bool:
    if previous_manifest is None:
        return True
    return previous_manifest.get("publication_fingerprint") != next_manifest.get(
        "publication_fingerprint"
    )


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PublicationError(f"timestamp is not valid ISO 8601 UTC: {value}") from exc
    return _require_utc(parsed)


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise PublicationError("timestamps must be timezone-aware UTC")
    return value.astimezone(UTC).replace(microsecond=0)


def _format_utc(value: datetime) -> str:
    return _require_utc(value).isoformat().replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _combined_secret_values(secret_values: Sequence[str]) -> tuple[str, ...]:
    values = [
        *secret_values,
        os.environ.get("FOOTBALL_DATA_API_KEY", ""),
        os.environ.get("API_FOOTBALL_KEY", ""),
    ]
    return tuple(value for value in values if len(value) >= 8)


def _assert_no_secret_text(files: Mapping[str, bytes], *, secret_values: Sequence[str]) -> None:
    for relative_path, content in files.items():
        text = _bytes_for_secret_scan(content, relative_path=relative_path)
        _assert_no_secret_string(relative_path, text, secret_values=secret_values)


def _assert_no_secret_string(
    relative_path: str,
    text: str,
    *,
    secret_values: Sequence[str],
) -> None:
    for value in secret_values:
        if value and value in text:
            raise PublicationError(f"secret value detected in publishable output: {relative_path}")
    if SECRET_ASSIGNMENT_PATTERN.search(text):
        raise PublicationError(
            f"secret-like assignment detected in publishable output: {relative_path}"
        )


def _file_text_for_secret_scan(path: Path) -> str:
    content = path.read_bytes()
    return _bytes_for_secret_scan(content, relative_path=path.as_posix())


def _bytes_for_secret_scan(content: bytes, *, relative_path: str) -> str:
    if relative_path.endswith(".gz"):
        try:
            content = gzip.decompress(content)
        except OSError as exc:
            raise PublicationError(
                f"failed to inspect compressed output for secrets: {relative_path}"
            ) from exc
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublicationError(f"publishable output must be UTF-8 text: {relative_path}") from exc
