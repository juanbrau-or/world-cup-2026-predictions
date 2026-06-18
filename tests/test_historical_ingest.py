import json
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import pytest
import respx
from typer.testing import CliRunner

from worldcup2026.cli import app
from worldcup2026.data.historical_ingest import (
    HistoricalIngestError,
    load_historical_source_config,
    run_historical_ingest,
)
from worldcup2026.data.sources import sha256_bytes

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "international_results"
RESULTS_FIXTURE = FIXTURE_DIR / "results.csv"
SHOOTOUTS_FIXTURE = FIXTURE_DIR / "shootouts.csv"
SOURCE_REVISION = "c44451d1a07f736502f364a62b6fbc947a544809"


def test_historical_source_config_uses_explicit_commit_revision() -> None:
    config = load_historical_source_config()

    assert config.source_revision == SOURCE_REVISION
    assert config.source_revision != "master"
    for file_config in config.files.values():
        assert file_config.logical_uri.endswith(f"@{SOURCE_REVISION}")
        assert f"/{SOURCE_REVISION}/" in file_config.url
        assert "master" not in file_config.logical_uri
        assert "/master/" not in file_config.url


def test_historical_source_config_rejects_branch_revision(tmp_path: Path) -> None:
    source_config = Path("configs/sources.yaml").read_text(encoding="utf-8")
    config_path = tmp_path / "sources.yaml"
    config_path.write_text(source_config.replace(SOURCE_REVISION, "master"), encoding="utf-8")

    with pytest.raises(HistoricalIngestError, match="source_revision"):
        load_historical_source_config(config_path)


def test_historical_ingest_from_local_fixture_is_idempotent(tmp_path: Path) -> None:
    config = load_historical_source_config()
    output_path = tmp_path / "processed" / "matches.parquet"
    quarantine_path = tmp_path / "interim" / "invalid.jsonl"
    report_path = tmp_path / "interim" / "report.json"

    first = run_historical_ingest(
        config,
        results_file=RESULTS_FIXTURE,
        shootouts_file=SHOOTOUTS_FIXTURE,
        raw_root=tmp_path / "raw",
        output_path=output_path,
        quarantine_path=quarantine_path,
        report_path=report_path,
    )
    second = run_historical_ingest(
        config,
        results_file=RESULTS_FIXTURE,
        shootouts_file=SHOOTOUTS_FIXTURE,
        raw_root=tmp_path / "raw",
        output_path=output_path,
        quarantine_path=quarantine_path,
        report_path=report_path,
    )

    assert first.report.results_rows_downloaded == 5
    assert first.report.results_rows_processed == 5
    assert first.report.valid_rows == 2
    assert first.report.invalid_rows == 4
    assert first.report.duplicate_rows == 1
    assert first.report.original_team_names == 7
    assert first.report.resolved_team_names == 6
    assert [name.source_name for name in first.report.unresolved_team_names] == ["Atlantis"]
    assert first.report.rows_with_resolved_team_names == 4
    assert [match.match_id for match in first.matches] == [
        match.match_id for match in second.matches
    ]
    assert {match.retrieved_at_utc for match in first.matches} == {
        config.snapshot_retrieved_at_utc
    }
    assert output_path.is_file()
    assert quarantine_path.read_text(encoding="utf-8").count("\n") == 4
    table = pq.read_table(output_path)
    assert table.num_rows == 2
    clean_output_path = tmp_path / "clean" / "processed" / "matches.parquet"
    run_historical_ingest(
        config,
        results_file=RESULTS_FIXTURE,
        shootouts_file=SHOOTOUTS_FIXTURE,
        raw_root=tmp_path / "clean" / "raw",
        output_path=clean_output_path,
        quarantine_path=tmp_path / "clean" / "interim" / "invalid.jsonl",
        report_path=tmp_path / "clean" / "interim" / "report.json",
    )
    assert clean_output_path.read_bytes() == output_path.read_bytes()


def test_historical_ingest_invalid_records_include_source_coordinates(tmp_path: Path) -> None:
    config = load_historical_source_config()

    result = run_historical_ingest(
        config,
        results_file=RESULTS_FIXTURE,
        shootouts_file=SHOOTOUTS_FIXTURE,
        raw_root=tmp_path / "raw",
        output_path=tmp_path / "processed" / "matches.parquet",
        quarantine_path=tmp_path / "interim" / "invalid.jsonl",
        report_path=tmp_path / "interim" / "report.json",
    )

    duplicate = next(record for record in result.invalid_records if record.stage == "deduplicate")
    missing_alias = next(
        record for record in result.invalid_records if "missing team alias" in record.reason
    )
    result_shootout = next(
        record for record in result.invalid_records if "shootout row lacks" in record.reason
    )
    shootout_marker = next(
        record for record in result.invalid_records if record.source_file == "shootouts.csv"
    )
    assert duplicate.row_number == 5
    assert missing_alias.row_number == 6
    assert result_shootout.row_number == 3
    assert shootout_marker.row_number == 2
    assert shootout_marker.stage == "shootout_validate"
    for record in result.invalid_records:
        assert record.logical_uri.endswith(f"@{SOURCE_REVISION}")
        assert record.source_revision == SOURCE_REVISION
        assert record.payload


def test_historical_ingest_records_local_input_uri_in_manifest(tmp_path: Path) -> None:
    config = load_historical_source_config()

    result = run_historical_ingest(
        config,
        results_file=RESULTS_FIXTURE,
        shootouts_file=SHOOTOUTS_FIXTURE,
        raw_root=tmp_path / "raw",
        output_path=tmp_path / "processed" / "matches.parquet",
        quarantine_path=tmp_path / "interim" / "invalid.jsonl",
        report_path=tmp_path / "interim" / "report.json",
    )

    assert result.report.snapshot_manifests[0].input_uri == RESULTS_FIXTURE.resolve().as_uri()
    assert result.report.snapshot_manifests[1].input_uri == SHOOTOUTS_FIXTURE.resolve().as_uri()


def test_historical_ingest_quarantines_malformed_csv_rows(tmp_path: Path) -> None:
    config = load_historical_source_config()
    malformed_results = tmp_path / "results-extra-field.csv"
    malformed_results.write_text(
        "\n".join(
            [
                "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral",
                "1872-11-30,Scotland,England,0,0,Friendly,Glasgow,Scotland,FALSE,unexpected",
                "2024-03-21,United States,Brazil,2,1,Friendly,Austin,United States,FALSE",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_historical_ingest(
        config,
        results_file=malformed_results,
        shootouts_file=SHOOTOUTS_FIXTURE,
        raw_root=tmp_path / "raw",
        output_path=tmp_path / "processed" / "matches.parquet",
        quarantine_path=tmp_path / "interim" / "invalid.jsonl",
        report_path=tmp_path / "interim" / "report.json",
    )

    parse_record = next(record for record in result.invalid_records if record.stage == "parse")
    assert parse_record.row_number == 2
    assert parse_record.source_file == "results.csv"
    assert parse_record.payload["_extra_fields"] == ["unexpected"]
    assert result.report.results_rows_downloaded == 2
    assert result.report.valid_rows == 1


def test_historical_ingest_quarantines_conflicting_duplicate_source_ids(tmp_path: Path) -> None:
    config = load_historical_source_config()
    conflicting_results = tmp_path / "results-conflict.csv"
    conflicting_results.write_text(
        "\n".join(
            [
                "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral",
                "2024-03-21,United States,Brazil,2,1,Friendly,Austin,United States,FALSE",
                "2024-03-21,United States,Brazil,3,1,Friendly,Austin,United States,FALSE",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_historical_ingest(
        config,
        results_file=conflicting_results,
        shootouts_file=SHOOTOUTS_FIXTURE,
        raw_root=tmp_path / "raw",
        output_path=tmp_path / "processed" / "matches.parquet",
        quarantine_path=tmp_path / "interim" / "invalid.jsonl",
        report_path=tmp_path / "interim" / "report.json",
    )

    conflicts = [
        record for record in result.invalid_records if "conflicting duplicate" in record.reason
    ]
    assert [record.row_number for record in conflicts] == [2, 3]
    assert result.report.duplicate_rows == 2
    assert result.report.valid_rows == 0


def test_historical_ingest_rejects_limit_when_writing_outputs(tmp_path: Path) -> None:
    config = load_historical_source_config()

    with pytest.raises(HistoricalIngestError, match="limit is only allowed"):
        run_historical_ingest(
            config,
            results_file=RESULTS_FIXTURE,
            shootouts_file=SHOOTOUTS_FIXTURE,
            raw_root=tmp_path / "raw",
            output_path=tmp_path / "processed" / "matches.parquet",
            quarantine_path=tmp_path / "interim" / "invalid.jsonl",
            report_path=tmp_path / "interim" / "report.json",
            limit=2,
        )

    assert not (tmp_path / "processed" / "matches.parquet").exists()
    assert not (tmp_path / "interim" / "invalid.jsonl").exists()
    assert not (tmp_path / "interim" / "report.json").exists()


def test_historical_ingest_allows_limit_for_dry_run_only(tmp_path: Path) -> None:
    config = load_historical_source_config()

    result = run_historical_ingest(
        config,
        results_file=RESULTS_FIXTURE,
        shootouts_file=SHOOTOUTS_FIXTURE,
        raw_root=tmp_path / "raw",
        output_path=tmp_path / "processed" / "matches.parquet",
        quarantine_path=tmp_path / "interim" / "invalid.jsonl",
        report_path=tmp_path / "interim" / "report.json",
        dry_run=True,
        limit=2,
    )

    assert result.report.results_rows_downloaded == 5
    assert result.report.results_rows_processed == 2
    assert result.report.output_path is None
    assert result.report.quarantine_path is None
    assert not (tmp_path / "processed" / "matches.parquet").exists()
    assert not any((tmp_path / "raw").rglob("*"))


def test_historical_ingest_rejects_stale_snapshot_manifest(tmp_path: Path) -> None:
    config = load_historical_source_config()
    first = run_historical_ingest(
        config,
        results_file=RESULTS_FIXTURE,
        shootouts_file=SHOOTOUTS_FIXTURE,
        raw_root=tmp_path / "raw",
        output_path=tmp_path / "processed" / "matches.parquet",
        quarantine_path=tmp_path / "interim" / "invalid.jsonl",
        report_path=tmp_path / "interim" / "report.json",
    )
    manifest_path = first.report.snapshot_manifests[0].raw_path.with_name(
        "results.csv.manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["content_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(HistoricalIngestError, match="snapshot manifest mismatch"):
        run_historical_ingest(
            config,
            results_file=RESULTS_FIXTURE,
            shootouts_file=SHOOTOUTS_FIXTURE,
            raw_root=tmp_path / "raw",
            output_path=tmp_path / "processed" / "matches.parquet",
            quarantine_path=tmp_path / "interim" / "invalid.jsonl",
            report_path=tmp_path / "interim" / "report.json",
        )


def test_historical_ingest_rejects_incomplete_snapshot_pair(tmp_path: Path) -> None:
    config = load_historical_source_config()
    first = run_historical_ingest(
        config,
        results_file=RESULTS_FIXTURE,
        shootouts_file=SHOOTOUTS_FIXTURE,
        raw_root=tmp_path / "raw",
        output_path=tmp_path / "processed" / "matches.parquet",
        quarantine_path=tmp_path / "interim" / "invalid.jsonl",
        report_path=tmp_path / "interim" / "report.json",
    )
    first.report.snapshot_manifests[0].raw_path.with_name("results.csv.manifest.json").unlink()

    with pytest.raises(HistoricalIngestError, match="incomplete immutable snapshot"):
        run_historical_ingest(
            config,
            results_file=RESULTS_FIXTURE,
            shootouts_file=SHOOTOUTS_FIXTURE,
            raw_root=tmp_path / "raw",
            output_path=tmp_path / "processed" / "matches.parquet",
            quarantine_path=tmp_path / "interim" / "invalid.jsonl",
            report_path=tmp_path / "interim" / "report.json",
        )


@respx.mock
def test_historical_ingest_fetches_http_sources_with_snapshots(tmp_path: Path) -> None:
    config = load_historical_source_config()
    results_config = config.files["results"]
    shootouts_config = config.files["shootouts"]
    results_content = RESULTS_FIXTURE.read_bytes()
    shootouts_content = SHOOTOUTS_FIXTURE.read_bytes()
    results_route = respx.get(results_config.url).mock(
        return_value=httpx.Response(200, content=results_content)
    )
    shootouts_route = respx.get(shootouts_config.url).mock(
        return_value=httpx.Response(200, content=shootouts_content)
    )

    result = run_historical_ingest(
        config,
        raw_root=tmp_path / "raw",
        output_path=tmp_path / "processed" / "matches.parquet",
        quarantine_path=tmp_path / "interim" / "invalid.jsonl",
        report_path=tmp_path / "interim" / "report.json",
    )

    assert results_route.called
    assert shootouts_route.called
    manifests = result.report.snapshot_manifests
    assert manifests[0].content_sha256 == sha256_bytes(results_content)
    assert manifests[1].content_sha256 == sha256_bytes(shootouts_content)
    assert manifests[0].source_revision == SOURCE_REVISION
    assert manifests[1].source_revision == SOURCE_REVISION
    assert manifests[0].raw_path.is_file()
    assert manifests[0].raw_path.parent == manifests[1].raw_path.parent
    assert manifests[0].raw_path.parent.name == "2026-06-18T00:00:00Z"
    persisted_manifest = json.loads(
        manifests[0].raw_path.with_name("results.csv.manifest.json").read_text(encoding="utf-8")
    )
    assert persisted_manifest["source_revision"] == SOURCE_REVISION
    assert persisted_manifest["input_uri"] == results_config.url
    assert result.report.output_path == tmp_path / "processed" / "matches.parquet"


@respx.mock
def test_historical_ingest_dry_run_does_not_write_snapshots(tmp_path: Path) -> None:
    config = load_historical_source_config()
    results_config = config.files["results"]
    shootouts_config = config.files["shootouts"]
    respx.get(results_config.url).mock(
        return_value=httpx.Response(200, content=RESULTS_FIXTURE.read_bytes())
    )
    respx.get(shootouts_config.url).mock(
        return_value=httpx.Response(200, content=SHOOTOUTS_FIXTURE.read_bytes())
    )

    result = run_historical_ingest(
        config,
        raw_root=tmp_path / "raw",
        output_path=tmp_path / "processed" / "matches.parquet",
        quarantine_path=tmp_path / "interim" / "invalid.jsonl",
        report_path=tmp_path / "interim" / "report.json",
        dry_run=True,
    )

    assert result.report.output_path is None
    assert not any((tmp_path / "raw").rglob("*"))
    assert not (tmp_path / "interim" / "invalid.jsonl").exists()


@respx.mock
def test_historical_ingest_retries_timeout_then_succeeds(tmp_path: Path) -> None:
    config = load_historical_source_config()
    results_config = config.files["results"]
    shootouts_config = config.files["shootouts"]
    results_content = RESULTS_FIXTURE.read_bytes()
    shootouts_content = SHOOTOUTS_FIXTURE.read_bytes()
    results_route = respx.get(results_config.url).mock(
        side_effect=[
            httpx.TimeoutException("timed out"),
            httpx.Response(200, content=results_content),
        ]
    )
    shootouts_route = respx.get(shootouts_config.url).mock(
        return_value=httpx.Response(200, content=shootouts_content)
    )

    result = run_historical_ingest(
        config,
        raw_root=tmp_path / "raw",
        output_path=tmp_path / "processed" / "matches.parquet",
        quarantine_path=tmp_path / "interim" / "invalid.jsonl",
        report_path=tmp_path / "interim" / "report.json",
        dry_run=True,
    )

    assert result.report.results_rows_downloaded == 5
    assert results_route.call_count == 2
    assert shootouts_route.call_count == 1


@respx.mock
def test_historical_ingest_reports_http_failure_after_retries(tmp_path: Path) -> None:
    config = load_historical_source_config().model_copy(update={"retries": 1})
    results_config = config.files["results"]
    results_route = respx.get(results_config.url).mock(
        return_value=httpx.Response(500, text="upstream error")
    )

    with pytest.raises(HistoricalIngestError, match="failed to download"):
        run_historical_ingest(
            config,
            raw_root=tmp_path / "raw",
            output_path=tmp_path / "processed" / "matches.parquet",
            quarantine_path=tmp_path / "interim" / "invalid.jsonl",
            report_path=tmp_path / "interim" / "report.json",
            dry_run=True,
        )

    assert results_route.call_count == 2


def test_historical_ingest_cli_accepts_local_fixture(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ingest",
            "historical",
            "--results-file",
            str(RESULTS_FIXTURE),
            "--shootouts-file",
            str(SHOOTOUTS_FIXTURE),
            "--raw-root",
            str(tmp_path / "raw"),
            "--output",
            str(tmp_path / "processed" / "matches.parquet"),
            "--quarantine",
            str(tmp_path / "interim" / "invalid.jsonl"),
            "--report",
            str(tmp_path / "interim" / "report.json"),
        ],
    )

    assert result.exit_code == 0
    assert "Normalized rows: 2" in result.stdout
    assert "Invalid rows: 4" in result.stdout


def test_historical_ingest_cli_reports_invalid_config(tmp_path: Path) -> None:
    config_path = tmp_path / "sources.yaml"
    config_path.write_text("historical:\n  provider: international_results_csv\n", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(app, ["ingest", "historical", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "not configured" in result.stdout


def test_historical_ingest_cli_reports_missing_local_file(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ingest",
            "historical",
            "--results-file",
            str(tmp_path / "missing-results.csv"),
            "--shootouts-file",
            str(SHOOTOUTS_FIXTURE),
            "--raw-root",
            str(tmp_path / "raw"),
            "--output",
            str(tmp_path / "processed" / "matches.parquet"),
            "--quarantine",
            str(tmp_path / "interim" / "invalid.jsonl"),
            "--report",
            str(tmp_path / "interim" / "report.json"),
        ],
    )

    assert result.exit_code == 1
    assert "failed to read local source file" in result.stdout


def test_alias_audit_cli_reports_fixture_coverage(tmp_path: Path) -> None:
    clean_results = tmp_path / "results-clean.csv"
    clean_results.write_text(
        "\n".join(
            [
                "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral",
                "1872-11-30,Scotland,England,0,0,Friendly,Glasgow,Scotland,FALSE",
                "2024-03-21,United States,Brazil,2,1,Friendly,Austin,United States,FALSE",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "alias-audit.json"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "audit",
            "aliases",
            "--results-file",
            str(clean_results),
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert "Original team names: 4" in result.stdout
    assert "Resolved team names: 4" in result.stdout
    assert "Rows with resolved team names: 2/2 (100.00%)" in result.stdout
    assert json.loads(report_path.read_text(encoding="utf-8"))["unresolved_team_names"] == []


def test_alias_audit_cli_fails_when_names_are_unresolved() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "audit",
            "aliases",
            "--results-file",
            str(RESULTS_FIXTURE),
        ],
    )

    assert result.exit_code == 1
    assert "Unresolved team names: 1" in result.stdout
