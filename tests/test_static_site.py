from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from worldcup2026.site import SiteBuildError, build_site

CUTOFF = "2026-07-01T21:40:07Z"
GENERATED_AT = "2026-07-01T21:45:58Z"


def test_site_build_renders_all_pages_from_full_public_data(tmp_path: Path) -> None:
    data_root = _write_public_data(tmp_path / "data")
    output_root = tmp_path / "site"

    result = build_site(data_root=data_root, output_root=output_root)

    assert result.page_count == 8
    assert result.manifest_path.is_file()
    assert result.checksum_report_path.is_file()
    assert result.coverage_report_path.is_file()
    assert (output_root / "assets" / "styles.css").is_file()
    assert (output_root / "assets" / "site.js").is_file()

    index = (output_root / "index.html").read_text(encoding="utf-8")
    upcoming = (output_root / "upcoming" / "index.html").read_text(encoding="utf-8")
    shadow = (output_root / "shadow" / "index.html").read_text(encoding="utf-8")
    simulation = (output_root / "simulation" / "index.html").read_text(encoding="utf-8")
    groups = (output_root / "groups" / "index.html").read_text(encoding="utf-8")
    bracket = (output_root / "bracket" / "index.html").read_text(encoding="utf-8")

    assert "poisson_goal_v1" in index
    assert "contextual_logit_v1" in index
    assert "random_lot_proxy=2" in index
    assert "Mexico &amp; Co" in upcoming
    assert "<script>alert" not in upcoming
    assert "Challenger experimental" in shadow
    assert "No se combinan probabilidades" in shadow
    assert "sim-run-1" in simulation
    assert "resultado observado" in groups
    assert "Cruce simulado frecuente" in bracket
    assert 'href="../assets/styles.css"' in upcoming
    assert "javascript:" not in (output_root / "methodology" / "index.html").read_text(
        encoding="utf-8"
    )


def test_site_build_supports_empty_optional_outputs(tmp_path: Path) -> None:
    data_root = _write_public_data(
        tmp_path / "data",
        rows=[],
        include_shadow=False,
        include_simulation=False,
        scorecard_matches=0,
    )

    result = build_site(data_root=data_root, output_root=tmp_path / "site")

    assert result.page_count == 8
    upcoming = (tmp_path / "site" / "upcoming" / "index.html").read_text(encoding="utf-8")
    simulation = (tmp_path / "site" / "simulation" / "index.html").read_text(encoding="utf-8")
    assert "No hay próximos partidos publicados" in upcoming
    assert "La simulación todavía no está publicada" in simulation
    assert any("El challenger shadow no está publicado" in warning for warning in result.warnings)


def test_invalid_manifest_checksum_is_rejected(tmp_path: Path) -> None:
    data_root = _write_public_data(tmp_path / "data")
    manifest_path = data_root / "manifest.json"
    manifest = _json(manifest_path)
    manifest["checksums"]["latest.csv"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(SiteBuildError, match="checksum mismatch"):
        build_site(data_root=data_root, output_root=tmp_path / "site")


def test_invalid_probabilities_are_rejected(tmp_path: Path) -> None:
    row = _prediction_row(probability_home_win="0.9")
    data_root = _write_public_data(tmp_path / "data", rows=[row])

    with pytest.raises(SiteBuildError, match="probabilities do not sum"):
        build_site(data_root=data_root, output_root=tmp_path / "site")


def test_invalid_timestamps_are_rejected(tmp_path: Path) -> None:
    row = _prediction_row(kickoff_utc="2026-07-02T12:00:00-05:00")
    data_root = _write_public_data(tmp_path / "data", rows=[row])

    with pytest.raises(SiteBuildError, match="timezone-aware UTC"):
        build_site(data_root=data_root, output_root=tmp_path / "site")


def test_official_shadow_mixing_is_rejected(tmp_path: Path) -> None:
    row = _prediction_row(model_family="contextual_challenger")
    data_root = _write_public_data(tmp_path / "data", rows=[row])

    with pytest.raises(SiteBuildError, match="official row is mixed"):
        build_site(data_root=data_root, output_root=tmp_path / "site")


def test_shadow_official_mixing_is_rejected(tmp_path: Path) -> None:
    data_root = _write_public_data(
        tmp_path / "data",
        shadow_rows=[_shadow_row(model_version="poisson_goal_v1")],
    )

    with pytest.raises(SiteBuildError, match="shadow row is mixed"):
        build_site(data_root=data_root, output_root=tmp_path / "site")


def test_security_scans_reject_secrets_env_raw_and_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "SECRET_VALUE_123456")
    data_root = _write_public_data(
        tmp_path / "secret-data",
        rows=[_prediction_row(home_team_name="SECRET_VALUE_123456")],
    )
    with pytest.raises(SiteBuildError, match="secret value detected"):
        build_site(data_root=data_root, output_root=tmp_path / "site-secret")

    env_root = _write_public_data(tmp_path / "env-data")
    (env_root / ".env").write_text("FOOTBALL_DATA_API_KEY=SECRET_VALUE_123456\n")
    with pytest.raises(SiteBuildError, match="env paths|path is not allowed"):
        build_site(data_root=env_root, output_root=tmp_path / "site-env")

    raw_root = _write_public_data(tmp_path / "raw-data")
    (raw_root / "raw").mkdir()
    (raw_root / "raw" / "payload.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SiteBuildError, match="raw/model/env"):
        build_site(data_root=raw_root, output_root=tmp_path / "site-raw")

    parquet_root = _write_public_data(tmp_path / "parquet-data")
    (parquet_root / "latest.parquet").write_bytes(b"not public")
    with pytest.raises(SiteBuildError, match="Parquet"):
        build_site(data_root=parquet_root, output_root=tmp_path / "site-parquet")


def test_site_build_is_reproducible(tmp_path: Path) -> None:
    data_root = _write_public_data(tmp_path / "data")
    first = build_site(data_root=data_root, output_root=tmp_path / "site-a")
    second = build_site(data_root=data_root, output_root=tmp_path / "site-b")

    assert first.site_checksum == second.site_checksum
    assert _tree_digest(tmp_path / "site-a") == _tree_digest(tmp_path / "site-b")
    manifest = _json(tmp_path / "site-a" / "build-manifest.json")
    checksums = _json(tmp_path / "site-a" / "checksum-report.json")
    coverage = _json(tmp_path / "site-a" / "page-coverage.json")
    assert manifest["schema_version"] == "wc2026_static_site_build_v1"
    assert checksums["site_checksum"] == first.site_checksum
    assert len(coverage["pages"]) == 8


def _write_public_data(
    root: Path,
    *,
    rows: list[dict[str, str]] | None = None,
    shadow_rows: list[dict[str, str]] | None = None,
    include_shadow: bool = True,
    include_simulation: bool = True,
    scorecard_matches: int = 1,
) -> Path:
    root.mkdir(parents=True)
    official_rows = [_prediction_row()] if rows is None else rows
    _write_latest(root / "latest.csv", rows=official_rows)
    _write_json(
        root / "latest.json",
        _latest_payload("predictions_latest_v1", official_rows, model_family="poisson"),
    )
    (root / "upcoming.md").write_text("# Upcoming\n", encoding="utf-8")
    _write_json(root / "prospective_scorecard.json", _scorecard(scorecard_matches))
    (root / "prospective_scorecard.md").write_text("# Scorecard\n", encoding="utf-8")
    (root / "prospective_matches.csv").write_text(
        "source_fixture_id,prediction_id\n",
        encoding="utf-8",
    )

    published_files = [
        "latest.csv",
        "latest.json",
        "upcoming.md",
        "prospective_scorecard.json",
        "prospective_scorecard.md",
        "prospective_matches.csv",
    ]
    shadow_manifest: dict[str, Any] | None = None
    if include_shadow:
        shadow = root / "shadow"
        shadow.mkdir()
        shadow_payload_rows = [_shadow_row()] if shadow_rows is None else shadow_rows
        _write_latest(shadow / "contextual_latest.csv", rows=shadow_payload_rows)
        _write_json(
            shadow / "contextual_latest.json",
            _latest_payload(
                "shadow_contextual_latest_v1",
                shadow_payload_rows,
                model_family="contextual_challenger",
                model_version="contextual_logit_v1",
            ),
        )
        (shadow / "contextual_upcoming.md").write_text("# Shadow\n", encoding="utf-8")
        _write_json(shadow / "contextual_scorecard.json", _scorecard(0, shadow=True))
        (shadow / "contextual_scorecard.md").write_text("# Shadow Scorecard\n", encoding="utf-8")
        (shadow / "contextual_comparison.md").write_text("# Comparison\n", encoding="utf-8")
        shadow_files = [
            "shadow/contextual_latest.csv",
            "shadow/contextual_latest.json",
            "shadow/contextual_upcoming.md",
            "shadow/contextual_scorecard.json",
            "shadow/contextual_scorecard.md",
            "shadow/contextual_comparison.md",
        ]
        shadow_manifest = {
            "schema_version": "shadow_contextual_publication_manifest_v1",
            "version": "publication_v1",
            "generated_at": GENERATED_AT,
            "data_cutoff": CUTOFF,
            "model": {"family": "contextual_challenger", "version": "contextual_logit_v1"},
            "prediction_context": "shadow_contextual_v1",
            "prediction_count": len(shadow_payload_rows),
            "scorecard": {"official_predictions_evaluated": 0},
            "published_files": shadow_files,
            "checksums": _checksums(root, shadow_files),
            "checksum": _sha256(root / "shadow" / "contextual_latest.csv"),
        }
        _write_json(shadow / "manifest.json", shadow_manifest)
        published_files.extend([*shadow_files, "shadow/manifest.json"])

    simulation_manifest: dict[str, Any] | None = None
    if include_simulation:
        simulation = root / "simulation"
        simulation.mkdir()
        simulation_manifest = {
            "schema_version": "world_cup_simulation_v1",
            "simulation_run_id": "sim-run-1",
            "data_cutoff_utc": CUTOFF,
            "runs": 2,
            "seed": 2026,
            "model": {"family": "poisson", "version": "poisson_goal_v1"},
            "rules": {"version": "world_cup_2026_rules_v1"},
            "fallback_counts": {"random_lot_proxy": 2},
        }
        _write_json(simulation / "manifest.json", simulation_manifest)
        (simulation / "team_probabilities.csv").write_text(
            "team_id,champion,final,semi_final,quarter_final,round_of_16,round_of_32\n"
            "mexico,0.2,0.3,0.4,0.5,0.6,1.0\n"
            "canada,0.1,0.2,0.3,0.4,0.5,1.0\n",
            encoding="utf-8",
        )
        _write_json(
            simulation / "team_probabilities.json",
            {
                "schema_version": "team_simulation_probabilities_v1",
                "teams": [
                    {
                        "team_id": "mexico",
                        "champion": 0.2,
                        "final": 0.3,
                        "semi_final": 0.4,
                        "quarter_final": 0.5,
                        "round_of_16": 0.6,
                        "round_of_32": 1.0,
                    },
                    {
                        "team_id": "canada",
                        "champion": 0.1,
                        "final": 0.2,
                        "semi_final": 0.3,
                        "quarter_final": 0.4,
                        "round_of_16": 0.5,
                        "round_of_32": 1.0,
                    },
                ],
            },
        )
        (simulation / "champion_probabilities.md").write_text("# Champions\n", encoding="utf-8")
        (simulation / "round_probabilities.md").write_text("# Rounds\n", encoding="utf-8")
        (simulation / "group_tables_summary.md").write_text(
            "| Group | Team | 1st | 2nd | 3rd | 4th | Direct | Best 3rd | Eliminated |\n"
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
            "| A | mexico | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |\n"
            "| A | canada | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |\n",
            encoding="utf-8",
        )
        (simulation / "bracket_summary.md").write_text(
            "| Pair | Probability |\n"
            "| --- | ---: |\n"
            "| round_of_32:mexico v canada | 1.000 |\n"
            "| round_of_16:mexico v brazil | 0.420 |\n",
            encoding="utf-8",
        )
        published_files.extend(
            [
                "simulation/manifest.json",
                "simulation/team_probabilities.csv",
                "simulation/team_probabilities.json",
                "simulation/champion_probabilities.md",
                "simulation/round_probabilities.md",
                "simulation/group_tables_summary.md",
                "simulation/bracket_summary.md",
            ]
        )

    manifest = {
        "schema_version": "prediction_publication_manifest_v1",
        "version": "publication_v1",
        "generated_at": GENERATED_AT,
        "data_cutoff": CUTOFF,
        "model": {"family": "poisson", "version": "poisson_goal_v1"},
        "prediction_count": len(official_rows),
        "prospective_observations": scorecard_matches,
        "prospective_policy_version": "early_v1_2026_06_30",
        "shadow": shadow_manifest,
        "simulation": simulation_manifest,
        "published_files": sorted(published_files),
        "checksums": _checksums(root, published_files),
        "checksum": _sha256(root / "latest.csv"),
    }
    _write_json(root / "manifest.json", manifest)
    return root


def _prediction_row(**overrides: str) -> dict[str, str]:
    row = {
        "prediction_id": "prediction-1",
        "source_fixture_id": "fixture-1",
        "prediction_created_at_utc": "2026-07-01T21:00:00Z",
        "data_cutoff_utc": CUTOFF,
        "kickoff_utc": "2026-07-02T12:00:00Z",
        "hours_before_kickoff": "14.0",
        "home_team_id": "mexico",
        "away_team_id": "canada",
        "home_team_name": "Mexico & Co <script>alert(1)</script>",
        "away_team_name": "Canada",
        "expected_home_goals": "1.4",
        "expected_away_goals": "1.1",
        "probability_home_win": "0.400000",
        "probability_draw": "0.300000",
        "probability_away_win": "0.300000",
        "modal_score": "1-1",
        "model_family": "poisson",
        "model_version": "poisson_goal_v1",
        "prediction_context": "early_v1",
    }
    row.update(overrides)
    return row


def _shadow_row(**overrides: str) -> dict[str, str]:
    row = {
        **_prediction_row(),
        "prediction_id": "shadow-1",
        "model_family": "contextual_challenger",
        "model_version": "contextual_logit_v1",
        "prediction_context": "shadow_contextual_v1",
        "probability_home_win": "0.450000",
        "probability_draw": "0.250000",
        "probability_away_win": "0.300000",
    }
    row.update(overrides)
    return row


def _latest_payload(
    schema_version: str,
    rows: list[dict[str, str]],
    *,
    model_family: str,
    model_version: str = "poisson_goal_v1",
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "generated_at": GENERATED_AT,
        "data_cutoff": CUTOFF,
        "model": {"family": model_family, "version": model_version},
        "prediction_count": len(rows),
        "predictions": rows,
    }


def _scorecard(matches: int, *, shadow: bool = False) -> dict[str, Any]:
    return {
        "schema_version": "prospective_scorecard_v1",
        "generated_at_utc": GENERATED_AT,
        "results_cutoff_utc": CUTOFF,
        "ledger": {"snapshots": 1, "predictions": matches, "unique_fixtures": matches},
        "official_selection_policy": {
            "policy_id": "shadow_contextual_early_v1" if shadow else "early_v1",
            "policy_version": "shadow_contextual_early_v1_2026_06_30"
            if shadow
            else "early_v1_2026_06_30",
            "prediction_context": "shadow_contextual_v1" if shadow else "early_v1",
        },
        "official_predictions_selected": matches,
        "official_predictions_evaluated": matches,
        "small_sample_warning": {
            "applies": matches < 30,
            "threshold": 30,
            "message": "Sample is too small for firm statistical conclusions.",
        },
        "baselines": {
            "uniform_1x2": {
                "status": "computed",
                "metrics": {
                    "matches": matches,
                    "log_loss": 1.0986,
                    "brier_score": 0.6667,
                    "ranked_probability_score": 0.2778,
                },
            }
        },
        "metrics": {
            "matches": matches,
            "log_loss": 0.7,
            "brier_score": 0.38,
            "ranked_probability_score": 0.16,
            "accuracy": 0.75,
        },
        "matches": [
            {
                "home_team_name": "Mexico",
                "away_team_name": "Canada",
                "kickoff_utc": "2026-07-01T16:00:00Z",
                "data_cutoff_utc": CUTOFF,
                "prediction_created_at_utc": "2026-07-01T09:00:00Z",
                "actual_result_90": "home_win",
                "predicted_result": "home_win",
                "probability_home_win": 0.4,
                "probability_draw": 0.3,
                "probability_away_win": 0.3,
                "stage": "LAST_32",
            }
        ]
        if matches
        else [],
    }


def _write_latest(path: Path, *, rows: list[dict[str, str]]) -> None:
    fieldnames = list(_prediction_row())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checksums(root: Path, relative_paths: list[str]) -> dict[str, str]:
    return {
        relative_path: _sha256(root / relative_path)
        for relative_path in sorted(relative_paths)
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
