from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from worldcup2026.simulation.tournament import audit_simulation_outputs


def test_simulation_audit_reports_random_lot_runs(tmp_path: Path) -> None:
    latest = tmp_path / "simulations" / "latest"
    latest.mkdir(parents=True)
    (latest / "team_probabilities.csv").write_text(
        "team_id,champion,final,semi_final,quarter_final,round_of_16,round_of_32\n"
        "mexico,0.1,0.2,0.3,0.4,0.5,1.0\n",
        encoding="utf-8",
    )
    (latest / "team_probabilities.json").write_text('{"teams":[]}\n', encoding="utf-8")
    (latest / "group_probabilities.csv").write_text("group,team_id\n", encoding="utf-8")
    for name in (
        "group_tables_summary.md",
        "round_probabilities.md",
        "champion_probabilities.md",
        "bracket_summary.md",
        "stability_report.md",
    ):
        (latest / name).write_text("# Report\n", encoding="utf-8")
    (latest / "manifest.json").write_text(
        json.dumps(
            {
                "simulation_run_id": "sim-run",
                "rules": {"version": "world_cup_2026_rules_v1"},
                "model": {"version": "poisson_goal_v1"},
                "fallback_counts": {"random_lot_proxy": 2},
            }
        ),
        encoding="utf-8",
    )
    table = pa.Table.from_pylist(
        [
            {"run_index": 0, "random_lot_proxy_count": 1},
            {"run_index": 1, "random_lot_proxy_count": 1},
        ]
    )
    pq.write_table(table, latest / "simulation_results.parquet")

    summary = audit_simulation_outputs(simulations_root=tmp_path / "simulations")

    assert summary["fallback_counts"] == {"random_lot_proxy": 2}
    assert summary["runs_with_random_lot_proxy"] == 2
