from __future__ import annotations

import pytest

from worldcup2026.simulation.rules import (
    ANNEX_C_SLOT_ORDER,
    GROUPS,
    RULE_VERSION,
    TournamentRuleError,
    annex_c_assignment,
    annex_c_table,
    normalize_stage,
    team_slot,
)


def test_annex_c_table_contains_all_official_combinations() -> None:
    table = annex_c_table()

    assert RULE_VERSION == "world_cup_2026_rules_v1"
    assert len(table) == 495
    assert all(len(groups) == 8 for groups in table)
    assert all(set(assignment) == set(ANNEX_C_SLOT_ORDER) for assignment in table.values())


def test_annex_c_known_combination_mapping() -> None:
    assignment = annex_c_assignment("EFGHIJKL")

    assert assignment == {
        "1A": "E",
        "1B": "J",
        "1D": "I",
        "1E": "F",
        "1G": "H",
        "1I": "G",
        "1K": "L",
        "1L": "K",
    }


def test_annex_c_rejects_invalid_combination() -> None:
    with pytest.raises(TournamentRuleError, match="unsupported Annex C"):
        annex_c_assignment("ABCDEFG")


def test_stage_and_team_slot_normalization() -> None:
    assert GROUPS[0] == "A"
    assert normalize_stage("LAST_32") == "round_of_32"
    assert normalize_stage("FINAL") == "final"
    assert team_slot("3L").group == "L"
    assert team_slot("3L").position == 3

