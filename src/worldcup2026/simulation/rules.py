"""Versioned FIFA World Cup 2026 tournament rules used by the simulator."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

RULE_VERSION = "world_cup_2026_rules_v1"
RULE_SOURCE_URL = (
    "https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/"
    "FWC2026_regulations_EN.pdf"
)
RULE_SOURCE_VERSION = "FIFA World Cup 26 Regulations, May 2026"
RULE_SOURCE_ACCESSED = "2026-07-01"

GROUPS = tuple("ABCDEFGHIJKL")
GROUP_COUNT = 12
TEAMS_PER_GROUP = 4
GROUP_MATCHES_PER_GROUP = 6
DIRECT_QUALIFIERS_PER_GROUP = 2
BEST_THIRD_QUALIFIERS = 8
ROUND_OF_32_TEAMS = 32

ANNEX_C_SLOT_ORDER = ("1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L")
ANNEX_C_SLOT_TO_MATCH = {
    "1A": 79,
    "1B": 85,
    "1D": 81,
    "1E": 74,
    "1G": 82,
    "1I": 77,
    "1K": 87,
    "1L": 80,
}
THIRD_PLACE_SLOT_ELIGIBILITY = {
    "1A": frozenset("CEFHI"),
    "1B": frozenset("EFGIJ"),
    "1D": frozenset("BEFIJ"),
    "1E": frozenset("ABCDF"),
    "1G": frozenset("AEHIJ"),
    "1I": frozenset("CDFGH"),
    "1K": frozenset("DEIJL"),
    "1L": frozenset("EHIJK"),
}


@dataclass(frozen=True)
class TeamSlot:
    """A group-stage finishing position used by a knockout fixture."""

    group: str
    position: int

    @property
    def label(self) -> str:
        return f"{self.position}{self.group}"


@dataclass(frozen=True)
class KnockoutTemplate:
    """Official bracket dependency for one knockout match."""

    match_number: int
    stage: str
    home: str
    away: str


ROUND_OF_32_FIXED_TEMPLATES = (
    KnockoutTemplate(73, "round_of_32", "2A", "2B"),
    KnockoutTemplate(75, "round_of_32", "1F", "2C"),
    KnockoutTemplate(76, "round_of_32", "1C", "2F"),
    KnockoutTemplate(78, "round_of_32", "2E", "2I"),
    KnockoutTemplate(83, "round_of_32", "2K", "2L"),
    KnockoutTemplate(84, "round_of_32", "1H", "2J"),
    KnockoutTemplate(86, "round_of_32", "1J", "2H"),
    KnockoutTemplate(88, "round_of_32", "2D", "2G"),
)
ROUND_OF_32_THIRD_PLACE_HOME_SLOTS = {
    74: "1E",
    77: "1I",
    79: "1A",
    80: "1L",
    81: "1D",
    82: "1G",
    85: "1B",
    87: "1K",
}
ROUND_OF_16_TEMPLATES = (
    KnockoutTemplate(89, "round_of_16", "W74", "W77"),
    KnockoutTemplate(90, "round_of_16", "W73", "W75"),
    KnockoutTemplate(91, "round_of_16", "W76", "W78"),
    KnockoutTemplate(92, "round_of_16", "W79", "W80"),
    KnockoutTemplate(93, "round_of_16", "W83", "W84"),
    KnockoutTemplate(94, "round_of_16", "W81", "W82"),
    KnockoutTemplate(95, "round_of_16", "W86", "W88"),
    KnockoutTemplate(96, "round_of_16", "W85", "W87"),
)
QUARTER_FINAL_TEMPLATES = (
    KnockoutTemplate(97, "quarter_final", "W89", "W90"),
    KnockoutTemplate(98, "quarter_final", "W93", "W94"),
    KnockoutTemplate(99, "quarter_final", "W91", "W92"),
    KnockoutTemplate(100, "quarter_final", "W95", "W96"),
)
SEMI_FINAL_TEMPLATES = (
    KnockoutTemplate(101, "semi_final", "W97", "W98"),
    KnockoutTemplate(102, "semi_final", "W99", "W100"),
)
THIRD_PLACE_TEMPLATE = KnockoutTemplate(103, "third_place", "L101", "L102")
FINAL_TEMPLATE = KnockoutTemplate(104, "final", "W101", "W102")

KNOCKOUT_ROUNDS = (
    ROUND_OF_16_TEMPLATES,
    QUARTER_FINAL_TEMPLATES,
    SEMI_FINAL_TEMPLATES,
    (THIRD_PLACE_TEMPLATE, FINAL_TEMPLATE),
)

STAGE_ALIASES = {
    "GROUP_STAGE": "group_stage",
    "LAST_32": "round_of_32",
    "LAST_16": "round_of_16",
    "QUARTER_FINALS": "quarter_final",
    "SEMI_FINALS": "semi_final",
    "THIRD_PLACE": "third_place",
    "FINAL": "final",
}

_ANNEX_C_ROWS = """
EJIFHGLK
HGIDJFLK
EJIDHGLK
EJIDHFLK
EGIDJFLK
EGJDHFLK
EGIDHFLK
EGJDHFLI
EGJDHFIK
HGICJFLK
EJICHGLK
EJICHFLK
EGICJFLK
EGJCHFLK
EGICHFLK
EGJCHFLI
EGJCHFIK
HGICJDLK
CJIDHFLK
CGIDJFLK
CGJDHFLK
CGIDHFLK
CGJDHFLI
CGJDHFIK
EJICHDLK
EGICJDLK
EGJCHDLK
EGICHDLK
EGJCHDLI
EGJCHDIK
CJEDIFLK
CJEDHFLK
CEIDHFLK
CJEDHFLI
CJEDHFIK
CGEDJFLK
CGEDIFLK
CGEDJFLI
CGEDJFIK
CGEDHFLK
CGJDHFLE
CGJDHFEK
CGEDHFLI
CGEDHFIK
CGJDHFEI
HJBFIGLK
EJIBHGLK
EJBFIHLK
EJBFIGLK
EJBFHGLK
EGBFIHLK
EJBFHGLI
EJBFHGIK
HJBDIGLK
HJBDIFLK
IGBDJFLK
HGBDJFLK
HGBDIFLK
HGBDJFLI
HGBDJFIK
EJBDIHLK
EJBDIGLK
EJBDHGLK
EGBDIHLK
EJBDHGLI
EJBDHGIK
EJBDIFLK
EJBDHFLK
EIBDHFLK
EJBDHFLI
EJBDHFIK
EGBDJFLK
EGBDIFLK
EGBDJFLI
EGBDJFIK
EGBDHFLK
HGBDJFLE
HGBDJFEK
EGBDHFLI
EGBDHFIK
HGBDJFEI
HJBCIGLK
HJBCIFLK
IGBCJFLK
HGBCJFLK
HGBCIFLK
HGBCJFLI
HGBCJFIK
EJBCIHLK
EJBCIGLK
EJBCHGLK
EGBCIHLK
EJBCHGLI
EJBCHGIK
EJBCIFLK
EJBCHFLK
EIBCHFLK
EJBCHFLI
EJBCHFIK
EGBCJFLK
EGBCIFLK
EGBCJFLI
EGBCJFIK
EGBCHFLK
HGBCJFLE
HGBCJFEK
EGBCHFLI
EGBCHFIK
HGBCJFEI
HJBCIDLK
IGBCJDLK
HGBCJDLK
HGBCIDLK
HGBCJDLI
HGBCJDIK
CJBDIFLK
CJBDHFLK
CIBDHFLK
CJBDHFLI
CJBDHFIK
CGBDJFLK
CGBDIFLK
CGBDJFLI
CGBDJFIK
CGBDHFLK
CGBDHFLJ
HGBCJFDK
CGBDHFLI
CGBDHFIK
HGBCJFDI
EJBCIDLK
EJBCHDLK
EIBCHDLK
EJBCHDLI
EJBCHDIK
EGBCJDLK
EGBCIDLK
EGBCJDLI
EGBCJDIK
EGBCHDLK
HGBCJDLE
HGBCJDEK
EGBCHDLI
EGBCHDIK
HGBCJDEI
CJBDEFLK
CEBDIFLK
CJBDEFLI
CJBDEFIK
CEBDHFLK
CJBDHFLE
CJBDHFEK
CEBDHFLI
CEBDHFIK
CJBDHFEI
CGBDEFLK
CGBDJFLE
CGBDJFEK
CGBDEFLI
CGBDEFIK
CGBDJFEI
CGBDHFLE
CGBDHFEK
HGBCJFDE
CGBDHFEI
HJIFAGLK
EJIAHGLK
EJIFAHLK
EJIFAGLK
EGJFAHLK
EGIFAHLK
EGJFAHLI
EGJFAHIK
HJIDAGLK
HJIDAFLK
IGJDAFLK
HGJDAFLK
HGIDAFLK
HGJDAFLI
HGJDAFIK
EJIDAHLK
EJIDAGLK
EGJDAHLK
EGIDAHLK
EGJDAHLI
EGJDAHIK
EJIDAFLK
HJEDAFLK
HEIDAFLK
HJEDAFLI
HJEDAFIK
EGJDAFLK
EGIDAFLK
EGJDAFLI
EGJDAFIK
HGEDAFLK
HGJDAFLE
HGJDAFEK
HGEDAFLI
HGEDAFIK
HGJDAFEI
HJICAGLK
HJICAFLK
IGJCAFLK
HGJCAFLK
HGICAFLK
HGJCAFLI
HGJCAFIK
EJICAHLK
EJICAGLK
EGJCAHLK
EGICAHLK
EGJCAHLI
EGJCAHIK
EJICAFLK
HJECAFLK
HEICAFLK
HJECAFLI
HJECAFIK
EGJCAFLK
EGICAFLK
EGJCAFLI
EGJCAFIK
HGECAFLK
HGJCAFLE
HGJCAFEK
HGECAFLI
HGECAFIK
HGJCAFEI
HJICADLK
IGJCADLK
HGJCADLK
HGICADLK
HGJCADLI
HGJCADIK
CJIDAFLK
HJFCADLK
HFICADLK
HJFCADLI
HJFCADIK
CGJDAFLK
CGIDAFLK
CGJDAFLI
CGJDAFIK
HGFCADLK
CGJDAFLH
HGJCAFDK
HGFCADLI
HGFCADIK
HGJCAFDI
EJICADLK
HJECADLK
HEICADLK
HJECADLI
HJECADIK
EGJCADLK
EGICADLK
EGJCADLI
EGJCADIK
HGECADLK
HGJCADLE
HGJCADEK
HGECADLI
HGECADIK
HGJCADEI
CJEDAFLK
CEIDAFLK
CJEDAFLI
CJEDAFIK
HEFCADLK
HJFCADLE
HJECAFDK
HEFCADLI
HEFCADIK
HJECAFDI
CGEDAFLK
CGJDAFLE
CGJDAFEK
CGEDAFLI
CGEDAFIK
CGJDAFEI
HGFCADLE
HGECAFDK
HGJCAFDE
HGECAFDI
HJBAIGLK
HJBAIFLK
IJBFAGLK
HJBFAGLK
HGBAIFLK
HJBFAGLI
HJBFAGIK
EJBAIHLK
EJBAIGLK
EJBAHGLK
EGBAIHLK
EJBAHGLI
EJBAHGIK
EJBAIFLK
EJBFAHLK
EIBFAHLK
EJBFAHLI
EJBFAHIK
EJBFAGLK
EGBAIFLK
EJBFAGLI
EJBFAGIK
EGBFAHLK
HJBFAGLE
HJBFAGEK
EGBFAHLI
EGBFAHIK
HJBFAGEI
IJBDAHLK
IJBDAGLK
HJBDAGLK
IGBDAHLK
HJBDAGLI
HJBDAGIK
IJBDAFLK
HJBDAFLK
HIBDAFLK
HJBDAFLI
HJBDAFIK
FJBDAGLK
IGBDAFLK
FJBDAGLI
FJBDAGIK
HGBDAFLK
HGBDAFLJ
HGBDAFJK
HGBDAFLI
HGBDAFIK
HGBDAFIJ
EJBAIDLK
EJBDAHLK
EIBDAHLK
EJBDAHLI
EJBDAHIK
EJBDAGLK
EGBAIDLK
EJBDAGLI
EJBDAGIK
EGBDAHLK
HJBDAGLE
HJBDAGEK
EGBDAHLI
EGBDAHIK
HJBDAGEI
EJBDAFLK
EIBDAFLK
EJBDAFLI
EJBDAFIK
HEBDAFLK
HJBDAFLE
HJBDAFEK
HEBDAFLI
HEBDAFIK
HJBDAFEI
EGBDAFLK
EGBDAFLJ
EGBDAFJK
EGBDAFLI
EGBDAFIK
EGBDAFIJ
HGBDAFLE
HGBDAFEK
HGBDAFEJ
HGBDAFEI
IJBCAHLK
IJBCAGLK
HJBCAGLK
IGBCAHLK
HJBCAGLI
HJBCAGIK
IJBCAFLK
HJBCAFLK
HIBCAFLK
HJBCAFLI
HJBCAFIK
CJBFAGLK
IGBCAFLK
CJBFAGLI
CJBFAGIK
HGBCAFLK
HGBCAFLJ
HGBCAFJK
HGBCAFLI
HGBCAFIK
HGBCAFIJ
EJBAICLK
EJBCAHLK
EIBCAHLK
EJBCAHLI
EJBCAHIK
EJBCAGLK
EGBAICLK
EJBCAGLI
EJBCAGIK
EGBCAHLK
HJBCAGLE
HJBCAGEK
EGBCAHLI
EGBCAHIK
HJBCAGEI
EJBCAFLK
EIBCAFLK
EJBCAFLI
EJBCAFIK
HEBCAFLK
HJBCAFLE
HJBCAFEK
HEBCAFLI
HEBCAFIK
HJBCAFEI
EGBCAFLK
EGBCAFLJ
EGBCAFJK
EGBCAFLI
EGBCAFIK
EGBCAFIJ
HGBCAFLE
HGBCAFEK
HGBCAFEJ
HGBCAFEI
IJBCADLK
HJBCADLK
HIBCADLK
HJBCADLI
HJBCADIK
CJBDAGLK
IGBCADLK
CJBDAGLI
CJBDAGIK
HGBCADLK
HGBCADLJ
HGBCADJK
HGBCADLI
HGBCADIK
HGBCADIJ
CJBDAFLK
CIBDAFLK
CJBDAFLI
CJBDAFIK
HFBCADLK
CJBDAFLH
HJBCAFDK
HFBCADLI
HFBCADIK
HJBCAFDI
CGBDAFLK
CGBDAFLJ
CGBDAFJK
CGBDAFLI
CGBDAFIK
CGBDAFIJ
CGBDAFLH
HGBCAFDK
HGBCAFDJ
HGBCAFDI
EJBCADLK
EIBCADLK
EJBCADLI
EJBCADIK
HEBCADLK
HJBCADLE
HJBCADEK
HEBCADLI
HEBCADIK
HJBCADEI
EGBCADLK
EGBCADLJ
EGBCADJK
EGBCADLI
EGBCADIK
EGBCADIJ
HGBCADLE
HGBCADEK
HGBCADEJ
HGBCADEI
CEBDAFLK
CJBDAFLE
CJBDAFEK
CEBDAFLI
CEBDAFIK
CJBDAFEI
HFBCADLE
HEBCAFDK
HJBCAFDE
HEBCAFDI
CGBDAFLE
CGBDAFEK
CGBDAFEJ
CGBDAFEI
HGBCAFDE
""".strip().splitlines()


class TournamentRuleError(RuntimeError):
    """Raised when the versioned tournament rules are internally inconsistent."""


def annex_c_table() -> Mapping[frozenset[str], Mapping[str, str]]:
    """Return the official Annex C mapping keyed by the eight qualifying third groups."""

    table: dict[frozenset[str], dict[str, str]] = {}
    for row in _ANNEX_C_ROWS:
        groups = tuple(row)
        key = frozenset(groups)
        assignment = dict(zip(ANNEX_C_SLOT_ORDER, groups, strict=True))
        if key in table:
            raise TournamentRuleError(f"duplicate Annex C combination: {''.join(sorted(key))}")
        table[key] = assignment
    _validate_annex_c_table(table)
    return table


def annex_c_assignment(qualified_third_groups: Iterable[str]) -> Mapping[str, str]:
    """Return the official third-place assignment for one qualifying group combination."""

    key = frozenset(_normalize_group(group) for group in qualified_third_groups)
    table = annex_c_table()
    try:
        return table[key]
    except KeyError as exc:
        groups = "".join(sorted(key))
        raise TournamentRuleError(f"unsupported Annex C third-place combination: {groups}") from exc


def team_slot(label: str) -> TeamSlot:
    """Parse labels such as ``1A`` or ``3L``."""

    if len(label) != 2 or label[0] not in {"1", "2", "3"}:
        raise TournamentRuleError(f"invalid team slot: {label}")
    return TeamSlot(group=_normalize_group(label[1]), position=int(label[0]))


def normalize_stage(stage: str | None) -> str | None:
    """Normalize provider and internal stage names to simulator names."""

    if stage is None:
        return None
    return STAGE_ALIASES.get(stage, stage.lower())


def _validate_annex_c_table(table: Mapping[frozenset[str], Mapping[str, str]]) -> None:
    if len(table) != 495:
        raise TournamentRuleError(f"Annex C must contain 495 combinations, got {len(table)}")
    for groups, assignment in table.items():
        if len(groups) != BEST_THIRD_QUALIFIERS:
            raise TournamentRuleError("Annex C combination must contain eight unique groups")
        if set(assignment) != set(ANNEX_C_SLOT_ORDER):
            raise TournamentRuleError("Annex C assignment has invalid slot columns")
        for slot, group in assignment.items():
            if group not in groups:
                raise TournamentRuleError("Annex C assignment references a non-qualifying group")
            if group not in THIRD_PLACE_SLOT_ELIGIBILITY[slot]:
                raise TournamentRuleError(f"Annex C assigns group {group} to invalid slot {slot}")


def _normalize_group(group: str) -> str:
    normalized = group.removeprefix("GROUP_").upper()
    if normalized not in GROUPS:
        raise TournamentRuleError(f"invalid group: {group}")
    return normalized
