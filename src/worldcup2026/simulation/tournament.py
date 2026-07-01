"""Monte Carlo tournament simulator for the FIFA World Cup 2026."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml  # type: ignore[import-untyped]

from worldcup2026.data.contracts import TeamAlias, resolve_team_alias
from worldcup2026.data.historical_ingest import load_team_aliases, load_team_catalog
from worldcup2026.data.world_cup_ingest import _football_data_after_extra_time
from worldcup2026.evaluation.dixon_coles_backtest import load_dixon_coles_config
from worldcup2026.models.dixon_coles import DixonColesGoalModel
from worldcup2026.pipelines.operational_predictions import (
    _assert_frozen_goal_selection,
    _fit_goal_model,
    _live_data_cutoff,
    _live_snapshot_checksum,
    _operational_training_rows,
    _read_parquet_rows,
    _selected_config_payload,
)
from worldcup2026.simulation.matches import (
    MatchSimulation,
    MatchSimulationError,
    ScoreMatrix,
    observed_knockout_match,
    score_matrix_from_mapping,
    score_matrix_from_prediction,
    simulate_group_score,
    simulate_knockout_match,
)
from worldcup2026.simulation.rules import (
    BEST_THIRD_QUALIFIERS,
    FINAL_TEMPLATE,
    GROUP_MATCHES_PER_GROUP,
    GROUPS,
    QUARTER_FINAL_TEMPLATES,
    ROUND_OF_16_TEMPLATES,
    ROUND_OF_32_FIXED_TEMPLATES,
    ROUND_OF_32_TEAMS,
    ROUND_OF_32_THIRD_PLACE_HOME_SLOTS,
    RULE_SOURCE_ACCESSED,
    RULE_SOURCE_URL,
    RULE_SOURCE_VERSION,
    RULE_VERSION,
    SEMI_FINAL_TEMPLATES,
    TEAMS_PER_GROUP,
    THIRD_PLACE_TEMPLATE,
    annex_c_assignment,
    annex_c_table,
    normalize_stage,
    team_slot,
)
from worldcup2026.simulation.standings import (
    GroupMatchResult,
    StandingRow,
    assign_group_classification,
    rank_best_thirds,
    rank_group,
)

SIMULATION_SCHEMA_VERSION = "world_cup_simulation_v1"
SIMULATION_OUTPUT_VERSION = "simulation_outputs_v2"
DEFAULT_CONFIG_PATH = Path("configs/simulation.yaml")
OFFICIAL_MODEL_VERSION = "poisson_goal_v1"
OFFICIAL_MODEL_FAMILY = "poisson"
PREDICTION_HISTORY_GLOB = "*.parquet"
RANKING_FIELDS = ("round_of_32", "round_of_16", "quarter_final", "semi_final", "final")
OBSERVED_FINALITY_BUFFER = timedelta(hours=3)


class TournamentSimulationError(RuntimeError):
    """Raised when the tournament simulation cannot proceed safely."""


@dataclass(frozen=True)
class SimulationConfig:
    """Versioned Monte Carlo configuration."""

    schema_version: str
    config_version: str
    default_runs: int
    seed: int
    chunk_size: int
    processes: int
    official_model_version: str
    rules_version: str
    extra_time_goal_scale: float
    output_root: Path


@dataclass(frozen=True)
class TournamentFixture:
    """Provider fixture normalized for simulation."""

    source_fixture_id: str
    stage: str
    group: str | None
    kickoff_utc: datetime
    status: str
    source_status: str
    home_team_id: str | None
    away_team_id: str | None
    home_team_name: str | None
    away_team_name: str | None
    home_goals_90: int | None
    away_goals_90: int | None
    home_goals_after_extra_time: int | None
    away_goals_after_extra_time: int | None
    home_penalty_goals: int | None
    away_penalty_goals: int | None


@dataclass(frozen=True)
class TournamentState:
    """All fixture and metadata inputs needed for a simulation run."""

    fixtures: tuple[TournamentFixture, ...]
    data_cutoff_utc: datetime
    snapshot_checksum: str
    snapshot_reference: str
    raw_snapshot_path: Path | None
    source_fixture_count: int


@dataclass(frozen=True)
class SimulationRunResult:
    """Summary of generated simulation outputs."""

    simulation_run_id: str
    manifest_path: Path
    latest_root: Path
    history_root: Path
    runs: int
    seed: int
    cutoff: datetime
    elapsed_seconds: float
    favorite_team_id: str | None
    favorite_champion_probability: float | None
    fallback_counts: Mapping[str, int]
    warnings: tuple[str, ...]


class PredictionStore:
    """Lookup persisted official score matrices by source fixture."""

    def __init__(self, rows: Sequence[Mapping[str, object]]) -> None:
        self._rows_by_fixture: dict[str, list[Mapping[str, object]]] = {}
        for row in rows:
            if str(row.get("model_version") or "") != OFFICIAL_MODEL_VERSION:
                continue
            if str(row.get("model_family") or "") not in {"", OFFICIAL_MODEL_FAMILY}:
                continue
            if not isinstance(row.get("score_probabilities_json"), str):
                continue
            source_fixture_id = _optional_str(row.get("source_fixture_id"))
            if source_fixture_id is None:
                continue
            self._rows_by_fixture.setdefault(source_fixture_id, []).append(row)

    @classmethod
    def from_predictions_root(cls, predictions_root: Path) -> PredictionStore:
        rows: list[Mapping[str, object]] = []
        latest = predictions_root / "latest.csv"
        if latest.is_file():
            rows.extend(_read_csv_rows(latest))
        history = predictions_root / "history"
        if history.is_dir():
            for path in sorted(history.glob(PREDICTION_HISTORY_GLOB)):
                rows.extend(_read_parquet_rows(path))
        published = predictions_root / "published-history" / "history"
        if published.is_dir():
            for path in sorted(published.glob("*.csv.gz")):
                rows.extend(_read_gzip_csv_rows(path))
        return cls(rows)

    def score_matrix_for_fixture(
        self,
        fixture: TournamentFixture,
        *,
        cutoff: datetime,
        allow_in_progress_snapshot: bool,
    ) -> ScoreMatrix | None:
        rows = self._rows_by_fixture.get(fixture.source_fixture_id, [])
        compatible: list[Mapping[str, object]] = []
        for row in rows:
            if _optional_str(row.get("home_team_id")) != fixture.home_team_id:
                continue
            if _optional_str(row.get("away_team_id")) != fixture.away_team_id:
                continue
            row_cutoff = _optional_datetime(row.get("data_cutoff_utc"))
            created_at = _optional_datetime(row.get("prediction_created_at_utc"))
            kickoff = _optional_datetime(row.get("kickoff_utc"))
            if row_cutoff is None or created_at is None or kickoff is None:
                continue
            if row_cutoff > cutoff:
                continue
            if created_at >= kickoff:
                continue
            if allow_in_progress_snapshot and row_cutoff >= kickoff:
                continue
            compatible.append(row)
        if not compatible:
            return None
        selected = max(
            compatible,
            key=lambda row: (
                _optional_datetime(row.get("data_cutoff_utc")) or datetime.min.replace(tzinfo=UTC),
                _optional_datetime(row.get("prediction_created_at_utc"))
                or datetime.min.replace(tzinfo=UTC),
            ),
        )
        return score_matrix_from_prediction(selected)


class OfficialScoreProvider:
    """Official Poisson goal-model score provider for hypothetical fixtures."""

    def __init__(self, model: DixonColesGoalModel) -> None:
        self.model = model

    @classmethod
    def fit_from_inputs(
        cls,
        *,
        model_config_path: Path,
        modeling_matches_path: Path,
        live_matches_path: Path,
        ingest_report_path: Path,
        cutoff: datetime,
    ) -> OfficialScoreProvider:
        goal_config = load_dixon_coles_config(model_config_path)
        _assert_frozen_goal_selection(goal_config)
        historical_rows = _read_parquet_rows(modeling_matches_path)
        live_rows = _read_parquet_rows(live_matches_path)
        training_rows, _ = _operational_training_rows(historical_rows, live_rows, cutoff=cutoff)
        del ingest_report_path
        model = _fit_goal_model(training_rows, goal_config=goal_config, cutoff=cutoff)
        return cls(model)

    def score_matrix(self, home_team_id: str, away_team_id: str) -> ScoreMatrix:
        distribution = self.model.predict_match(
            {
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "competition_category": "world_cup",
                "home_advantage_eligible": False,
            }
        )
        return score_matrix_from_mapping(
            dict(distribution.score_probabilities),
            expected_home_goals=distribution.expected_home_goals,
            expected_away_goals=distribution.expected_away_goals,
            source="as_of_official_model",
        )


def load_simulation_config(path: Path = DEFAULT_CONFIG_PATH) -> SimulationConfig:
    """Load the versioned simulation configuration."""

    if not path.is_file():
        raise TournamentSimulationError(f"simulation config is missing: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TournamentSimulationError("simulation config must contain a mapping")
    raw = payload.get("simulation")
    if not isinstance(raw, Mapping):
        raise TournamentSimulationError("simulation config must contain `simulation`")
    config = SimulationConfig(
        schema_version=_required_str(raw, "schema_version"),
        config_version=_required_str(raw, "config_version"),
        default_runs=_positive_int(raw.get("default_runs"), field="default_runs"),
        seed=_non_negative_int(raw.get("seed"), field="seed"),
        chunk_size=_positive_int(raw.get("chunk_size"), field="chunk_size"),
        processes=_positive_int(raw.get("processes"), field="processes"),
        official_model_version=_required_str(raw, "official_model_version"),
        rules_version=_required_str(raw, "rules_version"),
        extra_time_goal_scale=_positive_float(
            raw.get("extra_time_goal_scale"),
            field="extra_time_goal_scale",
            allow_zero=True,
        ),
        output_root=Path(_required_str(raw, "output_root")),
    )
    if config.official_model_version != OFFICIAL_MODEL_VERSION:
        raise TournamentSimulationError("simulator official model must remain poisson_goal_v1")
    if config.rules_version != RULE_VERSION:
        raise TournamentSimulationError(f"unsupported simulation rules: {config.rules_version}")
    if config.processes != 1:
        raise TournamentSimulationError("parallel simulation is not enabled for v1")
    return config


def run_tournament_simulation(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    runs: int | None = None,
    seed: int | None = None,
    as_of: datetime | None = None,
    offline_fixture: Path | None = None,
    official_model: bool = True,
    shadow_contextual: bool = False,
    model_config_path: Path = Path("configs/model.yaml"),
    modeling_matches_path: Path = Path("data/processed/modeling_matches.parquet"),
    live_matches_path: Path = Path("data/processed/world_cup_2026/matches.parquet"),
    ingest_report_path: Path = Path("data/interim/world_cup_2026_ingest_report.json"),
    predictions_root: Path = Path("predictions"),
    teams_path: Path = Path("data/static/teams.csv"),
    aliases_path: Path = Path("data/static/team_aliases.csv"),
    output_root: Path | None = None,
    generated_at: datetime | None = None,
) -> SimulationRunResult:
    """Run deterministic tournament Monte Carlo and write official simulation outputs."""

    if not official_model:
        raise TournamentSimulationError("official tournament simulation requires poisson_goal_v1")
    if shadow_contextual:
        raise TournamentSimulationError(
            "full contextual shadow tournament simulation is not valid for hypothetical fixtures"
        )
    config = load_simulation_config(config_path)
    run_count = config.default_runs if runs is None else runs
    if run_count < 0:
        raise TournamentSimulationError("runs cannot be negative")
    run_seed = config.seed if seed is None else seed
    if run_seed < 0:
        raise TournamentSimulationError("seed cannot be negative")
    generated = _utc_now() if generated_at is None else _require_utc(generated_at)
    state = load_tournament_state(
        live_matches_path=live_matches_path,
        ingest_report_path=ingest_report_path,
        raw_fixture_path=offline_fixture,
        teams_path=teams_path,
        aliases_path=aliases_path,
    )
    cutoff = state.data_cutoff_utc if as_of is None else _require_utc(as_of)
    if as_of is not None:
        state = _state_as_of(state, cutoff=cutoff)
    prediction_store = PredictionStore.from_predictions_root(predictions_root)
    score_provider = OfficialScoreProvider.fit_from_inputs(
        model_config_path=model_config_path,
        modeling_matches_path=modeling_matches_path,
        live_matches_path=live_matches_path,
        ingest_report_path=ingest_report_path,
        cutoff=cutoff,
    )
    active_output_root = config.output_root if output_root is None else output_root
    started = time.perf_counter()
    result = _simulate_many(
        state=state,
        prediction_store=prediction_store,
        score_provider=score_provider,
        runs=run_count,
        seed=run_seed,
        chunk_size=config.chunk_size,
        extra_time_goal_scale=config.extra_time_goal_scale,
        cutoff=cutoff,
    )
    elapsed = time.perf_counter() - started
    input_checksum = _simulation_input_checksum(
        state=state,
        config=config,
        runs=run_count,
        seed=run_seed,
        cutoff=cutoff,
    )
    run_id = _simulation_run_id(
        state=state,
        config=config,
        runs=run_count,
        seed=run_seed,
        cutoff=cutoff,
        input_checksum=input_checksum,
    )
    latest_root = active_output_root / "latest"
    history_root = active_output_root / "history" / run_id
    paths = _write_outputs(
        result,
        latest_root=latest_root,
        history_root=history_root,
        state=state,
        config=config,
        runs=run_count,
        seed=run_seed,
        cutoff=cutoff,
        generated_at=generated,
        elapsed_seconds=elapsed,
        run_id=run_id,
        input_checksum=input_checksum,
        model_config_payload=_selected_config_payload(load_dixon_coles_config(model_config_path)),
    )
    favorite = result.champion_top10[0] if result.champion_top10 else None
    return SimulationRunResult(
        simulation_run_id=run_id,
        manifest_path=paths["manifest"],
        latest_root=latest_root,
        history_root=history_root,
        runs=run_count,
        seed=run_seed,
        cutoff=cutoff,
        elapsed_seconds=elapsed,
        favorite_team_id=None if favorite is None else str(favorite["team_id"]),
        favorite_champion_probability=None
        if favorite is None
        else _object_float(favorite["champion_probability"]),
        fallback_counts=result.fallback_counts,
        warnings=result.warnings,
    )


def load_tournament_state(
    *,
    live_matches_path: Path,
    ingest_report_path: Path,
    raw_fixture_path: Path | None,
    teams_path: Path,
    aliases_path: Path,
) -> TournamentState:
    """Load current tournament fixtures from the raw provider snapshot."""

    report = _read_json(ingest_report_path)
    live_rows = _read_parquet_rows(live_matches_path) if live_matches_path.is_file() else []
    cutoff = _live_data_cutoff(live_rows, ingest_report_path=ingest_report_path)
    snapshot_checksum = _live_snapshot_checksum(
        live_matches_path,
        ingest_report_path=ingest_report_path,
    )
    snapshot_reference = str(report.get("snapshot_reference") or snapshot_checksum[:12])
    raw_path = raw_fixture_path or _collection_raw_path(report)
    if raw_path is None:
        raise TournamentSimulationError("raw World Cup fixture snapshot is required")
    fixtures = _read_provider_fixtures(
        raw_path,
        teams_path=teams_path,
        aliases_path=aliases_path,
    )
    if not fixtures:
        raise TournamentSimulationError("tournament snapshot contains no fixtures")
    return TournamentState(
        fixtures=fixtures,
        data_cutoff_utc=cutoff,
        snapshot_checksum=snapshot_checksum,
        snapshot_reference=snapshot_reference,
        raw_snapshot_path=raw_path,
        source_fixture_count=len(fixtures),
    )


def _state_as_of(state: TournamentState, *, cutoff: datetime) -> TournamentState:
    """Return an as-of view that clears results and future knockout participants after cutoff."""

    return replace(
        state,
        data_cutoff_utc=cutoff,
        fixtures=tuple(_fixture_as_of(fixture, cutoff=cutoff) for fixture in state.fixtures),
    )


def _fixture_as_of(fixture: TournamentFixture, *, cutoff: datetime) -> TournamentFixture:
    if fixture.kickoff_utc > cutoff:
        return _clear_unobserved_fixture(
            fixture,
            status="scheduled",
            clear_participants=fixture.stage != "group_stage",
        )
    if fixture.status == "played" and cutoff < fixture.kickoff_utc + OBSERVED_FINALITY_BUFFER:
        return _clear_unobserved_fixture(
            fixture,
            status="in_progress",
            clear_participants=False,
        )
    if fixture.status == "in_progress" and cutoff < fixture.kickoff_utc:
        return _clear_unobserved_fixture(
            fixture,
            status="scheduled",
            clear_participants=fixture.stage != "group_stage",
        )
    return fixture


def _clear_unobserved_fixture(
    fixture: TournamentFixture,
    *,
    status: str,
    clear_participants: bool,
) -> TournamentFixture:
    return replace(
        fixture,
        status=status,
        home_team_id=None if clear_participants else fixture.home_team_id,
        away_team_id=None if clear_participants else fixture.away_team_id,
        home_team_name=None if clear_participants else fixture.home_team_name,
        away_team_name=None if clear_participants else fixture.away_team_name,
        home_goals_90=None,
        away_goals_90=None,
        home_goals_after_extra_time=None,
        away_goals_after_extra_time=None,
        home_penalty_goals=None,
        away_penalty_goals=None,
    )


@dataclass(frozen=True)
class _SimulationAggregate:
    team_rows: tuple[dict[str, object], ...]
    group_rows: tuple[dict[str, object], ...]
    group_summary_md: str
    round_probabilities_md: str
    champion_probabilities_md: str
    bracket_summary_md: str
    manifest_payload: Mapping[str, object]
    simulation_rows: tuple[dict[str, object], ...]
    champion_top10: tuple[dict[str, object], ...]
    fallback_counts: Mapping[str, int]
    warnings: tuple[str, ...]


def _simulate_many(
    *,
    state: TournamentState,
    prediction_store: PredictionStore,
    score_provider: OfficialScoreProvider,
    runs: int,
    seed: int,
    chunk_size: int,
    extra_time_goal_scale: float,
    cutoff: datetime,
) -> _SimulationAggregate:
    rng = np.random.default_rng(seed)
    teams = sorted(_team_ids_from_group_stage(state.fixtures))
    if len(teams) != 48:
        raise TournamentSimulationError(f"expected 48 tournament teams, found {len(teams)}")
    counts = _initial_counts(teams)
    group_position_counts: Counter[tuple[str, str, int]] = Counter()
    group_status_counts: Counter[tuple[str, str]] = Counter()
    bracket_pair_counts: Counter[str] = Counter()
    fallback_counts: Counter[str] = Counter()
    simulation_rows: list[dict[str, object]] = []
    warnings: set[str] = set()
    if runs == 0:
        return _aggregate_outputs(
            teams=teams,
            counts=counts,
            group_position_counts=group_position_counts,
            group_status_counts=group_status_counts,
            bracket_pair_counts=bracket_pair_counts,
            simulation_rows=(),
            fallback_counts=fallback_counts,
            warnings=("zero simulations requested",),
            runs=runs,
        )

    for chunk_start in range(0, runs, chunk_size):
        chunk_end = min(runs, chunk_start + chunk_size)
        for run_index in range(chunk_start, chunk_end):
            run_result = _simulate_one(
                state=state,
                prediction_store=prediction_store,
                score_provider=score_provider,
                rng=rng,
                extra_time_goal_scale=extra_time_goal_scale,
                cutoff=cutoff,
            )
            _update_counts(counts, run_result)
            group_position_counts.update(run_result.group_position_counts)
            group_status_counts.update(run_result.group_status_counts)
            bracket_pair_counts.update(run_result.bracket_pair_counts)
            fallback_counts.update(run_result.fallback_counts)
            warnings.update(run_result.warnings)
            simulation_rows.append(
                {
                    "run_index": run_index,
                    "champion_team_id": run_result.champion_team_id,
                    "runner_up_team_id": run_result.runner_up_team_id,
                    "third_place_team_id": run_result.third_place_team_id,
                    "fourth_place_team_id": run_result.fourth_place_team_id,
                    "random_lot_proxy_count": run_result.fallback_counts.get(
                        "random_lot_proxy",
                        0,
                    ),
                }
            )

    return _aggregate_outputs(
        teams=teams,
        counts=counts,
        group_position_counts=group_position_counts,
        group_status_counts=group_status_counts,
        bracket_pair_counts=bracket_pair_counts,
        simulation_rows=tuple(simulation_rows),
        fallback_counts=fallback_counts,
        warnings=tuple(sorted(warnings)),
        runs=runs,
    )


@dataclass(frozen=True)
class _OneRunResult:
    group_position_counts: Counter[tuple[str, str, int]]
    group_status_counts: Counter[tuple[str, str]]
    bracket_pair_counts: Counter[str]
    fallback_counts: Counter[str]
    warnings: tuple[str, ...]
    round_of_32: frozenset[str]
    round_of_16: frozenset[str]
    quarter_final: frozenset[str]
    semi_final: frozenset[str]
    final: frozenset[str]
    champion_team_id: str
    runner_up_team_id: str
    third_place_team_id: str
    fourth_place_team_id: str


def _simulate_one(
    *,
    state: TournamentState,
    prediction_store: PredictionStore,
    score_provider: OfficialScoreProvider,
    rng: np.random.Generator,
    extra_time_goal_scale: float,
    cutoff: datetime,
) -> _OneRunResult:
    group_matches, warnings = _group_match_results(
        state=state,
        prediction_store=prediction_store,
        score_provider=score_provider,
        rng=rng,
        cutoff=cutoff,
    )
    group_teams = _group_teams(state.fixtures)
    fallback_counts: Counter[str] = Counter()
    standings_by_group: dict[str, tuple[StandingRow, ...]] = {}
    group_position_counts: Counter[tuple[str, str, int]] = Counter()
    for group in GROUPS:
        result = rank_group(
            group=group,
            teams=sorted(group_teams[group]),
            matches=group_matches,
            rng=rng,
        )
        fallback_counts.update(result.fallback_counts)
        standings_by_group[group] = result.rows
    third_result = rank_best_thirds(
        [standings_by_group[group][2] for group in GROUPS],
        qualifiers=BEST_THIRD_QUALIFIERS,
        rng=rng,
    )
    fallback_counts.update(third_result.fallback_counts)
    best_third_team_ids = {row.team_id for row in third_result.rows[:BEST_THIRD_QUALIFIERS]}
    classified_rows: dict[str, StandingRow] = {}
    group_status_counts: Counter[tuple[str, str]] = Counter()
    for group, rows in standings_by_group.items():
        for row in assign_group_classification(rows, best_third_team_ids):
            classified_rows[row.team_id] = row
            group_position_counts[(group, row.team_id, row.position)] += 1
            group_status_counts[(row.team_id, row.classification_status)] += 1

    round_of_32_matches = _build_round_of_32(standings_by_group, third_result.rows)
    _validate_round_of_32(round_of_32_matches, classified_rows)
    winners: dict[int, str] = {}
    losers: dict[int, str] = {}
    bracket_pair_counts: Counter[str] = Counter()
    round_of_32_teams: set[str] = set()
    round_of_16_teams: set[str] = set()
    quarter_final_teams: set[str] = set()
    semi_final_teams: set[str] = set()
    final_teams: set[str] = set()

    for match_number, home, away in round_of_32_matches:
        match_result = _play_knockout(
            match_number=match_number,
            stage="round_of_32",
            home=home,
            away=away,
            state=state,
            prediction_store=prediction_store,
            score_provider=score_provider,
            rng=rng,
            cutoff=cutoff,
            extra_time_goal_scale=extra_time_goal_scale,
        )
        winners[match_number] = match_result.advancing_team_id
        losers[match_number] = match_result.losing_team_id
        round_of_32_teams.update((home, away))
        round_of_16_teams.add(match_result.advancing_team_id)
        bracket_pair_counts[_pair_key("round_of_32", home, away)] += 1

    for template in ROUND_OF_16_TEMPLATES:
        home = _resolve_dependency(template.home, winners, losers)
        away = _resolve_dependency(template.away, winners, losers)
        match_result = _play_knockout(
            match_number=template.match_number,
            stage=template.stage,
            home=home,
            away=away,
            state=state,
            prediction_store=prediction_store,
            score_provider=score_provider,
            rng=rng,
            cutoff=cutoff,
            extra_time_goal_scale=extra_time_goal_scale,
        )
        winners[template.match_number] = match_result.advancing_team_id
        losers[template.match_number] = match_result.losing_team_id
        quarter_final_teams.add(match_result.advancing_team_id)
        bracket_pair_counts[_pair_key(template.stage, home, away)] += 1

    for template in QUARTER_FINAL_TEMPLATES:
        home = _resolve_dependency(template.home, winners, losers)
        away = _resolve_dependency(template.away, winners, losers)
        match_result = _play_knockout(
            match_number=template.match_number,
            stage=template.stage,
            home=home,
            away=away,
            state=state,
            prediction_store=prediction_store,
            score_provider=score_provider,
            rng=rng,
            cutoff=cutoff,
            extra_time_goal_scale=extra_time_goal_scale,
        )
        winners[template.match_number] = match_result.advancing_team_id
        losers[template.match_number] = match_result.losing_team_id
        semi_final_teams.add(match_result.advancing_team_id)
        bracket_pair_counts[_pair_key(template.stage, home, away)] += 1

    for template in SEMI_FINAL_TEMPLATES:
        home = _resolve_dependency(template.home, winners, losers)
        away = _resolve_dependency(template.away, winners, losers)
        match_result = _play_knockout(
            match_number=template.match_number,
            stage=template.stage,
            home=home,
            away=away,
            state=state,
            prediction_store=prediction_store,
            score_provider=score_provider,
            rng=rng,
            cutoff=cutoff,
            extra_time_goal_scale=extra_time_goal_scale,
        )
        winners[template.match_number] = match_result.advancing_team_id
        losers[template.match_number] = match_result.losing_team_id
        final_teams.add(match_result.advancing_team_id)
        bracket_pair_counts[_pair_key(template.stage, home, away)] += 1

    third_home = _resolve_dependency(THIRD_PLACE_TEMPLATE.home, winners, losers)
    third_away = _resolve_dependency(THIRD_PLACE_TEMPLATE.away, winners, losers)
    third_result_match = _play_knockout(
        match_number=THIRD_PLACE_TEMPLATE.match_number,
        stage=THIRD_PLACE_TEMPLATE.stage,
        home=third_home,
        away=third_away,
        state=state,
        prediction_store=prediction_store,
        score_provider=score_provider,
        rng=rng,
        cutoff=cutoff,
        extra_time_goal_scale=extra_time_goal_scale,
    )
    final_home = _resolve_dependency(FINAL_TEMPLATE.home, winners, losers)
    final_away = _resolve_dependency(FINAL_TEMPLATE.away, winners, losers)
    final_result_match = _play_knockout(
        match_number=FINAL_TEMPLATE.match_number,
        stage=FINAL_TEMPLATE.stage,
        home=final_home,
        away=final_away,
        state=state,
        prediction_store=prediction_store,
        score_provider=score_provider,
        rng=rng,
        cutoff=cutoff,
        extra_time_goal_scale=extra_time_goal_scale,
    )
    return _OneRunResult(
        group_position_counts=group_position_counts,
        group_status_counts=group_status_counts,
        bracket_pair_counts=bracket_pair_counts,
        fallback_counts=fallback_counts,
        warnings=tuple(sorted(warnings)),
        round_of_32=frozenset(round_of_32_teams),
        round_of_16=frozenset(round_of_16_teams),
        quarter_final=frozenset(quarter_final_teams),
        semi_final=frozenset(semi_final_teams),
        final=frozenset(final_teams),
        champion_team_id=final_result_match.advancing_team_id,
        runner_up_team_id=final_result_match.losing_team_id,
        third_place_team_id=third_result_match.advancing_team_id,
        fourth_place_team_id=third_result_match.losing_team_id,
    )


def _group_match_results(
    *,
    state: TournamentState,
    prediction_store: PredictionStore,
    score_provider: OfficialScoreProvider,
    rng: np.random.Generator,
    cutoff: datetime,
) -> tuple[list[GroupMatchResult], set[str]]:
    results: list[GroupMatchResult] = []
    warnings: set[str] = set()
    for fixture in state.fixtures:
        if fixture.stage != "group_stage":
            continue
        if fixture.group is None:
            raise TournamentSimulationError("group-stage fixture is missing group")
        home = _required_team(fixture.home_team_id, fixture=fixture)
        away = _required_team(fixture.away_team_id, fixture=fixture)
        if fixture.status == "played":
            home_goals = _required_int(fixture.home_goals_90, "home_goals_90", fixture)
            away_goals = _required_int(fixture.away_goals_90, "away_goals_90", fixture)
            observed = True
        elif fixture.status == "in_progress":
            matrix = prediction_store.score_matrix_for_fixture(
                fixture,
                cutoff=cutoff,
                allow_in_progress_snapshot=True,
            )
            if matrix is None:
                matrix = score_provider.score_matrix(home, away)
                warnings.add("in_progress_fixture_used_as_of_model_without_pregame_snapshot")
            else:
                warnings.add("in_progress_fixture_used_last_pregame_prediction")
            home_goals, away_goals = simulate_group_score(matrix, rng)
            observed = False
        elif fixture.status == "scheduled":
            matrix = prediction_store.score_matrix_for_fixture(
                fixture,
                cutoff=cutoff,
                allow_in_progress_snapshot=False,
            )
            if matrix is None:
                matrix = score_provider.score_matrix(home, away)
            home_goals, away_goals = simulate_group_score(matrix, rng)
            observed = False
        else:
            continue
        results.append(
            GroupMatchResult(
                group=fixture.group,
                home_team_id=home,
                away_team_id=away,
                home_goals=home_goals,
                away_goals=away_goals,
                source_fixture_id=fixture.source_fixture_id,
                observed=observed,
            )
        )
    expected = len(GROUPS) * GROUP_MATCHES_PER_GROUP
    if len(results) != expected:
        raise TournamentSimulationError(f"expected {expected} group matches, got {len(results)}")
    return results, warnings


def _build_round_of_32(
    standings_by_group: Mapping[str, Sequence[StandingRow]],
    third_ranking: Sequence[StandingRow],
) -> list[tuple[int, str, str]]:
    positions: dict[tuple[str, int], str] = {}
    for group, rows in standings_by_group.items():
        for row in rows:
            positions[(group, row.position)] = row.team_id
    best_third_groups = [row.group for row in third_ranking[:BEST_THIRD_QUALIFIERS]]
    assignment = annex_c_assignment(best_third_groups)
    matches: dict[int, tuple[str, str]] = {}
    for template in ROUND_OF_32_FIXED_TEMPLATES:
        home_slot = team_slot(template.home)
        away_slot = team_slot(template.away)
        matches[template.match_number] = (
            positions[(home_slot.group, home_slot.position)],
            positions[(away_slot.group, away_slot.position)],
        )
    for match_number, home_label in ROUND_OF_32_THIRD_PLACE_HOME_SLOTS.items():
        home_slot = team_slot(home_label)
        third_group = assignment[home_label]
        matches[match_number] = (
            positions[(home_slot.group, home_slot.position)],
            positions[(third_group, 3)],
        )
    return [(match_number, *matches[match_number]) for match_number in range(73, 89)]


def _validate_round_of_32(
    matches: Sequence[tuple[int, str, str]],
    classified_rows: Mapping[str, StandingRow],
) -> None:
    if len(matches) != 16:
        raise TournamentSimulationError("round of 32 must contain 16 fixtures")
    teams = [team for _, home, away in matches for team in (home, away)]
    if len(teams) != ROUND_OF_32_TEAMS or len(set(teams)) != ROUND_OF_32_TEAMS:
        raise TournamentSimulationError("round of 32 must contain 32 unique teams")
    qualifiers = {
        row.team_id
        for row in classified_rows.values()
        if row.classification_status in {"direct", "best_third"}
    }
    if set(teams) != qualifiers:
        raise TournamentSimulationError("round of 32 teams do not match group qualifiers")
    eliminated = {
        row.team_id
        for row in classified_rows.values()
        if row.classification_status == "group_eliminated"
    }
    if set(teams) & eliminated:
        raise TournamentSimulationError("round of 32 contains a group-eliminated team")


def _play_knockout(
    *,
    match_number: int,
    stage: str,
    home: str,
    away: str,
    state: TournamentState,
    prediction_store: PredictionStore,
    score_provider: OfficialScoreProvider,
    rng: np.random.Generator,
    cutoff: datetime,
    extra_time_goal_scale: float,
) -> MatchSimulation:
    fixture = _find_fixture(state.fixtures, stage=stage, home=home, away=away)
    if fixture is not None and fixture.status == "played":
        return observed_knockout_match(
            match_number=match_number,
            source_fixture_id=fixture.source_fixture_id,
            stage=stage,
            home_team_id=home,
            away_team_id=away,
            home_goals_90=_required_int(fixture.home_goals_90, "home_goals_90", fixture),
            away_goals_90=_required_int(fixture.away_goals_90, "away_goals_90", fixture),
            home_goals_after_extra_time=fixture.home_goals_after_extra_time,
            away_goals_after_extra_time=fixture.away_goals_after_extra_time,
            home_penalty_goals=fixture.home_penalty_goals,
            away_penalty_goals=fixture.away_penalty_goals,
        )
    matrix: ScoreMatrix | None = None
    source_fixture_id: str | None = None
    if fixture is not None:
        source_fixture_id = fixture.source_fixture_id
        matrix = prediction_store.score_matrix_for_fixture(
            fixture,
            cutoff=cutoff,
            allow_in_progress_snapshot=fixture.status == "in_progress",
        )
    if matrix is None:
        matrix = score_provider.score_matrix(home, away)
    try:
        return simulate_knockout_match(
            match_number=match_number,
            source_fixture_id=source_fixture_id,
            stage=stage,
            home_team_id=home,
            away_team_id=away,
            matrix=matrix,
            rng=rng,
            extra_time_goal_scale=extra_time_goal_scale,
        )
    except MatchSimulationError as exc:
        raise TournamentSimulationError(str(exc)) from exc


def _aggregate_outputs(
    *,
    teams: Sequence[str],
    counts: Mapping[str, Counter[str]],
    group_position_counts: Counter[tuple[str, str, int]],
    group_status_counts: Counter[tuple[str, str]],
    bracket_pair_counts: Counter[str],
    simulation_rows: Sequence[Mapping[str, object]],
    fallback_counts: Mapping[str, int],
    warnings: Sequence[str],
    runs: int,
) -> _SimulationAggregate:
    denominator = float(runs) if runs else 1.0
    team_rows: list[dict[str, object]] = []
    for team in teams:
        row = {
            "team_id": team,
            "finish_group_1": counts["group_1"][team] / denominator,
            "finish_group_2": counts["group_2"][team] / denominator,
            "finish_group_3": counts["group_3"][team] / denominator,
            "finish_group_4": counts["group_4"][team] / denominator,
            "classify_direct": counts["direct"][team] / denominator,
            "classify_best_third": counts["best_third"][team] / denominator,
            "eliminated_group": counts["group_eliminated"][team] / denominator,
            "round_of_32": counts["round_of_32"][team] / denominator,
            "round_of_16": counts["round_of_16"][team] / denominator,
            "quarter_final": counts["quarter_final"][team] / denominator,
            "semi_final": counts["semi_final"][team] / denominator,
            "final": counts["final"][team] / denominator,
            "third_place": counts["third_place"][team] / denominator,
            "fourth_place": counts["fourth_place"][team] / denominator,
            "runner_up": counts["runner_up"][team] / denominator,
            "champion": counts["champion"][team] / denominator,
        }
        _assert_monotonic(row)
        team_rows.append(row)
    champion_top10 = tuple(
        {
            "team_id": row["team_id"],
            "champion_probability": row["champion"],
        }
        for row in sorted(
            team_rows,
            key=lambda item: _object_float(item["champion"]),
            reverse=True,
        )[:10]
    )
    group_rows = _group_probability_rows(
        group_position_counts=group_position_counts,
        group_status_counts=group_status_counts,
        runs=runs,
    )
    return _SimulationAggregate(
        team_rows=tuple(team_rows),
        group_rows=tuple(group_rows),
        group_summary_md=_group_summary_markdown(group_rows),
        round_probabilities_md=_round_probabilities_markdown(team_rows),
        champion_probabilities_md=_champion_probabilities_markdown(team_rows),
        bracket_summary_md=_bracket_summary_markdown(bracket_pair_counts, runs=runs),
        manifest_payload={},
        simulation_rows=tuple(dict(row) for row in simulation_rows),
        champion_top10=champion_top10,
        fallback_counts=dict(fallback_counts),
        warnings=tuple(warnings),
    )


def _initial_counts(teams: Sequence[str]) -> dict[str, Counter[str]]:
    keys = (
        "group_1",
        "group_2",
        "group_3",
        "group_4",
        "direct",
        "best_third",
        "group_eliminated",
        "round_of_32",
        "round_of_16",
        "quarter_final",
        "semi_final",
        "final",
        "third_place",
        "fourth_place",
        "runner_up",
        "champion",
    )
    return {key: Counter({team: 0 for team in teams}) for key in keys}


def _update_counts(counts: Mapping[str, Counter[str]], result: _OneRunResult) -> None:
    for group, team, position in result.group_position_counts:
        counts[f"group_{position}"][team] += result.group_position_counts[(group, team, position)]
    for team, status in result.group_status_counts:
        counts[status][team] += result.group_status_counts[(team, status)]
    for team in result.round_of_32:
        counts["round_of_32"][team] += 1
    for team in result.round_of_16:
        counts["round_of_16"][team] += 1
    for team in result.quarter_final:
        counts["quarter_final"][team] += 1
    for team in result.semi_final:
        counts["semi_final"][team] += 1
    for team in result.final:
        counts["final"][team] += 1
    counts["third_place"][result.third_place_team_id] += 1
    counts["fourth_place"][result.fourth_place_team_id] += 1
    counts["runner_up"][result.runner_up_team_id] += 1
    counts["champion"][result.champion_team_id] += 1


def _write_outputs(
    aggregate: _SimulationAggregate,
    *,
    latest_root: Path,
    history_root: Path,
    state: TournamentState,
    config: SimulationConfig,
    runs: int,
    seed: int,
    cutoff: datetime,
    generated_at: datetime,
    elapsed_seconds: float,
    run_id: str,
    input_checksum: str,
    model_config_payload: Mapping[str, object],
) -> Mapping[str, Path]:
    latest_root.mkdir(parents=True, exist_ok=True)
    history_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "team_csv": latest_root / "team_probabilities.csv",
        "team_json": latest_root / "team_probabilities.json",
        "group_csv": latest_root / "group_probabilities.csv",
        "group_summary": latest_root / "group_tables_summary.md",
        "rounds": latest_root / "round_probabilities.md",
        "champions": latest_root / "champion_probabilities.md",
        "bracket": latest_root / "bracket_summary.md",
        "stability": latest_root / "stability_report.md",
        "manifest": latest_root / "manifest.json",
        "parquet": latest_root / "simulation_results.parquet",
    }
    _write_csv(paths["team_csv"], aggregate.team_rows)
    _write_json(
        paths["team_json"],
        {
            "schema_version": "team_simulation_probabilities_v1",
            "simulation_run_id": run_id,
            "generated_at": _format_utc(generated_at),
            "data_cutoff_utc": _format_utc(cutoff),
            "runs": runs,
            "seed": seed,
            "teams": list(aggregate.team_rows),
        },
    )
    _write_csv(paths["group_csv"], aggregate.group_rows)
    paths["group_summary"].write_text(
        _report_header(cutoff, runs, seed) + aggregate.group_summary_md
    )
    paths["rounds"].write_text(
        _report_header(cutoff, runs, seed) + aggregate.round_probabilities_md
    )
    paths["champions"].write_text(
        _report_header(cutoff, runs, seed) + aggregate.champion_probabilities_md
    )
    paths["bracket"].write_text(_report_header(cutoff, runs, seed) + aggregate.bracket_summary_md)
    paths["stability"].write_text(
        _report_header(cutoff, runs, seed)
        + _stability_report_markdown(
            aggregate.team_rows,
            runs=runs,
            elapsed_seconds=elapsed_seconds,
        )
    )
    _write_parquet(paths["parquet"], aggregate.simulation_rows)
    fixture_summary = _fixture_manifest_summary(state)
    manifest = {
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "output_version": SIMULATION_OUTPUT_VERSION,
        "simulation_run_id": run_id,
        "generated_at": _format_utc(generated_at),
        "data_cutoff_utc": _format_utc(cutoff),
        "runs": runs,
        "seed": seed,
        "chunk_size": config.chunk_size,
        "processes": config.processes,
        "model": {"family": OFFICIAL_MODEL_FAMILY, "version": OFFICIAL_MODEL_VERSION},
        "model_config": dict(model_config_payload),
        "rules": {
            "version": RULE_VERSION,
            "source": RULE_SOURCE_URL,
            "source_version": RULE_SOURCE_VERSION,
            "accessed": RULE_SOURCE_ACCESSED,
        },
        "simulation_config_version": config.config_version,
        "extra_time_goal_scale": config.extra_time_goal_scale,
        "penalties_model": "symmetric_50_50",
        "snapshot_checksum": state.snapshot_checksum,
        "snapshot_reference": state.snapshot_reference,
        "raw_snapshot_path": (
            None if state.raw_snapshot_path is None else str(state.raw_snapshot_path)
        ),
        "source_fixture_count": state.source_fixture_count,
        "fixtures": fixture_summary,
        "input_checksum": input_checksum,
        "elapsed_seconds": elapsed_seconds,
        "fallback_counts": dict(aggregate.fallback_counts),
        "warnings": list(aggregate.warnings),
        "outputs": {name: str(path) for name, path in paths.items()},
        "checksums": {
            name: _file_sha256(path)
            for name, path in paths.items()
            if name not in {"manifest"} and path.is_file()
        },
    }
    _write_json(paths["manifest"], manifest)
    _copy_latest_to_history(paths, history_root)
    return paths


def _fixture_manifest_summary(state: TournamentState) -> Mapping[str, object]:
    observed_ids = sorted(
        fixture.source_fixture_id for fixture in state.fixtures if fixture.status == "played"
    )
    in_progress_ids = sorted(
        fixture.source_fixture_id for fixture in state.fixtures if fixture.status == "in_progress"
    )
    future_known_ids = sorted(
        fixture.source_fixture_id
        for fixture in state.fixtures
        if fixture.status == "scheduled"
        and fixture.home_team_id is not None
        and fixture.away_team_id is not None
    )
    future_tbd_ids = sorted(
        fixture.source_fixture_id
        for fixture in state.fixtures
        if fixture.status == "scheduled"
        and fixture.home_team_id is None
        and fixture.away_team_id is None
    )
    future_partially_known_ids = sorted(
        fixture.source_fixture_id
        for fixture in state.fixtures
        if fixture.status == "scheduled"
        and ((fixture.home_team_id is None) != (fixture.away_team_id is None))
    )
    simulated_ids = sorted(
        fixture.source_fixture_id for fixture in state.fixtures if fixture.status != "played"
    )
    return {
        "observed": len(observed_ids),
        "in_progress": len(in_progress_ids),
        "future_known": len(future_known_ids),
        "future_tbd": len(future_tbd_ids),
        "future_partially_known": len(future_partially_known_ids),
        "simulated_or_resolved_per_run": len(simulated_ids),
        "observed_fixture_ids": observed_ids,
        "in_progress_fixture_ids": in_progress_ids,
        "future_known_fixture_ids": future_known_ids,
        "future_tbd_fixture_ids": future_tbd_ids,
        "future_partially_known_fixture_ids": future_partially_known_ids,
        "simulated_source_fixture_ids": simulated_ids,
    }


def _copy_latest_to_history(paths: Mapping[str, Path], history_root: Path) -> None:
    for path in paths.values():
        if not path.is_file():
            continue
        target = history_root / path.name
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise TournamentSimulationError(f"immutable simulation history collision at {target}")
        if not target.exists():
            target.write_bytes(path.read_bytes())


def _read_provider_fixtures(
    path: Path,
    *,
    teams_path: Path,
    aliases_path: Path,
) -> tuple[TournamentFixture, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TournamentSimulationError(f"failed to read raw fixture snapshot: {path}") from exc
    raw_matches = payload.get("matches") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_matches, list):
        raise TournamentSimulationError("raw fixture snapshot must contain a matches list")
    teams = load_team_catalog(teams_path)
    aliases = load_team_aliases(aliases_path, teams=teams)
    canonical_names = {team.canonical_name: team.canonical_team_id for team in teams}
    fixtures = [
        _provider_fixture(item, aliases=aliases, canonical_names=canonical_names)
        for item in raw_matches
        if isinstance(item, Mapping)
    ]
    return tuple(sorted(fixtures, key=lambda fixture: fixture.kickoff_utc))


def _provider_fixture(
    item: Mapping[str, object],
    *,
    aliases: Sequence[TeamAlias],
    canonical_names: Mapping[str, str],
) -> TournamentFixture:
    source_fixture_id = str(_required_raw(item, "id"))
    kickoff = _parse_utc(str(_required_raw(item, "utcDate")))
    match_date = kickoff.date()
    source = "world_cup_2026_football_data"
    home = _raw_mapping(_required_raw(item, "homeTeam"))
    away = _raw_mapping(_required_raw(item, "awayTeam"))
    home_name = _optional_str(home.get("name"))
    away_name = _optional_str(away.get("name"))
    home_team_id = _resolve_optional_team(
        home_name,
        source=source,
        match_date=match_date,
        aliases=aliases,
        canonical_names=canonical_names,
    )
    away_team_id = _resolve_optional_team(
        away_name,
        source=source,
        match_date=match_date,
        aliases=aliases,
        canonical_names=canonical_names,
    )
    score = _raw_mapping(item.get("score"))
    duration = str(score.get("duration") or "REGULAR")
    full_time = _raw_mapping(score.get("fullTime"))
    regular_time = _raw_mapping(score.get("regularTime")) or full_time
    extra_time = _raw_mapping(score.get("extraTime"))
    penalties = _raw_mapping(score.get("penalties"))
    home_goals_90 = _optional_int(regular_time.get("home"))
    away_goals_90 = _optional_int(regular_time.get("away"))
    home_after_extra, away_after_extra = _football_data_after_extra_time(
        duration=duration,
        home_goals_90=home_goals_90,
        away_goals_90=away_goals_90,
        full_time=full_time,
        extra_time=extra_time,
    )
    group = _optional_str(item.get("group"))
    return TournamentFixture(
        source_fixture_id=source_fixture_id,
        stage=normalize_stage(_optional_str(item.get("stage"))) or "unknown",
        group=None if group is None else group.removeprefix("GROUP_"),
        kickoff_utc=kickoff,
        status=_provider_status(str(_required_raw(item, "status"))),
        source_status=str(_required_raw(item, "status")),
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_team_name=home_name,
        away_team_name=away_name,
        home_goals_90=home_goals_90,
        away_goals_90=away_goals_90,
        home_goals_after_extra_time=home_after_extra,
        away_goals_after_extra_time=away_after_extra,
        home_penalty_goals=_optional_int(penalties.get("home")),
        away_penalty_goals=_optional_int(penalties.get("away")),
    )


def _provider_status(status: str) -> str:
    if status == "FINISHED":
        return "played"
    if status in {"IN_PLAY", "PAUSED"}:
        return "in_progress"
    if status in {"TIMED", "SCHEDULED"}:
        return "scheduled"
    if status in {"POSTPONED", "CANCELLED", "SUSPENDED"}:
        return status.lower()
    return "unknown"


def _resolve_optional_team(
    name: str | None,
    *,
    source: str,
    match_date: date,
    aliases: Sequence[TeamAlias],
    canonical_names: Mapping[str, str],
) -> str | None:
    if name is None:
        return None
    try:
        return resolve_team_alias(
            source=source,
            source_name=name,
            match_date=match_date,
            aliases=aliases,
        )
    except ValueError:
        resolved = canonical_names.get(name)
        if resolved is not None:
            return resolved
        raise TournamentSimulationError(
            f"missing team alias for {source}:{name} on {match_date.isoformat()}"
        ) from None


def _team_ids_from_group_stage(fixtures: Sequence[TournamentFixture]) -> set[str]:
    teams: set[str] = set()
    for fixture in fixtures:
        if fixture.stage != "group_stage":
            continue
        teams.add(_required_team(fixture.home_team_id, fixture=fixture))
        teams.add(_required_team(fixture.away_team_id, fixture=fixture))
    return teams


def _group_teams(fixtures: Sequence[TournamentFixture]) -> dict[str, set[str]]:
    groups = {group: set[str]() for group in GROUPS}
    for fixture in fixtures:
        if fixture.stage != "group_stage":
            continue
        if fixture.group not in groups:
            raise TournamentSimulationError(f"invalid group-stage group: {fixture.group}")
        groups[fixture.group].add(_required_team(fixture.home_team_id, fixture=fixture))
        groups[fixture.group].add(_required_team(fixture.away_team_id, fixture=fixture))
    for group, teams in groups.items():
        if len(teams) != TEAMS_PER_GROUP:
            raise TournamentSimulationError(f"group {group} must contain four teams")
    return groups


def _find_fixture(
    fixtures: Sequence[TournamentFixture],
    *,
    stage: str,
    home: str,
    away: str,
) -> TournamentFixture | None:
    exact = [
        fixture
        for fixture in fixtures
        if fixture.stage == stage and fixture.home_team_id == home and fixture.away_team_id == away
    ]
    if len(exact) > 1:
        raise TournamentSimulationError(f"duplicate provider fixture for {stage} {home} v {away}")
    if exact:
        return exact[0]
    return None


def _resolve_dependency(
    label: str,
    winners: Mapping[int, str],
    losers: Mapping[int, str],
) -> str:
    kind = label[0]
    match_number = int(label[1:])
    if kind == "W":
        return winners[match_number]
    if kind == "L":
        return losers[match_number]
    raise TournamentSimulationError(f"unsupported bracket dependency: {label}")


def _group_probability_rows(
    *,
    group_position_counts: Counter[tuple[str, str, int]],
    group_status_counts: Counter[tuple[str, str]],
    runs: int,
) -> tuple[dict[str, object], ...]:
    denominator = float(runs) if runs else 1.0
    teams_by_group: dict[str, set[str]] = {group: set() for group in GROUPS}
    for group, team, _ in group_position_counts:
        teams_by_group[group].add(team)
    rows: list[dict[str, object]] = []
    for group in GROUPS:
        for team in sorted(teams_by_group[group]):
            rows.append(
                {
                    "group": group,
                    "team_id": team,
                    "finish_1": group_position_counts[(group, team, 1)] / denominator,
                    "finish_2": group_position_counts[(group, team, 2)] / denominator,
                    "finish_3": group_position_counts[(group, team, 3)] / denominator,
                    "finish_4": group_position_counts[(group, team, 4)] / denominator,
                    "classify_direct": group_status_counts[(team, "direct")] / denominator,
                    "classify_best_third": group_status_counts[(team, "best_third")]
                    / denominator,
                    "group_eliminated": group_status_counts[(team, "group_eliminated")]
                    / denominator,
                }
            )
    return tuple(rows)


def _group_summary_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Group Probabilities",
        "",
        "| Group | Team | 1st | 2nd | 3rd | 4th | Direct | Best 3rd | Eliminated |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {_object_str(row['group'])} | {_object_str(row['team_id'])} | "
            f"{_object_float(row['finish_1']):.3f} | {_object_float(row['finish_2']):.3f} | "
            f"{_object_float(row['finish_3']):.3f} | {_object_float(row['finish_4']):.3f} | "
            f"{_object_float(row['classify_direct']):.3f} | "
            f"{_object_float(row['classify_best_third']):.3f} | "
            f"{_object_float(row['group_eliminated']):.3f} |"
        )
    return "\n".join(lines) + "\n"


def _round_probabilities_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Round Probabilities",
        "",
        "| Team | R32 | R16 | QF | SF | Final | Champion |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: _object_float(item["round_of_32"]), reverse=True):
        lines.append(
            f"| {_object_str(row['team_id'])} | {_object_float(row['round_of_32']):.3f} | "
            f"{_object_float(row['round_of_16']):.3f} | "
            f"{_object_float(row['quarter_final']):.3f} | "
            f"{_object_float(row['semi_final']):.3f} | {_object_float(row['final']):.3f} | "
            f"{_object_float(row['champion']):.3f} |"
        )
    return "\n".join(lines) + "\n"


def _champion_probabilities_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Champion Probabilities",
        "",
        "| Rank | Team | Champion | Final | Semi-final |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    ordered = sorted(rows, key=lambda item: _object_float(item["champion"]), reverse=True)
    for index, row in enumerate(ordered[:20], start=1):
        lines.append(
            f"| {index} | {_object_str(row['team_id'])} | {_object_float(row['champion']):.3f} | "
            f"{_object_float(row['final']):.3f} | {_object_float(row['semi_final']):.3f} |"
        )
    mexico = next((row for row in ordered if row["team_id"] == "mexico"), None)
    if mexico is not None:
        lines.extend(
            [
                "",
                "## Mexico",
                "",
                (
                    "Mexico champion probability: "
                    f"{_object_float(mexico['champion']):.3f}; "
                    f"final: {_object_float(mexico['final']):.3f}; "
                    f"semi-final: {_object_float(mexico['semi_final']):.3f}."
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def _bracket_summary_markdown(pair_counts: Counter[str], *, runs: int) -> str:
    denominator = float(runs) if runs else 1.0
    lines = [
        "# Bracket Summary",
        "",
        "| Pair | Probability |",
        "| --- | ---: |",
    ]
    for pair, count in pair_counts.most_common(30):
        lines.append(f"| {pair} | {count / denominator:.3f} |")
    return "\n".join(lines) + "\n"


def _stability_report_markdown(
    rows: Sequence[Mapping[str, object]],
    *,
    runs: int,
    elapsed_seconds: float,
) -> str:
    lines = [
        "# Stability Report",
        "",
        f"Elapsed seconds: {elapsed_seconds:.3f}",
        f"Monte Carlo runs: {runs}",
        "",
        "| Team | Champion | MC SE | 95% half-width |",
        "| --- | ---: | ---: | ---: |",
    ]
    ordered = sorted(rows, key=lambda item: _object_float(item["champion"]), reverse=True)
    denominator = float(runs) if runs > 0 else 1.0
    for row in ordered[:20]:
        probability = _object_float(row["champion"])
        standard_error = math.sqrt(probability * (1.0 - probability) / denominator)
        lines.append(
            f"| {_object_str(row['team_id'])} | {probability:.4f} | "
            f"{standard_error:.4f} | {1.96 * standard_error:.4f} |"
        )
    lines.extend(
        [
            "",
            "Seed and sample-size comparisons should be interpreted against this MC error. "
            "Operational validation can compare immutable history runs with different seeds "
            "or runs.",
        ]
    )
    return "\n".join(lines) + "\n"


def _report_header(cutoff: datetime, runs: int, seed: int) -> str:
    return (
        f"Data cutoff UTC: {_format_utc(cutoff)}\n"
        f"Model: {OFFICIAL_MODEL_FAMILY} ({OFFICIAL_MODEL_VERSION})\n"
        f"Rules: {RULE_VERSION}\n"
        f"Simulations: {runs}\n"
        f"Seed: {seed}\n"
        "Extra time: expected goals scaled by 30/90 from 90-minute rates.\n"
        "Penalties: symmetric 50/50 baseline.\n"
        "Unmodelled fair play/FIFA-ranking ties use deterministic random_lot_proxy.\n\n"
    )


def _assert_monotonic(row: Mapping[str, object]) -> None:
    fields = (
        "champion",
        "final",
        "semi_final",
        "quarter_final",
        "round_of_16",
        "round_of_32",
    )
    values = [_object_float(row[field]) for field in fields]
    if any(
        left > right + 1e-12
        for left, right in zip(values[:-1], values[1:], strict=True)
    ):
        raise TournamentSimulationError(
            f"round probabilities are not monotonic for {row['team_id']}"
        )


def _pair_key(stage: str, home: str, away: str) -> str:
    return f"{stage}:{home} v {away}"


def _simulation_run_id(
    *,
    state: TournamentState,
    config: SimulationConfig,
    runs: int,
    seed: int,
    cutoff: datetime,
    input_checksum: str,
) -> str:
    payload = {
        "snapshot_checksum": state.snapshot_checksum,
        "cutoff": _format_utc(cutoff),
        "model_version": OFFICIAL_MODEL_VERSION,
        "rule_version": RULE_VERSION,
        "config_version": config.config_version,
        "output_version": SIMULATION_OUTPUT_VERSION,
        "runs": runs,
        "seed": seed,
        "input_checksum": input_checksum,
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()[:24]


def _simulation_input_checksum(
    *,
    state: TournamentState,
    config: SimulationConfig,
    runs: int,
    seed: int,
    cutoff: datetime,
) -> str:
    payload = {
        "fixtures": [
            {
                "source_fixture_id": fixture.source_fixture_id,
                "stage": fixture.stage,
                "group": fixture.group,
                "kickoff_utc": _format_utc(fixture.kickoff_utc),
                "status": fixture.status,
                "home_team_id": fixture.home_team_id,
                "away_team_id": fixture.away_team_id,
                "home_goals_90": fixture.home_goals_90,
                "away_goals_90": fixture.away_goals_90,
                "home_goals_after_extra_time": fixture.home_goals_after_extra_time,
                "away_goals_after_extra_time": fixture.away_goals_after_extra_time,
                "home_penalty_goals": fixture.home_penalty_goals,
                "away_penalty_goals": fixture.away_penalty_goals,
            }
            for fixture in state.fixtures
        ],
        "snapshot_checksum": state.snapshot_checksum,
        "config": {
            "config_version": config.config_version,
            "chunk_size": config.chunk_size,
            "extra_time_goal_scale": config.extra_time_goal_scale,
        },
        "runs": runs,
        "seed": seed,
        "cutoff": _format_utc(cutoff),
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _collection_raw_path(report: Mapping[str, object]) -> Path | None:
    manifests = report.get("snapshot_manifests")
    if not isinstance(manifests, list):
        return None
    for manifest in manifests:
        if not isinstance(manifest, Mapping):
            continue
        if manifest.get("source_fixture_id") == "__collection__":
            raw_path = _optional_str(manifest.get("raw_path"))
            return None if raw_path is None else Path(raw_path)
    return None


def _read_json(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _read_gzip_csv_rows(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, mode="rt", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload) + b"\n")


def _write_parquet(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(row) for row in rows])
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


def _json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_value,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_raw(row: Mapping[str, object], field: str) -> object:
    value = row.get(field)
    if value is None:
        raise TournamentSimulationError(f"raw fixture is missing {field}")
    return value


def _raw_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _required_str(row: Mapping[object, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise TournamentSimulationError(f"simulation config field is required: {field}")
    return value


def _positive_int(value: object, *, field: str) -> int:
    parsed = _non_negative_int(value, field=field)
    if parsed <= 0:
        raise TournamentSimulationError(f"{field} must be positive")
    return parsed


def _non_negative_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TournamentSimulationError(f"{field} must be a non-negative integer")
    return value


def _positive_float(value: object, *, field: str, allow_zero: bool) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TournamentSimulationError(f"{field} must be a finite float") from exc
    if not math.isfinite(parsed) or parsed < 0 or (parsed == 0 and not allow_zero):
        raise TournamentSimulationError(f"{field} must be positive")
    return parsed


def _object_float(value: object) -> float:
    if isinstance(value, bool):
        raise TournamentSimulationError("boolean value cannot be parsed as a float")
    if isinstance(value, int | float | np.integer | np.floating | str):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    raise TournamentSimulationError(f"value cannot be parsed as a finite float: {value!r}")


def _object_str(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _required_team(value: str | None, *, fixture: TournamentFixture) -> str:
    if value is None:
        raise TournamentSimulationError(f"fixture {fixture.source_fixture_id} has unresolved team")
    return value


def _required_int(value: int | None, field: str, fixture: TournamentFixture) -> int:
    if value is None:
        raise TournamentSimulationError(f"fixture {fixture.source_fixture_id} is missing {field}")
    return value


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _optional_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return _require_utc(value)
    if isinstance(value, str) and value:
        return _parse_utc(value)
    return None


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TournamentSimulationError(f"invalid UTC timestamp: {value}") from exc
    return _require_utc(parsed)


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise TournamentSimulationError("timestamps must be timezone-aware UTC")
    return value.astimezone(UTC)


def _format_utc(value: datetime) -> str:
    return _require_utc(value).isoformat().replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _csv_value(value: object) -> object:
    if isinstance(value, datetime):
        return _format_utc(value)
    if isinstance(value, float):
        return f"{value:.10f}"
    return value


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return _format_utc(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def audit_simulation_outputs(simulations_root: Path = Path("simulations")) -> Mapping[str, object]:
    """Validate latest simulation outputs and return a compact audit summary."""

    latest = simulations_root / "latest"
    required = [
        "team_probabilities.csv",
        "team_probabilities.json",
        "group_probabilities.csv",
        "group_tables_summary.md",
        "round_probabilities.md",
        "champion_probabilities.md",
        "bracket_summary.md",
        "stability_report.md",
        "manifest.json",
    ]
    missing = [name for name in required if not (latest / name).is_file()]
    if missing:
        raise TournamentSimulationError("missing simulation outputs: " + ", ".join(missing))
    rows = _read_csv_rows(latest / "team_probabilities.csv")
    for row in rows:
        _assert_monotonic(row)
    manifest = _read_json(latest / "manifest.json")
    rules = manifest.get("rules")
    model = manifest.get("model")
    if not isinstance(rules, Mapping) or rules.get("version") != RULE_VERSION:
        raise TournamentSimulationError("simulation manifest has unexpected rule version")
    if not isinstance(model, Mapping) or model.get("version") != OFFICIAL_MODEL_VERSION:
        raise TournamentSimulationError("simulation manifest has unexpected model version")
    annex_c_table()
    return {
        "team_rows": len(rows),
        "rule_version": RULE_VERSION,
        "model_version": OFFICIAL_MODEL_VERSION,
        "simulation_run_id": manifest.get("simulation_run_id"),
    }


def simulation_report_summary(simulations_root: Path = Path("simulations")) -> Mapping[str, object]:
    """Return a concise summary of latest simulation outputs."""

    manifest = _read_json(simulations_root / "latest" / "manifest.json")
    champions = _read_csv_rows(simulations_root / "latest" / "team_probabilities.csv")
    top = sorted(champions, key=lambda row: _object_float(row["champion"]), reverse=True)[:10]
    mexico = next((row for row in champions if row.get("team_id") == "mexico"), None)
    return {
        "simulation_run_id": manifest.get("simulation_run_id"),
        "data_cutoff_utc": manifest.get("data_cutoff_utc"),
        "runs": manifest.get("runs"),
        "seed": manifest.get("seed"),
        "top_champions": top,
        "mexico": mexico,
        "fallback_counts": manifest.get("fallback_counts"),
    }
