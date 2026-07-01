# Contextual Features

Fase 4 agrega una capa auditable de features contextuales as-of. Estos artefactos son diagnosticos
para futuros modelos candidatos; no alimentan el modelo operativo congelado `poisson_goal_v1`.

## Comandos

```bash
uv run wc2026 prepare contextual-features
uv run wc2026 prepare contextual-features --as-of 2026-06-20T18:00:00Z
uv run wc2026 prepare contextual-features --no-live
uv run wc2026 prepare contextual-features --no-historical --offline-fixture \
  --as-of 2026-06-20T18:00:00Z --data-cutoff 2026-06-20T18:00:00Z
uv run wc2026 audit contextual-features
uv run wc2026 report contextual-features
```

Outputs generados:

- `data/processed/contextual_features/team_fixture_contextual_features.parquet`
- `data/processed/contextual_features/match_contextual_features.parquet`
- `data/interim/contextual_features/contextual_features_manifest.json`
- `data/interim/contextual_features/contextual_features_coverage.json`
- `data/interim/contextual_features/contextual_features_coverage.md`
- `data/interim/contextual_features/contextual_features_missing_data.json`
- `data/interim/contextual_features/contextual_features_leakage_audit.json`
- `data/interim/contextual_features/contextual_features_descriptive_quality.json`

Los Parquet son regenerables e ignorados por Git.

## Semantica As-Of

Cada fila team-fixture contiene `fixture_id`, `team_id`, `feature_generated_at_utc`,
`data_cutoff_utc`, `kickoff_utc`, `feature_set_version`, `source_dataset_revision`,
`source_row_checksum` y `venue_catalog_checksum`.

Validaciones obligatorias:

- `data_cutoff_utc <= feature_generated_at_utc`.
- Para filas prospectivas, `feature_generated_at_utc < kickoff_utc`.
- El partido previo de un equipo debe ser estrictamente anterior por `kickoff_utc`; si no hay hora
  exacta, debe ser estrictamente anterior por `match_date`.
- Fixtures TBD no se convierten en equipos sinteticos.
- Coordenadas, elevacion y timezones IANA invalidos fallan la auditoria.

## Nivel A

Features historicamente entrenables o parcialmente entrenables:

- `rest_hours`, `rest_days`, `hours_since_previous_match`
- `matches_last_7d`, `matches_last_14d`, `matches_last_30d`
- `minutes_equivalent_last_7d`, `minutes_equivalent_last_14d`,
  `minutes_equivalent_last_30d`
- `previous_match_extra_time`, `previous_match_penalty_shootout`
- `consecutive_matches_without_7d_rest`
- `tournament_match_number`, `is_first_tournament_match`

`minutes_equivalent` se define como 90 para partido regular y 120 si hubo prorroga; los penales no
agregan minutos. La fuente historica `international_results_csv` no permite distinguir todas las
prorrogas de forma segura, por lo que las cargas por minutos quedan missing cuando dependen de
partidos previos de esa fuente.

El historico actual es date-only. Por eso `rest_hours` queda missing historicamente y los conteos
usan solo partidos con fecha estrictamente anterior, sin inventar orden intradia.

## Nivel B

Features operativas para Mundial 2026:

- `venue_id`, ciudad, pais, coordenadas, elevacion y timezone IANA.
- `previous_venue_id`, distancia Haversine, acumulados de viaje a 7/14/30 dias.
- Cambio real de offset horario en la fecha de kickoff usando `zoneinfo`.
- Cambio de elevacion y valor absoluto.
- `cross_border_travel`, `host_country_match`, `is_neutral_venue`.

Estas features se reportan como operativas. No se presentan como historicamente validadas.

## Catalogo De Sedes

`data/static/venues.csv` versiona 16 sedes del Mundial 2026. Cada fila incluye nombre canonico,
aliases de proveedor, ciudad, pais, latitud, longitud, elevacion, timezone IANA, fuente, fecha de
consulta y version. Las fuentes usadas son la lista oficial de sedes de FIFA, coordenadas de
Wikidata, elevacion de Open-Meteo Elevation API e identificadores de timezone IANA.

No se imputan coordenadas, elevaciones ni timezones. Si un proveedor no entrega sede o el alias no
matchea el catalogo, las features geograficas quedan nulas y el reporte de missing data lo cuenta.

## Produccion

El workflow operativo calcula estas features como diagnostico y artifact despues de
`evaluate prospective`. No modifica:

- `configs/model.yaml`
- `poisson_goal_v1`
- `predictions/latest.csv`
- `predictions/prediction_ledger.parquet`
- la politica `early_v1_2026_06_30`
- la rama `predictions-data`
