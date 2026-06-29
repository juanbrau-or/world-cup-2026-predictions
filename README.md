# World Cup 2026 Predictions

Repositorio reproducible para obtener datos, construir variables disponibles antes de cada partido,
entrenar modelos probabilísticos y publicar predicciones actualizadas del Mundial 2026.

## Principios

- Separar datos crudos, transformaciones, features, modelos y predicciones.
- No utilizar información posterior al instante de predicción.
- Conservar snapshots y el `data_cutoff` de cada pronóstico.
- Validar de forma temporal, nunca con un split aleatorio de partidos.
- Empezar con baselines interpretables antes de modelos complejos.

## Inicio rápido

```bash
cp .env.example .env
uv sync --group dev
uv run wc2026 doctor
uv run ruff check .
uv run mypy src
uv run pytest
```

## Flujo previsto

```bash
uv run wc2026 ingest historical
uv run wc2026 ingest world-cup
uv run wc2026 audit aliases
uv run wc2026 build-features
uv run wc2026 train
uv run wc2026 predict
uv run wc2026 simulate
```

La ingesta histórica de Fase 1B puede ejecutarse contra la fuente pública configurada o contra
archivos locales equivalentes:

```bash
uv run wc2026 ingest historical
uv run wc2026 ingest historical \
  --results-file tests/fixtures/international_results/results.csv \
  --shootouts-file tests/fixtures/international_results/shootouts.csv
```

Los comandos posteriores a `ingest historical` son contratos previstos y se implementarán por fases.
Consulta `docs/ROADMAP.md` y `prompts/CODEX_PROMPTS.md`.

## Ingesta viva del Mundial 2026

La fuente principal se elige en `.env` con `WORLD_CUP_PROVIDER` (`football_data` por defecto o
`api_football`). Las claves se leen únicamente desde el entorno; no se muestran en consola ni se
incluyen en snapshots. Si ambas claves existen, la fuente no principal se consulta solo para validar
equipos, kickoff y marcadores.

```bash
uv run wc2026 ingest world-cup
uv run wc2026 ingest world-cup --dry-run
uv run wc2026 ingest world-cup --offline-fixture --dry-run
uv run wc2026 predict upcoming
uv run wc2026 evaluate prospective
```

Cada respuesta de colección y cada miembro de fixture se conserva sin sobrescribir en
`data/raw/world_cup_2026/<provider>/<fetched_at>_<checksum>/`. Sus manifiestos registran proveedor,
endpoint, `source_fixture_id`, instante de fetch, checksum y revisión de schema. La vista operativa
actual queda en `data/processed/world_cup_2026/`, mientras que las vistas canónicas históricas se
guardan en `data/processed/world_cup_2026/snapshots/`. Los reportes de freshness, equipos no
resueltos y discrepancias de validación se escriben bajo `data/interim/`.

`predict upcoming` usa la selección congelada de Poisson configurada en `configs/model.yaml`, agrega
solo resultados terminados disponibles antes del cutoff vivo y escribe vistas actuales más snapshots
históricos bajo `predictions/`. `evaluate prospective` evalúa únicamente predicciones históricas ya
guardadas cuyos fixtures estén terminados en la vista viva actual.
