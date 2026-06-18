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
