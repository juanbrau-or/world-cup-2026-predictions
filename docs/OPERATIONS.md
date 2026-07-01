# Operacion del pipeline operativo

Este flujo automatiza el MVP manual:

```bash
uv run wc2026 ingest historical
uv run wc2026 prepare modeling-data
uv run wc2026 model dixon-coles
uv run wc2026 ingest world-cup
uv run wc2026 predict upcoming
uv run wc2026 prepare contextual-features
uv run wc2026 model contextual-challenger
uv run wc2026 predict shadow-contextual
uv run wc2026 evaluate prospective
uv run wc2026 evaluate shadow-contextual
uv run wc2026 simulate tournament
```

El workflow vive en `.github/workflows/operational-predictions.yml`, se ejecuta manualmente con
`workflow_dispatch` y cada 4 horas con el cron UTC `0 */4 * * *`.

## Secrets de GitHub

Configura los secrets en GitHub:

1. Abre `Settings` -> `Secrets and variables` -> `Actions`.
2. Crea `FOOTBALL_DATA_API_KEY` con la clave de la fuente principal.
3. Crea `API_FOOTBALL_KEY` solo si quieres habilitar el validador secundario opcional.

No configures estos valores como variables visibles ni los pegues en logs. El workflow los pasa por
entorno a los comandos de ingesta y el publicador escanea los archivos publicables para evitar que
aparezcan en la rama de datos.

## Ejecucion manual

En GitHub:

1. Abre `Actions`.
2. Selecciona `Operational Predictions`.
3. Pulsa `Run workflow`.
4. Ejecuta desde `main`.

El job `run-pipeline` usa permisos de solo lectura sobre el repositorio. El job `publish-data` es el
unico con `contents: write` y solo publica archivos pequenos en `predictions-data`.

## Rama predictions-data

La rama `predictions-data` se crea automaticamente si no existe. Contiene solo:

- `latest.csv`
- `latest.json`
- `upcoming.md`
- `prospective_scorecard.json`
- `prospective_scorecard.md`
- `prospective_matches.csv`
- `manifest.json`
- `history/*.csv.gz`
- `shadow/contextual_latest.csv`
- `shadow/contextual_latest.json`
- `shadow/contextual_upcoming.md`
- `shadow/contextual_scorecard.json`
- `shadow/contextual_scorecard.md`
- `shadow/contextual_comparison.md`
- `shadow/manifest.json`
- `simulation/manifest.json`
- `simulation/team_probabilities.csv`
- `simulation/team_probabilities.json`
- `simulation/champion_probabilities.md`
- `simulation/round_probabilities.md`
- `simulation/group_tables_summary.md`
- `simulation/bracket_summary.md`

`manifest.json` registra `generated_at`, `data_cutoff`, modelo, version, checksums, numero de
predicciones, version de politica prospectiva, snapshots vistos, predicciones oficiales,
observaciones evaluables y cutoff de resultados. El historial usa nombres deterministas:

```text
history/<DATA_CUTOFF_UTC>_<CHECKSUM>.csv.gz
```

El publicador no escribe Parquet, snapshots raw, trayectorias completas ni modelos en la rama. El ledger
`predictions/prediction_ledger.parquet` y el ledger shadow
`predictions/shadow/contextual_ledger.parquet` quedan como GitHub Actions artifacts junto con el
resto de Parquet operativo. El Parquet completo del simulador se guarda como artifact, no en
`predictions-data`.

Para leer la rama localmente:

```bash
git fetch origin predictions-data
git switch --detach origin/predictions-data
```

O sin cambiar de rama:

```bash
git fetch origin predictions-data
git show origin/predictions-data:manifest.json
git show origin/predictions-data:latest.csv
```

## Recuperar una ejecucion fallida

1. Abre la ejecucion fallida en `Actions`.
2. Revisa el primer step fallido y descarga los artifacts disponibles.
3. Si fallo la fuente principal, reintenta manualmente cuando el proveedor vuelva a responder.
4. Si fallo la validacion del publicador, revisa `operational-reports`, `operational-logs` y
   `operational-manifests`.
5. Si fallo el push a `predictions-data`, reintenta el workflow. No hagas commits manuales a
   `main` con outputs generados.

El flujo no debe fallar por fixtures TBD correctamente reportados, ausencia de fixtures elegibles o
ausencia de predicciones ya finalizadas para evaluar.

## Desactivar temporalmente el schedule

Edita `.github/workflows/operational-predictions.yml` y comenta el bloque:

```yaml
schedule:
  - cron: "0 */4 * * *"
```

Deja `workflow_dispatch` activo para poder ejecutar manualmente. Reactiva el bloque cuando quieras
reanudar la ejecucion periodica.

## Ejecutar el flujo localmente

Desde la raiz del repositorio:

```bash
uv sync --group dev
uv run wc2026 doctor
uv run wc2026 ingest historical
uv run wc2026 prepare modeling-data
uv run wc2026 model dixon-coles
uv run wc2026 ingest world-cup
uv run wc2026 predict upcoming
uv run wc2026 prepare contextual-features
uv run wc2026 model contextual-challenger
uv run wc2026 predict shadow-contextual
uv run wc2026 evaluate prospective
uv run wc2026 evaluate shadow-contextual
uv run wc2026 simulate tournament
uv run pytest tests/test_publication.py
uv run wc2026 publish prepare --predictions-root predictions --simulations-root simulations --output-root dist/predictions-data
```

Para simular la rama de datos con historial existente:

```bash
git fetch origin predictions-data
git worktree add ../predictions-data predictions-data
uv run wc2026 publish prepare --predictions-root predictions --output-root ../predictions-data
```

No hagas commit ni push desde local salvo que la tarea lo pida explicitamente.

## Evaluacion prospectiva oficial

La evaluacion prospectiva no elige retrospectivamente el snapshot mas favorable. Primero construye
`predictions/prediction_ledger.parquet` con todas las predicciones historicas validas y despues
aplica la politica versionada de `configs/prospective_evaluation.yaml`. La politica actual
`early_v1_2026_06_30` selecciona, para cada fixture y contexto `early_v1`, la prediccion valida mas
reciente generada al menos 6 horas antes del kickoff; si no existe, usa la prediccion valida mas
temprana disponible antes del kickoff.

En GitHub Actions, el job operativo restaura `predictions-data/history/*.csv.gz` en
`predictions/published-history/history/` antes de evaluar. Ese directorio es temporal y permite que
el ledger acumulado incluya snapshots publicados por runs anteriores sin guardar Parquet en la rama
publica.

Los reportes oficiales son:

- `predictions/prospective_scorecard.json`
- `predictions/prospective_scorecard.md`
- `predictions/prospective_matches.csv`

La metrica 1X2 usa exclusivamente el resultado a 90 minutos (`result_90`). Prorroga, penales y
ganador de clasificacion se conservan como campos separados para auditoria.

## Shadow contextual

El challenger contextual se ejecuta en modo shadow con `prediction_context=shadow_contextual_v1`.
Usa exactamente los fixtures elegibles del baseline oficial y el mismo `data_cutoff_utc`, pero
escribe bajo `predictions/shadow/`. No modifica `predictions/latest.csv`,
`predictions/latest.parquet`, `predictions/upcoming.md`, el ledger oficial ni el scorecard oficial.

El workflow publica solo archivos pequenos y etiquetados bajo `shadow/`; no publica modelos,
Parquet, datasets contextuales ni el ledger completo. Si no hay observaciones prospectivas
evaluables, el scorecard shadow reporta muestra cero y la comparacion pareada queda vacia.
