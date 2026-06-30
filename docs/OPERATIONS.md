# Operacion del pipeline operativo

Este flujo automatiza el MVP manual:

```bash
uv run wc2026 ingest historical
uv run wc2026 prepare modeling-data
uv run wc2026 model dixon-coles
uv run wc2026 ingest world-cup
uv run wc2026 predict upcoming
uv run wc2026 evaluate prospective
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
- `prospective_evaluation.json`
- `prospective_evaluation.md`
- `manifest.json`
- `history/*.csv.gz`

`manifest.json` registra `generated_at`, `data_cutoff`, modelo, version, checksums, numero de
predicciones y observaciones prospectivas evaluadas. El historial usa nombres deterministas:

```text
history/<DATA_CUTOFF_UTC>_<CHECKSUM>.csv.gz
```

El publicador no escribe Parquet, snapshots raw ni modelos en la rama. Esos archivos quedan como
GitHub Actions artifacts.

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
uv run wc2026 evaluate prospective
uv run pytest tests/test_publication.py
uv run wc2026 publish prepare --predictions-root predictions --output-root dist/predictions-data
```

Para simular la rama de datos con historial existente:

```bash
git fetch origin predictions-data
git worktree add ../predictions-data predictions-data
uv run wc2026 publish prepare --predictions-root predictions --output-root ../predictions-data
```

No hagas commit ni push desde local salvo que la tarea lo pida explicitamente.
