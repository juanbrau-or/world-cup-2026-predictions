# Directorios de datos

- `raw/`: snapshots originales e inmutables.
- `interim/`: resultados de limpieza y normalización.
- `processed/`: datasets listos para features o entrenamiento.
- `cache/`: respuestas regenerables de APIs.
- `static/`: tablas pequeñas revisadas manualmente y aptas para Git.

Los archivos grandes y generados están ignorados. Versiona manifiestos, esquemas, alias y tablas
estáticas pequeñas; no subas API keys ni dumps innecesarios.

## Artefactos de ingesta historica

`uv run wc2026 ingest historical` genera:

- `data/raw/international_results_csv/<retrieved_at>/results.csv`: snapshot crudo inmutable.
- `data/raw/international_results_csv/<retrieved_at>/shootouts.csv`: snapshot auxiliar para detectar
  partidos que requieren cuarentena por falta de marcadores a 90 y penales anotados.
- `data/processed/international_matches.parquet`: tabla canonica validada.
- `data/interim/historical_ingest_invalid_records.jsonl`: registros rechazados o duplicados con
  motivo.
- `data/interim/historical_ingest_report.json`: conteos de calidad, checksums y rutas de snapshots.

Los directorios `raw/`, `interim/` y `processed/` estan ignorados por Git. Para desarrollo o tests
sin red usa `--results-file` y `--shootouts-file` con CSVs equivalentes pequenos. `--dry-run`
ejecuta la descarga o lectura, parseo y validacion sin escribir snapshots ni artefactos derivados.

## Catalogo y alias de equipos

- `data/static/teams.csv`: catalogo canonico de equipos con estado y tipo auditable.
- `data/static/team_aliases.csv`: alias exactos por fuente hacia el catalogo canonico.

`uv run wc2026 audit aliases` valida cobertura de nombres originales, alias duplicados o
conflictivos, alias hacia IDs inexistentes e IDs canonicos sin alias.

En la revision fijada, alias e ingesta miden coberturas distintas:

- Alias: 336/336 nombres originales y 49,477/49,477 filas de `results.csv` resuelven equipos,
  equivalente a 100.00%.
- Ingesta valida: 48,746/49,477 filas producen registros canonicos, equivalente a 98.52%.

La diferencia no viene de alias faltantes. `shootouts.csv` tiene 678 filas: 677 coinciden con
`results.csv` y provocan cuarentena por falta de marcador a 90 minutos y penales anotados, y una no
tiene partido correspondiente. Tambien hay 52 filas con marcador faltante y 2 duplicados
conflictivos en `results.csv`.
