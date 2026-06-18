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
