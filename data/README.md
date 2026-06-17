# Directorios de datos

- `raw/`: snapshots originales e inmutables.
- `interim/`: resultados de limpieza y normalización.
- `processed/`: datasets listos para features o entrenamiento.
- `cache/`: respuestas regenerables de APIs.
- `static/`: tablas pequeñas revisadas manualmente y aptas para Git.

Los archivos grandes y generados están ignorados. Versiona manifiestos, esquemas, alias y tablas
estáticas pequeñas; no subas API keys ni dumps innecesarios.
