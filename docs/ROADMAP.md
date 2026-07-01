# Roadmap

## Fase 0 — Bootstrap

- Entorno reproducible con `uv`.
- CLI mínima y configuración.
- Ruff, mypy, pytest y CI.
- Contratos de datos y documentación.

**Criterio de salida:** instalación limpia, `doctor`, lint y tests exitosos.

## Fase 1 — Ingestión histórica

- Elegir una fuente histórica inicial.
- Guardar snapshots crudos y manifiestos.
- Normalizar nombres de selecciones, fechas, torneos y neutralidad.
- Crear un dataset canónico de partidos.

**Criterio de salida:** reconstrucción determinista de `matches.parquet`.

## Fase 2 — Baseline Elo

- Elo prepartido y actualización posterior.
- Pesos configurables por tipo de competencia.
- Backtest temporal y probabilidades 1X2 calibrables.

**Criterio de salida:** baseline evaluado sin leakage.

## Fase 3 — Modelo de goles

- Poisson independiente como referencia.
- Ataque y defensa dinámicos.
- Corrección Dixon–Coles si mejora validación.
- Matriz de marcadores y probabilidades 1X2.

## Fase 4 — Features contextuales

- Localía real y país anfitrión.
- Descanso, viaje y cambio de huso horario.
- Altitud, aclimatación y clima.
- Variables del estado del torneo.

Cada grupo de features debe pasar una prueba de ablación.

## Fase 5 — Challenger y ensemble

- LightGBM multiclase.
- Calibración de probabilidades.
- Pesos de ensemble elegidos con validación temporal.

## Fase 6 — Actualización del Mundial

- Cliente de fixtures/resultados actuales.
- Detección idempotente de partidos nuevos.
- Snapshots con timestamp.
- Predicciones con `model_version` y `data_cutoff_utc`.

## Fase 7 — Simulación del torneo

- Desempates de grupos.
- Bracket de eliminatorias.
- Monte Carlo reproducible.
- Probabilidades de clasificar, llegar a cada ronda y ser campeón.

**Estado:** implementada como `world_cup_2026_rules_v1` con simulacion oficial basada en
`poisson_goal_v1`. Ver `docs/TOURNAMENT_SIMULATION.md`.

## Fase 8 — Publicación

- GitHub Actions para tests y actualización programada.
- Artefactos CSV/JSON y reporte legible.
- Opcional: dashboard estático o Streamlit.
