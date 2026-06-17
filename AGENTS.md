# AGENTS.md

## Objetivo del repositorio

Construir un pipeline reproducible para pronosticar partidos del Mundial 2026 y actualizar las
predicciones conforme aparecen resultados nuevos. El resultado principal son probabilidades
calibradas, no solamente una clase ganadora.

## Forma de trabajar

1. Lee `README.md`, `docs/ROADMAP.md` y el prompt de la fase actual antes de modificar código.
2. Trabaja en una sola fase o issue por cambio. No implementes silenciosamente fases futuras.
3. Antes de editar, resume el estado actual y propón un plan breve.
4. Mantén cambios pequeños, revisables y compatibles con el código existente.
5. No hagas `git commit`, `git push`, merges ni cambios de secretos salvo instrucción explícita.
6. No borres datos o archivos del usuario para “limpiar” un error.

## Reglas de datos y ML

- Nunca uses información conocida después del kickoff para construir una feature prepartido.
- Todas las tablas de entrenamiento deben incluir `match_id`, `kickoff_utc` y `data_cutoff_utc`.
- Los rolling features deben usar solamente partidos anteriores (`shift(1)` o equivalente).
- Los splits de validación deben respetar el tiempo; queda prohibido el split aleatorio por filas.
- Conserva los datos crudos sin mutarlos. Las correcciones pertenecen a capas posteriores.
- Normaliza equipos mediante identificadores canónicos y una tabla explícita de alias.
- Distingue `designated_home` de la ventaja de jugar en el país anfitrión.
- No impongas efectos de altitud, viaje o clima sin evaluarlos fuera de muestra.
- Reporta log loss, Brier score y calibración; accuracy es secundaria.
- Usa seeds explícitas cuando haya aleatoriedad.
- Evita serializar modelos con datos sensibles o secretos.

## Arquitectura

- `src/worldcup2026/data/`: clientes, snapshots, validación y normalización.
- `src/worldcup2026/features/`: Elo, forma, descanso, viaje, altitud y clima.
- `src/worldcup2026/models/`: baselines, Poisson/Dixon-Coles y challengers.
- `src/worldcup2026/evaluation/`: backtesting temporal, métricas y calibración.
- `src/worldcup2026/simulation/`: grupos, eliminatorias y Monte Carlo.
- `src/worldcup2026/pipelines/`: orquestación de etapas.
- `data/raw/`: respuestas originales; no se editan manualmente.
- `data/processed/`: datasets derivados regenerables.
- `predictions/history/`: snapshots inmutables de predicciones.

## Convenciones de implementación

- Python >= 3.11, layout `src/`, type hints y docstrings en interfaces públicas.
- Usa `pathlib`, UTC y formatos ISO 8601.
- Los clientes HTTP deben tener timeout, retries acotados y caché local.
- No registres API keys ni respuestas completas que puedan contener secretos.
- Prefiere funciones pequeñas y transformaciones puras para features.
- Una feature nueva debe incluir prueba contra leakage o una justificación documentada.
- Agregar dependencias de producción requiere justificar por qué la biblioteca estándar o las
  dependencias existentes no bastan.

## Verificación obligatoria

Antes de declarar terminado un cambio, ejecuta lo que corresponda:

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

Si algún comando no puede ejecutarse, explica la causa exacta. Revisa el diff final y menciona
riesgos, supuestos y archivos modificados.
