# Simulacion del torneo

La version inicial del simulador usa reglas `world_cup_2026_rules_v1` y el modelo oficial
`poisson_goal_v1`. No promociona challengers ni mezcla resultados shadow con la simulacion oficial.

## Fuente de reglas

- Fuente: FIFA World Cup 26 Regulations, PDF oficial de FIFA.
- URL: `https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/FWC2026_regulations_EN.pdf`
- Version del documento: `FIFA World Cup 26 Regulations, May 2026`.
- Fecha de consulta: `2026-07-01`.
- Version local: `world_cup_2026_rules_v1`.

Reglas confirmadas:

- 48 selecciones.
- 12 grupos, de A a L.
- 4 equipos por grupo.
- 6 partidos por grupo.
- Avanzan los dos primeros de cada grupo y los 8 mejores terceros.
- La eliminatoria inicia en round of 32 y termina con final; existe partido por tercer lugar.
- El round of 32 usa los emparejamientos FIFA M73-M88.
- Las combinaciones de mejores terceros usan Annex C completo, con 495 combinaciones.
- Si un partido eliminatorio termina empatado a 90 minutos, se juega prorroga y despues penales.

La parte no modelada por ahora es fair play y ranking FIFA como criterios finales de desempate. El
simulador no inventa tarjetas ni rankings como feature. Si una trayectoria llega a ese punto, usa
un sorteo determinista con semilla y registra `random_lot_proxy`.

## Modelo oficial

La simulacion oficial usa exclusivamente matrices de marcador de `poisson_goal_v1`.

Para fixtures conocidos:

- Si existe una prediccion oficial persistida y compatible con el cutoff, usa esa matriz.
- Si no existe, ajusta la configuracion oficial congelada as-of y genera la matriz de marcador.
- En partidos en curso, no usa marcador parcial. Usa el ultimo snapshot prepartido valido cuando
  existe; si no existe, genera una matriz as-of y reporta una advertencia.

El challenger contextual permanece separado. Una simulacion contextual completa no es valida por
defecto para fixtures hipoteticos porque sus features dependen de trayectorias futuras aun no
materializadas.

## Partidos

Fase de grupos:

- Se muestrean goles local y visitante desde la matriz de marcador de 90 minutos.
- Se actualizan puntos, goles a favor, goles en contra y diferencia.
- Los partidos terminados conservan el resultado real a 90 minutos.

Eliminatoria:

- Se simulan los 90 minutos.
- Si hay empate, la prorroga usa tasas esperadas escaladas por `30/90`.
- Si sigue empatado, penales usa baseline simetrico `50/50`.
- El ganador por penales no se registra como victoria 1X2 a 90 minutos.

La prorroga es una suposicion declarativa, no recalibrada con resultados de 2026.

## Outputs

Una corrida escribe `simulations/latest/`:

- `team_probabilities.csv`
- `team_probabilities.json`
- `group_probabilities.csv`
- `group_tables_summary.md`
- `round_probabilities.md`
- `champion_probabilities.md`
- `bracket_summary.md`
- `manifest.json`
- `simulation_results.parquet`

Tambien copia los mismos archivos a `simulations/history/<simulation_run_id>/`. El Parquet completo
es artifact operativo, no archivo publicable en `predictions-data`.

El `simulation_run_id` es determinista para la combinacion de snapshot, cutoff, modelo, reglas,
configuracion, seed e input checksum.

## Publicacion

La rama `predictions-data` recibe solo archivos pequenos bajo `simulation/`:

- `simulation/manifest.json`
- `simulation/team_probabilities.csv`
- `simulation/team_probabilities.json`
- `simulation/champion_probabilities.md`
- `simulation/round_probabilities.md`
- `simulation/group_tables_summary.md`
- `simulation/bracket_summary.md`

No se publican Parquet, raw API, modelos, secretos ni trayectorias completas.

## Comandos

```bash
uv run wc2026 simulate tournament
uv run wc2026 simulate tournament --runs 50000
uv run wc2026 simulate tournament --seed 2026
uv run wc2026 simulate tournament --as-of 2026-07-01T00:00:00Z
uv run wc2026 simulate tournament --offline-fixture
uv run wc2026 audit simulation
uv run wc2026 report simulation
```

`--shadow-contextual` existe como opt-in, pero no es el default y no mezcla resultados con el
baseline oficial.

