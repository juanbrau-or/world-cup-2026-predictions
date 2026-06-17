# Contrato mínimo de datos

## Tabla canónica de partidos

Campos mínimos:

| Campo | Tipo | Descripción |
|---|---|---|
| `match_id` | string | Identificador estable dentro del proyecto |
| `kickoff_utc` | datetime UTC | Inicio del partido |
| `home_team_id` | string | Equipo designado como local |
| `away_team_id` | string | Equipo designado como visitante |
| `home_score` | integer nullable | Goles a 90 minutos cuando terminó |
| `away_score` | integer nullable | Goles a 90 minutos cuando terminó |
| `competition` | string | Torneo |
| `stage` | string nullable | Fase o ronda |
| `venue_id` | string nullable | Estadio canónico |
| `host_country` | string nullable | País donde se disputa |
| `neutral` | boolean nullable | Indicador de sede neutral de la fuente |
| `source` | string | Proveedor de origen |
| `source_match_id` | string | Identificador del proveedor |
| `retrieved_at_utc` | datetime UTC | Momento de descarga |

## Tabla de features prepartido

Debe conservar, como mínimo:

- `match_id`
- `kickoff_utc`
- `data_cutoff_utc`
- versión del código o feature set
- todas las variables calculadas antes del kickoff

## Tabla de predicciones

Campos mínimos:

- `generated_at_utc`
- `data_cutoff_utc`
- `match_id`
- `model_name`
- `model_version`
- `prob_home_win`
- `prob_draw`
- `prob_away_win`
- `expected_home_goals` cuando aplique
- `expected_away_goals` cuando aplique

Las tres probabilidades deben ser finitas, estar en `[0, 1]` y sumar 1 dentro de una tolerancia.
