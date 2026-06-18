# Motor de ratings Elo

Este documento describe el motor Elo implementado para la Fase 2. El objetivo de esta etapa es
producir ratings prepartido y ratings actuales a partir de `modeling_matches.parquet`; no calibra
probabilidades finales W/D/L, no modela goles y no simula torneos.

## Orden temporal

El motor procesa solamente partidos con `model_eligible=true`.

Los partidos se agrupan por `match_date`. Para cada fecha:

1. Se calculan todos los ratings prepartido y scores esperados con los ratings disponibles antes de
   esa fecha.
2. Se calculan los cambios de rating de cada partido.
3. Se aplican todos los cambios al terminar el lote de la fecha.

Esto evita inventar un orden intradia cuando la fuente solo conoce la fecha del partido.

## Ecuaciones

Para el partido entre local designado `H` y visitante designado `A`:

```text
home_advantage_adjustment =
  +home_advantage  si home_advantage_eligible=true y home_advantage_status=home_team
  -home_advantage  si home_advantage_eligible=true y home_advantage_status=away_team
  0                en otro caso

elo_difference_pre = home_elo_pre + home_advantage_adjustment - away_elo_pre

home_expected_score = 1 / (1 + 10 ^ (-elo_difference_pre / 400))
```

El resultado observado usa goles a 90 minutos:

```text
actual_home_score =
  1.0  si home_goals_90 > away_goals_90
  0.5  si home_goals_90 = away_goals_90
  0.0  si home_goals_90 < away_goals_90
```

La actualizacion base es:

```text
k_effective = k_base * competition_importance[competition_category]
rating_change = k_effective * margin_multiplier * (actual_home_score - home_expected_score)

home_elo_post = home_elo_pre + rating_change
away_elo_post = away_elo_pre - rating_change
```

Cuando varios partidos caen en la misma fecha, los `*_elo_post` reflejan el rating del equipo al
terminar el lote completo de esa fecha.

## Margen de goles

El margen de victoria esta deshabilitado por defecto. Cuando se habilita:

```text
margin_multiplier =
  1                                                     si goal_difference <= 1
  1 + (goal_difference - 1) * goal_difference_weight    si goal_difference > 1
```

El multiplicador no se presenta como optimo; solo deja el comportamiento parametrizable para
evaluacion posterior.

## Regresion por inactividad

La regresion por inactividad esta deshabilitada por defecto. Cuando se habilita, se aplica antes de
calcular ratings prepartido para una fecha y solo usa la fecha del ultimo partido ya procesado:

```text
rating_regressed =
  initial_rating + (current_rating - initial_rating) * (1 - regression_fraction)
```

Se aplica si:

```text
match_date - last_match_date >= inactivity_days
```

El primer partido de un equipo no se regresa, porque no existe actividad previa observada dentro del
historial procesado.

## Artefactos

El comando principal es:

```bash
uv run wc2026 model elo-ratings
```

Produce:

- `data/processed/elo_match_ratings.parquet`
- `data/processed/elo_current_ratings.parquet`
- `data/interim/elo_ratings_report.json`
