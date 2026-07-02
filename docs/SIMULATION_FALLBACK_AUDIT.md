# Simulation fallback audit

Audit target: latest published simulation in `origin/predictions-data` as of
`2026-07-01T21:40:07Z`, run ID `f104b7acd8e5ffd95f698a97`.

## Finding

`random_lot_proxy=25000` is an aggregate count of fallback decisions, not a count of affected
teams, groups, ties or fixtures. The operational Parquet artifact for the same run contains
25,000 rows and every row has `random_lot_proxy_count=1`, so this is one fallback decision repeated
once per Monte Carlo trajectory.

The fallback occurs in the ranking of best third-placed teams after all group tables are already
observed. It does not occur while simulating group matches, while resolving group positions, while
building the round of 32, or in knockout matches.

## Affected tie

The affected teams are Ghana, third in Group L, and Ecuador, third in Group E:

| Team | Group | Points | Goal difference | Goals for | Best-third rank in local output |
| --- | --- | ---: | ---: | ---: | ---: |
| Ghana | L | 4 | 0 | 2 | 3 or 4 |
| Ecuador | E | 4 | 0 | 2 | 3 or 4 |

Relevant fixtures:

| Group | Fixture | Score |
| --- | --- | --- |
| E | Ivory Coast vs Ecuador | 1-0 |
| E | Ecuador vs Curacao | 0-0 |
| E | Ecuador vs Germany | 2-1 |
| L | Ghana vs Panama | 1-0 |
| L | England vs Ghana | 0-0 |
| L | Croatia vs Ghana | 2-1 |

## Criterion reached

For best third-placed teams, FIFA regulations rank tied teams by points, goal difference, goals
scored, team conduct score, and then successive FIFA/Coca-Cola Men's World Ranking editions. The
project does not publish fair-play points or pre-tournament ranking snapshots and must not invent
them. The current simulator therefore uses `random_lot_proxy` when the modeled criteria are
exhausted.

This is a fallback caused by absent data, not evidence of a score-model error. It is not a rule
change and does not introduce a new model.

## Impact

Measured impact for this run:

| Surface | Impact |
| --- | --- |
| Group positions | None. Group E and Group L positions are already determined before this tie. |
| Qualified teams | None. Ghana and Ecuador are both among the top eight third-placed teams. |
| Best-third cutoff | None. The tie is for ranks 3/4, not the 8/9 qualification boundary. |
| Annex C group set | None. Annex C is keyed by the set of qualifying third-place groups, not this rank order. |
| Round of 32 | None. Published round-of-32 pairings are unchanged. |
| Champion probabilities | None measurable; no output probabilities require correction. |

No correction to models, probabilities, tournament rules or published results is warranted. The
dashboard surfaces the fallback count and includes a generated `simulation-fallback-audit.json`
explaining that fair play/ranking data were not invented.
