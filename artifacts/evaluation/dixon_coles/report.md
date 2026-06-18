# Dixon-Coles Backtest Report

Selected model: `poisson`
Selected half-life days: `730.0`
Validation matches: 195
Prospective 2026 matches: 287

## Fold Metrics

| fold | matches | log_loss | brier_score | ranked_probability_score | goals_log_likelihood | goals_negative_log_likelihood | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| copa_america_2024 | 28 | 0.816975 | 0.465024 | 0.158307 | -2.529961 | 2.529961 | 0.642857 |
| euro_2024 | 48 | 0.973426 | 0.578342 | 0.188912 | -2.595949 | 2.595949 | 0.562500 |
| world_cup_2018 | 60 | 0.928572 | 0.547252 | 0.203898 | -2.811234 | 2.811234 | 0.600000 |
| world_cup_2022 | 59 | 0.999154 | 0.576717 | 0.216611 | -2.999681 | 2.999681 | 0.559322 |

## Comparison With Elo

| fold | dixon_coles_log_loss | dixon_coles_brier_score | dixon_coles_ranked_probability_score | dixon_coles_goals_log_likelihood | elo_log_loss | elo_brier_score | elo_ranked_probability_score | log_loss_delta_vs_elo | brier_delta_vs_elo | rps_delta_vs_elo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| copa_america_2024 | 0.816975 | 0.465024 | 0.158307 | -2.529961 | 0.779037 | 0.442461 | 0.152548 | 0.037938 | 0.022562 | 0.005759 |
| euro_2024 | 0.973426 | 0.578342 | 0.188912 | -2.595949 | 1.032136 | 0.614799 | 0.200547 | -0.058709 | -0.036458 | -0.011635 |
| world_cup_2018 | 0.928572 | 0.547252 | 0.203898 | -2.811234 | 0.957798 | 0.570447 | 0.218793 | -0.029225 | -0.023195 | -0.014895 |
| world_cup_2022 | 0.999154 | 0.576717 | 0.216611 | -2.999681 | 1.019244 | 0.602083 | 0.232018 | -0.020089 | -0.025366 | -0.015407 |

Validation reuses the Elo temporal folds. The 2026 holdout is not used for model selection.
