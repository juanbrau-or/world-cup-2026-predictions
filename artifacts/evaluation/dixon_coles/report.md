# Dixon-Coles Backtest Report

Selected model: `poisson`
Selected half-life days: `730.0`
Validation matches: 195
Holdout 2026 matches: 287

## Fold Metrics

| model | fold | matches | date_start | date_end | competitions | log_loss | brier_score | ranked_probability_score | goals_log_likelihood |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dixon_coles | copa_america_2024 | 28 | 2024-06-20 | 2024-07-14 | Copa América | 0.817047 | 0.465103 | 0.158319 | -2.529946 |
| dixon_coles | euro_2024 | 48 | 2024-06-14 | 2024-07-14 | UEFA Euro | 0.973173 | 0.578180 | 0.188885 | -2.595416 |
| dixon_coles | world_cup_2018 | 60 | 2018-06-14 | 2018-07-15 | FIFA World Cup | 0.929923 | 0.548134 | 0.204067 | -2.813300 |
| dixon_coles | world_cup_2022 | 59 | 2022-11-20 | 2022-12-17 | FIFA World Cup | 0.998883 | 0.576621 | 0.216597 | -2.999652 |
| elo | copa_america_2024 | 28 | 2024-06-20 | 2024-07-14 | Copa América | 0.779037 | 0.442461 | 0.152548 |  |
| elo | euro_2024 | 48 | 2024-06-14 | 2024-07-14 | UEFA Euro | 1.032136 | 0.614799 | 0.200547 |  |
| elo | world_cup_2018 | 60 | 2018-06-14 | 2018-07-15 | FIFA World Cup | 0.957798 | 0.570447 | 0.218793 |  |
| elo | world_cup_2022 | 59 | 2022-11-20 | 2022-12-17 | FIFA World Cup | 1.019244 | 0.602083 | 0.232018 |  |
| poisson | copa_america_2024 | 28 | 2024-06-20 | 2024-07-14 | Copa América | 0.816975 | 0.465024 | 0.158307 | -2.529961 |
| poisson | euro_2024 | 48 | 2024-06-14 | 2024-07-14 | UEFA Euro | 0.973426 | 0.578342 | 0.188912 | -2.595949 |
| poisson | world_cup_2018 | 60 | 2018-06-14 | 2018-07-15 | FIFA World Cup | 0.928572 | 0.547252 | 0.203898 | -2.811234 |
| poisson | world_cup_2022 | 59 | 2022-11-20 | 2022-12-17 | FIFA World Cup | 0.999154 | 0.576717 | 0.216611 | -2.999681 |

## Poisson Comparison With Elo

| fold | matches | date_start | date_end | competitions | poisson_log_loss | poisson_brier_score | poisson_ranked_probability_score | poisson_goals_log_likelihood | elo_log_loss | elo_brier_score | elo_ranked_probability_score | log_loss_delta_vs_elo | brier_delta_vs_elo | rps_delta_vs_elo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| copa_america_2024 | 28 | 2024-06-20 | 2024-07-14 | Copa América | 0.816975 | 0.465024 | 0.158307 | -2.529961 | 0.779037 | 0.442461 | 0.152548 | 0.037938 | 0.022562 | 0.005759 |
| euro_2024 | 48 | 2024-06-14 | 2024-07-14 | UEFA Euro | 0.973426 | 0.578342 | 0.188912 | -2.595949 | 1.032136 | 0.614799 | 0.200547 | -0.058709 | -0.036458 | -0.011635 |
| world_cup_2018 | 60 | 2018-06-14 | 2018-07-15 | FIFA World Cup | 0.928572 | 0.547252 | 0.203898 | -2.811234 | 0.957798 | 0.570447 | 0.218793 | -0.029225 | -0.023195 | -0.014895 |
| world_cup_2022 | 59 | 2022-11-20 | 2022-12-17 | FIFA World Cup | 0.999154 | 0.576717 | 0.216611 | -2.999681 | 1.019244 | 0.602083 | 0.232018 | -0.020089 | -0.025366 | -0.015407 |

## Paired Comparison Summary

| pair | metric | matches | mean_delta | ci_low | ci_high | bootstrap_iterations | random_seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dixon_coles_minus_elo | log_loss | 195 | -0.023793 | -0.059030 | 0.010593 | 10000 | 2026 |
| dixon_coles_minus_elo | brier_score | 195 | -0.020332 | -0.043173 | 0.002488 | 10000 | 2026 |
| dixon_coles_minus_elo | ranked_probability_score | 195 | -0.011239 | -0.021492 | -0.001364 | 10000 | 2026 |
| poisson_minus_dixon_coles | log_loss | 195 | -0.000281 | -0.000692 | 0.000148 | 10000 | 2026 |
| poisson_minus_dixon_coles | brier_score | 195 | -0.000214 | -0.000450 | 0.000037 | 10000 | 2026 |
| poisson_minus_dixon_coles | ranked_probability_score | 195 | -0.000043 | -0.000087 | 0.000002 | 10000 | 2026 |
| poisson_minus_elo | log_loss | 195 | -0.024075 | -0.059141 | 0.010567 | 10000 | 2026 |
| poisson_minus_elo | brier_score | 195 | -0.020546 | -0.043538 | 0.002061 | 10000 | 2026 |
| poisson_minus_elo | ranked_probability_score | 195 | -0.011281 | -0.021297 | -0.001455 | 10000 | 2026 |

Validation reuses the Elo temporal folds. The 2026 rows are a retrospective holdout, not prospective predictions, and are not used for model selection.
