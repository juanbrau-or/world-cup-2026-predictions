# Elo Backtest Report

Selected method: `ordinal_logistic`
Validation matches: 195
Prospective 2026 matches: 287

## Selected Rating Parameters

```json
{
  "competition_importance": {
    "confederation_championship": 3.0,
    "confederation_qualifier": 2.0,
    "friendly": 1.0,
    "nations_league": 2.0,
    "other": 1.0,
    "other_official": 2.0,
    "world_cup": 3.0,
    "world_cup_qualifier": 2.0
  },
  "competition_weight_profile": "compressed",
  "home_advantage": 50.0,
  "k_base": 20.0,
  "margin_of_victory": {
    "enabled": true,
    "goal_difference_weight": 0.15,
    "name": "modest"
  },
  "rating_regression_after_inactivity": {
    "enabled": false,
    "inactivity_days": 365,
    "name": "off",
    "regression_fraction": 0.0
  }
}
```

## Fold Metrics

| fold | matches | log_loss | brier_score | ranked_probability_score | calibration_error | accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| copa_america_2024 | 28 | 0.779037 | 0.442461 | 0.152548 | 0.107090 | 0.678571 |
| euro_2024 | 48 | 1.032136 | 0.614799 | 0.200547 | 0.132956 | 0.541667 |
| world_cup_2018 | 60 | 0.957798 | 0.570447 | 0.218793 | 0.102984 | 0.583333 |
| world_cup_2022 | 59 | 1.019244 | 0.602083 | 0.232018 | 0.071798 | 0.559322 |

## Prospective 2026 Metrics

| matches | log_loss | brier_score | ranked_probability_score | calibration_error | accuracy |
| --- | --- | --- | --- | --- | --- |
| 287 | 0.874914 | 0.514345 | 0.174846 | 0.038763 | 0.602787 |

## Segment Metrics

| evaluation_set | segment | segment_value | matches | log_loss | brier_score | ranked_probability_score | calibration_error | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | friendly_official | official | 195 | 0.969019 | 0.572559 | 0.208791 | 0.047288 | 0.579487 |
| validation | neutral | False | 14 | 0.858030 | 0.500337 | 0.206503 | 0.219273 | 0.714286 |
| validation | neutral | True | 181 | 0.977604 | 0.578145 | 0.208968 | 0.048681 | 0.569061 |
| validation | year | 2018 | 60 | 0.957798 | 0.570447 | 0.218793 | 0.102984 | 0.583333 |
| validation | year | 2022 | 59 | 1.019244 | 0.602083 | 0.232018 | 0.071798 | 0.559322 |
| validation | year | 2024 | 76 | 0.938889 | 0.551306 | 0.182863 | 0.096940 | 0.592105 |
| validation | elo_difference | -150_to_-50 | 35 | 1.033294 | 0.623934 | 0.224510 | 0.060429 | 0.514286 |
| validation | elo_difference | -300_to_-150 | 19 | 0.765201 | 0.425129 | 0.166172 | 0.050341 | 0.736842 |
| validation | elo_difference | -50_to_50 | 42 | 1.047371 | 0.631000 | 0.225910 | 0.098954 | 0.500000 |
| validation | elo_difference | 150_to_300 | 39 | 0.815933 | 0.465496 | 0.158129 | 0.027213 | 0.692308 |
| validation | elo_difference | 50_to_150 | 42 | 1.030305 | 0.618429 | 0.231747 | 0.064156 | 0.523810 |
| validation | elo_difference | <-300 | 5 | 1.194560 | 0.667668 | 0.330143 | 0.247211 | 0.600000 |
| validation | elo_difference | >300 | 13 | 1.015242 | 0.597324 | 0.204597 | 0.159543 | 0.615385 |
| validation | tournament | Copa América | 28 | 0.779037 | 0.442461 | 0.152548 | 0.107090 | 0.678571 |
| validation | tournament | FIFA World Cup | 119 | 0.988263 | 0.586132 | 0.225350 | 0.068336 | 0.571429 |
| validation | tournament | UEFA Euro | 48 | 1.032136 | 0.614799 | 0.200547 | 0.132956 | 0.541667 |
| prospective_2026 | friendly_official | friendly | 208 | 0.886629 | 0.526495 | 0.169166 | 0.047563 | 0.576923 |
| prospective_2026 | friendly_official | official | 79 | 0.844071 | 0.482354 | 0.189801 | 0.111160 | 0.670886 |
| prospective_2026 | neutral | False | 169 | 0.827626 | 0.486380 | 0.161133 | 0.037978 | 0.644970 |
| prospective_2026 | neutral | True | 118 | 0.942641 | 0.554395 | 0.194485 | 0.078546 | 0.542373 |
| prospective_2026 | year | 2026 | 287 | 0.874914 | 0.514345 | 0.174846 | 0.038763 | 0.602787 |
| prospective_2026 | elo_difference | -150_to_-50 | 40 | 1.048884 | 0.636082 | 0.220900 | 0.081148 | 0.425000 |
| prospective_2026 | elo_difference | -300_to_-150 | 17 | 0.946749 | 0.555936 | 0.178487 | 0.132660 | 0.647059 |
| prospective_2026 | elo_difference | -50_to_50 | 59 | 1.056119 | 0.639964 | 0.226618 | 0.062200 | 0.440678 |
| prospective_2026 | elo_difference | 150_to_300 | 52 | 0.719007 | 0.414023 | 0.121182 | 0.058506 | 0.730769 |
| prospective_2026 | elo_difference | 50_to_150 | 62 | 0.980448 | 0.582399 | 0.211495 | 0.048608 | 0.548387 |
| prospective_2026 | elo_difference | <-300 | 7 | 0.486951 | 0.235565 | 0.111006 | 0.142383 | 0.857143 |
| prospective_2026 | elo_difference | >300 | 50 | 0.583090 | 0.313558 | 0.094975 | 0.039314 | 0.820000 |
| prospective_2026 | tournament | African Cup of Nations | 13 | 0.576328 | 0.304854 | 0.132919 | 0.163977 | 0.769231 |
| prospective_2026 | tournament | Baltic Cup | 3 | 0.684550 | 0.386109 | 0.179674 | 0.315525 | 0.666667 |
| prospective_2026 | tournament | Diamond Jubilee International Football Tournament | 4 | 1.276498 | 0.847740 | 0.411241 | 0.471001 | 0.250000 |
| prospective_2026 | tournament | FIFA Series | 27 | 0.699830 | 0.376302 | 0.179638 | 0.172673 | 0.740741 |
| prospective_2026 | tournament | FIFA World Cup | 20 | 1.197232 | 0.712008 | 0.199938 | 0.264596 | 0.550000 |
| prospective_2026 | tournament | FIFA World Cup qualification | 12 | 0.765800 | 0.432772 | 0.186113 | 0.215321 | 0.750000 |
| prospective_2026 | tournament | Friendly | 208 | 0.886629 | 0.526495 | 0.169166 | 0.047563 | 0.576923 |

Validation uses only matches strictly before each fold start. The 2026 holdout is excluded from parameter selection.
