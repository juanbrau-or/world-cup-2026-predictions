# Prospective Scorecard

Generated UTC: 2026-07-02T07:35:39Z
Results cutoff UTC: 2026-07-02T07:35:28.558834Z
1X2 metric basis: 90-minute result (`result_90`). Extra time and penalties are reported separately and are not mixed into the 1X2 metric.

## Official Policy

Policy: early_v1 (early_v1_2026_06_30)
Context: early_v1
Primary rule: latest_valid_at_least_6h_before_kickoff at >= 6.0 hours
Fallback rule: earliest_valid_before_kickoff

> Sample is too small for firm statistical conclusions. Reported aggregates are monitoring diagnostics, not evidence of model improvement.

## Metrics

Official matches evaluated: 6

| Metric | Value |
| --- | ---: |
| log loss | 0.781565 |
| Brier score | 0.436296 |
| RPS | 0.145557 |
| accuracy | 0.666667 |
| calibration error | n/a |
| mean hours before kickoff | 7.270880 |
| median hours before kickoff | 6.812917 |

## Baselines

| Baseline | Status | Matches | Log loss | Brier | RPS | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| uniform_1x2 | computed | 6 | 1.098612 | 0.666667 | 0.250000 | 0.666667 |
| historical_frequency_frozen | computed | 6 | 0.935359 | 0.550119 | 0.203932 | 0.666667 |
| elo_operational | not_available | 6 | n/a | n/a | n/a | n/a |

## Matches

| Kickoff UTC | Match | Pick | Actual 90 | Rule | Log-loss input |
| --- | --- | --- | --- | --- | ---: |
| 2026-06-30T17:00:00Z | Ivory Coast vs Norway | away_win | away_win | latest_valid_at_least_6h_before_kickoff | 0.449251 |
| 2026-06-30T21:00:00Z | France vs Sweden | home_win | home_win | latest_valid_at_least_6h_before_kickoff | 0.695163 |
| 2026-07-01T02:00:00Z | Mexico vs Ecuador | away_win | home_win | latest_valid_at_least_6h_before_kickoff | 0.307169 |
| 2026-07-01T16:00:00Z | England vs Congo DR | home_win | home_win | latest_valid_at_least_6h_before_kickoff | 0.617803 |
| 2026-07-01T20:00:00Z | Belgium vs Senegal | home_win | draw | latest_valid_at_least_6h_before_kickoff | 0.243586 |
| 2026-07-02T00:00:00Z | United States vs Bosnia-Herzegovina | home_win | home_win | latest_valid_at_least_6h_before_kickoff | 0.636751 |
