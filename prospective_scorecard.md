# Prospective Scorecard

Generated UTC: 2026-07-01T21:40:19Z
Results cutoff UTC: 2026-07-01T21:40:07.099957Z
1X2 metric basis: 90-minute result (`result_90`). Extra time and penalties are reported separately and are not mixed into the 1X2 metric.

## Official Policy

Policy: early_v1 (early_v1_2026_06_30)
Context: early_v1
Primary rule: latest_valid_at_least_6h_before_kickoff at >= 6.0 hours
Fallback rule: earliest_valid_before_kickoff

> Sample is too small for firm statistical conclusions. Reported aggregates are monitoring diagnostics, not evidence of model improvement.

## Metrics

Official matches evaluated: 4

| Metric | Value |
| --- | ---: |
| log loss | 0.706432 |
| Brier score | 0.385844 |
| RPS | 0.160400 |
| accuracy | 0.750000 |
| calibration error | n/a |
| mean hours before kickoff | 7.263403 |
| median hours before kickoff | 6.812917 |

## Baselines

| Baseline | Status | Matches | Log loss | Brier | RPS | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| uniform_1x2 | computed | 4 | 1.098612 | 0.666667 | 0.277778 | 0.750000 |
| historical_frequency_frozen | computed | 4 | 0.850183 | 0.495796 | 0.222656 | 0.750000 |
| elo_operational | not_available | 4 | n/a | n/a | n/a | n/a |

## Matches

| Kickoff UTC | Match | Pick | Actual 90 | Rule | Log-loss input |
| --- | --- | --- | --- | --- | ---: |
| 2026-06-30T17:00:00Z | Ivory Coast vs Norway | away_win | away_win | latest_valid_at_least_6h_before_kickoff | 0.449251 |
| 2026-06-30T21:00:00Z | France vs Sweden | home_win | home_win | latest_valid_at_least_6h_before_kickoff | 0.695163 |
| 2026-07-01T01:00:00Z | Mexico vs Ecuador | away_win | home_win | latest_valid_at_least_6h_before_kickoff | 0.307169 |
| 2026-07-01T16:00:00Z | England vs Congo DR | home_win | home_win | latest_valid_at_least_6h_before_kickoff | 0.617803 |
