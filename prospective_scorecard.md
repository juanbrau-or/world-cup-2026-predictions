# Prospective Scorecard

Generated UTC: 2026-07-01T01:04:37Z
Results cutoff UTC: 2026-07-01T01:04:27.996693Z
1X2 metric basis: 90-minute result (`result_90`). Extra time and penalties are reported separately and are not mixed into the 1X2 metric.

## Official Policy

Policy: early_v1 (early_v1_2026_06_30)
Context: early_v1
Primary rule: latest_valid_at_least_6h_before_kickoff at >= 6.0 hours
Fallback rule: earliest_valid_before_kickoff

> Sample is too small for firm statistical conclusions. Reported aggregates are monitoring diagnostics, not evidence of model improvement.

## Metrics

Official matches evaluated: 2

| Metric | Value |
| --- | ---: |
| log loss | 0.581892 |
| Brier score | 0.297967 |
| RPS | 0.126900 |
| accuracy | 1.000000 |
| calibration error | n/a |
| mean hours before kickoff | 6.302917 |
| median hours before kickoff | 6.302917 |

## Baselines

| Baseline | Status | Matches | Log loss | Brier | RPS | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| uniform_1x2 | computed | 2 | 1.098612 | 0.666667 | 0.277778 | 0.500000 |
| historical_frequency_frozen | computed | 2 | 0.981986 | 0.595681 | 0.272598 | 0.500000 |
| elo_operational | not_available | 2 | n/a | n/a | n/a | n/a |

## Matches

| Kickoff UTC | Match | Pick | Actual 90 | Rule | Log-loss input |
| --- | --- | --- | --- | --- | ---: |
| 2026-06-30T17:00:00Z | Ivory Coast vs Norway | away_win | away_win | latest_valid_at_least_6h_before_kickoff | 0.449251 |
| 2026-06-30T21:00:00Z | France vs Sweden | home_win | home_win | latest_valid_at_least_6h_before_kickoff | 0.695163 |
