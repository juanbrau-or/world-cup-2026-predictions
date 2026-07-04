# Prospective Scorecard

Generated UTC: 2026-07-04T21:00:36Z
Results cutoff UTC: 2026-07-04T21:00:25.905569Z
1X2 metric basis: 90-minute result (`result_90`). Extra time and penalties are reported separately and are not mixed into the 1X2 metric.

## Official Policy

Policy: early_v1 (early_v1_2026_06_30)
Context: early_v1
Primary rule: latest_valid_at_least_6h_before_kickoff at >= 6.0 hours
Fallback rule: earliest_valid_before_kickoff

> Sample is too small for firm statistical conclusions. Reported aggregates are monitoring diagnostics, not evidence of model improvement.

## Metrics

Official matches evaluated: 13

| Metric | Value |
| --- | ---: |
| log loss | 0.846534 |
| Brier score | 0.490298 |
| RPS | 0.152514 |
| accuracy | 0.692308 |
| calibration error | n/a |
| mean hours before kickoff | 7.686645 |
| median hours before kickoff | 7.822222 |

## Baselines

| Baseline | Status | Matches | Log loss | Brier | RPS | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| uniform_1x2 | computed | 13 | 1.098612 | 0.666667 | 0.239316 | 0.615385 |
| historical_frequency_frozen | computed | 13 | 0.978258 | 0.578696 | 0.200572 | 0.615385 |
| elo_operational | not_available | 13 | n/a | n/a | n/a | n/a |

## Matches

| Kickoff UTC | Match | Pick | Actual 90 | Rule | Log-loss input |
| --- | --- | --- | --- | --- | ---: |
| 2026-06-30T17:00:00Z | Ivory Coast vs Norway | away_win | away_win | latest_valid_at_least_6h_before_kickoff | 0.449251 |
| 2026-06-30T21:00:00Z | France vs Sweden | home_win | home_win | latest_valid_at_least_6h_before_kickoff | 0.695163 |
| 2026-07-01T02:00:00Z | Mexico vs Ecuador | away_win | home_win | latest_valid_at_least_6h_before_kickoff | 0.307169 |
| 2026-07-01T16:00:00Z | England vs Congo DR | home_win | home_win | latest_valid_at_least_6h_before_kickoff | 0.617803 |
| 2026-07-01T20:00:00Z | Belgium vs Senegal | home_win | draw | latest_valid_at_least_6h_before_kickoff | 0.243586 |
| 2026-07-02T00:00:00Z | United States vs Bosnia-Herzegovina | home_win | home_win | latest_valid_at_least_6h_before_kickoff | 0.636751 |
| 2026-07-02T19:00:00Z | Spain vs Austria | home_win | home_win | latest_valid_at_least_6h_before_kickoff | 0.649742 |
| 2026-07-02T23:00:00Z | Portugal vs Croatia | home_win | home_win | latest_valid_at_least_6h_before_kickoff | 0.558464 |
| 2026-07-03T03:00:00Z | Switzerland vs Algeria | home_win | home_win | latest_valid_at_least_6h_before_kickoff | 0.431648 |
| 2026-07-03T18:00:00Z | Australia vs Egypt | home_win | draw | latest_valid_at_least_6h_before_kickoff | 0.311112 |
| 2026-07-03T22:00:00Z | Argentina vs Cape Verde Islands | home_win | draw | latest_valid_at_least_6h_before_kickoff | 0.105834 |
| 2026-07-04T01:30:00Z | Colombia vs Ghana | home_win | home_win | latest_valid_at_least_6h_before_kickoff | 0.751287 |
| 2026-07-04T17:00:00Z | Canada vs Morocco | away_win | away_win | latest_valid_at_least_6h_before_kickoff | 0.466634 |
