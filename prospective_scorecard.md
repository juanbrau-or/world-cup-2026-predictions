# Prospective Scorecard

Generated UTC: 2026-07-10T03:31:57Z
Results cutoff UTC: 2026-07-10T03:31:47.503980Z
1X2 metric basis: 90-minute result (`result_90`). Extra time and penalties are reported separately and are not mixed into the 1X2 metric.

## Official Policy

Policy: early_v1 (early_v1_2026_06_30)
Context: early_v1
Primary rule: latest_valid_at_least_6h_before_kickoff at >= 6.0 hours
Fallback rule: earliest_valid_before_kickoff

> Sample is too small for firm statistical conclusions. Reported aggregates are monitoring diagnostics, not evidence of model improvement.

## Metrics

Official matches evaluated: 21

| Metric | Value |
| --- | ---: |
| log loss | 0.849582 |
| Brier score | 0.490821 |
| RPS | 0.164837 |
| accuracy | 0.714286 |
| calibration error | n/a |
| mean hours before kickoff | 7.871402 |
| median hours before kickoff | 8.025833 |

## Baselines

| Baseline | Status | Matches | Log loss | Brier | RPS | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| uniform_1x2 | computed | 21 | 1.098612 | 0.666667 | 0.246032 | 0.476190 |
| historical_frequency_frozen | computed | 21 | 1.041672 | 0.629226 | 0.236930 | 0.476190 |
| elo_operational | not_available | 21 | n/a | n/a | n/a | n/a |

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
| 2026-07-04T21:00:00Z | Paraguay vs France | away_win | away_win | latest_valid_at_least_6h_before_kickoff | 0.643298 |
| 2026-07-05T20:00:00Z | Brazil vs Norway | home_win | away_win | latest_valid_at_least_6h_before_kickoff | 0.191035 |
| 2026-07-06T01:00:00Z | Mexico vs England | away_win | away_win | latest_valid_at_least_6h_before_kickoff | 0.509034 |
| 2026-07-06T19:00:00Z | Portugal vs Spain | away_win | away_win | latest_valid_at_least_6h_before_kickoff | 0.472595 |
| 2026-07-07T00:00:00Z | United States vs Belgium | away_win | away_win | latest_valid_at_least_6h_before_kickoff | 0.562494 |
| 2026-07-07T16:00:00Z | Argentina vs Egypt | home_win | home_win | latest_valid_at_least_6h_before_kickoff | 0.678693 |
| 2026-07-07T20:00:00Z | Switzerland vs Colombia | away_win | draw | latest_valid_at_least_6h_before_kickoff | 0.240988 |
| 2026-07-09T20:00:00Z | France vs Morocco | home_win | home_win | latest_valid_at_least_6h_before_kickoff | 0.394904 |
