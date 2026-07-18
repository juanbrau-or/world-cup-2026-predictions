# Prospective Scorecard

Generated UTC: 2026-07-18T17:00:37Z
Results cutoff UTC: 2026-07-18T17:00:27.304171Z
1X2 metric basis: 90-minute result (`result_90`). Extra time and penalties are reported separately and are not mixed into the 1X2 metric.

## Official Policy

Policy: early_v1 (early_v1_2026_06_30)
Context: early_v1
Primary rule: latest_valid_at_least_6h_before_kickoff at >= 6.0 hours
Fallback rule: earliest_valid_before_kickoff

> Sample is too small for firm statistical conclusions. Reported aggregates are monitoring diagnostics, not evidence of model improvement.

## Metrics

Official matches evaluated: 26

| Metric | Value |
| --- | ---: |
| log loss | 0.883106 |
| Brier score | 0.516600 |
| RPS | 0.166551 |
| accuracy | 0.692308 |
| calibration error | n/a |
| mean hours before kickoff | 7.955267 |
| median hours before kickoff | 8.035139 |

## Baselines

| Baseline | Status | Matches | Log loss | Brier | RPS | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| uniform_1x2 | computed | 26 | 1.098612 | 0.666667 | 0.239316 | 0.423077 |
| historical_frequency_frozen | computed | 26 | 1.079645 | 0.655530 | 0.238989 | 0.423077 |
| elo_operational | not_available | 26 | n/a | n/a | n/a | n/a |

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
| 2026-07-10T19:00:00Z | Spain vs Belgium | home_win | home_win | latest_valid_at_least_6h_before_kickoff | 0.549056 |
| 2026-07-11T21:00:00Z | Norway vs England | away_win | draw | latest_valid_at_least_6h_before_kickoff | 0.230692 |
| 2026-07-12T01:00:00Z | Argentina vs Switzerland | home_win | draw | latest_valid_at_least_6h_before_kickoff | 0.229109 |
| 2026-07-14T19:00:00Z | France vs Spain | away_win | away_win | latest_valid_at_least_6h_before_kickoff | 0.447977 |
| 2026-07-15T19:00:00Z | England vs Argentina | away_win | away_win | latest_valid_at_least_6h_before_kickoff | 0.459911 |
