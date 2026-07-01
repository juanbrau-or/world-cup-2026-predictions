# Contextual Challenger Evaluation

Official baseline remains: `poisson_goal_v1`
Selected shadow model: `contextual_logit_v1`
Promotion status: `shadow_monitoring`
Holdout 2026 matches scored: 287

## Aggregate Metrics

| model_name | ablation | matches | log_loss | brier_score | ranked_probability_score | calibration_error | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| contextual_lgbm_v1 | lgbm_contextual | 195 | 0.939507 | 0.552279 | 0.200252 | 0.054147 | 0.589744 |
| contextual_lgbm_v1 | lgbm_stack | 195 | 0.962393 | 0.565501 | 0.204165 | 0.083664 | 0.589744 |
| contextual_logit_v1 | contextual_logistic | 195 | 0.938325 | 0.551183 | 0.199956 | 0.048092 | 0.579487 |
| contextual_logit_v1 | logistic_stack | 195 | 0.948271 | 0.554792 | 0.200639 | 0.051040 | 0.600000 |
| poisson_goal_v1 | poisson_official | 195 | 0.944945 | 0.552013 | 0.197509 | 0.062019 | 0.584615 |

## Ablations

| ablation | model_name | include_contextual | matches | log_loss | brier_score | ranked_probability_score |
| --- | --- | --- | --- | --- | --- | --- |
| poisson_official | poisson_goal_v1 | False | 195 | 0.944945 | 0.552013 | 0.197509 |
| logistic_stack | contextual_logit_v1 | False | 195 | 0.927947 | 0.541046 | 0.193302 |
| contextual_logistic | contextual_logit_v1 | True | 195 | 0.920344 | 0.538481 | 0.192983 |
| lgbm_stack | contextual_lgbm_v1 | False | 195 | 0.941357 | 0.550420 | 0.196264 |
| lgbm_contextual | contextual_lgbm_v1 | True | 195 | 0.919157 | 0.538196 | 0.192736 |

## Fold Metrics

| model_name | ablation | fold | matches | log_loss | brier_score | ranked_probability_score |
| --- | --- | --- | --- | --- | --- | --- |
| contextual_lgbm_v1 | lgbm_contextual | copa_america_2024 | 28 | 0.769905 | 0.430877 | 0.145587 |
| contextual_lgbm_v1 | lgbm_contextual | euro_2024 | 48 | 0.980364 | 0.592827 | 0.193890 |
| contextual_lgbm_v1 | lgbm_contextual | world_cup_2018 | 60 | 0.933795 | 0.558381 | 0.209412 |
| contextual_lgbm_v1 | lgbm_contextual | world_cup_2022 | 59 | 0.992565 | 0.570698 | 0.222057 |
| contextual_lgbm_v1 | lgbm_stack | copa_america_2024 | 28 | 0.785977 | 0.435183 | 0.145875 |
| contextual_lgbm_v1 | lgbm_stack | euro_2024 | 48 | 1.006663 | 0.608887 | 0.199547 |
| contextual_lgbm_v1 | lgbm_stack | world_cup_2018 | 60 | 0.945035 | 0.561903 | 0.210930 |
| contextual_lgbm_v1 | lgbm_stack | world_cup_2022 | 59 | 1.027753 | 0.595709 | 0.228705 |
| contextual_logit_v1 | contextual_logistic | copa_america_2024 | 28 | 0.784201 | 0.441141 | 0.150195 |
| contextual_logit_v1 | contextual_logistic | euro_2024 | 48 | 0.985977 | 0.588747 | 0.191370 |
| contextual_logit_v1 | contextual_logistic | world_cup_2018 | 60 | 0.928357 | 0.550645 | 0.208629 |
| contextual_logit_v1 | contextual_logistic | world_cup_2022 | 59 | 0.982838 | 0.573392 | 0.221737 |
| contextual_logit_v1 | logistic_stack | copa_america_2024 | 28 | 0.777338 | 0.436687 | 0.147754 |
| contextual_logit_v1 | logistic_stack | euro_2024 | 48 | 0.991293 | 0.591609 | 0.192644 |
| contextual_logit_v1 | logistic_stack | world_cup_2018 | 60 | 0.918942 | 0.542666 | 0.204833 |
| contextual_logit_v1 | logistic_stack | world_cup_2022 | 59 | 1.024216 | 0.593221 | 0.227975 |
| poisson_goal_v1 | poisson_official | copa_america_2024 | 28 | 0.816975 | 0.465024 | 0.158307 |
| poisson_goal_v1 | poisson_official | euro_2024 | 48 | 0.973426 | 0.578342 | 0.188912 |
| poisson_goal_v1 | poisson_official | world_cup_2018 | 60 | 0.928572 | 0.547252 | 0.203898 |
| poisson_goal_v1 | poisson_official | world_cup_2022 | 59 | 0.999154 | 0.576717 | 0.216611 |

## Paired Bootstrap

| pair | metric | matches | mean_delta | ci_low | ci_high | proportion_favorable |
| --- | --- | --- | --- | --- | --- | --- |
| contextual_lgbm_v1::lgbm_contextual_minus_poisson_goal_v1::poisson_official | log_loss | 195 | -0.005438 | -0.037242 | 0.028355 | 0.635800 |
| contextual_lgbm_v1::lgbm_contextual_minus_poisson_goal_v1::poisson_official | brier_score | 195 | 0.000266 | -0.019446 | 0.019480 | 0.496600 |
| contextual_lgbm_v1::lgbm_contextual_minus_poisson_goal_v1::poisson_official | ranked_probability_score | 195 | 0.002743 | -0.004106 | 0.009838 | 0.224000 |
| contextual_lgbm_v1::lgbm_stack_minus_poisson_goal_v1::poisson_official | log_loss | 195 | 0.017449 | -0.011955 | 0.047938 | 0.133000 |
| contextual_lgbm_v1::lgbm_stack_minus_poisson_goal_v1::poisson_official | brier_score | 195 | 0.013488 | -0.004351 | 0.030477 | 0.066600 |
| contextual_lgbm_v1::lgbm_stack_minus_poisson_goal_v1::poisson_official | ranked_probability_score | 195 | 0.006655 | 0.000466 | 0.013047 | 0.017200 |
| contextual_logit_v1::contextual_logistic_minus_poisson_goal_v1::poisson_official | log_loss | 195 | -0.006619 | -0.026129 | 0.013254 | 0.743400 |
| contextual_logit_v1::contextual_logistic_minus_poisson_goal_v1::poisson_official | brier_score | 195 | -0.000830 | -0.013294 | 0.012039 | 0.550400 |
| contextual_logit_v1::contextual_logistic_minus_poisson_goal_v1::poisson_official | ranked_probability_score | 195 | 0.002447 | -0.002217 | 0.007363 | 0.158400 |
| contextual_logit_v1::logistic_stack_minus_poisson_goal_v1::poisson_official | log_loss | 195 | 0.003326 | -0.019620 | 0.026831 | 0.403400 |
| contextual_logit_v1::logistic_stack_minus_poisson_goal_v1::poisson_official | brier_score | 195 | 0.002779 | -0.011834 | 0.018107 | 0.367800 |
| contextual_logit_v1::logistic_stack_minus_poisson_goal_v1::poisson_official | ranked_probability_score | 195 | 0.003130 | -0.002333 | 0.008856 | 0.123600 |

Feature importance is predictive diagnostics only; it is not causal evidence.
