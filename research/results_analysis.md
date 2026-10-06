# Results analysis

All reported findings come from `experiments/results/metrics.csv`, `controls_metrics.csv`, saved per-sentence probabilities, and analysis scripts executed on 6 October 2026. Means and sample standard deviations are in `summary.csv`; machine-readable values are in `facts.json`. Results are for SST five-level root sentences, not Amazon ratings or transformers.

## Main hypotheses

H1 receives qualified support at the full budget: CE argmax MAE is 0.8299 and quadratic distance-penalized CE (DP2) is 0.8032. Severe errors fall from 18.82% to 17.19%. Accuracy increases only from 40.77% to 41.04%, while macro-F1 falls from 35.06% to 34.27%. This is an ordinal trade-off, not a comprehensive classification improvement. The conditional paired bootstrap MAE difference is -0.0267, with 95% interval [-0.0407, -0.0127]. Severe-error difference is -1.629 percentage points, interval [-2.308, -0.995]. Accuracy difference has an interval containing zero. These intervals concern resampling the shared test sentences with fitted models fixed, not new training corpora.

H2 is condition-dependent. Full-budget DP2 reduces NLL from 1.3754 to 1.3588 and ECE15 from 3.374% to 2.139%. CE+TS reaches NLL 1.3686 and ECE15 2.375%. Thus DP2 has a small full-budget probability-score benefit. At 2,000 examples, DP2 has slightly lower MAE (0.9768 versus 0.9902) but higher ECE15 (2.142% versus 1.739%). This contradicts a universal claim that distance penalties improve calibration.

H3 is not uniformly true for binned ECE. Temperature scaling leaves all argmax predictions unchanged and slightly improves full-budget CE NLL, but its ECE effect depends on bins and label budget. It fits calibration-set NLL, not ECE. Full-budget CE temperatures are below one, so this baseline is not simply an overconfidence story. Do not claim that every model is overconfident or that scaling always helps.

## Which baseline is best?

Cumulative logistic regression beats DP2 on full-budget argmax MAE (0.7833 versus 0.8032), severe-error rate (15.70% versus 17.19%), and accuracy (41.18% versus 41.04%), although DP2 has lower NLL. CE posterior-median decisions obtain MAE 0.8023 and severe errors 14.21%, but accuracy falls to 35.38% and central predictions rise to 41.99%. This shows that ordinal error can change through the decision rule alone. DP2's central fraction rises only from 10.09% to 11.99%; it does not collapse entirely to neutral.

At 2,000 examples, cumulative logistic regression again obtains lower MAE than DP2. No candidate is best on every metric. Optimizing accuracy, macro-F1, distance, and confidence produces different rankings.

## Why OLL is suspicious, and what the investigation found

Raw OLL has full-budget NLL 11.713 despite MAE 0.7905. Numerical gradients were independently checked against finite differences; probability normalization and metrics passed; all reported optimizers converged (with documented numerical continuation where necessary). Its fitted test predictions never choose either extreme label. The probabilities assigned to extreme labels are extremely small. NLL is computed by sklearn with machine-epsilon clipping, so 11.713 is already a clipped probability penalty.

An analytic risk diagnostic using the actual training class proportions explains a possible mechanism: for constant inputs, expected OLL with exponent one is `-sum_k a_k log(1-p_k)`, where `a_k` is expected absolute distance to class k. Its simplex solution is `p_k=max(0,1-a_k/nu)` with nu chosen so probabilities sum to one. The executed diagnostic yields [0, 0.2792, 0.4029, 0.3179, 0], whereas CE's optimum preserves the empirical class proportions. This is a mathematical diagnostic, not a second dataset or proof of the mechanism in every trained feature vector. An unpenalized intercept permits approaching these boundary distributions.

The separate OLL regularization sweep improves raw NLL to 5.649 at the full budget but lowers accuracy to 30.00%; subsequent scaling gives NLL 1.600, still worse than CE. Temperatures reach the predefined upper bound. Therefore the evidence does not support attributing the entire problem merely to a mismatched regularization strength, nor claiming temperature scaling repairs OLL here. The result is specific to this linear representation and exponent-one OLL, not a refutation of previously reported transformer results.

## Controls, seed consistency, and confounds

The secondary controls were written in `reviewer_control_plan.md` before being run. They are exploratory additions after primary outcomes.

- CE selected by MAE gives full-budget MAE 0.8293; DP2's improvement remains. The two selection criteria alone do not explain it.
- DP2 selected by NLL chooses lambda=2 in all full-budget runs and matches the primary DP2 model up to numerical precision.
- Fixed lambda=0.5 gives MAE 0.8217, a smaller benefit. The chosen weight is at the edge of the original grid; there is no evidence that it is globally optimal. No grid expansion based on test outcomes was performed.
- At low budget, lambda selection and sampled examples vary, and calibration improvements are inconsistent. Full-budget CE and primary DP2 use identical training data, deterministic zero initialization, and identical selected settings across seeds. Their zero standard deviations are expected, not evidence of robustness to random initialization. Temperature and NB selection can still vary with the development halves.
- ECE15 favors full-budget DP2 over CE, but ECE5 favors DP1 over DP2. ECE30 favors the cumulative baseline. Aggregate ECE is insufficient for a universal ranking.

## Class-level analysis

DP2's aggregate gains hide worse probability scores for both extreme classes and a worse MAE for the most positive class. The row-normalized confusion figure and class diagnostics show that it predicts fewer extreme labels. Central and moderately positive/negative labels dominate the corpus; the reduced macro-F1 warns against equating aggregate ordinal gains with improvement for every label.

## Conclusions not supported

No evidence for novelty of the loss, state-of-the-art accuracy, generalization to product ratings or unseen domains, transformer behavior, causal effects, improved confidence for every class, population-level significance, or elimination of severe errors. One corpus and two budgets support a narrow evaluation. The study supports measuring ordinal costs and probability quality separately and comparing against an ordinal baseline and a changed decision rule.
