# Strict internal reviewer report

This is a self-review, not external peer review. Scores are judgment on a 1 (weak) to 5 (strong) scale. Assessments concern an undergraduate empirical study, not publication acceptance.

| Criterion | Score | Assessment |
|---|---:|---|
| Novelty | 2 | Established loss family; contribution is a modest controlled evaluation, not an algorithm. |
| Technical correctness | 4 | Analytic gradients checked, documented clipping, loss derivation, converged fits, and split audits. Sparse modeling remains limited. |
| Experimental design | 4 | Matched features/data, explicit validation/calibration separation, strong ordinal baseline, decision control, repeated splits, and secondary selection controls. One corpus limits scope. |
| Reproducibility | 5 | Raw archive with hash, fixed dependencies, exact IDs, configurations, model artifacts, saved probabilities, generated figures/tables/macros, and replay verification. |
| Quality of analysis | 4 | Negative findings, objective/decision/calibration separation, classwise failures, bin sensitivity, and OLL diagnostic. Mechanistic explanations remain limited. |
| Clarity | 4 | Concrete question, modest claims, full metric definitions, distinctions between planned and secondary work. |
| Strength of evidence | 3 | Modest effect with matched conditional intervals; no independent corpus replication or pretrained comparison. |
| Limitations | 4 | Explicit constraints and evidence boundaries, including correlated seeds and boundary lambda. |

## Five strongest weaknesses and revisions

1. **The PDF overstates novelty and recommends transformers that were not run.** Fixed by selecting the CPU variant explicitly, citing direct OLL prior work, removing first-method/state-of-the-art claims, and identifying the work as a sparse-model study throughout. Remaining weakness: modest novelty and one corpus. A larger neural or independent-corpus replication is future work, not implied evidence.
2. **CE is selected by NLL but DP2 by MAE, confounding method with selection.** Fixed through executed CE-MAE, DP2-NLL, and fixed-lambda controls on the exact saved splits. Primary results remain unchanged; controls are labeled post-primary and exploratory.
3. **OLL's different loss scale could make its probability failure an unfair regularization comparison.** Fixed through a separately specified five-value C sweep, scaling each selected OLL variant on independent calibration labels, finite-difference loss checks, and a constant-feature risk diagnostic based on actual training priors. The failure remains in this setting, but claims about all OLL models are disallowed.
4. **Three seeds share a test set, and zero SD could be misread as independent replication; ECE depends on binning.** Fixed by naming what the seeds randomize, disclosing deterministic identical full-budget fits, reporting conditional paired test-sentence bootstrap intervals without decorative three-run tests, and analyzing 5/10/15/30-bin ECE plus proper probability scores.
5. **Aggregate ordinal improvement hides harmed extreme classes and a posterior-median explanation.** Fixed through an executed decision-rule control, central prediction fractions, classwise MAE/NLL, and row-normalized confusion matrices. Discussion now includes reduced macro-F1 and worse extreme-class probability scores.

## Remaining evidence limits

No independent-corpus, product-rating, domain-shift, or transformer claim is supported. No publication submission, faculty endorsement, author identity, or external review is implied. The selected lambda at the grid boundary does not establish a globally optimal value. Bootstrap intervals condition on fitted models and do not capture dataset/model-selection uncertainty. These limits cannot be removed by rewriting and are retained plainly in the manuscript.

Final acceptance criteria: compile the revised PDF; inspect every rendered page; check figures/tables/citations; replay metrics from saved predictions; verify generated numerical macros; preserve all result and failure logs. The final verification report records those checks after execution.
