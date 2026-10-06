# Research plan (written before training)

Date: 6 October 2026. Tentative title: **Ordinal Error and Probability Calibration in Lightweight Five-Level Sentiment Classification**.

## Problem, motivation, and scope
Adjacent sentiment errors and extreme polarity errors have different ordinal costs, while trustworthy confidence is a separate requirement. Existing ordinal classification losses and temperature scaling address different objectives. The narrow empirical gap investigated here is how these objectives interact with sparse CPU-trained classifiers under fixed representations and limited labels. This does not establish that earlier papers overlooked ordinal classification or introduce a new algorithm.

Primary question: Does adding expected ordinal distance to cross-entropy reduce severe test errors while also improving probability calibration compared with cross-entropy and its temperature-scaled version?

H1: The validation-selected quadratic distance penalty reduces argmax MAE and the rate of errors of at least two levels relative to CE.
H2: It also reduces 15-bin ECE and NLL relative to CE. H1 and H2 can diverge.
H3 (secondary): Temperature scaling improves probability quality without changing argmax decisions. A CE posterior-median decision provides a training-free ordinal control.

## Variables and data
- Independent variables: objective (CE, CE+linear distance, CE+quadratic distance, ordinal log-loss, smoothed CE), post-hoc scaling, decision rule, training budget (2,000 versus full), seed (42, 123, 999).
- Dependent variables: accuracy, macro-F1, argmax MAE, severe error rate (absolute error >=2), ECE (5/10/15/30 equal-width bins), multiclass Brier (sum convention), NLL, ranked probability score (mean over four thresholds), central prediction fraction, fit time.
- Dataset: official Stanford Sentiment Treebank root sentences only, five sentiment intervals. Preserve official train/dev/test boundaries; remove duplicate normalized texts within splits and from lower-priority splits (test > dev > train). Exclude conflicting-label text groups from all splits. Log all changes and identifiers. Never train on phrase descendants.
- Split each official development set into stratified model-selection and temperature-calibration halves separately per seed. Train subsets sampled stratified without replacement. Full-budget variation primarily measures split/selection variability, not independent data replication.

## Baselines and proposed comparison
- Simple: multinomial naive Bayes on common TF-IDF features, alpha selected from {0.1, 1, 10} by selection NLL.
- Stronger: multinomial linear softmax CE, regularization C selected from {1, 4, 16} by selection NLL.
- Ordinal baseline: four cumulative binary logistic classifiers; enforce non-increasing threshold probabilities by cumulative minimum; same CE-selected C. This is a cumulative reduction, not proportional-odds regression.
- Prior-work baseline: ordinal log-loss with raw distance exponent 1; same regularization strength as CE. State that loss scale is different and regularization equivalence is imperfect.
- Main comparison: CE + lambda * sum_k p_k (|k-y|/4)^q; q=2 primary, q=1 ablation, lambda in {0.1, 0.5, 1, 2}. Choose by selection MAE, ties by NLL. Shared CE-selected C.
- Calibration control: label smoothing epsilon=0.1; same C. Temperature scaling for CE, selected distance models, OLL, and smoothing, fitted only on calibration labels by NLL. Bound temperature to [0.05, 20].
- Decision control: posterior median of unscaled CE probabilities minimizes expected absolute error; report decision metrics separately because posterior probabilities are unchanged.

## Protocol
Common lowercased word unigram/bigram TF-IDF, min_df=2, max_features=12,000, sublinear TF, L2 normalized; vocabulary and IDF fitted only on the sampled training split. Keep sklearn default token pattern (two or more word characters). Intercept unpenalized. Analytic-gradient L-BFGS optimization, maxiter=400, ftol=1e-10, gtol=1e-6. Any nonconvergence triggers investigation and a documented numerical repair, not test-based tuning. Three seeds; matched data across methods. Save selections before scoring test predictions. Retain candidate validation records, final test probabilities, model weights, split IDs, dataset hash, versions, configuration, and hardware.

Means and sample standard deviations describe runs. Primary paired bootstrap: resample test sentence indices, preserving matched method predictions, average the three seed-specific per-example differences, 5,000 resamples with seed 2026; report conditional 95% intervals for full-budget quadratic penalty versus CE and versus CE+TS. These do not cover model-selection or corpus variation. No t-tests on three correlated seeds and no universal significance claims. ECE-bin and classwise analyses are secondary and descriptive.

Before experiments, check analytic gradients numerically, hand-calculated calibration/severity cases, probability normalization, threshold monotonicity, and duplicate/split disjointness. Then inspect convergence and predictions for central-class collapse. All quantitative paper values must be generated from saved files. Test outcomes will not change the primary lambda grid or selection criteria.

## Limitations and threats
One corpus, sentence-level sentiment rather than product star ratings; subjective labels and artificial equal spacing; sparse features miss composition; no pretrained encoders or domain transfer; different losses have different objective scales; limited tuning budget; MAE-based selection may trade accuracy/calibration; development halves small; seeds share test data and full training data; test bootstrap is conditional; duplicates are audited only under exact normalized matching. Numerical cost is measured on this shared machine and is not a hardware benchmark.
