# Verified references

Verified against the linked primary publication pages on 6 October 2026. The candidate PDF's bibliography is not treated as verified evidence. Numerical comparisons with prior papers are not used in our results tables.

| Title | Authors | Year / source | Identifier | Relevance |
|---|---|---|---|---|
| Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank | Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, Christopher Potts | 2013, EMNLP, pp. 1631-1642 | https://aclanthology.org/D13-1170/ | Official SST dataset and five-level sentiment task. |
| A Simple Log-based Loss Function for Ordinal Text Classification | Francois Castagnos, Martin Mihelich, Charles Dognin | 2022, COLING, pp. 4604-4609 | https://aclanthology.org/2022.coling-1.407/ | Direct ordinal-loss prior work; Eq. 2 gives the OLL baseline. It explicitly discusses expected-distance losses, preventing a novelty claim for our penalty. |
| On Calibration of Modern Neural Networks | Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger | 2017, ICML, PMLR 70, pp. 1321-1330 | https://proceedings.mlr.press/v70/guo17a.html ; arXiv:1706.04599 | Confidence calibration, ECE, and temperature scaling control. Their neural-network findings are not assumed to hold for our sparse models. |
| When Does Label Smoothing Help? | Rafael Muller, Simon Kornblith, Geoffrey E. Hinton | 2019, NeurIPS 32 | https://proceedings.neurips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html ; arXiv:1906.02629 | Nonordinal probability regularization baseline. |
| Squared Earth Mover's Distance-based Loss for Training Deep Neural Networks | Le Hou, Chen-Ping Yu, Dimitris Samaras | 2016 preprint, revised 2017; arXiv | https://arxiv.org/abs/1611.05916 | Establishes that exploiting class distances is prior art. Our expected-distance term is not this squared cumulative-distribution objective. |
| The Multilingual Amazon Reviews Corpus | Phillip Keung, Yichao Lu, Gyorgy Szarvas, Noah A. Smith | 2020, EMNLP, pp. 4563-4568 | https://aclanthology.org/2020.emnlp-main.369/ ; DOI:10.18653/v1/2020.emnlp-main.369 | Motivates ordinal evaluation of review ratings and a possible future second corpus; not evaluated in the primary experiment. |

## Synthesis

Ordinal text classification is an established field. Castagnos et al. directly evaluate OLL on SST-5 and other tasks, and identify drawbacks of pure expected-distance objectives. We therefore retain CE and add a controlled penalty rather than presenting distance-aware training as novel. Hou et al. show a different distribution-aware distance formulation. Calibration is a distinct objective: Guo et al. fit a positive scalar temperature without changing argmax labels, while Muller et al. train with soft targets. Our study measures ordinal and probability metrics together in the same sparse representation, and includes CE posterior-median decisions to check whether changing the decision rule can explain ordinal gains.

## Source-document corrections

The PDF attributes `2025.uncertainlp-main.2` to Balasubramanian et al.; its primary page lists Aneesh Durai and the title *Phases of Uncertainty: Confidence-Calibration Dynamics in Language Model Training*. This is not cited in the paper because training-phase transformer calibration is outside our selected scope. Claims that ordinal losses are almost universally overlooked, that a first application is established, or that attention gives causal token effects are not adopted. No unverified bibliography entries are included.
