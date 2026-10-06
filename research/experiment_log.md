# Experiment log

Date: 6 October 2026 (Asia/Kolkata).

## Data and numerical validation

Objective: reproduce official SST root labels and prevent exact-text leakage.
Archive SHA-256: `3f5209483b46bbf129cacbbbe6ae02fe780407034f61cf6342b7833257c3f1db`.
Original counts: {'train': 8544, 'dev': 1101, 'test': 2210}; audited counts: {'train': 8531, 'dev': 1100, 'test': 2210}.
Excluded 14 repeated rows; conflicting-label groups: 0.
Finite-difference gradients, known metric cases, scaling invariance, cumulative probabilities, and split-disjointness checks passed. Full error values are in validation_checks.json.

## Environment

Python: 3.14.7; CPU device; numerical-library threads: 1.
Packages: {'numpy': '2.5.3', 'scipy': '1.18.1', 'scikit-learn': '1.9.1', 'pandas': '3.0.6', 'matplotlib': '3.11.2'}. Hardware: AMD Ryzen 5 5500U, 12 logical CPUs; no detected NVIDIA GPU.

Protocol: research_plan.md and protocol_sha256.txt were written before model training. Numerical continuation is documented in protocol_amendments.md; initial failed output is preserved.

## Primary experiment

Objective: test whether expected ordinal distance improves ordinal error and calibration under a common sparse representation.
Configuration: experiments/configs/main.json; fitted TF-IDF per training sample; selection and calibration halves kept separate. Candidate records preserve every validation setting.
Completed primary candidate fits: 102. Fits using iteration continuation: 3.

| Run identifier | Method | Train size | Features | Accuracy | MAE | Severe rate | ECE15 | NLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| budget-2000_seed-42 | CE | 2000 | 5627 | 0.343891 | 0.992760 | 0.262896 | 0.008526 | 1.488651 |
| budget-2000_seed-42 | CE+TS | 2000 | 5627 | 0.343891 | 0.992760 | 0.262896 | 0.018303 | 1.487779 |
| budget-2000_seed-42 | CE-median | 2000 | 5627 | 0.281448 | 0.952489 | 0.217647 | 0.008526 | 1.488651 |
| budget-2000_seed-42 | NB | 2000 | 5627 | 0.349321 | 0.986425 | 0.252489 | 0.023212 | 1.526991 |
| budget-2000_seed-42 | Cumulative | 2000 | 5627 | 0.358824 | 0.933484 | 0.230769 | 0.014077 | 1.499938 |
| budget-2000_seed-42 | DP1 | 2000 | 5627 | 0.351584 | 0.966063 | 0.251131 | 0.029846 | 1.482510 |
| budget-2000_seed-42 | DP1+TS | 2000 | 5627 | 0.351584 | 0.966063 | 0.251131 | 0.015461 | 1.480750 |
| budget-2000_seed-42 | DP2 | 2000 | 5627 | 0.347964 | 0.968778 | 0.250679 | 0.021973 | 1.484217 |
| budget-2000_seed-42 | DP2+TS | 2000 | 5627 | 0.347964 | 0.968778 | 0.250679 | 0.013450 | 1.483474 |
| budget-2000_seed-42 | OLL | 2000 | 5627 | 0.347511 | 0.932579 | 0.224887 | 0.115552 | 10.054627 |
| budget-2000_seed-42 | OLL+TS | 2000 | 5627 | 0.347511 | 0.932579 | 0.224887 | 0.053020 | 1.686039 |
| budget-2000_seed-42 | LS | 2000 | 5627 | 0.344344 | 0.992308 | 0.262443 | 0.020790 | 1.491535 |
| budget-2000_seed-42 | LS+TS | 2000 | 5627 | 0.344344 | 0.992308 | 0.262443 | 0.019984 | 1.487602 |
| budget-2000_seed-123 | CE | 2000 | 5573 | 0.356109 | 1.000905 | 0.269683 | 0.021810 | 1.479224 |
| budget-2000_seed-123 | CE+TS | 2000 | 5573 | 0.356109 | 1.000905 | 0.269683 | 0.019675 | 1.476514 |
| budget-2000_seed-123 | CE-median | 2000 | 5573 | 0.277376 | 0.952489 | 0.214027 | 0.021810 | 1.479224 |
| budget-2000_seed-123 | NB | 2000 | 5573 | 0.342534 | 0.996833 | 0.257014 | 0.026346 | 1.521039 |
| budget-2000_seed-123 | Cumulative | 2000 | 5573 | 0.360181 | 0.950679 | 0.237104 | 0.022841 | 1.476448 |
| budget-2000_seed-123 | DP1 | 2000 | 5573 | 0.357466 | 0.999548 | 0.269231 | 0.023162 | 1.478367 |
| budget-2000_seed-123 | DP1+TS | 2000 | 5573 | 0.357466 | 0.999548 | 0.269231 | 0.020619 | 1.476075 |
| budget-2000_seed-123 | DP2 | 2000 | 5573 | 0.358371 | 0.996833 | 0.268326 | 0.023199 | 1.478549 |
| budget-2000_seed-123 | DP2+TS | 2000 | 5573 | 0.358371 | 0.996833 | 0.268326 | 0.021087 | 1.476180 |
| budget-2000_seed-123 | OLL | 2000 | 5573 | 0.357014 | 0.930317 | 0.223982 | 0.096393 | 9.861688 |
| budget-2000_seed-123 | OLL+TS | 2000 | 5573 | 0.357014 | 0.930317 | 0.223982 | 0.064164 | 1.680973 |
| budget-2000_seed-123 | LS | 2000 | 5573 | 0.356561 | 1.000452 | 0.269683 | 0.035356 | 1.483511 |
| budget-2000_seed-123 | LS+TS | 2000 | 5573 | 0.356561 | 1.000452 | 0.269683 | 0.019429 | 1.476303 |
| budget-2000_seed-999 | CE | 2000 | 5585 | 0.350679 | 0.976923 | 0.248869 | 0.021830 | 1.481593 |
| budget-2000_seed-999 | CE+TS | 2000 | 5585 | 0.350679 | 0.976923 | 0.248869 | 0.022830 | 1.480342 |
| budget-2000_seed-999 | CE-median | 2000 | 5585 | 0.269231 | 0.961086 | 0.213122 | 0.021830 | 1.481593 |
| budget-2000_seed-999 | NB | 2000 | 5585 | 0.351584 | 0.971493 | 0.242986 | 0.020270 | 1.520053 |
| budget-2000_seed-999 | Cumulative | 2000 | 5585 | 0.358371 | 0.932579 | 0.224887 | 0.017483 | 1.477708 |
| budget-2000_seed-999 | DP1 | 2000 | 5585 | 0.352489 | 0.962443 | 0.240271 | 0.027023 | 1.475179 |
| budget-2000_seed-999 | DP1+TS | 2000 | 5585 | 0.352489 | 0.962443 | 0.240271 | 0.019175 | 1.474146 |
| budget-2000_seed-999 | DP2 | 2000 | 5585 | 0.351584 | 0.964706 | 0.241176 | 0.019085 | 1.477505 |
| budget-2000_seed-999 | DP2+TS | 2000 | 5585 | 0.351584 | 0.964706 | 0.241176 | 0.018837 | 1.477586 |
| budget-2000_seed-999 | OLL | 2000 | 5585 | 0.346154 | 0.928959 | 0.215385 | 0.114481 | 9.891718 |
| budget-2000_seed-999 | OLL+TS | 2000 | 5585 | 0.346154 | 0.928959 | 0.215385 | 0.052823 | 1.681484 |
| budget-2000_seed-999 | LS | 2000 | 5585 | 0.351131 | 0.975566 | 0.248416 | 0.030771 | 1.485198 |
| budget-2000_seed-999 | LS+TS | 2000 | 5585 | 0.351131 | 0.975566 | 0.248416 | 0.022590 | 1.480254 |
| budget-full_seed-42 | CE | 8531 | 12000 | 0.407692 | 0.829864 | 0.188235 | 0.033740 | 1.375394 |
| budget-full_seed-42 | CE+TS | 8531 | 12000 | 0.407692 | 0.829864 | 0.188235 | 0.026114 | 1.367751 |
| budget-full_seed-42 | CE-median | 8531 | 12000 | 0.353846 | 0.802262 | 0.142081 | 0.033740 | 1.375394 |
| budget-full_seed-42 | NB | 8531 | 12000 | 0.393665 | 0.837104 | 0.180090 | 0.027059 | 1.433895 |
| budget-full_seed-42 | Cumulative | 8531 | 12000 | 0.411765 | 0.783258 | 0.157014 | 0.025292 | 1.370242 |
| budget-full_seed-42 | DP1 | 8531 | 12000 | 0.409502 | 0.819005 | 0.182805 | 0.030192 | 1.362944 |
| budget-full_seed-42 | DP1+TS | 8531 | 12000 | 0.409502 | 0.819005 | 0.182805 | 0.027520 | 1.360880 |
| budget-full_seed-42 | DP2 | 8531 | 12000 | 0.410407 | 0.803167 | 0.171946 | 0.021393 | 1.358837 |
| budget-full_seed-42 | DP2+TS | 8531 | 12000 | 0.410407 | 0.803167 | 0.171946 | 0.025372 | 1.357832 |
| budget-full_seed-42 | OLL | 8531 | 12000 | 0.383710 | 0.790498 | 0.143439 | 0.120734 | 11.712994 |
| budget-full_seed-42 | OLL+TS | 8531 | 12000 | 0.383710 | 0.790498 | 0.143439 | 0.084177 | 1.697566 |
| budget-full_seed-42 | LS | 8531 | 12000 | 0.409502 | 0.827149 | 0.187783 | 0.044402 | 1.384840 |
| budget-full_seed-42 | LS+TS | 8531 | 12000 | 0.409502 | 0.827149 | 0.187783 | 0.027124 | 1.368154 |
| budget-full_seed-123 | CE | 8531 | 12000 | 0.407692 | 0.829864 | 0.188235 | 0.033740 | 1.375394 |
| budget-full_seed-123 | CE+TS | 8531 | 12000 | 0.407692 | 0.829864 | 0.188235 | 0.019428 | 1.367945 |
| budget-full_seed-123 | CE-median | 8531 | 12000 | 0.353846 | 0.802262 | 0.142081 | 0.033740 | 1.375394 |
| budget-full_seed-123 | NB | 8531 | 12000 | 0.402715 | 0.827602 | 0.182805 | 0.080723 | 1.403833 |
| budget-full_seed-123 | Cumulative | 8531 | 12000 | 0.411765 | 0.783258 | 0.157014 | 0.025292 | 1.370242 |
| budget-full_seed-123 | DP1 | 8531 | 12000 | 0.409502 | 0.819005 | 0.182805 | 0.030192 | 1.362944 |
| budget-full_seed-123 | DP1+TS | 8531 | 12000 | 0.409502 | 0.819005 | 0.182805 | 0.024011 | 1.361016 |
| budget-full_seed-123 | DP2 | 8531 | 12000 | 0.410407 | 0.803167 | 0.171946 | 0.021393 | 1.358837 |
| budget-full_seed-123 | DP2+TS | 8531 | 12000 | 0.410407 | 0.803167 | 0.171946 | 0.023088 | 1.357857 |
| budget-full_seed-123 | OLL | 8531 | 12000 | 0.383710 | 0.790498 | 0.143439 | 0.120734 | 11.712994 |
| budget-full_seed-123 | OLL+TS | 8531 | 12000 | 0.383710 | 0.790498 | 0.143439 | 0.084177 | 1.697566 |
| budget-full_seed-123 | LS | 8531 | 12000 | 0.409502 | 0.827149 | 0.187783 | 0.044402 | 1.384840 |
| budget-full_seed-123 | LS+TS | 8531 | 12000 | 0.409502 | 0.827149 | 0.187783 | 0.021427 | 1.368391 |
| budget-full_seed-999 | CE | 8531 | 12000 | 0.407692 | 0.829864 | 0.188235 | 0.033740 | 1.375394 |
| budget-full_seed-999 | CE+TS | 8531 | 12000 | 0.407692 | 0.829864 | 0.188235 | 0.025700 | 1.370036 |
| budget-full_seed-999 | CE-median | 8531 | 12000 | 0.353846 | 0.802262 | 0.142081 | 0.033740 | 1.375394 |
| budget-full_seed-999 | NB | 8531 | 12000 | 0.402715 | 0.827602 | 0.182805 | 0.080723 | 1.403833 |
| budget-full_seed-999 | Cumulative | 8531 | 12000 | 0.411765 | 0.783258 | 0.157014 | 0.025292 | 1.370242 |
| budget-full_seed-999 | DP1 | 8531 | 12000 | 0.409502 | 0.819005 | 0.182805 | 0.030192 | 1.362944 |
| budget-full_seed-999 | DP1+TS | 8531 | 12000 | 0.409502 | 0.819005 | 0.182805 | 0.029782 | 1.362897 |
| budget-full_seed-999 | DP2 | 8531 | 12000 | 0.410407 | 0.803167 | 0.171946 | 0.021393 | 1.358837 |
| budget-full_seed-999 | DP2+TS | 8531 | 12000 | 0.410407 | 0.803167 | 0.171946 | 0.018405 | 1.359364 |
| budget-full_seed-999 | OLL | 8531 | 12000 | 0.383710 | 0.790498 | 0.143439 | 0.120734 | 11.712994 |
| budget-full_seed-999 | OLL+TS | 8531 | 12000 | 0.383710 | 0.790498 | 0.143439 | 0.084177 | 1.697566 |
| budget-full_seed-999 | LS | 8531 | 12000 | 0.409502 | 0.827149 | 0.187783 | 0.044402 | 1.384840 |
| budget-full_seed-999 | LS+TS | 8531 | 12000 | 0.409502 | 0.827149 | 0.187783 | 0.028751 | 1.370539 |

Interpretation: DP2 improves full-budget ordinal scores over CE, but cumulative logistic regression has lower MAE. Low-budget calibration does not consistently improve. OLL probability failure requires investigation rather than optimistic interpretation.

## Secondary reviewer controls

Objective: address selection-criterion and loss-scale regularization confounds. Plan and grid were written in reviewer_control_plan.md and reviewer_controls.json before execution.
Controls: CE-MAE, DP2-NLL, fixed lambda=0.5, and independently tuned OLL with/without scaling. Same saved split IDs and vectorizer as each primary run.

| Run identifier | Method | Accuracy | MAE | Severe rate | ECE15 | NLL |
|---|---|---:|---:|---:|---:|---:|
| budget-2000_seed-42 | CE-MAE | 0.346606 | 0.985068 | 0.256561 | 0.092353 | 1.510875 |
| budget-2000_seed-42 | DP2-NLL | 0.347964 | 0.968778 | 0.250679 | 0.021973 | 1.484217 |
| budget-2000_seed-42 | DP2-fixed | 0.345701 | 0.982805 | 0.258824 | 0.023608 | 1.486042 |
| budget-2000_seed-42 | OLL-reg | 0.176018 | 1.130769 | 0.306787 | 0.224341 | 6.889826 |
| budget-2000_seed-42 | OLL-reg+TS | 0.176018 | 1.130769 | 0.306787 | 0.091227 | 1.618312 |
| budget-2000_seed-123 | CE-MAE | 0.356109 | 1.000905 | 0.269683 | 0.021810 | 1.479224 |
| budget-2000_seed-123 | DP2-NLL | 0.355204 | 0.976923 | 0.253846 | 0.016189 | 1.474028 |
| budget-2000_seed-123 | DP2-fixed | 0.361086 | 0.985068 | 0.261991 | 0.020270 | 1.476436 |
| budget-2000_seed-123 | OLL-reg | 0.176018 | 1.130769 | 0.306787 | 0.224485 | 6.200475 |
| budget-2000_seed-123 | OLL-reg+TS | 0.176018 | 1.130769 | 0.306787 | 0.084808 | 1.608173 |
| budget-2000_seed-999 | CE-MAE | 0.350679 | 0.976923 | 0.248869 | 0.021830 | 1.481593 |
| budget-2000_seed-999 | DP2-NLL | 0.352036 | 0.959276 | 0.238462 | 0.021892 | 1.477061 |
| budget-2000_seed-999 | DP2-fixed | 0.349774 | 0.971946 | 0.245249 | 0.017241 | 1.478933 |
| budget-2000_seed-999 | OLL-reg | 0.202262 | 1.063801 | 0.263348 | 0.178995 | 6.446783 |
| budget-2000_seed-999 | OLL-reg+TS | 0.202262 | 1.063801 | 0.263348 | 0.060345 | 1.611208 |
| budget-full_seed-42 | CE-MAE | 0.407240 | 0.830317 | 0.188235 | 0.034858 | 1.375394 |
| budget-full_seed-42 | DP2-NLL | 0.410407 | 0.803167 | 0.171946 | 0.021392 | 1.358830 |
| budget-full_seed-42 | DP2-fixed | 0.409955 | 0.821719 | 0.184615 | 0.031844 | 1.369095 |
| budget-full_seed-42 | OLL-reg | 0.300000 | 0.886878 | 0.172398 | 0.087678 | 5.649092 |
| budget-full_seed-42 | OLL-reg+TS | 0.300000 | 0.886878 | 0.172398 | 0.044513 | 1.599533 |
| budget-full_seed-123 | CE-MAE | 0.407240 | 0.830317 | 0.188235 | 0.034858 | 1.375394 |
| budget-full_seed-123 | DP2-NLL | 0.410407 | 0.803167 | 0.171946 | 0.021392 | 1.358830 |
| budget-full_seed-123 | DP2-fixed | 0.409955 | 0.821719 | 0.184615 | 0.031844 | 1.369095 |
| budget-full_seed-123 | OLL-reg | 0.300000 | 0.886878 | 0.172398 | 0.087678 | 5.649092 |
| budget-full_seed-123 | OLL-reg+TS | 0.300000 | 0.886878 | 0.172398 | 0.044513 | 1.599533 |
| budget-full_seed-999 | CE-MAE | 0.406787 | 0.827149 | 0.180090 | 0.088196 | 1.392279 |
| budget-full_seed-999 | DP2-NLL | 0.410407 | 0.803167 | 0.171946 | 0.021392 | 1.358830 |
| budget-full_seed-999 | DP2-fixed | 0.409955 | 0.821719 | 0.184615 | 0.031844 | 1.369095 |
| budget-full_seed-999 | OLL-reg | 0.300000 | 0.886878 | 0.172398 | 0.087678 | 5.649092 |
| budget-full_seed-999 | OLL-reg+TS | 0.300000 | 0.886878 | 0.172398 | 0.044513 | 1.599533 |

Completed secondary candidate fits: 72; total recorded candidate fits: 174.
Interpretation: alternate selection criteria do not remove the full-budget DP2 finding; fixed smaller weight has a weaker benefit. OLL regularization reduces its raw NLL but does not yield competitive probabilities.

## OLL risk diagnostic

Objective: distinguish objective behavior from a software bug. Using actual training priors, compute the constant-feature OLL-alpha=1 probability-simplex optimum by a scalar root solve. Output: oll_risk_diagnostic.json. It assigns no probability to extreme labels; this is a mathematical diagnostic and not a second dataset experiment.

## Analysis and artifacts

Executed analyze.py to produce means/sample SD, conditional 5,000-resample paired intervals, four figures with PDF/PNG variants, LaTeX result macros/tables, and class diagnostics. Test outcomes did not expand the primary hyperparameter grid. Source and final-output verification are recorded separately in final_verification.json and final_verification.md.

Issues: interrupted dependency download was retried successfully; one original full-budget C=16 fit hit its iteration cap; the documented continuation repaired numerical convergence. The built-in editor compiler did not return diagnostics for the multi-file paper, so a portable Tectonic compiler was used to export it. No empirical result is inferred from a compiler/editor state.
