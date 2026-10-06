# Secondary controls (specified before their execution)

The initial results and code review identify two remediable confounds: CE uses selection NLL while DP2 uses selection MAE, and standalone OLL has a different loss scale but shares CE regularization. Do not change the primary analysis. Add secondary controls on the same saved splits and representation:

1. CE-MAE: choose C in the original grid by selection MAE, ties by NLL.
2. DP2-NLL: choose the original lambda grid by selection NLL.
3. DP2-fixed: lambda=0.5, shared original CE C, removing lambda selection.
4. OLL-reg: tune C in {0.01, 0.1, 1, 4, 16} by selection NLL, and fit a separate temperature on the calibration half.

These controls are post-primary, exploratory additions. No test-driven grid expansion is performed after this specification. Fit the same zero-initialized models with the numerical continuation rule and retain every validation candidate and test probability. They can test whether the primary interpretation survives fairer criteria, but cannot establish broad generalization or a new loss contribution.
