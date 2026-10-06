# Final verification

Completed 6 October 2026.

- All required research notes, sources, results, figures, README, and LaTeX/PDF artifacts exist.
- Final PDF: 9 pages, including references; successfully compiled with Tectonic 0.17.0 and BibTeX.
- All nine rendered pages were inspected. Final reproduction's page images exactly match those inspected. Figures and tables are legible and within the margins.
- No overfull boxes or unresolved citations/references remain. Minor underfull line-spacing warnings remain and do not cause clipping or overlap.
- 108 metric rows were recalculated; 66 saved models reproduced their probabilities exactly (maximum absolute difference 0.0).
- Six metrics were independently recalculated, all generated numerical macros were checked, and every used citation key resolves.
- The one-command reproduction completed successfully and exported the final PDF. Its transcript is in experiments/results/reproduction_stdout.log.
- The original research-plan hash still matches the pretraining hash. Numerical continuation and post-primary controls are explicitly logged.
- No TODO/TBD/FIXME placeholders remain in the manuscript. No experiments, datasets, citations, or quantitative findings were invented.
- Revised claims retain the single-corpus, sparse-model, correlated-seed, bin-sensitivity, and limited-tuning boundaries. The reviewer report's fixable weaknesses were addressed through executed controls, diagnostics, and rewriting.

PDF SHA-256: `b39493dcd2158be37bf95e5553a36caf3827544228c6a8dedeab343b4c3bb60d`.

Machine-readable audit: experiments/results/final_verification.json. File integrity manifest: experiments/results/artifact_manifest.json.
