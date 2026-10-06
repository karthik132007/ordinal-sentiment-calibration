# Ordinal sentiment error and probability calibration

Completed CPU research study selected from candidate 10 in `Sentiment Analysis Research Directions.pdf`. Research question: **does an expected ordinal distance penalty reduce severe sentiment errors and improve probability calibration in sparse five-level sentiment classifiers?**

The answer is qualified: full-budget quadratic penalization improves MAE from 0.830 to 0.803 and severe errors from 18.82% to 17.19%, but cumulative logistic regression achieves lower MAE. Calibration does not improve consistently at the 2,000-example budget. This is one SST-5 corpus, not a transformer or domain-transfer experiment. All findings come from executed code and retained outputs.

## Outputs

- `paper/paper.pdf`, `paper/paper.tex`, and `paper/references.bib`: compiled manuscript and editable sources.
- `research/`: candidate evaluation, plan, source extraction, references, amendments, experiment log, analysis, and internal reviewer report.
- `src/`: official-data preparation, analytic loss implementations, experiment runners, numerical checks, analysis, and final replay audit.
- `experiments/configs/`: primary protocol and secondary control settings.
- `experiments/results/`: individual candidate records, selected settings, exact split IDs, vectorizers/models, compressed test probabilities, summaries, bootstrap intervals, diagnostics, and logs.
- `figures/`: four generated figures in PDF and PNG formats.
- `data/raw/stanfordSentimentTreebank.zip` and `data/processed/sst5.json`: official archive and audited root sentences.
- `requirements.lock.txt`: exact installed dependency versions. `tools/tectonic/README.md`: setup instructions for the portable compiler used for the exported PDF; its executable is excluded from Git.

## Environment setup

The executed environment uses Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, pandas 3.0.6, and Matplotlib 3.11.2. The full transitive freeze is in `requirements.lock.txt`; hardware and dates are in `experiments/results/environment.json`. Models use CPU and one numerical-library thread. Use this environment for exact replay; numerical-library changes can produce small differences.

```bash
uv venv --python 3.14 .venv
uv pip sync --python .venv/bin/python requirements.lock.txt
```

The workspace already contains this environment. If the host restricts the default uv cache, set `UV_CACHE_DIR` to a writable directory, such as `./tmp/uv-cache`.

## Dataset preparation

The official archive is already saved. If restoring on another machine:

```bash
mkdir -p data/raw
curl -fL https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip \
  -o data/raw/stanfordSentimentTreebank.zip
uv run --no-sync python -m src.data.prepare
```

Expected archive SHA-256: `3f5209483b46bbf129cacbbbe6ae02fe780407034f61cf6342b7833257c3f1db`. Root sentences only; no phrase-descendant expansion. The processed train/dev/test counts are 8,531/1,100/2,210. Exact normalized-text deduplication removes 14 rows; IDs and class counts are in `data_audit.json`. Consult the originating dataset source for use terms.

## Reproduce everything

From the repository root after setup:

```bash
uv run --no-sync python -m src.reproduce
```

This runs numerical validation, primary experiments, secondary controls, the OLL diagnostic, figure/table/macro generation, experiment logging, saved-model/metric replay, and LaTeX compilation. Completed runs resume from their saved artifacts. To retrain without deleting the existing evidence, first copy the repository into a new directory, remove only that copy's `experiments/results/budget-*` directories, and run the same command. The original fits included 102 primary and 72 secondary candidates; these counts are recorded in the log. The study fits small sparse heads rather than pretrained encoders.

Individual stages:

```bash
uv run --no-sync python -m src.validate
uv run --no-sync python -m src.run_experiments
uv run --no-sync python -m src.run_controls
uv run --no-sync python -m src.diagnose_oll
uv run --no-sync python -m src.analyze
uv run --no-sync python -m src.verify_results
```

Plots, LaTeX tables, and numerical macros regenerate directly from `metrics.csv`, `controls_metrics.csv`, and per-run probabilities. Edit prose in `paper/paper.tex`; do not hand-edit generated `numbers.tex` or table files. The audit independently recalculates accuracy, macro-F1, MAE, severity, NLL, and Brier, replays model predictions, verifies scaling decisions, checks CSV agreement, and confirms macro/citation resolution.

## Compile the paper

Tectonic 0.17.0 was downloaded from the official `tectonic-typesetting/tectonic` GitHub release. Its executable is present in the original workspace but excluded from Git. Follow `tools/tectonic/README.md` to install it before running the full reproduction command. No system-wide TeX installation is required. LaTeX resources are cached under `tmp/tex-cache`; a clean copy may need internet access on its first compile. If using another OS, place a matching official release executable at the same location.

```bash
XDG_CACHE_HOME="$PWD/tmp/tex-cache" tools/tectonic/tectonic \
  --keep-logs --keep-intermediates --outdir paper paper/paper.tex
```

The built-in source editor can open the manuscript, but this paper uses external figures, generated tables, and BibTeX. Export/verification uses the portable compiler; a queued editor preview is not compilation evidence. `paper/build_stdout.log` and `paper/paper.log` record the build. PDF layout is visually reviewed after rendering with `pdftoppm`; the final verification report records the outcome.

## Interpretation and evidence boundaries

The original plan predates training; `protocol_sha256.txt` preserves its hash. `protocol_amendments.md` documents a numerical optimizer continuation. Secondary reviewer controls are explicitly post-primary. Seeds randomize subset/development selection, not initialization; identical full-budget fits can have zero SD. Bootstrap intervals condition on fixed models and resample the common test sentences. No independent-corpus replication, faculty endorsement, external review, submission, or new loss invention is claimed. Stored pickle files are generated locally; only load these trusted artifacts.
