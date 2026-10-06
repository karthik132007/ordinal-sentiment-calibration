"""Write the experiment log from primary and secondary run artifacts."""
import json
from pathlib import Path
import pandas as pd
from src.data.prepare import ROOT

def main():
    base = ROOT / "experiments/results"
    env = json.loads((base / "environment.json").read_text())
    audit = json.loads((base / "data_audit.json").read_text())
    entries = json.loads((base / "candidates.json").read_text())
    lines = ["# Experiment log", "", "Date: 6 October 2026 (Asia/Kolkata).", "",
        "## Data and numerical validation", "",
        "Objective: reproduce official SST root labels and prevent exact-text leakage.",
        f"Archive SHA-256: `{audit['archive_sha256']}`.",
        f"Original counts: {audit['original_counts']}; audited counts: {audit['clean_counts']}.",
        f"Excluded {len(audit['excluded'])} repeated rows; conflicting-label groups: {audit['conflicting_text_groups']}.",
        "Finite-difference gradients, known metric cases, scaling invariance, cumulative probabilities, and split-disjointness checks passed. Full error values are in validation_checks.json.", "",
        "## Environment", "", f"Python: {env['python']}; CPU device; numerical-library threads: 1.",
        f"Packages: {env['packages']}. Hardware: AMD Ryzen 5 5500U, 12 logical CPUs; no detected NVIDIA GPU.", "",
        "Protocol: research_plan.md and protocol_sha256.txt were written before model training. Numerical continuation is documented in protocol_amendments.md; initial failed output is preserved.", "",
        "## Primary experiment", "",
        "Objective: test whether expected ordinal distance improves ordinal error and calibration under a common sparse representation.",
        "Configuration: experiments/configs/main.json; fitted TF-IDF per training sample; selection and calibration halves kept separate. Candidate records preserve every validation setting.",
        f"Completed primary candidate fits: {len(entries)}. Fits using iteration continuation: {sum(e['fit'].get('numerical_retry', False) for e in entries)}.", "",
        "| Run identifier | Method | Train size | Features | Accuracy | MAE | Severe rate | ECE15 | NLL |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    main = pd.read_csv(base / "metrics.csv", dtype={"budget": str})
    for _, r in main.iterrows():
        run = f"budget-{r['budget']}_seed-{int(r['seed'])}"
        sel = json.loads((base / run / "selection.json").read_text())
        lines.append(f"| {run} | {r['method']} | {sel['n_train']} | {sel['n_features']} | {r['accuracy']:.6f} | {r['mae']:.6f} | {r['severe']:.6f} | {r['ece15']:.6f} | {r['nll']:.6f} |")
    lines += ["", "Interpretation: DP2 improves full-budget ordinal scores over CE, but cumulative logistic regression has lower MAE. Low-budget calibration does not consistently improve. OLL probability failure requires investigation rather than optimistic interpretation.", "",
        "## Secondary reviewer controls", "",
        "Objective: address selection-criterion and loss-scale regularization confounds. Plan and grid were written in reviewer_control_plan.md and reviewer_controls.json before execution.",
        "Controls: CE-MAE, DP2-NLL, fixed lambda=0.5, and independently tuned OLL with/without scaling. Same saved split IDs and vectorizer as each primary run.", "",
        "| Run identifier | Method | Accuracy | MAE | Severe rate | ECE15 | NLL |", "|---|---|---:|---:|---:|---:|---:|"]
    controlfits = 0
    for p in base.glob("budget-*/controls_candidates.json"):
        controlfits += len(json.loads(p.read_text()))
    controls = pd.read_csv(base / "controls_metrics.csv", dtype={"budget": str})
    for _, r in controls.iterrows():
        lines.append(f"| budget-{r['budget']}_seed-{int(r['seed'])} | {r['method']} | {r['accuracy']:.6f} | {r['mae']:.6f} | {r['severe']:.6f} | {r['ece15']:.6f} | {r['nll']:.6f} |")
    lines += ["", f"Completed secondary candidate fits: {controlfits}; total recorded candidate fits: {controlfits + len(entries)}.",
        "Interpretation: alternate selection criteria do not remove the full-budget DP2 finding; fixed smaller weight has a weaker benefit. OLL regularization reduces its raw NLL but does not yield competitive probabilities.", "",
        "## OLL risk diagnostic", "",
        "Objective: distinguish objective behavior from a software bug. Using actual training priors, compute the constant-feature OLL-alpha=1 probability-simplex optimum by a scalar root solve. Output: oll_risk_diagnostic.json. It assigns no probability to extreme labels; this is a mathematical diagnostic and not a second dataset experiment.", "",
        "## Analysis and artifacts", "",
        "Executed analyze.py to produce means/sample SD, conditional 5,000-resample paired intervals, four figures with PDF/PNG variants, LaTeX result macros/tables, and class diagnostics. Test outcomes did not expand the primary hyperparameter grid. Source and final-output verification are recorded separately in final_verification.json and final_verification.md.", "",
        "Issues: interrupted dependency download was retried successfully; one original full-budget C=16 fit hit its iteration cap; the documented continuation repaired numerical convergence. The built-in editor compiler did not return diagnostics for the multi-file paper, so a portable Tectonic compiler was used to export it. No empirical result is inferred from a compiler/editor state.", ""]
    (ROOT / "research/experiment_log.md").write_text("\n".join(lines))

if __name__ == "__main__":
    main()
