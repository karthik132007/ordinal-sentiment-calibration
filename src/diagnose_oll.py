"""Constant-feature population-risk diagnostic using actual SST training priors."""
import json
import numpy as np
from scipy.optimize import brentq
from src.data.prepare import ROOT

def main():
    audit = json.loads((ROOT / "experiments/results/data_audit.json").read_text())
    q = np.array(audit["class_counts"]["train"], dtype=float)
    q /= q.sum()
    a = q @ np.abs(np.arange(5)[:, None] - np.arange(5)[None, :])
    nu = brentq(lambda t: np.maximum(0, 1 - a / t).sum() - 1, a.min(), a.max() * 5)
    p = np.maximum(0, 1 - a / nu)
    assert np.isclose(p.sum(), 1)
    result = {"description": "Analytic OLL-alpha=1 simplex optimum for a constant feature vector with empirical training priors. This is a risk diagnostic, not a second dataset or classifier test result.",
              "train_priors": q.tolist(), "expected_absolute_cost": a.tolist(),
              "lagrange_multiplier": float(nu), "optimal_OLL_probabilities": p.tolist(),
              "CE_optimum": q.tolist()}
    (ROOT / "experiments/results/oll_risk_diagnostic.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
