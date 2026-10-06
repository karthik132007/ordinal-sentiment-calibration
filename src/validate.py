"""Numerical and methodological tests for the scientific implementation."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize._numdiff import approx_derivative
from src.models.linear import objective, CumulativeLogistic
from src.evaluation.metrics import metrics, scale
from src.data.prepare import prepare, canonical

def main():
    rng = np.random.default_rng(2026)
    X = csr_matrix(rng.normal(size=(7, 4)))
    y = np.array([0, 1, 2, 3, 4, 0, 4])
    theta = rng.normal(scale=.15, size=25)
    checks = []
    for params in [dict(kind="ce"), dict(kind="distance", lam=.5, power=1),
                   dict(kind="distance", lam=2, power=2), dict(kind="oll", power=1),
                   dict(kind="ce", smoothing=.1)]:
        f = lambda t: objective(t, X, y, C=4, **params)[0]
        numeric = approx_derivative(f, theta, method="3-point").ravel()
        analytic = objective(theta, X, y, C=4, **params)[1]
        err = float(np.max(np.abs(numeric - analytic)))
        assert err < 1e-7, (params, err)
        checks.append({"gradient": params, "max_abs_error": err})
    p = np.array([[.6, .1, .1, .1, .1], [.6, .1, .1, .1, .1]])
    m = metrics(p, np.array([0, 4]))
    assert np.isclose(m["ece15"], .1) and m["mae"] == 2 and m["severe"] == .5
    assert np.array_equal(scale(p, 2).argmax(axis=1), p.argmax(axis=1))
    assert np.allclose(scale(p, 1), p)
    model = CumulativeLogistic(1).fit(X, y)
    assert (model.predict_proba(X) >= 0).all()
    rows, audit = prepare()
    sets = {s: {canonical(r["text"]) for r in rows if r["split"] == s}
            for s in ["train", "dev", "test"]}
    assert not (sets["train"] & sets["test"] or sets["train"] & sets["dev"] or sets["dev"] & sets["test"])
    import json
    from pathlib import Path
    Path("experiments/results/validation_checks.json").write_text(json.dumps(checks + [{"metrics": "passed", "split_audit": "passed"}], indent=2))
    print("Gradient, metric, scaling, cumulative probability, and split checks passed.")

if __name__ == "__main__":
    main()
