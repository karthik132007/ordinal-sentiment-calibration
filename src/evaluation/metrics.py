import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, log_loss

def scale(p, temperature):
    return softmax(np.log(np.maximum(p, 1e-14)) / temperature, axis=1)

def fit_temperature(p, y):
    result = minimize_scalar(lambda t: log_loss(y, scale(p, np.exp(t)), labels=np.arange(5)),
                             bounds=(np.log(.05), np.log(20)), method="bounded")
    return float(np.exp(result.x))

def reliability(p, y, bins=15):
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    ids = np.minimum((confidence * bins).astype(int), bins - 1)
    return [{"bin": i, "count": int((ids == i).sum()),
             "confidence": float(confidence[ids == i].mean()),
             "accuracy": float(correct[ids == i].mean())}
            for i in range(bins) if (ids == i).any()]

def metrics(p, y, decision="argmax"):
    assert np.isfinite(p).all() and (p >= -1e-12).all()
    assert np.allclose(p.sum(axis=1), 1)
    pred = p.argmax(axis=1) if decision == "argmax" else (np.cumsum(p, axis=1) < .5).sum(axis=1)
    error = np.abs(pred - y)
    target = np.eye(5)[y]
    result = {"accuracy": float(accuracy_score(y, pred)),
              "macro_f1": float(f1_score(y, pred, average="macro", labels=np.arange(5), zero_division=0)),
              "mae": float(error.mean()), "severe": float((error >= 2).mean()),
              "nll": float(log_loss(y, p, labels=np.arange(5))),
              "brier": float(np.square(p - target).sum(axis=1).mean()),
              "rps": float(np.square(np.cumsum(p, axis=1)[:, :4] - np.cumsum(target, axis=1)[:, :4]).mean()),
              "central_fraction": float((pred == 2).mean())}
    # Calibration concerns the argmax confidence, including when ordinal decision is reported.
    for bins in [5, 10, 15, 30]:
        result[f"ece{bins}"] = sum(r["count"] / len(y) * abs(r["confidence"] - r["accuracy"])
                                  for r in reliability(p, y, bins))
    return result
