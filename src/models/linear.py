"""Sparse linear softmax models with analytic loss gradients."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, logsumexp
from sklearn.linear_model import LogisticRegression

def objective(theta, X, y, C, kind="ce", lam=0., power=2, smoothing=0.):
    d = X.shape[1]
    W = theta[:d * 5].reshape(d, 5)
    b = theta[d * 5:]
    z = X @ W + b
    p = softmax(z, axis=1)
    n = len(y)
    target = np.eye(5)[y] * (1 - smoothing) + smoothing / 5
    if kind in ("ce", "distance"):
        loss = np.mean(logsumexp(z, axis=1) - np.sum(target * z, axis=1))
        grad_z = p - target
        if kind == "distance":
            cost = (np.abs(np.arange(5)[None, :] - y[:, None]) / 4.) ** power
            expected = np.sum(p * cost, axis=1, keepdims=True)
            loss += lam * expected.mean()
            grad_z += lam * p * (cost - expected)
    elif kind == "oll":
        cost = np.abs(np.arange(5)[None, :] - y[:, None]).astype(float) ** power
        safe = np.maximum(1 - p, 1e-14)
        loss = np.mean(-np.sum(cost * np.log(safe), axis=1))
        grad_p = cost / safe
        grad_z = p * (grad_p - np.sum(grad_p * p, axis=1, keepdims=True))
    else:
        raise ValueError(kind)
    penalty = 1 / (n * C)
    loss += .5 * penalty * np.sum(W * W)
    grad_W = X.T @ (grad_z / n) + penalty * W
    grad_b = grad_z.mean(axis=0)
    return float(loss), np.concatenate([np.asarray(grad_W).ravel(), grad_b])

class LinearSoftmax:
    def __init__(self, C=4., kind="ce", lam=0., power=2, smoothing=0., maxiter=400):
        self.params = dict(C=C, kind=kind, lam=lam, power=power, smoothing=smoothing)
        self.maxiter = maxiter

    def fit(self, X, y):
        start = time.perf_counter()
        # Deterministic zero initialization isolates selection and sampling variability.
        result = minimize(
            lambda theta: objective(theta, X, y, **self.params),
            np.zeros((X.shape[1] + 1) * 5), jac=True, method="L-BFGS-B",
            options={"maxiter": self.maxiter, "ftol": 1e-10, "gtol": 1e-6})
        initial_iterations = int(result.nit)
        numerical_retry = False
        if not result.success and result.nit >= self.maxiter:
            # Numerical repair only: identical objective/data/tolerances, more iterations.
            numerical_retry = True
            result = minimize(lambda theta: objective(theta, X, y, **self.params),
                              result.x, jac=True, method="L-BFGS-B",
                              options={"maxiter": 1600, "ftol": 1e-10, "gtol": 1e-6})
        self.W = result.x[:-5].reshape(X.shape[1], 5)
        self.b = result.x[-5:]
        self.info = {"success": bool(result.success), "message": str(result.message),
                     "iterations": int(result.nit) + (initial_iterations if numerical_retry else 0),
                     "numerical_retry": numerical_retry, "objective": float(result.fun),
                     "max_abs_gradient": float(np.abs(result.jac).max()),
                     "fit_seconds": time.perf_counter() - start}
        return self

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b, axis=1)

class CumulativeLogistic:
    """Four independent binary heads, with monotonic probability repair."""
    def __init__(self, C):
        self.C = C

    def fit(self, X, y):
        start = time.perf_counter()
        self.models = [LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000,
                                         tol=1e-7).fit(X, y > k) for k in range(4)]
        self.info = {"fit_seconds": time.perf_counter() - start,
                     "iterations": [int(m.n_iter_[0]) for m in self.models]}
        return self

    def predict_proba(self, X):
        q = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models])
        q = np.minimum.accumulate(q, axis=1)
        return np.diff(np.column_stack([np.ones(len(q)), q, np.zeros(len(q))]), axis=1) * -1
