"""Run the locked protocol; save selections before evaluating held-out test data."""
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from threadpoolctl import threadpool_limits

from src.data.prepare import prepare, ROOT
from src.evaluation.metrics import metrics, fit_temperature, scale
from src.models.linear import LinearSoftmax, CumulativeLogistic

OUT = ROOT / "experiments/results"

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2))

def main():
    config = json.loads((ROOT / "experiments/configs/main.json").read_text())
    rows, audit = prepare()
    arrays = {s: [r for r in rows if r["split"] == s] for s in ("train", "dev", "test")}
    texts = {s: np.array([r["text"] for r in rs]) for s, rs in arrays.items()}
    ys = {s: np.array([r["label"] for r in rs]) for s, rs in arrays.items()}
    started = datetime.now().astimezone().isoformat()
    dump(OUT / "environment.json", {"started": started, "python": platform.python_version(),
        "platform": platform.platform(), "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
        "device": "CPU", "blas_threads": 1,
        "packages": {p: importlib.metadata.version(p) for p in
                     ["numpy", "scipy", "scikit-learn", "pandas", "matplotlib"]},
        "config": config, "protocol_sha256": hashlib.sha256((ROOT / "research/research_plan.md").read_bytes()).hexdigest()})
    all_metrics = []
    all_candidates = []
    for budget in config["budgets"]:
        for seed in config["seeds"]:
            run = f"budget-{budget}_seed-{seed}"
            dest = OUT / run
            dest.mkdir(exist_ok=True)
            if (dest / "complete.json").exists():
                print(f"RESUME {run}", flush=True)
                all_metrics.extend(json.loads((dest / "metrics.json").read_text()))
                all_candidates.extend(json.loads((dest / "candidates.json").read_text()))
                continue
            print(f"START {run}", flush=True)
            run_start = time.perf_counter()
            train_ids = np.arange(len(ys["train"]))
            if budget != "full":
                train_ids, _ = train_test_split(train_ids, train_size=budget, random_state=seed,
                                                stratify=ys["train"])
            sel_ids, cal_ids = train_test_split(np.arange(len(ys["dev"])), test_size=.5,
                                                 random_state=seed, stratify=ys["dev"])
            dump(dest / "splits.json", {"train": [arrays["train"][i]["id"] for i in train_ids],
                "selection": [arrays["dev"][i]["id"] for i in sel_ids],
                "calibration": [arrays["dev"][i]["id"] for i in cal_ids],
                "test": [r["id"] for r in arrays["test"]]})
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=config["min_df"],
                                        max_features=config["max_features"], sublinear_tf=True)
            X = vectorizer.fit_transform(texts["train"][train_ids])
            y = ys["train"][train_ids]
            Xsel = vectorizer.transform(texts["dev"][sel_ids])
            Xcal = vectorizer.transform(texts["dev"][cal_ids])
            ysel, ycal = ys["dev"][sel_ids], ys["dev"][cal_ids]
            candidates = []

            def record(name, model, params):
                prob = model.predict_proba(Xsel)
                value = metrics(prob, ysel)
                item = {"budget": str(budget), "seed": seed, "family": name,
                        "params": params, "selection": value, "fit": getattr(model, "info", {})}
                candidates.append(item)
                print(f"  {name} {params}: val MAE={value['mae']:.4f} NLL={value['nll']:.4f}", flush=True)
                if hasattr(model, "info") and model.info.get("success") is False:
                    # Fail rather than silently report underoptimized objectives.
                    dump(dest / "failed_candidate.json", item)
                    raise RuntimeError(f"Optimizer did not converge: {item}")
                return value

            ce_options = []
            for C in config["C_grid"]:
                model = LinearSoftmax(C=C, maxiter=config["maxiter"]).fit(X, y)
                value = record("CE", model, {"C": C})
                ce_options.append((value["nll"], C, model))
            _, C, ce = min(ce_options, key=lambda t: t[0])
            models = {"CE": ce}
            selected = {"CE": {"C": C}}
            nb_options = []
            for alpha in config["nb_alpha_grid"]:
                start = time.perf_counter()
                model = MultinomialNB(alpha=alpha).fit(X, y)
                model.info = {"fit_seconds": time.perf_counter() - start}
                value = record("NB", model, {"alpha": alpha})
                nb_options.append((value["nll"], alpha, model))
            _, alpha, models["NB"] = min(nb_options, key=lambda t: t[0])
            selected["NB"] = {"alpha": alpha}
            models["Cumulative"] = CumulativeLogistic(C).fit(X, y)
            selected["Cumulative"] = {"C": C, "monotonic_repair": "cumulative_minimum"}
            record("Cumulative", models["Cumulative"], selected["Cumulative"])
            for power in [1, 2]:
                options = []
                for lam in config["lambda_grid"]:
                    model = LinearSoftmax(C=C, kind="distance", lam=lam, power=power,
                                          maxiter=config["maxiter"]).fit(X, y)
                    value = record(f"DP{power}", model, {"C": C, "lambda": lam, "power": power})
                    options.append((value["mae"], value["nll"], lam, model))
                _, _, lam, model = min(options, key=lambda t: t[:2])
                models[f"DP{power}"] = model
                selected[f"DP{power}"] = {"C": C, "lambda": lam, "power": power}
            models["OLL"] = LinearSoftmax(C=C, kind="oll", power=1, maxiter=config["maxiter"]).fit(X, y)
            selected["OLL"] = {"C": C, "alpha": 1, "distance": "raw"}
            record("OLL", models["OLL"], selected["OLL"])
            models["LS"] = LinearSoftmax(C=C, smoothing=.1, maxiter=config["maxiter"]).fit(X, y)
            selected["LS"] = {"C": C, "smoothing": .1}
            record("LS", models["LS"], selected["LS"])
            temperatures = {name: fit_temperature(m.predict_proba(Xcal), ycal)
                            for name, m in models.items() if name in ["CE", "DP1", "DP2", "OLL", "LS"]}
            # Write all choices before computing test probabilities or metrics.
            dump(dest / "selection.json", {"models": selected, "temperatures": temperatures,
                 "n_train": len(y), "n_selection": len(ysel), "n_calibration": len(ycal),
                 "n_features": X.shape[1]})
            dump(dest / "candidates.json", candidates)
            with (dest / "models.pkl").open("wb") as f:
                pickle.dump({"vectorizer": vectorizer, "models": models}, f)
            Xtest = vectorizer.transform(texts["test"])
            predictions = {"y": ys["test"], "ids": np.array([r["id"] for r in arrays["test"]])}
            run_metrics = []
            for name, model in models.items():
                p = model.predict_proba(Xtest)
                predictions[name] = p
                value = {"budget": str(budget), "seed": seed, "method": name,
                         **metrics(p, ys["test"]), "fit_seconds": model.info["fit_seconds"]}
                run_metrics.append(value)
                if name in temperatures:
                    tsname = name + "+TS"
                    scaled = scale(p, temperatures[name])
                    assert np.array_equal(p.argmax(axis=1), scaled.argmax(axis=1))
                    predictions[tsname] = scaled
                    run_metrics.append({"budget": str(budget), "seed": seed, "method": tsname,
                                        **metrics(scaled, ys["test"]), "fit_seconds": model.info["fit_seconds"]})
                if name == "CE":
                    run_metrics.append({"budget": str(budget), "seed": seed, "method": "CE-median",
                                        **metrics(p, ys["test"], decision="median"),
                                        "fit_seconds": model.info["fit_seconds"]})
            np.savez_compressed(dest / "predictions.npz", **predictions)
            dump(dest / "metrics.json", run_metrics)
            dump(dest / "complete.json", {"wall_seconds": time.perf_counter() - run_start,
                                         "completed": datetime.now().astimezone().isoformat()})
            all_metrics.extend(run_metrics)
            all_candidates.extend(candidates)
            pd.DataFrame(all_metrics).to_csv(OUT / "metrics.csv", index=False)
            dump(OUT / "candidates.json", all_candidates)
            print(f"DONE {run} in {time.perf_counter() - run_start:.1f}s", flush=True)
    pd.DataFrame(all_metrics).to_csv(OUT / "metrics.csv", index=False)
    dump(OUT / "candidates.json", all_candidates)

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
