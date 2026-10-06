"""Replay saved models and independently audit final numerical results."""
import hashlib
import json
import pickle
import re
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.prepare import ROOT, canonical
from src.evaluation.metrics import metrics, scale

def independent_scores(p, y, median=False):
    pred = p.argmax(axis=1) if not median else np.array([next(k for k in range(5) if sum(row[:k+1]) >= .5) for row in p])
    scores = {"accuracy": float(np.mean(pred == y)), "mae": float(np.mean(np.abs(pred-y))),
              "severe": float(np.mean(np.abs(pred-y) >= 2))}
    fs = []
    for k in range(5):
        tp = ((pred == k) & (y == k)).sum()
        fp = ((pred == k) & (y != k)).sum()
        fn = ((pred != k) & (y == k)).sum()
        fs.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    scores["macro_f1"] = float(np.mean(fs))
    eps = np.finfo(p.dtype).eps
    scores["nll"] = float(np.mean([-np.log(np.clip(row[label], eps, 1-eps)) for row, label in zip(p, y)]))
    scores["brier"] = float(np.mean([sum((row[k] - int(k == label)) ** 2 for k in range(5)) for row, label in zip(p, y)]))
    return scores

def main():
    base = ROOT / "experiments/results"
    rows = {r["id"]: r for r in json.loads((ROOT / "data/processed/sst5.json").read_text())}
    checked_rows = 0
    replay_models = 0
    max_difference = 0.
    hashes = {}
    for folder in sorted(base.glob("budget-*")):
        if not folder.is_dir(): continue
        splits = json.loads((folder / "splits.json").read_text())
        assert not (set(splits["train"]) & set(splits["test"]))
        assert not (set(splits["calibration"]) & set(splits["selection"]))
        y = np.array([rows[i]["label"] for i in splits["test"]])
        pred = np.load(folder / "predictions.npz")
        assert np.array_equal(pred["ids"], splits["test"]) and np.array_equal(pred["y"], y)
        choices = json.loads((folder / "selection.json").read_text())
        with (folder / "models.pkl").open("rb") as f:
            fitted = pickle.load(f)
        Xtest = fitted["vectorizer"].transform([rows[i]["text"] for i in splits["test"]])
        for name, model in fitted["models"].items():
            p = model.predict_proba(Xtest)
            diff = float(np.max(np.abs(p - pred[name])))
            assert diff < 1e-12, (folder, name, diff)
            max_difference = max(diff, max_difference)
            replay_models += 1
            if name in choices["temperatures"]:
                ts = pred[name + "+TS"]
                assert np.allclose(ts, scale(p, choices["temperatures"][name]), atol=1e-12)
                assert np.array_equal(ts.argmax(axis=1), p.argmax(axis=1))
        for file, values_file, controls in [("predictions.npz", "metrics.json", False),
                                            ("controls_predictions.npz", "controls_metrics.json", True)]:
            pp = np.load(folder / file)
            if controls:
                with (folder / "controls_models.pkl").open("rb") as f:
                    models = pickle.load(f)
                for name, model in models.items():
                    diff = float(np.max(np.abs(model.predict_proba(Xtest) - pp[name])))
                    assert diff < 1e-12
                    max_difference = max(diff, max_difference)
                    replay_models += 1
            for result in json.loads((folder / values_file).read_text()):
                name = result["method"]
                p = pp["CE"] if name == "CE-median" else pp[name]
                recomputed = metrics(p, y, decision="median" if name == "CE-median" else "argmax")
                independent = independent_scores(p, y, median=name == "CE-median")
                for key, value in recomputed.items():
                    assert np.isclose(value, result[key], atol=1e-12), (folder, name, key)
                for key, value in independent.items():
                    assert np.isclose(value, result[key], atol=1e-12), (folder, name, key, value, result[key])
                checked_rows += 1
            hashes[str((folder / file).relative_to(ROOT))] = hashlib.sha256((folder / file).read_bytes()).hexdigest()
    # Audit aggregate CSV values against the individual run records.
    for csv, jsonname in [("metrics.csv", "metrics.json"), ("controls_metrics.csv", "controls_metrics.json")]:
        df = pd.read_csv(base / csv, dtype={"budget": str}).sort_values(["budget", "seed", "method"]).reset_index(drop=True)
        expected = pd.DataFrame([r for folder in sorted(base.glob("budget-*"))
                                 if folder.is_dir() for r in json.loads((folder / jsonname).read_text())])
        expected["budget"] = expected["budget"].astype(str)
        expected = expected.sort_values(["budget", "seed", "method"]).reset_index(drop=True)
        for col in df.select_dtypes("number").columns:
            assert np.allclose(df[col], expected[col], atol=1e-12)
    # Every referenced result macro must resolve in the generated numerical file.
    tex = (ROOT / "paper/paper.tex").read_text()
    macros = (ROOT / "paper/numbers.tex").read_text()
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", macros))
    used = set(re.findall(r"\\((?:Full|Low)[A-Za-z]+|TrainCount|DevCount|TestCount|RemovedCount)\b", tex))
    assert used <= defined, used - defined
    values = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]+)\}", macros))
    facts = json.loads((base / "facts.json").read_text())
    method_names = {"CE": "CE", "CE+TS": "CETS", "CE-median": "Median", "NB": "NB",
        "Cumulative": "Cum", "DP1": "DPone", "DP2": "DPtwo", "DP2+TS": "DPtwoTS",
        "OLL": "OLL", "OLL+TS": "OLLTS", "LS": "LS", "LS+TS": "LSTS",
        "CE-MAE": "CEMAE", "DP2-NLL": "DPNLL", "DP2-fixed": "DPfixed",
        "OLL-reg": "OLLreg", "OLL-reg+TS": "OLLregTS"}
    metric_names = {"accuracy": "Acc", "macro_f1": "Fone", "mae": "MAE", "severe": "Severe",
        "ece15": "ECE", "nll": "NLL", "brier": "Brier", "rps": "RPS", "central_fraction": "Central"}
    for budget in ["2000", "full"]:
        for method, suffix in method_names.items():
            for col, ending in metric_names.items():
                name = ("Low" if budget == "2000" else "Full") + suffix + ending
                number = facts[f"{budget}/{method}"][col]["mean"]
                expected = f"{number * 100:.2f}" if col in ["accuracy", "macro_f1", "severe", "ece15", "central_fraction"] else f"{number:.3f}"
                assert values[name] == expected, (name, values[name], expected)
    audit = json.loads((base / "data_audit.json").read_text())
    for name, number in [("TrainCount", audit["clean_counts"]["train"]), ("DevCount", audit["clean_counts"]["dev"]),
                         ("TestCount", audit["clean_counts"]["test"]), ("RemovedCount", len(audit["excluded"]))]:
        assert values[name] == str(number)
    cites = {key for group in re.findall(r"\\cite\{([^}]+)\}", tex) for key in group.split(",")}
    bibkeys = set(re.findall(r"@\w+\{([^,]+),", (ROOT / "paper/references.bib").read_text()))
    assert cites <= bibkeys
    assert not re.search(r"\bTODO\b|\bTBD\b|\bFIXME\b", tex)
    report = {"status": "passed", "metric_rows_recomputed": checked_rows,
              "saved_models_replayed": replay_models, "maximum_probability_replay_difference": max_difference,
              "independent_metrics": ["accuracy", "macro_f1", "mae", "severe", "nll", "brier"],
              "numeric_macros_used": len(used), "citations_resolve": sorted(cites), "prediction_hashes": hashes}
    (base / "final_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "prediction_hashes"}, indent=2))

if __name__ == "__main__":
    main()
