"""Post-primary reviewer controls, specified in reviewer_control_plan.md."""
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data.prepare import ROOT
from src.models.linear import LinearSoftmax
from src.evaluation.metrics import metrics, fit_temperature, scale

def main():
    base = ROOT / "experiments/results"
    config = json.loads((ROOT / "experiments/configs/main.json").read_text())
    controls = json.loads((ROOT / "experiments/configs/reviewer_controls.json").read_text())
    rows = {r["id"]: r for r in json.loads((ROOT / "data/processed/sst5.json").read_text())}
    output = []
    for budget in config["budgets"]:
        for seed in config["seeds"]:
            dest = base / f"budget-{budget}_seed-{seed}"
            print(dest.name, flush=True)
            if (dest / "controls_metrics.json").exists():
                output.extend(json.loads((dest / "controls_metrics.json").read_text()))
                continue
            splits = json.loads((dest / "splits.json").read_text())
            selection = json.loads((dest / "selection.json").read_text())
            with (dest / "models.pkl").open("rb") as f:
                artifact = pickle.load(f)
            vect = artifact["vectorizer"]
            Xs, ys = {}, {}
            for s, ids in splits.items():
                Xs[s] = vect.transform([rows[i]["text"] for i in ids])
                ys[s] = np.array([rows[i]["label"] for i in ids])
            candidates = []
            def fit(family, **kwargs):
                model = LinearSoftmax(**kwargs).fit(Xs["train"], ys["train"])
                assert model.info["success"], model.info
                val = metrics(model.predict_proba(Xs["selection"]), ys["selection"])
                candidates.append({"family": family, "params": kwargs, "selection": val, "fit": model.info})
                return model, val, kwargs
            ce_options = [fit("CE-MAE", C=c) for c in config["C_grid"]]
            ce = min(ce_options, key=lambda t: (t[1]["mae"], t[1]["nll"]))
            C = selection["models"]["CE"]["C"]
            dp_options = [fit("DP2-NLL", C=C, kind="distance", power=2, lam=lam)
                          for lam in config["lambda_grid"]]
            dp = min(dp_options, key=lambda t: t[1]["nll"])
            fixed = next(t for t in dp_options if t[2]["lam"] == controls["fixed_quadratic_lambda"])
            oll_options = [fit("OLL-reg", C=c, kind="oll", power=1) for c in controls["OLL_C_grid"]]
            oll = min(oll_options, key=lambda t: t[1]["nll"])
            selected = {"CE-MAE": ce, "DP2-NLL": dp, "DP2-fixed": fixed, "OLL-reg": oll}
            temp = fit_temperature(oll[0].predict_proba(Xs["calibration"]), ys["calibration"])
            (dest / "controls_selection.json").write_text(json.dumps(
                {"choices": {name: t[2] for name, t in selected.items()}, "OLL_temperature": temp}, indent=2))
            (dest / "controls_candidates.json").write_text(json.dumps(candidates, indent=2))
            pred = {"y": ys["test"]}
            values = []
            for name, t in selected.items():
                p = t[0].predict_proba(Xs["test"])
                pred[name] = p
                values.append({"budget": str(budget), "seed": seed, "method": name,
                               **metrics(p, ys["test"]), "fit_seconds": t[0].info["fit_seconds"]})
            p = scale(pred["OLL-reg"], temp)
            pred["OLL-reg+TS"] = p
            values.append({"budget": str(budget), "seed": seed, "method": "OLL-reg+TS",
                           **metrics(p, ys["test"]), "fit_seconds": oll[0].info["fit_seconds"]})
            np.savez_compressed(dest / "controls_predictions.npz", **pred)
            (dest / "controls_metrics.json").write_text(json.dumps(values, indent=2))
            with (dest / "controls_models.pkl").open("wb") as f:
                pickle.dump({name: t[0] for name, t in selected.items()}, f)
            output.extend(values)
    pd.DataFrame(output).to_csv(base / "controls_metrics.csv", index=False)

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
