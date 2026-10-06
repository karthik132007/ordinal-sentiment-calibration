"""Regenerate summaries, conditional uncertainty, figures, and LaTeX numbers."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.data.prepare import ROOT
from src.evaluation.metrics import metrics, reliability

OUT = ROOT / "experiments/results"
FIG = ROOT / "figures"
PAPER = ROOT / "paper"

def main():
    d = pd.read_csv(OUT / "metrics.csv", dtype={"budget": str})
    controls_path = OUT / "controls_metrics.csv"
    if controls_path.exists():
        d = pd.concat([d, pd.read_csv(controls_path, dtype={"budget": str})], ignore_index=True)
    stats = d.groupby(["budget", "method"]).agg({c: ["mean", "std"] for c in
        ["accuracy", "macro_f1", "mae", "severe", "ece5", "ece10", "ece15", "ece30",
         "nll", "brier", "rps", "central_fraction", "fit_seconds"]})
    stats.to_csv(OUT / "summary.csv")
    facts = {f"{budget}/{method}": {col: {"mean": float(stats.loc[(budget, method), (col, "mean")]),
                                         "std": float(stats.loc[(budget, method), (col, "std")])}
             for col in ["accuracy", "macro_f1", "mae", "severe", "ece15", "nll", "brier", "rps", "central_fraction"]}
             for budget, method in stats.index}
    (OUT / "facts.json").write_text(json.dumps(facts, indent=2))
    # All scalar result macros originate here, never hand-edited into the manuscript.
    names = {"CE": "CE", "CE+TS": "CETS", "CE-median": "Median", "NB": "NB",
             "Cumulative": "Cum", "DP1": "DPone", "DP2": "DPtwo", "DP2+TS": "DPtwoTS",
             "OLL": "OLL", "OLL+TS": "OLLTS", "LS": "LS", "LS+TS": "LSTS",
             "CE-MAE": "CEMAE", "DP2-NLL": "DPNLL", "DP2-fixed": "DPfixed",
             "OLL-reg": "OLLreg", "OLL-reg+TS": "OLLregTS"}
    metricnames = {"accuracy": "Acc", "macro_f1": "Fone", "mae": "MAE", "severe": "Severe",
                   "ece15": "ECE", "nll": "NLL", "brier": "Brier", "rps": "RPS",
                   "central_fraction": "Central"}
    macros = []
    for (budget, method), row in stats.iterrows():
        if method not in names:
            continue
        prefix = "Full" if budget == "full" else "Low"
        for col, suffix in metricnames.items():
            number = row[(col, "mean")]
            value = f"{100 * number:.2f}" if col in ["accuracy", "macro_f1", "severe", "ece15", "central_fraction"] else f"{number:.3f}"
            macros.append(chr(92) + "newcommand{" + chr(92) + prefix + names[method] + suffix + "}{" + value + "}")
    audit = json.loads((OUT / "data_audit.json").read_text())
    for name, value in [("TrainCount", audit["clean_counts"]["train"]), ("DevCount", audit["clean_counts"]["dev"]),
                        ("TestCount", audit["clean_counts"]["test"]), ("RemovedCount", len(audit["excluded"]))]:
        macros.append(chr(92) + "newcommand{" + chr(92) + name + "}{" + str(value) + "}")
    (PAPER / "numbers.tex").write_text("\n".join(macros) + "\n")

    def table(budget, methods, columns, filename):
        labels = {"accuracy": "Acc. (\\%)", "macro_f1": "F1 (\\%)", "mae": "MAE", "severe": "Sev. (\\%)",
                  "ece15": "ECE (\\%)", "nll": "NLL", "brier": "Brier", "rps": "RPS"}
        lines = [r"\begin{tabular}{l" + "r" * len(columns) + "}", r"\toprule",
                 "Method & " + " & ".join(labels[c] for c in columns) + r" \\", r"\midrule"]
        for method in methods:
            if (budget, method) not in stats.index:
                continue
            values = []
            for col in columns:
                mean = stats.loc[(budget, method), (col, "mean")]
                sd = stats.loc[(budget, method), (col, "std")]
                factor = 100 if col in ["accuracy", "macro_f1", "severe", "ece15"] else 1
                fmt = ".2f" if factor == 100 else ".3f"
                values.append("$" + format(mean * factor, fmt) + r"\!\pm\!" + format(sd * factor, fmt) + "$")
            lines.append(method.replace("CE-median", "CE median").replace("Cumulative", "Cumulative LR") + " & " + " & ".join(values) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        (PAPER / filename).write_text("\n".join(lines))
    methods = ["NB", "CE", "CE+TS", "CE-median", "Cumulative", "DP1", "DP2", "DP2+TS", "LS", "LS+TS", "OLL", "OLL+TS"]
    table("full", methods, ["accuracy", "macro_f1", "mae", "severe", "ece15", "nll"], "main_table.tex")
    table("2000", methods, ["accuracy", "mae", "severe", "ece15", "nll"], "low_table.tex")
    table("full", ["CE", "CE-MAE", "DP2", "DP2-NLL", "DP2-fixed", "OLL-reg", "OLL-reg+TS"],
          ["accuracy", "mae", "severe", "ece15", "nll"], "controls_table.tex")

    preds = {seed: np.load(OUT / f"budget-full_seed-{seed}/predictions.npz") for seed in [42, 123, 999]}
    y = preds[42]["y"]
    assert all(np.array_equal(y, p["y"]) for p in preds.values())
    confidence_intervals = []
    rng = np.random.default_rng(2026)
    for comparison in ["CE", "CE+TS", "Cumulative"]:
        for metric in ["mae", "severe", "nll", "accuracy"]:
            def per_sample(p):
                pred = p.argmax(axis=1)
                if metric == "mae": return np.abs(pred - y)
                if metric == "severe": return (np.abs(pred - y) >= 2).astype(float)
                if metric == "accuracy": return (pred == y).astype(float)
                # Match sklearn clipping at machine epsilon.
                clipped = np.clip(p, np.finfo(p.dtype).eps, 1 - np.finfo(p.dtype).eps)
                return -np.log(clipped[np.arange(len(y)), y])
            delta = np.mean([per_sample(p["DP2"]) - per_sample(p[comparison]) for p in preds.values()], axis=0)
            boot = np.array([delta[rng.integers(0, len(y), len(y))].mean() for _ in range(5000)])
            lo, hi = np.quantile(boot, [.025, .975])
            confidence_intervals.append({"method": "DP2", "reference": comparison, "metric": metric,
                                         "difference": float(delta.mean()), "low": float(lo), "high": float(hi)})
    (OUT / "bootstrap.json").write_text(json.dumps(confidence_intervals, indent=2))
    lines = [r"\begin{tabular}{llrrr}", r"\toprule", r"Reference & Metric & Difference & Lower & Upper \\", r"\midrule"]
    for item in confidence_intervals:
        if item["reference"] == "CE+TS": continue
        factor = 100 if item["metric"] in ["severe", "accuracy"] else 1
        label = "Acc." if item["metric"] == "accuracy" else item["metric"].upper()
        lines.append(item["reference"].replace("Cumulative", "Cumulative LR") + " & " + label + " & " +
                     " & ".join(f"{item[c] * factor:.3f}" for c in ["difference", "low", "high"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "bootstrap_table.tex").write_text("\n".join(lines))

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
    colors = {"CE": "#4b5563", "CE+TS": "#2563eb", "DP2": "#b45309", "DP2+TS": "#047857", "OLL": "#9333ea"}
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.15))
    for budget, marker in [("2000", "o"), ("full", "s")]:
        for method in ["CE", "DP2", "Cumulative", "CE-median"]:
            row = stats.loc[(budget, method)]
            label = f"{method}, {'2k' if budget == '2000' else 'full'}"
            ax[0].scatter(100 * row[("accuracy", "mean")], row[("mae", "mean")], marker=marker,
                          color=colors.get(method, "#0f766e" if method == "Cumulative" else "#9333ea"), s=35, label=label)
    ax[0].set(xlabel="Accuracy (%)", ylabel="Mean absolute error (lower is better)")
    ax[0].legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.27), frameon=False)
    for budget, offset in [("2000", -.15), ("full", .15)]:
        methods2 = ["CE", "CE+TS", "DP2", "DP2+TS"]
        means = [100 * stats.loc[(budget, m), ("ece15", "mean")] for m in methods2]
        sds = [100 * stats.loc[(budget, m), ("ece15", "std")] for m in methods2]
        ax[1].bar(np.arange(4) + offset, means, .28, yerr=sds, label=f"{budget} labels", alpha=.85,
                  color="#64748b" if budget == "2000" else "#b45309", capsize=2)
    ax[1].set_xticks(np.arange(4), methods2)
    ax[1].set(ylabel="15-bin ECE (%)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    for ext in ["pdf", "png"]: fig.savefig(FIG / f"tradeoffs.{ext}", dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for method in ["CE", "CE+TS", "DP2", "DP2+TS"]:
        # Representative seed, not pooled predictions treated as independent data.
        curve = reliability(preds[42][method], y)
        axs[0].plot([r["confidence"] for r in curve], [r["accuracy"] for r in curve],
                    marker="o", markersize=3, label=method, color=colors[method])
        conf = preds[42][method].max(axis=1)
        axs[1].hist(conf, bins=np.linspace(0, 1, 16), histtype="step", label=method, color=colors[method])
    axs[0].plot([0, 1], [0, 1], "k--", linewidth=.8)
    axs[0].set(xlabel="Mean confidence in bin", ylabel="Accuracy in bin", xlim=(0, 1), ylim=(0, 1))
    axs[0].legend(fontsize=7)
    axs[1].set(xlabel="Maximum predicted probability", ylabel="Sentence count")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    for ext in ["pdf", "png"]: fig.savefig(FIG / f"reliability.{ext}", dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5))
    diagnostics = {}
    for ax, method in zip(axs, ["CE", "DP2", "OLL"]):
        p = preds[42][method]
        cm = confusion_matrix(y, p.argmax(axis=1), labels=np.arange(5), normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for a in range(5):
            for b in range(5):
                ax.text(b, a, f"{cm[a,b]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cm[a,b] > .55 else "black")
        ax.set(title=method, xlabel="Predicted level", xticks=range(5), yticks=range(5))
        diagnostics[method] = {"mean_probability": p.mean(axis=0).tolist(),
            "predicted_class_counts": np.bincount(p.argmax(axis=1), minlength=5).tolist(),
            "min_probability": float(p.min()),
            "true_class_mean_probability": [float(p[y == k, k].mean()) for k in range(5)],
            "class_mae": [float(np.abs(p[y == k].argmax(axis=1) - k).mean()) for k in range(5)],
            "class_nll": [float(-np.log(np.clip(p[y == k,k], np.finfo(float).eps, 1)).mean()) for k in range(5)]}
    axs[0].set_ylabel("True level")
    fig.colorbar(im, ax=axs, fraction=.02, pad=.025, label="Row fraction")
    fig.subplots_adjust(left=.07, right=.89, bottom=.2, top=.84, wspace=.3)
    for ext in ["pdf", "png"]: fig.savefig(FIG / f"confusions.{ext}", dpi=200)
    plt.close(fig)
    (OUT / "class_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    all_candidates = json.loads((OUT / "candidates.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.55))
    for budget, color in [("2000", "#64748b"), ("full", "#b45309")]:
        entries = [c for c in all_candidates if c["family"] == "DP2" and c["budget"] == budget]
        for index, metric in enumerate(["mae", "nll"]):
            v = pd.DataFrame([{"lam": c["params"]["lambda"], "value": c["selection"][metric]} for c in entries])
            group = v.groupby("lam")["value"].agg(["mean", "std"])
            axs[index].errorbar(group.index, group["mean"], yerr=group["std"], marker="o", capsize=3,
                                label=f"{budget} labels", color=color)
            axs[index].set(xlabel="Quadratic penalty weight", ylabel=f"Selection {metric.upper()}")
            axs[index].legend(fontsize=7)
    fig.tight_layout()
    for ext in ["pdf", "png"]: fig.savefig(FIG / f"lambda_selection.{ext}", dpi=200)
    plt.close(fig)
    print(stats.loc[("full", ["CE", "DP2", "Cumulative", "OLL"]),
                    [("accuracy", "mean"), ("mae", "mean"), ("ece15", "mean"), ("nll", "mean")]].to_string())

if __name__ == "__main__":
    main()
