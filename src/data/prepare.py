"""Extract official root sentence labels and audit exact normalized duplicates."""
import hashlib
import json
import re
import zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def canonical(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

def prepare():
    archive = ROOT / "data/raw/stanfordSentimentTreebank.zip"
    with zipfile.ZipFile(archive) as z:
        def read(name):
            return z.read("stanfordSentimentTreebank/" + name).decode("utf-8").splitlines()
        dictionary = {}
        for line in read("dictionary.txt"):
            text, pid = line.rsplit("|", 1)
            dictionary[text] = int(pid)
        sentiments = {int(a): float(b) for a, b in
                      (s.split("|") for s in read("sentiment_labels.txt")[1:])}
        split = {int(a): int(b) for a, b in
                 (s.split(",") for s in read("datasetSplit.txt")[1:])}
        roots = read("SOStr.txt")
        rows = []
        for sid, line in enumerate(roots, 1):
            text = " ".join(line.split("|"))
            if text not in dictionary:
                raise ValueError(f"Root missing from phrase dictionary: {sid}: {text}")
            value = sentiments[dictionary[text]]
            # Official bins: [0,.2], (.2,.4], (.4,.6], (.6,.8], (.8,1].
            y = int(np.searchsorted([.2, .4, .6, .8], value, side="left"))
            rows.append({"id": sid, "text": text, "label": y,
                         "split": {1: "train", 2: "test", 3: "dev"}[split[sid]]})
    original_counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "dev", "test")}
    groups = {}
    for r in rows:
        groups.setdefault(canonical(r["text"]), []).append(r)
    conflicts = {k for k, rs in groups.items() if len({r["label"] for r in rs}) > 1}
    kept, excluded = [], []
    priority = {"test": 0, "dev": 1, "train": 2}
    for key, rs in groups.items():
        if key in conflicts:
            excluded.extend({**r, "reason": "conflicting_labels"} for r in rs)
        else:
            ordered = sorted(rs, key=lambda r: (priority[r["split"]], r["id"]))
            kept.append(ordered[0])
            excluded.extend({**r, "reason": "duplicate"} for r in ordered[1:])
    kept.sort(key=lambda r: r["id"])
    audit = {"source_url": "https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip",
             "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
             "original_counts": original_counts, "conflicting_text_groups": len(conflicts),
             "excluded": excluded, "clean_counts": {}, "class_counts": {}}
    for s in original_counts:
        ys = [r["label"] for r in kept if r["split"] == s]
        audit["clean_counts"][s] = len(ys)
        audit["class_counts"][s] = np.bincount(ys, minlength=5).tolist()
    (ROOT / "data/processed").mkdir(exist_ok=True)
    (ROOT / "data/processed/sst5.json").write_text(json.dumps(kept, ensure_ascii=False, indent=2))
    (ROOT / "experiments/results/data_audit.json").write_text(json.dumps(audit, indent=2))
    return kept, audit

if __name__ == "__main__":
    _, audit = prepare()
    print(json.dumps({k: v for k, v in audit.items() if k != "excluded"}, indent=2))
