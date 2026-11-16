"""Read official masks and record cross-split label-role overlap."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .plan import DATASETS

def audit(data_dir):
    rows = []
    for dataset in DATASETS:
        with np.load(data_dir / (dataset.replace("-", "_") + ".npz")) as data:
            roles = {}
            for role in ("train", "val", "test"):
                mask = data[role + "_masks"].astype(bool)
                roles[role] = mask.T if mask.shape[0] < mask.shape[1] else mask
            for pilot in range(3):
                for later in range(3, 10):
                    for role in ("train", "val"):
                        first, second = roles[role][:, pilot], roles["test"][:, later]
                        overlap = int(np.sum(first & second))
                        rows.append(dict(dataset=dataset, pilot_split=pilot, pilot_role=role,
                            later_split=later, later_role="test", overlap_nodes=overlap,
                            later_test_nodes=int(second.sum()),
                            later_test_fraction_seen=overlap / max(1, int(second.sum()))))
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = audit(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "split_overlap.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(cross_split_pairs=len(rows),
                   max_later_test_fraction_seen=max(row["later_test_fraction_seen"] for row in rows),
                   interpretation="Subsequent split evaluation on fixed graphs; overlapping label roles prevent an independent-holdout claim")
    (args.out / "split_overlap.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
if __name__ == "__main__":
    main()
