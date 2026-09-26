"""Plot every audited initial-update case. No accuracy values are used."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--audit", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
audit = json.loads(a.audit.read_text())
assert audit["status"] == "PASS"
rows = audit["rows"]
graphs = ("cora", "wikics", "actor", "chameleon_filtered")
labels = ("Cora", "WikiCS", "Actor", "Filtered\nChameleon")
assert {(r["dataset"], r["seed"]) for r in rows} == {
    (g, s) for g in graphs for s in range(3)}
assert len(rows) == 12
lookup = {(r["dataset"], r["seed"]): r for r in rows}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.65), sharex=True)
styles = [("#235e83", "o"), ("#ba6336", "s"), ("#4f7d48", "^")]
for ax, key, ylabel, title in zip(
    axes,
    ("update_cosine", "sync_over_tied_l2_norm_ratio"),
    ("Cosine of the two update vectors", "SYNC / TIED update norm"),
    ("Different update directions", "Different update magnitudes"),
):
    for seed, (color, marker) in enumerate(styles):
        x = [i + (seed - 1) * .13 for i in range(4)]
        y = [lookup[g, seed][key] for g in graphs]
        ax.scatter(x, y, color=color, marker=marker, s=43,
                   label=f"Seed {seed}", zorder=3)
    ax.axhline(1, color="#777777", linestyle="--", linewidth=1)
    ax.set_ylim(.45, 1.045)
    ax.set_xticks(range(4), labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, pad=10)
    ax.grid(axis="y", color="#dddddd", linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)
axes[1].legend(loc="lower right", frameon=False, fontsize=9)
fig.tight_layout(w_pad=2)
a.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.output, dpi=240, bbox_inches="tight", facecolor="white")
plt.close(fig)
metadata = {
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "audit_sha256": hashlib.sha256(a.audit.read_bytes()).hexdigest(),
    "figure_sha256": hashlib.sha256(a.output.read_bytes()).hexdigest(),
    "selection": "all 12 frozen graph/seed rows, none excluded",
    "scope": "first AdamW graph-parameter update, CPU, training labels, default settings",
    "interpretation": "directions and norms at initialization, no generalization claim",
}
a.output.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
table = [
    r"\begin{table}[t]", r"\centering",
    r"\caption{All twelve initial-update diagnostics. Cosine compares the graph-parameter update vectors. The norm ratio is $\|\Delta\theta_{\rm SYNC}\|_2/\|\Delta\theta_{\rm TIED}\|_2$. Both are computed at matched initial weights and dropout draws using training labels only.}",
    r"\label{tab:initial-update}", r"\small",
    r"\begin{tabular}{lrrr}", r"\toprule",
    r"Graph & Seed & Cosine & Norm ratio\\", r"\midrule",
]
for graph, label in zip(graphs, labels):
    for seed in range(3):
        row = lookup[graph, seed]
        table.append(f"{label.replace(chr(10), ' ')} & {seed} & {row['update_cosine']:.4f} & {row['sync_over_tied_l2_norm_ratio']:.4f}" + r"\\")
table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
a.output.with_suffix(".table.tex").write_text("\n".join(table) + "\n")
print(json.dumps(metadata, indent=2))
