"""Build a bounded real-data DAG smoke using the scientific entry points."""
import argparse
import json
from pathlib import Path
from .plan import Builder, node_spec
from .schema import SCHEMA, validate_plan
from ablation.p05_hparam_grid.grid import REFERENCE, REFERENCE_NAME

def build(include_transfer=False):
    builder = Builder()
    cells = []
    for split in range(3):
        cell, _ = builder.cell("roman-empire", "GAT", 1, split, REFERENCE,
                               REFERENCE_NAME, "engineering_smoke", 1)
        cells.append(cell)
    decision = "decision/smoke"
    builder.add(decision, "select", dict(cells=cells, dataset="roman-empire",
                model="GAT", pilot_splits=[0, 1, 2]), cells, 2)
    scores = []
    for setting, methods in (("BASE", ["native0"]),
                            ("ENS", ["native0", "native1", "native2", "native3"]),
                            ("GNNM", ["gnnm"])):
        fits = [builder.fit(node_spec("roman-empire", "GAT", 1, 3, method),
                    "engineering_smoke", 3, [decision],
                    resolve=dict(selection=decision, setting=setting)) for method in methods]
        score = f"score/smoke/{setting}"
        builder.add(score, "node_score", dict(fits=fits, setting=setting, dataset="roman-empire",
                    model="GAT", split=3, stage="engineering_smoke", roles=["val"]),
                    [*fits, decision], 4, "gpu", memory=24)
        scores.append(score)
        builder.add(f"diversity/smoke/{setting}", "diversity",
                    dict(score=score, setting=setting, dataset="roman-empire", model="GAT", split=3),
                    [score], 5, "gpu", memory=24)
    if include_transfer:
        for pair, module in [('molbace_gcn', 'p09_molecular'),
                             ('collab_buddy', 'p11_collab_buddy')]:
            builder.add(f'transfer/smoke/{pair}', 'transfer_fit',
                        dict(pair=pair, module=module, cohort=0, method='gnnm',
                             min_epochs=0, epochs=1000, patience=50,
                             continuation_epochs=2000, defer_test=True),
                        priority=6, kind='gpu', memory=40 if 'collab' in pair else 12)
    builder.add("report/smoke", "final_report",
                dict(reports=[decision, *scores]), optional=list(builder.tasks), priority=9)
    for task in builder.tasks.values():
        if task["metadata"]["action"] in ("node_fit", "transfer_fit"):
            task["argv"] += ["--smoke-updates", "2"]
    plan = dict(schema=SCHEMA, campaign_id="engineering-dag-smoke", tasks=list(builder.tasks.values()),
                role="engineering_smoke", scientific_results=False, heldout_scored=False)
    validate_plan(plan)
    return plan

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-transfer", action="store_true")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plan = build(include_transfer=args.include_transfer)
    args.out.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"tasks": len(plan["tasks"]), "scientific_results": False}))

if __name__ == "__main__":
    main()
