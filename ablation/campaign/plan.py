"""Build the prospective rebuttal DAG without launching scientific training."""
import argparse
from collections import Counter
from itertools import product
import json
import csv
from pathlib import Path

from .schema import SCHEMA, canonical_hash, validate_plan
from ablation.p05_hparam_grid.grid import configs_for, REFERENCE, REFERENCE_NAME

DATASETS = ["roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"]
MODELS = ["ResNet", "GCN", "SAGE", "GAT", "GAT-sep", "GT", "GT-sep"]
NODE_PROTOCOL = dict(min_steps=5000, patience=1000, num_steps=20000,
                     continuation_steps=40000, selection_interval=1,
                     hidden_dim=512, defer_test=True, save_selected=True)
VARIANTS = {
    "ones_all": ("p02_full_ablations", 4),
    "input_only": ("p01_placement", 4),
    "output_only_identity": ("p01_placement", 4),
    "output_only_signs": ("p01_placement", 4),
    "all_layer_be": ("p01_placement", 4),
    "m2": ("p02_full_ablations", 2),
    "m8": ("p02_full_ablations", 8),
    "xavier_all": ("p02_full_ablations", 4),
    "xavier_rms": ("p03_rms_init", 4),
    "xavier_row_rms": ("p03_rms_init", 4),
    "xavier_same_weights": ("p03_rms_init", 4),
    "xavier_activation_rms": ("p03_rms_init", 4),
    "identity_both": ("p03_rms_init", 4),
    "signs_both": ("p03_rms_init", 4),
    "identity_input_signs_output": ("p03_rms_init", 4),
    "independent_projectors": ("p04_shared_heads", 4),
    "heads_single_pass": ("p04_shared_heads", 4),
}


# Only the colleague's requested studies and the owner's recorded additions
# enter the active plan. Extra controls remain available as historical code.
ACTIVE_ROMAN_VARIANTS = (
    "input_only", "output_only_identity", "output_only_signs", "all_layer_be",
    "m2", "m8", "xavier_all", "xavier_rms", "ones_all", "heads_single_pass",
)

def historical_spec(dataset, model, depth, split, method, contrast=False):
    spec = node_spec(dataset, model, depth, split, method)
    cap = 20000 if dataset == "tolokers" else 5000
    spec.update(min_steps=0, patience=300, num_steps=cap, continuation_steps=cap,
                selection_interval=10, seed=split if method == "gnnm" else
                1000 * split + int(method[-1]))
    if contrast:
        spec.update(model_module="ablation.p13_node_contrast.models",
                    variant="gnnm_contrast", method="gnnm", members=4)
    return spec


def node_spec(dataset, model, depth, split, method, hp=None):
    hp = dict(hp or REFERENCE)
    member = int(method[-1]) if method.startswith("native") else 0
    module, members = VARIANTS.get(method, ("p05_hparam_grid", 4 if method == "gnnm" else 1))
    return dict(NODE_PROTOCOL, dataset=dataset, model=model, num_layers=depth,
                split=split, seed=1000 * split + member, members=members,
                method="gnnm" if method == "gnnm" else "base" if method.startswith("native") else method,
                variant=method if method in VARIANTS else None,
                model_module=f"ablation.{module}.models",
                weight_decay=hp.pop("weight_decay", 0.0), **hp)

class Builder:
    def __init__(self):
        self.tasks = {}
        self.groups = {}

    def add(self, task_id, action, metadata, deps=(), priority=20, kind="cpu",
            when=None, memory=8, exclusive=False, optional=()):
        task = dict(id=task_id, kind=kind,
                    argv=["{python}", "-m", "ablation.campaign.tasks",
                          "--plan", "{plan}", "--root", "{root}",
                          "--task", task_id, "--device", "{device}"],
                    dependencies=sorted(set(deps)), optional_dependencies=sorted(set(optional)),
                    priority=priority, resources=dict(memory_gb=memory, cpus=2, exclusive=exclusive),
                    max_attempts=3, required_outputs=["output/done.json"],
                    metadata=dict(action=action, **metadata))
        if when:
            task["when"] = when
        if task_id in self.tasks:
            self.tasks[task_id]["priority"] = min(priority, self.tasks[task_id]["priority"])
        else:
            self.tasks[task_id] = task
        return task_id

    def fit(self, spec, stage, priority, deps=(), when=None, resolve=None):
        payload = dict(spec=spec, resolve=resolve)
        task_id = "fit/" + canonical_hash(payload)[:20]
        memory = (64 if spec.get("variant") == "gnnm_contrast" else
                  24 if spec["model"] in ("GAT", "GAT-sep", "GT", "GT-sep") else 12)
        self.add(task_id, "node_fit", dict(**payload, stage=stage), deps,
                 priority, "gpu", when, memory=memory)
        return task_id

    def cell(self, dataset, model, depth, split, hp, label, stage, priority,
             deps=(), when=None, resolve=None):
        fits = [self.fit(node_spec(dataset, model, depth, split, method, hp),
                         stage, priority, deps, when, resolve)
                for method in ("native0", "native1", "native2", "native3", "gnnm")]
        task_id = "cell/" + canonical_hash(fits)[:20]
        self.add(task_id, "node_cell", dict(fits=fits, dataset=dataset, model=model,
                 depth=depth, split=split, config_name=label, stage=stage),
                 [*deps, *fits], priority + 1, when=when)
        if split == 0 and stage in ("gat_pilot", "expanded_pilot"):
            self.add("budget_profile/" + canonical_hash(fits)[:20], "budget_profile",
                     dict(fits=fits, dataset=dataset, model=model, depth=depth,
                          config_name=label, stage=stage),
                     [*deps, *fits], priority + 2, "gpu", when,
                     memory=24, exclusive=True)
        return task_id, fits

    def selected(self, decision, dataset, model, stage, when=None, gate=None):
        all_scores = []
        for split in range(3, 10):
            fits = {}
            for setting, methods in (("BASE", ["native0"]), ("ENS", [f"native{i}" for i in range(4)]),
                                     ("GNNM", ["gnnm"])):
                setting_condition = dict(task=decision, path="output/done.json",
                                         field="ready_" + setting, equals=True)
                fits[setting] = []
                for method in methods:
                    spec = node_spec(dataset, model, 1, split, method)
                    resolve = dict(selection=decision, setting=setting)
                    dependencies = [decision] + ([gate] if gate else [])
                    fits[setting].append(self.fit(spec, stage, 12 if model == "GAT" else 32,
                                                   dependencies, setting_condition, resolve))
                score_id = f"score/{stage}/{dataset}/{model}/split{split}/{setting}"
                self.add(score_id, "node_score", dict(fits=fits[setting], setting=setting,
                         dataset=dataset, model=model, split=split, stage=stage),
                         [*fits[setting], decision] + ([gate] if gate else []),
                         15 if model == "GAT" else 35, "gpu", setting_condition, memory=24)
                all_scores.append(score_id)
        report_id = f"report/{stage}/{dataset}/{model}"
        self.add(report_id, "comparison", dict(scores=all_scores, stage=stage),
                 [gate] if gate else [], 16, when=when,
                 optional=all_scores + [decision])
        return report_id

def build_plan():
    b = Builder()
    decisions, reports = [], []
    grid = configs_for("combined")
    for dataset in DATASETS:
        cells = []
        for name, hp in grid.items():
            for depth, split in product(range(1, 6), range(3)):
                cell, _ = b.cell(dataset, "GAT", depth, split, hp, name, "gat_pilot", 2)
                cells.append(cell)
        decision = f"decision/gat/{dataset}"
        b.add(decision, "select", dict(cells=cells, dataset=dataset, model="GAT",
              pilot_splits=[0, 1, 2]), cells, 5)
        profile_ids = [task["id"] for task in b.tasks.values()
                       if task["metadata"]["action"] == "budget_profile"
                       and task["metadata"]["dataset"] == dataset
                       and task["metadata"]["model"] == "GAT"]
        budget = f"budget/gat/{dataset}"
        b.add(budget, "budget_report", dict(cells=cells, profiles=profile_ids),
              [*cells, *profile_ids], 21)
        reports.append(budget)
        reports.append(b.selected(budget, dataset, "GAT", "gat_budget_confirmation"))
        decisions.append(decision)
        reports.append(b.selected(decision, dataset, "GAT", "gat_confirmation"))
    gate = "decision/expand"
    b.add(gate, "expand", dict(decisions=decisions,
          rule="GNNM best VALID exceeds independently tuned ENS on at least 3 of 5 datasets",
          common_grid_rule="reference plus two LR/WD configurations with best mean within-dataset within-method VALID rank",
          candidate_names=list(configs_for("lrwd"))), decisions, 7)
    when = dict(task=gate, path="output/done.json", field="expand", equals=True)
    for model in [name for name in MODELS if name != "GAT"]:
        for dataset in DATASETS:
            cells = []
            for slot, depth, split in product(range(3), range(1, 6), range(3)):
                cell, _ = b.cell(dataset, model, depth, split, REFERENCE, f"slot{slot}",
                    "expanded_pilot", 25, [gate], when, dict(gate=gate, grid_slot=slot))
                cells.append(cell)
            decision = f"decision/expanded/{dataset}/{model}"
            b.add(decision, "select", dict(cells=cells, dataset=dataset, model=model,
                  pilot_splits=[0, 1, 2]), [*cells, gate], 28, when=when)
            profile_ids = [task["id"] for task in b.tasks.values()
                           if task["metadata"]["action"] == "budget_profile"
                           and task["metadata"]["dataset"] == dataset
                           and task["metadata"]["model"] == model]
            budget = f"budget/expanded/{dataset}/{model}"
            b.add(budget, "budget_report", dict(cells=cells, profiles=profile_ids),
                  [*cells, *profile_ids, gate], 40, when=when)
            reports.append(budget)
            reports.append(b.selected(budget, dataset, model, "expanded_budget_confirmation", when, gate))
            reports.append(b.selected(decision, dataset, model, "expanded_confirmation", when, gate))
    # Full paired Roman Empire matrix. Canonically identical GAT pilot fits are reused.
    roman_scores = []
    for model, depth, split in product(MODELS, range(1, 6), range(10)):
        cell, fits = b.cell("roman-empire", model, depth, split, REFERENCE, REFERENCE_NAME,
                            "roman_controls", 10)
        groups = {"BASE": fits[:1], "ENS": fits[:4], "gnnm": fits[4:]}
        for variant in ACTIVE_ROMAN_VARIANTS:
            groups[variant] = [b.fit(node_spec("roman-empire", model, depth, split, variant),
                                     "roman_ablation", 11)]
        for setting, group in groups.items():
            score_id = f"score/roman/{model}/L{depth}/split{split}/{setting}"
            b.add(score_id, "node_score", dict(fits=group, setting=setting, dataset="roman-empire",
                  model=model, depth=depth, split=split, stage="roman_ablation"),
                  group, 18, "gpu", memory=24)
            roman_scores.append(score_id)
    b.add("report/roman", "comparison", dict(scores=roman_scores, stage="roman_ablation"),
          roman_scores, 20)
    reports.append("report/roman")
    # Dedicated diversity study: one fixed split and each method's saved
    # historical VALID-selected depth, across all five graphs and seven backbones.
    selected_path = Path(__file__).resolve().parents[1] / "p06_member_diversity/selected_depths.csv"
    diversity_reports = []
    with selected_path.open(newline="") as stream:
        selected_rows = list(csv.DictReader(stream))
    for row in selected_rows:
        dataset, model, method, depth = row["dataset"], row["model"], row["method"], int(row["num_layers"])
        members = ["gnnm"] if method == "gnnm" else [f"native{i}" for i in range(4)]
        fits = [b.fit(historical_spec(dataset, model, depth, 0, member),
                      "selected_diversity", 18) for member in members]
        score = f"selected_diversity_score/{dataset}/{model}/{method}"
        b.add(score, "node_score", dict(fits=fits, setting=method, dataset=dataset,
              model=model, split=0, stage="selected_diversity", roles=["val"]),
              fits, 20, "gpu", memory=24)
        identifier = f"selected_diversity/{dataset}/{model}/{method}"
        b.add(identifier, "diversity", dict(score=score, setting=method,
              dataset=dataset, model=model, split=0), [score], 22, "gpu", memory=24)
        diversity_reports.append(identifier)
    b.add("report/selected_diversity", "diversity_report", dict(results=diversity_reports),
          [], 30, optional=diversity_reports)
    reports.append("report/selected_diversity")

    # Native/ensemble/GNNM and requested ablations retain ordinary cost profiles.
    for model, setting in product(MODELS, ["BASE", "ENS", "gnnm", *ACTIVE_ROMAN_VARIANTS]):
        candidates = [f"score/roman/{model}/L{depth}/split0/{setting}" for depth in range(1, 6)]
        identifier = f"profile/roman/{model}/{setting}"
        b.add(identifier, "profile", dict(candidates=candidates, setting=setting,
              dataset="roman-empire", model=model, split=0), candidates, 23, "gpu",
              memory=24, exclusive=True)
        reports.append(identifier)

    # Fixed historical winners with exactly one extra objective; no new HPO.
    contrast_scores = []
    baseline_path = Path(__file__).resolve().parents[1] / "p13_node_contrast/historical_baseline.csv"
    with baseline_path.open(newline="") as stream:
        frozen_baselines = list(csv.DictReader(stream))
    for row in frozen_baselines:
        dataset, model, split, depth = row["dataset"], row["model"], int(row["split"]), int(row["num_layers"])
        fit = b.fit(historical_spec(dataset, model, depth, split, "gnnm", contrast=True),
                    "node_contrast", 16)
        score = f"node_contrast_score/{dataset}/{model}/split{split}"
        b.add(score, "node_score", dict(fits=[fit], setting="gnnm_contrast", dataset=dataset,
              model=model, split=split, stage="node_contrast"), [fit], 21, "gpu", memory=64)
        contrast_scores.append(score)
    b.add("report/node_contrast", "historical_contrast_report", dict(results=contrast_scores),
          [], 31, optional=contrast_scores)
    reports.append("report/node_contrast")

    for model, depth, setting in product(MODELS, range(1, 6),
            ["BASE", "ENS", "gnnm", "m2", "m8", "heads_single_pass"]):
        score = f"score/roman/{model}/L{depth}/split0/{setting}"
        task_id = f"scaling/{model}/L{depth}/{setting}"
        b.add(task_id, "scaling_profile", dict(score=score, setting=setting,
              dataset="roman-empire", model=model, depth=depth, split=0),
              [score], 24, "gpu", memory=24, exclusive=True)
        reports.append(task_id)
    for model, depth, variant in product(MODELS, range(1, 6),
                 ["gnnm", "xavier_all", "xavier_rms", "ones_all"]):
        identifier = f"initialization/{model}/L{depth}/{variant}"
        b.add(identifier, "initialization", dict(spec=node_spec("roman-empire", model, depth, 0, variant)),
              priority=14, kind="gpu", memory=24)
        reports.append(identifier)
    transfer_scores = []
    for pair, module, cohorts, methods in [
        ("molbace_gcn", "p09_molecular", 10,
         ["native0", "native1", "native2", "native3", "gnnm", "gnnm_contrast", "private_bn", "gnnm_private_bn_contrast"]),
        ("collab_buddy", "p11_collab_buddy", 5,
         ["native0", "native1", "native2", "native3", "gnnm", "gnnm_contrast", "private_bn", "link_same_query_global"]),
    ]:
        for cohort in range(cohorts):
            method_scores = []
            for method in methods:
                task_id = f"transfer/{pair}/cohort{cohort}/{method}"
                b.add(task_id, "transfer_fit", dict(pair=pair, module=module, cohort=cohort,
                      method=method, min_epochs=0, epochs=1000, patience=50,
                      continuation_epochs=2000, defer_test=True), priority=8, kind="gpu",
                      memory=40 if "collab" in pair else 12)
                method_score = f"transfer_method_score/{pair}/cohort{cohort}/{method}"
                b.add(method_score, "transfer_score",
                      dict(fits=[task_id], methods=[method], pair=pair, module=module, cohort=cohort),
                      [task_id], 12, "gpu", memory=40 if "collab" in pair else 12)
                method_scores.append(dict(method=method, task=method_score, fit=task_id))
            score_id = f"transfer_score/{pair}/cohort{cohort}"
            b.add(score_id, "transfer_score",
                  dict(method_scores=method_scores, methods=methods, pair=pair, module=module, cohort=cohort),
                  [], 13, "cpu", memory=40 if "collab" in pair else 12,
                  optional=[entry["task"] for entry in method_scores])
            transfer_scores.append(score_id)
    b.add("report/transfer", "transfer_report", dict(scores=transfer_scores),
          [], 17, optional=transfer_scores)
    reports.append("report/transfer")
    b.add("report/final", "final_report", dict(reports=reports),
          [], 100, optional=reports)
    plan = dict(schema=SCHEMA, campaign_id="iclr-rebuttal-owner-20261112",
                protocol=dict(node=NODE_PROTOCOL, pilot_splits=[0, 1, 2],
                  confirmation_splits=list(range(3, 10)), grid=grid,
                  freeze="Decisions use VALID only; all outcomes and failures retained",
                  original_paper_scores="unchanged"),
                tasks=list(b.tasks.values()))
    validate_plan(plan)
    return plan

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2) + "\n")
    counts = Counter((task["metadata"]["action"], bool(task.get("when"))) for task in plan["tasks"])
    fits = [task for task in plan["tasks"] if task["metadata"]["action"] in ("node_fit", "transfer_fit")]
    expanded = [task for task in fits if task["metadata"].get("stage", "").startswith("expanded")]
    summary = dict(campaign_id=plan["campaign_id"], tasks=len(plan["tasks"]),
                   scientific_fit_counts=dict(total=len(fits), conditional_expansion=len(expanded),
                                              planned_core=len(fits) - len(expanded)),
                   counts={f"{action}/{'conditional' if conditional else 'unconditional'}": n
                           for (action, conditional), n in sorted(counts.items())})
    args.out.with_suffix(".counts.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
