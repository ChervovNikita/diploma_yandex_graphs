"""Frozen validation-only decisions and complete search accounting."""
from collections import defaultdict
import math
import statistics
from ablation.p05_hparam_grid.grid import configs_for, REFERENCE_NAME

def select(cells):
    grouped = defaultdict(list)
    for cell in cells:
        if cell.get("role") != "val":
            raise ValueError("Selection requires explicitly validation-only cell artifacts")
        grouped[(cell["config_name"], cell["depth"])].append(cell)
    rankings = {}
    selections = {}
    for setting in ("BASE", "ENS", "GNNM"):
        ranked = []
        for (name, depth), rows in grouped.items():
            splits = sorted(row["split"] for row in rows)
            if splits != [0, 1, 2]:
                raise ValueError(f"Incomplete pilot cell {name}/{depth}: {splits}")
            values = [row["scores"][setting] for row in rows]
            if any(not math.isfinite(value) for value in values):
                raise ValueError("Non-finite validation score")
            ranked.append(dict(config_name=name, depth=depth,
                               mean=statistics.mean(values), values=values,
                               spec=rows[0]["specs"][setting], cells=[row["task_id"] for row in rows]))
        ranked.sort(key=lambda row: (-row["mean"], row["depth"], row["config_name"]))
        rankings[setting] = ranked
        selections[setting] = ranked[0]
    unresolved = sorted({task for cell in cells for task in cell.get("unresolved_convergence", [])})
    selected_cells = {name for selected in selections.values() for name in selected["cells"]}
    selected_unresolved = sorted({task for cell in cells if cell["task_id"] in selected_cells
                                  for task in cell.get("unresolved_convergence", [])})
    return dict(role="val", selections=selections, rankings=rankings,
                unresolved_convergence=unresolved, selected_unresolved=selected_unresolved,
                no_test_read=True, pilot_splits=[0, 1, 2],
                tie_break="lower depth then lexical configuration name")

def expansion(decisions):
    if len(decisions) != 5 or any(item.get("role") != "val" for item in decisions):
        raise ValueError("Expansion requires all five validation-only decisions")
    gains = [item["selections"]["GNNM"]["mean"] - item["selections"]["ENS"]["mean"]
             for item in decisions]
    candidate_names = set(configs_for("lrwd"))
    rank_totals = defaultdict(list)
    for item in decisions:
        for setting in ("BASE", "ENS", "GNNM"):
            best_depth = {}
            for row in item["rankings"][setting]:
                name = row["config_name"]
                if name in candidate_names:
                    best_depth[name] = max(best_depth.get(name, -math.inf), row["mean"])
            if set(best_depth) != candidate_names:
                raise ValueError("Incomplete common LR/WD candidate set")
            ordered = sorted(best_depth, key=lambda name: (-best_depth[name], name))
            for name in ordered:
                # Average rank for ties avoids lexical preference in aggregate ranking.
                larger = sum(value > best_depth[name] for value in best_depth.values())
                equal = sum(value == best_depth[name] for value in best_depth.values())
                rank_totals[name].append(1 + larger + (equal - 1) / 2)
    ranks = {name: statistics.mean(values) for name, values in rank_totals.items()}
    ordered = sorted(ranks, key=lambda name: (ranks[name], name))
    chosen = [REFERENCE_NAME] + [name for name in ordered if name != REFERENCE_NAME][:2]
    convergence_ready = not any(item.get("selected_unresolved") for item in decisions)
    return dict(role="val", expand=convergence_ready and sum(gain > 0 for gain in gains) >= 3,
                convergence_ready=convergence_ready,
                dataset_gains=gains, positive_datasets=sum(gain > 0 for gain in gains),
                chosen_names=chosen, chosen_configs=[configs_for("lrwd")[name] for name in chosen],
                common_ranks=ranks, no_test_read=True,
                rule="at least three positive dataset differences against independently tuned ENS",
                caveat="Validation-adaptive branch; confirmation splits 3–9 provide subsequent evaluation")
