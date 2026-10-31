"""Stdlib complete-family byte/work bound; no scientific input is loaded."""
from .contract import ARMS, EPOCHS, SEEDS, TASKS, TIMING


def forecast(specification):
    if set(specification) != set(TASKS):
        raise ValueError("Both complete actual task payload specifications required")
    rows, total = [], 0
    validations = EPOCHS // 5
    for task in TASKS:
        spec = specification[task]
        if any(type(spec[k]) is not int or spec[k] <= 0 for k in ("validation_nodes", "classes")) or set(spec["arms"]) != set(ARMS):
            raise ValueError("Actual complete validation sizes and all arm payload bounds required")
        for arm in ARMS:
            m = 1 if arm.startswith("single_") else 4
            state = spec["arms"][arm]
            fields = ("model_tensor_bytes", "optimizer_tensor_bytes", "RNG_tensor_bytes")
            if any(type(state[k]) is not int or state[k] < 0 for k in fields):
                raise ValueError("Nonnegative actual/full-state payload bounds required")
            checkpoint = sum(state[k] for k in fields) + 2**20
            logits = (m + 1) * spec["validation_nodes"] * spec["classes"] * 4 + spec["validation_nodes"] * 8 + 2**20
            # Worst case each of60 validation checkpoints improves. No per-epoch logits.
            per_fit = (validations + 2) * checkpoint + 4 * logits + 16 * 2**20
            subtotal = len(SEEDS) * per_fit
            total += subtotal
            rows.append({"task": task, "arm": arm, "maximum_improved_checkpoints_per_seed": validations,
                         "initial_and_final_extra_checkpoint_bound": 2, "selected_final_fit_and_replay_logit_artifacts": 4,
                         "checkpoint_bytes_bound": checkpoint, "validation_logit_bytes_bound": logits,
                         "all_three_seeds_bytes_bound": subtotal})
    return {"schema": "gnnm-shared-message-complete30-artifact-bound-v2", "fit_count": 30,
            "updates": 9000, "rows": rows, "scientific_artifact_bytes_bound": total,
            "work_counts": {"training_updates": 9000, "complete_validation_forwards": 1800,
                            "scalar_validation_records": 1800, "initial_checkpoint_writes": 30,
                            "improved_checkpoint_write_upper": 1800, "fixed_final_checkpoint_writes": 30,
                            "fit_selected_final_logit_writes": 60, "independent_validation_forward_replays": 60,
                            "replay_logit_writes": 60, "trained_timing_forwards": 30 * (TIMING["warmups"] + TIMING["samples"]),
                            "fresh_fit_and_replay_input_loads": 60},
            "validation_exports_and_prepared_operator_prior_phase_bytes_must_be_added": True,
            "hashes_transfers_CPU_metrics_supervisor_and_manifest_costs_must_be_charged": True,
            "scientific_launch_or_resource_admission_established": False}
