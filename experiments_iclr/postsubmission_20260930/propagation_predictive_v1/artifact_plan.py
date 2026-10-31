"""Stdlib byte/work counts for the full fixed30 family; no data is loaded."""
from .contract import ARMS, TASKS


def forecast(specification):
    """Root supplies actual split sizes and outcome-blind state payload sizes.

    Optimizer bytes must be measured after AdamW state exists, or conservatively
    upper-bounded by two moment tensors per trainable parameter. Values are
    storage bytes, not inferred scientific outcomes. Include exact pickle
    overhead measurements when available; reserve one MiB per tensor artifact.
    """
    if set(specification) != set(TASKS):
        raise ValueError("Both complete task specifications required")
    rows, total = [], 0
    for task in TASKS:
        spec = specification[task]
        for field in ("validation_nodes", "test_nodes", "classes"):
            if type(spec[field]) is not int or spec[field] <= 0:
                raise ValueError("Actual positive complete split sizes required")
        if set(spec["arms"]) != set(ARMS):
            raise ValueError("All five measured or bounded state-payload counts required")
        for arm in ARMS:
            n = 1 if arm == "single_base" else 4
            state = spec["arms"][arm]
            for field in ("canonical_state_bytes", "training_layout_state_bytes", "optimizer_tensor_bytes", "RNG_tensor_bytes"):
                if type(state[field]) is not int or state[field] < 0:
                    raise ValueError("Nonnegative frozen actual/bounded payload bytes required")
            checkpoint = sum(state[field] for field in
                ("canonical_state_bytes", "training_layout_state_bytes", "optimizer_tensor_bytes", "RNG_tensor_bytes")) + 2**20
            # Validation saved both member and exact source raw pooled logits,
            # node IDs; selected/epoch100 replay uses validation slices only.
            valid = (n + 1) * spec["validation_nodes"] * spec["classes"] * 4 + spec["validation_nodes"] * 8 + 2**20
            test = (n + 1) * spec["test_nodes"] * spec["classes"] * 4 + spec["test_nodes"] * 8 + 2**20
            initial = state["canonical_state_bytes"] + 2**20
            # Two validation replay artifacts even when selection==epoch100
            # produces only one. No full-node logits are persisted.
            per_seed = 100 * (checkpoint + valid) + 2 * valid + test + initial + 16 * 2**20
            total += 3 * per_seed
            rows.append({"task": task, "arm": arm, "seeds": 3,
                "all100_checkpoint_bytes_per_seed_bound": 100 * checkpoint,
                "all100_validation_logits_bytes_per_seed_bound": 100 * valid,
                "two_validation_replay_bytes_per_seed_bound": 2 * valid,
                "single_test_logits_bytes_per_seed_bound": test,
                "total_bytes_three_seeds_bound": 3 * per_seed,
                "whole_node_logits_persisted": False})
    return {"schema": "gnnm-propagation-complete30-artifact-byte-plan-v1",
            "fit_count": 30, "epochs": 100, "tasks": specification,
            "rows": rows, "scientific_artifact_bytes_bound": total,
            "metadata_bound_included": True, "source_only_no_artifacts_or_fits_created": True,
            "additional_work_counts": {"stored_metric_replays": 3000,
                "selected_or_epoch100_forward_replay_upper": 60,
                "selected_test_metric_forwards": 30, "cold_pooled_timing_forwards": 30,
                "raw_warmups_and_samples": 1800, "separate_component_warmups_and_samples": 1800,
                "candidate_full_graph_trace_forwards": 6, "extra_resident_deployment_setups": 30},
            "prior_phase_bytes_prepared_operators_and_public_label_bundles_must_be_added": True,
            "resource_or_scientific_admission_established": False}
