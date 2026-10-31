"""Payload arithmetic only. Metadata edge counts are illustrative scenarios."""

from __future__ import annotations

import json

from .contract import ROOT, SCHEMA, confined_output


def csr_payload(nodes, edges, cached_halves):
    entries = edges * (8 if cached_halves else 4)
    csr_count = 18 if cached_halves else 10
    return entries * 12 + csr_count * (nodes + 1) * 8


def row(name, n, f, edge_scenario, scenario_description):
    def size(count):
        return {"bytes": count, "GiB": count / (2**30)}
    d, m = 64, 4
    return {
        "task": name, "nodes_from_scout_metadata": n,
        "canonical_edges_actual": None,
        "canonical_edges_illustrative_scenario": edge_scenario,
        "scenario_description": scenario_description,
        "one_full_member_state_float32": size(n * m * d * 4),
        "one_source_FFN_concatenation_float32": size(n * m * 2 * d * 4),
        "one_serial_exact_member_state_float32": size(n * d * 4),
        "complete_feature_matrix_float32": size(n * f * 4),
        "full_all_colors_and_adjoints_CSR_payload": size(csr_payload(n, edge_scenario, False)),
        "plus_all_cached_halves_and_adjoints_CSR_payload": size(csr_payload(n, edge_scenario, True)),
        "preparation_metadata_payload": size(n * 9 + edge_scenario * 9),
        "excluded_from_CSR_payload": ["metadata", ".npy headers", "allocator bookkeeping",
                                      "sorting workspace", "features", "states", "autograd activations"],
        "peak_reserved_memory_measured": False,
        "time_measured": False,
    }


def calculate():
    return {
        "schema": SCHEMA, "kind": "symbolic_payload_arithmetic",
        "not_a_feasibility_or_latency_measurement": True,
        "dtype_policy": {"states": "float32", "CSR_values": "float32",
                         "CSR_columns_and_rowptr": "int64"},
        "scenarios": [
            row("Reddit2", 232965, 602, 23213838,
                "Use reported raw edge entries as an upper-bound scenario for coalesced CSR entries; provider has not been opened."),
            row("ogbn-products", 2449029, 100, 61859140,
                "Reported metadata count used illustratively; actual returned directed/coalesced E is unknown."),
            row("ogbn-products", 2449029, 100, 123718280,
                "Twice the metadata count is another illustrative scenario, not a claim about provider conventions."),
        ],
        "expected_sparse_edge_channels_per_layer": {
            "candidate": "E*D + mean_c(E_c)*4*D = 2*E*D",
            "exact_member": "4*E*D",
            "single_base": "E*D",
            "naive_shared_sample": "mean_c(E_c+E_(c+1))*4*D = 2*E*D",
        },
        "actual_color_entry_counts_required": True,
        "family_size": {"datasets": 2, "arms": 4, "seeds": 3, "fits": 24,
                        "full_epochs_per_fit": 100, "full_validations_per_fit": 100},
        "forecast_ceiling_seconds": 6.5 * 3600,
        "mean_training_plus_validation_seconds_per_epoch_allotment_upper_bound": 6.5 * 3600 / 2400,
        "allotment_limit": "Final evaluation and profiling reduce this illustrative per-epoch allotment; it is not an ETA.",
    }


if __name__ == "__main__":
    path = confined_output(ROOT / "SYMBOLIC_RESOURCES_local_run01.json")
    if path.exists():
        raise FileExistsError("Preserve earlier arithmetic record")
    path.write_text(json.dumps(calculate(), indent=2) + "\n")
    print(str(path))
