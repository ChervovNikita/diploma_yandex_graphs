"""Frozen TRAIN-only diagnostic recipe. Not generated in this preparation.

Future root must seal record IDs, negative rows and support/teacher digests
BEFORE model values. The diagnostic adds no fit or selection criterion.
"""
import heapq
import random
import numpy as np
import torch
from cardinality_counter import diagnostic_record_key, separated_seed
from cardinality_density import require
from cardinality_graph import Graph
from cardinality_teacher import tensor_sha, sealed_support_digest
from cardinality_model import ordered_support


def diagnostic_record_ids(record_count=1179052):
    require(record_count == 1179052, "Complete ascending TRAIN record population required")
    # Tuple order implements byte-hash ordering then ascending ID tie-break.
    chosen = heapq.nsmallest(65536, ((diagnostic_record_key(i), i) for i in range(record_count)))
    return sorted(i for _, i in chosen)


def diagnostic_native_negatives(data, sampler):
    """Pinned native law with separate seed; restore native state on failure."""
    py_state, np_state = random.getstate(), np.random.get_state()
    cpu_state, cuda_states = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    try:
        seed = separated_seed("diagnostic-negatives")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        negatives = sampler(data["raw_edge_index"], len(data["x"]))
        require(negatives.dtype == torch.long and negatives.ndim == 2 and negatives.shape[0] == 2 and negatives.shape[1] >= 65536,
                "Complete diagnostic native negative draw required")
        return negatives.T[:65536].clone()
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(cuda_states)


def prepare_diagnostic_support(data, sampler, teacher):
    require(data["pairs"].shape == (1179052, 2) and data["x"].shape == (235868, 128), "Complete representative TRAIN snapshot required")
    require(teacher.scope == "complete_authenticated_TRAIN" and teacher.train_records_sha256 == tensor_sha(data["pairs"]), "TRAIN-only diagnostic teacher required")
    ids = torch.tensor(diagnostic_record_ids(), dtype=torch.long, device=data["pairs"].device)
    negatives = diagnostic_native_negatives(data, sampler)
    graph = Graph.mask_train_batch(data["pairs"], ids, len(data["x"]))
    queries = (data["pairs"][ids], negatives)
    supports = [ordered_support(graph, current) for current in queries]
    labels = [teacher.labels(support.counterpart_pairs) for support in supports]
    receipt = {"TRAIN_records_sha256": tensor_sha(data["pairs"]), "selected_record_ids_sha256": tensor_sha(ids),
               "negative_query_rows_sha256": tensor_sha(negatives), "records_removed_before_coalescing": 65536,
               "support_without_labels_sha256": [sealed_support_digest(current, support) for current, support in zip(queries, supports)],
               "support_with_labels_sha256": [sealed_support_digest(current, support, z) for current, support, z in zip(queries, supports, labels)],
               "model_values_read": False, "fit_or_selection_work": False,
               "source_zero_counts": [int((z == 0).sum()) for z in labels], "native_scientific_rng_restored": True}
    # This return must be committed to root-owned custody BEFORE any model
    # diagnostic forward. Model input is graph/queries, never labels/support
    # teacher features. Observation labels remain outside its forward API.
    return graph, queries, supports, labels, receipt
