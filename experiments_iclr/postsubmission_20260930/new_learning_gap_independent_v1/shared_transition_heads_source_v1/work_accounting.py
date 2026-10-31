"""Unexecuted shared-transition counters; retained graphs are not backwards."""
from dataclasses import dataclass

from ..ar_mixture_source_v1.work_accounting import WorkLedger


@dataclass
class SharedTransitionLedger(WorkLedger):
    """Reuse audited encoder/matrix/reduction counters; add exact route counts.

    These are authored symbolic work counts, not timings or memory measurements.
    The inherited private-decoder record_decoder method is not used here.
    """

    def record_shared_trajectory(self, *, kind, n_nodes, n_edges,
                                 embedding_dimension, hidden_dimension,
                                 heads_per_step, factored_heads,
                                 retain_generating_score_graph):
        p, h = embedding_dimension + 2, hidden_dimension
        self.add(kind + "_shared_recurrent_trajectories", 1)
        self.add(kind + "_shared_recurrent_steps", n_nodes)
        self.add(kind + "_head_step_evaluations", n_nodes * heads_per_step)
        self.add("recurrent_dense_multiply_terms", n_nodes * (3 * h * p + 3 * h * h))
        self.add("head_dense_multiply_terms", n_nodes * heads_per_step * 2 * h)
        self.add("shared_recurrent_bias_additions", n_nodes * 6 * h)
        self.add("private_head_bias_additions", n_nodes * heads_per_step * 2)
        self.add("shared_recurrent_nonlinearity_elements", n_nodes * 3 * h)
        if factored_heads:
            self.add("private_head_diagonal_multiply_elements", n_nodes * heads_per_step * (h + 2))
        self.add("hard_core_neighbor_inspection_bound", 2 * n_edges)
        if retain_generating_score_graph:
            self.add("retained_generating_score_graphs", 1)
        if kind == "generation":
            self.add("caller_uniform_values_consumed", n_nodes)
