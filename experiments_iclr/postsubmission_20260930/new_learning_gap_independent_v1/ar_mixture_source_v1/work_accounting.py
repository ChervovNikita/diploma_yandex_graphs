"""Unexecuted operation counters; no timing, memory measurement or forecast."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EncoderAudit:
    source_recipe_id: str
    source_sha256: str
    deterministic_module_audit_id: str
    normalization: str
    node_embedding_dimension: int
    message_passes_per_encoding: int
    edge_visits_per_encoding: int
    dense_multiply_terms_per_encoding: int

    def __post_init__(self):
        if not self.source_recipe_id or not self.source_sha256 or not self.deterministic_module_audit_id:
            raise ValueError("Authorized-source and deterministic-module provenance required")
        if self.normalization not in ("LayerNorm", "None"):
            raise ValueError("Only audited node-wise LayerNorm or Identity is admitted")
        counts = (self.node_embedding_dimension, self.message_passes_per_encoding,
                  self.edge_visits_per_encoding, self.dense_multiply_terms_per_encoding)
        if any(type(value) is not int or value < 0 for value in counts) or self.node_embedding_dimension <= 0:
            raise ValueError("Explicit audited encoder dimensions/work counts required")


@dataclass
class WorkLedger:
    audit: EncoderAudit
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, key, amount):
        if type(amount) is not int or amount < 0:
            raise ValueError("Nonnegative integral work count required")
        self.counts[key] = self.counts.get(key, 0) + amount

    def record_encoding(self, n_nodes):
        self.add("static_graph_encodings", 1)
        self.add("encoder_message_passes", self.audit.message_passes_per_encoding)
        self.add("encoder_edge_visits", self.audit.edge_visits_per_encoding)
        self.add("encoder_dense_multiply_terms", self.audit.dense_multiply_terms_per_encoding)
        self.add("encoded_node_values", n_nodes * self.audit.node_embedding_dimension)

    def record_decoder(self, *, kind, n_nodes, n_edges, embedding_dimension,
                       hidden_dimension, factored, retain_score_gradient):
        p, h = embedding_dimension + 2, hidden_dimension
        self.add(kind + "_complete_decoder_trajectories", 1)
        self.add(kind + "_decoder_steps", n_nodes)
        self.add("decoder_dense_multiply_terms", n_nodes * (3 * h * p + 3 * h * h + 2 * h))
        if factored:
            self.add("decoder_diagonal_multiply_elements", n_nodes * (p + 8 * h + 2))
        self.add("decoder_gate_bias_additions", n_nodes * (6 * h + 2))
        self.add("decoder_nonlinearity_elements", n_nodes * 3 * h)
        # any() can short-circuit; the complete adjacency scan is an upper bound.
        self.add("hard_core_neighbor_inspection_bound", 2 * n_edges)
        if retain_score_gradient:
            self.add("generating_score_backward_trajectories", 1)
        if kind == "generation":
            self.add("caller_uniform_values_consumed", n_nodes)

    def record_cross_matrix(self, n_samples, n_branches):
        self.add("cross_member_density_entries", n_samples * n_branches)

    def record_mixture_reduction(self, n_samples, n_branches):
        self.add("mixture_logsumexp_entries", n_samples * n_branches)
