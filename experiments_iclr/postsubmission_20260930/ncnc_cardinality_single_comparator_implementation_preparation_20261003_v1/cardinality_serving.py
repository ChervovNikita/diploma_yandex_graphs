"""Complete D4 VALID utilities and selection-closed fixed-bank M4 control.

SOURCE ONLY. No current checkpoint, score, selection, evaluator invocation,
training/fit loop or TEST accessor. The future root owns state-file custody.
"""
from dataclasses import dataclass
from time import perf_counter
import torch
from cardinality_counter import CompletionKeys
from cardinality_graph import Graph
from cardinality_density import require
from cardinality_teacher import tensor_sha


@dataclass(frozen=True)
class SelectedD4Bank:
    checkpoint_sha256: str
    epoch: int
    completed_selection_states: int
    selection_rule: str = "first_strict_best_complete_VALID_C64-D4"

    def __post_init__(self):
        require(len(self.checkpoint_sha256) == 64 and all(c in "0123456789abcdef" for c in self.checkpoint_sha256), "Selected checkpoint hash required")
        require(1 <= self.epoch <= 100 and self.completed_selection_states == 100 and self.selection_rule == "first_strict_best_complete_VALID_C64-D4",
                "M4 requires closed frozen D4 selection")


class FirstStrictBest:
    """Tiny prospective policy; future root writes/loads only its own states."""
    def __init__(self):
        self.states = 0
        self.best_value = None
        self.best_epoch = None

    def consider(self, epoch, complete_d4_hits50):
        require(epoch == self.states + 1 and epoch <= 100, "D4 selection must visit100 ordered states")
        require(type(complete_d4_hits50) is float and 0. <= complete_d4_hits50 <= 1., "Complete strict official VALID metric required")
        self.states += 1
        improved = self.best_value is None or complete_d4_hits50 > self.best_value
        if improved:
            self.best_value, self.best_epoch = complete_d4_hits50, epoch
        return improved  # exact ties retain the first bank


def strict_official_hits50(metric, positive, negative):
    require(positive.shape == (60084,) and negative.shape == (100000,), "Complete official VALID pools required")
    require(positive.dtype == negative.dtype == torch.float32 and bool(torch.isfinite(positive).all()) and bool(torch.isfinite(negative).all()),
            "Finite mean raw FP32 logits required")
    require(metric.name == "ogbl-collab" and metric.eval_metric == "hits@50" and metric.K == 50, "Pinned official collab evaluator required")
    actual = float(metric.eval({"y_pred_pos": positive, "y_pred_neg": negative})["hits@50"])
    explicit = float((positive > torch.topk(negative, 50).values[-1]).sum()) / len(positive)
    require(actual == explicit, "Official shared-pool strict Hits50 arithmetic differs")
    return actual


@torch.no_grad()
def score_complete_valid(model, data, native_utils, *, route="D4", selected_bank=None, loaded_checkpoint_sha256=None):
    require(route in ("D4", "M4"), "Unknown declared serving route")
    if route == "M4":
        require(type(selected_bank) is SelectedD4Bank and loaded_checkpoint_sha256 == selected_bank.checkpoint_sha256,
                "M4 can score only the exactly loaded D4-selected bank")
    else:
        require(selected_bank is None, "D4 selection does not take an M4 bank")
    require(data["pairs"].shape == (1179052, 2) and data["x"].shape == (235868, 128), "Complete TRAIN graph/features required")
    require(data["valid_positive"].shape == (60084, 2) and data["valid_negative"].shape == (100000, 2), "Complete official VALID queries required")
    model.eval()
    torch.cuda.synchronize(0)
    started = perf_counter()
    graph = Graph.from_pairs(data["pairs"], len(data["x"]))
    h = model.encode(data["x"], graph)
    outputs, work = [], []
    for queries in (data["valid_positive"], data["valid_negative"]):
        # Native evaluation iterator keeps every tail, fixed131072 batch.
        scores = torch.empty(len(queries), dtype=torch.float32, device="cpu")
        visited = torch.zeros(len(queries), dtype=torch.bool, device="cpu")
        for indices in native_utils.PermIterator(queries.device, len(queries), 131072, False):
            logits, detail, draws = model.query_forward(h, graph, queries[indices], CompletionKeys("eval"), route=route)
            host_ids = indices.detach().cpu()
            require(not bool(visited[host_ids].any()), "Repeated VALID query row")
            scores[host_ids] = model.serve(logits).detach().cpu()
            visited[host_ids] = True
            work.append({**draws["work"], "native_full_node_xlin_calls": 3, "native_depth_zero_calls": 2,
                         "downstream_decodes": 4, "route": route})
        require(bool(visited.all()), "VALID query/tail coverage incomplete")
        outputs.append(scores)
    torch.cuda.synchronize(0)
    receipt = {"route": "C64-" + route, "positive_queries": 60084, "negative_queries": 100000,
               "encoder_calls": 1, "serving": "mean_four_raw_logits", "graph": "complete_TRAIN_only", "all_rows_complete": True,
               "score_digests": [tensor_sha(value) for value in outputs], "query_work": work,
               "wall_seconds": perf_counter() - started, "M4_extra_fits": 0,
               "selected_bank": selected_bank.__dict__ if selected_bank is not None else None}
    return outputs[0], outputs[1], receipt
