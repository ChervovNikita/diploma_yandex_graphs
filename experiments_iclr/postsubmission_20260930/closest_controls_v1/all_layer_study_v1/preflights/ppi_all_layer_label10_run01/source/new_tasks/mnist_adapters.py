"""MNIST graph adapters around frozen SAGE and repaired common/narrow sources.

All arms return member graph logits[M,G,10]. They first compute node logits
after node LayerNorm, then average logits within each graph. Ensemble pooling
averages member raw logits. No data access or training loop is included.
"""
import importlib.util
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from common_readouts import ALL_LAYER_ARM


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASELINE = _load("mnist_frozen_baseline_adapters", ROOT / "protocols" / "baseline_adapters.py")
PATHS = _load("mnist_repaired_centered_paths", ROOT / "efficient_paths" / "centered_paths.py")
PORT = _load("amended_mnist_all_layer_port", ROOT / "closest_port" / "all_layer_sage_adapter.py")
ARMS = (ALL_LAYER_ARM, "centered_feedback", "centered_no_feedback", "uncentered_feedback", "uncentered_no_feedback",
        "base", "gnnm", "untied", "first_private", "middle_private", "last_private",
        "multihead", "ens", "packed_gnnm", "packed_full", "packed_channels",
        "packed_parameters", "base_parameters")


def graph_options(hidden_dim=128, rank=16, layers=3, members=4, dropout=0.2):
    return {"input_dim": 3, "output_dim": 10, "hidden_dim": hidden_dim,
            "rank": rank, "layers": layers, "members": members, "dropout": dropout}


class GraphClassifier(nn.Module):
    """Consistent mean node-logit graph readout for every execution path."""
    def __init__(self, node_model, execution, members, metadata):
        super().__init__()
        self.node_model, self.execution, self.members = node_model, execution, members
        self.metadata = metadata
        self.last_operator_widths = None
        self.last_operator_build_seconds = None

    def node_logits(self, graph, features=None):
        import time
        features = graph.x if features is None else features
        if features.ndim != 2 or features.shape != (graph.num_nodes, 3) or not features.is_floating_point():
            raise ValueError("MNIST requires float intensity+released coordinates[N,3]")
        if self.execution == "source":
            self.last_operator_widths = None
            self.last_operator_build_seconds = None
            return self.node_model(graph, features)
        started = time.perf_counter()
        operator = PATHS.MeanOperator.from_edges(graph.edge_index, graph.num_nodes,
                                               dtype=features.dtype, device=features.device)
        self.last_operator_build_seconds = time.perf_counter() - started
        logits = self.node_model(operator, features)
        self.last_operator_widths = list(operator.widths)
        return logits

    def forward(self, graph, features=None):
        logits = self.node_logits(graph, features)
        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=logits.device)
            graphs = 1
        else:
            graphs = int(graph.num_graphs)
        if graph.num_nodes == 0 or bool((torch.bincount(batch, minlength=graphs) == 0).any()):
            raise ValueError("Mean node-logit readout requires nonempty graphs")
        return PATHS.graph_mean_logits(logits, batch, graphs)

    def pooled_forward(self, graph, features=None):
        return self(graph, features).mean(0)


def _target_parameters(options):
    # Preserve the caller's initialization RNG; shape matching uses no labels.
    with torch.random.fork_rng(devices=[]):
        target = PATHS.CenteredPathEnsemble(
            options["input_dim"], options["output_dim"], dim=options["hidden_dim"],
            rank=options["rank"], members=options["members"], layers=options["layers"],
            dropout=options["dropout"], recycle_mean=True)
        return sum(p.numel() for p in target.parameters())


def make_graph_model(arm, options=None):
    if arm not in ARMS:
        raise ValueError("Unknown MNIST arm")
    o = graph_options() if options is None else dict(options)
    d, r, m, layers = o["hidden_dim"], o["rank"], o["members"], o["layers"]
    if arm == ALL_LAYER_ARM:
        model, definition = PORT.make_mnist_model(hidden_dim=d, layers=layers, members=m, dropout=o["dropout"])
        definition.update({"node_features": "released intensity+coordinates3, no rescaling",
                           "edges": "released directed entries unchanged, no added self-loops; edge_attr retained but unused",
                           "adjacency_channels_per_layer": m * d,
                           "adjacency_applications_per_layer": m, "execution": "source"})
        return model, definition
    source_options = BASELINE.model_options(o["input_dim"], o["output_dim"],
                                          hidden_dim=d, layers=layers, dropout=o["dropout"])
    kw = {"input_dim": o["input_dim"], "output_dim": o["output_dim"], "dim": d,
          "members": m, "layers": layers, "dropout": o["dropout"]}
    definition = {"arm": arm, "readout": "mean_node_logits_after_node_LayerNorm",
                  "pooling": "mean_member_raw_logits", "SAGE_MLP_multiplier": 1,
                  "hidden_dim": d, "rank": r, "members": m, "layers": layers,
                  "node_features": "released intensity+coordinates3, no rescaling",
                  "edges": "released directed entries unchanged, no added self-loops; edge_attr retained but unused"}
    execution = "source"
    if arm.startswith("centered_") or arm.startswith("uncentered_"):
        centered = arm.startswith("centered_")
        feedback = arm.endswith("_feedback") and not arm.endswith("_no_feedback")
        cls = PATHS.CenteredPathEnsemble if centered else PATHS.IndependentNarrowPaths
        model = cls(**kw, rank=r, recycle_mean=feedback)
        execution = "sparse"
        definition.update({"centered": centered, "mean_feedback": feedback,
                           "feedback_parameter_present": model.expansion is not None,
                           "adjacency_channels_per_layer": d + (m - 1 if centered else m) * r,
                           "adjacency_applications_per_layer": 1,
                           "initialization": "same shared/narrow parameter draws across four cells; disabled expansion absent"})
    elif arm == "base":
        model, m = BASELINE.SingleModel(source_options), 1
    elif arm == "ens":
        model = BASELINE.IndependentEnsemble(source_options, members=m)
        definition["initialization"] = "independent complete frozen-source models"
    elif arm == "multihead":
        model = BASELINE.SharedTrunkMultihead(source_options, members=m)
        definition["propagation_paths"] = 1
    elif arm in ("gnnm", "untied", "first_private", "middle_private", "last_private"):
        original = BASELINE.ProjectorEnsemble(source_options, members=m)
        indices = {"gnnm": [], "untied": list(range(layers)), "first_private": [0],
                   "middle_private": [layers // 2], "last_private": [layers - 1]}[arm]
        model = original if not indices else BASELINE.SelectiveProjectorEnsemble(original, indices)
        definition.update({"private_blocks": indices,
                           "initialization": "same frozen-source GNNM initialization; selected complete blocks copied",
                           "shared_boundary_projectors": True})
    elif arm == "packed_gnnm":
        original = BASELINE.ProjectorEnsemble(source_options, members=m)
        model = PATHS.PackedSharedSAGEEnsemble(**kw)
        model.copy_from_source_tabm(original.model)
        execution = "sparse"
        definition["initialization"] = "exact frozen-source GNNM parameters copied; eval equivalence, not identical dropout trajectory"
    elif arm == "packed_full":
        original = BASELINE.IndependentEnsemble(source_options, members=m)
        model = PATHS.PackedIndependentSAGEEnsemble(**kw)
        model.copy_from_source_members(original.models)
        execution = "sparse"
        definition["initialization"] = "exact independent frozen-source ENS parameters copied; eval equivalence, not identical dropout trajectory"
    else:
        target = _target_parameters(o)
        if arm == "packed_channels":
            channels = d + (m - 1) * r
            if channels % m:
                raise ValueError("The fixed centered channel target must divide by members")
            width = channels // m
            definition.update({"matching_target": "centered_feedback adjacency channels", "target_channels": channels})
        else:
            matched_members = 1 if arm == "base_parameters" else m
            width = PATHS.nearest_parameter_width(target, o["input_dim"], o["output_dim"],
                                                  layers=layers, members=matched_members)
            definition.update({"matching_target": "centered_feedback registered parameters",
                               "target_parameters": target,
                               "width_rule": "closest exact source count over integer widths8..512; ties smaller; no labels or scores"})
        if arm == "base_parameters":
            model, m = PATHS.PlainSAGE(o["input_dim"], o["output_dim"], dim=width,
                                      layers=layers, dropout=o["dropout"]), 1
        else:
            model = PATHS.PackedIndependentSAGEEnsemble(**{**kw, "dim": width})
        execution = "sparse"
        definition["hidden_dim"] = width
        actual = sum(p.numel() for p in model.parameters())
        if arm != "packed_channels":
            definition.update({"parameter_difference": actual - target,
                               "relative_parameter_difference": (actual - target) / target})
    definition["members"] = m
    definition["execution"] = execution
    width = definition["hidden_dim"]
    if "adjacency_channels_per_layer" not in definition:
        paths = 1 if arm in ("base", "base_parameters", "multihead") else m
        definition["adjacency_channels_per_layer"] = paths * width
        definition["adjacency_applications_per_layer"] = paths if execution == "source" else 1
    definition["parameters"] = sum(p.numel() for p in model.parameters())
    return GraphClassifier(model, execution, m, definition), definition


def arm_plan(arms=ARMS, options=None):
    with torch.random.fork_rng(devices=[]):
        result = {}
        for arm in arms:
            torch.manual_seed(0)
            model, definition = make_graph_model(arm, options)
            result[arm] = definition
            del model
        return result
