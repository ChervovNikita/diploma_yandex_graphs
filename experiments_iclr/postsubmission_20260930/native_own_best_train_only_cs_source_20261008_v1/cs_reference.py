"""Disabled TRAIN-only C&S using exact pinned author diffusion functions.

No dataset loader, heldout-label interface, stock evaluator or hyperparameter
defaults. Role-safe explicit-C initializers replace only full-label helpers.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
AUTHOR_FUNCTIONS = (
    "process_adj", "gen_normalized_adjs", "general_outcome_correlation",
    "double_correlation_autoscale", "double_correlation_fixed",
)
REQUIRED = {
    "train_only", "autoscale", "normalization_correction", "normalization_smoothing",
    "correction_alpha", "smoothing_alpha", "num_correction_layers",
    "num_smoothing_layers", "scale", "graph_policy",
}


def _require(value, message):
    if not value:
        raise ValueError(message)


def correct_and_smooth_train_only(*, probabilities, edge_index, train_ids,
                                  train_labels, classes, configuration,
                                  device="cpu", later_execution_authorized=False):
    """Return author corrected/smoothed scores on all11701nodes.

    No development IDs/truths or TEST fields accepted. Caller must prospectively
    bind settings and any probability-scoring map; no selection is done here.
    """
    if later_execution_authorized is not True:
        raise PermissionError("Disabled C&S reference; prospective root budget required")
    _require(classes == 10 and set(configuration) == REQUIRED,
             "Explicit C10 and complete prospectively declared configuration")
    cfg = dict(configuration)
    _require(cfg["train_only"] is True and type(cfg["autoscale"]) is bool,
             "TRAIN-only anchors required; no default VALID union")
    _require(cfg["graph_policy"] == "author_as_supplied",
             "Explicit caller-bound graph/self-loop policy; no silent private-hop graph")
    for key in ("normalization_correction", "normalization_smoothing"):
        _require(cfg[key] in ("DAD", "DA", "AD"), "Explicit author normalization")
    for key in ("correction_alpha", "smoothing_alpha"):
        _require(type(cfg[key]) in (int, float) and math.isfinite(cfg[key])
                 and 0 <= cfg[key] <= 1, "Explicit finite author propagation alpha")
    for key in ("num_correction_layers", "num_smoothing_layers"):
        _require(type(cfg[key]) is int and cfg[key] > 0, "Explicit positive propagation count")
    _require(type(cfg["scale"]) in (int, float) and math.isfinite(cfg["scale"]),
             "Explicit finite scale (author autoscale ignores this argument)")
    pins = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
    source_path = PHASE / pins["author_code"]["path"]
    source_bytes = source_path.read_bytes()
    source = source_bytes.decode("utf-8")
    _require(hashlib.sha256(source_bytes).hexdigest() == pins["author_code"]["sha256"],
             "Exact cached author code")
    tree = ast.parse(source)
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in AUTHOR_FUNCTIONS]
    _require({node.name for node in nodes} == set(AUTHOR_FUNCTIONS), "Author function inventory")
    # Imports occur only after authorization/config/source checks. The original
    # module's optuna/OGB/Logger/evaluator/top-level code is never imported.
    import torch
    from torch_sparse import SparseTensor
    from torch_geometric.utils import to_undirected
    from tqdm import tqdm
    _require(tuple(probabilities.shape) == (11701, 10)
             and probabilities.dtype == torch.float32
             and bool(torch.isfinite(probabilities).all())
             and bool((probabilities >= 0).all())
             and bool((probabilities.sum(-1) - 1).abs().le(1e-4).all()),
             "Full native class probabilities, not logits or corrected-arm scores")
    _require(edge_index.dtype == torch.long and edge_index.ndim == 2
             and edge_index.shape[0] == 2 and edge_index.shape[1] > 0
             and int(edge_index.min()) >= 0 and int(edge_index.max()) < 11701,
             "Full allowed graph input")
    _require(train_ids.dtype == torch.long and train_labels.dtype == torch.long,
             "Explicit TRAIN ID/label dtypes")
    anchors = train_ids.detach().cpu().reshape(-1).clone()
    labels_A = train_labels.detach().cpu().reshape(-1).clone()
    _require(anchors.numel() == labels_A.numel() == 580 and anchors.unique().numel() == 580
             and int(anchors.min()) >= 0 and int(anchors.max()) < 11701
             and int(labels_A.min()) >= 0 and int(labels_A.max()) < classes,
             "Only the declared complete TRAIN role")
    p0 = probabilities.detach().cpu().clone()
    one_hot_A = torch.nn.functional.one_hot(labels_A, num_classes=classes).to(p0.dtype)

    def allowed_indices(label_idx):
        _require(torch.equal(label_idx.detach().cpu(), anchors),
                 "Author initializer may use only exact TRAIN anchors")

    def pre_residual_correlation(labels, model_out, label_idx):
        # `labels` is an empty compatibility slot, never a full truth array.
        allowed_indices(label_idx)
        value = torch.zeros((11701, classes), dtype=p0.dtype)
        value[anchors] = one_hot_A - model_out.cpu()[anchors]
        return value

    def pre_outcome_correlation(labels, model_out, label_idx):
        allowed_indices(label_idx)
        value = model_out.cpu().clone()
        value[anchors] = one_hot_A
        return value

    namespace = dict(torch=torch, tqdm=tqdm, SparseTensor=SparseTensor,
                     to_undirected=to_undirected,
                     pre_residual_correlation=pre_residual_correlation,
                     pre_outcome_correlation=pre_outcome_correlation)
    selected = ast.Module(body=nodes, type_ignores=[])
    exec(compile(selected, str(source_path), "exec"), namespace)
    data = SimpleNamespace(num_nodes=11701, edge_index=edge_index.detach().cpu().clone(),
                           y=torch.empty(0, dtype=torch.long))
    adjacency, inverse_sqrt = namespace["process_adj"](data)
    maps = dict(zip(("DAD", "DA", "AD"),
                    namespace["gen_normalized_adjs"](adjacency, inverse_sqrt)))
    split = dict(train=anchors, valid=torch.empty(0, dtype=torch.long),
                 test=torch.empty(0, dtype=torch.long))
    name = "double_correlation_autoscale" if cfg["autoscale"] else "double_correlation_fixed"
    with torch.no_grad():
        corrected, smoothed = namespace[name](
            data, p0, split, maps[cfg["normalization_correction"]], cfg["correction_alpha"],
            cfg["num_correction_layers"], maps[cfg["normalization_smoothing"]],
            cfg["smoothing_alpha"], cfg["num_smoothing_layers"],
            scale=cfg["scale"], train_only=True, device=device, display=False)
    _require(tuple(corrected.shape) == tuple(smoothed.shape) == (11701, classes)
             and bool(torch.isfinite(corrected).all()) and bool(torch.isfinite(smoothed).all()),
             "Finite author class-score outputs")
    return dict(corrected_scores=corrected, smoothed_scores=smoothed,
                configuration=cfg, classes=classes, TRAIN_anchors=580,
                source_sha256=pins["author_code"]["sha256"],
                train_only=True, heldout_truths_supplied=False, scoring_performed=False,
                probability_map_applied=False,
                input_edge_records=int(edge_index.shape[1]), graph_policy=cfg["graph_policy"],
                autoscale_ignores_explicit_scale=cfg["autoscale"],
                propagation_passes=cfg["num_correction_layers"]+cfg["num_smoothing_layers"])
