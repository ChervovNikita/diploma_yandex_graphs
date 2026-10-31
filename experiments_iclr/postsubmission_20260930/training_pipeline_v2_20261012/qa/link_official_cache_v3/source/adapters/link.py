"""NCN/NCNC full-epoch bridge using pinned author classes and native sampling.

Stdlib-only import. The fit factory opens plain-tensor GRAPH/TRAIN/SELECT packs,
never an OGB dataset/split accessor or CAL/DEV/TEST authority. Each I4 body is a
separate paid engine trajectory with its own optimizer/RNG/selected checkpoint.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import math
from pathlib import Path
import random
import re
import sys

try:
    from ..engine import AdapterStepError
    from ..metrics import EpochMeans
except ImportError:
    from engine import AdapterStepError
    from metrics import EpochMeans

AUTHOR_REPOSITORY = "GraphPKU/NeuralCommonNeighbor"
AUTHOR_COMMIT = "11d597013750da17ce7468e344bec756a7af39a4"
AUTHOR_SOURCE = {
    "model": {"bytes": 39783, "sha256": "5b6cfe26074c83d57900f589b37a4ef57fc7b8eb6dff3179fc52e3f6943aa5e3"},
    "utils": {"bytes": 9208, "sha256": "29df3b9cf82ae5885ece0fac010b81383f1fb5391068649c88b7d2bc98ad86be"},
}
ARMS = {"native_single", "native_independent4_body"}
METADATA_FIELDS = {"stage", "reservation_mib", "estimated_seconds", "selected_config_from", "resolved_selection"}


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid integer " + name)


def _number(value, name, minimum=0., maximum=None, strict=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= minimum if strict else value < minimum)
            or (maximum is not None and value >= maximum)):
        raise ValueError("Invalid finite number " + name)


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("An exact SHA256 binding is required")


def _receipt(value):
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError("Bind every link input/source with path/bytes/SHA256")
    if not isinstance(value["path"], str) or not value["path"]:
        raise ValueError("Link paths must be project-relative")
    path = Path(value["path"])
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Link paths must be project-relative")
    _integer(value["bytes"], "receipt bytes", 0)
    _hash(value["sha256"])


def _keys(value, expected, message):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(message)


def validate_config(config, run):
    required = {"link_inputs", "link_source", "arm", "encoder", "predictor", "optimizer",
                "training", "support", "sampler", "metric", "initialization_seed"}
    optional = {"independence_group", "maximum_cuda_bytes"} | METADATA_FIELDS
    if not isinstance(config, dict) or not required <= set(config) or set(config)-required-optional:
        raise ValueError("Incomplete/unsupported link config; heldout fit inputs are forbidden")
    _keys(config["link_inputs"], {"graph", "train", "select"}, "Link fit opens exactly GRAPH, TRAIN and SELECT authorities")
    for item in config["link_inputs"].values():
        _receipt(item)
    if len({r["path"] for r in config["link_inputs"].values()}) != 3:
        raise ValueError("Link graph/TRAIN/SELECT must be separate files")
    _keys(config["link_source"], {"model", "utils"}, "Bind exact author model and common-neighbor utility source")
    for name, item in config["link_source"].items():
        _receipt(item)
        if any(item[key] != AUTHOR_SOURCE[name][key] for key in ("bytes", "sha256")):
            raise ValueError("Link model/utils must equal the pinned author commit source")
    if not isinstance(config["arm"], str) or config["arm"] not in ARMS:
        raise ValueError("This packet implements native NCN/NCNC single and own I4 bodies")
    if config["arm"] == "native_independent4_body":
        if type(run["body"]) is not int or not 0 <= run["body"] < 4 or not isinstance(config.get("independence_group"), str) or not config["independence_group"]:
            raise ValueError("I4 needs a named group and separate body IDs0..3")
    elif "independence_group" in config:
        raise ValueError("An I4 group belongs only to separate I4 body trajectories")
    seed = config["initialization_seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Native Torch/NumPy/Python initialization needs an explicit uint32 seed")
    encoder = config["encoder"]
    _keys(encoder, {"hidden", "layers", "dropout", "ln", "res", "edrop", "xdropout", "taildropout", "jk", "conv_fn"}, "Declare the complete native GCN constructor recipe")
    for key in ("hidden", "layers"):
        _integer(encoder[key], key)
    for key in ("dropout", "edrop", "xdropout", "taildropout"):
        _number(encoder[key], key, maximum=1.)
    if any(type(encoder[k]) is not bool for k in ("ln", "res", "jk")) or encoder["conv_fn"] != "gcn":
        raise ValueError("This source packet binds the native uncached GCN operator")
    predictor = config["predictor"]
    _keys(predictor, {"name", "layers", "dropout", "edrop", "ln", "cndeg", "use_xlin", "tailact", "twolayerlin", "beta", "completion"}, "Declare the complete native NCN/NCNC predictor recipe")
    if not isinstance(predictor["name"], str) or predictor["name"] not in {"NCN", "NCNC"}:
        raise ValueError("Use pinned CNLinkPredictor or IncompleteCN1Predictor")
    _integer(predictor["layers"], "predictor layers")
    for key in ("dropout", "edrop"):
        _number(predictor[key], key, maximum=1.)
    _number(predictor["beta"], "predictor beta")
    if any(type(predictor[k]) is not bool for k in ("ln", "use_xlin", "tailact", "twolayerlin")) or predictor["cndeg"] != -1:
        raise ValueError("Bind unsampled native common neighbors and explicit normalization flags")
    completion = predictor["completion"]
    if predictor["name"] == "NCN":
        if completion is not None:
            raise ValueError("NCN must not inherit NCNC completion settings")
    else:
        if not isinstance(completion, dict) or set(completion) != {"alpha", "scale", "offset", "trainresdeg", "testresdeg", "pt", "learnablept", "depth", "splitsize"}:
            raise ValueError("Declare every NCNC clamp/recursion setting")
        for key in ("alpha", "scale", "pt"):
            _number(completion[key], key, strict=True)
        _number(completion["offset"], "completion offset", minimum=-1e300)
        if completion["pt"] >= 1 or completion["trainresdeg"] != -1 or completion["testresdeg"] != -1 or type(completion["learnablept"]) is not bool:
            raise ValueError("Bind native unsampled NCNC residual neighbors and a valid prior threshold")
        _integer(completion["depth"], "completion depth")
        if type(completion["splitsize"]) is not int or completion["splitsize"] == 0 or completion["splitsize"] < -1:
            raise ValueError("NCNC recursion splitsize is -1 or a positive integer")
    optimizer = config["optimizer"]
    _keys(optimizer, {"name", "encoder_lr", "predictor_lr", "weight_decay", "eps", "betas", "amsgrad"}, "Declare the native two-group Adam recipe")
    if optimizer["name"] != "Adam":
        raise ValueError("Declare the native two-group Adam recipe")
    for key in ("encoder_lr", "predictor_lr", "eps"):
        _number(optimizer[key], key, strict=True)
    _number(optimizer["weight_decay"], "weight decay")
    if type(optimizer["amsgrad"]) is not bool or not isinstance(optimizer["betas"], (list, tuple)) or len(optimizer["betas"]) != 2:
        raise ValueError("Declare Adam betas and amsgrad")
    for beta in optimizer["betas"]:
        _number(beta, "Adam beta", maximum=1.)
    training = config["training"]
    _keys(training, {"batch_size", "eval_batch_size", "tail_policy"}, "Declare native floor batching or an explicit complete-tail adaptation")
    if not isinstance(training["tail_policy"], str) or training["tail_policy"] not in {"native_drop_tail", "complete_tail"}:
        raise ValueError("Declare native floor batching or an explicit complete-tail adaptation")
    for key in ("batch_size", "eval_batch_size"):
        _integer(training[key], key)
    support = config["support"]
    _keys(support, {"graph", "maskinput", "record_duplicate_policy", "select_overlap", "select_negative_self_policy"}, "TRAIN-only native record masking and heldout-edge guards are mandatory")
    if support["graph"] != "train_only" or support["maskinput"] is not True or support["select_overlap"] != "reject" or not isinstance(support["record_duplicate_policy"], str) or support["record_duplicate_policy"] not in {"require_unique_pairs", "native_record_multiset"}:
        raise ValueError("TRAIN-only native record masking and heldout-edge guards are mandatory")
    if support["select_negative_self_policy"] not in {"reject", "preserve_official_records"}:
        raise ValueError("Declare whether official SELECT negative self records are preserved")
    sampler = config["sampler"]
    _keys(sampler, {"name", "source_sha256", "function_sha256"}, "Bind the actual native-default PyG negative sampler")
    if sampler["name"] != "pyg_native_defaults":
        raise ValueError("Bind the actual native-default PyG negative sampler")
    _hash(sampler["source_sha256"])
    _hash(sampler["function_sha256"])
    metric = config["metric"]
    _keys(metric, {"hits_k", "evaluator_source_sha256"}, "Bind the shared-pool OGB Hits evaluator")
    _integer(metric["hits_k"], "Hits K")
    _hash(metric["evaluator_source_sha256"])
    if run["budget"]["unit"] != "epochs" or run["budget"]["policy"] != "full_budget":
        raise ValueError("Link fitting requires a full declared epoch budget")
    _integer(run["budget"]["maximum"], "epoch budget")
    selector = run["selection"]
    if (selector["metric"] not in {"hits@"+str(metric["hits_k"]), "ce"}
            or selector["direction"] != ("min" if selector["metric"] == "ce" else "max")
            or selector["ties"] not in {"first", "last"} or selector.get("save_every", 1) != 1):
        raise ValueError("Declare a native Hits or balanced-BCE selector")
    return copy.deepcopy(config)


def validate_roster(study, run):
    config = run["config"]
    if study["data"]["task"] != "link_prediction":
        raise ValueError("Link packet requires task=link_prediction")
    for role in ("train", "select"):
        if study["data"]["roles"][role] != config["link_inputs"][role]:
            raise ValueError("Link input differs from prospective " + role + " authority")
    if config["arm"] != "native_independent4_body":
        return
    rows = [r for r in study["runs"] if r["kind"] == "model_fit" and r["config"].get("independence_group") == config["independence_group"]]
    if len(rows) != 4 or {r["body"] for r in rows} != {0, 1, 2, 3}:
        raise ValueError("Genuine link I4 requires four separate paid body trajectories")
    if len({r["run_id"] for r in rows}) != 4:
        raise ValueError("Genuine link I4 requires four distinct paid run IDs")
    if len({r["config"].get("initialization_seed") for r in rows}) != 4:
        raise ValueError("Genuine link I4 requires four distinct native fit seeds")
    recipe = {k: v for k, v in config.items() if k not in {"initialization_seed"} | METADATA_FIELDS}
    for row in rows:
        validate_config(row["config"], row)
        other = {k: v for k, v in row["config"].items() if k not in {"initialization_seed"} | METADATA_FIELDS}
        if other != recipe or any(row.get(k) != run.get(k) for k in ("adapter", "method", "split", "budget", "selection")):
            raise ValueError("I4 body source/role/architecture/optimizer/budget/selector recipes differ")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bound_file(project_root, item):
    _receipt(item)
    root = Path(project_root).resolve()
    path = root / item["path"]
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("Missing, unconfined or symlinked link file")
    if path.stat().st_size != item["bytes"] or file_sha(path) != item["sha256"]:
        raise ValueError("Link source/input receipt changed")
    return path


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def native_modules(root, receipts):
    paths = {k: bound_file(root, r) for k, r in receipts.items()}
    utils = _module(paths["utils"], "gnnm_link_utils_"+receipts["utils"]["sha256"][:16])
    previous = sys.modules.get("utils")
    try:
        sys.modules["utils"] = utils
        native = _module(paths["model"], "gnnm_link_native_"+receipts["model"]["sha256"][:16])
    finally:
        if previous is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous
    return native, utils


def pair_keys(torch, pairs, nodes, *, undirected=True):
    if undirected:
        lo, hi = pairs.min(1).values, pairs.max(1).values
        return lo*nodes+hi
    return pairs[:, 0]*nodes+pairs[:, 1]


def intersects(torch, left, right):
    if not left.numel() or not right.numel():
        return False
    ordered = right.sort().values
    indices = torch.searchsorted(ordered, left)
    return bool(((indices < len(ordered)) & (ordered[indices.clamp(max=len(ordered)-1)] == left)).any())


def validate_inputs(torch, graph, train, select, config):
    for pack, fields in ((graph, {"x", "raw_edge_index"}), (train, {"positive"}), (select, {"positive", "negative"})):
        _keys(pack, fields, "Link GRAPH/TRAIN/SELECT tensor keys differ or contain heldout data")
    tensors = list(graph.values())+list(train.values())+list(select.values())
    if any(not torch.is_tensor(v) or v.device.type != "cpu" or v.layout != torch.strided or v.requires_grad for v in tensors):
        raise ValueError("Use plain immutable CPU tensors for link roles")
    x = graph["x"]
    if x.dtype != torch.float32 or x.ndim != 2 or not 0 < len(x) < 2**31 or x.shape[1] < 1 or not bool(torch.isfinite(x).all()):
        raise ValueError("Bind complete float32 node features and declared isolated nodes")
    raw = graph["raw_edge_index"]
    if raw.dtype != torch.long or raw.ndim != 2 or raw.shape[0] != 2 or not raw.shape[1]:
        raise ValueError("Native sampler needs complete int64 directed TRAIN raw edges")
    for role, pairs in (("raw", raw.T), ("train_positive", train["positive"]), ("select_positive", select["positive"]), ("select_negative", select["negative"])):
        if pairs.dtype != torch.long or pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs):
            raise ValueError("Expected complete [queries,2] int64 edge pairs")
        allow_self = role == "select_negative" and config["support"].get("select_negative_self_policy", "reject") == "preserve_official_records"
        if int(pairs.min()) < 0 or int(pairs.max()) >= len(x) or (not allow_self and bool((pairs[:, 0] == pairs[:, 1]).any())):
            raise ValueError("Invalid endpoint or self-link in source role pairs")
    # Preserve raw edge order but require its exact bidirectional TRAIN multiset.
    positive = train["positive"]
    expected = torch.cat((positive, positive.flip(1)), dim=0)
    if not torch.equal(pair_keys(torch, raw.T, len(x), undirected=False).sort().values,
                       pair_keys(torch, expected, len(x), undirected=False).sort().values):
        raise ValueError("Raw sampler graph is not the exact bidirectional TRAIN multiset")
    train_keys = pair_keys(torch, positive, len(x))
    if config["support"]["record_duplicate_policy"] == "require_unique_pairs" and len(torch.unique(train_keys)) != len(positive):
        raise ValueError("This recipe requires unique canonical TRAIN edge records")
    selected_positive = pair_keys(torch, select["positive"], len(x))
    selected_negative = pair_keys(torch, select["negative"], len(x))
    if intersects(torch, selected_positive, train_keys) or intersects(torch, selected_negative, train_keys):
        raise ValueError("SELECT target edges leak into TRAIN graph support")
    if intersects(torch, selected_positive, selected_negative):
        raise ValueError("SELECT positive and negative canonical identities contradict")
    if len(select["negative"]) < config["metric"]["hits_k"]:
        raise ValueError("SELECT negative pool is smaller than declared Hits K")
    return {"nodes": len(x), "features": x.shape[1], "train_records": len(positive),
            "train_unique_pairs": len(torch.unique(train_keys)), "raw_directed_records": raw.shape[1],
            "select_positive": len(selected_positive), "select_negative": len(selected_negative),
            "select_negative_self_pairs": int((select["negative"][:, 0] == select["negative"][:, 1]).sum()),
            "train_only_support": True, "heldout_edges_in_support": False}


def sampler_custody(sampler):
    path = Path(inspect.getsourcefile(sampler)).resolve()
    signature = inspect.signature(sampler)
    if not {"num_neg_samples", "method", "force_undirected"} <= set(signature.parameters):
        raise ValueError("PyG native negative-sampler signature changed")
    if (signature.parameters["num_neg_samples"].default is not None
            or signature.parameters["method"].default != "sparse"
            or signature.parameters["force_undirected"].default is not False):
        raise ValueError("PyG native negative-sampler defaults changed")
    return {"name": "pyg_native_defaults", "source_path": str(path),
            "source_sha256": file_sha(path),
            "function_sha256": hashlib.sha256(inspect.getsource(sampler).encode()).hexdigest(),
            "call": "negative_sampling(raw_TRAIN_edge_index, num_nodes); all other defaults untouched"}


def rng_state(torch, numpy):
    name, keys, position, gauss, cached = numpy.random.get_state()
    return {"python": random.getstate(), "numpy": (name, keys.tolist(), position, gauss, cached),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []}


def restore_rng(torch, numpy, state):
    random.setstate(state["python"])
    name, keys, position, gauss, cached = state["numpy"]
    numpy.random.set_state((name, numpy.asarray(keys, dtype="uint32"), position, gauss, cached))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class LinkAdapter:
    def __init__(self, torch, numpy, model, optimizer, *, utils, sparse_tensor, sampler,
                 evaluator, graph, train, select, config, selector, provenance, fit_rng):
        self.t, self.np, self.model, self.optimizer = torch, numpy, model, optimizer
        self.F, self.utils, self.SparseTensor = torch.nn.functional, utils, sparse_tensor
        self.sampler, self.evaluator = sampler, evaluator
        self.graph, self.train, self.select = graph, train, select
        self.config, self.selector, self.provenance = config, dict(selector), copy.deepcopy(provenance)
        self.device = next(model.parameters()).device
        self._rng, self._best, self._best_step, self._epoch = fit_rng, None, None, 0
        self._active_counts = None
        self._setup_charged = False
        # Call the original bound source method unchanged; count recursive
        # NCNC invocations too, rather than claiming two top-level calls are
        # the whole completion-decoder cost.
        original = self.model.decoder.multidomainforward
        def counted_decoder(*args, **kwargs):
            counters = self._active_counts
            if counters is not None:
                counters["native_decoder_call_attempts"] = counters.get("native_decoder_call_attempts", 0)+1
                query = args[2] if len(args) > 2 else kwargs["tar_ei"]
                counters["native_decoder_query_rows"] = counters.get("native_decoder_query_rows", 0)+int(query.shape[1])
            value = original(*args, **kwargs)
            if counters is not None:
                counters["native_decoder_completed_calls"] = counters.get("native_decoder_completed_calls", 0)+1
            return value
        self.model.decoder.multidomainforward = counted_decoder
        self.full_graph = self._support(train["positive"])
        self.forbidden_directed = pair_keys(torch, graph["raw_edge_index"].T, len(graph["x"]), undirected=False).sort().values
        self.provenance["adapter_setup_operations"] = {"initial_full_support_builds": 1,
            "initial_support_input_records": len(train["positive"]),
            "input_tensor_bytes": sum(v.numel()*v.element_size() for pack in (graph, train, select) for v in pack.values())}

    @contextmanager
    def _scope(self, *, training):
        outer = rng_state(self.t, self.np)
        restore_rng(self.t, self.np, self._rng)
        try:
            yield
        finally:
            if training:
                self._rng = rng_state(self.t, self.np)
            restore_rng(self.t, self.np, outer)

    def _support(self, pairs):
        # Literal source-native graph construction; no VALID/TEST augmentation.
        return self.SparseTensor.from_edge_index(pairs.T, sparse_sizes=(len(self.graph["x"]), len(self.graph["x"]))).to_device(
            pairs.device, non_blocking=True).to_symmetric()

    def _finite_state(self, *, gradients):
        for name, parameter in self.model.named_parameters():
            if not bool(self.t.isfinite(parameter).all()):
                raise FloatingPointError("Nonfinite link model parameter")
            unused_ptlin = ".ptlin." in name and not self.config["predictor"]["completion"]["learnablept"]
            if gradients and parameter.grad is None and not unused_ptlin:
                raise ValueError("Missing active native gradient: "+name)
            if parameter.grad is not None and not bool(self.t.isfinite(parameter.grad).all()):
                raise FloatingPointError("Nonfinite native link gradient")
        for state in self.optimizer.state.values():
            for value in state.values():
                if self.t.is_tensor(value) and not bool(self.t.isfinite(value).all()):
                    raise FloatingPointError("Nonfinite link Adam state")

    def _number(self, value):
        result = float(value.detach().item())
        if not math.isfinite(result):
            raise FloatingPointError("Nonfinite link loss")
        return result

    def train_step(self, epoch):
        self.model.train()
        means = EpochMeans()
        counts = {k: 0 for k in ("optimizer_updates", "optimizer_update_attempts", "backward_calls", "backward_attempts",
            "encoder_calls", "encoder_call_attempts", "decoder_query_passes", "positive_records", "negative_records", "sampler_calls", "sampler_completed_calls",
            "negative_records_drawn", "unused_negative_records_drawn", "dropped_positive_tail", "graph_rebuilds",
            "support_input_records", "graph_rebuild_attempts", "supervised_examples", "native_decoder_call_attempts",
            "native_decoder_completed_calls", "native_decoder_query_rows")}
        counts.update({k: (0 if self._setup_charged else v) for k, v in self.provenance["adapter_setup_operations"].items()})
        self._setup_charged = True
        self._active_counts = counts
        try:
            with self._scope(training=True):
                counts["sampler_calls"] += 1
                negatives = self.sampler(self.graph["raw_edge_index"], len(self.graph["x"]))
                counts["sampler_completed_calls"] += 1
                if self.t.is_tensor(negatives) and negatives.ndim == 2:
                    counts["negative_records_drawn"] = negatives.shape[1]
                n = len(self.train["positive"])
                if negatives.dtype != self.t.long or negatives.ndim != 2 or negatives.shape[0] != 2 or negatives.shape[1] < n:
                    raise ValueError("Native negative draw is too short for every TRAIN record index")
                if bool((negatives[0] == negatives[1]).any()) or int(negatives.min()) < 0 or int(negatives.max()) >= len(self.graph["x"]):
                    raise ValueError("Native negative sampler produced self/outside endpoints")
                if intersects(self.t, pair_keys(self.t, negatives.T, len(self.graph["x"]), undirected=False), self.forbidden_directed):
                    raise ValueError("Native negative sampler returned forbidden TRAIN edges")
                counts["negative_records_drawn"] = negatives.shape[1]
                iterator = self.utils.PermIterator(self.device, n, self.config["training"]["batch_size"], True)
                if self.config["training"]["tail_policy"] == "complete_tail":
                    records = iterator.idx.split(self.config["training"]["batch_size"])
                else:
                    records = iterator
                for ids in records:
                    keep = self.t.ones(n, device=self.device, dtype=self.t.bool)
                    keep[ids] = False
                    support_pairs = self.train["positive"][keep]
                    counts["graph_rebuild_attempts"] += 1
                    graph = self._support(support_pairs)
                    counts["graph_rebuilds"] += 1
                    counts["support_input_records"] += len(support_pairs)
                    self.optimizer.zero_grad(set_to_none=True)
                    counts["encoder_call_attempts"] += 1
                    h = self.model.encoder(self.graph["x"], graph)
                    counts["encoder_calls"] += 1
                    positive = self.model.decoder.multidomainforward(h, graph, self.train["positive"][ids].T, cndropprobs=[]).flatten()
                    counts["decoder_query_passes"] += 1
                    negative = self.model.decoder.multidomainforward(h, graph, negatives[:, ids], cndropprobs=[]).flatten()
                    counts["decoder_query_passes"] += 1
                    if positive.shape != (len(ids),) or negative.shape != (len(ids),):
                        raise ValueError("Native link predictor query coverage changed")
                    if not bool(self.t.isfinite(positive).all()) or not bool(self.t.isfinite(negative).all()):
                        raise FloatingPointError("Nonfinite native link training scores")
                    positive_loss = -self.F.logsigmoid(positive).mean()
                    negative_loss = -self.F.logsigmoid(-negative).mean()
                    loss = positive_loss+negative_loss
                    metrics = {"objective": self._number(loss), "ce": self._number(loss), "regularization": 0.,
                               "positive_ce": self._number(positive_loss), "negative_ce": self._number(negative_loss)}
                    counts["positive_records"] += len(ids)
                    counts["negative_records"] += len(ids)
                    counts["supervised_examples"] += 2*len(ids)
                    counts["backward_attempts"] += 1
                    loss.backward()
                    counts["backward_calls"] += 1
                    self._finite_state(gradients=True)
                    counts["optimizer_update_attempts"] += 1
                    self.optimizer.step()
                    counts["optimizer_updates"] += 1
                    self._finite_state(gradients=False)
                    means.add(metrics, samples=len(ids))
                counts["dropped_positive_tail"] = n-counts["positive_records"]
                counts["unused_negative_records_drawn"] = negatives.shape[1]-counts["negative_records"]
            result = means.finish()
            expected = n if self.config["training"]["tail_policy"] == "complete_tail" else n//self.config["training"]["batch_size"]*self.config["training"]["batch_size"]
            if result["sample_count"] != expected:
                raise ValueError("Incomplete declared native link epoch")
            result.update(operation_counts=counts, positive_denominator=counts["positive_records"],
                negative_denominator=counts["negative_records"], loss_reduction="mean_positive_logistic + mean_negative_logistic",
                loss_timing="preupdate; dropout_on; balanced sum; record-weighted whole declared epoch",
                tail_policy=self.config["training"]["tail_policy"])
            self._epoch = epoch
            return result
        except BaseException as error:
            counts["unused_negative_records_drawn"] = max(0, counts["negative_records_drawn"]-counts["negative_records"])
            raise AdapterStepError(str(error), operation_counts=counts) from error
        finally:
            self._active_counts = None

    def predict_queries(self, positive, negative):
        """Raw query scores using fixed full TRAIN support; no role files opened."""
        self.model.eval()
        rows = []
        counters = {"native_decoder_call_attempts": 0, "native_decoder_completed_calls": 0, "native_decoder_query_rows": 0,
                    "encoder_call_attempts": 0, "encoder_calls": 0, "optimizer_updates": 0, "decoder_query_passes": 0,
                    "positive_records": 0, "negative_records": 0}
        self._active_counts = counters
        try:
            with self._scope(training=False), self.t.no_grad():
                counters["encoder_call_attempts"] += 1
                h = self.model.encoder(self.graph["x"], self.full_graph)
                counters["encoder_calls"] += 1
                for role, queries in (("positive", positive), ("negative", negative)):
                    if not self.t.is_tensor(queries) or queries.dtype != self.t.long or queries.ndim != 2 or queries.shape[1] != 2 or not len(queries) or queries.device != self.device:
                        raise ValueError("Serving queries must be nonempty int64 [rows,2] tensors on the body device")
                    if int(queries.min()) < 0 or int(queries.max()) >= len(self.graph["x"]):
                        raise ValueError("Serving query endpoints exceed the frozen TRAIN node authority")
                    scores = []
                    for ids in self.utils.PermIterator(self.device, len(queries), self.config["training"]["eval_batch_size"], False):
                        value = self.model.decoder(h, self.full_graph, queries[ids].T).flatten()
                        counters["decoder_query_passes"] += 1
                        if value.shape != (len(ids),) or not bool(self.t.isfinite(value).all()):
                            raise FloatingPointError("Incomplete/nonfinite native link serving")
                        scores.append(value.detach().cpu())
                        counters[role+"_records"] += len(ids)
                    output = self.t.cat(scores)
                    if len(output) != len(queries):
                        raise ValueError("A SELECT query tail was omitted")
                    rows.append(output)
            return rows[0], rows[1], counters
        except BaseException as error:
            raise AdapterStepError(str(error), operation_counts=counters) from error
        finally:
            self._active_counts = None

    def validate(self, epoch):
        positive, negative, counts = self.predict_queries(self.select["positive"], self.select["negative"])
        counts.update(metric_evaluator_attempts=0, metric_evaluator_calls=0)
        try:
            k = self.config["metric"]["hits_k"]
            counts["metric_evaluator_attempts"] += 1
            hits = float(self.evaluator.eval({"y_pred_pos": positive, "y_pred_neg": negative})["hits@"+str(k)])
            counts["metric_evaluator_calls"] += 1
            threshold = self.t.topk(negative, k).values[-1]
            explicit = int((positive > threshold).sum().item())/len(positive)
            if hits != explicit:
                raise ValueError("OGB strict shared-pool Hits tie rule changed")
            pos = self._number(-self.F.logsigmoid(positive).mean())
            neg = self._number(-self.F.logsigmoid(-negative).mean())
            result = {"objective": pos+neg, "ce": pos+neg, "regularization": 0., "positive_ce": pos,
                      "negative_ce": neg, "hits@"+str(k): hits, "positive_count": len(positive),
                      "negative_count": len(negative), "support": "full_TRAIN_only",
                      "operation_counts": counts, "loss_reduction": "balanced_positive_negative_sum",
                      "fusion": "one raw native scorer; later I4 mean of own-selected raw scores"}
            result["selection_metric"] = result[self.selector["metric"]]
            return result
        except BaseException as error:
            raise AdapterStepError(str(error), operation_counts=counts) from error

    def observe_validation(self, epoch, metrics):
        score = float(metrics["selection_metric"])
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite link selector")
        if self._best is None:
            better = True
        elif self.selector["direction"] == "max":
            better = score > self._best if self.selector["ties"] == "first" else score >= self._best
        else:
            better = score < self._best if self.selector["ties"] == "first" else score <= self._best
        if better:
            self._best, self._best_step = score, epoch
        return {"save_selected": better, "stop_requested": False, "best_saved_metric": self._best,
                "best_saved_step": self._best_step, "selector_scope": "one paid native body", "full_budget": True}

    def capture_state(self):
        return {"provenance": copy.deepcopy(self.provenance), "fit_rng": copy.deepcopy(self._rng),
                "outer_rng": rng_state(self.t, self.np), "epoch": self._epoch,
                "setup_charged": self._setup_charged,
                "selector": {"best": self._best, "best_step": self._best_step}}

    def restore_state(self, state):
        if state["provenance"] != self.provenance:
            raise ValueError("Link source/input/model/sampler/recipe ancestry differs")
        self._rng, self._epoch = copy.deepcopy(state["fit_rng"]), state["epoch"]
        self._setup_charged = state["setup_charged"]
        self._best, self._best_step = state["selector"]["best"], state["selector"]["best_step"]
        restore_rng(self.t, self.np, state["outer_rng"])


def make_adapter(*, torch, project_root, study, run, device):
    config = validate_config(run["config"], run)
    validate_roster(study, run)
    import numpy as np
    from torch_sparse import SparseTensor
    from torch_geometric.utils import negative_sampling
    from ogb.linkproppred import Evaluator
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Native NCN requires explicit float32 arithmetic")
    inputs = {k: torch.load(bound_file(project_root, item), map_location="cpu", weights_only=True)
              for k, item in config["link_inputs"].items()}
    input_audit = validate_inputs(torch, inputs["graph"], inputs["train"], inputs["select"], config)
    if config["training"]["tail_policy"] == "native_drop_tail" and input_audit["train_records"] < config["training"]["batch_size"]:
        raise ValueError("Native floor batching would yield an empty epoch")
    sampler_pin = sampler_custody(negative_sampling)
    if any(config["sampler"][key] != sampler_pin[key] for key in ("source_sha256", "function_sha256")):
        raise ValueError("Qualified negative-sampler source differs")
    evaluator_path = Path(inspect.getfile(Evaluator))
    if file_sha(evaluator_path) != config["metric"]["evaluator_source_sha256"]:
        raise ValueError("Qualified OGB evaluator source differs")
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.name, evaluator.K = study["dataset"], config["metric"]["hits_k"]
    evaluator.eval_metric = "hits@"+str(evaluator.K)
    native, utils = native_modules(project_root, config["link_source"])
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.set_device(target)
    outer = rng_state(torch, np)
    try:
        seed = config["initialization_seed"]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if target.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        e, p = config["encoder"], config["predictor"]
        class Body(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = native.GCN(input_audit["features"], e["hidden"], e["hidden"], e["layers"],
                    e["dropout"], ln=e["ln"], res=e["res"], max_x=-1, conv_fn="gcn", jk=e["jk"],
                    edrop=e["edrop"], xdropout=e["xdropout"], taildropout=e["taildropout"])
                cls = native.CNLinkPredictor if p["name"] == "NCN" else native.IncompleteCN1Predictor
                kwargs = {k: p[k] for k in ("edrop", "ln", "cndeg", "use_xlin", "tailact", "twolayerlin", "beta")}
                if p["name"] == "NCNC":
                    kwargs.update(p["completion"])
                self.decoder = cls(e["hidden"], e["hidden"], 1, p["layers"], p["dropout"], **kwargs)
        model = Body().to(target)
        fit_rng = rng_state(torch, np)
    finally:
        restore_rng(torch, np, outer)
    o = config["optimizer"]
    optimizer = torch.optim.Adam([{"params": model.encoder.parameters(), "lr": o["encoder_lr"]},
                                  {"params": model.decoder.parameters(), "lr": o["predictor_lr"]}],
        betas=tuple(o["betas"]), eps=o["eps"], weight_decay=o["weight_decay"], amsgrad=o["amsgrad"])
    inputs = {role: {key: value.to(target) for key, value in pack.items()} for role, pack in inputs.items()}
    provenance = {"run_id": run["run_id"], "body": run["body"], "dataset": study["dataset"],
        "source": study["source"], "data": study["data"], "config": config, "budget": run["budget"],
        "selection": run["selection"], "input_audit": input_audit, "negative_sampler": sampler_pin,
        "evaluator_source_sha256": config["metric"]["evaluator_source_sha256"],
        "author_repository": AUTHOR_REPOSITORY, "author_commit": AUTHOR_COMMIT,
        "source_forward_reused": True, "author_protocol_replication": False,
        "graph_policy": "TRAIN-only during fit and SELECT; no future-positive rejection in TRAIN negative sampling"}
    return LinkAdapter(torch, np, model, optimizer, utils=utils, sparse_tensor=SparseTensor,
        sampler=negative_sampling, evaluator=evaluator, graph=inputs["graph"], train=inputs["train"],
        select=inputs["select"], config=config, selector=run["selection"], provenance=provenance, fit_rng=fit_rng)
