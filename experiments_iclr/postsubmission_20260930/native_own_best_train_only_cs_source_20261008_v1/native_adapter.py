"""Disabled feature-only reconstruction of authentic native-own-best states.

No checkpoint/data loader, correction bank, fitting or metric evaluation.
Imports only stdlib until explicit later execution authorization.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct_native_own_best(*, state, train_data, polynormer,
                                development_ids, development_ids_sha256,
                                device="cpu", later_execution_authorized=False):
    """Return a serving-only wrapper; caller owns trusted state/role loading.

    development_ids_sha256 uses the pinned integration common.tensor_digest.
    No development truths, corrected checkpoints or checkpoint-kind rewriting.
    """
    if later_execution_authorized is not True:
        raise PermissionError("Disabled native-own-best reference; root release required")
    pins = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
    for row in pins["native_sources"]:
        path = PHASE / row["path"]
        _require(_sha(path) == row["sha256"], "Pinned native reference source changed")
    _require(state["schema"] == "label-only-four-bank-coherent-state-v1"
             and state["kind"] == "native_own_best" and state["arm"] is None
             and state["selector_performed"] is True
             and state["snapshot_purpose"] == "first_screen",
             "Authentic native-own-best kind only; never relabel an arm state")
    _require(type(state["epoch"]) is int and 1 <= state["epoch"] <= 1100,
             "Native selected epoch")
    screen_path = PHASE / pins["screen_directory"] / "screen.py"
    spec = importlib.util.spec_from_file_location("_cs_reference_native_screen", screen_path)
    screen = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = screen
    spec.loader.exec_module(screen)
    _require(state["run"]["source"] == screen.source_identity(),
             "Exact authentic scientific screen identity")
    _, _, common, _, public, _, _, _, _, _, _ = screen._dependencies()
    seed = state["run"]["seed"]
    _require(seed in (6101, 6203, 6307), "Frozen native seed")
    # The existing native factory is used directly. No screen.make_session,
    # corrector construction or state['banks'] access occurs here.
    native = public.Session("wikics", "single", seed, device, polynormer)
    torch = native.torch
    _require(native.config == state["run"]["native_recipe"]
             and native.core_provenance == state["run"]["native_core"]
             and native.native_provenance == state["run"]["native_source"]
             and common.context_identity(native, train_data) == state["run"]["data_context"],
             "Same complete TRAIN/graph/native source/recipe context")
    _require(tuple(train_data["x"].shape) == (11701, 300)
             and tuple(train_data["edge_index"].shape) == (2, 442907)
             and train_data["ids"].numel() == 580,
             "Full frozen native inputs")
    ids = development_ids.detach().to(device="cpu", dtype=torch.long).clone()
    _require(ids.ndim == 1 and ids.numel() == 5274 and ids.unique().numel() == 5274
             and int(ids.min()) >= 0 and int(ids.max()) < 11701
             and not bool(torch.isin(ids, train_data["ids"].cpu()).any())
             and common.tensor_digest(ids) == development_ids_sha256,
             "Audited full development ID role, no truths")
    payload = state["native"]
    _require(payload["epoch"] == state["epoch"]
             and payload["run"] == state["run"]
             and payload["config"] == native.config
             and bool(payload["global"]) == bool(state["global_mode"])
             and payload["checkpoint_kind"].startswith("native_own_best;"),
             "Authentic coherent native payload/model/mode")
    native.model.load_state_dict(payload["model"], strict=True)
    native.model.set_global(state["global_mode"])
    native.streams = common.clone_streams(payload["streams"])
    native.model.eval()
    for parameter in native.model.parameters():
        parameter.requires_grad_(False)

    class FeatureOnlyReference:
        """No update/selection/corrector interface; native optimizer is unused."""
        def probabilities(self):
            native.model.eval()
            with torch.no_grad():
                batch = dict(x=train_data["x"].to(native.device),
                             edge_index=train_data["edge_index"].to(native.device),
                             ids=torch.arange(11701, device=native.device, dtype=torch.long))
                logits, _ = native.forward(batch)
                _require(tuple(logits.shape) == (1, 11701, 10)
                         and bool(torch.isfinite(logits).all()), "Finite full native logits")
                full = logits[0].softmax(-1).detach()
                return dict(full_probabilities=full,
                            development_probabilities=full[ids.to(native.device)],
                            development_ids=ids.clone(),
                            native_epoch=state["epoch"], global_mode=bool(state["global_mode"]),
                            checkpoint_kind="native_own_best", classes=10,
                            source=state["run"]["source"], corrected_parameters_loaded=False,
                            native_forward_calls=1, metric_or_truth_read=False)

    return FeatureOnlyReference()
