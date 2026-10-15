"""Source-mapped four-route private residuals; no native program is rewritten.

This is a candidate implementation, not evidence of predictive improvement or
methodological novelty. Attach only before the first optimizer update. Both
variants retain the original shared native body and mean per-route native loss.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent
PHASE = EXPANSION.parent
METHODS = {"portable_input_residual_be_v1": "BE_plus_residual",
           "portable_input_residual_only_v1": "residual_only"}


def _receipt(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _load_core():
    path = HERE/"core.py"
    name = "portable_private_input_"+_receipt(path)["sha256"]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _source_bindings(family, root, expected_manifest=None):
    root = Path(root).resolve()
    manifest_path = HERE/"MANIFEST.json"
    manifest_pin = _receipt(manifest_path)
    if expected_manifest is not None and manifest_pin["sha256"] != expected_manifest:
        raise ValueError("Portable private-input manifest differs from the declared method")
    manifest = json.loads(manifest_path.read_text())
    receipts = [manifest_pin]
    for row in manifest["files"]:
        if row["path"].endswith(".py") or row["path"] in ("SOURCES.json", "METHODS.json"):
            path = (HERE/row["path"]).resolve()
            if not path.is_relative_to(HERE):
                raise ValueError("Portable method source escapes its package")
            actual = _receipt(path)
            if any(actual[key] != row[key] for key in ("bytes", "sha256")):
                raise ValueError("Portable private-input source changed: "+row["path"])
            receipts.append(actual)
    source = json.loads((HERE/"SOURCES.json").read_text())
    for row in source["source_files"]:
        if family not in row["families"]:
            continue
        path = (PHASE/row["path_relative_to_phase"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Native source binding escapes the repository")
        actual = _receipt(path)
        if any(actual[key] != row[key] for key in ("bytes", "sha256")):
            raise ValueError("Frozen source interface changed: "+row["path_relative_to_phase"])
        receipts.append(actual)
    return receipts


def _link_native(session):
    # The frozen LinkSession proxy stores both fields in its own instance.
    # Do not infer another model through a generic recursive attribute search.
    return session.native if "native_binding" in vars(session) and "native" in vars(session) else session


def source_mapping(session, family):
    """Resolve only declared source sites, never search for convenient layers."""
    if family == "node":
        core = session.core
        if core is None or session.model is not core or not hasattr(core, "native"):
            raise TypeError("Actual four-route native PolyFormer boundary session required")
        native = core.native
        if not all(hasattr(native, name) for name in ("lin1", "lin3")):
            raise TypeError("The declared PolyFormer input/output interface is unavailable")
        first, last = native.lin1, native.lin3
        if first.weight.ndim != 2 or last.weight.ndim != 2:
            raise TypeError("Native PolyFormer affines must retain their original shapes")
        return session, native, core, [("lin1", first, int(first.weight.shape[0]), first.weight)], ("lin3", last)
    if family == "ogb":
        core = session.model
        if not hasattr(core, "native") or not hasattr(core, "input_bank") or not hasattr(core, "output_bank"):
            raise TypeError("Actual full-AtomEncoder four-route OGB boundary session required")
        native = core.native
        encoder, head = native.gnn_node.atom_encoder, native.graph_pred_linear
        tables = tuple(encoder.atom_embedding_list)
        if len(tables) != 9 or any(table.embedding_dim != 300 for table in tables) \
                or head.in_features != 300 or session.config.get("feature") != "full":
            raise TypeError("This candidate requires the complete native nine-field300-wide AtomEncoder")
        return session, native, core, [("gnn_node.atom_encoder", encoder, 300, tables[0].weight)], ("graph_pred_linear", head)
    if family == "link":
        owner = _link_native(session)
        core, native = owner.core, owner.model
        if core is None or owner.members != 4 or owner.procedure != "base" or not hasattr(core, "banks"):
            raise TypeError("Actual four-route source link own-loss boundary session required")
        source_family = str(owner.metadata["family"]).upper()
        if source_family in ("NCN", "NCNC"):
            first, last = owner.encoder.convs[0].lin, owner.predictor.lin[-1]
            if tuple(first.weight.shape) != (64, 128) or tuple(last.weight.shape) != (1, 64):
                raise TypeError("Only the declared Collab NCN/NCNC logical boundary shapes are implemented")
            inputs = [("encoder.convs.0.lin", first, 64, first.weight)]
            output = ("predictor.lin."+str(len(owner.predictor.lin)-1), last)
            if source_family == "NCNC" and owner.task_call_scope is None:
                raise TypeError("The corrected NCNC top-level task scope is mandatory")
        elif source_family == "BUDDY":
            structural, feature, last = native.label_lin_layer, native.lin_feat, native.lin
            if tuple(structural.weight.shape) != (16, 16) or tuple(feature.weight.shape) != (1024, 128) \
                    or tuple(last.weight.shape) != (1, 1040) or native.sign_k != 0:
                raise TypeError("Only the declared full-feature BUDDY Collab SIGN-k0 input branches are implemented")
            inputs = [("label_lin_layer", structural, 16, structural.weight),
                      ("lin_feat", feature, 1024, feature.weight)]
            output = ("lin", last)
        else:
            raise TypeError("No residual mapping for native link family "+source_family)
        declared = [row["site"] for row in core.binding_metadata if row["first_input"]]
        if declared != [row[0] for row in inputs]:
            raise ValueError("Current source-mapped link input boundaries differ from the frozen core")
        return owner, native, core, inputs, output
    raise ValueError("No declared portable input boundary for family "+str(family))


def _assert_ownership(owner, source_parameters, removed_ids=()):
    model_parameters = {id(p) for p in owner.model.parameters()}
    current = dict(owner.model.named_parameters())
    if any(current.get(name) is not parameter for name, parameter in source_parameters.items()):
        raise ValueError("An original native Parameter was removed or renamed")
    owned = [p for group in owner.optimizer.param_groups for p in group["params"]]
    if len(owned) != len({id(p) for p in owned}) or {id(p) for p in owned} != model_parameters:
        raise ValueError("Model/optimizer ownership is incomplete or aliased")
    if model_parameters.intersection(removed_ids):
        raise ValueError("Residual-only model still registers a removed BE factor")


def transform(session, job):
    family, method = job.get("family"), job.get("procedure")
    if method not in METHODS or job.get("TEST_access") is not False:
        raise ValueError("Explicit portable method ID and TEST_access=false required")
    if job.get("policy", {}).get("track") == "source_reference":
        raise ValueError("A new residual method cannot be called a source-reference baseline")
    settings = job.get("method_settings", {})
    allowed = {"rank_cap", "frame_policy", "manifest_sha256"}
    if set(settings)-allowed or settings.get("rank_cap") != 8 \
            or settings.get("frame_policy") != "disjoint_if_fit_else_separate_orthonormal_frames":
        raise ValueError("Declare fixed rank cap8 and the dimension-aware frame policy; no implicit tuning")
    if not isinstance(settings.get("manifest_sha256"), str) or len(settings["manifest_sha256"]) != 64:
        raise ValueError("Pin the actual portable method manifest SHA256 before execution")
    root = Path(job.get("recipe", {}).get("_repository_root", PHASE.parent)).resolve()
    if root != PHASE.parent.resolve():
        raise ValueError("The declared repository root differs from this checked-out method source")
    receipts = _source_bindings(family, root, settings.get("manifest_sha256"))
    corelib = _load_core()
    owner, native, oldcore, inputs, output = source_mapping(session, family)
    if int(getattr(owner, "updates", -1)) != 0 or getattr(owner, "_portable_private_input_attached", False):
        raise ValueError("Attach once to a freshly constructed untrained source session")
    torch = owner.t if family == "link" else owner.torch
    mode = METHODS[method]
    before = dict(owner.source_parameter_references)
    _assert_ownership(owner, before)
    oldprivate = tuple(oldcore.private_parameters())
    seed = int(job["seed"])+901003
    locations, removed_ids, removed_count = [], frozenset(), 0
    if mode == "residual_only":
        corelib.remove_private_ownership(owner.optimizer, oldprivate)
        removed_ids = frozenset(id(p) for p in oldprivate)
        removed_count = sum(p.numel() for p in oldprivate)
        if family == "link":
            for handle in oldcore._handles:
                handle.remove()
            oldcore._handles = []
            if native._modules.get("_canonical_boundary_banks") is not oldcore:
                raise ValueError("The actual source boundary-bank registration is unavailable")
            del native._modules["_canonical_boundary_banks"]
            newcore = corelib.make_link_residual_core(torch, native, inputs, output,
                seed=seed, task_scope=owner.task_call_scope)
            native.add_module("_portable_private_input_residuals", newcore)
            owner.core = newcore
            residuals = tuple(newcore.residuals)
        else:
            oldcore.remove_hooks() if family == "node" else oldcore.close()
            newcore = corelib.make_residual_only_wrapper(torch, native, inputs, output, seed=seed, family=family)
            owner.model = newcore
            if family == "node":
                owner.core = newcore
            residuals = (newcore.input_residual,)
        # Existing callbacks retain the same four-route own-loss update.
        # Only the source wrapper's BE hooks/private banks were removed.
        for (_, _, _, anchor), residual in zip(inputs, residuals):
            group = corelib.assign_private_parameters(owner.optimizer, anchor, residual.parameters())
            locations.append({"optimizer_group": group, **residual.metadata()})
    else:
        residuals = torch.nn.ModuleList(corelib.build_residual(torch, anchor, width=width,
            seed=seed+104729*index) for index, (_, _, width, anchor) in enumerate(inputs))
        oldcore.add_module("_portable_private_input_residuals", residuals)
        original_private = oldcore.private_parameters
        def private_parameters(self):
            return tuple(original_private())+tuple(self._portable_private_input_residuals.parameters())
        oldcore.private_parameters = types.MethodType(private_parameters, oldcore)
        for index, ((path, module, width, anchor), residual) in enumerate(zip(inputs, residuals)):
            def post(source, arguments, encoded, adapter=residual):
                member = oldcore.route if family == "link" else oldcore._active_member if family == "node" else oldcore._member
                return None if member is None else adapter(encoded, int(member))
            # Append after the existing logical boundary's BE output transform.
            oldcore._handles.append(module.register_forward_hook(post))
            group = corelib.assign_private_parameters(owner.optimizer, anchor, residual.parameters())
            locations.append({"optimizer_group": group, **residual.metadata()})
    metadata = {"schema": "portable-private-input-session-v1", "method_id": method, "mode": mode,
        "family": family, "native_family": owner.metadata["family"] if family == "link" else "PolyFormer" if family == "node" else owner.config["gnn"],
        "input_sites": [{"path": path, "position": "after complete native logical input encoder and existing BE when retained",
                         "encoded_width": width, **details} for (path, _, width, _), details in zip(inputs, locations)],
        "output_boundary": output[0], "BE_retained": mode == "BE_plus_residual",
        "BE_factor_parameters_removed": removed_count,
        "native_backbone_copies": 1, "routes": 4, "native_parameters_replaced": 0,
        "native_operators_or_graph_preprocessing_changed": False,
        "training_objective": "mean of four native per-route losses; one optimizer transition",
        "prediction_pool": "mean raw logits", "source_optimizer_options_changed": False,
        "private_optimizer_assignment": "same existing native input-anchor group",
        "private_dropout_RNG_policy": "unchanged source session", "shared_BN_policy": "unchanged native BN; four route forwards",
        "initial_added_residual": "exact zero", "residual_only_initial_eval_routes": "identical native function before learning, with dropout disabled",
        "NCNC_output_scope": "unchanged top-level task scope; residual-only output hook is observer-only" if family == "link" and owner.metadata["family"] == "NCNC" else None,
        "native_runtime_qualification": "pending; no model/task/server execution in source preparation",
        "novelty_or_quality_claim": False, "TEST_accessed": False, "source_receipts": receipts}
    owner._portable_private_input_attached = True
    owner.portable_private_input_metadata = metadata
    original_state, original_load = owner.state, owner.load_state
    def state(self):
        return {"schema": "portable-private-input-wrapped-state-v1", "method": copy.deepcopy(metadata),
                "source_state": copy.deepcopy(original_state())}
    def load_state(self, value):
        if value.get("schema") != "portable-private-input-wrapped-state-v1" or value.get("method") != metadata:
            raise ValueError("Checkpoint portable method/source/initialization differs")
        original_load(copy.deepcopy(value["source_state"]))
        _assert_ownership(owner, before, removed_ids)
    owner.state, owner.load_state = types.MethodType(state, owner), types.MethodType(load_state, owner)
    if family == "link":
        owner.metadata["portable_private_input"] = metadata
        if mode == "residual_only":
            owner.metadata["boundary_bindings"] = copy.deepcopy(owner.core.binding_metadata)
            owner.boundary_bindings = copy.deepcopy(owner.core.binding_metadata)
            owner.metadata["canonical_scope"] = "four private nonlinear residuals at declared native input encoders; no BE factors; final task affine unchanged"
            if owner.task_call_scope is not None:
                owner.metadata["NCNC_output_scope"] = "observer-only original top-level task eligibility; no output factors"
        previous_close = getattr(owner, "close", None)
        def close(self):
            if previous_close is not None:
                previous_close()
            for handle in self.core._handles:
                handle.remove()
            self.core._handles = []
        owner.close = types.MethodType(close, owner)
        if mode == "residual_only" and owner.task_call_scope is not None:
            def rename_observer(value):
                work = value.get("work", value)
                if "NCNC_output_routing" in work:
                    work["NCNC_task_output_observer"] = work.pop("NCNC_output_routing")
                    work["NCNC_task_output_observer"]["inherited_applied_bypassed_names_mean_scope_eligibility"] = True
                value["NCNC_observer_only_no_output_factors"] = True
                return value
            original_update = owner.train_update
            def train_update(self):
                return rename_observer(original_update())
            owner.train_update = types.MethodType(train_update, owner)
            original_validate = session.validate
            def validate(self):
                return rename_observer(original_validate())
            session.validate = types.MethodType(validate, session)
        if owner is not session:
            session.model, session.optimizers = owner.model, owner.optimizers
    elif family == "node":
        owner.source_metadata["portable_private_input"] = metadata
    else:
        assignment = [{"portable_input_site": path, "group_index": details["optimizer_group"],
                       "extra_parameters": details["extra_parameters"]}
                      for (path, _, _, _), details in zip(inputs, locations)]
        owner.optimizer_assignment = (copy.deepcopy(owner.optimizer_assignment) if mode == "BE_plus_residual" else [])+assignment
        original_metadata = owner.metadata
        def combined_metadata(self):
            return {**original_metadata(), "portable_private_input": metadata}
        owner.metadata = types.MethodType(combined_metadata, owner)
    _assert_ownership(owner, before, removed_ids)
    return session


extend_session = transform
