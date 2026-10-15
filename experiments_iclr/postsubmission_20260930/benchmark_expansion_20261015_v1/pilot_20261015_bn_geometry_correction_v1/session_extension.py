"""Compose the unchanged completed molecular-ingredient and private-BN ports."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

METHODS = {
    "gnnm_private_bn_graph_alignment_variance_v1": (True, False),
    "gnnm_private_bn_input_residual_v1": (False, True),
    "gnnm_private_bn_input_residual_graph_alignment_variance_v1": (True, True),
}
PINS = {
    "molecular/session_extension.py": "8e2cc88b08d8a704c198af3d84a7a65cc5e63262fa3426776aaf4b83ecd1fe9d",
    "molecular/objectives.py": "75e7d59f8360aa2688b9cd974e9e28863e475c683f3dd073e2eacd6665aff71a",
    "adapters/input_residual.py": "c0def85281f68e91461474812202a6cffb1df2e4dabcf2c78b35d84c76f1776d",
    "private_bn_state_v3/session_extension.py": "3d9d9c47d2dd421d7bb54560f7d4466fff839c6d360e8571328dcf3f7874e14c",
    "private_bn_state_v3/banks.py": "d90a74bf98be32408cad306f35ab5ebd1ec80153981a0b7f92cba07e7f447fea",
}
NATIVE_CONFIG = {"JK": "last", "batch_size": 32, "dataset": "ogbg-molbace",
                 "drop_ratio": 0.5, "emb_dim": 300, "feature": "full", "gnn": "gin",
                 "graph_pooling": "mean", "lr": 0.001, "num_layer": 5, "residual": False}


def settings_for(method):
    geometry, correction = METHODS[method]
    return {"ingredient": {"auxiliary": geometry, "input_residual": correction, "adapter_rank": 8,
                           "dependency_sha256": {k: PINS[k] for k in
                                                 ("molecular/objectives.py", "adapters/input_residual.py")}},
            "banks_sha256": PINS["private_bn_state_v3/banks.py"]}


def _load(root, relative):
    name = "_bn_geometry_composition_" + PINS[relative]
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def transform(session, job):
    method = job.get("procedure")
    if method not in METHODS or job.get("method_settings") != settings_for(method):
        raise ValueError("Only the three exact frozen ingredient combinations are admitted")
    if job.get("family") != "ogb" or job.get("base_procedure") != "gnnm_boundary_native" \
            or job.get("seed") not in (101, 202, 303) or job.get("split") != 0 \
            or job.get("TEST_access") is not False or session.config != NATIVE_CONFIG \
            or session.procedure != "gnnm_boundary_native":
        raise ValueError("The frozen native MolBACE boundary session is required")
    if hasattr(session, "extension_metadata") or hasattr(session.model, "route_bn_state") \
            or hasattr(session.model, "private_input_residual"):
        raise ValueError("Composition requires a fresh native boundary session")
    root = Path(__file__).resolve().parents[1]
    for relative, expected in PINS.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Completed parent source changed: " + relative)
    native_model, native_optimizer = session.model, session.optimizer
    ingredient_job = copy.deepcopy(job)
    ingredient_job["method_settings"] = copy.deepcopy(job["method_settings"]["ingredient"])
    _load(root, "molecular/session_extension.py").transform(session, ingredient_job)
    ingredient_metadata = copy.deepcopy(session.extension_metadata)
    ingredient_update = session.train_update
    bn_job = copy.deepcopy(job)
    bn_job["method_settings"] = {"banks_sha256": PINS["private_bn_state_v3/banks.py"]}
    _load(root, "private_bn_state_v3/session_extension.py").transform(session, bn_job)
    bn_metadata = copy.deepcopy(session.metadata()["candidate_extension"])
    if session.model is not native_model or session.optimizer is not native_optimizer \
            or session.train_update != ingredient_update:
        raise ValueError("Composition replaced the native model/optimizer or ingredient update")
    binding = {"schema": "molbace-bn-geometry-correction-composition-v1", "method": method,
               "seed": job["seed"], "settings": copy.deepcopy(job["method_settings"]),
               "source_sha256": dict(PINS)}
    combined = {**binding, "ingredient": ingredient_metadata, "normalization": bn_metadata,
                "composition_order": ["unchanged parent ingredient transform", "unchanged private BN transform"],
                "TEST_accessed": False, "novelty_claim": False}
    original_metadata, original_state, original_load = session.metadata, session.state, session.load_state

    def metadata(self):
        value = original_metadata()
        value["candidate_extension"] = copy.deepcopy(combined)
        return value

    def state(self):
        value = original_state()
        value["bn_geometry_correction_binding"] = copy.deepcopy(binding)
        return value

    def load_state(self, value):
        if value.get("bn_geometry_correction_binding") != binding:
            raise ValueError("Checkpoint combination/seed/source binding differs")
        native = dict(value)
        native.pop("bn_geometry_correction_binding")
        original_load(native)

    session.extension_metadata = copy.deepcopy(combined)
    session.metadata = types.MethodType(metadata, session)
    session.state = types.MethodType(state, session)
    session.load_state = types.MethodType(load_state, session)
    return session
