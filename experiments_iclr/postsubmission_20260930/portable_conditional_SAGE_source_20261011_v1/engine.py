"""Portable binding over the preserved E owner and Physics sequential source.

Import and source rendering use only the standard library.  Numerical imports,
role loading and model construction occur only when an explicit caller invokes
construct/acquire after local validation.  No author admission is synthesized.
"""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types


SEEDS = (8101, 8219, 8363, 8501, 8663, 8819, 9001, 9199, 9403, 9623)
ARMS = ("BASE", "E4", "N4", "equipped_M1", "equipped_I4")
FIXED = {"depth": 2, "hidden": 128, "dropout": .2, "learning_rate": .001,
         "max_updates": 1000, "patience_updates": 300, "rank": 16,
         "member_seed_stride": 1000003, "factor_seed_offset": 2000003,
         "dropout_seed_offset": 3000007, "block_seed_offset": 4000037,
         "stalk_dim": 4, "baseline_policy": "fresh_required",
         "accumulation": "sequential_own_ce"}
PINNED = {
    "training_family": "dbe58a63e2976e865b475b3557ec8fad7568dcf31bfcd721de816772c51d55bd",
    "value_installer": "e92df73611dd2cf4c063dc7bee1c7182a1c07907a77777f10d2a748fd92bb23a",
    "value_helper": "cf97376779782482148ed90b57ea47d58fe18fd5554aab543d41cf3f0e9cf263",
    "contracts": "b50d8e24c2decf8c696f10bf694b724456011a0248fb47de99f47bceddc763d1",
    "sequential_port": "bf6d8d6694f2bc6946dde54d0aca2f50c9cd9e4dfa115a55fbfb1b92ec23d2fb",
    "own_ce": "5bb257313c3371f26881f1cf3131e5ec68e11b4afa3b3025c1bc56af41f02322",
    "common_routes": "84ff13b28ef471ac0a4199e10d4284f24647d396e021646e2a21a06f96fb0773",
    "factors": "9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3",
    "models": "07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303",
    "run_common": "e5256b39aef9fdc688832320a9e7d8aa24f244f48f6939d0b0daa342639a1ebc",
    "sage_conv": "db5a4998387eb0ae2ee1cb3d957699938e6c0fd1b0a1c8b6135c691dfe5af7c7",
    "message_passing": "66e4ef4afa1d1b2d46c805b8987b6ea0c52ec5c95a9beee8a62758358e481e6f",
}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _file(item, expected=None):
    if (not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}
            or type(item["bytes"]) is not int or item["bytes"] <= 0):
        raise ValueError("Exact local path/bytes/sha256 receipt required")
    path = Path(item["path"]).resolve()
    if not path.is_file() or path.stat().st_size != item["bytes"]:
        raise ValueError("Local receipt size differs: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    if digest.hexdigest() != item["sha256"] or (expected and expected != item["sha256"]):
        raise ValueError("Local receipt hash differs: " + str(path))
    return path


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _replace(source, before, after, count=1):
    if source.count(before) != count:
        raise ValueError("Portable source boundary anchor differs: " + before[:70])
    return source.replace(before, after)


def _function(source, name, class_name=None):
    tree = ast.parse(source)
    nodes = tree.body
    if class_name is not None:
        nodes = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


def render(cfg):
    """Validate source/config metadata and render without importing model/data code.

    Role payload bytes are checked by construct/acquire, not this source-only API.
    """
    cfg = copy.deepcopy(cfg)
    if any(k.startswith("root_") for k in cfg):
        raise ValueError("Portable config uses local protocol and validation records")
    if cfg.get("seeds") != list(SEEDS) or any(cfg.get(k) != v for k, v in FIXED.items()):
        raise ValueError("Exact Physics ten-mask sequential own-CE recipe required")
    if cfg.get("stage", "F") != "F" or cfg.get("K") is not None:
        raise ValueError("Only the complete core five-arm acquisition is exposed")
    limits = cfg.get("family_limits")
    if (not isinstance(limits, dict) or set(limits) != {"wall_seconds", "cuda_bytes", "rss_bytes"}
            or any(type(v) is not int or v <= 0 for v in limits.values())):
        raise ValueError("Finite local wall/CUDA/RSS limits required")
    if not isinstance(cfg.get("device"), str) or not cfg["device"]:
        raise ValueError("One explicit local device required")
    if not isinstance(cfg.get("protocol"), dict):
        raise ValueError("Explicit prospective protocol required")
    receipts = cfg.get("source_receipts")
    if not isinstance(receipts, dict) or set(receipts) != set(PINNED):
        raise ValueError("Exact canonical numerical dependency receipts required")
    paths = {key: _file(receipts[key], value) for key, value in PINNED.items()}
    e_directory = paths["training_family"].parent
    for key, filename in (("training_family", "run_family.py"),
                          ("value_installer", "value_installer.py"),
                          ("value_helper", "value_transport.py"), ("contracts", "contracts.py")):
        if paths[key] != e_directory / filename:
            raise ValueError("Canonical E source path mapping differs")
    if (Path(cfg["native_repo"]).resolve() != paths["models"].parent
            or paths["run_common"] != paths["models"].with_name("run_common.py")
            or Path(cfg["common_routes"]).resolve() != paths["common_routes"]
            or Path(cfg["factors"]).resolve() != paths["factors"]):
        raise ValueError("Local native/route/factor path mapping differs")
    return _render_sources(cfg, paths)


def _render_sources(cfg, paths):
    """Source-only rendering after the public metadata checks above."""
    receipts = cfg["source_receipts"]
    e_directory = paths["training_family"].parent
    port = _load("portable_physics_source_renderer", paths["sequential_port"])
    if cfg.get("profile", {}).get("dataset") != "CoauthorPhysics":
        raise ValueError("Only the fixed CoauthorPhysics profile is exposed")
    sources, profile_changes = port.render_sources(e_directory, cfg["profile"])
    owner = sources["owner"]
    owner = _replace(owner, "        from run_base import train_step\n", "")
    owner = _replace(owner, "        self.Model, self.train_step, self.set_seed = Model, train_step, set_seed",
                     "        self.Model, self.set_seed = Model, set_seed")
    owner = _replace(owner, '(("models", Model), ("run_base", train_step), ("run_common", set_seed),',
                     '(("models", Model), ("run_common", set_seed),')
    owner = _replace(owner, '"root_qualification_is_separate_paid_work": True,',
                     '"portable_binding_runs_qualification": False,')
    owner = _replace(owner, '"runtime_admission": self.learning_lock["root_runtime_admission"],',
                     '"local_runtime_validation": self.learning_lock["runtime"],')
    owner = _replace(owner, '"root external finite watchdog required"',
                     '"external finite watchdog required"')
    owner = _replace(owner, '"native sequential route acquisition; no packed execution claim; selected fullgraph forward/metrics/CPU transfer/pool; lifetime RSS peak; overlap supplied by root"',
                     '"native sequential route acquisition; no packed execution claim; selected fullgraph forward/metrics/CPU transfer/pool; lifetime RSS peak; overlap supplied by caller"')
    tree = ast.parse(owner)
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.FunctionDef)
                         and node.name in ("validate_config", "file_receipt", "main"))
                 and not (isinstance(node, ast.If)
                          and ast.unparse(node.test) == "__name__ == '__main__'")]
    owner = ast.unparse(tree) + "\n"
    role_source = _function(paths["sequential_port"].read_text(),
                            "activate_role_block", "ProfiledFamily")
    count_tree = ast.parse(paths["contracts"].read_text())
    count_value = next(ast.literal_eval(node.value) for node in count_tree.body
                       if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "VALUE_COUNT_KEYS" for t in node.targets))
    result = {"owner": owner, "installer": sources["installer"],
              "helper": sources["helper"], "role_activation": role_source,
              "value_count_keys": count_value}
    receipt = {"schema_version": 1, "binding_kind": "portable_local_numerical_binding",
               "config_sha256": _sha(_json_bytes(cfg)),
               "canonical_sources": {str(path): receipts[key]["sha256"] for key, path in paths.items()},
               "derived_sources": {k: _sha(v.encode()) for k, v in result.items() if isinstance(v, str)},
               "profile_changes": profile_changes,
               "engine_sha256": _sha(Path(__file__).read_bytes()),
               "author_host_admissions_reused": False}
    return result, receipt


def _roles(cfg):
    blocks, receipts = cfg.get("role_blocks"), cfg.get("role_receipts")
    seed_keys = {str(seed) for seed in SEEDS}
    if (not isinstance(blocks, dict) or set(blocks) != seed_keys
            or not isinstance(receipts, dict) or set(receipts) != seed_keys):
        raise ValueError("All ten paired TRAIN/SELECT blocks and receipts required")
    hashes = {}
    for seed in map(str, SEEDS):
        if set(blocks[seed]) != {"train_npz", "select_npz"} or set(receipts[seed]) != {"train", "select"}:
            raise ValueError("Exact portable TRAIN/SELECT role fields required")
        for role in ("train", "select"):
            item = receipts[seed][role]
            if _file(item) != Path(blocks[seed][role + "_npz"]).resolve():
                raise ValueError("Role file and receipt paths differ")
            hashes[seed + ":" + role] = item["sha256"]
    active = cfg.get("active_role_seed")
    safe = cfg.get("safe_roles")
    if active not in SEEDS or not isinstance(safe, dict) or set(safe) != {"train", "valid"}:
        raise ValueError("One exact initial active safe block required")
    for role, field in (("train", "train_npz"), ("valid", "select_npz")):
        expected = blocks[str(active)][field]
        if (Path(cfg["train_npz" if role == "train" else "valid_npz"]).resolve() != Path(expected).resolve()
                or _file(safe[role]) != Path(expected).resolve()
                or safe[role]["sha256"] != receipts[str(active)]["train" if role == "train" else "select"]["sha256"]):
            raise ValueError("Initial safe loader differs from active role block")
    return hashes


def _construct(cfg, output, local_validate, action):
    cfg = copy.deepcopy(cfg)
    sources, receipt = render(cfg)
    role_hashes = _roles(cfg)
    receipt["role_file_sha256"] = role_hashes
    if not callable(local_validate):
        raise ValueError("Explicit local runtime/data/protocol validator required")
    check = local_validate(copy.deepcopy(cfg), action=action,
                           executing_source=str(Path(__file__).resolve()),
                           source_receipt=copy.deepcopy(receipt))
    if (not isinstance(check, dict) or check.get("status") != "LOCAL_VALIDATED"
            or not isinstance(check.get("runtime"), dict)):
        raise ValueError("Local validator must return LOCAL_VALIDATED and runtime evidence")
    if "config_sha256" in check and check["config_sha256"] != receipt["config_sha256"]:
        raise ValueError("Local validator config digest differs")
    _json_bytes(check)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock = {"status": "LOCAL_VALIDATED", "action": action,
            "config_sha256": receipt["config_sha256"],
            "source_sha256": receipt["canonical_sources"],
            "safe_role_sha256": {k: v["sha256"] for k, v in cfg["safe_roles"].items()},
            "role_file_sha256": role_hashes, "runtime": copy.deepcopy(check["runtime"]),
            "local_checks": copy.deepcopy(check), "binding": receipt,
            "author_host_admissions_reused": False}
    module = types.ModuleType("portable_local_physics_owner")
    module.__file__ = str(Path(cfg["source_receipts"]["training_family"]["path"]).resolve())
    sys.modules[module.__name__] = module
    exec(compile(sources["owner"], module.__file__, "exec"), module.__dict__)
    own = _load("portable_physics_sequential_own_ce", cfg["source_receipts"]["own_ce"]["path"])
    module.own_ce_update = own.update
    installer = types.ModuleType("portable_local_value_installer")
    installer.__file__ = str(Path(cfg["source_receipts"]["value_installer"]["path"]).resolve())
    sys.modules[installer.__name__] = installer
    exec(compile(sources["installer"], installer.__file__, "exec"), installer.__dict__)
    counts = types.SimpleNamespace(VALUE_COUNT_KEYS=sources["value_count_keys"],
                                   PROTOCOL=copy.deepcopy(cfg["protocol"]))
    original_load = module.load_source
    def load(name, path):
        if name == "endpoint_value_installer":
            return installer
        if name == "endpoint_value_contracts_counts":
            return counts
        return original_load(name, path)
    module.load_source = load
    def validate(actual_cfg, qualification_only=False):
        if actual_cfg != cfg or qualification_only:
            raise ValueError("Only the exact locally validated acquisition/export config is exposed")
        return copy.deepcopy(lock)
    module.validate_config = validate
    namespace = {"cfg": cfg, "dataset": "CoauthorPhysics", "module": module}
    exec(compile(sources["role_activation"], module.__file__, "exec"), namespace)
    original = module.Family
    class PortableFamily(original):
        activate_role_block = namespace["activate_role_block"]
        def __init__(self):
            super().__init__(cfg, output, receipt["config_sha256"])
            profile = cfg["profile"]
            if self.edge_metadata != {"nodes": profile["nodes"], "edges": profile["edges"], "self_edges": profile["self_edges"]}:
                raise ValueError("Full factual Physics support differs from local profile")
            self.active_role_seed = cfg["active_role_seed"]
            self.port_receipt = receipt
    module.write_json(output / "LOCAL_VALIDATION.json", lock)
    try:
        return PortableFamily()
    except BaseException as error:
        module.write_json(output / "FAILURE.json", {"status": "FAILED_RETAINED",
                          "complete": False, "type": type(error).__name__, "message": str(error),
                          "config_sha256": receipt["config_sha256"], "retry_or_rescue": False})
        raise


def construct(cfg, output, local_validate):
    """Construct the fixed full-graph Family for selected-state export; never fit."""
    return _construct(cfg, output, local_validate, "construct_for_export")


def acquire(cfg, output, local_validate):
    """Acquire all five arms for all ten masks, preserving failures and selections."""
    family = _construct(cfg, output, local_validate, "fixed_family_acquisition")
    try:
        family.run()
    except BaseException as error:
        temporary = Path(output) / "FAILURE.json"
        temporary.write_text(json.dumps({"status": "FAILED_RETAINED", "complete": False,
                             "type": type(error).__name__, "message": str(error),
                             "current": family.current, "completed_optimizer_bundles": family.completed_units,
                             "config_sha256": family.config_sha256, "retry_or_rescue": False}, indent=2) + "\n")
        raise
    return json.loads((Path(output) / "COMPLETE_FAMILY.json").read_text())
