"""One explicit same-runtime portable-binding check; never a fit or scorer.

Import is standard-library only.  Root owns the single execution and an external
600-second watchdog.  The sealed portable bundle and scientific configs are read
only.  No current scientific state, CAL/TEST target or predictive metric is used.
"""
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
MANIFEST_SHA256 = "d52b185077d81692472a0bedc002df864c1c7a34f6de55b4e398126c2ce80781"
LIMITS = {"wall_seconds": 600, "cuda_bytes": 48 * 2**30, "rss_bytes": 64 * 2**30}
SOURCE_PATHS = {
    "training_family": "deps/e_owner/run_family.py",
    "value_installer": "deps/e_owner/value_installer.py",
    "value_helper": "deps/e_owner/value_transport.py", "contracts": "deps/e_owner/contracts.py",
    "sequential_port": "deps/sequential/port.py", "own_ce": "deps/sequential/own_ce.py",
    "common_routes": "deps/routes/common_routes.py", "factors": "deps/factors/factors.py",
    "models": "deps/native/models.py", "run_common": "deps/native/run_common.py",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def exact_state(torch, actual, expected):
    if torch.is_tensor(actual):
        return (torch.is_tensor(expected) and actual.dtype == expected.dtype
                and actual.shape == expected.shape
                and torch.equal(actual.detach().cpu(), expected.detach().cpu()))
    if isinstance(actual, dict):
        return (isinstance(expected, dict) and actual.keys() == expected.keys()
                and all(exact_state(torch, actual[k], expected[k]) for k in actual))
    if isinstance(actual, (list, tuple)):
        return (type(actual) is type(expected) and len(actual) == len(expected)
                and all(exact_state(torch, a, b) for a, b in zip(actual, expected)))
    return type(actual) is type(expected) and actual == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permit-one-binding-check", action="store_true", required=True)
    args = parser.parse_args()
    package, output = args.package.resolve(), args.output.resolve()
    if not args.permit_one_binding_check or digest(package / "MANIFEST.json") != MANIFEST_SHA256:
        raise ValueError("Explicit one-check permission and exact sealed portable manifest required")
    seal = json.loads((package / "SEAL.json").read_text())
    if seal["manifest_sha256"] != MANIFEST_SHA256:
        raise ValueError("Portable package seal differs")
    inputs_path = Path(__file__).with_name("INPUT_BINDINGS.json")
    inputs = json.loads(inputs_path.read_text())
    if inputs["portable_manifest_sha256"] != MANIFEST_SHA256:
        raise ValueError("Check source/input binding differs")
    # Only the public package is on the project import path.  Family then binds
    # its own deps/native, and source inspection checks every imported identity.
    sys.path.insert(0, str(package))
    from common import bundle, canonical, checked, load, local_validate, receipt, require, write
    import engine
    require(Path(inspect.getfile(engine)).resolve() == package / "engine.py", "Portable engine import required")
    require(Path(sys.modules["common"].__file__).resolve() == package / "common.py", "Portable common import required")
    package_receipt = bundle()
    require(package_receipt["sha256"] == MANIFEST_SHA256, "Full portable manifest verification differs")
    for key in ("core_config", "frozen_bindings", "preparation"):
        checked(inputs[key])  # Metadata custody only; never execute original sources.
    sources = {key: receipt(package / relative) for key, relative in SOURCE_PATHS.items()}
    pyg = Path(importlib.util.find_spec("torch_geometric").origin).parent
    sources.update({key: receipt(pyg / ("nn/conv/" + filename)) for key, filename in
                    (("sage_conv", "sage_conv.py"), ("message_passing", "message_passing.py"))})
    require(all(sources[k]["sha256"] == engine.PINNED[k] for k in engine.PINNED), "Exact portable/PyG source pins required")
    roles = copy.deepcopy(inputs["role_receipts"])
    blocks = {seed: {"train_npz": row["train"]["path"], "select_npz": row["select"]["path"]}
              for seed, row in roles.items()}
    seed = inputs["active_role_seed"]
    cfg = dict(engine.FIXED, permission="LOCAL_RUN_PERMITTED", seeds=list(engine.SEEDS),
               native_repo=str(package / "deps/native"), common_routes=str(package / "deps/routes/common_routes.py"),
               factors=str(package / "deps/factors/factors.py"), source_receipts=sources,
               profile=copy.deepcopy(inputs["profile"]), role_blocks=blocks, role_receipts=roles,
               active_role_seed=seed, train_npz=blocks[str(seed)]["train_npz"], valid_npz=blocks[str(seed)]["select_npz"],
               safe_roles={"train": roles[str(seed)]["train"], "valid": roles[str(seed)]["select"]},
               family_limits=copy.deepcopy(LIMITS), device="cuda:0", stage="F", K=None,
               k_selection_receipt=None, protocol=json.loads((package / "PROTOCOL.json").read_text()),
               preparation_receipt=inputs["preparation"],
               file_bindings={key + ":" + role: item for key, row in roles.items() for role, item in row.items()})
    require(cfg["max_updates"] == 1000 and cfg["seeds"] == inputs["seeds"], "Preserve full fixed source recipe")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "CHECK_CONFIG.json", cfg)  # New check output only.
    started = time.monotonic()
    family = None
    def budget():
        if time.monotonic() - started > LIMITS["wall_seconds"]:
            raise RuntimeError("One-check wall budget exceeded; external watchdog also required")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if rss > LIMITS["rss_bytes"]:
            raise RuntimeError("One-check RSS budget exceeded")
        if family is not None and family.torch.cuda.max_memory_allocated(family.device) > LIMITS["cuda_bytes"]:
            raise RuntimeError("One-check CUDA budget exceeded")
    def validate(actual_cfg, **context):
        require(actual_cfg == cfg and context.get("action") == "construct_for_export", "Only constructor binding is exercised")
        value = local_validate(actual_cfg, **context)
        require(value["status"] == "LOCAL_VALIDATED", "Actual local validation required")
        budget()
        return value
    try:
        budget()
        family = engine.construct(cfg, output / "constructor", validate)
        budget()
        family._operation = "endpoint"
        model, optimizer, streams = family.make(seed, 0, "baseline", True, 4, "shared")
        own = load("portable_binding_check_own_ce", package / "deps/sequential/own_ce.py")
        stats = own.update(family, model, optimizer, streams, True, mode="sequential")
        family.synchronize(); budget()
        require((stats["backwards"], stats["adam_steps"], stats["native_trajectories"]) == (4, 1, 4), "Exactly one sequential own-CE update")
        torch = family.torch
        state_path = output / "fresh_binding_state.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "streams": streams, "manual_updates": 1}, state_path)
        saved = torch.load(state_path, map_location="cpu", weights_only=True)
        # One deliberate mutation in each saved category exercises restoration;
        # these are not optimizer steps or additional forwards.
        with torch.no_grad():
            next(model.parameters()).add_(1)
            next(value["exp_avg"] for value in optimizer.state.values() if "exp_avg" in value).add_(1)
            streams[0]["cpu"][0].bitwise_xor_(1)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        streams = copy.deepcopy(saved["streams"])
        require(exact_state(torch, model.state_dict(), saved["model"]), "Exact model state restore")
        require(exact_state(torch, optimizer.state_dict(), saved["optimizer"]), "Exact Adam state restore")
        require(exact_state(torch, streams, saved["streams"]), "Exact route RNG restore")
        model.eval()
        with torch.no_grad():
            logits = family.logits(model, streams, True, family.valid_ids)
        family.synchronize(); budget()
        require(tuple(logits.shape) == (4, 150, 5) and logits.dtype == torch.float32
                and bool(torch.isfinite(logits).all()), "Finite label-free full-graph SELECT output")
        project_imports = {}
        for name in ("models", "run_common", "sage_family_routes", "sage_family_factors",
                     "portable_physics_source_renderer", "portable_physics_sequential_own_ce",
                     "portable_local_physics_owner", "portable_local_value_installer", "endpoint_value_helper"):
            path = Path(sys.modules[name].__file__).resolve()
            require(path.is_relative_to(package), "Project import escaped portable bundle: " + name)
            project_imports[name] = str(path.relative_to(package))
        write(output / "COMPLETE_BINDING_CHECK.json", {
            "complete": True, "status": "SAME_RUNTIME_PORTABLE_BINDING_CHECK_COMPLETE",
            "package_manifest": package_receipt, "check_source": receipt(Path(__file__)),
            "input_bindings": receipt(inputs_path), "config_sha256": canonical(cfg),
            "manual_own_ce_updates": 1, "backwards": 4, "AdamW_steps": 1,
            "strict_model_Adam_route_RNG_restore": True, "SELECT_shape": [4, 150, 5], "SELECT_finite": True,
            "accuracy_NLL_or_selection_metrics_computed_or_reported": False, "CAL_TEST_target_access": False,
            "current_scientific_states_accessed": False, "original_scientific_config_modified": False,
            "acquire_fit_export_called": False, "project_imports": project_imports,
            "source_recipe_max_updates": cfg["max_updates"], "runtime": family.learning_lock["runtime"],
            "seconds_after_binding_metadata": time.monotonic() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(family.device),
            "process_lifetime_peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            "scope": "Existing same .venv binding/update/restore path only; not clean installation, external portability, full CLI or predictive evidence."})
    except BaseException as error:
        write(output / "FAILURE.json", {"complete": False, "status": "FAILED_RETAINED",
              "type": type(error).__name__, "message": str(error), "retry": False,
              "predictive_evidence": False, "external_portability_verified": False})
        raise


if __name__ == "__main__":
    main()
