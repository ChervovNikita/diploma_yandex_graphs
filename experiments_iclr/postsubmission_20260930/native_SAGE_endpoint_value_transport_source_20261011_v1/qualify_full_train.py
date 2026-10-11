"""Inactive disposable qualifier on actual full WikiCS TRAIN; stdlib import.

Root supplies metadata admission and a finite external watchdog. No VALID
payload is deserialized, no quality selection is called, no scientific fit or
continuation follows from PASS. All discarded states start from native owners.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time


def qualify(cfg, output, config_sha256, *, wall_seconds, cuda_bytes, rss_bytes):
    if (not math.isfinite(wall_seconds) or not 0 < wall_seconds <= 300
            or type(cuda_bytes) is not int or cuda_bytes <= 0
            or type(rss_bytes) is not int or rss_bytes <= 0):
        raise ValueError("Root must supply finite full-TRAIN wall/GPU/RSS budgets")
    source = Path(__file__).with_name("run_family.py")
    spec = importlib.util.spec_from_file_location("endpoint_value_qualification_owner", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    receipt = {"status": "RUNNING", "scope": "actual full11701node/580TRAIN native width128 graph",
        "scientific_fits": 0, "VALID_payload_deserializations": 0,
        "VALID_quality_calls": 0, "TEST_access": False, "units": [],
        "TRAIN_native_trajectories": 0, "discarded_backwards": 0, "discarded_AdamW_steps": 0,
        "limits": {"wall_seconds": wall_seconds, "cuda_bytes": cuda_bytes, "rss_bytes": rss_bytes},
        "hard_pending_call_limit": "root must enforce an external finite watchdog"}

    def budget(family=None):
        if time.perf_counter() - started > wall_seconds:
            raise RuntimeError("Full-TRAIN qualification wall budget exceeded")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if rss > rss_bytes:
            raise RuntimeError("Process-lifetime qualification RSS cap exceeded")
        if family is not None and family.device.type == "cuda":
            if family.torch.cuda.max_memory_allocated(family.device) > cuda_bytes:
                raise RuntimeError("Actual full-graph qualification CUDA cap exceeded")

    def clone(torch, value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: clone(torch, item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(clone(torch, item) for item in value)
        return value

    def equal(torch, left, right):
        if torch.is_tensor(left):
            return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(equal(torch, left[key], right[key]) for key in left)
        if isinstance(left, (tuple, list)):
            return len(left) == len(right) and all(equal(torch, x, y) for x, y in zip(left, right))
        return left == right

    def gradients(torch, parameters):
        rows = [parameter.grad for parameter in parameters if parameter.grad is not None]
        return {"available": len(rows),
            "finite": bool(rows) and all(torch.isfinite(row).all().item() for row in rows),
            "nonzero": bool(rows) and any(row.ne(0).any().item() for row in rows)}

    try:
        budget()
        family = module.Family(cfg, output, config_sha256, training_only=True)
        torch, seed = family.torch, 7301
        if (family.data.x.shape != (11701, 300) or family.train_ids.shape != (580,)
                or family.valid_ids.numel() != 0 or family.valid_y.numel() != 0):
            raise ValueError("Actual full TRAIN graph only; no toy/subgraph/VALID tensors")
        receipt["actual_graph"] = family.edge_metadata
        receipt["source_sha256"] = family.source_hashes
        receipt["safe_role_sha256"] = family.learning_lock["safe_role_sha256"]
        initial_BASE = None
        independent_models = []
        for arm in module.ARM_SPECS:
            genuine = arm["bundles"] == 4
            for member in range(arm["bundles"]):
                budget(family)
                family._operation = arm["operation"]
                members = 1 if genuine else arm["members"]
                model, optimizer, streams = family.make(seed, member, "baseline", True, members, arm["architecture"])
                body = model.body if members > 1 else model
                ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
                if len(ids) != len(set(ids)) or set(ids) != {id(parameter) for parameter in model.parameters() if parameter.requires_grad}:
                    raise ValueError("Native/builder optimizer ownership is not exact")
                added = [(name, parameter) for name, parameter in model.named_parameters() if "value_source_builder" in name or "value_target_builder" in name]
                expected_added = 0 if arm["operation"] == "none" else 2 * (8224 + 544 * members)
                if sum(parameter.numel() for _, parameter in added) != expected_added:
                    raise ValueError("Actual native-dimensional builder parameter count differs")
                if any(parameter.ne(0).any().item() for name, parameter in added if name.rsplit(".", 1)[-1] in ("weight", "bias")):
                    raise ValueError("Registered builder W/b must begin at zero")
                if any(not parameter.eq(1).all().item() for name, parameter in added if name.rsplit(".", 1)[-1] in ("r", "s")):
                    raise ValueError("Added factors must follow existing FactorLinear ones law")
                first_fast = second_fast = None
                for update in range(2):
                    model.train(); optimizer.zero_grad()
                    logits, hidden = family.training_outputs(model, streams, members > 1)
                    receipt["TRAIN_native_trajectories"] += members
                    if logits.shape != (members, 580, 10) or hidden.shape != (members, 580, 128):
                        raise ValueError("Actual full-TRAIN native output/head dimensions differ")
                    if update == 0 and arm["stem"] == "BASE":
                        initial_BASE = logits.detach().cpu().clone()
                    if update == 0 and arm["stem"] in ("E4", "N4"):
                        if initial_BASE is None or not torch.allclose(logits.detach().cpu(), initial_BASE, rtol=0, atol=1e-6):
                            raise ValueError("Identity VALUE initialization or native factor/dropout starts differ")
                    targets = family.data.y[family.train_ids] if members > 1 else family.data.y[family.data.train_mask]
                    loss = family.F.cross_entropy(logits.flatten(0, 1), targets.repeat(members))
                    if not math.isfinite(loss.item()):
                        raise ValueError("Nonfinite full-TRAIN own CE")
                    loss.backward(); receipt["discarded_backwards"] += 1
                    native = gradients(torch, [parameter for name, parameter in model.named_parameters() if "value_" not in name])
                    if not native["finite"] or not native["nonzero"]:
                        raise ValueError("Finite live native TRAIN gradient path missing")
                    if added:
                        slow = gradients(torch, [parameter for name, parameter in added if name.rsplit(".", 1)[-1] in ("weight", "bias")])
                        fast = gradients(torch, [parameter for name, parameter in added if name.rsplit(".", 1)[-1] in ("r", "s")])
                        if not slow["finite"] or not slow["nonzero"] or not fast["finite"]:
                            raise ValueError("Finite live registered builder gradient path missing")
                        if update == 0:
                            first_fast = fast
                        else:
                            second_fast = fast
                            if not fast["nonzero"]:
                                raise ValueError("Builder factors remain inactive after the first discarded AdamW step")
                    optimizer.step(); receipt["discarded_AdamW_steps"] += 1
                    family.synchronize(); budget(family)
                snapshot = {"model": clone(torch, model.state_dict()),
                            "optimizer": clone(torch, optimizer.state_dict()), "streams": clone(torch, streams)}
                state_path = output / (arm["stem"] + "_member" + str(member) + "_discarded_TRAIN_state.pt")
                torch.save(snapshot, state_path)
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.zero_()
                    for values in optimizer.state.values():
                        for value in values.values():
                            if torch.is_tensor(value):
                                value.zero_()
                streams = []
                restored = torch.load(state_path, map_location="cpu", weights_only=True)
                model.load_state_dict(restored["model"], strict=True)
                optimizer.load_state_dict(restored["optimizer"]); streams = restored["streams"]
                if not equal(torch, snapshot, restored) or not equal(torch, snapshot, {
                        "model": clone(torch, model.state_dict()), "optimizer": clone(torch, optimizer.state_dict()),
                        "streams": clone(torch, streams)}):
                    raise ValueError("Native/builder/optimizer/persistent-stream strict restoration failed")
                if genuine:
                    independent_models.append(model)
                    parameters = [parameter for item in independent_models for parameter in item.parameters()]
                    if (len({id(parameter) for parameter in parameters}) != len(parameters)
                            or len({parameter.untyped_storage().data_ptr() for parameter in parameters}) != len(parameters)):
                        raise ValueError("Genuine equipped independent native bodies/builders alias")
                receipt["units"].append({"arm": arm["stem"], "member": member, "members": members,
                    "native_parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "added_VALUE_parameters": expected_added, "identity_BASE_match": arm["stem"] in ("E4", "N4"),
                    "builder_factor_grad_first_step": first_fast, "builder_factor_grad_second_step": second_fast,
                    "strict_full_state_restore": True, "discarded_state_bytes": state_path.stat().st_size,
                    "value_installations": family._value_installations,
                    "actual_VALUE_TRAIN_operation_counts": [residual.module.conv.value_counts_train for residual in body.residual_modules]
                        if added else [], "VALID_quality_calls": 0})
                family._fit_model = family._single_adapter = None
                del optimizer, logits, hidden, loss, snapshot, restored
                if not genuine:
                    del model
                budget(family)
        if (len(receipt["units"]) != 8 or len(independent_models) != 4
                or receipt["TRAIN_native_trajectories"] != 34
                or receipt["discarded_backwards"] != 16 or receipt["discarded_AdamW_steps"] != 16):
            raise ValueError("Entire fixed eight-unit actual-TRAIN qualification must close")
        receipt["status"] = "PASS"
        receipt["scientific_launch_admitted"] = False
        receipt["elapsed_seconds"] = time.perf_counter() - started
        receipt["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(family.device) if family.device.type == "cuda" else 0
        receipt["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(family.device) if family.device.type == "cuda" else 0
        receipt["process_peak_RSS_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        module.write_json(output / "QUALIFICATION.json", receipt)
        return receipt
    except BaseException as error:
        receipt["status"] = "FAIL"
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
        receipt["elapsed_seconds"] = time.perf_counter() - started
        module.write_json(output / "QUALIFICATION.json", receipt)
        raise


if __name__ == "__main__":
    raise SystemExit("Inactive source entry. Root supplies actual full-TRAIN cfg/admission and finite budgets to qualify().")
