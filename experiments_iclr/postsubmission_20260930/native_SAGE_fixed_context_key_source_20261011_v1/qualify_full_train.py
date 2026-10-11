"""Inactive bounded full-TRAIN qualifier template. Root is the executor.

No automatic invocation, VALID scoring, TEST, scientific fit, or imported
numerical library occurs on loading this module. The caller must supply the
already reviewed normal-provider cfg and finite wall/memory budgets.
"""
import hashlib
import importlib.util
import math
from pathlib import Path
import time


def qualify(cfg, output, config_sha256, wall_seconds, cuda_bytes, rss_bytes):
    source = Path(__file__).with_name("run_family.py")
    spec = importlib.util.spec_from_file_location("fixed_key_family_qualification_v1", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Family, ARMS, CONTEXT_MODES, write_json = module.Family, module.ARMS, module.CONTEXT_MODES, module.write_json
    import resource
    import sys
    if (not math.isfinite(wall_seconds) or not 0 < wall_seconds <= 300
            or not isinstance(cuda_bytes, int) or cuda_bytes <= 0
            or not isinstance(rss_bytes, int) or rss_bytes <= 0):
        raise ValueError("Explicit finite root qualification budgets required")
    output = Path(output)
    output.mkdir(exist_ok=False)
    start = time.perf_counter()
    receipt = {"status": "RUNNING", "scope": "disposable full580TRAIN interface qualification only",
        "scientific_fits": 0, "VALID_quality_calls": 0, "TEST_access": False,
        "units": [], "diagnostic_autograd_VJPs": 0, "combined_backwards": 0,
        "optimizer_steps": 0, "native_TRAIN_trajectories": 0,
        "limits": {"wall_seconds": wall_seconds, "cuda_bytes": cuda_bytes, "rss_bytes": rss_bytes},
        "hard_wall_enforcement": "root finite owner/watchdog must also bound a pending native call"}

    def check_budget(family=None):
        if time.perf_counter() - start > wall_seconds:
            raise RuntimeError("Full-TRAIN qualification wall budget exceeded")
        current_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if current_rss > rss_bytes:
            raise RuntimeError("Process-lifetime RSS exceeds root qualification budget")
        if family is not None and family.device.type == "cuda":
            if family.torch.cuda.max_memory_allocated(family.device) > cuda_bytes:
                raise RuntimeError("CUDA qualification budget exceeded")

    def clone_tree(torch, value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: clone_tree(torch, v) for k, v in value.items()}
        if isinstance(value, list):
            return [clone_tree(torch, v) for v in value]
        if isinstance(value, tuple):
            return tuple(clone_tree(torch, v) for v in value)
        return value

    def equal_tree(torch, left, right):
        if torch.is_tensor(left):
            return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(equal_tree(torch, left[k], right[k]) for k in left)
        if isinstance(left, (tuple, list)):
            return len(left) == len(right) and all(equal_tree(torch, a, b) for a, b in zip(left, right))
        return left == right

    def finite_nonzero(torch, values):
        available = [v for v in values if v is not None]
        return bool(available) and all(torch.isfinite(v).all().item() for v in available) and any(v.ne(0).any().item() for v in available)

    try:
        check_budget()
        family = Family(cfg, output, config_sha256)
        torch = family.torch
        check_budget(family)
        receipt["source_sha256"] = family.source_hashes
        receipt["target_preparation"] = family.context_preparation
        receipt["target_objective_contract"] = family.context_contract
        # Both actual orders come through the production tensor-bank entry.
        for order_name, buffers in family.context_targets.items():
            ids, labels = buffers["train_ids"], buffers["train_y"]
            expected_ids = family.train_ids if order_name == "bank" else family.data.train_mask.nonzero().flatten()
            if not torch.equal(ids, expected_ids) or not torch.equal(labels, family.data.y[expected_ids]):
                raise ValueError("Actual H/role/key row order mismatch")
            allowed = ids[:, None].ne(ids[None, :])
            positive = labels[:, None].eq(labels[None, :]) & allowed
            if not torch.equal(allowed, buffers["full_allowed"]) or not torch.equal(positive, buffers["positive"]):
                raise ValueError("Key denominator/self/class support mismatch")
            if any(v.requires_grad or not torch.isfinite(v).all().item() for v in buffers.values()):
                raise ValueError("Frozen finite target/key/moment/order buffers required")
            permutation = buffers["shuffle_permutation"]
            if (not torch.equal(labels[permutation], labels)
                    or not torch.equal(buffers["shuffled_weights"],
                                       buffers["weights"][:, permutation[:, None], permutation[None, :]])):
                raise ValueError("Both-axis class-preserving Q shuffle mismatch")
            for q_name, moment_name in [("weights", "moments"), ("shuffled_weights", "shuffled_moments"),
                                        ("uniform_weights", "uniform_moments")]:
                q = buffers[q_name]
                if q.ndim == 2:
                    q = q[None].expand(4, -1, -1)
                if (q[:, ~positive].ne(0).any().item() or q[:, positive].le(0).any().item()
                        or not torch.allclose(q.sum(-1), torch.ones_like(q.sum(-1)), rtol=0, atol=1e-6)
                        or not torch.equal(torch.bmm(q, buffers["keys"]), buffers[moment_name])):
                    raise ValueError("Actual Q support/rowmass/unnormalized moment reduction mismatch")
            norms = buffers["keys"].norm(dim=-1)
            if not (((norms - 1).abs() <= 1e-6) | norms.eq(0)).all().item():
                raise ValueError("Fixed normalized keys/zero-key policy mismatch")
        receipt["buffer_qualification_extra_moment_matrix_products"] = 24
        receipt["buffer_qualification_extra_moment_MACs"] = 24 * 580 * 580 * 300
        seed = 7301
        initial_query_sha256 = None
        for arm in ARMS:
            genuine = arm in ("genuinefactorizedI4_key_mixture", "genuinefactorizedI4_key_assigned")
            bank = arm.startswith("shared4_")
            factorized = arm != "ordinaryM1_key_mixture"
            mode = CONTEXT_MODES[arm]
            for member in range(4 if genuine else 1):
                check_budget(family)
                model, optimizer, streams = family.make(seed, member, "baseline", factorized, 4 if bank else 1)
                native_before_query = clone_tree(torch, model.state_dict())
                cpu_rng = torch.get_rng_state().clone()
                cuda_rng = torch.cuda.get_rng_state(family.device).clone() if family.device.type == "cuda" else None
                query = family.context_objective.make_query_map(torch, family.F, seed, family.device)
                query_sha = hashlib.sha256(query.weight.detach().cpu().numpy().tobytes(order="C")).hexdigest()
                if initial_query_sha256 is None:
                    initial_query_sha256 = query_sha
                if (query_sha != initial_query_sha256 or not torch.equal(cpu_rng, torch.get_rng_state())
                        or cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(family.device))
                        or not equal_tree(torch, native_before_query, clone_tree(torch, model.state_dict()))):
                    raise ValueError("A initialization changed native/RNG bytes or differs across matched arms")
                optimizer.add_param_group({"params": list(query.parameters()), "lr": .001, "weight_decay": 0})
                parameters = list(model.parameters()) + list(query.parameters())
                owners = [id(v) for group in optimizer.param_groups for v in group["params"]]
                if len(owners) != len(parameters) or set(owners) != {id(v) for v in parameters}:
                    raise ValueError("Native/A optimizer ownership mismatch")
                model.train(); query.train(); optimizer.zero_grad()
                logits, hidden = family.training_outputs(model, streams, bank)
                members = 4 if bank else 1
                receipt["native_TRAIN_trajectories"] += members
                target_order = "bank" if bank else "single"
                buffers = family.context_targets[target_order]
                context_index = member if arm == "genuinefactorizedI4_key_assigned" else None
                own_y = family.data.y[family.train_ids] if bank else family.data.y[family.data.train_mask]
                own = family.F.cross_entropy(logits.flatten(0, 1), own_y.repeat(members))
                auxiliary, routes, pairs, operations = family.context_objective.context_auxiliary(
                    torch, family.F, hidden, buffers, mode, context_index=context_index, query_map=query)
                # These two diagnostic VJPs exercise actual live H/A and native
                # slow/fast pullbacks; no extra native forward or oracle selector.
                hidden_grad, query_grad = torch.autograd.grad(auxiliary, (hidden, query.weight), retain_graph=True)
                named_native = list(model.named_parameters())
                native_aux = torch.autograd.grad(auxiliary, [v for _, v in named_native],
                                                 allow_unused=True, retain_graph=True)
                receipt["diagnostic_autograd_VJPs"] += 2
                slow_aux = [g for (name, _), g in zip(named_native, native_aux) if name.rsplit(".", 1)[-1] not in ("r", "s")]
                fast_aux = [g for (name, _), g in zip(named_native, native_aux) if name.rsplit(".", 1)[-1] in ("r", "s")]
                if (not finite_nonzero(torch, [hidden_grad]) or not finite_nonzero(torch, [query_grad])
                        or not finite_nonzero(torch, slow_aux)
                        or factorized and not finite_nonzero(torch, fast_aux)):
                    raise ValueError("Required finite live query/native/fast auxiliary gradient path missing")
                loss = own + .05 * auxiliary
                if not math.isfinite(loss.item()) or not math.isfinite(auxiliary.item()):
                    raise ValueError("Nonfinite full-TRAIN objective")
                loss.backward(); receipt["combined_backwards"] += 1
                if not finite_nonzero(torch, [v.grad for v in model.parameters()]) or not finite_nonzero(torch, [query.weight.grad]):
                    raise ValueError("Combined native/query gradient path missing")
                optimizer.step(); receipt["optimizer_steps"] += 1
                snapshot = {"model": clone_tree(torch, model.state_dict()),
                    "query_map": clone_tree(torch, query.state_dict()),
                    "optimizer": clone_tree(torch, optimizer.state_dict()), "streams": clone_tree(torch, streams),
                    "context_index": context_index, "mode": mode,
                    "target_tensor_hashes": family.context_preparation["actual_loss_tensor_hashes"][target_order],
                    "objective_contract": family.context_contract, "query_initial_sha256": query_sha}
                snapshot_path = output / (arm + "_member" + str(member) + "_TRAIN_only_snapshot.pt")
                torch.save(snapshot, snapshot_path)
                # Make restore meaningful without an extra predictive trajectory.
                with torch.no_grad():
                    for parameter in parameters:
                        parameter.zero_()
                    for values in optimizer.state.values():
                        for value in values.values():
                            if torch.is_tensor(value):
                                value.zero_()
                streams = []
                restored = torch.load(snapshot_path, map_location="cpu", weights_only=True)
                model.load_state_dict(restored["model"], strict=True)
                query.load_state_dict(restored["query_map"], strict=True)
                optimizer.load_state_dict(restored["optimizer"])
                streams = restored["streams"]
                if (not equal_tree(torch, snapshot, restored)
                        or not equal_tree(torch, snapshot["model"], clone_tree(torch, model.state_dict()))
                        or not equal_tree(torch, snapshot["query_map"], clone_tree(torch, query.state_dict()))
                        or not equal_tree(torch, snapshot["optimizer"], clone_tree(torch, optimizer.state_dict()))
                        or not equal_tree(torch, snapshot["streams"], clone_tree(torch, streams))):
                    raise ValueError("Native/A/optimizer/stream full-state restoration mismatch")
                receipt["units"].append({"arm": arm, "member": member, "seed": seed,
                    "mode": mode, "context_index": context_index, "TRAIN_H_order": target_order,
                    "initial_query_sha256": query_sha, "native_parameters": sum(v.numel() for v in model.parameters()),
                    "training_query_parameters": query.weight.numel(), "serving_query_parameters": 0,
                    "query_and_hidden_aux_gradient_nonzero": True, "slow_aux_gradient_nonzero": True,
                    "fast_aux_gradient_nonzero": True if factorized else None,
                    "native_parameter_aux_gradient_present": [name for (name, _), g in zip(named_native, native_aux) if g is not None],
                    "operations": operations, "TRAIN_own_CE": own.item(), "TRAIN_auxiliary": auxiliary.item(),
                    "TRAIN_combined": loss.item(), "route_losses": routes.detach().tolist(),
                    "pair_losses": pairs.detach().tolist(), "snapshot_bytes": snapshot_path.stat().st_size,
                    "full_state_restore_equal": True})
                family.synchronize(); check_budget(family)
                del model, query, optimizer, logits, hidden, own, auxiliary, loss
                del native_aux, hidden_grad, query_grad, parameters, native_before_query, snapshot, restored
                del routes, pairs, named_native, slow_aux, fast_aux
        if (len(receipt["units"]) != 16 or receipt["native_TRAIN_trajectories"] != 34
                or receipt["combined_backwards"] != 16 or receipt["optimizer_steps"] != 16):
            raise ValueError("Complete ten-arm16unit34trajectory qualifier roster missing")
        receipt["status"] = "PASS"
        receipt["scientific_launch_admitted"] = False
        receipt["query_map_initial_sha256_same_all_units"] = initial_query_sha256
        receipt["elapsed_seconds"] = time.perf_counter() - start
        write_json(output / "QUALIFICATION.json", receipt)
        return receipt
    except BaseException as error:
        receipt["status"] = "FAIL"
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
        receipt["elapsed_seconds"] = time.perf_counter() - start
        write_json(output / "QUALIFICATION.json", receipt)
        raise


if __name__ == "__main__":
    raise SystemExit("Inactive source template; root supplies reviewed cfg/provider/budgets and invokes qualify explicitly")
