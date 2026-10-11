"""Inactive full-graph qualification entry; root supplies the admitted factory.

Two discarded update pairs per native unit compare sequential vs simultaneous
at identical states. No fitting, quality selection, VALID/CAL/TEST labels or
automatic continuation. Caller enforces the external pending-call watchdog.
"""
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time


def clone(torch, value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone(torch, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone(torch, v) for v in value)
    return value


def compare(torch, left, right, *, atol=0., rtol=0.):
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.shape != right.shape or left.dtype != right.dtype:
            return False
        if left.is_floating_point():
            return torch.isfinite(left).all().item() and torch.isfinite(right).all().item() and torch.allclose(left, right.cpu(), atol=atol, rtol=rtol)
        return torch.equal(left, right.cpu())
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(compare(torch, left[k], right[k], atol=atol, rtol=rtol) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(compare(torch, a, b, atol=atol, rtol=rtol) for a, b in zip(left, right))
    return left == right


def snapshot(family, model, optimizer, streams):
    torch = family.torch
    state = {"model": clone(torch, model.state_dict()), "optimizer": clone(torch, optimizer.state_dict()),
             "streams": clone(torch, streams), "cpu_rng": torch.get_rng_state().clone()}
    if family.device.type == "cuda":
        state["cuda_rng"] = torch.cuda.get_rng_state(family.device).cpu().clone()
    return state


def restore(family, model, optimizer, state):
    model.load_state_dict(state["model"], strict=True)
    # Adam step tensors may stay on CPU; prevent load_state_dict from sharing
    # those tensors with the immutable common-state/selected snapshot.
    optimizer.load_state_dict(clone(family.torch, state["optimizer"]))
    family.torch.set_rng_state(state["cpu_rng"])
    if family.device.type == "cuda":
        family.torch.cuda.set_rng_state(state["cuda_rng"], family.device)
    return clone(family.torch, state["streams"])


def qualify(make_family, selector_ids, output, *, seed, limits, tolerances, release):
    """Root factory constructs training_only=True on the complete safe TRAIN graph.

    selector_ids are root-bound label-free SELECT membership only. Tolerances
    are prospectively fixed engineering bounds, never predictive criteria.
    """
    if release.get("status") != "ROOT_FULL_TRAIN_QUALIFICATION_ADMITTED":
        raise ValueError("Inactive source: qualification root release required")
    if set(limits) != {"wall_seconds", "cuda_bytes", "rss_bytes"} or any(type(v) is not int or v <= 0 for v in limits.values()):
        raise ValueError("Finite root limits required")
    if set(tolerances) != {"gradient_atol", "gradient_rtol", "update_atol", "update_rtol"} or any(not math.isfinite(v) or v < 0 for v in tolerances.values()):
        raise ValueError("Root must freeze all numerical equivalence tolerances")
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    receipt = {"status": "RUNNING", "units": [], "scientific_fits": 0, "quality_selector_calls": 0,
               "VALID_CAL_TEST_label_access": False, "automatic_fit_release": False,
               "limits": limits, "tolerances": tolerances, "discarded_backwards": 0,
               "discarded_AdamW_steps": 0, "TRAIN_native_trajectories": 0,
               "cuda_peak_allocated_bytes": 0, "cuda_peak_reserved_bytes": 0,
               "scope": "one complete official ogbn-arxiv safe graph; eight native units, two common-state update pairs each"}
    def budget(family=None):
        if time.perf_counter() - started > limits["wall_seconds"]:
            raise RuntimeError("Finite qualification wall cap exceeded")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if rss > limits["rss_bytes"]:
            raise RuntimeError("Qualification lifetime RSS cap exceeded")
        if family is not None and family.device.type == "cuda":
            t = family.torch
            peak = t.cuda.max_memory_allocated(family.device)
            receipt["cuda_peak_allocated_bytes"] = max(receipt["cuda_peak_allocated_bytes"], peak)
            receipt["cuda_peak_reserved_bytes"] = max(receipt["cuda_peak_reserved_bytes"], t.cuda.max_memory_reserved(family.device))
            if peak > limits["cuda_bytes"]:
                raise RuntimeError("Qualification complete-graph CUDA cap exceeded")
    try:
        budget()
        family = make_family(output)
        torch, profile = family.torch, family.cfg["profile"]
        if (not family.training_only or family.data.x.shape != (169343, 128)
                or family.train_ids.numel() != profile["train_rows"]
                or family.valid_ids.numel() or family.valid_y.numel()
                or seed not in family.cfg["seeds"]):
            raise ValueError("Root-bound actual complete TRAIN only; no miniature case")
        if (selector_ids.dtype != torch.long or selector_ids.device != family.device
                or selector_ids.shape != (profile["select_rows"],)
                or selector_ids.unique().numel() != selector_ids.numel()
                or selector_ids.min().item() < 0 or selector_ids.max().item() >= 169343
                or torch.isin(selector_ids, family.train_ids).any().item()):
            raise ValueError("Complete root-bound disjoint label-free SELECT IDs required")
        path = Path(__file__).with_name("own_ce.py")
        spec = importlib.util.spec_from_file_location("qualification_own_ce", path)
        own = importlib.util.module_from_spec(spec); spec.loader.exec_module(own)
        arms = (("BASE", "none", 4, 1), ("E4", "endpoint", 4, 1), ("N4", "node", 4, 1),
                ("equipped_M1", "endpoint", 1, 1), ("equipped_I4", "endpoint", 1, 4))
        independent = []
        receipt["profile"] = profile; receipt["source_sha256"] = family.source_hashes
        for arm, operation, members, bundles in arms:
            for member in range(bundles):
                budget(family); family._operation = operation
                model, optimizer, streams = family.make(seed, member, "baseline", True, members,
                                                         "shared" if members == 4 else "single")
                added = [(n, p) for n, p in model.named_parameters() if "value_source_builder" in n or "value_target_builder" in n]
                if sum(p.numel() for n, p in added) != (0 if operation == "none" else 2 * (8224 + 544 * members)):
                    raise ValueError("Native width128 builder ownership/count differs")
                if any(not p.eq(0 if n.rsplit('.', 1)[-1] in ('weight', 'bias') else 1).all().item() for n, p in added):
                    raise ValueError("Exact builder zero/ones initialization differs")
                unit = {"arm": arm, "member": member, "members": members, "pairs": []}
                for step in range(2):
                    common = snapshot(family, model, optimizer, streams)
                    ends, gradients = {}, {}
                    for mode in ("sequential", "simultaneous"):
                        streams = restore(family, model, optimizer, common)
                        if family.device.type == "cuda":
                            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(family.device)
                        family.synchronize(); tick = time.perf_counter()
                        result = own.update(family, model, optimizer, streams, members == 4,
                                            mode=mode, qualification=True, capture_gradients=True)
                        family.synchronize(); budget(family)
                        gradients[mode] = result.pop("gradients")
                        if any(g is None or not torch.isfinite(g).all().item() for g in gradients[mode].values()):
                            raise ValueError("All registered gradients must exist and be finite")
                        native = [g for n, g in gradients[mode].items() if 'value_' not in n]
                        if not any(g.ne(0).any().item() for g in native):
                            raise ValueError("Native own-CE gradient path is inactive")
                        if added:
                            slow = [g for n, g in gradients[mode].items() if 'value_' in n and n.rsplit('.', 1)[-1] in ('weight', 'bias')]
                            fast = [g for n, g in gradients[mode].items() if 'value_' in n and n.rsplit('.', 1)[-1] in ('r', 's')]
                            if not any(g.ne(0).any().item() for g in slow) or (step == 1 and not any(g.ne(0).any().item() for g in fast)):
                                raise ValueError("Live builder/second-step private-factor gradients missing")
                        ends[mode] = snapshot(family, model, optimizer, streams)
                        if compare(torch, common['model'], ends[mode]['model']):
                            raise ValueError("Discarded AdamW step did not change native/builder state")
                        if not optimizer.state or any(v['step'].item() != step + 1 for v in optimizer.state.values()):
                            raise ValueError("Exactly one AdamW moment/update advancement required")
                        unit["pairs"].append(dict(result, update=step + 1, seconds=time.perf_counter() - tick,
                            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(family.device) if family.device.type == 'cuda' else 0))
                        for key, source in (("discarded_backwards", "backwards"), ("discarded_AdamW_steps", "adam_steps"), ("TRAIN_native_trajectories", "native_trajectories")):
                            receipt[key] += result[source]
                    if not compare(torch, gradients['sequential'], gradients['simultaneous'], atol=tolerances['gradient_atol'], rtol=tolerances['gradient_rtol']):
                        raise ValueError("Complete-TRAIN accumulated gradients differ beyond fixed bounds")
                    for key in ('model', 'optimizer'):
                        if not compare(torch, ends['sequential'][key], ends['simultaneous'][key], atol=tolerances['update_atol'], rtol=tolerances['update_rtol']):
                            raise ValueError("One AdamW update/state differs beyond fixed bounds")
                    for key in ('streams', 'cpu_rng') + (('cuda_rng',) if family.device.type == 'cuda' else ()):
                        if not compare(torch, ends['sequential'][key], ends['simultaneous'][key]):
                            raise ValueError("Persistent/global RNG differs")
                    streams = restore(family, model, optimizer, ends['sequential'])
                    del common, ends, gradients
                selected = snapshot(family, model, optimizer, streams)
                state_path = output / f'{arm}_member{member}_discarded.pt'
                torch.save(selected, state_path)
                model.eval(); family.synchronize(); tick = time.perf_counter()
                with torch.no_grad():
                    logits = family.logits(model, streams, members == 4, selector_ids)
                    probabilities = logits.softmax(-1).mean(0)
                    if logits.shape != (members, profile['select_rows'], 40) or not torch.isfinite(probabilities).all().item():
                        raise ValueError("Complete-graph label-free selector/serving shape or finiteness differs")
                    expected = logits.detach().cpu().clone()
                family.synchronize(); budget(family)
                unit['selector_probability_serving_seconds'] = time.perf_counter() - tick
                del logits, probabilities
                with torch.no_grad():
                    for p in model.parameters(): p.add_(1)
                    for values in optimizer.state.values():
                        for value in values.values():
                            if torch.is_tensor(value): value.add_(1)
                streams = []; torch.manual_seed(997)
                restored = torch.load(state_path, map_location='cpu', weights_only=True)
                streams = restore(family, model, optimizer, restored)
                if not compare(torch, selected, snapshot(family, model, optimizer, streams)):
                    raise ValueError("Meaningful native/builder/moment/stream/global-RNG restoration failed")
                model.eval()
                with torch.no_grad(): replay = family.logits(model, streams, members == 4, selector_ids).cpu()
                if not compare(torch, expected, replay, atol=tolerances['update_atol'], rtol=tolerances['update_rtol']):
                    raise ValueError("Selected label-free serving does not restore")
                family.synchronize(); budget(family)
                if arm == 'equipped_I4':
                    independent.append(model)
                    params = [p for body in independent for p in body.parameters()]
                    if len({id(p) for p in params}) != len(params) or len({p.untyped_storage().data_ptr() for p in params}) != len(params):
                        raise ValueError("Genuine independently initialized bodies/builders alias")
                unit.update(strict_restore=True, gradients_and_one_AdamW_state_compared=True,
                            persistent_and_global_RNG_equal=True, label_free_selector_native_trajectories=2 * members,
                            actual_VALUE_TRAIN_operation_counts=[r.module.conv.value_counts_train for r in (model.body if members == 4 else model).residual_modules] if added else [])
                receipt['units'].append(unit)
                family._fit_model = family._single_adapter = None
                del optimizer, selected, restored, replay, expected
                if arm != 'equipped_I4': del model
        if len(receipt['units']) != 8 or receipt['discarded_backwards'] != 50 or receipt['discarded_AdamW_steps'] != 32 or receipt['TRAIN_native_trajectories'] != 68:
            raise ValueError("Complete fixed qualification did not close")
        receipt['status'] = 'PASS'
        receipt['elapsed_seconds'] = time.perf_counter() - started
        receipt['process_lifetime_peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    except BaseException as error:
        receipt['status'] = 'FAIL'; receipt['failure'] = {'type': type(error).__name__, 'message': str(error)}
        (output / 'QUALIFICATION.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
        raise
    (output / 'QUALIFICATION.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    return receipt


if __name__ == '__main__':
    raise SystemExit('Inactive source. Root supplies the admitted full-TRAIN factory, SELECT IDs, fixed tolerances and budgets.')
