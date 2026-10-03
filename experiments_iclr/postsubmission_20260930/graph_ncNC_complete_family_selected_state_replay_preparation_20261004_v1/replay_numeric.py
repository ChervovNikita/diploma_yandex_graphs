"""Deferred numerical replay using only the authenticated original scoring APIs."""
from time import perf_counter
from pathlib import Path
import importlib
import sys
from replay_gate import require, bound, ARMS
from replay_contract import validate_journal_payload, validate_selected_payload, validate_valid_receipt


def original_modules(context):
    root = context["paths"]["driver_root"]
    sys.path.insert(0, str(root))
    result = {}
    for name in ("pilot_common", "pilot_model", "pilot_data", "pilot_state", "pilot_evaluate"):
        module = importlib.import_module(name)
        require(Path(module.__file__).resolve() == root / (name + ".py"), "Original driver module shadowed")
        result[name] = module
    return result


def trusted_load(torch, root, pin):
    """Reauthenticate the open file, then deserialize that same authenticated FD."""
    from hashlib import sha256
    path = bound(pin, root=root)
    with path.open("rb") as handle:
        digest = sha256()
        size = 0
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
        require(size == pin["bytes"] and digest.hexdigest() == pin["sha256"], "Open checkpoint custody changed")
        handle.seek(0)
        return torch.load(handle, map_location="cpu", weights_only=False)


def run(context, gate, accounting):
    require(gate["full_success"] and gate["unique_fits_completed"] == 35 and gate["complete_served_cells"] == 25, "Full immutable35/25 gate required before numerical imports")
    api = original_modules(context)
    common, model, data_api, state_api, evaluate = (api[n] for n in ("pilot_common", "pilot_model", "pilot_data", "pilot_state", "pilot_evaluate"))
    accounting.phase("original_stdlib_interpreter_distribution_admission")
    common.runtime_stdlib(context)
    accounting.phase("original_normal_CUDA_source_binary_admission")
    device, _ = model.runtime(context)
    import torch
    accounting.cuda_started = True
    torch.cuda.reset_peak_memory_stats(0)
    accounting.phase("original_qualified_model_source_imports")
    mods = model.modules(context)
    mods["design"].validate_plan(context["plan"])
    select = mods["design"].select_validation_candidate
    accounting.phase("all20_own_journal_all25_selected_snapshot_contracts")
    prepared = {}
    load_failures = []
    for (unit, seed), row in sorted(gate["units"].items()):
        try:
            payload = trusted_load(torch, row["root"], row["journal"]["state_file"])
            state = validate_journal_payload(payload, row["journal"], row["complete"], context["identity"], unit, seed, select)
            if unit == "native_bank4":
                cost = next(c for c in gate["costs"] if (c["unit"], c["base_seed"]) == (unit, seed))
                cost["candidate101_extra_VALID_wall_seconds"] = sum(r["wall_seconds"] for r in state["ensemble_candidates"][-1]["extra_VALID_evaluations"])
            for cell in [c for c in gate["cells"] if (c["unit"], c["base_seed"]) == (unit, seed)]:
                selected = trusted_load(torch, row["root"], cell["checkpoint"])
                saved, digests = validate_selected_payload(selected, cell, state, context["identity"], state_api.state_digest)
                prepared[(cell["arm"], seed)] = {"snapshots": saved, "score_digests": digests}
            del state, payload
        except Exception as error:
            load_failures.append({"unit": unit, "base_seed": seed, "exception_type": type(error).__name__, "condition": str(error)})
            for cell in gate["cells"]:
                if (cell["unit"], cell["base_seed"]) == (unit, seed):
                    cell["status"] = "FAILED_SELECTED_STATE_CONTRACT"
    if load_failures:
        accounting.private({"selected_state_failures": load_failures})
        return gate["cells"], {"status": "FAILED_SELECTED_STATE_CONTRACT", "score_valid_calls": 0, "no_partial_replay": True}
    require(len(prepared) == 25, "All25 snapshots must validate before any scoring")
    accounting.phase("original_authenticated_complete_TRAIN_raw_VALID_load_transfer")
    data = data_api.load_data(context, device)
    metric = evaluate.evaluator(context)
    accounting.phase("all25_selected_state_replays")
    total_calls = 0
    private_rows = []
    for index, cell in enumerate(gate["cells"]):
        arm, seed = cell["arm"], cell["base_seed"]
        item = prepared[(arm, seed)]
        started = perf_counter()
        current = {"arm": arm, "base_seed": seed, "score_valid_calls": 0, "member_receipts": []}
        model_instance = optimizer = None
        try:
            scores = []
            for member, (saved, score_digest) in enumerate(zip(item["snapshots"], item["score_digests"])):
                construction = perf_counter()
                if arm in (ARMS[0], ARMS[1], ARMS[4]):
                    model_instance, optimizer = model.make_native(mods, seed + 5 * member if arm == ARMS[1] else seed, 70 if arm == ARMS[4] else 64, device)
                else:
                    model_instance, optimizer = model.make_factorized(mods, seed, device)
                state_api.restore_snapshot(model_instance, optimizer, saved, restore_random=True)
                restored = state_api.snapshot(model_instance, optimizer)
                require(state_api.state_digest(restored) == state_api.state_digest(saved), "Strict restored complete model/Adam/flags/RNG digest differs")
                fixed_digest = state_api.state_digest({"models": restored["models"], "optimizer": restored["optimizer"]})
                restore_wall = perf_counter() - construction
                before = state_api.rng_digest(state_api.rng_state())
                total_calls += 1
                current["score_valid_calls"] += 1
                mode = "private" if arm == ARMS[2] else "pooled_after_clamp" if arm == ARMS[3] else None
                positive, negative, receipt = evaluate.score_valid(model_instance, data, mods, mode=mode)
                validate_valid_receipt(receipt)
                require(state_api.rng_digest(state_api.rng_state()) == before, "Replay scoring changed restored fit RNG")
                after = state_api.snapshot(model_instance, optimizer, rng=saved["rng"])
                require(state_api.state_digest({"models": after["models"], "optimizer": after["optimizer"]}) == fixed_digest, "Replay scoring changed model/optimizer state")
                receipt["construction_restore_transfer_digest_wall_seconds"] = restore_wall
                receipt["original_score_digest_equal"] = receipt["score_digests"] == score_digest
                current["member_receipts"].append(receipt)
                require(receipt["original_score_digest_equal"], "Original per-model raw score digest mismatch")
                scores.append((positive, negative, receipt))
                del model_instance, optimizer, restored, after
                model_instance = optimizer = None
            positive, negative = evaluate.mean_native_scores(scores) if arm == ARMS[1] else scores[0][:2]
            replayed = evaluate.hits50(metric, positive, negative)
            current["replayed_VALID_hits50"] = replayed
            current["selected_hits50_equal"] = replayed == cell["selection"]["hits50"]
            require(current["selected_hits50_equal"], "Exact selected Hits50 replay mismatch; no tolerance admitted")
            cell.update(status="PASS", replayed_VALID_hits50=replayed, selected_hits50_equal=True)
            current["status"] = "PASS"
        except Exception as error:
            cell["status"] = "FAILED_REPLAY"
            current.update(status="FAILED_REPLAY", failure={"exception_type": type(error).__name__, "condition": str(error)})
        finally:
            if model_instance is not None:
                del model_instance, optimizer
            current["inclusive_cell_wall_seconds"] = perf_counter() - started
            private_rows.append(current)
            accounting.private({"cells": private_rows, "all25_required": True, "aggregate_contrasts": None})
            accounting.progress(index + 1, total_calls)
    full = all(c["status"] == "PASS" for c in gate["cells"])
    require(not full or total_calls == 40, "Successful all25 replay call denominator differs")
    return gate["cells"], {"status": "PASS" if full else "FAILED_REPLAY", "score_valid_calls": total_calls, "nominal_score_valid_calls": 40, "original_pooled_I4_score_digest_available": False, "per_model_score_digests_checked": True, "deterministic_algorithms": False, "no_tolerance_or_retry": True}
