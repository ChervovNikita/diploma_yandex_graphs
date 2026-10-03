"""Frozen 100-epoch fits and private validation-selected states."""
from time import perf_counter
from pilot_common import atomic_json, require, utc, file_sha
from pilot_model import make_native, make_factorized
from pilot_state import (rng_state, rng_digest, restore_rng, state_digest, snapshot,
                         restore_snapshot, write_journal, read_journal, write_selected)
from pilot_train import train_epoch
from pilot_evaluate import score_valid, mean_native_scores, hits50

ARM = {"factor_private4": "factorized_private_4", "factor_pooled4": "factorized_pooled_after_clamp_4", "native70": "native_single_70"}
MODE = {"factor_private4": "private", "factor_pooled4": "pooled_after_clamp"}


def build(context, mods, unit, base_seed, device):
    if unit == "native_bank4":
        return [make_native(mods, mods["design"].native_member_seed(base_seed, m), 64, device) for m in range(4)]
    if unit == "native70":
        return [make_native(mods, base_seed, 70, device)]
    return [make_factorized(mods, base_seed, device)]


def initialize(context, mods, unit, seed, device):
    # Capture each member's RNG immediately after its constructor. Building all
    # four then capturing global RNG would give the bank an incorrect schedule.
    fits, records = [], []
    count = 4 if unit == "native_bank4" else 1
    for member in range(count):
        fit_seed = mods["design"].native_member_seed(seed, member) if count == 4 else seed
        pair = make_native(mods, fit_seed, 64 if count == 4 else 70, device) if unit in ("native_bank4", "native70") else make_factorized(mods, seed, device)
        initial = snapshot(*pair)
        fits.append(pair)
        records.append({"fit_seed": fit_seed, "member": member, "initial_state_sha256": state_digest(initial),
                        "initial_rng_sha256": rng_digest(initial["rng"]), "current": initial,
                        "best": None, "best_state": None, "epochs": []})
    return fits, {"epoch": 0, "phase": "epochs", "fits": records,
                  "ensemble_best": None, "ensemble_best_states": None, "ensemble_candidates": [],
                  "cuda_visible_devices": context["release"]["cuda_visible_devices"],
                  "fresh_scientific_initialization": True, "resource_state_donor": False}


def select(mods, best, order, score, candidate_id):
    return mods["design"].select_validation_candidate(best, candidate_id=candidate_id, hits50=score, order=order)


def evaluate_without_rng(model, data, mods, *, mode=None):
    before = rng_digest(rng_state())
    result = score_valid(model, data, mods, mode=mode)
    require(rng_digest(rng_state()) == before, "VALID consumed the TRAIN RNG schedule")
    return result


def fit(context, mods, data, sampler, metric, device, output, unit, seed, resume, attempts):
    import torch
    if resume:
        attempts.phase("own_epoch_resume")
        state = read_journal(output, context, unit, seed)
        require(state["cuda_visible_devices"] == context["release"]["cuda_visible_devices"], "Resume device differs from own run")
        require(state["fresh_scientific_initialization"] is True and state["resource_state_donor"] is False, "Untrusted scientific donor")
        fits = build(context, mods, unit, seed, device)
        require(len(fits) == len(state["fits"]), "Resume fit cardinality differs")
        for pair, record in zip(fits, state["fits"]):
            restore_snapshot(*pair, record["current"], restore_random=False)
    else:
        attempts.phase("fresh_model_optimizer_RNG_initialization")
        fits, state = initialize(context, mods, unit, seed, device)
        write_journal(output, context, unit, seed, state)
    require(0 <= state["epoch"] <= 100 and state["phase"] in ("epochs", "complete"), "Resume epoch/phase differs")
    for epoch in range(state["epoch"] + 1, 101):
        scored = []
        for member, ((model, optimizer), record) in enumerate(zip(fits, state["fits"])):
            attempts.phase("native_TRAIN_epoch", epoch=epoch, member=member, completed_batches=0, attempted_batch=0)
            restore_rng(record["current"]["rng"])
            start_rng = rng_digest(rng_state())
            train_receipt = train_epoch(model, optimizer, data, mods, sampler,
                                       mode=MODE.get(unit), progress=attempts.progress)
            train_rng = rng_state()
            attempts.phase("complete_official_VALID", epoch=epoch, member=member)
            positive, negative, valid_receipt = evaluate_without_rng(model, data, mods, mode=MODE.get(unit))
            quality = hits50(metric, positive, negative)
            current = snapshot(model, optimizer, rng=train_rng)
            best, replace = select(mods, record["best"], epoch, quality, "epoch_" + str(epoch))
            record["best"] = best
            if replace:
                record["best_state"] = current
            record["current"] = current
            row = {"epoch": epoch, "start_rng_sha256": start_rng, "end_rng_sha256": rng_digest(train_rng),
                   "train": train_receipt, "VALID": valid_receipt, "private_hits50": quality}
            record["epochs"].append(row)
            scored.append((positive, negative, valid_receipt))
        if unit == "native_bank4":
            positive, negative = mean_native_scores(scored)
            quality = hits50(metric, positive, negative)
            best, replace = select(mods, state["ensemble_best"], epoch, quality, "same_epoch_" + str(epoch))
            state["ensemble_best"] = best
            if replace:
                state["ensemble_best_states"] = [record["current"] for record in state["fits"]]
            state["ensemble_candidates"].append({"order": epoch, "candidate_id": "same_epoch_" + str(epoch), "private_hits50": quality})
        state["epoch"] = epoch
        attempts.phase("durable_epoch_commit", epoch=epoch)
        write_journal(output, context, unit, seed, state)
        print("EPOCH_COMPLETE unit=" + unit + " base_seed=" + str(seed) + " epoch=" + str(epoch) + "/100", flush=True)
    if unit == "native_bank4" and state["phase"] != "complete":
        attempts.phase("ordered_candidate_101_individual_best_bank", epoch=100)
        scored = []
        receipts = []
        for (model, optimizer), record in zip(fits, state["fits"]):
            restore_snapshot(model, optimizer, record["best_state"])
            row = evaluate_without_rng(model, data, mods)
            scored.append(row)
            receipts.append(row[2])
        positive, negative = mean_native_scores(scored)
        quality = hits50(metric, positive, negative)
        best, replace = select(mods, state["ensemble_best"], 101, quality, "individual_validation_best_bank")
        state["ensemble_best"] = best
        if replace:
            state["ensemble_best_states"] = [record["best_state"] for record in state["fits"]]
        state["ensemble_candidates"].append({"order": 101, "candidate_id": "individual_validation_best_bank", "private_hits50": quality,
            "extra_VALID_evaluations": receipts, "member_individual_best_epochs": [record["best"]["order"] for record in state["fits"]]})
    if state["phase"] != "complete":
        state["phase"] = "complete"
        attempts.phase("durable_final_selection_commit", epoch=100)
        write_journal(output, context, unit, seed, state)
    require(all(len(record["epochs"]) == 100 for record in state["fits"]), "Complete 100-epoch family unit required")
    attempts.phase("validation_selected_checkpoint_export", epoch=100)
    if unit == "native_bank4":
        require(len(state["ensemble_candidates"]) == 101, "Incomplete frozen independent candidate bank")
        selections = {"native_single_64": (state["fits"][0]["best"], [state["fits"][0]["best_state"]]),
                      "independent_native_4": (state["ensemble_best"], state["ensemble_best_states"])}
    else:
        selections = {ARM[unit]: (state["fits"][0]["best"], [state["fits"][0]["best_state"]])}
    selected = {label: write_selected(output, context, label, unit, seed, candidate, saved)
                for label, (candidate, saved) in selections.items()}
    torch.cuda.synchronize(0)
    stream_metadata = [{"fit_seed": record["fit_seed"], "member": record["member"],
        "initial_state_sha256": record["initial_state_sha256"], "initial_rng_sha256": record["initial_rng_sha256"],
        "epochs": [{"epoch": e["epoch"], "start_rng_sha256": e["start_rng_sha256"], "end_rng_sha256": e["end_rng_sha256"],
                    "stream": e["train"]["stream"], "full_batches": e["train"]["full_batches"],
                    "optimizer_steps": e["train"]["optimizer_steps"], "train_wall_seconds": e["train"]["wall_seconds"],
                    "valid_wall_seconds": e["VALID"]["wall_seconds"], "positive_queries": e["VALID"]["positive_queries"],
                    "negative_queries": e["VALID"]["negative_queries"]} for e in record["epochs"]]} for record in state["fits"]]
    return {"schema": "ncnc-pilot-complete-unit-v1", "identity": context["identity"], "unit": unit,
        "base_seed": seed, "epochs": 100, "unique_fits": len(fits), "scientific_initialization": "fresh",
        "resource_state_donor": False, "native64_donor_reuse": unit == "native_bank4",
        "member_metadata": stream_metadata, "selected_checkpoints": selected,
        "selection_artifacts": {label: {"path": "PRIVATE_SELECTION_" + label + ".json",
            "bytes": (output / ("PRIVATE_SELECTION_" + label + ".json")).stat().st_size,
            "sha256": file_sha(output / ("PRIVATE_SELECTION_" + label + ".json"))} for label in selected},
        "independent_candidate_count": len(state["ensemble_candidates"]),
        "candidate101_extra_VALID_evaluations": 4 if unit == "native_bank4" else 0,
        "completed_optimizer_steps": len(fits) * 1700, "all_official_VALID_rows_complete": True,
        "predictive_values_exposed": False, "test_file_opened": False, "UTC": utc()}
