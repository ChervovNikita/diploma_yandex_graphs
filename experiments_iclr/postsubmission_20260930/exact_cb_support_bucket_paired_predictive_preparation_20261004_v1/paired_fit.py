"""Nine prospective cells use this same fixed100-epoch fresh-fit procedure."""
from pathlib import Path
import json
import os
import torch
from pilot_common import require, utc, file_sha
from pilot_state import (snapshot, restore_snapshot, rng_state, rng_digest, state_digest,
                         write_journal, write_selected, atomic_torch)
from pilot_evaluate import score_valid, hits50
from pilot_data import tensor_sha
from pattern_model import make_pattern
from pattern_teacher import ObservationTeacher
from paired_train import train_epoch


def fit(context, mods, data, sampler, metric, device, output, arm, seed, attempts):
    require(seed in (0, 1, 2) and arm in ("target_only", "joint", "separate"), "Outside fixed nine-cell family")
    teacher = ObservationTeacher.from_train(data["pairs"], len(data["x"]))
    model, optimizer = make_pattern(mods, seed, device)
    initial = snapshot(model, optimizer)
    state = {"arm": arm, "seed": seed, "epoch": 0, "phase": "epochs", "current": initial,
             "best": None, "best_state": None, "epochs": [], "initial_state_sha256": state_digest(initial),
             "initial_rng_sha256": rng_digest(initial["rng"]), "fresh_scientific_initialization": True,
             "resource_state_donor": False, "teacher": teacher.receipt()}
    write_journal(output, context, arm, seed, state)
    for epoch in range(1, 101):
        restore_snapshot(model, optimizer, state["current"])
        start_rng = rng_digest(rng_state())
        attempts.phase("native_TRAIN_fixed_objective", arm=arm, seed=seed, epoch=epoch, completed_batches=0)
        train = train_epoch(model, optimizer, data, mods, sampler, teacher, arm, attempts.progress)
        train_rng = rng_state()
        before = rng_digest(train_rng)
        attempts.phase("complete_official_VALID_selection", epoch=epoch)
        positive, negative, valid = score_valid(model, data, mods, mode="private")
        require(rng_digest(rng_state()) == before, "VALID consumed training RNG")
        quality = hits50(metric, positive, negative)
        current = snapshot(model, optimizer, rng=train_rng)
        best, replace = mods["design"].select_validation_candidate(state["best"],
            candidate_id="epoch_"+str(epoch), hits50=quality, order=epoch)
        state["best"] = best
        if replace:
            state["best_state"] = current
        state["current"] = current
        row = {"epoch": epoch, "start_rng_sha256": start_rng, "end_rng_sha256": before,
               "TRAIN": train, "VALID": valid, "private_hits50": quality}
        state["epochs"].append(row)
        state["epoch"] = epoch
        attempts.phase("durable_complete_epoch_commit", epoch=epoch)
        write_journal(output, context, arm, seed, state)
        with (Path(output)/"EPOCHS.jsonl").open("a") as stream:
            json.dump(row, stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        print("EPOCH_COMPLETE arm="+arm+" seed="+str(seed)+" epoch="+str(epoch)+"/100", flush=True)
    require(len(state["epochs"]) == state["epoch"] == 100, "Incomplete fixed fit")
    state["phase"] = "complete"
    write_journal(output, context, arm, seed, state)
    attempts.phase("selected_checkpoint_and_complete_VALID_replay")
    restore_snapshot(model, optimizer, state["best_state"])
    reference_rng = rng_digest(rng_state())
    positive, negative, reference = score_valid(model, data, mods, mode="private")
    quality = hits50(metric, positive, negative)
    require(quality == state["best"]["hits50"] and rng_digest(rng_state()) == reference_rng,
            "Selected serving does not reproduce selection/RNG")
    selected = write_selected(output, context, arm, arm, seed, state["best"], [state["best_state"]])
    path = Path(output)/selected["path"]
    require(path.stat().st_size == selected["bytes"] and file_sha(path) == selected["sha256"], "Selected file custody differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(payload["identity"] == context["identity"] and payload["arm"] == arm and payload["seed"] == seed
            and payload["test_stage_supported"] is False, "Selected own provenance differs")
    restored, restored_optimizer = make_pattern(mods, seed, device)
    restore_snapshot(restored, restored_optimizer, payload["states"][0])
    require(state_digest(snapshot(restored, restored_optimizer)) == state_digest(state["best_state"]),
            "Selected model/Adam/RNG/flags roundtrip differs")
    replay_rng = rng_digest(rng_state())
    rp, rn, replay = score_valid(restored, data, mods, mode="private")
    rule = context["plan"]["arithmetic_rule"]
    for actual, expected in ((rp, positive), (rn, negative)):
        require(actual.shape == expected.shape and actual.dtype == expected.dtype == torch.float32
                and bool(((actual-expected).abs() <= rule["atol"]+rule["rtol"]*expected.abs()).all()),
                "Selected complete VALID logits fail original replay rule")
    require(hits50(metric, rp, rn) == quality and rng_digest(rng_state()) == replay_rng, "Serialized replay metric/RNG differs")
    logits = atomic_torch(Path(output)/"VALID_LOGITS.pt", {"schema": "ncnc-selected-VALID-logits-v1",
        "identity": context["identity"], "arm": arm, "seed": seed, "selected_checkpoint": selected,
        "selection": state["best"], "graph": "complete_TRAIN_only", "serving_pool": "mean_raw_logits",
        "query_digests": {"positive": tensor_sha(data["valid_positive"]), "negative": tensor_sha(data["valid_negative"])},
        "positive": positive, "negative": negative, "TEST_read": False})
    def receipt(name):
        path = Path(output)/name
        return {"path": name, "bytes": path.stat().st_size, "sha256": file_sha(path)}
    return {"schema": "ncnc-pattern-complete-fit-v1", "identity": context["identity"], "arm": arm, "seed": seed,
        "epochs": 100, "optimizer_steps": 1700, "selection_candidates": 100, "selected_checkpoint": selected,
        "private_selection": receipt("PRIVATE_SELECTION_"+arm+".json"), "journal": receipt("JOURNAL.json"),
        "epoch_log": receipt("EPOCHS.jsonl"), "selected_VALID_logits": logits, "selected_VALID_score_digests": reference["score_digests"],
        "initial_state_sha256": state["initial_state_sha256"], "initial_rng_sha256": state["initial_rng_sha256"],
        "teacher": teacher.receipt(), "epoch_streams": [{"epoch": r["epoch"], "start_rng_sha256": r["start_rng_sha256"],
            "end_rng_sha256": r["end_rng_sha256"], "stream": r["TRAIN"]["stream"],
            "batches": [{k: b[k] for k in ("batch", "record_ids_sha256", "start_rng_sha256", "end_rng_sha256")}
                        for b in r["TRAIN"]["batch_receipts"]]} for r in state["epochs"]],
        "selected_roundtrip_and_full_served_replay": True, "extra_complete_VALID_replay_evaluations": 2,
        "full_VALID_evaluations": 102, "reference_replay": reference, "serialized_replay": replay,
        "predictive_values_exposed": False, "resource_state_donor": False, "test_file_opened": False, "UTC": utc()}
