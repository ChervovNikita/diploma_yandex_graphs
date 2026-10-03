"""Complete TRAIN/VALID resource checks, with no project ranking metric."""
from pilot_common import require, utc
from pilot_model import make_native, make_factorized
from pilot_state import rng_state, restore_rng, rng_digest, snapshot, state_digest
from pilot_train import train_epoch
from pilot_evaluate import score_valid, mean_native_scores
from pilot_fit import MODE
from pilot_data import tensor_sha


def run(context, mods, data, sampler, device, output, stage, unit, seed, attempts):
    import torch
    release = context["release"]
    engineering_seed = release["engineering_rng_seed"]
    require(engineering_seed == 20261003, "Engineering must use its separate root-bound seed")
    sign_seed = release["engineering_factor_sign_seed"]
    require(type(sign_seed) is int and 0 <= sign_seed < 2**63, "Explicit engineering sign seed required")
    count = 4 if unit == "native_bank4" else 1
    records, scored = [], []
    fits, initial_states = [], []
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    for member in range(count):
        attempts.phase("fresh_engineering_initialization", member=member, completed_batches=0)
        model, optimizer = (make_native(mods, engineering_seed + member, 64 if count == 4 else 70, device)
            if unit in ("native_bank4", "native70") else
            make_factorized(mods, engineering_seed, device, engineering_sign_seed=sign_seed))
        initial = snapshot(model, optimizer)
        fits.append((model, optimizer))
        initial_states.append(initial)
    for member, ((model, optimizer), initial) in enumerate(zip(fits, initial_states)):
        restore_rng(initial["rng"])
        record = {"member": member, "initial_state_sha256": state_digest(initial),
                  "initial_rng_sha256": rng_digest(initial["rng"]), "engineering_seed": engineering_seed + member}
        if stage == "resource_engineering":
            attempts.phase("complete_engineering_TRAIN_epoch", member=member, completed_batches=0, attempted_batch=0)
            record["TRAIN"] = train_epoch(model, optimizer, data, mods, sampler,
                                          mode=MODE.get(unit), progress=attempts.progress)
        attempts.phase("complete_engineering_official_VALID", member=member)
        before = rng_digest(rng_state())
        positive, negative, receipt = score_valid(model, data, mods, mode=MODE.get(unit))
        require(rng_digest(rng_state()) == before, "Engineering VALID consumed training RNG")
        require(receipt["positive_queries"] == 60084 and receipt["negative_queries"] == 100000, "Complete official VALID required")
        record["VALID"] = receipt
        record["final_rng_sha256"] = before
        records.append(record)
        scored.append((positive, negative, receipt))
    if count == 4:
        positive, negative = mean_native_scores(scored)
        require(bool(torch.isfinite(positive).all()) and bool(torch.isfinite(negative).all()), "Nonfinite bank serving pool")
        pooled = {"positive_queries": len(positive), "negative_queries": len(negative),
                  "score_digests": {"positive": tensor_sha(positive), "negative": tensor_sha(negative)}}
    else:
        pooled = records[0]["VALID"]
    torch.cuda.synchronize(0)
    return {"schema": "ncnc-pilot-complete-engineering-v1", "identity": context["identity"],
            "stage": stage, "unit": unit, "base_seed_invocation_label": seed,
            "status": "COMPLETE_FINITE", "records": records, "served_coverage": pooled,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "full_TRAIN_epochs": count if stage == "resource_engineering" else 0,
            "full_official_VALID_evaluations": count, "metric_computed": False,
            "scores_returned_or_saved": False, "resource_state_donor": False,
            "model_optimizer_or_RNG_checkpoint_saved": False, "test_file_opened": False,
            "fresh_scientific_initialization_required_for_fit": True, "UTC": utc()}
