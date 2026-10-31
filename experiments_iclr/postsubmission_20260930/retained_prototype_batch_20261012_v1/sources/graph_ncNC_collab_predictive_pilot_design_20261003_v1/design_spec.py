"""Prospective pilot metadata helpers only; no model, data or fit imports."""
from hashlib import sha256
from math import isfinite

BASE_SEEDS = (0, 1, 2, 3, 4)
CORE_ARMS = ("native_single_64", "independent_native_4", "factorized_private_4", "factorized_pooled_after_clamp_4")
EPOCHS = 100
TRAIN_BATCH_SIZE = 65536
EVAL_BATCH_SIZE = 131072


def native_member_seed(base_seed, member):
    if base_seed not in BASE_SEEDS or member not in range(4):
        raise ValueError("Outside frozen five-block/four-member design")
    return base_seed + 5 * member


def factor_sign_seed(base_seed):
    if base_seed not in BASE_SEEDS:
        raise ValueError("Outside frozen five-block design")
    label = f"ncnc-collab-pilot-v1|base={base_seed}|domain=factor-signs"
    return int.from_bytes(sha256(label.encode("ascii")).digest()[:8], "little") % (2**31 - 1)


def native_parameter_count(width, features=128):
    # GCN/LN encoder + native decoder, including fixed-pt's unused ptlin.
    return 7 * width**2 + (features + 20) * width + 3


def native_active_parameter_count(width, features=128):
    return native_parameter_count(width, features) - (width**2 + 2 * width + 1)


def factorized_parameter_count(width=64, members=4, features=128):
    # All shared W/bias, private factors/LN/beta; includes unused ptlin.
    return 7 * width**2 + (features + 12) * width + 2 + members * (24 * width + 3)


def factorized_active_parameter_count(width=64, members=4, features=128):
    unused = width**2 + 2 * width + 1 + members * (3 * width + 1)
    return factorized_parameter_count(width, members, features) - unused


def minimum_no_smaller_native_width():
    total = factorized_parameter_count()
    active = factorized_active_parameter_count()
    width = 1
    while native_parameter_count(width) < total or native_active_parameter_count(width) < active:
        width += 1
    return width


def select_validation_candidate(best, *, candidate_id, hits50, order):
    """Strict served Hits@50 improvement; first exact ties are retained.

    Caller evaluates canonical positive/negative scores and persists complete
    state on every replacement, including the first candidate. TEST is absent.
    Independent ensemble candidates have order1..100, then101 for the fixed
    bank assembled from the four individual native validation-best states.
    """
    if not isfinite(hits50) or not 0 <= hits50 <= 1 or order < 1:
        raise ValueError("Invalid complete validation metric/candidate order")
    if best is not None and order <= best["order"]:
        raise ValueError("Candidates must be evaluated in frozen order")
    current = {"candidate_id": candidate_id, "hits50": hits50, "order": order}
    if best is None or hits50 > best["hits50"]:
        return current, True
    return best, False


def validate_plan(plan):
    if tuple(plan["base_seeds"]) != BASE_SEEDS or tuple(plan["core_arms"]) != CORE_ARMS:
        raise ValueError("Frozen seeds/arms differ")
    if plan["epochs"] != EPOCHS or plan["train_batch_size"] != TRAIN_BATCH_SIZE or plan["eval_batch_size"] != EVAL_BATCH_SIZE:
        raise ValueError("Native budget/batches differ")
    if plan["prototype_manifest_sha256"] != "a99b0e3b0e8ec597b03e2c390ac176597b5b6422d365fc20624ab98a113d42f9":
        raise ValueError("Qualified prototype differs")
    if plan["auxiliary_native_single"]["width"] != minimum_no_smaller_native_width():
        raise ValueError("Capacity control is smaller than target or not minimal")
    if plan["auxiliary_native_single"]["fit_count"] != 5 or plan["work_budget"]["full_family_unique_fits"] != 35:
        raise ValueError("Required width70 auxiliary/full family differs")
    if plan["scientific_initialization_donors"]["resource_epoch_states_permitted"] is not False:
        raise ValueError("Resource qualification states cannot become scientific donors")
    all_seeds = [native_member_seed(s, m) for s in BASE_SEEDS for m in range(4)]
    if len(set(all_seeds)) != 20:
        raise ValueError("Independent native seed donors repeat")
    if plan["fit_or_data_execution_authorized"] is not False or plan["project_validation_outcome_accessed"] is not False:
        raise ValueError("Design preparation is not an execution release")
    return {"design_consistent": True, "independent_native_unique_fits": 20,
            "factorized_unique_fits": 10, "core_unique_fits_with_disclosed_single_donor_reuse": 30,
            "required_auxiliary_unique_fits": 5, "full_family_unique_fits": 35,
            "served_arm_seed_cells": 25, "parameter_control_width": minimum_no_smaller_native_width()}
