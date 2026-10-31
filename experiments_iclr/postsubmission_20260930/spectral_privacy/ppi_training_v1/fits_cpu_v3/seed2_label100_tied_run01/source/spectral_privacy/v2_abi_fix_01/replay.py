"""Recompute all spectral choices from hashed matrix evidence; no labels."""

import argparse
import json
from pathlib import Path

import torch

if __package__:
    from .collector import LOSS_RULE, environment_contract, validate_partition
    from .contracts import (code_hashes, json_hash, save_new_json, unpack_float64, verify_seal)
    from .spectral_core import RULE, choose_bases, serialize_selection
else:
    from collector import LOSS_RULE, environment_contract, validate_partition
    from contracts import (code_hashes, json_hash, save_new_json, unpack_float64, verify_seal)
    from spectral_core import RULE, choose_bases, serialize_selection


def replay_record(record, *, strict_environment=True, strict_code=True,
                  expected_source_sha256=None, expected_probe_state_sha256=None):
    """Exact content replay, intended for the recorded CPU torch environment.

    Even with an explicit environment override, derived tensor/basis bytes
    must agree exactly. This is not a claim of bitwise eigensystem identity
    across BLAS/torch versions or at degenerate eigenvalues. Inputs and saved
    bases remain content-addressed. There is no tolerant success mode.
    """
    verify_seal(record)
    validate_partition(record["partition"])
    for field in ("source_models_sha256", "model_state_sha256"):
        digest = record[field]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("Malformed source/probe SHA256")
    if expected_source_sha256 is not None and record["source_models_sha256"] != expected_source_sha256:
        raise ValueError("Record belongs to a different frozen source")
    if expected_probe_state_sha256 is not None and record["model_state_sha256"] != expected_probe_state_sha256:
        raise ValueError("Record belongs to a different tied probe checkpoint")
    if strict_code and record["code_sha256"] != code_hashes():
        raise ValueError("Replay code hashes differ from collector record")
    if strict_environment and record["environment"] != environment_contract():
        raise ValueError("Replay environment differs; use the recorded CPU environment")
    if (record["members"] != 4 or record["checkpoint_epoch"] != 25
            or record["loss_rule"] != LOSS_RULE
            or record["gradient_scale"] != "unscaled own-member own-graph mean loss"
            or record["fold_weighting"] != "equal_complete_graph"
            or record["selection"]["rule"] != RULE
            or record["state_grads_modes_and_rng_preserved"] is not True):
        raise ValueError("Record does not satisfy the frozen v2 recipe")
    ids = record["partition"]["discovery"] + record["partition"]["screen"]
    fingerprints = record["input_fingerprints"]
    if ([row["unit_id"] for row in fingerprints] != ids
            or any(row["split"] != "train" for row in fingerprints)
            or record["input_identity_sha256"] != json_hash(fingerprints)
            or set(record["graph_metadata"]) != set(ids)):
        raise ValueError("Training graph identity/order mismatch")
    target_keys = set(record["targets"])
    expected_keys = {f"residual_modules.{layer}.module.conv.lin_l.weight" for layer in range(3)}
    if target_keys != expected_keys:
        raise ValueError("Fixed source neighbor target names differ")
    for key, metadata in record["targets"].items():
        expected_parameter = "source." + key.removesuffix(".weight") + ".base.weight"
        if (metadata["actual_parameter_name"] != expected_parameter
                or metadata["dtype"] not in ("torch.float32", "torch.float64")):
            raise ValueError("Actual named matrix parameter/dtype contract mismatch")
    raw = {}
    for group in ("discovery", "screen"):
        evidence = record["groups"][group]
        if evidence["unit_ids"] != record["partition"][group] or set(evidence["matrix_gradients"]) != target_keys:
            raise ValueError("Gradient graph IDs or target names disagree")
        raw[group] = {key: unpack_float64(payload)
                      for key, payload in evidence["matrix_gradients"].items()}
        for key, value in raw[group].items():
            if list(value.shape) != [4, 4, 128, 128] or record["targets"][key]["shape"] != [128, 128]:
                raise ValueError("Matrix evidence dimensions differ from v2 recipe")
    bases, details, statistics = choose_bases(raw["discovery"], raw["screen"],
                                              n0=RULE["n0"], rank_max=RULE["rank_max_per_target"])
    derived = serialize_selection(bases, details, statistics, raw["discovery"],
                                  record["selection"]["controls"]["random_seed"])
    if json_hash(derived) != json_hash(record["selection"]):
        raise ValueError("Derived covariance, eigenbasis, ranks, or controls do not replay exactly")
    # Decode selected/control bases too, validating their independent raw-byte hashes.
    for group in (record["selection"]["basis"],
                  record["selection"]["controls"]["random_matched_rank_basis"],
                  record["selection"]["controls"]["raw_centered_gradient_matched_rank_basis"]):
        for key, payload in group.items():
            value = unpack_float64(payload)
            if list(value.shape) != [128, derived["ranks"][key]]:
                raise ValueError("Saved matched-rank basis shape mismatch")
    return {"status": "EXACT_REPLAY_PASS", "record_sha256": record["record_sha256"],
            "partition_sha256": record["partition"]["sha256"],
            "input_identity_sha256": record["input_identity_sha256"],
            "ranks": derived["ranks"],
            "basis_sha256": {key: payload["sha256"] for key, payload in derived["basis"].items()},
            "selected_score_is_unbiased": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--interop-threads", type=int)
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    if args.interop_threads is not None:
        torch.set_num_interop_threads(args.interop_threads)
    result = replay_record(json.loads(args.record.read_text()))
    kwargs = {} if args.artifact_root is None else {"artifact_root": args.artifact_root}
    save_new_json(result, args.output, **kwargs)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
