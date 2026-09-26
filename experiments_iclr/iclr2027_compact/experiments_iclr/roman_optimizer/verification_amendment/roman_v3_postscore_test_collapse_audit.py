"""Post-score inference-layout test-node audit; never changes selected weights/scores.

The 1e-4 GPU diagnostic tolerance and exact CPU identity were fixed by the
validation-only amendment before any Roman24 test score was opened. This
script runs only after the original 24-cell CUDA test replay audit passed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
STUDY = HERE / "roman_mechanism_v3_prepared"
sys.path.insert(0, str(STUDY))
import mechanism as mech  # noqa: E402
import roman_mechanism as study  # noqa: E402
import tuning as base  # noqa: E402

TOLERANCE = 1e-4
OUT = STUDY / "post_score_test_collapse_audit.json"


def read(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    assert not OUT.exists()
    torch.set_num_threads(2)
    gpu = torch.device("cuda:0")
    cpu = torch.device("cpu")
    assert torch.cuda.is_available()
    source_sha = study.check_freeze()
    amendment = read(STUDY / "ROMAN_V3_VERIFICATION_AMENDMENT_FREEZE.json")
    assert amendment["original_v3_source_freeze_sha256"] == source_sha
    assert amendment["gpu_diagnostic_tolerance"] == TOLERANCE
    lock_path = STUDY / "validation_lock.json"
    lock = read(lock_path)
    assert lock["status"] == "COMPLETE_24_CELL_VALIDATION_REPLAY_LOCK"
    assert lock["source_freeze_sha256"] == source_sha
    assert lock["verification_amendment_sha256"] == study.sha(
        STUDY / "ROMAN_V3_VERIFICATION_AMENDMENT_FREEZE.json")
    locked = {(r["depth"], r["seed"], r["arm"]): r for r in lock["records"]}
    assert len(locked) == 24
    score_audit_path = STUDY / "complete_score_audit.json"
    score_audit = read(score_audit_path)
    assert score_audit["status"] == "COMPLETE_24_CELL_CUDA_TEST_REPLAY_PASS"
    assert score_audit["validation_lock_sha256"] == study.sha(lock_path)
    assert score_audit["source_freeze_sha256"] == source_sha
    assert len(score_audit["records"]) == 24
    audited_scores = {(r["depth"], r["seed"], r["arm"]): r
                      for r in score_audit["records"]}
    expected = {(d, s, a) for d in study.DEPTHS for s in study.SEEDS
                for a in study.ARMS}
    assert set(audited_scores) == expected and len(audited_scores) == 24
    rows = []
    for depth in study.DEPTHS:
        base.configure(depth)
        gpu_bundle, gpu_descriptor = base.load_graph("roman", gpu, include_test=True)
        cpu_bundle, cpu_descriptor = base.load_graph("roman", cpu, include_test=True)
        assert gpu_descriptor == cpu_descriptor
        gpu_idx = gpu_bundle.test_idx_cpu.to(gpu)
        cpu_idx = cpu_bundle.test_idx_cpu
        assert torch.equal(gpu_bundle.test_idx_cpu, cpu_idx)
        for seed in study.SEEDS:
            for arm in ("sync", "norm_sync"):
                key = depth, seed, arm
                folder = study.run_dir(*key)
                checkpoint_path = folder / "checkpoint.pt"
                assert study.sha(checkpoint_path) == locked[key]["checkpoint_sha256"]
                score_path = STUDY / "test_scores" / f"depth{depth}" / f"seed{seed}" / arm / "test_result.json"
                score = read(score_path)
                assert study.sha(score_path) == audited_scores[key]["test_result_sha256"]
                assert score["validation_lock_sha256"] == study.sha(lock_path)
                assert score["selected_checkpoint_sha256"] == study.sha(checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location=gpu, weights_only=True)
                assert (checkpoint["depth"], checkpoint["seed"], checkpoint["arm"],
                        checkpoint["epoch"]) == (depth, seed, arm, locked[key]["selected_epoch"])
                base.seed_all(seed)
                model, _ = study.make_arm(arm, gpu_bundle, gpu)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                mech.assert_equal_graph_weights(model)
                compact = mech.collapse_model(model, gpu_bundle, gpu)
                mapped = mech.collapsed_state(model)
                assert set(mapped) == set(compact.state_dict())
                assert all(torch.equal(value, compact.state_dict()[name])
                           for name, value in mapped.items())
                model.eval()
                compact.eval()
                with torch.no_grad():
                    original = torch.stack(base.member_logits(
                        model, "untied", gpu_bundle))[:, gpu_idx]
                    collapsed = torch.stack(base.member_logits(
                        compact, "tied", gpu_bundle))[:, gpu_idx]
                gpu_difference = float((original - collapsed).abs().max().item())
                gpu_member_mismatch = int((original.argmax(-1) !=
                                           collapsed.argmax(-1)).sum().item())
                gpu_pooled_mismatch = int((original.mean(0).argmax(-1) !=
                                           collapsed.mean(0).argmax(-1)).sum().item())
                assert math.isfinite(gpu_difference) and gpu_difference <= TOLERANCE
                assert gpu_member_mismatch == gpu_pooled_mismatch == 0

                cpu_model, _ = base.make_model("untied", cpu_bundle, cpu)
                cpu_model.load_state_dict(checkpoint["state_dict"], strict=True)
                mech.assert_equal_graph_weights(cpu_model)
                cpu_compact = mech.collapse_model(cpu_model, cpu_bundle, cpu)
                cpu_mapped = mech.collapsed_state(cpu_model)
                assert set(cpu_mapped) == set(cpu_compact.state_dict())
                assert all(torch.equal(value, cpu_compact.state_dict()[name])
                           for name, value in cpu_mapped.items())
                cpu_model.eval()
                cpu_compact.eval()
                with torch.no_grad():
                    cpu_original = torch.stack(base.member_logits(
                        cpu_model, "untied", cpu_bundle))[:, cpu_idx]
                    cpu_collapsed = torch.stack(base.member_logits(
                        cpu_compact, "tied", cpu_bundle))[:, cpu_idx]
                assert torch.equal(cpu_original, cpu_collapsed)
                rows.append({"depth": depth, "seed": seed, "arm": arm,
                             "selected_checkpoint_sha256": study.sha(checkpoint_path),
                             "test_result_sha256": study.sha(score_path),
                             "gpu_max_abs_member_logit_difference": gpu_difference,
                             "gpu_member_decision_mismatches": gpu_member_mismatch,
                             "gpu_pooled_decision_mismatches": gpu_pooled_mismatch,
                             "gpu_state_exactly_mapped": True,
                             "cpu_state_exactly_mapped": True,
                             "cpu_member_logits_bitwise_equal": True,
                             "collapsed_state_sha256": base.state_sha(compact.state_dict())})
                print("TEST_COLLAPSE_PASS", depth, seed, arm, flush=True)
    assert len(rows) == 12
    study.write_json(OUT, {
        "status": "COMPLETE_12_CELL_POST_SCORE_TEST_COLLAPSE_AUDIT_PASS",
        "scope": "same selected Roman24 checkpoints; test-node inference-layout equivalence only",
        "source_freeze_sha256": source_sha,
        "verification_amendment_sha256": study.sha(
            STUDY / "ROMAN_V3_VERIFICATION_AMENDMENT_FREEZE.json"),
        "validation_lock_sha256": study.sha(lock_path),
        "complete_score_audit_sha256": study.sha(score_audit_path),
        "audit_source_sha256": study.sha(Path(__file__)),
        "gpu_diagnostic_tolerance": TOLERANCE,
        "test_scores_or_selected_weights_modified": False,
        "records": rows})
    print("COMPLETE_12_CELL_POST_SCORE_TEST_COLLAPSE_AUDIT_PASS", flush=True)


if __name__ == "__main__":
    main()
