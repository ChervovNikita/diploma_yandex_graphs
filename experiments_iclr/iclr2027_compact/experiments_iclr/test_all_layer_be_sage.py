"""CPU-only checks for the standalone all-layer SAGE BatchEnsemble control."""

from __future__ import annotations

import csv
import fcntl
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import all_layer_be_sage as pilot


def small_graph() -> tuple[SimpleNamespace, torch.Tensor, torch.Tensor]:
    n = 12
    graph = SimpleNamespace(edge_index=torch.tensor([
        list(range(n - 1)) + list(range(1, n)),
        list(range(1, n)) + list(range(n - 1)),
    ], dtype=torch.long))
    x = torch.arange(n * 4, dtype=torch.float32).reshape(n, 4) / 48
    y = torch.tensor([0, 1] * (n // 2), dtype=torch.long)
    return graph, x, y


def test_identity_and_count() -> None:
    assert pilot.OFFICIAL_MASKS == (0, 1, 2, 3, 4)
    assert pilot.RESULT_ROOT.resolve().is_relative_to(pilot.REPO.resolve())
    assert pilot.RESULT_REL != Path("experiments_iclr/results")
    for split in pilot.OFFICIAL_MASKS:
        result = pilot.identity_check(split)
        assert result["initial_logit_max_abs_difference"] == 0.0
        assert result["post_construction_rng_matches"] is True
        assert result["base_trainable_parameters"] == pilot.BASE_PARAMS
        assert result["added_trainable_parameters"] == pilot.ADDED_PARAMS
        assert result["total_trainable_parameters"] == pilot.TOTAL_PARAMS


def test_gradient_and_checkpoint() -> None:
    graph, x, y = small_graph()
    args = SimpleNamespace(model="SAGE", num_layers=1, hidden_dim=8, m=4)
    torch.manual_seed(11)
    model = pilot.AllLayerBESAGEModel(args, 4, 2, torch.device("cpu"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for member in range(4):
        logits = model(graph, x, member)
        (F.cross_entropy(logits[:6], y[:6]) / 4).backward()
    for layer in model._factor_layers:
        for factor in (layer.R, layer.S, layer.B):
            assert factor.grad is not None
            assert torch.isfinite(factor.grad).all()
            assert bool(factor.grad.abs().sum() > 0)
    optimizer.step()
    assert any(not torch.equal(layer.R, torch.ones_like(layer.R))
               for layer in model._factor_layers)
    model.eval()
    with torch.no_grad():
        expected = [model(graph, x, member) for member in range(4)]
    restored = pilot.AllLayerBESAGEModel(args, 4, 2, torch.device("cpu"))
    restored.load_state_dict(model.state_dict())
    restored.eval()
    with torch.no_grad():
        actual = [restored(graph, x, member) for member in range(4)]
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_resume_integrity() -> None:
    temp_parent = pilot.REPO / "experiments_iclr/.tmp/all_layer_be_tests"
    temp_parent.mkdir(parents=True, exist_ok=True)
    old = (pilot.RESULT_REL, pilot.RESULT_ROOT, pilot.RESULT_CSV,
           pilot.HASHES_PATH)
    try:
        with tempfile.TemporaryDirectory(dir=temp_parent) as temporary:
            root = Path(temporary)
            pilot.RESULT_REL = root.relative_to(pilot.REPO)
            pilot.RESULT_ROOT = root
            pilot.RESULT_CSV = root / "projector_controls.csv"
            pilot.HASHES_PATH = root / "artifact_hashes.json"
            _, checkpoint = pilot.expected_artifact(0, "checkpoint")
            _, prediction = pilot.expected_artifact(0, "prediction_file")
            checkpoint.parent.mkdir(parents=True)
            prediction.parent.mkdir(parents=True)
            model = pilot.AllLayerBESAGEModel(
                pilot.fixed_args(), 300, 18, torch.device("cpu"))
            torch.save(model.state_dict(), checkpoint)
            n = 5666
            labels = np.zeros(n, dtype=np.int64)
            logits = np.zeros((4, n, 18), dtype=np.float32)
            np.savez_compressed(
                prediction,
                node_index=np.arange(n), y_true=labels,
                ensemble_pred=labels, member_pred=np.zeros((4, n), dtype=np.int64),
                member_logits=logits,
                ensemble_prob=np.full((n, 18), 1 / 18, dtype=np.float32),
                confidence=np.full(n, 1 / 18, dtype=np.float32),
                degree=np.zeros(n), local_homophily=np.zeros(n),
            )
            row = {field: "0" for field in pilot.controls.FIELDS}
            row.update({
                "dataset": "roman-empire", "model": "SAGE",
                "variant": pilot.VARIANT, "split": "0", "seed": "0",
                "num_layers": "5", "hidden_dim": "512", "lr": "3e-5",
                "m": "4", "num_steps": "5000", "best_step": "1",
                "num_params": str(pilot.TOTAL_PARAMS),
                "train_seconds": "1", "val_metric": "1",
                "test_metric": "1", "test_acc": "1",
                "mean_member_acc": "1", "n_test": str(n),
                "checkpoint": str(pilot.RESULT_REL / "checkpoints" / checkpoint.name),
                "prediction_file": str(pilot.RESULT_REL / "predictions" / prediction.name),
            })
            with pilot.RESULT_CSV.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=pilot.controls.FIELDS)
                writer.writeheader()
                writer.writerow(row)
            pilot.write_json_atomic(pilot.HASHES_PATH, {
                "0": {"checkpoint": pilot.sha256(checkpoint),
                      "prediction_file": pilot.sha256(prediction)},
            })
            rows, _ = pilot.validate_resume()
            assert len(rows) == 1
            with prediction.open("ab") as stream:
                stream.write(b"changed")
            try:
                pilot.validate_resume()
            except ValueError as exc:
                assert "Changed prediction_file" in str(exc)
            else:
                raise AssertionError("A changed prediction archive was accepted")
    finally:
        (pilot.RESULT_REL, pilot.RESULT_ROOT, pilot.RESULT_CSV,
         pilot.HASHES_PATH) = old


def test_optional_gat_gate() -> None:
    temp_parent = pilot.REPO / "experiments_iclr/.tmp/all_layer_be_tests"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_parent) as temporary:
        root = Path(temporary)
        log_dir = root / "experiments_iclr/logs"
        log_dir.mkdir(parents=True)
        (log_dir / "run_after_sage.log").write_text("ENSEMBLE_INFERENCE_PROFILE_COMPLETE\n")
        gat_root = root / "experiments_iclr/gat_failure_pair_results"
        gat_root.mkdir(parents=True)

        def process_list(command: str):
            return SimpleNamespace(stdout=command, returncode=0)

        with patch.object(pilot, "REPO", root), \
                patch.object(pilot, "GAT_ROOT", gat_root), \
                patch("verify_gat_failure_pair.queue_terminal_marker",
                      return_value="ENSEMBLE_INFERENCE_PROFILE_COMPLETE"):
            with patch.object(pilot.subprocess, "run", return_value=process_list("")):
                state = pilot.require_queue_finished_and_gat_idle()
                assert state["gat_status"] == "incomplete_or_failed_after_launcher_exit"
                assert state["gat_results_sha256"] is None
            for command in (
                    "123 bash experiments_iclr/run_gat_failure_pair.sh\n",
                    "124 .venv/bin/python experiments_iclr/projector_controls.py --model GAT\n"):
                with patch.object(pilot.subprocess, "run",
                                  return_value=process_list(command)):
                    try:
                        pilot.require_queue_finished_and_gat_idle()
                    except RuntimeError as exc:
                        assert "still active" in str(exc)
                    else:
                        raise AssertionError("An active GAT job was accepted")
            (gat_root / "run.log").write_text("GAT_PAIR_COMPLETE\n")
            (gat_root / "projector_controls.csv").write_text("header\n")
            (gat_root / "protocol.json").write_text("{}\n")
            verified = SimpleNamespace(returncode=0, stdout="ok", stderr="")
            with patch.object(pilot.subprocess, "run",
                              side_effect=[process_list(""), verified]):
                state = pilot.require_queue_finished_and_gat_idle()
                assert state["gat_status"] == "complete"
                assert state["gat_final_marker_present"] is True
                assert state["gat_verification_exit_code"] == 0


def test_selected_comparator_verifier_gate() -> None:
    success = SimpleNamespace(returncode=0, stdout="verified", stderr="")
    with patch.object(pilot.subprocess, "run", return_value=success) as invoked:
        pilot.verify_selected_sage_controls()
    assert invoked.call_args.args[0][-2:] == [
        "--require-complete", "--check-checkpoints"]
    assert invoked.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    failure = SimpleNamespace(returncode=1, stdout="", stderr="bad artifact")
    with patch.object(pilot.subprocess, "run", return_value=failure):
        try:
            pilot.verify_selected_sage_controls()
        except RuntimeError as exc:
            assert "bad artifact" in str(exc)
        else:
            raise AssertionError("A failed selected-control audit was accepted")


def test_comparator_freeze_and_gat_lock() -> None:
    temp_parent = pilot.REPO / "experiments_iclr/.tmp/all_layer_be_tests"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_parent) as temporary:
        root = Path(temporary)
        comparator = root / "projector_controls.csv"
        adoption = root / "gnnm_split0_verification_audit.json"
        with comparator.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=pilot.controls.FIELDS)
            writer.writeheader()
            for variant in ("gnnm", "ens_pooled", "untied_backbone", "base"):
                for split in range(5):
                    row = {field: "" for field in pilot.controls.FIELDS}
                    row.update(dataset="roman-empire", model="SAGE",
                               variant=variant, split=str(split))
                    writer.writerow(row)
        adoption.write_text('{"adopted": "fresh"}\n')
        with patch.object(pilot, "SAGE_CSV", comparator), \
                patch.object(pilot, "SPLIT0_AUDIT", adoption):
            frozen = pilot.completed_sage_comparators()
            assert frozen["sage_controls_csv_sha256"] == pilot.sha256(comparator)
            assert frozen["split0_adoption_audit_sha256"] == pilot.sha256(adoption)
            adoption.write_text('{"adopted": "pilot"}\n')
            changed = pilot.completed_sage_comparators()
            assert changed["split0_adoption_audit_sha256"] != \
                frozen["split0_adoption_audit_sha256"]
        lock_root = root / "gat_failure_pair_results"
        with patch.object(pilot, "GAT_ROOT", lock_root):
            with pilot.hold_gat_run_lock():
                with (lock_root / ".run.lock").open("a") as competitor:
                    try:
                        fcntl.flock(competitor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        raise AssertionError("GAT could acquire a held run lock")


def test_verify_only_is_read_only() -> None:
    temp_parent = pilot.REPO / "experiments_iclr/.tmp/all_layer_be_tests"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_parent) as temporary:
        root = Path(temporary) / "missing_results"
        manifest = root / "protocol.json"
        with patch.object(pilot, "RESULT_ROOT", root), \
                patch.object(pilot, "MANIFEST_PATH", manifest), \
                patch.object(sys, "argv", ["all_layer_be_sage.py", "--verify-only"]):
            try:
                pilot.main()
            except FileNotFoundError as exc:
                assert "Frozen all-layer protocol" in str(exc)
            else:
                raise AssertionError("Missing frozen manifest was accepted")
            assert not root.exists()
            root.mkdir()
            frozen = {"schema": 1, "source_sha256": {}}
            pilot.write_json_atomic(manifest, frozen)
            before = (manifest.read_bytes(), manifest.stat().st_mtime_ns)
            with patch.object(pilot, "verify_selected_sage_controls"), \
                    patch.object(pilot, "require_queue_finished_and_gat_idle",
                                 return_value={}), \
                    patch.object(pilot, "expected_manifest", return_value=frozen), \
                    patch.object(pilot, "validate_resume", return_value=([], {})):
                pilot.main()
            assert (manifest.read_bytes(), manifest.stat().st_mtime_ns) == before
            assert not (root / ".run.lock").exists()


def test_failed_preflight_does_not_freeze_manifest() -> None:
    temp_parent = pilot.REPO / "experiments_iclr/.tmp/all_layer_be_tests"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_parent) as temporary:
        root = Path(temporary) / "new_results"
        check = {"base_trainable_parameters": pilot.BASE_PARAMS,
                 "added_trainable_parameters": pilot.ADDED_PARAMS,
                 "total_trainable_parameters": pilot.TOTAL_PARAMS}
        with patch.object(pilot, "RESULT_ROOT", root), \
                patch.object(pilot, "RESULT_CSV", root / "projector_controls.csv"), \
                patch.object(pilot, "HASHES_PATH", root / "artifact_hashes.json"), \
                patch.object(pilot, "MANIFEST_PATH", root / "protocol.json"), \
                patch.object(pilot, "IDENTITY_PATH", root / "initialization_audit.json"), \
                patch("verify_gat_failure_pair.queue_terminal_marker",
                      return_value="ENSEMBLE_INFERENCE_PROFILE_COMPLETE"), \
                patch.object(pilot, "hold_gat_run_lock", return_value=nullcontext()), \
                patch.object(pilot, "require_queue_finished_and_gat_idle",
                              return_value={}), \
                patch.object(pilot, "verify_selected_sage_controls"), \
                patch.object(pilot, "expected_manifest",
                              return_value={"source_sha256": {}}), \
                patch.object(pilot, "validate_resume", return_value=([], {})), \
                patch.object(pilot, "identity_check", return_value=check), \
                patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(sys, "argv", ["all_layer_be_sage.py"]):
            try:
                pilot.main()
            except RuntimeError as exc:
                assert "requires CUDA" in str(exc)
            else:
                raise AssertionError("Unavailable CUDA was accepted")
        assert not (root / "protocol.json").exists()
        assert not (root / "initialization_audit.json").exists()


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_identity_and_count()
    test_gradient_and_checkpoint()
    test_resume_integrity()
    test_optional_gat_gate()
    test_selected_comparator_verifier_gate()
    test_comparator_freeze_and_gat_lock()
    test_verify_only_is_read_only()
    test_failed_preflight_does_not_freeze_manifest()
    print("CPU all-layer checks passed: identity, gradients, restore, resume, comparator freeze, GAT lock, read-only verification, and preflight")
