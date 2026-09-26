"""All-six initial-logit numerical diagnosis; no test labels or scores."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "roman_mechanism_v3_prepared"))
import tuning as base  # noqa: E402
import roman_narrow_untied as study  # noqa: E402

OUT = study.ROOT / "initial_replay_all6_diagnostic.json"


def summarize(reference: torch.Tensor, replay: torch.Tensor) -> dict:
    diff = (reference - replay).abs()
    changed = (reference.argmax(-1) != replay.argmax(-1)).nonzero(as_tuple=False)
    details = []
    for member, node in changed.tolist():
        old = torch.topk(reference[member, node], 2)
        new = torch.topk(replay[member, node], 2)
        details.append({"member": member, "node": node,
                        "archived_top2_class": old.indices.tolist(),
                        "archived_top2_logit": old.values.tolist(),
                        "archived_margin": float((old.values[0] - old.values[1]).item()),
                        "replay_top2_class": new.indices.tolist(),
                        "replay_top2_logit": new.values.tolist(),
                        "replay_margin": float((new.values[0] - new.values[1]).item())})
    pooled_old, pooled_new = reference.mean(0), replay.mean(0)
    changed_pool = (pooled_old.argmax(-1) != pooled_new.argmax(-1)).nonzero(
        as_tuple=False).flatten()
    return {"max_abs_member_logit_difference": float(diff.max().item()),
            "mean_abs_member_logit_difference": float(diff.mean().item()),
            "member_decision_mismatch_count": len(details),
            "member_decision_mismatches": details,
            "pooled_decision_mismatch_count": int(changed_pool.numel()),
            "pooled_mismatch_nodes": changed_pool.tolist()}


def main() -> None:
    assert not OUT.exists()
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    assert study.check_freeze()
    records = []
    gpu, cpu = torch.device("cuda:0"), torch.device("cpu")
    for depth in study.DEPTHS:
        study.configure(depth)
        gpu_bundle, gpu_descriptor = base.load_graph("roman", gpu, include_test=False)
        cpu_bundle, cpu_descriptor = base.load_graph("roman", cpu, include_test=False)
        assert gpu_descriptor == cpu_descriptor
        assert not hasattr(gpu_bundle, "test_idx_cpu")
        assert not hasattr(cpu_bundle, "test_idx_cpu")
        for seed in study.SEEDS:
            folder = study.run_dir(depth, seed)
            row = json.loads((folder / "result.json").read_text())
            archive = np.load(folder / "initial_logits.npy", allow_pickle=False)
            assert archive.shape == (4, 22662, gpu_bundle.classes)
            archived = torch.from_numpy(archive).to(gpu)
            base.seed_all(seed)
            model, canonical = study.make_model(gpu_bundle, gpu)
            initial = base.initial_audit(model, "untied", gpu_bundle,
                                         seed, gpu, canonical)
            assert canonical == row["initialization"]["canonical_projector_state_sha256"]
            for field in ("python_rng_sha256", "numpy_rng_sha256",
                          "cpu_rng_sha256", "cuda_rng_sha256", "parameter_count"):
                assert initial[field] == row["initialization"][field]
            first_state = {key: value.detach().clone()
                           for key, value in model.state_dict().items()}
            first_hash = base.state_sha(first_state)
            base.seed_all(seed)
            repeated_model, repeated_canonical = study.make_model(gpu_bundle, gpu)
            second_state = repeated_model.state_dict()
            assert canonical == repeated_canonical
            assert set(first_state) == set(second_state)
            assert all(torch.equal(first_state[key], second_state[key])
                       for key in first_state)
            assert first_hash == base.state_sha(second_state)
            model.eval()
            gpu_replays = []
            with torch.no_grad():
                for _ in range(5):
                    gpu_replays.append(torch.stack(base.member_logits(
                        model, "untied", gpu_bundle)).detach().clone())
            cpu_model, _ = base.make_model("untied", cpu_bundle, cpu)
            cpu_model.load_state_dict(model.state_dict(), strict=True)
            assert all(torch.equal(value.cpu(), cpu_model.state_dict()[key])
                       for key, value in model.state_dict().items())
            cpu_model.eval()
            cpu_replays = []
            with torch.no_grad():
                for _ in range(3):
                    cpu_replays.append(torch.stack(base.member_logits(
                        cpu_model, "untied", cpu_bundle)).detach().clone())
            gpu_vs_archive = [summarize(archived, x) for x in gpu_replays]
            gpu_vs_gpu = [summarize(gpu_replays[0], x) for x in gpu_replays[1:]]
            cpu_vs_cpu = [summarize(cpu_replays[0], x) for x in cpu_replays[1:]]
            cpu_vs_archive = summarize(archived.cpu(), cpu_replays[0])
            records.append({"depth": depth, "seed": seed,
                            "width": study.WIDTHS[depth],
                            "archived_initial_logits_sha256": study.sha(
                                folder / "initial_logits.npy"),
                            "canonical_full_tied_initial_state_sha256": canonical,
                            "untied_initial_state_sha256": first_hash,
                            "reconstructed_untied_initial_state_bitwise_equal": True,
                            "cpu_model_state_bitwise_equal_to_gpu_state": True,
                            "gpu_vs_archive": gpu_vs_archive,
                            "gpu_vs_gpu": gpu_vs_gpu,
                            "cpu_vs_cpu": cpu_vs_cpu,
                            "cpu_vs_archive": cpu_vs_archive})
            print("INITIAL_DIAG", depth, seed,
                  "archive_vs_gpu_mismatch", gpu_vs_archive[0]["member_decision_mismatch_count"],
                  "same_gpu_mismatch", gpu_vs_gpu[0]["member_decision_mismatch_count"],
                  flush=True)
    assert len(records) == 6
    study.write_json(OUT, {
        "status": "ALL_6_INITIAL_REPLAY_VALIDATION_ONLY_DIAGNOSTIC",
        "source_freeze_sha256": study.sha(study.ROOT / "SOURCE_FREEZE.json"),
        "diagnostic_source_sha256": study.sha(Path(__file__)),
        "test_ids_or_labels_accessed": False,
        "records": records})
    print("ALL_6_INITIAL_REPLAY_DIAGNOSTIC_COMPLETE", study.sha(OUT), flush=True)


if __name__ == "__main__":
    main()
