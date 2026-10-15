"""Finite real-TRAIN native-forward BN audit; zero optimizer steps and no TEST."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    import torch
    from torch_geometric.data import Batch
    from banks import attach_molecular
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    job_path, output = Path(args.job).resolve(), Path(args.output).resolve()
    if not all(p.is_relative_to(root) for p in (job_path, output, Path(__file__).resolve())):
        raise ValueError("Only own repository paths")
    if output.exists():
        raise ValueError("Preserve previous qualification; choose fresh output")
    job = json.loads(job_path.read_text())
    if job["family"] != "ogb" or job["procedure"] != "gnnm_boundary_native" or job["TEST_access"] is not False:
        raise ValueError("Use the frozen molecular GNNM TRAIN/VALID descriptor")
    torch.set_num_threads(2)
    callback_path = Path(job["callback_module"]).resolve()
    if not callback_path.is_relative_to(root) or pin(callback_path)["sha256"] != job["callback_sha256"]:
        raise ValueError("Frozen native callback changed")
    spec = importlib.util.spec_from_file_location("_bn_real_native_session", callback_path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    recipe = copy.deepcopy(job["recipe"]); recipe["_seed"] = job["seed"]
    sessions = []
    start = time.perf_counter()
    for _ in range(2):
        torch.manual_seed(job["seed"])
        sessions.append(module.create_session(copy.deepcopy(recipe), "gnnm_boundary_native", job["data_root"], "cpu"))
    baseline, candidate = sessions
    candidate.model.load_state_dict(copy.deepcopy(baseline.model.state_dict()))
    candidate_meta = attach_molecular(candidate)
    batch = Batch.from_data_list([baseline.train_view[i] for i in range(32)])
    reference_rng = torch.get_rng_state().clone()
    train_routes, gradients = [], []
    for session in sessions:
        torch.set_rng_state(reference_rng)
        session.model.train(); session.optimizer.zero_grad()
        routes = [session.model.run_member(m, batch) for m in range(4)]
        loss = torch.stack([session._native_loss(r[0], batch) for r in routes]).mean()
        loss.backward()
        train_routes.append(torch.stack([r[0].detach().clone() for r in routes]))
        gradients.append({name: None if p.grad is None else p.grad.detach().clone()
                          for name, p in session.model.named_parameters()})
        session.model.assert_source_identity()
    if set(gradients[0]) != set(gradients[1]):
        raise AssertionError("Private BN must add zero learned parameters")
    diffs = []
    for name in gradients[0]:
        a, b = gradients[0][name], gradients[1][name]
        if (a is None) != (b is None):
            raise AssertionError("Changed gradient support: " + name)
        if a is not None:
            diffs.append(float((a-b).abs().max()))
    max_output = float((train_routes[0]-train_routes[1]).abs().max())
    max_gradient = max(diffs, default=0.)
    if max_output > 1e-6 or max_gradient > 1e-6:
        raise AssertionError("Unexpected TRAIN semantic difference")
    manager = candidate.model.route_bn_state
    counters = [[int(bank.num_batches_tracked) for bank in module_banks] for module_banks in manager.banks]
    if any(counts != [1, 1, 1, 1] for counts in counters):
        raise AssertionError("Unexpected real native BN call count")
    candidate.model.eval()
    before = copy.deepcopy(candidate.model.state_dict())
    with torch.no_grad():
        eval_routes = torch.stack([candidate.model.run_member(m, batch)[0] for m in range(4)])
    if any(not torch.equal(v, candidate.model.state_dict()[k]) for k, v in before.items()):
        raise AssertionError("Eval mutated tracked state")
    if not torch.isfinite(eval_routes).all():
        raise AssertionError("Nonfinite candidate TRAIN diagnostic")
    result = {"schema": "native-bn-state-cpu-qualification-v1", "status": "passed",
              "task": job["recipe"]["dataset"], "role": "TRAIN", "graphs": 32,
              "TRAIN_graph_ids": batch.smoke_graph_id.tolist(),
              "task_validation_or_test_scoring": False, "TEST_accessed": False,
              "optimizer_transitions": 0, "model_constructions": 2,
              "route_forwards": {"TRAIN": 8, "eval_TRAIN": 4},
              "max_train_output_abs_difference": max_output,
              "max_train_gradient_abs_difference": max_gradient,
              "route_module_counters": counters, "candidate": candidate_meta,
              "job": pin(job_path), "callback": pin(callback_path),
              "banks": pin(Path(__file__).with_name("banks.py")),
              "script": pin(Path(__file__)), "torch": torch.__version__,
              "elapsed_seconds": time.perf_counter()-start,
              "limitations": ["CPU forward/gradient ownership only", "no learned fit or predictive effect",
                              "not CUDA, BUDDY, activation-checkpoint or resume qualification"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("status", "graphs", "optimizer_transitions",
                     "max_train_output_abs_difference", "max_train_gradient_abs_difference", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
