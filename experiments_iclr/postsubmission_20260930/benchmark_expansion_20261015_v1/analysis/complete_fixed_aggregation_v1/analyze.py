"""Complete fixed readout sensitivity on saved selected VALID, CPU only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback


def receipt(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    assert socket.gethostname() == "anogena-2-0"
    assert subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True).splitlines() == ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    root = Path(sys.argv[1]).resolve()
    here, output = Path(__file__).resolve().parent, Path(sys.argv[2]).resolve()
    assert here.is_relative_to(root) and output.is_relative_to(root) and not output.exists()
    inputs = json.loads((here / "INPUTS.json").read_text())
    manifest = json.loads((here / "MANIFEST.json").read_text())
    for pin in manifest["files"]:
        path = here / pin["path"]
        assert receipt(path)["sha256"] == pin["sha256"]

    def verified(row):
        path = Path(row["path"]).resolve()
        assert path.is_relative_to(root) and receipt(path) == row
        return path

    def timeout(*_):
        raise TimeoutError("Fixed60second CPU-only analysis limit")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(60)
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="running", schema="complete-fixed-aggregation-result-v1",
        inputs=receipt(here / "INPUTS.json"), plan=receipt(here / "PLAN.md"),
        manifest=receipt(here / "MANIFEST.json"), scope="posthoc selected-VALID sensitivity",
        new_fits=0, new_updates=0, new_forwards=0, new_models=0, TEST_accessed=False,
        selection_changed=False, chosen_readout=None, cases=[], summaries=[],
        automatic_retries=0, device="CPU", threads=2)
    try:
        import torch
        from ogb.graphproppred import Evaluator
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        helper_path = verified(inputs["helper"])
        spec = importlib.util.spec_from_file_location("fixed_readouts", helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        evaluator = Evaluator(name="ogbg-molbace")
        loaded = {}
        for row in inputs["fits"]:
            analysis = json.loads(verified(row["analysis"]).read_text())
            fit = json.loads(verified(row["fit"]).read_text())
            bank = torch.load(verified(row["routes"]), map_location="cpu", weights_only=False)
            assert analysis["status"] == fit["status"] == "complete"
            assert analysis["raw_tensor_receipt"] == row["routes"]
            assert bank["role"] == "VALID" and bank["TEST_accessed"] is False
            assert bank["job_hash"] == analysis["job_hash"] == fit["job_hash"]
            assert bank["selected_checkpoint"] == row["checkpoint"] == fit["selected_checkpoint"]
            assert bank["selected_epoch"] == analysis["selected_epoch"] == fit["selected_epoch"]
            assert bank["sanity_against_saved_export"]["passed"]
            assert analysis["saved_VALID_export"] == row["export"] == fit["selected_VALID_export"]
            assert len(bank["ids"]) == 151 and len(torch.unique(bank["ids"])) == 151
            assert torch.isfinite(bank["member_raw_logits"]).all()
            loaded[row["job_id"]] = (row, bank)
        assert len(loaded) == 27
        methods = {
            "G0": "gnnm_boundary_native", "geometry": "gnnm_graph_alignment_variance",
            "correction": "gnnm_private_input_residual",
            "both": "gnnm_input_residual_plus_graph_alignment_variance",
            "privateBN": "gnnm_private_bn_state_v3"}
        for seed in inputs["seed_order"]:
            prefix = "molecular_input_steering_and_class_geometry_20261015_v1/split-0/seed-" + str(seed)
            natives = [loaded[prefix + "/native-body-" + str(i)] for i in range(4)]
            candidates = [("M1", [natives[0]]), ("I4", natives)] + [
                (name, [loaded[prefix + "/method-" + method]]) for name, method in methods.items()]
            ids, labels = natives[0][1]["ids"], natives[0][1]["labels"]
            for name, banks in candidates:
                for _, bank in banks:
                    assert torch.equal(bank["ids"], ids) and helper.same_labels(torch, bank["labels"], labels)
                logits = torch.cat([bank["member_raw_logits"] for _, bank in banks], dim=0)
                assert len(logits) == (1 if name == "M1" else 4)
                _, metrics = helper.fixed_readouts(torch, evaluator, logits, labels)
                result["cases"].append(dict(name=name, seed=seed, members=len(logits), graphs=len(ids),
                    source_jobs=[row["job_id"] for row, _ in banks], **metrics))
        assert len(result["cases"]) == 21
        for name in ["M1", "I4", *methods]:
            rows = [r for r in result["cases"] if r["name"] == name]
            entry = dict(name=name, seeds=[r["seed"] for r in rows])
            for rule in ("baseline_mean_raw_logits", "fixed_mean_sigmoid_probabilities"):
                values = [100*r[rule]["rocauc"] for r in rows]
                bce = [r[rule]["observed_label_BCE"] for r in rows]
                entry[rule] = dict(AUC_mean_percent=statistics.mean(values),
                    AUC_sample_SD_percent=statistics.stdev(values), BCE_mean=statistics.mean(bce),
                    BCE_sample_SD=statistics.stdev(bce))
            for metric in ("alternative_minus_baseline_ROC_AUC_points", "alternative_minus_baseline_BCE"):
                values = [r[metric] for r in rows]
                entry[metric] = dict(mean=statistics.mean(values), sample_SD=statistics.stdev(values), values=values)
            result["summaries"].append(entry)
        reference = next(r for r in result["summaries"] if r["name"] == "I4")
        for row in result["summaries"]:
            row["AUC_delta_vs_I4_points"] = {rule: row[rule]["AUC_mean_percent"] - reference[rule]["AUC_mean_percent"]
                for rule in ("baseline_mean_raw_logits", "fixed_mean_sigmoid_probabilities")}
        assert not torch.cuda.is_initialized()
        result.update(status="complete", exact_graph_alignment=True, tensor_files_read=27)
    except BaseException as error:
        result.update(status="failed", failure=dict(type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(limit=12)))
    finally:
        signal.alarm(0)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result.update(elapsed_seconds=time.monotonic()-started, peak_RSS_KiB=usage.ru_maxrss,
            CPU_user_seconds=usage.ru_utime, CPU_system_seconds=usage.ru_stime)
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print(json.dumps({k:result[k] for k in ("status", "elapsed_seconds", "new_updates", "new_forwards")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
