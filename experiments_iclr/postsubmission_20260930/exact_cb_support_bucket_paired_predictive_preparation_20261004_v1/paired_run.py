"""Fresh fixed full-budget TRAIN/VALID cell; source preparation is not a release."""
from argparse import ArgumentParser
from pathlib import Path
import os
import sys
from time import perf_counter
from pilot_common import (preflight, admit_invocation, fresh_output, runtime_stdlib, atomic_json,
                          file_sha, lock_output, driver_module_custody, require)
from pilot_accounting import Attempts


def host_peak_bytes():
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak*1024)


def receipt(path):
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path)}


def main():
    started = perf_counter()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root-release", required=True)
    parser.add_argument("--stage", required=True, choices=("fit",))
    parser.add_argument("--unit", required=True, choices=("target_only", "joint", "separate"))
    parser.add_argument("--base-seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    context = preflight(args.root_release, args.stage)
    admit_invocation(context, args.stage, args.unit, args.base_seed, args.output, False)
    output = fresh_output(args.output)
    lock = lock_output(output)
    attempts = Attempts(output, context, "fit", args.unit, args.base_seed, started=started)
    cuda_started = False
    try:
        require("torch" not in sys.modules and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
                "Qualified workspace must precede Torch")
        context["preimport_environment"] = {"Torch_absent_before_configuration": True,
            "Torch_absent_after_configuration": True, "CUBLAS_WORKSPACE_CONFIG_before": ":4096:8",
            "CUBLAS_WORKSPACE_CONFIG_after": ":4096:8"}
        context["versions"] = runtime_stdlib(context)
        attempts.phase("normal_CUDA_source_and_binary_admission")
        from pilot_model import runtime, modules, runtime_custody
        device, sampler = runtime(context)
        cuda_started = True
        atomic_json(output/"RUNTIME_PROFILE_TRANSITION.json", context["runtime_profile_transition"])
        attempts.phase("qualified_native_model_source_imports")
        mods = modules(context)
        driver_module_custody(context)
        runtime(context)
        from pilot_data import load_data
        attempts.phase("authenticated_complete_TRAIN_raw_VALID_load")
        data = load_data(context, device)
        from pilot_evaluate import evaluator
        from paired_fit import fit
        attempts.phase("fixed_fresh_fit")
        result = fit(context, mods, data, sampler, evaluator(context), device, output,
                     args.unit, args.base_seed, attempts)
        import torch
        attempts.phase("final_CUDA_and_inclusive_accounting")
        torch.cuda.synchronize(0)
        result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0), peak_host_rss_bytes=host_peak_bytes(),
                      runtime_profile_transition=context["runtime_profile_transition"],
                      runtime_profile_transition_receipt=receipt(output/"RUNTIME_PROFILE_TRANSITION.json"),
                      runtime_profile_final=runtime_custody(context))
        driver_module_custody(context)
        atomic_json(output/"COMPLETE.json", result)
        result["inclusive_accounting"] = attempts.finish("COMPLETE")
        atomic_json(output/"COMPLETE.json", result)
        print("STAGE_COMPLETE fit arm="+args.unit+" seed="+str(args.base_seed), flush=True)
    except BaseException as error:
        attempts.row["failure_peak_host_rss_bytes"] = host_peak_bytes()
        if cuda_started:
            try:
                import torch
                torch.cuda.synchronize(0)
                attempts.row.update(failure_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                                    failure_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
            except BaseException as observation_error:
                attempts.row["failed_CUDA_accounting"] = {"type": type(observation_error).__name__, "condition": str(observation_error)}
        accounting = attempts.finish("FAILED", error)
        atomic_json(output/"FAILED.json", {"schema": "ncnc-paired-predictive-failed-cell-v1", "identity": context["identity"],
            "arm": args.unit, "seed": args.base_seed, "status": "FAILED", "failure": attempts.row["failure"],
            "inclusive_accounting": accounting, "test_file_opened": False, "automatic_retry_or_resume": False})
        raise
    finally:
        os.close(lock)


if __name__ == "__main__":
    main()
