#!/usr/bin/env python3
"""Root-released driver. No TEST stage, implicit download, or installations."""
from argparse import ArgumentParser
from time import perf_counter
from pilot_common import (UNITS, preflight, admit_invocation, fresh_output,
                          runtime_stdlib, fit_qualification, atomic_json, file_sha)
from pilot_common import lock_output
from pilot_accounting import Attempts


def arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root-release", required=True)
    parser.add_argument("--stage", required=True, choices=("synthetic", "valid_engineering", "resource_engineering", "fit", "family_lock"))
    parser.add_argument("--unit", required=True, choices=UNITS)
    parser.add_argument("--base-seed", required=True, type=int, choices=range(5))
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    started = perf_counter()
    args = arguments()
    context = preflight(args.root_release, args.stage)
    admit_invocation(context, args.stage, args.unit, args.base_seed, args.output, args.resume)
    if args.stage == "fit":
        fit_qualification(context)
    output = fresh_output(args.output, resume=args.resume)
    output_lock = lock_output(output)
    context["output"] = output
    attempts = Attempts(output, context, args.stage, args.unit, args.base_seed, started=started)
    cuda_started = False
    try:
        if args.stage == "family_lock":
            from pilot_lock import run
            result = run(context, output, attempts)
            name = "FAMILY_LOCK.json"
        else:
            attempts.phase("stdlib_interpreter_and_distribution_admission")
            context["versions"] = runtime_stdlib(context)
            from pilot_model import runtime, modules
            attempts.phase("normal_host_CUDA_source_and_binary_admission")
            device, sampler = runtime(context)
            cuda_started = True
            attempts.phase("sealed_qualified_model_source_imports")
            mods = modules(context)
            mods["design"].validate_plan(context["plan"])
            if args.stage == "synthetic":
                from pilot_synthetic import run
                result = run(context, mods, sampler, device, output, attempts)
                name = "QUALIFICATION.json"
            else:
                from pilot_data import load_data
                attempts.phase("authenticated_complete_TRAIN_raw_VALID_load_and_transfer")
                data = load_data(context, device)
                if args.stage == "fit":
                    from pilot_fit import fit
                    from pilot_evaluate import evaluator
                    attempts.phase("bound_official_evaluator")
                    metric = evaluator(context)
                    result = fit(context, mods, data, sampler, metric, device, output,
                                 args.unit, args.base_seed, args.resume, attempts)
                    name = "COMPLETE.json"
                else:
                    from pilot_engineering import run
                    result = run(context, mods, data, sampler, device, output, args.stage,
                                 args.unit, args.base_seed, attempts)
                    name = "QUALIFICATION.json"
            import torch
            attempts.phase("final_CUDA_synchronization_and_memory_accounting")
            torch.cuda.synchronize(0)
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        # Charge the first complete-result serialization. Final accounting and
        # its own terminal writes have their conventional tiny unmeasured tail.
        attempts.phase("final_receipt_serialization")
        atomic_json(output / name, result)
        result["inclusive_accounting"] = attempts.finish("COMPLETE")
        atomic_json(output / name, result)
        if args.stage == "family_lock":
            from pilot_lock import commit_closure_markers
            commit_closure_markers(context, output, result)
        print("STAGE_COMPLETE stage=" + args.stage + " unit=" + args.unit + " base_seed=" + str(args.base_seed) + " receipt=" + str(output / name), flush=True)
    except Exception as error:
        if cuda_started:
            try:
                import torch
                torch.cuda.synchronize(0)
                attempts.row["failure_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
                attempts.row["failure_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
            except Exception as sync_error:
                attempts.row["failed_CUDA_accounting"] = {"exception_type": type(sync_error).__name__, "condition": str(sync_error)}
        accounting = attempts.finish("FAILED", error)
        def receipt(path):
            return {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path)}
        atomic_json(output / "FAILED.json", {"schema": "ncnc-pilot-failed-unit-v1",
            "identity": context["identity"], "stage": args.stage, "unit": args.unit, "base_seed": args.base_seed,
            "status": "FAILED", "failure": attempts.row["failure"], "inclusive_accounting": accounting,
            "attempts_receipt": receipt(output / "ATTEMPTS.json"),
            "journal_receipt": receipt(output / "JOURNAL.json") if (output / "JOURNAL.json").exists() else None,
            "unique_fits_planned": 4 if args.unit == "native_bank4" else 1,
            "predictive_values_exposed": False, "test_file_opened": False,
            "terminal_family_disposition_requires_explicit_root_release": True})
        raise
    finally:
        import os
        os.close(output_lock)


if __name__ == "__main__":
    main()
