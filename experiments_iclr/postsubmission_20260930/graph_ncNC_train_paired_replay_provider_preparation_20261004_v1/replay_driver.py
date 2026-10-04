"""Independent future root-gated provider CLI. No fit, evaluation or checkpoint stage."""
import argparse
import resource
import sys
from time import perf_counter
from replay_common import atomic_json, file_sha, preflight, qualification_pins, utc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True)
    parser.add_argument("--stage", choices=("fabricated", "full_batch", "generate_epoch"), required=True)
    parser.add_argument("--master-seed", type=int, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    context = preflight(args.release, args.stage, args.epoch, args.master_seed, args.output)
    output = context["output"]
    output.mkdir(parents=True, mode=0o700)
    attempt = {"schema": "ncnc-TRAIN-replay-attempt-v1", "invocation": context["invocation"],
               "provider_manifest_sha256": context["provider_manifest_sha256"], "release_sha256": context["release_sha256"],
               "status": "RUNNING", "UTC": utc(), "no_model_or_fit": True, "no_VALID_or_TEST": True}
    atomic_json(output/"ATTEMPT.json", attempt)
    started = perf_counter()
    try:
        # Ordinary runtime authentication is charged and precedes numerical work.
        from replay_runtime import load_train_only, runtime
        rt = runtime(context)
        torch = rt["torch"]
        from replay_qualification import fabricated, full_batch
        from replay_provider import EpochReplay, draw_epoch, isolated_rng, write_epoch
        # No qualification/provider state is a scientific model donor.
        with isolated_rng(rt, 2026100401):
            if args.stage == "fabricated":
                result = fabricated(rt, args.master_seed, args.epoch, output)
            else:
                if args.stage == "full_batch":
                    qualification_pins(context, rt["identity"])
                data = load_train_only(rt)
                if args.stage == "full_batch":
                    result = full_batch(rt, data, args.master_seed, args.epoch, output)
                else:
                    qualification_pins(context, data["identity"])
                    permutation, negatives = draw_epoch(rt, data, args.master_seed, args.epoch)
                    epoch = write_epoch(rt, data, permutation, negatives, args.master_seed, args.epoch, output/"trace")
                    epoch_sha = file_sha(output/"trace"/"EPOCH.json")
                    replayed = EpochReplay(rt, data, output/"trace", epoch_sha)
                    require_equal = torch.equal(replayed.permutation, permutation) and torch.equal(replayed.negatives, negatives)
                    if not require_equal:
                        raise RuntimeError("Generated epoch full integer roundtrip differs")
                    result = {"epoch_receipt": "trace/EPOCH.json", "full_batches": epoch["full_batches"],
                              "epoch_sha256": epoch_sha, "full_integer_roundtrip_verified": True,
                              "full_negative_rows": epoch["full_negative_rows"],
                              "trace_integer_file_bytes": sum(p["bytes"] for p in epoch["files"].values()),
                              "work": {"native_sampler_draws": 1, "native_permutations": 1,
                                       "record_graph_rebuilds": 17, "full_query_support_enumerations": 34,
                                       "model_forwards": 0, "optimizer_updates": 0}, "prospective_only": True}
        torch.cuda.synchronize(0)
        result.update(schema="ncnc-TRAIN-replay-qualification-v1" if args.stage != "generate_epoch" else "ncnc-TRAIN-replay-generation-v1",
                      stage=args.stage, status="PASS", provider_manifest_sha256=context["provider_manifest_sha256"],
                      runtime_identity=rt["identity"]["runtime"], source_identity=rt["identity"]["source"],
                      no_model_or_fit=True, no_VALID_or_TEST=True, state_donor=False, UTC=utc(),
                      wall_seconds=perf_counter()-started, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                      CPU_max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024),
                      runtime_profile_transition=rt["runtime_profile_transition"])
        atomic_json(output/"RECEIPT.json", result)
        attempt.update(status="PASS", receipt="RECEIPT.json", completed_UTC=utc())
        atomic_json(output/"ATTEMPT.json", attempt)
    except BaseException as exc:
        attempt.update(status="FAILED", error_type=type(exc).__name__, error=str(exc), completed_UTC=utc(),
                       wall_seconds=perf_counter()-started,
                       CPU_max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024),
                       no_success_receipt=True)
        atomic_json(output/"ATTEMPT.json", attempt)
        raise


if __name__ == "__main__":
    main()
