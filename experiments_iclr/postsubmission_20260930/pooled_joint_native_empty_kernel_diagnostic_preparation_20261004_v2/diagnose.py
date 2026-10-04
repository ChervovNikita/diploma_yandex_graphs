#!/usr/bin/env python3
"""Separately released native CUDA diagnostic; no qualification or fit."""
import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
CASES = ("exact", "isolated_depth0", "empty_depth0", "isolated_spmm", "empty_spmm")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def module(name, path):
    prior = sys.modules.get(name)
    if prior is not None:
        require(Path(prior.__file__).resolve() == Path(path).resolve(), "Module shadowed")
        return prior
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def verify_own_manifest(expected):
    # Authenticate binding bytes before they can name any importable module.
    require(sha(HERE / "MANIFEST.json") == expected, "Diagnostic manifest differs")
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    for row in manifest["files"]:
        path = (HERE / row["path"]).resolve()
        require(path.is_relative_to(HERE) and path.stat().st_size == row["bytes"]
                and sha(path) == row["sha256"], "Diagnostic payload differs: " + row["path"])


def preflight(args):
    require("torch" not in sys.modules, "Stdlib gate must precede Torch")
    release_path = Path(args.release).resolve()
    release = json.loads(release_path.read_text())
    bindings = json.loads((HERE / "SOURCE_BINDINGS.json").read_text())
    require(release.get("schema") == "pooled-native-empty-diagnostic-root-release-v1"
            and release.get("execution_enabled") is True and release.get("root_authorization_reference"),
            "Separate explicit diagnostic root release required")
    manifest_sha = sha(HERE / "MANIFEST.json")
    require(release.get("preparation_manifest_sha256") == manifest_sha
            and release.get("fits_authorized") is False and release.get("data_reads_authorized") is False
            and release.get("state_donor") is False and release.get("VALID_or_TEST_authorized") is False
            and release.get("old_checkpoint_reads_authorized") is False, "Diagnostic source/scope differs")
    verify_own_manifest(manifest_sha)
    research = Path(release["research_root"]).resolve()
    paths = {}
    for row in bindings["files"]:
        path = (research / row["relative_path"]).resolve()
        require(path.is_relative_to(research) and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
                "Bound source/failure bytes differ: " + row["key"])
        paths[row["key"]] = path
    base = module("qualification_common", paths["native_common"])
    base.verify_manifest(HERE, manifest_sha)
    base.verify_manifest(paths["native_common"].parent, bindings["native_preparation_manifest_sha256"])
    original_bindings, original_paths = base.source_custody(research)
    failure = json.loads(paths["failed_receipt"].read_text())
    supervisor = json.loads(paths["failed_supervisor"].read_text())
    require(failure["schema"] == "pooled-native-batch-qualification-v1" and failure["stage"] == "native"
            and failure["status"] == "FAIL" and failure["data_files_opened"] == [] and failure["fits"] == 0
            and failure["VALID_evaluations"] == 0 and failure["TEST_file_opened"] is False
            and failure["old_checkpoint_opened"] is False and failure["retained_engineering_state_files"] == []
            and supervisor["child_exit_code"] == 1 and supervisor["bound_failure"] is None,
            "Bound failure is not the preserved native kernel attempt")
    require(release.get("runtime_profile") == "ordinary-authenticated-then-V5-deterministic"
            and release["cuda_visible_devices"] == failure["runtime_identity"]["profile"]["CUDA_VISIBLE_DEVICES"]
            and os.environ.get("CUDA_VISIBLE_DEVICES") == release["cuda_visible_devices"]
            and os.environ.get("CUDA_LAUNCH_BLOCKING") is None
            and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "Preserved process profile/device required")
    runtime = json.loads(original_paths["runtime_authority"].read_text())
    require(Path(sys.executable).resolve() == Path(runtime["interpreter_path"]).resolve()
            and sha(sys.executable) == runtime["interpreter_sha256"] and sys.platform == "linux"
            and research == Path(runtime["research_root"]).resolve(), "Pinned ordinary interpreter/root required")
    output = Path(args.output).resolve()
    require(output.is_relative_to(research) and not output.is_relative_to(HERE)
            and not any((p/"MANIFEST.json").exists() for p in (output,*output.parents)),
            "Fresh unsealed project output required")
    require(not output.exists() if not args.worker else (output/"STARTED.json").is_file(), "Own fresh output required")
    invocation = {"case":args.case,"output_directory":str(output),"model_seed":610041,"fixture_seed":91234}
    require(invocation in release.get("authorized_invocations", []), "Exact diagnostic invocation missing")
    limits = json.loads((HERE/"LIMITS.json").read_text())
    require(release.get("limits") == limits, "Diagnostic bounds differ")
    return {"base":base,"paths":{**original_paths, **paths},"bindings":original_bindings,"release":release,
            "release_path":release_path,"release_sha256":sha(release_path),"research":research,"output":output,
            "plan":{"resource_limits":limits},"case":args.case,"failure":failure,"runtime":runtime,
            "diagnostic_bindings":bindings,
            "identity":{"diagnostic_manifest_sha256":manifest_sha,
                        "native_preparation_manifest_sha256":bindings["native_preparation_manifest_sha256"],
                        "failed_receipt_sha256":sha(paths["failed_receipt"]),
                        "runtime_authority_sha256":sha(original_paths["runtime_authority"])}}


def final_custody(context, *, imported):
    """File/module identity checks only, also safe after a failed CUDA call."""
    base = context["base"]
    base.verify_manifest(HERE, context["identity"]["diagnostic_manifest_sha256"])
    require(sha(context["release_path"]) == context["release_sha256"], "Release changed during diagnostic")
    for row in context["diagnostic_bindings"]["files"]:
        base.read_pin(context["research"] / row["relative_path"], row)
    base.source_custody(context["research"])
    runtime = context["runtime"]
    require(sha(runtime["interpreter_path"]) == runtime["interpreter_sha256"], "Interpreter changed")
    for pin in runtime["runtime_source_pins"]:
        require(sha(pin["path"]) == pin["sha256"], "Runtime source changed")
    for pin in runtime["runtime_binary_files"]:
        base.read_pin(pin["path"], pin)
    require(sha(runtime["negative_sampler"]["path"]) == runtime["negative_sampler"]["sha256"], "Sampler changed")
    if imported:
        for row in context["bindings"]["module_paths"]:
            loaded = sys.modules.get(row["module"])
            require(loaded is not None and Path(loaded.__file__).resolve() == context["paths"][row["key"]],
                    "Imported source changed: " + row["module"])
        for name,key in (("qualification_common","native_common"),("_diagnostic_native_checks","native_checks")):
            require(Path(sys.modules[name].__file__).resolve() == context["paths"][key], "Diagnostic module changed")
    for pin in context.get("actual_operator_sources",{}).values():
        require(sha(pin["path"]) == pin["sha256"], "Actual operator source changed")
    return {"status":"PASS_FILE_CUSTODY_ONLY","GPU_operations":False,"module_paths_checked":imported}


def run_case(context, rt, mods, checks, trace):
    torch = rt["torch"]
    x,pairs,graph,positive,negative = checks.fixture(rt["device"])
    # Preserve the original already-successful coordinate check before the
    # failing path; instrumentation fences this boundary as well.
    trace.call("same_fabricated_native_support", checks.native_support,
               (mods,graph,torch.cat((positive,negative)),rt["graphs"].enumerate_neighbors(graph,torch.cat((positive,negative)))))
    if context["case"] == "exact":
        return trace.call("unchanged_direct_native", checks.direct_native,
                          (mods,rt["device"],x,graph,positive,checks.Audit()))
    (encoder,decoder),_ = mods["count_model"].make_native(mods,610041,64,rt["device"])
    encoder.eval(); decoder.eval()
    adj = sys.modules["count_model"].adjacency(graph)
    h = encoder(x,adj)
    query = positive[-1:].T if context["case"].startswith("isolated") else positive.new_empty((2,0))
    trace.event("CASE_GEOMETRY", case=context["case"], query_columns=query.shape[1], full_nodes=graph.nodes,
                feature_width=64, expected_common_nonzeros=0)
    if context["case"].endswith("depth0"):
        # Full native xlin/overlap/spmm/decode remains intact, even Nq=0.
        return decoder(h,adj,query,depth=0)
    outer = h + decoder.xlin(h)
    # Build the exact source-native common-support SparseTensor rather than
    # substitute a numerical zero or hand-constructed sparse representation.
    cn = mods["native"].adjoverlap(adj,adj,query,calresadj=False,cnsampledeg=-1,ressampledeg=-1)
    require(tuple(cn.sizes()) == (query.shape[1],graph.nodes) and cn.nnz() == 0,
            "Control native common-support geometry differs")
    trace.event("CONTROL_NATIVE_COMMON_SUPPORT", sizes=list(cn.sizes()), nnz=cn.nnz())
    return mods["native"].spmm_add(cn,outer)


def worker(context):
    base,output = context["base"],context["output"]
    receipt = {"schema":"pooled-native-empty-kernel-diagnostic-v1","identity":context["identity"],
               "case":context["case"],"status":"IN_PROGRESS","source_only_result":False,
               "native_qualification_pass":False,"data_files_opened":[],"fits":0,"state_donor":False,
               "VALID_evaluations":0,"TEST_file_opened":False,"old_checkpoint_opened":False,
               "kernel_profile_server_or_model_source_changed":False,"synchronization_instrumentation":True}
    base.atomic_json(output/"RECEIPT.json", receipt)
    trace = rt = None
    started = time.monotonic()
    try:
        rt,mods = base.assemble(context)
        require(rt["identity"]["runtime"] == context["failure"]["runtime_identity"], "Preserved failure runtime identity differs")
        checks = base.load_module("_diagnostic_native_checks", context["paths"]["native_checks"])
        observer = base.load_module("_pooled_operator_trace", HERE/"operator_trace.py")
        receipt["runtime_identity"] = rt["identity"]["runtime"]
        # Pin actual Python operator/observer source identities alongside the
        # already authenticated sparse/scatter binaries; no source is edited.
        import inspect
        operator_sources = {}
        for name,fn in (("spmm_add",mods["native"].spmm_add),("adjoverlap",mods["native"].adjoverlap),
                        ("TorchDispatchMode",observer.TorchDispatchMode)):
            path = Path(inspect.getsourcefile(fn)).resolve()
            operator_sources[name] = {"path":str(path),"sha256":sha(path)}
        receipt["actual_operator_source_identity"] = operator_sources
        context["actual_operator_sources"] = operator_sources
        trace = observer.Trace(output,base.atomic_json,limits=context["plan"]["resource_limits"])
        rt["torch"].cuda.synchronize(0)
        with observer.native_boundaries(mods,trace), observer.SynchronizedOperators(trace):
            run_case(context,rt,mods,checks,trace)
        require(sys.modules["pilot_model"].runtime_settings() == rt["identity"]["runtime"]["profile"], "Final profile changed")
        receipt["final_source_custody"] = final_custody(context, imported=True)
        receipt["status"] = "COMPLETED_DIAGNOSTIC_NO_QUALIFICATION"
    except BaseException as exc:
        receipt.update(status="ERROR_OBSERVED_NO_QUALIFICATION",exception_type=type(exc).__name__,exception=str(exc))
        (output/"TRACEBACK.txt").write_text(traceback.format_exc())
        # Never continue numerical work or probe GPU RNG after a CUDA failure.
        receipt["first_error"] = None if trace is None else trace.first_error
        try:
            receipt["final_source_custody"] = final_custody(context, imported=rt is not None)
        except BaseException as custody_error:
            receipt["final_source_custody"] = {"status":"FAILED_FILE_CUSTODY_ONLY","exception":str(custody_error)}
        raise
    finally:
        receipt.update(wall_seconds=time.monotonic()-started,trace_events=0 if trace is None else trace.counter,
                       last_event=None if trace is None else trace.last,
                       synchronized_CUDA_peak_bytes=None if trace is None else trace.peak_bytes,
                       RNG_after_failed_operation="NOT_PROBED" if receipt["status"].startswith("ERROR") else "NO_INSTRUMENTATION_RNG_CALLS_SOURCE_ONLY")
        if trace is not None:
            trace.close()
        base.atomic_json(output/"RECEIPT.json",receipt)


def supervisor(context):
    import fcntl
    output,base = context["output"],context["base"]
    output.mkdir(parents=True,mode=0o700)
    lock = os.open(output/".RUN_LOCK",os.O_RDWR|os.O_CREAT,0o600)
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    limits = context["plan"]["resource_limits"]
    command = [sys.executable,"-B",str(HERE/"diagnose.py"),"--case",context["case"],"--release",str(context["release_path"]),
               "--output",str(output),"--worker"]
    receipt = {"schema":"pooled-native-empty-diagnostic-owned-supervisor-v1","identity":context["identity"],
               "command":command,"status":"IN_PROGRESS","limits":limits,"state_donor":False,"fits":0}
    base.atomic_json(output/"STARTED.json",receipt)
    started = time.monotonic(); process = None; bound = None; rss_peak = 0
    try:
        with (output/"STDOUT.txt").open("wb") as out,(output/"STDERR.txt").open("wb") as err:
            process = subprocess.Popen(command,stdout=out,stderr=err,cwd=context["research"])
            while process.poll() is None:
                status = Path('/proc')/str(process.pid)/'status'
                try:
                    rss = next((int(line.split()[1])*1024 for line in status.read_text().splitlines() if line.startswith('VmRSS:')),0)
                    rss_peak = max(rss_peak,rss)
                except FileNotFoundError:
                    pass
                size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
                if time.monotonic()-started > limits['wall_seconds']: bound='WALL'
                elif rss_peak > limits['host_RSS_bytes']: bound='HOST_RSS'
                elif size > limits['output_bytes']: bound='OUTPUT_BYTES'
                if bound:
                    process.kill(); break
                time.sleep(.5)
            process.wait()
        receipt['status'] = 'BOUNDED_PROCESS_COMPLETE' if bound is None else 'BOUND_STOP_NO_QUALIFICATION'
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.kill(); process.wait()
        receipt.update(status='SUPERVISOR_ERROR_NO_QUALIFICATION',exception=str(exc))
        raise
    finally:
        receipt.update(wall_seconds=time.monotonic()-started,owned_child_exit_code=None if process is None else process.returncode,
                       bound_failure=bound,observed_child_peak_RSS_bytes=rss_peak)
        base.atomic_json(output/'SUPERVISOR_RECEIPT.json',receipt)
        os.close(lock)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=CASES,required=True)
    parser.add_argument('--release',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    context=preflight(args)
    worker(context) if args.worker else supervisor(context)


if __name__=='__main__':
    main()
