"""Six terminal selected-VALID collections and a saved-prediction addition.

The collector and genuine structural v2 producer remain unchanged. Module
import is stdlib-only; scientific access is guarded and allocation-only.
"""
import argparse
import fcntl
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

PLAN_SHA = "68f884a0aae025a8ceca0196553760f85c9756b0e1c9aaab8a47e74044460e3f"
HOST = "anogena-2-0"
GPU = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"
EXPANSION = Path("experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1")
SEEDS = [101, 202, 303]
BANKS = ['structural_private_risk', 'permuted_private_risk']
FIELDS = ["mean_member_AUC", "worst_member_AUC", "pool_AUC", "pool_minus_mean_member_AUC_pp",
          "mean_member_BCE", "worst_member_BCE", "uniform_graph_BCE", "all_members_reversed",
          "pool_reversed_despite_some_member_correct", "pool_reversed_despite_strict_majority_correct"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def required_hash(value):
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            "Actual terminal SHA256 required; deferred/null pins cannot be admitted")
    return value


def module(path, name):
    definition = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(value)
    return value


def terminal_gate(end, outcomes, jobs):
    expected = [(seed, bank) for seed in SEEDS for bank in BANKS]
    require([(j['seed'], j['bank']) for j in jobs] == expected and len({j['id'] for j in jobs}) == 6,
            'The exact six ordered banks are required')
    require(end.get('status') == 'complete' and end.get('TEST_accessed') is False and end.get('outcomes') == outcomes and
            len(outcomes) == 6 and [r.get('id') for r in outcomes] == [j['id'] for j in jobs] and all(
                r.get('status') == 'complete' and r.get('returncode') == 0 and not r.get('timed_out') and
                r.get('RESULT', {}).get('sha256') for r in outcomes), 'All six successful terminal fits required')


def collection_gate(value, fit, result_pin):
    require(value.get("status") in ("complete", "complete_with_sanity_warning") and
            value.get("role") == "VALID" and value.get("TEST_accessed") is False and
            value.get("job_id") == fit["job_id"] and value.get("job_hash") == fit["job_hash"] and
            value.get("selected_epoch") == fit["selected_epoch"] and
            value.get("fit_receipt") == result_pin and value.get("members") == 4 and
            value.get("evaluated_VALID_graphs") == 151 and not value.get("close_error"),
            "Collection identity, complete VALID coverage or closure differs")
    require(value["selected_checkpoint"]["sha256"] == fit["selected_checkpoint"]["sha256"] and
            value["saved_VALID_export"]["sha256"] == fit["selected_VALID_export"]["sha256"] and
            value["actual_work"]["TRAIN_callbacks"] == value["actual_work"]["optimizer_steps"] == 0,
            "Collection selected authority or zero-training contract differs")


def contrast(seed, candidate, reference):
    return {'seed': seed, 'comparison': candidate['bank'] + '_vs_' + reference['bank'],
            'candidate_bank': candidate['bank'], 'reference_bank': reference['bank'],
            'delta_pool_AUC_pp': 100*(candidate['pool_AUC']-reference['pool_AUC']),
            'delta_uniform_graph_BCE': candidate['uniform_graph_BCE']-reference['uniform_graph_BCE'],
            'delta_mean_member_AUC_pp': 100*(candidate['mean_member_AUC']-reference['mean_member_AUC']),
            'delta_worst_member_AUC_pp': 100*(candidate['worst_member_AUC']-reference['worst_member_AUC']),
            'delta_mean_member_BCE': candidate['mean_member_BCE']-reference['mean_member_BCE'],
            'delta_saved_native_VALID_BCE': candidate['saved_native_VALID_BCE']-reference['saved_native_VALID_BCE']
                if reference['saved_native_VALID_BCE'] is not None else None,
            'delta_pool_minus_mean_member_AUC_pp': candidate['pool_minus_mean_member_AUC_pp']-reference['pool_minus_mean_member_AUC_pp'],
            'delta_common_member_reversals': candidate['all_members_reversed']-reference['all_members_reversed'],
            'delta_pool_losses_despite_correct_member': candidate['pool_reversed_despite_some_member_correct']-reference['pool_reversed_despite_some_member_correct']}


def matched_pairs(helper, reference, candidate, labels):
    """Aligned positive-negative pairs; descriptive counts, never replicates."""
    before = helper.margins(reference['pool'], labels)
    after = helper.margins(candidate['pool'], labels)
    old_members = list(zip(*(helper.margins(z, labels) for z in reference['members'])))
    new_members = list(zip(*(helper.margins(z, labels) for z in candidate['members'])))
    old_common = [all(v < 0 for v in col) for col in old_members]
    new_common = [all(v < 0 for v in col) for col in new_members]
    acquired = [any(v > 0 for v in col) for col in new_members]
    return {**helper.compare_pairs(before, after), 'pairs': len(before),
        'reference_all_members_reversed': sum(old_common),
        'candidate_all_members_reversed': sum(new_common),
        'reference_common_with_any_candidate_member_correct': sum(a and b for a, b in zip(old_common, acquired)),
        'reference_common_pool_strict_repairs': sum(a and b > 0 for a, b in zip(old_common, after)),
        'reference_common_acquired_correct_member_but_pool_reversed': sum(
            a and b and p < 0 for a, b, p in zip(old_common, acquired, after)),
        'reference_common_acquired_correct_member_but_pool_tied': sum(
            a and b and p == 0 for a, b, p in zip(old_common, acquired, after)),
        'reference_common_persisting_all_member_reversals': sum(a and b for a, b in zip(old_common, new_common)),
        'reference_common_no_longer_common': sum(a and not b for a, b in zip(old_common, new_common)),
        'new_common_member_reversals_outside_reference_common': sum(not a and b for a, b in zip(old_common, new_common)),
        'tie_law': 'AUC credit: positive 1, tied .5, reversed 0; strict repairs/harms exclude ties',
        'population': 'All aligned positive-negative VALID pairs; common-error subset is separately identified'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--mode", choices=("commands", "collect", "analyze"), default="commands")
    args = parser.parse_args()
    require(socket.gethostname() == HOST and subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=10).splitlines() == [GPU],
        "Dedicated allocation guard required before project reads")
    root = Path(args.project_root).resolve()
    expansion = root / EXPANSION

    def safe(path):
        path = Path(path)
        value = (path if path.is_absolute() else expansion / path).resolve()
        require(value.is_relative_to(root), "Project-contained paths required")
        return value

    def verified(pin):
        path = safe(pin["path"])
        require(sha(path) == pin["sha256"] and ("bytes" not in pin or path.stat().st_size == pin["bytes"]),
                "Receipted input changed: " + str(path))
        return path

    def read(path, expected=None):
        path = safe(path)
        raw = path.read_bytes()
        pin = {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        require(expected is None or pin["sha256"] == expected, "Snapshot differs: " + str(path))
        return json.loads(raw), pin

    plan, plan_pin = read(Path(__file__).with_name("COLLECTION_PLAN.json"), PLAN_SHA)
    require(Path(plan["project_root"]).resolve() == root, "Frozen project root differs")
    sources = {p["path"]: verified(p) for p in plan["source_pins"]}
    admission, admission_pin = read(args.admission, args.admission_sha256)
    manifest, manifest_pin = read(Path(__file__).with_name('MANIFEST.json'))
    for row in manifest['files']:
        verified(dict(row, path=str(Path(__file__).parent / row['path'])))
    require(admission.get("schema") == "root-reviewed-structural-six-VALID-addition-v1" and
            admission.get("approved") is True and admission.get("resource_admission_approved") is True and
            admission.get("PLAN_sha256") == PLAN_SHA and admission.get("selected_VALID_sha256") == sha(__file__) and
            admission.get('pilot_manifest_sha256') == manifest_pin['sha256'] and
            admission.get("allocation_GPU_UUID") == GPU and admission.get("maximum_concurrent_collectors") == 1,
            "Exact source and one-slot root resource admission required")
    require(Path(sys.prefix).resolve() == (root / '.venv').resolve(), 'Original repository virtualenv required')
    other_pids = admission["allowed_other_GPU_pids"]
    require(isinstance(other_pids, list) and len(other_pids) <= 1 and
            all(type(p) is int and p > 0 for p in other_pids), "Bind the older sole GPU peer or none")
    launch = module(sources[plan["launch_helpers"]], "_residual_selected_launch_helpers")
    helper = module(sources[plan["analysis_helpers"]], "_residual_saved_analysis_helpers")
    study = safe(plan["study_output"])
    end, end_pin = read(study / "END.json", required_hash(admission["terminal_END_sha256"]))
    outcomes, outcomes_pin = read(study / "OUTCOMES.json", required_hash(admission["terminal_OUTCOMES_sha256"]))
    terminal_gate(end, outcomes, plan["jobs"])
    # New score-bearing files are first read after the complete six-fit gate.
    spec, _ = read(sources[plan["SPEC"]])
    for pin in spec["source_pins"] + spec["data_role_pins"]:
        verified(pin)
    require(all(not safe(path).exists() for path in spec["absent_role_overrides"]), "Original VALID role authority required")
    qualification_pins = []
    for qualification in plan['CPU_qualifications']:
        value, qualifier_pin = read(verified(qualification['authority']))
        result = value['result']
        require(value['status'] == 'complete' and result['status'] == 'passed' and value['exit_code'] == 0 and
                result['fresh_restore_VALID_bit_exact'] is True and result['TEST_accessed'] is False,
                'Both actual v2 CPU lifecycle qualifications required')
        qualification_pins.append(qualifier_pin)
    fits, fit_pins, commands = [], [], []
    collected, analysis_output = safe(admission["collection_output"]), safe(admission["analysis_output"])
    require(collected != analysis_output and not collected.is_relative_to(study) and
            not analysis_output.is_relative_to(study), "Fresh analysis outputs must be outside scientific fit tree")
    for index, binding in enumerate(plan["jobs"]):
        fit_dir = safe(binding["fit"])
        fit, fit_pin = read(fit_dir / "RESULT.json", required_hash(outcomes[index]["RESULT"]["sha256"]))
        job, _ = read(fit_dir / "JOB.json")
        frozen_job, _ = read(sources[binding["source_JOB"]], binding["source_JOB_sha256"])
        require(job == frozen_job and digest(job) == binding["job_hash"] == fit["job_hash"] and
                fit["job_id"] == binding["id"] and fit["seed"] == binding["seed"] and
                fit["status"] == "complete" and fit.get("TEST_accessed") is False and
                job["TEST_access"] is False and job["device"] == "cuda:0" and
                fit["policy"] == job["policy"] and not fit.get("failure") and not fit.get("close_error"),
                "Exact genuine completed structural v2 JOB/RESULT required")
        runtime = fit["runtime_binding"]
        outcome = outcomes[index]
        require(all(outcome.get(k) == fit[k] for k in ("selected_epoch", "completed_epochs", "stop_reason")),
                "Scientific terminal outcome/fit result join differs")
        require(runtime["callback_sha256"] == job["callback_sha256"] and
                runtime["extension_sha256"] == job["session_transform"]["sha256"] and
                set(fit["runtime_sources"]) == set(runtime["runtime_sha256"]) ==
                {"run_fit.py", "adapters.py", "bookkeeping.py", "common.py"}, "Producer callback/STATE extension differs")
        for name, pin in fit["runtime_sources"].items():
            require(verified(pin) == safe("runtime/" + name) and
                    runtime["runtime_sha256"][name] == pin["sha256"], "Original runtime source differs")
        for path, expected in runtime["dependency_sha256"].items():
            verified({"path": path, "sha256": expected})
        for filename, key in (("selected.pt", "selected_checkpoint"), ("selected_VALID.pt", "selected_VALID_export")):
            require(safe(fit[key]["path"]) == fit_dir / filename and (fit_dir / filename).is_file(),
                    "Actual terminal selected checkpoint/export missing")
        fits.append(fit); fit_pins.append(fit_pin)
        commands.append([sys.executable, "-B", "-u", str(sources[plan["collector"]]), "--project-root", str(root),
                         "--fit", str(fit_dir), "--output", str(collected / binding["collection"])])
    if args.mode == "commands":
        print(json.dumps({"collectors": commands, "analysis": {"mode": "analyze", "env": {"CUDA_VISIBLE_DEVICES": ""}},
                          "terminal_END": end_pin, "terminal_OUTCOMES": outcomes_pin}, indent=2))
        return 0
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    if args.mode == "collect":
        require(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "0", GPU), "One authorized GPU required")
        lock_path = safe(admission["slot_lock_path"])
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            signal.signal(signal.SIGTERM, launch.interrupted)
            require(not collected.exists(), "Fresh collection root required; no overwrite or retry")
            collected.mkdir(parents=True)
            started = time.monotonic()
            children = [{"seed": j["seed"], "id": j["id"], "status": "not_started"} for j in plan["jobs"]]
            launch.write(collected / "START.json", {"PLAN": plan_pin, "admission": admission_pin,
                         "terminal_END": end_pin, "terminal_OUTCOMES": outcomes_pin, "fit_RESULT": fit_pins,
                         "maximum_concurrent_collectors": 1, "automatic_retries": 0})
            failure = None
            try:
                for index, command in enumerate(commands):
                    active = {int(p.strip()) for p in subprocess.check_output(
                        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=10).splitlines() if p.strip()}
                    require(active <= set(other_pids), "Unexpected GPU process; reserved collector slot unavailable")
                    before, child = time.monotonic(), None
                    children[index].update(status="started", observed_other_GPU_pids=sorted(active))
                    launch.write(collected / "OUTCOMES.json", children, True)
                    try:
                        with (collected / ("seed" + str(index).zfill(3) + ".log")).open("x") as log:
                            child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, env=env, start_new_session=True)
                            children[index]["pid"] = child.pid
                            try:
                                children[index]["exit_code"] = child.wait(timeout=plan["timeout_seconds_per_collection"])
                            except subprocess.TimeoutExpired:
                                children[index]["timed_out"] = True
                                launch.terminate(child)
                                children[index]["exit_code"] = child.returncode
                    finally:
                        if child is not None:
                            launch.terminate(child)
                        children[index]["wall_seconds"] = time.monotonic() - before
                    result, pin = read(Path(command[-1]) / "RESULT.json")
                    children[index].update(helper.collection_outcome(plan["jobs"][index]["id"], result, pin, "new"))
                    collection_gate(result, fits[index], fit_pins[index])
                    require(result["analysis_source"]["sha256"] == sha(sources[plan["collector"]]), "Unchanged collector required")
                    launch.write(collected / "OUTCOMES.json", children, True)
                    require(children[index]["exit_code"] == 0 and not children[index].get("timed_out"), "Collector failed; no retry")
            except BaseException as error:
                failure = {"type": type(error).__name__, "message": str(error), "unreceipted_work": "unknown if no child RESULT, not zero"}
                raise
            finally:
                launch.write(collected / "OUTCOMES.json", children, True)
                launch.write(collected / "END.json", {"status": "complete" if failure is None else "stopped_no_retry",
                    "children": children, "PLAN": plan_pin, "admission": admission_pin, "terminal_END": end_pin,
                    "terminal_OUTCOMES": outcomes_pin, "wall_seconds": time.monotonic() - started, "failure": failure,
                    "new_training_updates": 0, "TEST_accessed": False, "automatic_retries": 0})
        return 0
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Saved-prediction analysis requires CPU visibility only")
    return analyze(plan, plan_pin, admission_pin, fits, fit_pins, collected, analysis_output,
                   end_pin, outcomes_pin, qualification_pins, read, verified, helper, sources)


def analyze(plan, plan_pin, admission_pin, fits, fit_pins, collected, output,
            end_pin, outcomes_pin, qualification_pins, read, verified, helper, sources):
    started = time.monotonic()
    require(not output.exists(), "Fresh analysis output required")
    output.mkdir(parents=True)
    collection_records = []
    try:
        collection_end, collection_end_pin = read(collected / "END.json")
        require(collection_end.get("status") == "complete" and collection_end.get("PLAN") == plan_pin and
                collection_end.get("admission") == admission_pin and collection_end.get("terminal_END") == end_pin and
                collection_end.get("terminal_OUTCOMES") == outcomes_pin and len(collection_end.get("children", [])) == 6 and
                [r.get("id") for r in collection_end["children"]] == [j["id"] for j in plan["jobs"]] and
                all(r.get("status") in ("complete", "complete_with_sanity_warning") and
                    r.get("exit_code") == 0 and not r.get("timed_out") for r in collection_end["children"]),
                "All six admitted collections must finish before analysis")
        collections = []
        for fit, fit_pin, job, outcome in zip(fits, fit_pins, plan["jobs"], collection_end["children"]):
            value, pin = read(collected / job["collection"] / "RESULT.json")
            collection_records.append(helper.collection_outcome(job["id"], value, pin, "new"))
            collection_gate(value, fit, fit_pin)
            require(outcome["receipt"] == pin and outcome["id"] == job["id"] and
                    value["analysis_source"]["sha256"] == sha(sources[plan["collector"]]), "Collection END/source identity differs")
            verified(value["raw_tensor_receipt"])
            collections.append(value)
        prior, prior_pin = read(verified(plan["completed_33_bank_analysis"]))
        require(prior["schema"] == "residual-only-selected-VALID-33-bank-addition-v1" and
                prior["status"] in ("complete", "complete_with_warnings") and prior["TEST_accessed"] is False and
                len(prior["rows"]) == 33 and len(prior["training_fit_outcomes"]) == 39,
                "Completed 33-bank/39-fit authority required")
        uniform_inputs = []
        for seed in SEEDS:
            bank = [r for r in prior['rows'] if r['seed'] == seed and r['bank'] == plan['reference_banks']['uniform_residual']]
            require(len(bank) == 1 and len(bank[0]['source_job_ids']) == 1, 'Exact paid uniform bank required')
            identity = bank[0]['source_job_ids'][0]
            old_fit = [r for r in prior['training_fit_outcomes'] if r['job_id'] == identity]
            old_collection = [r for r in prior['route_collection_outcomes'] if r['job_id'] == identity]
            require(len(old_fit) == len(old_collection) == 1, 'Unique paid uniform fit and collection required')
            fit, fit_pin = read(verified(old_fit[0]['receipt']))
            value, value_pin = read(verified(old_collection[0]['receipt']))
            require(fit['status'] == 'complete' and fit['seed'] == seed and fit['job_id'] == identity and
                    fit['TEST_accessed'] is False and fit['selected_checkpoint'] == old_fit[0]['selected_checkpoint'] and
                    fit['selected_VALID_export'] == old_fit[0]['selected_VALID_export'], 'Paid uniform selected fit differs')
            collection_gate(value, fit, fit_pin)
            verified(fit['selected_VALID_export']); verified(value['raw_tensor_receipt'])
            uniform_inputs.append((seed, fit, value, fit_pin, value_pin))
        # Tensor libraries are first imported after every terminal/collection gate.
        import torch
        from ogb.graphproppred import Evaluator
        torch.set_num_threads(2)
        metrics = module(sources[plan["collector"]], "_residual_unchanged_binary_metrics")
        evaluator = Evaluator(name="ogbg-molbace")
        rows, warnings, comparisons = list(prior["rows"]), list(prior["warnings"]), []
        reference_ids = reference_labels = None
        predictions, reused_uniform_inputs = {}, []
        def prediction(fit, collection):
            saved = torch.load(verified(fit['selected_VALID_export']), map_location='cpu', weights_only=False)
            raw = torch.load(verified(collection['raw_tensor_receipt']), map_location='cpu', weights_only=False)
            for value in (saved, raw):
                require(value['role'] == 'VALID' and value['TEST_accessed'] is False and
                        value['job_id'] == fit['job_id'] and value['job_hash'] == fit['job_hash'] and
                        value['selected_epoch'] == fit['selected_epoch'] and
                        value['selected_checkpoint']['sha256'] == fit['selected_checkpoint']['sha256'],
                        'Selected prediction identity/epoch/authority differs')
            require(torch.equal(saved['ids'], raw['ids']) and torch.equal(saved['labels'], raw['labels']) and
                    tuple(saved['raw_logits'].shape) == tuple(saved['labels'].shape) == (151, 1) and
                    tuple(raw['member_raw_logits'].shape) == (4, 151, 1) and saved['ids'].numel() == 151 and
                    len(set(saved['ids'].reshape(-1).tolist())) == 151 and
                    int((saved['labels'] == 1).sum()) == 130 and int((saved['labels'] == 0).sum()) == 21 and
                    all(torch.isfinite(x).all().item() for x in (saved['raw_logits'], saved['labels'], raw['member_raw_logits'])) and
                    saved['validation'] == fit['selected_VALID'], 'Complete aligned selected binary VALID required')
            require(torch.equal(raw['member_raw_logits'].mean(0), raw['pooled_raw_logits']), 'Route mean law differs')
            return saved, raw
        for seed, fit, collection, fit_pin, collection_pin in uniform_inputs:
            saved, raw = prediction(fit, collection)
            if reference_ids is None: reference_ids, reference_labels = saved['ids'], saved['labels']
            require(torch.equal(saved['ids'], reference_ids) and torch.equal(saved['labels'], reference_labels),
                    'Paid uniform graph identity/order differs')
            predictions[(seed, plan['reference_banks']['uniform_residual'])] = {
                'pool': saved['raw_logits'][:, 0].double().tolist(),
                'members': [z[:, 0].double().tolist() for z in raw['member_raw_logits']]}
            pooled = metrics.binary_metrics(torch, evaluator, saved['raw_logits'], saved['labels'])
            old_row = next(r for r in prior['rows'] if r['seed'] == seed and r['bank'] == plan['reference_banks']['uniform_residual'])
            require(abs(pooled['rocauc'] - old_row['pool_AUC']) < 1e-12 and
                    abs(pooled['BCE'] - old_row['uniform_graph_BCE']) < 1e-12, 'Paid uniform saved metric differs')
            pairs, _, _ = helper.pair_diagnostics(predictions[(seed, plan['reference_banks']['uniform_residual'])]['members'],
                predictions[(seed, plan['reference_banks']['uniform_residual'])]['pool'], saved['labels'][:, 0].tolist())
            require(pairs['all_members_reversed'] == old_row['all_members_reversed'], 'Paid uniform common errors differ')
            reused_uniform_inputs.append({'seed': seed, 'fit_RESULT': fit_pin, 'collection_RESULT': collection_pin,
                'selected_VALID_export': fit['selected_VALID_export'], 'raw_tensor_receipt': collection['raw_tensor_receipt']})
        for fit, collection, job in zip(fits, collections, plan["jobs"]):
            saved, raw = prediction(fit, collection)
            for value in (saved, raw):
                require(value["role"] == "VALID" and value["TEST_accessed"] is False and
                        value["job_id"] == job["id"] and value["job_hash"] == fit["job_hash"] and
                        value["selected_epoch"] == fit["selected_epoch"] and
                        value["selected_checkpoint"]["sha256"] == fit["selected_checkpoint"]["sha256"],
                        "Actual selected prediction authority differs")
            ids, labels, pool, members = saved["ids"], saved["labels"], saved["raw_logits"], raw["member_raw_logits"]
            require(tuple(pool.shape) == tuple(labels.shape) == (151, 1) and tuple(members.shape) == (4, 151, 1) and
                    ids.numel() == 151 and len(set(ids.reshape(-1).tolist())) == 151 and
                    torch.isfinite(pool).all().item() and torch.isfinite(labels).all().item() and torch.isfinite(members).all().item() and
                    int((labels == 1).sum()) == 130 and int((labels == 0).sum()) == 21 and
                    torch.equal(ids, raw["ids"]) and torch.equal(labels, raw["labels"]), "Complete aligned binary VALID required")
            if reference_ids is None:
                reference_ids, reference_labels = ids, labels
            require(torch.equal(ids, reference_ids) and torch.equal(labels, reference_labels), "Seed graph identity/order differs")
            member_metrics = [metrics.binary_metrics(torch, evaluator, z, labels) for z in members]
            pooled = metrics.binary_metrics(torch, evaluator, pool, labels)
            require(saved["validation"] == fit["selected_VALID"] and
                    abs(pooled["rocauc"] - fit["selected_VALID"]["metrics"]["rocauc"]) < 1e-12,
                    "Saved serving metric differs from terminal selected VALID authority")
            pairs, _, manual = helper.pair_diagnostics([z[:, 0].double().tolist() for z in members],
                pool[:, 0].double().tolist(), labels[:, 0].tolist())
            require(all(abs(a - m["rocauc"]) < 1e-12 for a, m in zip(manual, member_metrics)), "Member evaluator/pair AUC differs")
            reconstructed = members.mean(0)
            require(torch.equal(reconstructed, raw["pooled_raw_logits"]), "Collector mean-logit serving law differs")
            row = {"seed": job["seed"], "bank": job["bank"], "cell": None, "origin": "new_structural_private_risk",
                   "source_job_ids": [job["id"]], "selected_epochs": [fit["selected_epoch"]], "members": member_metrics,
                   "mean_member_AUC": statistics.mean(m["rocauc"] for m in member_metrics),
                   "worst_member_AUC": min(m["rocauc"] for m in member_metrics), "pool_AUC": pooled["rocauc"],
                   "mean_member_BCE": statistics.mean(m["BCE"] for m in member_metrics),
                   "worst_member_BCE": max(m["BCE"] for m in member_metrics), "uniform_graph_BCE": pooled["BCE"],
                   "pooled_metrics": pooled, "saved_native_VALID_BCE": fit["selected_VALID"]["loss"], **pairs,
                   "reconstruction": {"sanity": collection["sanity_against_saved_export"],
                       "max_abs_pool_difference": float((reconstructed.double() - pool.double()).abs().max()),
                       "reconstructed_pool_AUC": metrics.binary_metrics(torch, evaluator, reconstructed, labels)["rocauc"]}}
            row["pool_minus_mean_member_AUC_pp"] = 100 * (row["pool_AUC"] - row["mean_member_AUC"])
            rows.append(row)
            predictions[(job['seed'], job['bank'])] = {'pool': pool[:, 0].double().tolist(),
                'members': [z[:, 0].double().tolist() for z in members]}
            if collection["status"] != "complete" or not collection["sanity_against_saved_export"]["passed"]:
                warnings.append({"seed": job["seed"], "bank": job["bank"], **row["reconstruction"]})
        for row in rows[33:]:
            for bank in plan['reference_banks'].values():
                matched = [r for r in prior['rows'] if r['seed'] == row['seed'] and r['bank'] == bank]
                require(len(matched) == 1, 'Exact paid reference bank required')
                comparisons.append(contrast(row['seed'], row, matched[0]))
            if row['bank'] == BANKS[0]:
                matched = [r for r in rows[33:] if r['seed'] == row['seed'] and r['bank'] == BANKS[1]]
                require(len(matched) == 1, 'Exact paired permutation required')
                comparisons.append(contrast(row['seed'], row, matched[0]))
        pair_comparisons = []
        for seed in SEEDS:
            for candidate, reference in ((BANKS[0], plan['reference_banks']['uniform_residual']),
                    (BANKS[1], plan['reference_banks']['uniform_residual']), (BANKS[0], BANKS[1])):
                value = matched_pairs(helper, predictions[(seed, reference)], predictions[(seed, candidate)],
                                      reference_labels[:, 0].tolist())
                scalar = next(r for r in comparisons if r['seed'] == seed and r['candidate_bank'] == candidate and r['reference_bank'] == reference)
                require(abs(value['delta_AUC_pp'] - scalar['delta_pool_AUC_pp']) < 1e-10, 'Aligned pair/scalar AUC contrast differs')
                pair_comparisons.append({'seed': seed, 'comparison': candidate + '_vs_' + reference,
                    'candidate_bank': candidate, 'reference_bank': reference, **value})
        training = list(prior["training_fit_outcomes"]) + [
            {"job_id": fit["job_id"], "receipt": pin, "status": fit["status"],
             **{k: fit.get(k) for k in helper.COST_FIELDS}, "selected_checkpoint": fit["selected_checkpoint"],
             "selected_VALID_export": fit["selected_VALID_export"], "source_metadata": fit["source_metadata"]}
            for fit, pin in zip(fits, fit_pins)]
        require(len(rows) == 39 and len(training) == len({r["job_id"] for r in training}) == 45, "39 banks/45 unique paid fits required")
        result = {"schema": "structural-private-risk-selected-VALID-39-bank-addition-v1", "status": "complete_with_warnings" if warnings else "complete",
            "scope": prior["scope"], "rows": rows, "summaries": helper.summarize(rows, "bank", FIELDS),
            "new_contrasts": comparisons, "new_contrast_summaries": helper.summarize(comparisons, "comparison",
                ["delta_pool_AUC_pp", "delta_uniform_graph_BCE", "delta_mean_member_AUC_pp", "delta_worst_member_AUC_pp",
                 "delta_mean_member_BCE", "delta_pool_minus_mean_member_AUC_pp", "delta_common_member_reversals",
                 "delta_pool_losses_despite_correct_member"]),
            'matched_pair_contrasts': pair_comparisons,
            'matched_pair_summaries': helper.summarize(pair_comparisons, 'comparison', [
                'strict_repairs', 'strict_harms', 'delta_AUC_pp', 'reference_all_members_reversed',
                'reference_common_with_any_candidate_member_correct', 'reference_common_pool_strict_repairs',
                'reference_common_acquired_correct_member_but_pool_reversed',
                'new_common_member_reversals_outside_reference_common']),
            'native_BCE_contrast_summaries': helper.summarize([r for r in comparisons if
                r['delta_saved_native_VALID_BCE'] is not None], 'comparison', ['delta_saved_native_VALID_BCE']),
            'reused_uniform_saved_inputs': reused_uniform_inputs,
            "primary_references": plan["primary_references"], "warnings": warnings, "PLAN": plan_pin, "admission": admission_pin,
            "terminal_END": end_pin, "terminal_OUTCOMES": outcomes_pin, "collection_END": collection_end_pin,
            "CPU_qualifications": qualification_pins, "reused_33_bank_analysis": prior_pin, "training_fit_outcomes": training,
            "route_collection_outcomes": prior["route_collection_outcomes"] + collection_records,
            "historical_full_original_collection": prior["historical_full_original_collection"],
            "cost_rule": "39 previously paid fits plus 6 new fits; M1/body0 counted once. Six new collectors are separate overhead. Preserve individual walls and peaks; never add concurrent walls or memory peaks.",
            "BCE_rule": prior["BCE_rule"], "serving_authority": "Saved selected mean raw logits; reconstructed route mean is a disclosed sanity check only.",
            "resource_accounting": "Source metadata retains method parameter/BN bytes and dense Adam moment estimates; actual fit peaks, optimizer maps, checkpoint/export bytes and collector work remain separate.",
            "interpretation": "All three paired seeds on one already-exposed VALID split; no independent TEST/generalization, novelty or attainable-gating claim. Signed AUC/BCE contrasts retain both ranking and loss tradeoffs.",
            "new_models": 0, "new_native_forwards": 0, "new_optimizer_steps": 0, "TEST_accessed": False,
            "source_sha256": sha(__file__), "wall_seconds": time.monotonic() - started,
            "peak_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        lines = ["# Structural private-risk selected VALID addition", "", result["scope"], "",
            "Saved selected mean-logit pools retain authority. AUC is percent; BCE here is uniform per graph.", "",
            "| Seed | Bank | Member AUCs | Member BCEs | Pool AUC | Pool BCE | Pool minus mean member AUC points |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |"]
        for row in rows:
            lines.append(f"| {row['seed']} | {row['bank']} | " + ", ".join(f"{100*m['rocauc']:.3f}" for m in row["members"]) +
                " | " + ", ".join(f"{m['BCE']:.4f}" for m in row["members"]) +
                f" | {100*row['pool_AUC']:.3f} | {row['uniform_graph_BCE']:.4f} | {row['pool_minus_mean_member_AUC_pp']:.3f} |")
        lines += ["", "## Candidate minus reference", "", "| Seed | Candidate | Reference | AUC points | Uniform graph BCE | Native loss BCE |",
                  "| --- | --- | --- | ---: | ---: | ---: |"]
        for row in comparisons:
            native = row["delta_saved_native_VALID_BCE"]
            lines.append(f"| {row['seed']} | {row['candidate_bank']} | {row['reference_bank']} | {row['delta_pool_AUC_pp']:.3f} | {row['delta_uniform_graph_BCE']:.4f} | " +
                         (f"{native:.4f}" if native is not None else "not common reduction") + " |")
        lines += ['', '## Aligned pair repairs and new common errors', '',
            '| Seed | Candidate vs reference | Strict pool repairs / harms | Reference common errors | Acquired correct member / actual pool repair | Member repair still lost in pool | New common errors |',
            '| --- | --- | --- | ---: | --- | ---: | ---: |']
        for row in pair_comparisons:
            lines.append(f"| {row['seed']} | {row['comparison']} | {row['strict_repairs']} / {row['strict_harms']} | " +
                f"{row['reference_all_members_reversed']} | {row['reference_common_with_any_candidate_member_correct']} / " +
                f"{row['reference_common_pool_strict_repairs']} | {row['reference_common_acquired_correct_member_but_pool_reversed']} | " +
                f"{row['new_common_member_reversals_outside_reference_common']} |")
        lines += ['', 'Pair and route counts are descriptive, correlated diagnostics. Ties receive half AUC credit; strict repair/harm counts exclude ties. RESULT.json retains ties and all-population harms. Greater disagreement or member coverage alone is insufficient.']
        lines += ["", "## New training and collection costs", "",
                  "| Seed | Selected / completed epochs | Adam maps attempted / completed | Fit wall seconds | Fit peak GPU allocated / reserved bytes | Collection wall seconds | Collection native forwards attempted / completed |",
                  "| --- | --- | --- | ---: | --- | ---: | --- |"]
        for fit, collection in zip(fits, collections):
            calls, work, peaks = fit["optimizer_calls"], collection["actual_work"], fit["resources"]
            lines.append(f"| {fit['seed']} | {fit['selected_epoch']} / {fit['completed_epochs']} | " +
                f"{calls['attempts']} / {calls['completed']} | {fit['invocation_wall_seconds']:.3f} | " +
                f"{peaks['cuda_peak_allocated_bytes']} / {peaks['cuda_peak_reserved_bytes']} | {collection['wall_seconds']:.3f} | " +
                f"{work['native_route_forward_attempts']} / {work['native_route_forwards_completed']} |")
        lines += ["", "The new method has 19,200 private learned coordinates (76,800 FP32 parameter bytes), 153,600 dense Adam moment bytes if all receive gradients, and 144,320 private BN buffer bytes. BN adds zero learned parameters. Checkpoint receipt bytes are actual serialized totals, separately retained in RESULT.json.",
                  "", "RESULT.json retains all 45 unique training outcomes, every individual cost, 33 route collection outcomes plus the original full collection receipt, three-seed summaries and reconstruction warnings.",
                  "No control was retrained or recollected. No TEST was accessed. This analyzer constructed no model and executed no native forward or optimizer step.", ""]
        (output / "REPORT.md").write_text("\n".join(lines))
        return 0
    except Exception as error:
        (output / "GATE.json").write_text(json.dumps({"status": "blocked_no_partial_summary", "TEST_accessed": False,
            "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(limit=6)},
            "collection_outcomes_read": collection_records, "terminal_END": end_pin, "terminal_OUTCOMES": outcomes_pin,
            "new_models": 0, "new_native_forwards": 0, "new_optimizer_steps": 0,
            "wall_seconds": time.monotonic() - started}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
