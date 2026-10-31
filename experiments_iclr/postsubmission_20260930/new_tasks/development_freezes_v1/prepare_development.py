"""Create exclusive source-only prospective freezes. No imports of graph libraries.

Reads source text and existing local shareable JSON; never follows frozen
server paths, loads tensors, computes predictions, or launches a process.
Runtime CPU gates and runner-generated freezes are required before fitting.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent.parent


def inside(path):
    resolved = Path(path).resolve()
    if PHASE not in resolved.parents:
        raise ValueError("All source preparation paths must stay inside this phase")
    return resolved


def sha(path):
    return hashlib.sha256(inside(path).read_bytes()).hexdigest()


def literal(path, name):
    tree = ast.parse(inside(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Missing literal: " + name)


def write_json(path, value):
    with inside(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def shared_guard():
    path = PHASE / "new_tasks/shareable_guard.py"
    spec = importlib.util.spec_from_file_location("preparation_shareable_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_paths(dataset):
    common = ["new_tasks/shareable_guard.py", "source_snapshot/models.py", "protocols/EXPERIMENT_RULES.md"]
    if dataset == "molhiv":
        return common + ["new_tasks/molhiv_pilot.py", "new_tasks/gine_adapters.py", "new_tasks/verify_molhiv_pilot.py",
                         "new_tasks/test_gine_adapters.py", "new_tasks/MOLHIV_PILOT_USAGE.md", "new_tasks/GINE_MODEL_SPEC.md"]
    return common + ["new_tasks/mnist_pilot.py", "new_tasks/mnist_adapters.py", "new_tasks/verify_mnist_pilot.py",
                     "protocols/baseline_adapters.py", "efficient_paths/centered_paths.py", "new_tasks/MNIST_PILOT_USAGE.md"]


def make_study(dataset, arms, guard):
    runner = "molhiv_pilot.py" if dataset == "molhiv" else "mnist_pilot.py"
    path = PHASE / "new_tasks" / runner
    registered = literal(path, "ARMS")
    if any(arm not in registered for arm in arms):
        raise ValueError("Prospective arm missing from current runner")
    manifest = PHASE / "data_freezes" / dataset / "frozen_shareable_manifest_v2.json"
    data = guard.require_shareable_manifest(json.loads(manifest.read_text()), dataset)
    training, validation = data["split_summaries"]["train"], data["split_summaries"]["val"]
    is_mol = dataset == "molhiv"
    candidate = "middle_private" if is_mol else "centered_feedback"
    practical = 0.005 if is_mol else 0.002
    return {
        "schema": "gnnm-source-only-development-freeze-v1",
        "study_id": "molhiv_dev_v2" if is_mol else "mnist_dev_v1",
        "status": "prospective_source_and_method_freeze_runtime_CPU_gates_pending",
        "dataset": "ogbg-molhiv official scaffold split" if is_mol else "released MNIST-superpixel official splits",
        "primary_candidate": candidate,
        "hypothesis": ("A copied-private middle bond-aware GINE block restores useful member capacity while retaining shared atom/boundary/other-block storage; it improves pooled ROC-AUC over source GNNM and capacity-matched controls."
                       if is_mol else "Centered common/narrow SAGE paths with mean feedback retain useful member variation with fewer adjacency channels and improve accuracy at a fixed stored-parameter budget; the four-cell interaction tests whether centering and feedback jointly matter."),
        "nearest_prior_boundary": ("Selective parameter sharing and boundary BatchEnsemble are established constructions; this is a prospective graph-stack intervention, not a cleared novelty claim."
                                   if is_mol else "BatchEnsemble, shared-stack multihead models, and packed source SAGE are controls; the repaired four-cell common/narrow implementation is the candidate, not a new general feedback principle."),
        "arms": arms, "optimizer_seeds": [0, 1, 2], "label_fractions": [0.1, 1.0],
        "recipe": literal(path, "RECIPE"), "runner_protocol_schema": literal(path, "SCHEMA"),
        "HPO_candidates_per_arm": 1, "validation_opportunities_per_fit": 100,
        "label_budget_rule": "training labels only; largest-remainder class quotas; fixed class permutations101+class; sorted official training positions; 10% subset nested in full training",
        "training_graph_counts": {"0.1": training["graphs"] // 10, "1.0": training["graphs"]},
        "official_validation_graphs": validation["graphs"], "validation_partition": "complete official validation at every epoch",
        "train_and_validation_identity": {key: {field: data["split_summaries"][key][field]
                                               for field in ("graphs", "nodes", "directed_edge_entries", "canonical_split_sha256")}
                                          for key in ("train", "val")},
        "data_manifest_local_path": str(manifest.relative_to(PHASE)), "data_manifest_sha256": sha(manifest),
        "original_private_manifest_sha256_from_amendment": data["manifest_amendment"]["parent_manifest_sha256"],
        "shareable_guard_sha256": sha(PHASE / "new_tasks/shareable_guard.py"),
        "bound_source_sha256": {key: sha(PHASE / key) for key in source_paths(dataset)},
        "preparation_source_sha256": sha(Path(__file__)),
        "test_boundary": {"test_predictive_scoring_permitted": False, "confirmation_access_permitted": False,
                          "private_freezer_previously_fingerprinted_test_targets": True,
                          "public_provider_cache_contains_test_labels": True,
                          "shareable_manifest_contains_individual_test_target_digests": False,
                          "confirmation_requires_new_independent_protocol_after_candidate_lock": True},
        "primary_contrasts": ([(candidate + " - " + arm) for arm in ["gnnm", "untied", "base", "multihead", "ens", "base_parameters", "ens_parameters"]]
                              if is_mol else [(candidate + " - " + arm) for arm in ["gnnm", "base", "multihead", "ens", "packed_gnnm", "packed_parameters", "base_parameters"]]),
        "mechanism_contrasts": (["middle_private - gnnm", "middle_private - untied"] if is_mol else
                                 ["(centered_feedback-centered_no_feedback)-(uncentered_feedback-uncentered_no_feedback)",
                                  "centered_feedback-centered_no_feedback", "centered_feedback-uncentered_feedback"]),
        "matching_contract": ("base_parameters and ens_parameters: deterministic actual CPU parameter-count integer search8..512, closest to middle_private width128; tie smaller; <=1% mismatch; same bond-aware GINE recipe; independent ENS executes sequential members. Capacities must be frozen by the repaired synthetic gate."
                              if is_mol else "Fixed centered_feedback target281006: packed_parameters width67 count277688 (-1.1808%); base_parameters width136 count281666 (+0.2349%). Exact structural plan rechecked and frozen by runtime runner. packed_gnnm separates source execution layout from candidate architecture."),
        "uncertainty": {"experimental_unit": "dataset; optimizer seeds are paired repeated fits on that dataset",
                        "paired_seed_summary": "For each contrast and fraction report all3 deltas, mean, sample SD, median, positive-sign count; descriptive two-sided95% t interval mean +/-4.302652729911275*SD/sqrt(3).",
                        "validation_resampling": "No test scores or graph/node bootstrap significance claims. Validation is reused for checkpoint selection; seed intervals describe optimization variation, not independent-dataset generalization.",
                        "missing_run_policy": "Keep every failure and attempt log; no favorable-seed replacement or silent omission. Unresolved missing arm/seed/budget => incomplete study.",
                        "multiple_comparisons": "All prespecified contrasts reported. No selection of winning depth, fraction, four-cell variant, width, rank, or metric."},
        "decision_rules": {"practical_metric_margin": practical,
                           "primary_metric": "pooled validation ROC-AUC in[0,1]" if is_mol else "pooled validation accuracy in[0,1]",
                           "go_to_independent_confirmation": "At both fractions candidate mean gain>=margin and 3/3 paired gains>0 versus BASE, source GNNM, multihead, parameter-matched BASE and parameter-matched independent ENS; for MNIST this also applies to packed source GNNM. Versus same-width ENS mean delta>=-margin and all 3 deltas>=-margin. Mechanism contrast has predicted direction. Completed evidence/replay and directly measured cost advantage required before promoting an efficiency claim. This is a development go decision, not confirmatory evidence.",
                           "mechanism_direction": "middle_private gain over GNNM; report untied contrast without choosing a different depth" if is_mol else "centering-by-feedback interaction>0 at both fractions; paired effects reported without switching the primary cell",
                           "cost_gate": "At equal device/precision/representative fixed batches retain actual stored/trainable counts, fit time and peak memory. Against same-width independent ENS require >=20% fewer stored parameters and >=20% lower measured fit time or median inference latency, with peak allocation <=110% of ENS. An inference-speed claim additionally requires >=20% lower synchronized median inference latency with no p95 regression and complete timing samples from a separately frozen profiler; parameter counts alone do not pass this gate.",
                           "stop_or_amend": "Stop on guard failure, nonfinite loss/gradients, OOM, incomplete evidence/replay, or resource-budget infeasibility. Preserve attempts. A changed batch/epoch/width/rank/recipe/source needs a new study/amendment before rerunning. If quality/mechanism rules fail after complete comparisons, report all outcomes and stop expansion of this candidate.",
                           "no_outcome_adaptive_truncation": "Complete both fractions and all 3 seeds under the fixed schedule unless a technical/resource stop occurs; no outcome-based early cancellation of full-label condition."},
        "pending_runtime_bindings": ["repaired synthetic helper gate", "real128-complete-training-graph CPU gate under new source", "runner-generated prospective protocol with exact train subset/index hashes", "actual package/provider-source environment", "authorized idle physical GPU UUID/model and serialized launch ownership", "fixed resource ceiling and separately frozen inference profiler"],
        "source_only_limits": {"runner_imported": False, "raw_processed_data_bytes_verified_locally": False,
                               "server_frozen_paths_followed": False, "train_subset_tensors_constructed": False,
                               "runtime_environment_or_hardware_freeze_fabricated": False,
                               "predictive_fit_or_score_run": False, "original_paper_scores_redefined": False},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = inside(args.output)
    if output.exists():
        raise ValueError("Refuse to overwrite any prospective package")
    guard = shared_guard()
    arms = {
        "molhiv": ["middle_private", "gnnm", "untied", "base", "multihead", "ens", "base_parameters", "ens_parameters"],
        "mnist_superpixels": ["centered_feedback", "centered_no_feedback", "uncentered_feedback", "uncentered_no_feedback",
                              "base", "gnnm", "multihead", "ens", "packed_gnnm", "packed_parameters", "base_parameters"],
    }
    studies = {dataset: make_study(dataset, values, guard) for dataset, values in arms.items()}
    created = datetime.now(timezone.utc).isoformat()
    for study in studies.values():
        study["frozen_at_UTC"] = created
    schedule = []
    for fraction, label in ((0.1, "frac01"), (1.0, "frac10")):
        for seed in (0, 1, 2):
            for position in range(max(map(len, arms.values()))):
                for dataset in ("molhiv", "mnist_superpixels"):
                    values = arms[dataset]
                    if position >= len(values):
                        continue
                    arm = values[(position + seed) % len(values)]
                    study = studies[dataset]
                    schedule.append({"sequence": len(schedule) + 1, "study_id": study["study_id"],
                                     "arm": arm, "seed": seed, "label_fraction": fraction,
                                     "run_output": "new_tasks/runs/" + study["study_id"] + "_" + arm + "_seed" + str(seed) + "_" + label + "_attempt01"})
    output.mkdir(parents=True, exist_ok=False)
    for dataset, study in studies.items():
        write_json(output / (dataset + "_source_freeze.json"), study)
    plan = {"schema": "gnnm-source-only-serial-launch-plan-v1", "created_at_UTC": created,
            "status": "no_jobs_launched_no_GPU_idle_at_preparation",
            "fits_total": len(schedule), "fits_per_task": {"molhiv": 48, "mnist_superpixels": 66},
            "execution": "one explicitly authorized idle GPU, strictly serial; no queued/background auto-launch",
            "order": "10% then100%; paired seeds0/1/2; task interleaving and deterministic arm rotation reduce systematic time/order drift",
            "full_fit_epochs": 100, "selection_opportunities_per_arm_seed_fraction": 100,
            "predictive_minibatches_excluding_validation": {"molhiv": 681600, "mnist_superpixels": 1560900},
            "time_estimate": "No GPU-hour ETA inferred from CPU preflight. After first complete GPU fits, forecast all remaining runs from observed task/arm training+validation+artifact timings without changing recipe.",
            "launch_gate": "Repaired CPU gates pass; exact runtime freezes bind these arms and source/data/env; independent coordinator reserves the authorized idle GPU; runner checks <=100MiB used, zero utilization, no compute contexts before model CUDA transfer. A check alone is not a reservation.",
            "attempt_logging": "Keep command, UTC, environment, stdout, stderr, exit status for every attempted launch in a new coordinator log outside immutable run directories. Any retry receives a new attemptNN path, remains an explicit failure/retry pair, and cannot replace a reported seed.",
            "freeze_command_templates": {dataset: ["python", "new_tasks/" + ("molhiv_pilot.py" if dataset == "molhiv" else "mnist_pilot.py"), "freeze", "--data-manifest", "new_tasks/data/" + dataset + "/frozen_shareable_manifest_v2.json", "--arms", *values, "--output", "new_tasks/protocols/" + studies[dataset]["study_id"] + "_runtime.json"] for dataset, values in arms.items()},
            "train_command_template": ["python", "new_tasks/{runner}.py", "train", "--data-manifest", "new_tasks/data/{dataset}/frozen_shareable_manifest_v2.json", "--protocol", "new_tasks/protocols/{study_id}_runtime.json", "--arm", "{arm}", "--seed", "{seed}", "--label-fraction", "{fraction}", "--device", "cuda:0", "--physical-gpu", "{authorized_idle_physical_gpu}", "--output", "{run_output}"],
            "environment_template": {"CUDA_VISIBLE_DEVICES": "{authorized_idle_physical_gpu}", "PYTHONDONTWRITEBYTECODE": "1"},
            "runs": schedule}
    write_json(output / "launch_schedule.json", plan)
    with (output / "README.md").open("x") as stream:
        stream.write(README)
    write_json(output / "artifact_manifest.json", {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    print(json.dumps({"status": "source_only_prospective_package_created", "output": str(output), "fits_planned": len(schedule),
                      "artifact_manifest_sha256": sha(output / "artifact_manifest.json"), "jobs_launched": False}, indent=2))


README = """# Prospective MolHIV and MNIST development freezes

These are finalized source and method plans, created before predictive fitting.
They are not runner-generated runtime protocols: exact training-label subset
hashes, the actual package/provider environment, matching capacities and GPU
identity still require the declared CPU gates and runtime freezes. No server,
GPU job, raw tensor load, validation score or test score was used in preparing
this package. Existing paper scores and prior artifacts retain their meaning.

MolHIV uses official scaffold train/validation partitions (32,901/4,113 graphs),
bond-aware residual GINE and a prespecified middle-private block candidate.
Eight arms, three paired optimizer seeds and two nested training-label fractions
make 48 fits. MNIST uses released 55,000/5,000 development partitions, SAGE and
mean node-logit graph readout. Eleven arms make 66 fits. The four common/narrow
cells distinguish centering from feedback; packed source GNNM and parameter
matched independent/single controls test whether the new architecture adds
value beyond execution layout or stored capacity. Every fit gets 100 epochs,
the complete validation partition and one fixed recipe.

Use `molhiv_source_freeze.json` and `mnist_superpixels_source_freeze.json` for
the bound recipes, hypotheses, contrasts, uncertainty and stop/go rules.
`launch_schedule.json` contains the exact114-fit serial order and command
templates as data. It launches nothing. Run all 10% comparisons then full-label
comparisons; neither budget nor seed is selected for a favorable result.
No GPU-hour estimate is claimed from the CPU checks. The workload contains
2,242,500 training minibatches before validation or evidence writing.

The latest source amendment adds `base_parameters` and `ens_parameters` to
MolHIV schema-v2. They use complete bond-aware GINE, not packed nonlinear GINE.
Width matching enumerates actual CPU parameter counts over integers 8..512,
targets `middle_private` at width 128, breaks ties toward smaller widths and
requires no more than 1% difference. The new synthetic gate must confirm the
actual chosen capacities, RNG preservation and finite gradients. The old
MolHIV usage document describes the earlier eight-arm version; this package's
explicit arm list and schema-v2 recipe supersede it for these new fits.

Both runners now bind `new_tasks/shareable_guard.py`, which requires explicit
unscored/unaccessed-confirmation status, unchanged shareable-v2 provenance,
absence of test targets and class summaries, consistent removal counts and
the canonical test input-fingerprint hash. The original freezer did fingerprint
test targets, and public provider files contain test labels. The boundary is
procedural and source enforced, not a cryptographic seal of public labels.

Before any fit, the coordinator must rerun:

1. `python new_tasks/verify_molhiv_pilot.py` and
   `python new_tasks/verify_mnist_pilot.py` in the existing graph environment;
   CUDA visibility must be empty. These are synthetic checks.
2. CPU preflights with 128 complete training graphs for MolHIV `middle_private`,
   `base_parameters` and `ens_parameters`, and MNIST `centered_feedback`,
   with new output paths and the original host's exact shareable data manifest.
3. The two runner freeze commands in `launch_schedule.json`, including their
   explicit reduced arm lists. Inspect actual matching counts and bind the
   same source/data/environment as these plans. Frozen server paths must not
   be rewritten or followed from this local preparation host.
4. Record an explicit authorized idle physical GPU, its UUID/model, serialized
   ownership and resource ceiling. None was idle when this plan was prepared.

The runner freeze creates the exact subset/index hashes using training labels
only. Runtime hardware belongs in the launch contract and fitted evidence;
this local package fabricates none of it. A source/data/environment change
requires a new amendment and output directory before fitting. Prior runners
and verifiers are preserved in sibling `before_runner_amendment/` with hashes.

Report every arm/seed/fraction and failed attempt. The primary validation metric
is ROC-AUC for MolHIV and accuracy for MNIST; margins are 0.005 and 0.002.
Paired seed summaries and descriptive 95% t intervals describe optimizer
variation on each corpus. They do not establish independent-task generality.
Efficiency promotion requires measured cost and the separately frozen
inference profiler; stored counts or adjacency widths cannot establish speed.
Failure of the prespecified quality or mechanism rules stops expansion rather
than initiating a width/rank/depth search. The same source MNIST images in
GOOD-CMNIST cannot count as independent image-corpus confirmation.

Test scoring and independent confirmation remain closed until a separate
candidate/confirmation protocol is locked. This package makes no manuscript,
acceptance, state-of-the-art or predictive performance claim.

To prepare another source-only amendment, run the sibling preparation script
with a new output path inside this phase. It rejects existing paths and writes
every JSON/README/artifact-manifest file exclusively. The manifest covers all
its siblings except itself. No prior files are overwritten.
"""


if __name__ == "__main__":
    main()
