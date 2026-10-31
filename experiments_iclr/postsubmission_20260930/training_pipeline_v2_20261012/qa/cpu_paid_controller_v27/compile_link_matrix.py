"""Compile a disabled original-source NCN/NCNC single/I4 development packet.

Source/data custody is mandatory. Resource and exposure decisions are required
to materialize study rows. No model, role tensor, score, download or GO is read
or produced by compilation. Sealed endpoints are bound as data receipts only.
"""
from __future__ import annotations
import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import socket
import subprocess

from adapters.link import AUTHOR_COMMIT, AUTHOR_REPOSITORY, AUTHOR_SOURCE, bound_file, file_sha, validate_config, validate_roster
from spec import fingerprint, validate

SEARCH_SEEDS = (71101, 71103)
COMPARATIVE_SEEDS = (71201, 71203, 71207)
PROFILES = ("legacy_widths", "source_controls")


def native_parameters(predictor):
    return {"encoder": {"hidden": 64, "layers": 1, "dropout": .1, "ln": True, "res": True, "edrop": .25,
            "xdropout": .25, "taildropout": .05, "jk": False, "conv_fn": "gcn"},
        "predictor": {"name": predictor, "layers": 3, "dropout": .3, "edrop": 0., "ln": True, "cndeg": -1,
            "use_xlin": True, "tailact": True, "twolayerlin": False, "beta": 1., "completion": None if predictor == "NCN" else
            {"alpha": 1.05, "scale": 2.5, "offset": 6., "trainresdeg": -1, "testresdeg": -1, "pt": .1,
             "learnablept": False, "depth": 1, "splitsize": -1}},
        "optimizer": {"name": "Adam", "encoder_lr": .0082, "predictor_lr": .0037, "weight_decay": 0.,
            "eps": 1e-8, "betas": [.9, .999], "amsgrad": False}}


def replace(point, key, value):
    result = copy.deepcopy(point)
    cursor = result
    parts = key.split(".")
    for part in parts[:-1]:cursor = cursor[part]
    cursor[parts[-1]] = value
    return result


def parameter_domain(predictor, profile="source_controls"):
    if predictor not in {"NCN", "NCNC"} or profile not in PROFILES:
        raise ValueError("Declare native NCN/NCNC and a named finite search profile")
    anchor = native_parameters(predictor)
    if profile == "legacy_widths":
        return [replace(anchor, "encoder.hidden", width) for width in (64, 128)]
    joint = {"encoder.hidden": (64, 128), "encoder.layers": (1, 2), "encoder.dropout": (.1, .3),
             "predictor.dropout": (.1, .3), "optimizer.encoder_lr": (.0041, .0082, .0164),
             "optimizer.predictor_lr": (.00185, .0037, .0074), "optimizer.weight_decay": (0., .0001)}
    points = []
    for values in itertools.product(*joint.values()):
        point = anchor
        for key, value in zip(joint, values):point = replace(point, key, value)
        points.append(point)
    # Extra controls are native-anchor neighbors, not an undeclared enormous
    # Cartesian product. Every point records its full actual source recipe.
    neighbors = {"encoder.edrop": (0., .5), "encoder.xdropout": (0., .5), "predictor.edrop": (.1,),
        "predictor.beta": (.5, 2.), "encoder.ln": (False,), "encoder.jk": (True,), "predictor.ln": (False,),
        "predictor.use_xlin": (False,), "predictor.tailact": (False,), "predictor.twolayerlin": (True,)}
    for key, values in neighbors.items():
        points.extend(replace(anchor, key, value) for value in values)
    # Residual is effective only when feature/hidden dimensions match for the
    # native one-layer encoder; the 128-width neighbor therefore has an effect.
    points.append(replace(replace(anchor, "encoder.hidden", 128), "encoder.res", False))
    if predictor == "NCNC":
        completion = {"alpha": (1., 1.1), "scale": (1.25, 5.), "offset": (3., 9.), "pt": (.05, .2), "learnablept": (True,)}
        for key, values in completion.items():
            points.extend(replace(anchor, "predictor.completion."+key, value) for value in values)
    unique = {fingerprint(point): point for point in points}
    return [unique[key] for key in sorted(unique)]


def fit_formula(profile="source_controls", epochs=100, search_seeds=SEARCH_SEEDS, comparative_seeds=COMPARATIVE_SEEDS, member0_reuse=True):
    if epochs != 100:
        raise ValueError("This packet freezes native full 100 epochs; extensions require another declaration")
    configurations = {name: len(parameter_domain(name, profile)) for name in ("NCN", "NCNC")}
    search = len(search_seeds)*sum(configurations.values())
    comparison = len(comparative_seeds)*2*(4 if member0_reuse else 1+4)
    fits = search+comparison
    return {"configuration_counts": configurations, "search_seed_count": len(search_seeds),
        "comparative_seed_count": len(comparative_seeds), "search_model_fits": search, "comparative_model_fits": comparison,
        "model_fit_formula": "S*(C_NCN+C_NCNC) + R*2*4" if member0_reuse else "S*(C_NCN+C_NCNC) + R*2*(1+4)", "model_fits": fits,
        "single_reuses_I4_body0": member0_reuse, "I4_reuses_single_search_configuration": True,
        "I4_search_member_rule": "own-selected member-equivalent single search; four separate comparative fits per I4 seed",
        "optimizer_starts": fits, "full_budget_epochs": fits*epochs, "native_updates_per_epoch": 17,
        "native_updates_per_fit": epochs*17, "prospective_optimizer_updates": fits*epochs*17,
        "native_positive_records_per_epoch": 1114112, "native_dropped_tail_per_epoch": 64940,
        "prospective_supervised_positive_records": fits*epochs*1114112,
        "prospective_supervised_negative_records": fits*epochs*1114112,
        "search_selection_jobs": 2, "comparative_raw_banks": len(comparative_seeds)*4,
        "calibration_jobs": 0, "evaluation_jobs": 0, "continuations": 0, "extensions": 0,
        "formula_is_measurement": False, "scientific_training_runs": 0}


def bind_custody(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):raise ValueError("Link custody must be within the project")
    value = json.loads(path.read_text())
    if value.get("schema") != "gnnm-link-collab-fit-data-v1" or value.get("dataset") != "ogbl-collab":
        raise ValueError("Require the actual official collab fit preparation manifest")
    if any(value.get(key) is not False for key in ("test_file_opened", "model_or_checkpoint_loaded", "predictive_metrics_computed", "original_paper_scores_changed", "science_ready", "author_protocol_replication")):
        raise ValueError("Fit custody must be no-model/no-score/no-TEST and disabled")
    audit = value["input_audit"]
    for key, expected in {"nodes": 235868, "features": 128, "train_records": 1179052, "raw_directed_records": 2358104,
                          "select_positive": 60084, "select_negative": 100000, "heldout_records_added_to_support": False,
                          "train_only_support": True, "train_unique_pairs": 967632, "select_positive_train_pair_recurrences": 28072,
                          "select_negative_train_pair_recurrences": 8, "select_negative_self_pairs": 1}.items():
        if audit.get(key) != expected:raise ValueError("Official full link role audit changed: "+key)
    if set(value["link_inputs"]) != {"graph", "train", "select"}:raise ValueError("Fit custody opens exactly three packs")
    rows = [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": file_sha(path)}]
    for item in list(value["link_inputs"].values())+list(value["link_source"].values())+[value["preparation_program"]]:
        bound_file(root, item);rows.append(item)
    for name, pin in value["link_source"].items():
        if name not in AUTHOR_SOURCE or any(pin[k] != AUTHOR_SOURCE[name][k] for k in ("bytes", "sha256")):
            raise ValueError("Native source pin changed")
    if "source_admission_receipt" in value:
        bound_file(root, value["source_admission_receipt"]);rows.append(value["source_admission_receipt"])
    return value, rows


def make_study(custody, source, profile, exposure, search_profile="source_controls", engine_prefix="training_pipeline_v2_20261012", sealed=None):
    for key in ("reservation_mib", "wall_seconds"):
        if type(profile.get(key)) is not int or profile[key] < 1:raise ValueError("Explicit positive link resource profile required: "+key)
    for key in ("estimated_seconds_per_epoch", "estimated_selection_seconds"):
        if type(profile.get(key)) not in (int, float) or not math.isfinite(profile[key]) or profile[key] <= 0:raise ValueError("Explicit link scheduling estimate required: "+key)
    if not profile.get("evidence"):raise ValueError("Resource assumption evidence is required")
    if exposure.get("protected_confirmation") is not False or exposure.get("evaluate_purpose") != "development" or not exposure.get("prior_exposure"):
        raise ValueError("Original collab development requires explicit prior exposure; confirmation is not admitted")
    budget = {"unit": "epochs", "policy": "full_budget", "maximum": 100}
    selector = {"metric": "hits@50", "direction": "max", "ties": "first", "evaluation_interval": 1, "checkpoint0": False, "save_every": 1}
    runs, selections, banks = [], {}, []
    def add(identifier, kind, method, seed, body, adapter, config, parents=()):
        row = {"run_id": identifier, "kind": kind, "method": method, "split": 0, "seed": seed, "body": body,
            "adapter": adapter, "config": config, "budget": copy.deepcopy(budget) if kind == "model_fit" else {},
            "selection": copy.deepcopy(selector) if kind == "model_fit" else {}, "parents": list(parents)}
        runs.append(row);return identifier
    def fit_config(parameters, seed, arm, stage, body=0, group=None):
        value = {key: copy.deepcopy(custody[key]) for key in ("link_inputs", "link_source", "sampler", "metric", "support")}
        value.update(copy.deepcopy(parameters), arm=arm, initialization_seed=seed*100+(20+body if group else 10),
            training={"batch_size": 65536, "eval_batch_size": 131072, "tail_policy": "native_drop_tail"}, stage=stage,
            reservation_mib=profile["reservation_mib"], maximum_cuda_bytes=profile["reservation_mib"]*1024**2,
            estimated_seconds=profile["estimated_seconds_per_epoch"]*100)
        if group:value["independence_group"] = group
        return value
    for predictor in ("NCN", "NCNC"):
        candidates = []
        for point in parameter_domain(predictor, search_profile):
            config_id = fingerprint(point)[:20]
            identifiers = []
            for seed in SEARCH_SEEDS:
                identifier = "search-"+predictor+"-"+config_id+"-s"+str(seed)
                identifiers.append(add(identifier, "model_fit", "original_"+predictor+"_native", seed, 0,
                    engine_prefix+"/adapters/link.py:make_adapter", fit_config(point, seed, "native_single", "search")))
            candidates.append({"config_id": config_id, "parameters": point, "run_ids": identifiers})
        selections[predictor] = add("select-"+predictor, "selection", "original_"+predictor+"_native", 0, 0,
            engine_prefix+"/adapters/link_search.py:select_configuration",
            {"stage": "search_selection", "ranking_metric": "hits@50", "ranking_direction": "max", "configurations": candidates,
             "reservation_mib": 256, "estimated_seconds": profile["estimated_selection_seconds"]},
            [identifier for candidate in candidates for identifier in candidate["run_ids"]])
        for seed in COMPARATIVE_SEEDS:
            for label, arm, bodies in (("I4", "native_independent4_body", 4),):
                group = "compare-"+predictor+"-"+label+"-s"+str(seed)
                fits = []
                for body in range(bodies):
                    config = fit_config(native_parameters(predictor), seed, arm, "comparative", body, group if bodies == 4 else None)
                    config["selected_config_from"] = selections[predictor]
                    fits.append(add(group+"-b"+str(body), "model_fit", "original_"+predictor+"_"+label, seed, body,
                        engine_prefix+"/adapters/link.py:make_adapter", config, [selections[predictor]]))
                full_bank = {"predictor": predictor, "arm": label, "seed": seed, "fit_parents": fits,
                    "own_selected_checkpoints": bodies, "fusion": "mean_raw_scores", "export_job": None,
                    "final_evaluation_status": "prepared_raw_endpoint" if sealed else "blocked_sealed_evaluate_custody_missing"}
                banks.append(full_bank)
                single_bank = {**copy.deepcopy(full_bank), "arm":"single", "fit_parents":fits[:1], "own_selected_checkpoints":1,
                    "singleton_reference":"prospectively_fixed_I4_body0", "paid_fit_reused":True}
                banks.append(single_bank)
                if sealed:
                    shared={"query_input":sealed["endpoint_inputs"]["evaluate_queries"],"fusion":"mean_raw_scores","support_policy":"full_TRAIN_only"}
                    export_id=group+"-export"
                    add(export_id,"ensemble_assembly","original_"+predictor+"_I4",seed,0,
                        engine_prefix+"/adapters/link_endpoints.py:export_bank",{**shared,"fit_parents":fits,"stage":"bank_export",
                        "reservation_mib":profile["reservation_mib"],"estimated_seconds":profile.get("estimated_export_seconds",3600)},fits)
                    single_id=group+"-single-export"
                    add(single_id,"ensemble_assembly","original_"+predictor+"_single",seed,0,
                        engine_prefix+"/adapters/link_endpoints.py:export_bank",{**shared,"fit_parents":fits[:1],"reuse_export_parent":export_id,
                        "single_reference":"fixed_I4_body0","stage":"bank_export","reservation_mib":512,"estimated_seconds":1},[export_id,fits[0]])
                    for bank,identifier in ((full_bank,export_id),(single_bank,single_id)):
                        evaluate_id=identifier+"-evaluate"
                        add(evaluate_id,"evaluation","original_"+predictor+"_"+bank["arm"],seed,0,
                            engine_prefix+"/adapters/link_endpoints.py:evaluate_bank",{**shared,"target_input":sealed["endpoint_inputs"]["evaluate_targets"],
                            "export_parent":identifier,"metric":custody["metric"],"stage":"evaluation","reservation_mib":512,"estimated_seconds":1},[identifier])
                        bank.update(export_job=identifier,evaluation_job=evaluate_id)
    spec = {"schema_version": 1, "study_id": "original-ncn-ncnc-collab-v2-"+search_profile, "stage": "development",
        "dataset": "ogbl-collab", "hypothesis": "Prospective native-source NCN and NCNC single versus separately selected I4 raw rankings under fixed full epoch budget.",
        "data": {"task": "link_prediction", "preprocessing": "official temporal TRAIN; full official VALID as SELECT; unweighted full TRAIN-only support",
            "roles": {"train": custody["link_inputs"]["train"], "select": custody["link_inputs"]["select"],
                "cal": {"status": "raw_only_absent", "confidence_claims_available": False},
                "evaluate": sealed["endpoint_inputs"]["evaluate_targets"] if sealed else {"status": "sealed_authority_missing", "test_file_opened": False}},
            "prior_exposure": exposure["prior_exposure"], "evidence": {"fit_custody_hash": fingerprint(custody),
                "exposure_decision": exposure, "author_protocol_replication": False, "SELECT_labels_reused_as_CAL": False,
                **({"endpoint_inputs":sealed["endpoint_inputs"],"endpoint_nodes":sealed["nodes"],"sealed_endpoint_manifest_hash":fingerprint(sealed)} if sealed else {})}},
        "source": source, "runs": runs, "comparisons": [{"references": ["original_NCN_single", "original_NCNC_single"],
            "candidates": ["original_NCN_I4", "original_NCNC_I4"], "seed_ids": list(COMPARATIVE_SEEDS),
            "report": "all costs and failures; raw ranking only after a separately sealed evaluate role; no confidence/calibration claim",
            "acceptance_guarantee": False, "author_replication": "separate unmaterialized track requiring native final TRAIN+VALID support"}],
        "release": {"policy": "whole_roster", "primary_endpoint": "all_closed_raw_official_TEST_Hits50_banks" if sealed else "blocked_until_sealed_evaluate_custody",
            "global_plan_gates_required": True, "scientific_launch_enabled": False, "human_GO_required": True},
        "resources": {"wall_seconds": profile["wall_seconds"], "profile": profile, "status": "DECLARED_ESTIMATES_REQUIRE_RESOURCE_QUALIFICATION"}}
    validate(spec)
    for row in runs:
        if row["kind"] == "model_fit":validate_config(row["config"], row);validate_roster(spec, row)
    return spec, banks


def bind_sealed(root,path):
    root,path=Path(root).resolve(),Path(path).resolve()
    if not path.is_relative_to(root):raise ValueError("Confine sealed custody")
    value=json.loads(path.read_text())
    if value.get("schema")!="gnnm-link-sealed-evaluate-v1" or value.get("complete") is not True or value.get("dataset")!="ogbl-collab":raise ValueError("Require actual complete sealed endpoint preparation")
    if any(value.get(key) is not False for key in ("training_or_selection_or_export_launched","model_constructed","optimizer_created","scoring_performed","scientific_launch_enabled","author_protocol_replication")):
        raise ValueError("Sealed custody must be no-fit/no-score and disabled")
    if set(value["endpoint_inputs"])!={"evaluate_queries","evaluate_targets"} or value["shuffle"]["opaque_query_ids"] is not True:
        raise ValueError("Separate label-free and target/index authorities required")
    if (value.get("nodes"),value.get("positive_records"),value.get("negative_records"),value.get("query_records"))!=(235868,46329,100000,146329):
        raise ValueError("Complete unchanged official TEST record counts required")
    if value["endpoint_inputs"]["evaluate_queries"]["path"]==value["endpoint_inputs"]["evaluate_targets"]["path"]:
        raise ValueError("Query and target/index files must be separate")
    rows=[{"path":str(path.relative_to(root)),"bytes":path.stat().st_size,"sha256":file_sha(path)}]
    for item in list(value["endpoint_inputs"].values())+[value["archive"],value["preparation_program"],value["fit_manifest"]]:
        bound_file(root,item);rows.append(item)
    return value,rows


def verify_allocation():
    if socket.gethostname() != "anogena-2-0":raise RuntimeError("CLI compilation belongs to the authorized allocation")
    if subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() != ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]:
        raise RuntimeError("Allocation GPU UUID custody changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "custody", "output"):parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--search-profile", choices=PROFILES, default="source_controls")
    parser.add_argument("--resources", type=Path, help="Optional explicit profile JSON; mandatory to materialize study rows")
    parser.add_argument("--exposure-decision", type=Path, help="Optional explicit exposure JSON; mandatory to materialize study rows")
    parser.add_argument("--source-commit")
    parser.add_argument("--sealed-endpoint",type=Path)
    args = parser.parse_args();verify_allocation()
    root, output = args.project_root.resolve(), args.output.resolve()
    if not output.is_relative_to(root) or output == root or output.exists():raise ValueError("Use a fresh project-confined disabled packet")
    custody, bindings = bind_custody(root, args.custody)
    sealed=None
    if args.sealed_endpoint:
        sealed,sealed_bindings=bind_sealed(root,args.sealed_endpoint);bindings+=sealed_bindings
        if sealed["fit_manifest"]!=bindings[0] or sealed["metric"]!=custody["metric"]:
            raise ValueError("Sealed endpoint differs from exact prepared fit/metric custody")
    for path in (args.resources,args.exposure_decision):
        if path is not None:
            path=path.resolve()
            if not path.is_relative_to(root):raise ValueError("Confine resource/exposure decisions")
            bindings.append({"path":str(path.relative_to(root)),"bytes":path.stat().st_size,"sha256":file_sha(path)})
    programs = [p for p in Path(__file__).resolve().parent.rglob("*.py") if "__pycache__" not in p.parts and "qa" not in p.relative_to(Path(__file__).resolve().parent).parts]
    for path in programs:
        if not path.is_relative_to(root):raise ValueError("Compiler and adapters must be inside project custody")
        bindings.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": file_sha(path)})
    source = {"git_commit": args.source_commit or subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "files": list({row["path"]: row for row in bindings}.values()), "author_repository": AUTHOR_REPOSITORY, "author_commit": AUTHOR_COMMIT,
        "working_tree_fingerprints_authoritative": True, "author_protocol_replication": False}
    formula = fit_formula(args.search_profile)
    manifest = {"schema": "disabled-link-ncn-ncnc-preparation-v1", "source_data_custody_ready": True,
        "scientific_launch_enabled": False, "human_GO_required": True, "scientific_training_runs": 0, "GO_receipt_created": False,
        "fit_custody": bindings[0], "source": source, "search_profile": args.search_profile,
        "full_epoch_budget": 100, "fit_formula": formula, "materialized_model_fits": 0,
        "implemented_models": ["original_NCN_single", "original_NCNC_single", "original_NCN_I4", "original_NCNC_I4"],
        "protected_adaptation": "TRAIN-only graph; official VALID SELECT; native constructor/objective/sampler; no TEST diagnostics",
        "author_replication": {"status": "blocked", "required": "separate author final TRAIN+VALID graph and sealed evaluate custody"},
        "calibration": {"status": "raw_only_absent", "SELECT_reuse": False, "confidence_claims_available": False},
        "original_GNNM_link": {"status":"blocked","required_comparison":True,
            "exact_gap":"Pinned TABMModel forwards graph/x/tabm_seed to node outputs. Node adapter accepts node IDs and multiclass CE only. No separate boundary-encoder/query-decoder link adaptation, link objective/sampler/selector/restore endpoint exists.",
            "NCN_banks_are_original_GNNM":False,"fits_counted":0},
        "blockers": ([] if sealed else ["sealed_evaluate_custody"])+["original_GNNM_boundary_encoder_query_decoder_link_adapter", "CUDA_sparse_forward_backward_resource_qualification", "root_global_plan_gates", "human_GO"],
        "domain_policy": "finite joint base plus declared native-anchor neighbors; stable config hash deduplication",
        "ineffective_locked_fields": {"predictor.layers": "native constructor accepts but ignores depth argument", "encoder.taildropout": "ignored by source gcn nonpure branch"},
        "additional_source_locks": {"encoder.conv_fn": "gcn", "cndeg": -1, "trainresdeg": -1, "testresdeg": -1,
            "NCNC_depth": 1, "NCNC_splitsize": -1, "sampler": "actual qualified PyG defaults", "tail_policy": "native_drop_tail"}}
    output.mkdir(parents=True)
    domain = {predictor: [{"config_id": fingerprint(point)[:20], "parameters": point} for point in parameter_domain(predictor, args.search_profile)] for predictor in ("NCN", "NCNC")}
    (output/"SEARCH_DOMAIN.json").write_text(json.dumps(domain, indent=2)+"\n")
    if args.resources is not None and args.exposure_decision is not None:
        profile, exposure = json.loads(args.resources.read_text()), json.loads(args.exposure_decision.read_text())
        engine_prefix = str(Path(__file__).resolve().parent.relative_to(root))
        spec, banks = make_study(custody, source, profile, exposure, args.search_profile, engine_prefix,sealed)
        spec["release"]["preparation_blockers"]=manifest["blockers"]
        spec["data"]["evidence"]["original_GNNM_link"]=manifest["original_GNNM_link"]
        (output/"STUDY_DISABLED.json").write_text(json.dumps(spec, indent=2)+"\n")
        (output/"BANKS_DISABLED.json").write_text(json.dumps(banks, indent=2)+"\n")
        manifest.update(materialized_model_fits=formula["model_fits"],study_hash=fingerprint(spec),status="SOURCE_DATA_READY_DISABLED_STUDY",
            bank_exports=12 if sealed else 0,evaluation_jobs=12 if sealed else 0,calibration_jobs=0,resources_provisional=True,
            fit_formula={**formula,"bank_exports":12 if sealed else 0,"evaluation_jobs":12 if sealed else 0})
    else:
        manifest["blockers"].extend(["explicit_resource_profile", "explicit_exposure_decision"])
        manifest["status"] = "SOURCE_DATA_READY_DISABLED_DOMAIN_ONLY"
    (output/"MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({key: manifest[key] for key in ("status", "source_data_custody_ready", "scientific_launch_enabled", "materialized_model_fits", "fit_formula", "blockers")}, indent=2))


if __name__ == "__main__":main()
