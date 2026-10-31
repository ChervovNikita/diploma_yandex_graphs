"""AST/source/metadata verification only: no imports or calls of the runner."""
import ast
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]


def constant(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        return {constant(k): constant(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [constant(n) for n in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Pow, ast.Mult)):
        a, b = constant(node.left), constant(node.right)
        return a ** b if isinstance(node.op, ast.Pow) else a * b
    raise ValueError("Not a literal-only contract constant")


def main():
    checks = []

    def check(name, truth):
        if not truth:
            raise ValueError(name)
        checks.append({"check": name, "passed": True})

    source = HERE / "decision_analysis.py"
    raw = source.read_bytes()
    text = raw.decode()
    tree = ast.parse(text, filename=str(source))
    for path in sorted(HERE.glob("*.py")):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        check("Python source syntax: " + path.name, True)
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {"PLAN_ROOT", "PLAN_SEAL_SHA", "PLAN_MANIFEST_SHA", "BOUND_SHA", "DESTINATION", "RUNTIME", "CAPS", "MAX_LOGIT_FILE_BYTES", "MAX_TARGET_FILE_BYTES", "MAX_METADATA_BYTES"}:
            constants[node.targets[0].id] = constant(node.value)
    draft = json.loads((HERE / "DRAFT_REQUEST_v1.json").read_text())["request_payload"]
    plan_root = PHASE / constants["PLAN_ROOT"]
    plan = json.loads((plan_root / "PROTOCOL.json").read_text())
    check("exact sealed plan manifest", hashlib.sha256((plan_root / "MANIFEST.json").read_bytes()).hexdigest() == constants["PLAN_MANIFEST_SHA"])
    check("exact sealed plan receipt", hashlib.sha256((plan_root / "SEAL.json").read_bytes()).hexdigest() == constants["PLAN_SEAL_SHA"] == draft["plan_seal_sha256"])
    check("original bounded wrapper source", hashlib.sha256((PHASE / "protocols/bounded_run_v1.py").read_bytes()).hexdigest() == constants["BOUND_SHA"])
    check("exact four-thread CPU caps", constants["CAPS"] == draft["caps"] == {"whole_wall_seconds": 600, "host_RSS_bytes": 2**31, "new_output_disk_bytes": 128 * 2**20, "CPU_threads": 4, "GPU_count": 0})
    check("fixed installed Torch version CPU policy", constants["RUNTIME"] == draft["runtime_policy"] == {"torch_version": "2.1.2+cu118", "CPU_threads": 4, "device": "cpu", "scientific_imports": ["torch"], "GPU_calls": 0})
    check("draft disabled", draft["root_authorized"] is False)
    check("all receipt and output gates missing", all(draft[k] is None for k in ("source_manifest", "consumer_audit", "implementation_review", "resource_approval", "root_approval", "output_relative_to_phase")))
    check("forbidden permissions false", all(draft[k] is False for k in ("training_or_inference_authorized", "test_target_access_authorized", "provider_access_authorized", "GPU_operations_authorized", "automatic_scientific_promotion_authorized")))
    check("all30cells both roles fixed by sealed plan", len(plan["study_scope"]["cells"]) == 30 and plan["study_scope"]["checkpoint_roles"] == ["final", "selected"])
    check("all ten arm pairs and six degree bins fixed", len(plan["cross_arm_comparisons"]["all_pairs"]) == 10 and len(plan["degree_analysis"]["bins"]) == 6)
    check("all empty groups retained with complete row arithmetic", plan["output_contract"]["expected_rows"] == {"arm_summary": 420, "cross_arm_decisions": 840, "correct_member_count": 1596, "member_pair": 1512, "seed_summary": 280})
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    check("separate gates loader kernels and renderer", all(name in functions for name in ("admission", "verify_plan_and_source", "verify_live", "verify_consumer", "load_targets", "load_degree", "load_decisions", "partition", "arm_tables", "descriptive_interval", "summarize_seeds", "figures", "analyze", "run", "main")))
    scientific_imports = []
    for node in ast.walk(tree):
        scientific = (isinstance(node, ast.Import) and any(x.name.split(".")[0] in {"torch", "numpy", "torch_geometric", "ogb"} for x in node.names)) or (isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in {"torch", "numpy", "torch_geometric", "ogb"})
        if scientific:
            owners = [f.name for f in functions.values() if f.lineno <= node.lineno <= f.end_lineno]
            scientific_imports.append((owners[0] if owners else "module", node.lineno))
    check("one lazy Torch import in run only", len(scientific_imports) == 1 and scientific_imports[0][0] == "run")
    run_text = ast.get_source_segment(text, functions["run"])
    check("full consumer qualification precedes Torch import", run_text.index("verify_consumer(") < run_text.index("import torch"))
    main_text = ast.get_source_segment(text, functions["main"])
    check("admission precedes execution", main_text.index("admission(") < main_text.index("run("))
    check("path guard precedes remaining imports", text.index("sys.path[:]") < text.index("import argparse"))
    check("no phase or graph/provider source imports", not any(isinstance(n, ast.ImportFrom) and (n.module or "").startswith(("shared_message", "torch_geometric", "ogb")) for n in ast.walk(tree)))
    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "torch" and n.func.attr == "load"]
    check("only two explicit Torch load sites", len(loads) == 2)
    for index, node in enumerate(loads):
        kwargs = {k.arg: constant(k.value) for k in node.keywords}
        check("CPU weights-only input load " + str(index), kwargs == {"map_location": "cpu", "weights_only": True})
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    forbidden = {"cuda", "save", "load_state_dict", "backward", "step", "fit", "train"}
    check("no checkpoint output learning or CUDA call", not any(isinstance(n.func, ast.Attribute) and n.func.attr in forbidden for n in calls))
    check("no shell subprocess network or remote code", not any(isinstance(n, (ast.Import, ast.ImportFrom)) and ((getattr(n, "module", "") or "").split(".")[0] in {"subprocess", "socket", "requests", "urllib"} or any(a.name.split(".")[0] in {"subprocess", "socket", "requests", "urllib"} for a in getattr(n, "names", []))) for n in ast.walk(tree)))
    admission_text = ast.get_source_segment(text, functions["admission"])
    check("independent review and resource/root payload receipts mandatory", all(k in admission_text for k in ("independent_of_implementation_author", "unresolved_blocking_issues", "frozen_training_allocation_extended", "approved_request_payload_sha256")))
    consumer_text = ast.get_source_segment(text, functions["verify_consumer"])
    check("full family and original audit mandatory", all(k in consumer_text for k in ("all30_fits_and_replays_and_lock_qualified", "completed_family_root_audit", "all30_fits_and_replays_qualified", "fixed_updates", "locked_rows_sha256")))
    check("canonical all30rows and exact64inputs", 'len(rows) == 30' in consumer_text and '== 64' in consumer_text)
    check("only already frozen validation target bundle admitted", 'freeze["validation_exports"][task]["bundle"]' in consumer_text and 'unrequested_target_values_indexed_or_summarized' in consumer_text)
    check("no checkpoint load: references compared only", "Replay checkpoint identity differs" in consumer_text and 'torch.load' not in consumer_text)
    degree_text = ast.get_source_segment(text, functions["load_degree"])
    check("existing degree NPY only, no array provider", all(k in degree_text for k in ("degree_file_relative_to_phase", "literal_eval", '"<i8"', 'shape["graph_nodes"]', "frombuffer", "index_select")))
    decisions_text = ast.get_source_segment(text, functions["load_decisions"])
    check("saved pool and ordered IDs used", 'torch.equal(value["node_ids"], ids)' in decisions_text and 'pooled.double().argmax(-1)' in decisions_text and '.mean(' not in decisions_text)
    check("finite literal shapes and single-model identity", all(k in decisions_text for k in ("torch.isfinite", "(m, n, c)", "(n, c)", "Single pooled/member decisions differ")))
    analyze_text = ast.get_source_segment(text, functions["analyze"])
    check("all task role seed arm pair stratum loops", all(k in analyze_text for k in ('plan["study_scope"]["tasks"]', 'plan["study_scope"]["checkpoint_roles"]', 'plan["study_scope"]["seeds"]', 'plan["study_scope"]["arms"]', 'plan["cross_arm_comparisons"]["all_pairs"]', 'masks.items()')))
    check("complete atlas row coverage checked", 'plan["output_contract"]["expected_rows"]' in analyze_text and 'Structural group changed across arms/seeds/roles' in analyze_text)
    check("locked accuracy used for consistency not NLL rescoring", 'context["accuracy_percent"]' in analyze_text and 'cross_entropy' not in text and 'softmax' not in text)
    partition_text = ast.get_source_segment(text, functions["partition"])
    check("full correctness/disagreement partition identities", all(k in partition_text for k in ("both_correct", "lost_correct", "recovered_correct", "both_wrong_same_class", "both_wrong_different_classes", "Decision partition count identity fails", "Disagreement identity fails")))
    arm_text = ast.get_source_segment(text, functions["arm_tables"])
    check("member counts recovery/loss and wrong disagreement retained", all(k in arm_text for k in ("range(m + 1)", "pool_correct_all_members_wrong", "pool_wrong_some_member_correct", "wrong_disagreement_given_both_wrong", "no_member_pairs", "no_either_wrong_nodes", "Member-correct group identity fails")))
    interval_text = ast.get_source_segment(text, functions["descriptive_interval"])
    check("df2 descriptive t formula with no clipping", '4.302652729911275 * sd / math.sqrt(3)' in interval_text and '"df": 2' in interval_text and 'min(' not in interval_text and 'max(' not in interval_text)
    check("three seed values and empty denominators required", 'len(values) == 3' in interval_text and 'all(x is None for x in values)' in interval_text)
    summary_text = ast.get_source_segment(text, functions["summarize_seeds"])
    check("seed pairing no pooling graphs/roles", '("task", "role", "reference", "candidate", "stratum")' in summary_text and 'Complete paired seed group required' in summary_text)
    check("three SVG outputs and complete atlas scaling", all(k in ast.get_source_segment(text, functions["figures"]) for k in ("lost_recovered.svg", "member_disagreement_by_correctness.svg", "degree_signed_map.svg", "summaries", "all_values", "NA")))
    check("rehash inputs and metadata before artifact writing", run_text.index("for ref in payload_refs") < run_text.index('write_json(output / "AGGREGATES.json"') and 'for ref in metadata_refs' in run_text)
    check("external completion and root audit still pending", 'complete_pending_successful_outer_terminal_and_root_audit' in run_text and 'automatic_scientific_promotion_authorized' in run_text)
    check("bounded cap checks and handled failure cleanup", 'budget.check()' in run_text and 'path.unlink()' in run_text and 'partial_scientific_summary_qualified' in run_text)
    for name in ("CONSUMER_AUDIT_SHAPE_v1.json", "RESOURCE_INTERFACE_v1.json"):
        value = json.loads((HERE / name).read_text())
        check("closed interface metadata: " + name, value["execution_authorized"] is False)
    check("no numerical library loaded by author verifier", not any(n == "torch" or n.startswith(("torch.", "torch_geometric", "ogb")) for n in sys.modules))
    helper = HERE / "family_consumer.py"
    helper_text = helper.read_text()
    helper_tree = ast.parse(helper_text)
    helper_functions = {n.name: n for n in helper_tree.body if isinstance(n, ast.FunctionDef)}
    check("separate actual-consumer helper exists", {"bootstrap", "receipts", "verify_witness", "main"}.issubset(helper_functions))
    consumer_main = ast.get_source_segment(helper_text, helper_functions["main"])
    check("actual original consumers called without copied audits", 'admitted, binding_ref = binding(' in consumer_main and 'freeze, freeze_ref = validation_freeze(' in consumer_main and 'value = audit_family_lock(' in consumer_main)
    check("root approval precedes scientific bootstrap", consumer_main.index('approved_request_payload_sha256') < consumer_main.index('bootstrap(request, c)'))
    check("actual-consumer result written after full consumer returns", consumer_main.index('value = audit_family_lock(') < consumer_main.index('output / "CONSUMER_RESULT.json"'))
    check("consumer has no numerical package import or tensor deserialization", not any(isinstance(n, (ast.Import, ast.ImportFrom)) and ((getattr(n, 'module', '') or '').split('.')[0] in {'torch', 'numpy', 'torch_geometric', 'ogb'} or any(a.name.split('.')[0] in {'torch', 'numpy', 'torch_geometric', 'ogb'} for a in getattr(n, 'names', []))) for n in ast.walk(helper_tree)) and 'torch.load' not in helper_text)
    witness_text = ast.get_source_segment(helper_text, helper_functions['verify_witness'])
    check("strict actual witness schema/status and function/source identity", all(k in witness_text for k in ('gnnm-shared-message-complete30-consumer-result-v1', 'passed_existing_complete_family_consumer', 'invoked_consumers', 'scientific_source_manifest', 'locked_rows_sha256')))
    check("same binding/freeze/family refs in result and invocation", 'for key in ("binding", "validation_freeze", "family_lock")' in witness_text and 'Witness qualified a different scientific family' in witness_text and 'Consumer request belongs to another family' in witness_text)
    check("existing completed-chain consumer reused", 'from shared_message_predictive_v3.contract import completed' in witness_text and 'chain = completed(supervisor, witness["live"]["outer_start"])' in witness_text)
    check("successful invocation exact source request and live linkage", all(k in witness_text for k in ('chain["argv"] == argv', 'chain["live"] == witness["live"]', 'chain["gpu_uuid"] == admitted["gpu_uuid"]', 'outer["root_request"]', 'approved_request_payload_sha256')))
    check("saved actual caller/import and artifact receipts bound", all(k in witness_text for k in ('caller_import_policy_sha256', 'imported_phase_modules', 'verify_imported_receipts', 'Consumer result artifact binding differs')))
    bootstrap_text = ast.get_source_segment(helper_text, helper_functions['bootstrap'])
    check("caller augmentation cannot override scientific policy", 'relative not in policy["importable_sha256"]' in bootstrap_text and 'guard["verify_packet"](SCIENCE["sha256"])' in bootstrap_text)
    check("old two-boolean witness gate removed", 'Original root full-family qualification missing' not in consumer_text and 'helper["verify_witness"]' in consumer_text)
    check("post-consumption metadata covers consumer terminals", '*witness_refs' in consumer_text)
    old_path = HERE.parent / 'decision_analysis_impl_v1/decision_analysis.py'
    old_text = old_path.read_text()
    old_tree = ast.parse(old_text)
    old_functions = {n.name: n for n in old_tree.body if isinstance(n, ast.FunctionDef)}
    frozen_functions = ['load_targets','load_degree','load_decisions','fraction','partition','strata','base_row','arm_tables','descriptive_interval','summarize_seeds','svg_text','svg_circle','figure_panels','figures','write_csv','analyze']
    numerical_correspondence = {}
    for name in frozen_functions:
        current_source = ast.get_source_segment(text, functions[name]).encode()
        original_source = ast.get_source_segment(old_text, old_functions[name]).encode()
        check('v1 numerical/figure/input function byte-identical: ' + name, current_source == original_source)
        numerical_correspondence[name] = hashlib.sha256(current_source).hexdigest()
    consumer_draft = json.loads((HERE/'DRAFT_CONSUMER_REQUEST_v1.json').read_text())['request_payload']
    check("consumer draft closed and witness absent", consumer_draft['root_authorized'] is False and all(consumer_draft[k] is None for k in ('source_manifest','binding','validation_freeze','family_lock','resource_approval','root_approval','output_relative_to_phase')))
    receipt = {"schema": "gnnm-shared-message-decision-analysis-AST-source-check-v1", "status": "passed_source_only_unexecuted", "checks": checks, "check_count": len(checks), "runner_sha256": hashlib.sha256(raw).hexdigest(), "sealed_plan_manifest_sha256": constants["PLAN_MANIFEST_SHA"], "Torch_imported": False, "numerical_kernel_or_integration_test_executed": False, "scientific_inputs_outcomes_labels_logits_checkpoints_graph_arrays_SSH_GPU_accessed": False, "independent_review_or_execution_authorized": False, "verification_limit": "Static syntax/contract/gate inspection only; neither numerical correctness nor successful integration is established."}
    receipt['v1_byte_identical_function_sha256'] = numerical_correspondence
    receipt['consumer_source_sha256'] = hashlib.sha256(helper.read_bytes()).hexdigest()
    with (HERE / "SOURCE_CHECK_v1.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"check_count": len(checks), "source_only": True, "diagnostic_unexecuted": True}))


if __name__ == "__main__":
    main()
