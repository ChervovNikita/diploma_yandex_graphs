"""Meaningful pure metadata/order/cap checks; no scientific package import."""
import ast
import collections
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]


def verify():
    checks = []
    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)
    def refuses(name, call):
        try:
            call()
        except (ValueError, RuntimeError):
            check(name, True)
        else:
            check(name, False)
    spec = importlib.util.spec_from_file_location("_trained_benchmark_stdlib_runner", HERE / "runner.py")
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    guard_spec = importlib.util.spec_from_file_location("_trained_benchmark_stdlib_guard", HERE / "guard.py")
    guard = importlib.util.module_from_spec(guard_spec)
    sys.modules[guard_spec.name] = guard
    guard_spec.loader.exec_module(guard)
    expected_modes = ("exact_packed", "exact_serial", "shared_packed")
    check("six_outcome_independent_children", [(r["task"], r["seed"]) for r in runner.children()] == [(t,s) for t in ("Reddit2","ogbn-products") for s in (17,29,43)])
    check("unique_outer_inner_output_paths", len({r[k] for r in runner.children() for k in ("outer_directory","supervisor_directory","output_directory")}) == 18)
    permutations = set(itertools.permutations(expected_modes))
    orders = []
    for task in runner.TASKS:
        for seed in runner.SEEDS:
            for role in runner.ROLES:
                order = runner.round_order(task,seed,role)
                counts = collections.Counter(map(tuple,order))
                check("balanced50:"+task+":"+str(seed)+":"+role, len(order)==50 and set(counts)==permutations and sorted(counts.values())==[8,8,8,8,9,9] and all(set(row)==set(expected_modes) and len(row)==3 for row in order))
                check("deterministic_order:"+task+":"+str(seed)+":"+role, order==runner.round_order(task,seed,role))
                orders.append(order)
    check("distinct_frozen_task_seed_role_orders", len({json.dumps(x) for x in orders})==12)
    refuses("unlisted_seed_refused", lambda:runner.round_order("Reddit2",0,"final"))
    refuses("unlisted_role_refused", lambda:runner.round_order("Reddit2",17,"best"))
    check("nearest_rank_p90", runner.p90(list(range(1,51)))==45)
    refuses("nonfinite_timing_refused", lambda:runner.p90([float('nan')]))
    refuses("nonpositive_timing_refused", lambda:runner.p90([0]))
    values={"exact_packed":[2.]*50,"exact_serial":[4.]*50,"shared_packed":[1.]*50}
    result=runner.summary(values)
    check("stronger_exact_and_signed_reduction", result["stronger_exact_mode"]=="exact_packed" and result["shared_reduction_fraction_against_stronger_exact"]==0.5)
    values["shared_packed"]=[3.]*50
    check("slower_shared_is_retained_not_clipped", runner.summary(values)["shared_reduction_fraction_against_stronger_exact"]==-0.5)
    refuses("partial_timing_family_refused", lambda:runner.summary({"exact_packed":[1.]*50}))
    check("final_then_selected_even_if_same_file", runner.ROLES==("final","selected"))
    check("frozen_tol_caps_and_grace", runner.FIXED["atol"]==runner.FIXED["rtol"]==1e-5 and runner.FIXED["child_cap_seconds"]==600 and runner.FIXED["termination_grace_seconds"]==5 and runner.FIXED["companion_wall_cap_seconds"]==3600)
    draft=json.loads((HERE/'REQUEST_TEMPLATE_v1.json').read_text())
    refuses("unapproved_draft_refused_before_bootstrap", lambda:runner.request_authorized(draft))
    prior=dict(combined_phase_seconds_before_declared_children=18000,
               combined_phase_allocated_disk_bytes_before_declared_children=0,
               companion_wall_seconds_before_declared_children=0,
               companion_artifact_allocated_bytes_before_declared_children=0)
    # First child paths are uncreated. Only path metadata is inspected.
    runner.Budget({}, {}, prior, runner.children()[0], None)
    check("full600_times6_fits_exact_remaining_wall", True)
    refuses("full_remaining_caps_cannot_exceed_phase", lambda:runner.Budget({}, {}, {**prior,"combined_phase_seconds_before_declared_children":18000.001},runner.children()[0],None))
    refuses("prior_companion_failure_cost_not_omitted", lambda:runner.Budget({}, {}, {**prior,"companion_wall_seconds_before_declared_children":0.001},runner.children()[0],None))
    refuses("allocated_artifact_cap_includes_metadata_reserve", lambda:runner.Budget({}, {}, {**prior,"companion_artifact_allocated_bytes_before_declared_children":64*2**20},runner.children()[0],None))
    outer_start=dict(whole_cap_seconds=600,termination_grace_seconds=5)
    failed_terminal=dict(schema="gnnm-whole-process-bound-terminal-v1",START_sha256="start",stdout_stderr_sha256="log",
                         root_request_unchanged=True,whole_supervised_seconds=600.25,complete=False)
    check("failed_cleanup_overrun_fully_charged", runner.terminal_charge(outer_start,failed_terminal,"start","log")==600.25)
    refuses("nonfinite_failure_wall_refused", lambda:runner.terminal_charge(outer_start,{**failed_terminal,"whole_supervised_seconds":float('nan')},"start","log"))
    refuses("negative_failure_wall_refused", lambda:runner.terminal_charge(outer_start,{**failed_terminal,"whole_supervised_seconds":-1},"start","log"))
    refuses("failed_terminal_source_change_refused", lambda:runner.terminal_charge(outer_start,failed_terminal,"changed","log"))
    refuses("outside_phase_path_refused", lambda:guard.confined(PHASE/'../outside'))
    all_sources=[]
    for path in sorted(HERE.glob('*.py')):
        tree=ast.parse(path.read_text(),filename=str(path))
        top_imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        names=[a.name for n in top_imports if isinstance(n,ast.Import) for a in n.names]+[n.module for n in top_imports if isinstance(n,ast.ImportFrom)]
        check("stdlib_top_level:"+path.name, not any(n and n.split('.')[0] in {'torch','numpy','torch_geometric','ogb','shared_message_quality_v3','shared_message_predictive_v3'} for n in names))
        check("no_dunder_file_assignment:"+path.name, not any(isinstance(n,(ast.Assign,ast.AnnAssign)) and any(isinstance(t,ast.Name) and t.id=='__file__' for t in (n.targets if isinstance(n,ast.Assign) else [n.target])) for n in ast.walk(tree)))
        all_sources.append(dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    source=(HERE/'runner.py').read_text()
    check("no_new_targets_training_or_prediction_write_calls", all(s not in source for s in ['load_validation(', 'torch.save(', 'train_step(', 'make_optimizer(', '.backward(', 'cross_entropy(']))
    check("whole_family_consumer_before_run", source.index('family = audit_family_lock(')<source.index('def run('))
    check("own_import_receipt_used", '**guard.receipt(guard_state)' in source and '**scientific.runtime_import_receipt(' not in source)
    check("real_main_explicitly_admitted_without_scientific_mutation", 'expected["importable_sha256"].update(own)' in (HERE/'guard.py').read_text())
    protocol=(PHASE/'trained_shared_message_benchmark_v1/PROTOCOL.md').read_bytes()
    freeze=json.loads((PHASE/'trained_shared_message_benchmark_v1/PROTOCOL_FREEZE_before_quality_v1.json').read_text())
    check("prospective_protocol_exact", hashlib.sha256(protocol).hexdigest()=='6879e4c1e258c7c65c5e178d72091f10956feb77d344e641e195c18802eb41df'==freeze['protocol']['sha256'])
    check("no_scientific_modules_imported_by_checks", not any(n=='torch' or n.startswith(('torch.','numpy','torch_geometric','ogb','shared_message_predictive_v3','shared_message_quality_v3')) for n in sys.modules))
    check("source_check_did_not_create_bytecode_tree", not any(p.name=='__pycache__' or p.suffix in {'.pyc','.pyo'} for p in HERE.rglob('*')))
    return dict(schema="gnnm-trained-inference-source-only-stdlib-check-v1", checks=checks, check_count=len(checks), inspected_sources=all_sources,
                GPU_data_targets_checkpoints_predictions_or_quality_outcomes_accessed=False, execution_authorized=False)


if __name__=='__main__':
    value=verify()
    print(json.dumps(dict(check_count=value['check_count'],all_passed=all(v['passed'] for v in value['checks'])),sort_keys=True))
