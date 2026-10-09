"""Stdlib source/hash/AST checks only; no entry point or numeric fixture runs."""
import ast
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapted(file, function, target):
    source = ast.parse((HERE / file).read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == function)
    ns = {'ast': ast, 'copy': copy}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), file + ':AST-only', 'exec'), ns)
    tree = ns[function](target.read_text()); compile(tree, str(target) + ':syntax-only', 'exec')
    return tree, hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def main():
    parsed = []
    for path in sorted(HERE.glob('*.py')):
        tree = ast.parse(path.read_text(), filename=str(path)); compile(tree, str(path), 'exec')
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or '']
                assert not any(n.split('.')[0] in ('torch', 'numpy', 'torch_geometric', 'ogb') for n in names), path
        parsed.append(path.name)
    pins = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text()); count = 0
    for row in pins['source_files']:
        path = PHASE / row['path']; assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], row['path']
    for manifest in pins['source_manifests']:
        path = PHASE / manifest['path']; assert sha(path) == manifest['sha256'] and path.stat().st_size == manifest['bytes']
        for row in json.loads(path.read_text())['files']:
            payload = path.parent / row['path']; assert payload.is_file() and sha(payload) == row['sha256'] and payload.stat().st_size == row['bytes'], str(payload)
            count += 1
    release = json.loads((HERE / 'RELEASE_TEMPLATE_DISABLED.json').read_text())
    flags = ('enabled', 'root_execution_authorized', 'source_review_approved', 'whole36_complete', 'actual_all_route_custody_closed',
        'trusted_selected_state_deserialization_authorized', 'all36_server_only_states_staged_and_transfer_cost_bound',
        'qualified_readout_route_confirmed', 'finite_external_owner_confirmed', 'inclusive_cost_accounting_confirmed')
    assert all(release[k] is False for k in flags)
    assert release['maximum_member_forwards'] == 108 and release['execution_source_commit'] == 'a128c164bc33d7843a13c290b22ef1c311f4184a'
    assert all(release[k] is False for k in ('training', 'TEST_access', 'automatic_retry', 'reselection', 'calibration'))
    evidence = json.loads((HERE / 'FRESH_CUSTODY_TEMPLATE_DISABLED.json').read_text())
    assert evidence['complete'] is evidence['root_observed'] is evidence['checked_after_all36_union'] is False
    assert set(evidence['routes']) == set(pins['plan']['route_ids'])
    supervision = json.loads((HERE / 'EXTERNAL_SUPERVISION_TEMPLATE_DISABLED.json').read_text())
    assert supervision['enabled'] is supervision['finite_owned_bound_confirmed'] is False
    assert supervision['entry_program']['sha256'] == sha(HERE / 'collect36.py')
    original_path = PHASE / pins['reuse']['variable_member_collect']['path']
    tree, collect_hash = adapted('collection_hooks.py', 'collector_tree', original_path)
    original = ast.parse(original_path.read_text()); old = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'collect_cell')
    def tail(function, normalize=False):
        function = copy.deepcopy(function); attempt = next(n for n in function.body if isinstance(n, ast.Try))
        index = next(i for i,n in enumerate(attempt.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'forward_start' for t in n.targets))
        attempt.body = attempt.body[index:]
        if normalize:
            for node in ast.walk(attempt):
                if isinstance(node,ast.If) and ast.unparse(node.test) == "state['operator'] == 'native_tied'":
                    node.test = ast.parse("state['arm'] == 'be_init'").body[0].value
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'cohort_path' for t in node.targets):
                    node.value = ast.parse("output/'raw'/('be_init_cohorts_'+str(seed)+'.npz')").body[0].value
        return ast.dump(attempt,include_attributes=False)
    assert tail(old) == tail(tree.body[0],True), 'Original variable-M inference/loss/persistence/failure tail changed'
    assert 'range(model.members)' in ast.unparse(tree) and 'math.log(model.members)' in ast.unparse(tree)
    relation = PHASE / pins['reuse']['relation_readout']['path']
    _, arrays_hash = adapted('readout36.py', 'arrays_tree', relation)
    report_tree, report_hash = adapted('readout36.py', 'report_tree', relation)
    assert "frozen[seed, right.split('/')[0]]" in ast.unparse(report_tree) or "frozen[(seed, right.split('/')[0])]" in ast.unparse(report_tree)
    hashes = dict(collect_cell_AST_sha256=collect_hash, arrays_AST_sha256=arrays_hash, analysis_run_AST_sha256=report_hash,
        pooled_pair_AST_sha256=pins['adapted_AST_sha256']['pooled_pair_AST_sha256'])
    assert hashes == pins['adapted_AST_sha256']
    code = ast.parse((HERE / 'readout36.py').read_text())
    fn = next(n for n in code.body if isinstance(n,ast.FunctionDef) and n.name == 'comparisons')
    ns = {}; exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])), 'prospective-labels-only','exec'),ns)
    comparisons = ns['comparisons'](); assert len(comparisons) == len(set(r[0] for r in comparisons)) == 17
    assert comparisons[0][0] == 'be_init/pre_sigmoid_split-be_init/native_tied' and comparisons[1][0] == 'be_init/pre_sigmoid_split-be_init/active_reversible_exp'
    gate_fn = next(n for n in code.body if isinstance(n,ast.FunctionDef) and n.name == 'pilot_decision')
    fixed = next(n for n in gate_fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'expected' for t in n.targets))
    expected = {k.arg:ast.literal_eval(k.value) for k in fixed.value.keywords}
    protocol = json.loads((PHASE / pins['reuse']['pilot_protocol']['path']).read_text())
    assert expected == protocol['prospective_decisions']['quality_and_member_gate']
    entry=(HERE/'collect36.py').read_text(); hook=(HERE/'collection_hooks.py').read_text(); gate=(HERE/'gate36.py').read_text()
    assert entry.index('_gate.consume(release_path, release_sha256)') < entry.index('import numpy as np')
    assert entry.index("'native_tied', seed") < entry.index('if frozen:') < entry.index('OPERATORS[1:]')
    for token in ('.backward(', '.step(', 'torch.load(', 'local_transition('): assert token not in entry, token
    for token in ('individual_best_bank_only', "saved['body_global']", "model.set_global(saved['global'])", 'later_execution_authorized=True', '2 * bodies'):
        assert token in hook, token
    assert 'queue.verify_union(' in gate and 'lane.work_receipt(' in gate and 'Fresh root custody after complete36 union' in gate
    assert 'boot_id' in gate and 'checked_after_all36_union' in gate
    if (HERE/'MANIFEST.json').exists():
        for row in json.loads((HERE/'MANIFEST.json').read_text())['files']:
            path=HERE/row['path'];assert sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],row['path']
    print(json.dumps(dict(AST_and_syntax=parsed,source_files=len(pins['source_files']),source_manifests=len(pins['source_manifests']),source_payloads=count,
        adapted_AST_sha256=hashes,unchanged_actual_M1_M4_inference_loss_failure_tail=True,
        exact_frozen_pilot_gate=True,ordered_comparisons=17,all9_native_baselines_first=True,maximum_member_calls=108,
        selected_state_installer_and_own_modes_wired=True,actual_two_Adam_generations_charged=True,
        templates_disabled=True,numerical_imports_or_execution=False,collector_or_gate_entry_points_executed=False,
        arrays_checkpoints_TEST_or_running_outcomes_opened=False,remote_calls=False,runtime_qualified=False),indent=2,sort_keys=True))


if __name__ == '__main__': main()
