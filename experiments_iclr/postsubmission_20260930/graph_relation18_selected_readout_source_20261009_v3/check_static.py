"""Text/hash/AST checks only. Never import a collector or numerical provider."""
import ast
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adaptation(function, file, target):
    parsed = ast.parse((HERE / file).read_text())
    node = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == function)
    namespace = {'ast': ast, 'copy': copy}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), file + ':pure-AST', 'exec'), namespace)
    tree = namespace[function](target.read_text())
    compile(tree, str(target) + ':syntax-only', 'exec')
    return tree, hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def main():
    parsed = []
    for path in sorted(HERE.glob('*.py')):
        tree = ast.parse(path.read_text(), filename=str(path)); compile(tree, str(path), 'exec')
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or '']
                assert not any(n.split('.')[0] in ('torch', 'numpy', 'ogb', 'torch_geometric') for n in names), path
        parsed.append(path.name)
    pins = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
    for row in pins['source_files']:
        path = PHASE / row['path']; assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['path']
    source_payloads = 0
    for manifest in pins['source_manifests']:
        path = PHASE / manifest['path']; assert path.stat().st_size == manifest['bytes'] and sha(path) == manifest['sha256']
        for row in json.loads(path.read_text())['files']:
            payload = path.parent / row['path']
            assert payload.is_file() and payload.stat().st_size == row['bytes'] and sha(payload) == row['sha256'], str(payload)
            source_payloads += 1
    release = json.loads((HERE / 'RELEASE_TEMPLATE_DISABLED.json').read_text())
    flags = ('enabled', 'root_execution_authorized', 'source_review_approved', 'whole_relation12_complete',
        'exact_original12_union_complete', 'historical_whole24_and_collection_closed', 'all_owners_and_children_terminal',
        'trusted_checkpoint_deserialization_authorized', 'historical_prediction_archive_opening_authorized',
        'runtime_resource_readiness_confirmed', 'collection_and_analysis_cost_charged', 'external_owned_bound_confirmed')
    assert all(release[k] is False for k in flags)
    assert release['maximum_member_forwards'] == 48 and release['historical_archive_choice'] == 'reuse_exact_six_archives'
    assert all(release[k] is False for k in ('TEST_access', 'training', 'reselection', 'calibration', 'automatic_retry'))
    terminal = json.loads((HERE / 'TERMINAL_EVIDENCE_TEMPLATE_DISABLED.json').read_text())
    assert terminal['complete'] is terminal['root_observed'] is terminal['all_owners_and_children_terminal'] is False
    assert len(terminal['relation_children']) == len(terminal['original12_children']) == 12
    supervision = json.loads((HERE / 'EXTERNAL_SUPERVISION_TEMPLATE_DISABLED.json').read_text())
    assert supervision['enabled'] is supervision['root_execution_authorized'] is supervision['finite_owned_bound_confirmed'] is False
    assert supervision['owned_entry_program']['sha256'] == sha(HERE / 'collect_relation18.py')
    collect_tree, collect_hash = adaptation('adapt_tree', 'collection_hooks.py', PHASE / pins['reuse']['legacy_collect']['path'])
    assert collect_hash == pins['adapted_collect_cell_AST_sha256']
    pair_tree, pair_hash = adaptation('pooled_pair_tree', 'readout.py', PHASE / pins['audited_Wiki24_analysis']['path'])
    assert pair_hash == pins['pooled_pair_AST_sha256']
    original = ast.parse((PHASE / pins['reuse']['legacy_collect']['path']).read_text())
    original_function = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'collect_cell')
    adapted = collect_tree.body[0]
    for tree in (original_function, adapted):
        forwards = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'member_forward']
        assert len(forwards) == 1
    # Every original statement from inference entry through serialization/failure/finally survives,
    # except the prospective baseline label/path/keys. Check via a normalized AST copy.
    def numerical_tail(function):
        function = copy.deepcopy(function); attempt = next(n for n in function.body if isinstance(n, ast.Try))
        index = next(i for i,n in enumerate(attempt.body) if isinstance(n,ast.Assign)
            and isinstance(n.value,ast.Constant) and n.value.value == 'inference_and_CPU_transfer')
        attempt.body = attempt.body[index:]
        for n in ast.walk(attempt):
            if isinstance(n,ast.Constant) and n.value in ('alphaF','alphaF_cohorts_'):
                n.value = {'alphaF':'plain','alphaF_cohorts_':'plain_cohorts_'}[n.value]
            if isinstance(n,ast.keyword) and n.arg in ('alphaF_cell','alphaF_archive'):
                n.arg = n.arg.replace('alphaF','plain')
        return ast.dump(attempt,include_attributes=False)
    assert numerical_tail(original_function) == numerical_tail(adapted)
    assert 'range(4)' not in ast.unparse(pair_tree)
    # Compile the unchanged completed-output historical metadata adaptation.
    gate_tree = ast.parse((HERE / 'readout_gate.py').read_text())
    history_fn = next(n for n in gate_tree.body if isinstance(n,ast.FunctionDef) and n.name == 'historical')
    adapter_class = next(n for n in history_fn.body if isinstance(n,ast.ClassDef) and n.name == 'CompletedOutput')
    namespace = {'ast': ast, 'require': lambda value, message: value or (_ for _ in ()).throw(AssertionError(message))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[adapter_class],type_ignores=[])), 'CompletedOutput:AST-only','exec'),namespace)
    source = ast.parse((PHASE / pins['reuse']['historical_collect']['path']).read_text())
    consume_fn = copy.deepcopy(next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name == 'consume'))
    adapter = namespace['CompletedOutput'](); consume_fn = adapter.visit(consume_fn)
    assert adapter.count == adapter.route_count == 1
    compile(ast.fix_missing_locations(ast.Module(body=[consume_fn],type_ignores=[])), 'historical-consume:syntax-only','exec')
    assert "namespace['preflight']" not in (HERE / 'readout_gate.py').read_text()
    old = PHASE / 'graph_relation18_selected_readout_source_20261009_v1'
    assert sha(HERE / 'readout.py') == sha(old / 'readout.py')
    assert sha(HERE / 'collection_hooks.py') == sha(old / 'collection_hooks.py')
    # The sole collection change selects the staged, byte-identical route.
    assert (HERE / 'collect_relation18.py').read_text().replace("row['raw_archive'] = row['staged_archive']",
        "row['raw_archive'] = row['original_archive']") == (old / 'collect_relation18.py').read_text()
    old_gate = ast.parse((old / 'readout_gate.py').read_text())
    def named_function(tree, name):
        return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == name)
    assert ast.dump(named_function(gate_tree,'original_union'),include_attributes=False) == ast.dump(named_function(old_gate,'original_union'),include_attributes=False)
    for tree in (old_gate,gate_tree):
        fn = named_function(tree,'relations')
        lane = next(n for n in fn.body if isinstance(n,ast.For) and 'lane_results' in ast.unparse(n.iter))
        if tree is old_gate:
            unchanged_lane = ast.dump(lane,include_attributes=False)
        else:
            assert ast.dump(lane,include_attributes=False) == unchanged_lane
    previous = PHASE / 'graph_relation18_selected_readout_source_20261009_v2'
    assert all(sha(HERE / name) == sha(previous / name) for name in ('collect_relation18.py', 'collection_hooks.py', 'readout.py'))
    assert "('PID', 'start_ticks', 'pgid', 'sid', 'argv')" in (HERE / 'readout_gate.py').read_text()
    assert "launch['owner_identity']['observation_complete'] is True" in (HERE / 'readout_gate.py').read_text()
    custody = json.loads((HERE / 'HISTORICAL_ORIGINAL_ROUTE_CUSTODY_TEMPLATE_DISABLED.json').read_text())
    stage = json.loads((HERE / 'HISTORICAL_ARCHIVE_STAGING_TEMPLATE_DISABLED.json').read_text())
    assert custody['root_observed'] is custody['original_reader_gate_passed'] is False
    assert stage['complete'] is stage['root_observed'] is stage['all_original_and_staged_hashes_verified'] is False
    entry = (HERE / 'collect_relation18.py').read_text(); hook = (HERE / 'collection_hooks.py').read_text(); gate = (HERE / 'readout_gate.py').read_text()
    assert entry.index('consume(release_path, release_sha256)') < entry.index('import numpy as np')
    assert entry.index("collect('alphaF')") < entry.index('if frozen:') < entry.index("for condition in POLICIES[1:]")
    assert 'relation.reconstruct_selected(state' in hook and "session.model.models[0].body._global is state['global']" in hook
    assert "session.optimizers[0].state" in hook and 'original_init(instance, *args, **kwargs)' in hook
    for token in ('.backward(', '.step(', 'load_state_dict(', 'local_transition(', 'torch.load('):
        assert token not in entry, token
    for token in ('graph-relation-full12-normal77-family-closure-v1', 'Wiki12-old10-plus-never-started2-union-closure-v1',
        'Same original whole24 gate', 'Original relation selected bytes; no reselection', 'Root exact relation child terminal custody'):
        assert token in gate
    references = pins['reference_pointers']['records']
    assert len(references) == 6 and {(r['seed'],r['arm']) for r in references} == {(s,c) for s in (6101,6203,6307) for c in ('single','independent4')}
    assert pins['original_review_manifest_sha256'] == pins['reuse']['review_manifest']['sha256']
    if (HERE / 'MANIFEST.json').exists():
        for row in json.loads((HERE / 'MANIFEST.json').read_text())['files']:
            path = HERE / row['path']; assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['path']
    result = dict(AST_and_syntax=parsed, bound_source_files=len(pins['source_files']), bound_source_manifests=len(pins['source_manifests']),
        bound_source_payloads=source_payloads, original_inference_loss_persistence_failure_tail_unchanged=True,
        historical_metadata_adaptation_syntax_only=True, original_route_custody_template_disabled=True,
        original_union_and_relation_child_lanes_AST_unchanged=True,
        original_readout_and_collection_hooks_byte_unchanged=True,
        collection_change_only_staged_byte_identical_archive_path=True,
        adapted_collect_cell_AST_sha256=collect_hash, pooled_pair_AST_sha256=pair_hash,
        actual_M1_M4_only=True, exact_six_archive_choice=True, templates_disabled=True,
        collector_or_gate_entry_points_executed=False, numerical_imports_or_execution=False,
        checkpoint_array_or_outcome_payload_reads=False, remote_calls=False, runtime_qualified=False)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
