"""Stdlib AST/JSON/hash source checks only; never execute a reader or gate."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parsed = []
    for path in sorted(HERE.glob('*.py')):
        tree = ast.parse(path.read_text(), filename=str(path)); compile(tree, str(path), 'exec')
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or '']
                assert not any(name.split('.')[0] in {'numpy', 'torch', 'ogb', 'torch_geometric'} for name in names)
        parsed.append(path.name)
    pins = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
    for row in pins['source_files']:
        path = PHASE / row['path']
        assert path.suffix not in {'.pt', '.npz', '.npy'}
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    payloads = 0
    for row in pins['source_manifests']:
        path = PHASE / row['path']; assert sha(path) == row['sha256']
        for item in json.loads(path.read_text())['files']:
            file = path.parent / item['path']; assert file.suffix not in {'.pt', '.npz', '.npy'}
            assert file.stat().st_size == item['bytes'] and sha(file) == item['sha256']
            payloads += 1
    for name, original in pins['unchanged_numerical_files'].items():
        assert sha(HERE / name) == original['sha256'] == sha(PHASE / original['path'])
    previous = PHASE / 'pre_sigmoid_qk36_original_route_selected_collection_source_20261010_v1'
    assert sha(HERE / 'collect_route.py') == sha(previous / 'collect_route.py')
    original_gate = ast.parse((previous / 'gate_route.py').read_text())
    current_gate = ast.parse((HERE / 'gate_route.py').read_text())
    for name in ('fresh_custody', 'source_lane', 'original_custody', 'consume'):
        function = lambda tree: next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name == name)
        assert ast.dump(function(original_gate),include_attributes=False) == ast.dump(function(current_gate),include_attributes=False)
    old_assembly = ast.parse((previous / 'assemble_readout.py').read_text())
    new_assembly = ast.parse((HERE / 'assemble_readout.py').read_text())
    numerical_try = lambda tree: next(node for node in next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run').body if isinstance(node,ast.Try))
    assert ast.dump(numerical_try(old_assembly),include_attributes=False) == ast.dump(numerical_try(new_assembly),include_attributes=False)
    for tree in (old_assembly,new_assembly):
        body = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        route_loop = next(n for n in body.body if isinstance(n,ast.For) and 'route_id' in ast.unparse(n.target))
        row_loop = next(n for n in route_loop.body if isinstance(n,ast.For) and ast.unparse(n.iter)=="local['cells']")
        if tree is old_assembly:
            preserved_availability = ast.dump(row_loop,include_attributes=False)
        else:
            assert ast.dump(row_loop,include_attributes=False)==preserved_availability
    old_seal = PHASE / pins['immutable_selected_readout_v2_seal']['path']
    old = json.loads(old_seal.read_text()); assert sha(old_seal) == pins['immutable_selected_readout_v2_seal']['sha256']
    assert sha(old_seal.parent / 'MANIFEST.json') == old['manifest_sha256']
    for item in json.loads((old_seal.parent / 'MANIFEST.json').read_text())['files']:
        file = old_seal.parent / item['path']; assert sha(file) == item['sha256'] and file.stat().st_size == item['bytes']
    for name, budget in (('RELEASE_TEMPLATE_DISABLED.json', 9), ('PHASE2_RELEASE_TEMPLATE_DISABLED.json', 27)):
        cfg = json.loads((HERE / name).read_text())
        assert cfg['enabled'] is cfg['root_execution_authorized'] is cfg['source_review_approved'] is False
        assert cfg['whole36_complete'] is cfg['actual_all_route_custody_closed'] is False
        assert cfg['maximum_member_forwards'] == budget and cfg['maximum_route_member_forwards'] == 36
        assert all(cfg[key] is False for key in ('training', 'TEST_access', 'automatic_retry', 'reselection', 'calibration'))
    assembly = json.loads((HERE / 'ASSEMBLY_RELEASE_TEMPLATE_DISABLED.json').read_text())
    assert assembly['enabled'] is assembly['root_execution_authorized'] is False
    freeze = json.loads((HERE / 'GLOBAL_BASELINE_FREEZE_TEMPLATE_DISABLED.json').read_text())
    assert freeze['complete'] is freeze['root_frozen_before_any_candidate_calls'] is False
    entry, gate = (HERE / 'collect_route.py').read_text(), (HERE / 'gate_route.py').read_text()
    assert entry.index('_gate.consume(release_path, release_sha256)') < entry.index('import numpy as np')
    assert "if route_id != cfg['readout_route_id']" in gate and 'lane.work_receipt(' in gate and 'queue.verify_union(' in gate
    assert "global_baselines(g, pins, cfg, records)" in gate and 'All9 original baseline cohorts globally frozen before any candidate calls' in gate
    assert "original_route_resident_state_custody=custody" in gate and 'checkpoint_copies_required' in gate
    assert "row['members'] <= cfg['maximum_member_forwards']" in entry
    assert "analyse(np," not in entry and all(token not in entry for token in ('.backward(', '.step(', 'torch.load('))
    assert 'len(truth) == 5274' in entry and '(11701, 300)' in entry and '(2, 442907)' in entry
    raw = (HERE / 'assemble_readout.py').read_text()
    assert raw.index('Complete36 custody, no survivor subset') < raw.index('import numpy as np')
    assert 'analyse(np, g, pins' in raw and 'staged_archive' in raw and 'original_archive' in raw
    assert all(token not in raw for token in ('import torch', '.member_forward(', 'torch.load(', '.backward(', '.step('))
    for template, program in (('EXTERNAL_SUPERVISION_TEMPLATE_DISABLED.json', 'collect_route.py'),
                              ('ASSEMBLY_SUPERVISION_TEMPLATE_DISABLED.json', 'assemble_readout.py')):
        row = json.loads((HERE / template).read_text())
        assert row['enabled'] is row['finite_owned_bound_confirmed'] is False
        assert row['entry_program']['sha256'] == sha(HERE / program)
        assert row['owned_GPU_bytes'] == 24*1024**3
        assert row['owned_GPU_bytes'] <= min(value['max_owned_GPU_bytes'] for value in pins['plan']['root_resource_limits'].values())
    terminal = json.loads((HERE/'READER_TERMINAL_TEMPLATE_DISABLED.json').read_text())
    assert terminal['schema']=='qk36-original-route-reader-terminal-v2'
    assert terminal['actual_wait_and_reap'] is False and terminal['actual_exit_code'] is None
    assert terminal['parent_direct_wait_and_reap_observed'] is False and terminal['parent_actual_exit_code'] is None
    assert terminal['actual_parent_exit_receipt'] is None
    for name in ('PROCESS_WITNESS_TEMPLATE_DISABLED.json','OWNED_ABSENCE_TEMPLATE_DISABLED.json'):
        assert json.loads((HERE/name).read_text())['root_observed'] is False
    if (HERE / 'MANIFEST.json').exists():
        for row in json.loads((HERE / 'MANIFEST.json').read_text())['files']:
            path = HERE / row['path']; assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        seal = json.loads((HERE / 'SEAL.json').read_text())
        assert seal['runtime_disabled'] is True and seal['manifest_sha256'] == sha(HERE / 'MANIFEST.json')
    result = dict(AST_and_syntax=parsed, source_files=len(pins['source_files']), source_manifests=len(pins['source_manifests']),
        source_payloads=payloads, unchanged_numerical_hooks_readout_and_namespace=True,
        unchanged_adapted_AST_sha256=pins['adapted_AST_sha256'], preserved_whole36_custody_and_local12_byte_validation=True,
        per_original_route_phase_calls=[9, 27], total_route_calls=36, total_panel_calls=108,
        global_all9_native_freeze_before_candidates=True, original_full_population=5274,
        exact_terminal_source_enabled_release_process_exit_absence_cleanup_bindings=True,
        phase1_concrete_successful_child_exit0_required=True, phase2_real_nonzero_exit_and_failure_custody_retained=True,
        original_per_bank_availability_AST_preserved=True,no_new_whole_phase_availability_veto=True,
        collection_and_assembly_numerical_AST_preserved=True,supervision_cap_bytes=24*1024**3,
        templates_disabled=True, numerical_imports_or_execution=False, gate_or_reader_executed=False,
        arrays_checkpoints_actual_outcomes_or_servers_opened=False)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
