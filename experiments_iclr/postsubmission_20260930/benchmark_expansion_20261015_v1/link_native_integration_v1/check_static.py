"""CPU-only epoch/import/source-custody checks; no tensor library or model."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


def run():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('_link_epoch_static', here/'epoch.py')
    epoch = importlib.util.module_from_spec(spec); spec.loader.exec_module(epoch)
    for family in ('NCN', 'NCNC'):
        layout = epoch.epoch_layout(family, 3*65536+123)
        assert layout['optimizer_maps_per_epoch'] == 3 and layout['dropped_positive_tail'] == 123
        assert layout['task_query_records_per_epoch'] == 6*65536
        assert epoch.progress(layout, 0, 0)['maps_to_next_completed_epoch'] == 3
        assert epoch.progress(layout, 1, 2)['maps_to_next_completed_epoch'] == 1
        assert epoch.progress(layout, 1, 3)['completed_epochs'] == 1
        assert epoch.progress(layout, 1, 3)['maps_to_next_completed_epoch'] == 3
        assert epoch.progress(layout, 2, 1)['completed_epochs'] == 1
    layout = epoch.epoch_layout('BUDDY', 2049, 4098)
    assert layout['optimizer_maps_per_epoch'] == 5 and layout['included_partial_tail'] == 2
    assert epoch.progress(layout, 1, 5)['completed_epochs'] == 1
    for values in ((0,1), (1,6), (-1,0)):
        try: epoch.progress(layout, *values)
        except ValueError: pass
        else: raise AssertionError('Inconsistent epoch cursor admitted')
    spec = importlib.util.spec_from_file_location('_link_integration_static', here/'session.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    library, roster, receipts = module._dependencies()
    assert len([row for row in roster.values() if row['kind'] == 'query_objective']) == 8
    def rejects(function, exception=ValueError):
        try: function()
        except exception: return
        raise AssertionError('Unsafe or incompatible binding admitted')
    checks = here/'.checks'; checks.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=checks) as temporary:
        base = Path(temporary)
        root = base/'repository'; root.mkdir(); root = root.resolve()
        redesign = root/'redesign'; redesign.mkdir()
        recipe = {'_repository_root': str(root), '_redesign_root': 'redesign',
                  'data_files': {role: 'data/'+role+'.pt' for role in ('graph', 'train', 'valid')}}
        config, data_root = module._prepare_recipe(recipe, 'data')
        assert config['_redesign_root'] == str(redesign)
        assert config['_native_callback'] == str(redesign/'smoke_20261014/link_port/callbacks.py')
        assert data_root == root/'data' and all(Path(path).is_relative_to(root) for path in config['data_files'].values())
        for field in ('_redesign_root', '_native_callback'):
            rejects(lambda field=field: module._prepare_recipe({**recipe, field: '../outside'}, 'data'))
        rejects(lambda: module._prepare_recipe(recipe, '../outside'))
        rejects(lambda: module._prepare_recipe({**recipe, 'data_files': {**recipe['data_files'], 'train': '../outside'}}, 'data'))
        rejects(lambda: module._prepare_recipe({**recipe, 'data_files': {**recipe['data_files'], 'test': 'never_open.pt'}}, 'data'))
        rejects(lambda: module._prepare_recipe({**recipe, 'data_files': None}, 'data'))
        (root/'escaped').symlink_to(base)
        rejects(lambda: module._prepare_recipe({**recipe, 'data_files': {**recipe['data_files'], 'valid': 'escaped/valid.pt'}}, 'data'))
        sample = root/'sample.bin'; sample.write_bytes(b'a'*(1024*1024+7))
        actual = module._receipt(sample)
        assert actual['bytes'] == 1024*1024+7 and actual['sha256'] == hashlib.sha256(sample.read_bytes()).hexdigest()
        assert module._native_hash_bindings({'_native_callback_sha256': actual['sha256']}, actual)
        rejects(lambda: module._native_hash_bindings({'_native_callback_sha256': '0'*64}, actual))
        rejects(lambda: module._native_hash_bindings({'callback_sha256': 'malformed'}, actual))
        assert module._job_callback_binding({'callback_module': 'sample.bin', 'callback_sha256': actual['sha256']}, root, sample)['actual'] == actual
        rejects(lambda: module._job_callback_binding({'callback_module': '../outside'}, root, sample))
        rejects(lambda: module._job_callback_binding({'callback_sha256': '0'*64}, root, sample))
        rejects(lambda: module.create_adapter(None, {'family': 'link', 'TEST_access': False}, SimpleNamespace(track='source_reference'), root), NotImplementedError)
    native = module._native_factory({})
    assert native.__file__.endswith('link_port/callbacks.py')
    rejects(lambda: module._native_factory({'_native_callback_sha256': '0'*64}))
    for path in here.glob('*.py'):
        compile(path.read_bytes(), str(path), 'exec')
    tree = ast.parse((here/'session.py').read_text())
    forbidden = {'os.system', 'subprocess', 'urllib', 'requests', 'socket', 'paramiko'}
    assert not any(isinstance(node, ast.Import) and any(alias.name in forbidden for alias in node.names) for node in ast.walk(tree))
    assert 'torch' not in sys.modules
    frozen = here.parents[1]/'reference_wrapper_redesign_20261013_v1/smoke_20261014/link_port'
    frozen_rows = {row['path']: row for row in json.loads((frozen/'MANIFEST.json').read_text())['files']}
    for name in ('callbacks.py', 'boundaries.py', 'source_loader.py'):
        raw = (frozen/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == frozen_rows[name]['sha256']
    return {'python_syntax_and_import_without_Torch': 'passed', 'frozen_link_helper_hashes': 'passed',
        'frozen_smoke_ports_unchanged': 'passed', 'NCN_NCNC_dropped_tail_epoch_layout': 'passed',
        'BUDDY_included_tail_epoch_layout': 'passed', 'partial_resume_and_next_epoch_boundaries': 'passed',
        'inconsistent_cursor_rejection': 'passed', 'new_source_model_or_fit_execution': False,
        'repository_root_paths_and_symlink_escape_rejection': 'passed',
        'explicit_GRAPH_TRAIN_VALID_only_data_binding': 'passed',
        'streamed_file_SHA256': 'passed', 'supplied_callback_SHA256_binding_and_rejection': 'passed',
        'frozen_native_callback_and_external_dependency_preflight': 'passed',
        'generic_policy_rejected_before_model_construction': 'passed',
        'actual_native_numerical_qualification': 'pending allocation', 'TEST_opened': False}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
