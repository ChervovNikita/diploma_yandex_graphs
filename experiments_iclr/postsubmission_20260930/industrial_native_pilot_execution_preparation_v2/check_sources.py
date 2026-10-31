"""Source-only AST/hash/JSON checks; never import, compile or execute a model.

Default is read-only. --write-receipt writes only this packet's static receipt
before sealing; sealed verification never changes a file.
"""
from __future__ import annotations
import argparse
import ast
import copy
import datetime
import hashlib
import html
import json
from pathlib import Path
import re
import sys
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
V1 = BASE / 'industrial_native_pilot_preparation_v1'
R17 = BASE / 'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v2'
FOLLOWUP = BASE / 'industrial_graph_followup_v1'
COMMIT = '3b9b115490249cc777227c846babfb55f35bd8c4'
CHECKS = []


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def check(name, passed, detail=None):
    CHECKS.append({'name': name, 'passed': bool(passed), **({'detail': detail} if detail else {})})
    if not passed:
        raise AssertionError(name)


def tree(path):
    return ast.parse(Path(path).read_text(), filename=str(path))


def definition(path, name):
    return next(node for node in tree(path).body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)


def segment(path, node):
    return ''.join(Path(path).read_text().splitlines(keepends=True)[node.lineno - 1:node.end_lineno])


def calls(node):
    return [ast.unparse(item.func) for item in ast.walk(node) if isinstance(item, ast.Call)]


def verify_payload(folder, manifest_sha, seal_sha):
    manifest = load(folder / 'MANIFEST.json')
    seal = load(folder / 'SEAL.json')
    check(folder.name + ' manifest identity', digest(folder / 'MANIFEST.json') == manifest_sha)
    check(folder.name + ' seal identity', digest(folder / 'SEAL.json') == seal_sha)
    check(folder.name + ' seal binds manifest', seal['manifest_sha256'] == manifest_sha)
    check(folder.name + ' payload count', len(manifest['payload']) == manifest['payload_files'])
    for item in manifest['payload']:
        path = folder / item['path']
        check(folder.name + ' unchanged payload ' + item['path'],
              path.is_file() and not path.is_symlink() and path.stat().st_size == item['bytes']
              and digest(path) == item['sha256'])


def reference_adaptation_check():
    source = FOLLOWUP.parent / 'industrial_native_pilot_preparation_v1/sources/paper__lib__graph__data.py'
    original = copy.deepcopy(definition(source, '_load_graphland_data'))
    candidate = definition(ROOT / 'public_reference.py', 'reference_public_load')
    original.name = candidate.name
    body = []
    removed_targets = removed_labels = removed_mask = False
    for node in original.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'targets_df' for t in node.targets):
            body.append(ast.parse('public_node_count = len(features_df)').body[0])
            removed_targets = True
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'labels' for t in node.targets):
            removed_labels = True
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == '_mask_labeled_nodes':
            removed_mask = True
        else:
            body.append(node)
    original.body = body
    for node in ast.walk(original):
        if isinstance(node, ast.keyword) and node.arg == 'num_nodes':
            node.value = ast.Name(id='public_node_count', ctx=ast.Load())
        if isinstance(node, ast.Dict):
            pairs = [(key, value) for key, value in zip(node.keys, node.values)
                     if not (isinstance(key, ast.Constant) and key.value == 'labels')]
            node.keys, node.values = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    check('Public oracle removes exactly target read/labels/masking', removed_targets and removed_labels and removed_mask)
    check('Public oracle exact pinned AST after declared closed-role adaptations',
          ast.dump(original, include_attributes=False) == ast.dump(candidate, include_attributes=False))


def main(write_receipt=False):
    bindings = load(ROOT / 'SOURCE_BINDINGS.json')
    check('Pinned native commit', bindings['native_source_commit'] == COMMIT)
    for item in bindings['external_inputs'] + bindings['root_runtime_capability_evidence']:
        path = Path(item['path'])
        check('External metadata/source binding ' + path.parent.name + '/' + path.name,
              path.is_file() and path.stat().st_size == item['bytes'] and digest(path) == item['sha256'])
    verify_payload(V1, '79ffa3c1da7e5f6a1472853abcb4ed1c8aaf82e9eb684f59365fcf113781dca7',
                   '55ba69ad2771d64aaede263ac5387e990ae02c9ba8ab2eaa21f148e4d6481038')
    verify_payload(R17, '2a4208581e67e44f8941a43b696fd8ee141b2eec6533893862c1f0dfbe8c0345',
                   'd21fc53935ef20e7f9278039a9a4a26fea4f3248c3a75157401fb0d95c345203')
    for name in ('CONFIG_MATRIX.json', 'ROLE_POLICY.json'):
        check(name + ' v1 bytes preserved', (ROOT / name).read_bytes() == (V1 / name).read_bytes())
    for target, source in (('boundary_adapter.py', 'native_gat_adapter.py'), ('ad_qualifier.py', 'route_initializer.py')):
        check(target + ' v1 bytes preserved', (ROOT / target).read_bytes() == (V1 / source).read_bytes())
    lineage = load(ROOT / 'EXTRACTION_LINEAGE.json')
    groups = {
        'native_layers.py': {'get_activation_module', 'ResidualModule', 'FFNModule', '_check_d_and_n_heads_consistency', 'GATModule', 'BaseGraphBackbone', 'make_module'},
        'native_model.py': {'Model'},
        'native_transforms.py': {'Setting', 'NumPolicy', 'CatPolicy', 'impute_nans', 'transform_num_features', 'transform_cat_features'},
        'native_optimizer.py': {'make_parameter_groups'},
        'public_reference.py': {'load_dict'},
    }
    for item in lineage['segments']:
        source = Path(item['source'])
        original = ''.join(source.read_text().splitlines(keepends=True)[item['start'] - 1:item['end']])
        check('Pinned extraction segment hash ' + item['name'], hashlib.sha256(original.encode()).hexdigest() == item['source_segment_sha256'])
        for target, names in groups.items():
            if item['name'] in names:
                check('Exact source text ' + target + ':' + item['name'], segment(ROOT / target, definition(ROOT / target, item['name'])) == original)
    reference_adaptation_check()
    pinned_tree = {row['path']: row for row in load(FOLLOWUP / 'discovery/graphpfn_tree.json')['tree']}
    retrieved = load(ROOT / 'SOURCE_RETRIEVALS.json')
    for item in retrieved:
        data = (ROOT / item['path']).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        check('Retrieved pinned source identity ' + item['remote_path'], len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'] and blob == item['git_blob_sha1'] == pinned_tree[item['remote_path']]['sha'] and item['commit'] == COMMIT)
    native_originals = {
        'paper/lib/graph/deep.py': V1 / 'sources/paper__lib__graph__deep.py',
        'paper/lib/graph/data.py': V1 / 'sources/paper__lib__graph__data.py',
        'paper/lib/deep.py': V1 / 'sources/paper__lib__deep.py',
        'paper/bin/base/gnn.py': FOLLOWUP / 'author_source/graphpfn_paper_gnn.py',
        'paper/pyproject.toml': FOLLOWUP / 'author_source/graphpfn_paper_pyproject.toml',
    }
    for remote, path in native_originals.items():
        data = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        check('Native original pinned Git blob ' + remote, blob == pinned_tree[remote]['sha'])
    for item in load(ROOT / 'CANDIDATE_WHEEL_INDEX_RECEIPTS.json'):
        path = ROOT / item['path']
        check('Wheel-index metadata identity ' + path.name, path.stat().st_size == item['bytes'] and digest(path) == item['sha256'] and item['wheel_bytes_fetched'] is False)
        check('Candidate cp312 Linux wheel is in saved index ' + path.name,
              len(item['matched_cp312_linux_x86_64_wheels']) == 1 and item['matched_cp312_linux_x86_64_wheels'][0]
              in {urljoin(item['final_url'], html.unescape(href)) for href in re.findall(r'href="([^"]+)"', path.read_text())})
    torch_receipt = load(ROOT / 'TORCH_SOURCE_RECEIPT.json')
    check('Torch CUDA source receipt identity', digest(ROOT / torch_receipt['path']) == torch_receipt['sha256'] and (ROOT / torch_receipt['path']).stat().st_size == torch_receipt['bytes'])
    torch_source = (ROOT / torch_receipt['path']).read_text()
    check('Torch2.4 CUDA property UUID is not exposed in inspected source', '.def_readonly("uuid"' not in torch_source and '.def_property_readonly("uuid"' not in torch_source)
    lock_scope = load(ROOT / 'PUBLISHED_LOCK_SCOPE.json')
    check('Published lock metadata binding', digest(ROOT / 'sources/paper__uv.lock') == lock_scope['lock_descriptor']['sha256'])
    lock_text = (ROOT / 'sources/paper__uv.lock').read_text()
    lock_packages = {}
    for section in lock_text.split('[[package]]')[1:]:
        fields = {key: re.search(r'^' + key + r' = (.+)$', section, re.M).group(1) for key in ('name', 'version', 'source')}
        lock_packages[json.loads(fields['name'])] = fields
    for item in lock_scope['selected_packages']:
        actual = lock_packages[item['name']]
        check('Published lock package ' + item['name'], json.loads(actual['version']) == item['version'] and actual['source'] == item['source_literal'])
    check('Published Python pin disclosed', lock_scope['requires-python'] == '==3.12.9' and 'requires-python = "==3.12.9"' in lock_text)
    check('torchdata compatibility addition disclosed', 'torchdata' not in lock_packages)
    runtime = load(ROOT / 'CANDIDATE_RUNTIME.json')
    observed = load(BASE / 'industrial_environment_setup_root_v1/RUNTIME_IMPORT_CUDA_RECEIPT.json')
    check('Root import/CUDA capability is passed and scoped', observed['passed'] is True and observed['software_capability_only'] is True and observed['datasets_or_native_models_accessed'] is False and observed['qualification_of_backbone_or_preprocessing_claimed'] is False)
    check('Candidate selected versions match root capability', observed['versions'] == runtime['selected_top_versions'])
    check('Candidate cu118 policy differs from published cu124', runtime['selected_top_versions']['torch'] == '2.4.0+cu118' and runtime['selected_top_versions']['dgl'] == '2.4.0+cu118' and runtime['selected_top_versions']['torchdata'] == '0.8.0')
    top = next(node for node in tree(ROOT / 'runtime_fingerprint.py').body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TOP_REQUIREMENTS' for t in node.targets))
    check('Worker exact top-version policy', ast.literal_eval(top.value) == {name: '==' + version for name, version in runtime['selected_top_versions'].items()})
    matrix = load(ROOT / 'CONFIG_MATRIX.json')
    check('Frozen nine cells and order', matrix['fit_cells'] == [{'config': c, 'seed': s, 'cell_id': c + '__s' + str(s)} for c in ('c0', 'c1', 'c2') for s in (17, 29, 43)])
    check('Frozen fit/update cap', matrix['maximum_native_fits'] == 9 and matrix['maximum_native_updates'] == 9000)
    for config, lr, fraction in zip(matrix['configurations'], (3e-4, 1e-3, 3e-4), ('none', 'none', 'quantile-normal')):
        actual = config['native_config']
        check('Frozen configuration ' + config['id'], actual['optimizer']['lr'] == lr and actual['transform']['features'] == {'seed': 0, 'cat_policy': 'one-hot', 'num_policy': 'quantile-normal', 'frac_policy': fraction} and actual['n_steps'] == 1000 and actual['patience'] == 100 and actual['amp_dtype'] is None)
    scientific_suffixes = {'.pt', '.pth', '.npy', '.npz', '.zip', '.csv', '.whl', '.so', '.pyc'}
    for path in ROOT.rglob('*'):
        check('Source-only filesystem entry ' + path.relative_to(ROOT).as_posix(), not path.is_symlink() and (path.is_dir() or path.is_file()) and path.suffix not in scientific_suffixes)
    local_modules = {path.stem for path in ROOT.glob('*.py')}
    package_names = {'torch': 'torch', 'dgl': 'dgl', 'numpy': 'numpy', 'scipy': 'scipy', 'sklearn': 'scikit-learn', 'delu': 'delu', 'pandas': 'pandas', 'yaml': 'PyYAML', 'packaging': 'packaging'}
    standard = set(getattr(sys, 'stdlib_module_names', ()))
    standard |= {'__future__', 'argparse', 'ast', 'collections', 'copy', 'csv', 'ctypes', 'dataclasses', 'datetime', 'enum', 'hashlib', 'html', 'importlib', 'io', 'itertools', 'json', 'os', 'pathlib', 'platform', 're', 'resource', 'shutil', 'socket', 'subprocess', 'sys', 'time', 'traceback', 'types', 'typing', 'urllib', 'zipfile'}
    for path in sorted(ROOT.glob('*.py')):
        parsed = tree(path)
        check('AST parses ' + path.name, isinstance(parsed, ast.Module))
        imports = set()
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module.split('.')[0])
        check('Declared direct import closure ' + path.name, imports <= standard | local_modules | set(package_names), sorted(imports))
        if path.name not in ('ad_qualifier.py', 'check_sources.py'):
            check('No initializer/continuation calls ' + path.name, not any(name in {'initialize_four_routes', 'bernstein_cubic_bands', 'permute_topology_nodes'} for name in calls(parsed)))
        if path.name not in ('custodian_export.py', 'check_sources.py'):
            check('No raw provider/archive calls ' + path.name, not any(name.split('.')[-1] in {'GraphDataset', 'GraphTask', '_load_graphland_data', '_mask_labeled_nodes', 'ZipFile'} for name in calls(parsed)))
    for name in ('namespace_bootstrap.py', 'launch_isolated.py', 'run_native.py'):
        parsed = tree(ROOT / name)
        choices = [ast.literal_eval(keyword.value) for node in ast.walk(parsed) if isinstance(node, ast.Call) and ast.unparse(node.func) == 'parser.add_argument' for keyword in node.keywords if keyword.arg == 'choices' and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 'mode']
        check('Closed CLI modes ' + name, choices == [('isolation-probe', 'qualify', 'run-nine')])
    prep = definition(ROOT / 'public_preprocessor.py', 'load_public')
    dgl_calls = [ast.unparse(node.value.func) for node in prep.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func).startswith('dgl.')]
    check('Public graph construction exact operation order', dgl_calls == ['dgl.graph', 'dgl.remove_self_loop', 'dgl.to_simple', 'dgl.to_bidirected'])
    for filename in ('native_layers.py', 'torch_coo.py', 'boundary_adapter.py'):
        text = (ROOT / filename).read_text()
        check('Native head-packing contract ' + filename, 'self.d_head, self.n_heads' in text or 'self.d_head,self.n_heads' in text)
    run_tree = definition(ROOT / 'run_native.py', 'main')
    import_lines = [node.lineno for node in ast.walk(run_tree) if isinstance(node, (ast.Import, ast.ImportFrom)) and any(alias.name == 'torch' for alias in node.names)]
    gate_lines = [node.lineno for node in ast.walk(run_tree) if isinstance(node, ast.Call) and ast.unparse(node.func) in {'observe_namespace', 'verify_source_packet', 'verify_runtime_image'}]
    check('OS/source/image observation precedes Torch import', max(gate_lines) < min(import_lines))
    bootstrap = (ROOT / 'namespace_bootstrap.py').read_text()
    check('Namespace creation is concrete and has no fallback', all(flag in (ROOT / 'launch_isolated.py').read_text() for flag in ('--user', '--map-root-user', '--mount', '--pid', '--fork', '--net', '--ipc', '--uts', '--kill-child')))
    check('Chroot precedes capability removal and scientific exec', bootstrap.index('os.chroot(root)') < bootstrap.index('    drop_capabilities()') < bootstrap.index('    os.execve('))
    check('Bootstrap clears environment/inherited descriptors', 'os.closerange(3, 65536)' in bootstrap and 'environment = {' in bootstrap and 'CUBLAS_WORKSPACE_CONFIG' in bootstrap)
    custodian = definition(ROOT / 'custodian_export.py', 'source_labels')
    probe_check = min(node.lineno for node in ast.walk(custodian) if isinstance(node, ast.Call) and ast.unparse(node.func) == 'verify_source_packet')
    target_open = min(node.lineno for node in ast.walk(custodian) if isinstance(node, ast.Call) and ast.unparse(node.func) == 'zip_file.open')
    check('Actual public-only probe binding precedes source target open', probe_check < target_open)
    native_text = (ROOT / 'native_fit.py').read_text()
    check('Bounded native driver has no extension family', all(token not in native_text for token in ('build_boundary_family', 'initialize_four_routes', 'transport_adamw')))
    check('Earliest strict best/native AP/fixed warm50 retained', all(token in native_text for token in ('ap > best_ap', 'if step == 50:', 'scipy.special.softmax(raw, axis=-1)[val_rows, 1]', "'warm50_selector': 'fixed_last'")))
    check('Selection/competence remains frozen', all(token in native_text for token in ('if scores[name] > scores[selected]', 'all(uplift > 0 for uplift in uplifts)', 'sum(uplifts) / 3 >= 0.05', 'np.mean(np.stack(probabilities), axis=0)')))
    check('Failure costs retained before dependent stop', "'NOT_STARTED_DEPENDENCY_FAILED'" in native_text and 'paid_update_progress' in native_text and 'completed_optimizer_updates' in native_text)
    qualifier = (ROOT / 'qualify_native.py').read_text()
    check('Disposable full-config/K1/K4/native-Torch/AD qualification registered', all(token in qualifier for token in ("for row in matrix['configurations']:", 'for members in (1, 4)', 'native_torch_parity(', 'qualify_gradient_interface(', "'scheduled_disposable_optimizer_steps': 7", "'useful_fit_updates': 0")))
    if (ROOT / 'SEAL.json').exists():
        seal = load(ROOT / 'SEAL.json')
        manifest = load(ROOT / 'MANIFEST.json')
        check('V2 seal binds actual manifest', digest(ROOT / 'MANIFEST.json') == seal['manifest_sha256'])
        declared = {item['path'] for item in manifest['payload']}
        actual = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob('*') if path.is_file()}
        check('V2 sealed file allowlist', actual == declared | {'MANIFEST.json', 'SEAL.json'})
        for item in manifest['payload']:
            path = ROOT / item['path']
            check('V2 sealed payload ' + item['path'], path.stat().st_size == item['bytes'] and digest(path) == item['sha256'])
        check('V2 seal source-only claims', seal['scientific_import_or_execution_by_author'] is False and seal['all_test_labels'] == 'CLOSED')
    receipt = {'schema': 'tolokers2-static-source-check-receipt-v2', 'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'passed': True, 'checks_passed': len(CHECKS), 'checks': CHECKS,
               'scope': 'Standard-library source text, AST, hashes and JSON metadata only; no scientific imports, compilation, archive/dataset/model/GPU execution, SSH or installation by this author.',
               'runtime_numerical_or_isolation_qualification': False}
    if write_receipt:
        if (ROOT / 'SEAL.json').exists():
            raise RuntimeError('A sealed packet cannot be changed by the checker')
        (ROOT / 'STATIC_SOURCE_CHECKS.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'passed': True, 'checks_passed': len(CHECKS), 'sealed': (ROOT / 'SEAL.json').exists(), 'source_only': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-receipt', action='store_true')
    main(parser.parse_args().write_receipt)
