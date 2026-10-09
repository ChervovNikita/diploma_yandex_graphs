"""AST/JSON/source-byte verification only; never import the training modules."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_KEYS = ('engine', 'adapter', 'metadata', 'helper', 'role_loader', 'source_views', 'diagnostics')
CONDITIONS = {
    'BE_P': ('shared', 'be', 1, 'none', False),
    'BE_PS': ('shared', 'be', 1, 'assigned', False),
    'ADD_P': ('shared', 'add', 1, 'none', False),
    'ADD_PS': ('shared', 'add', 1, 'assigned', False),
    'ADD_P_copied_dictionary': ('shared', 'add', 1, 'none', True),
    'ADD_PS_neutral_all_source_average': ('shared', 'add', 1, 'neutral', False),
    'independent_ADD_P': ('independent', 'add', 1, 'none', False),
    'independent_ADD_PS': ('independent', 'add', 1, 'assigned', False),
    'native_single_P': ('single', 'native', 0, 'none', False),
    'rank4_single_P': ('single', 'add', 4, 'none', False),
}


def require(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_rows(root, rows):
    require(len({row['path'] for row in rows}) == len(rows), 'Unique custody paths')
    for row in rows:
        require(not Path(row['path']).is_absolute(), 'Relative custody path')
        path = (root / row['path']).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file()
                and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Exact source bytes: ' + row['path'])


def assignment(tree, name):
    node = next(node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
    return ast.literal_eval(node.value)


def check(unsealed=False):
    sources = read(HERE / 'SOURCE_BINDINGS.json')
    check_rows(HERE.parent, sources['files'])
    require(set(sources['modules']) == set(MODULE_KEYS) | {'model', 'native_helpers', 'state_helpers'}, 'Exact ten code-source identities')
    require(tuple(sources['injected_module_keys']) == MODULE_KEYS, 'Exact seven injected modules')
    for key, row in sources['modules'].items():
        require(row in sources['files'], 'Every module bound in the source-file list')
        path = HERE.parent / row['path']
        old_manifest = read(path.parent / 'MANIFEST.json')
        old_row = next(item for item in old_manifest['files'] if item['path'] == path.name)
        require((old_row['bytes'], old_row['sha256']) == (row['bytes'], row['sha256']), 'Preserved original sealed module: ' + key)
    require(sources['source_view_seal_sha256'] == sha(HERE.parent / 'sehgnn_literal_raw_family_source_views_source_20261009_v1/SEAL.json'), 'Exact source-view seal')
    require(sources['modules']['model']['sha256'] == '0948239c4c4d06dd258cd106fd2fa7b6aaa2a62688fe70dd662413fa1eae532a', 'Pinned native model')
    require(sources['native_author_commit'] == 'e92bd37d0b803457339555684f139b4c8f3e160d', 'Pinned native commit')

    allowed_imports = {'argparse', 'ast', 'copy', 'dataclasses', 'gc', 'hashlib', 'json', 'pathlib',
                       'resource', 'sys', 'time', 'joint_contracts', 'joint_additive_adapter'}
    trees, python_rows = {}, []
    for path in sorted(HERE.glob('*.py')):
        tree = ast.parse(path.read_text(), filename=str(path))
        trees[path.name] = tree
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                require(all(alias.name.split('.')[0] in allowed_imports for alias in node.names), 'Only stdlib/local source imports')
            elif isinstance(node, ast.ImportFrom):
                require(node.module and node.module.split('.')[0] in allowed_imports, 'Only stdlib/local source imports')
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                require(node.func.id not in {'exec', 'eval', '__import__'}, 'No dynamic execution in source')
        python_rows.append(dict(path=path.name, sha256=sha(path), lines=len(path.read_text().splitlines())))
    require(set(trees) == {'joint_contracts.py', 'joint_additive_adapter.py', 'joint_training.py', 'verify_static.py'}, 'Exact source Python files; no launcher/fixture')

    contract = trees['joint_contracts.py']
    config = next(node for node in contract.body if isinstance(node, ast.ClassDef) and node.name == 'Config')
    defaults = {node.target.id: ast.literal_eval(node.value) for node in config.body
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
    flags = ('enabled', 'root_source_review_approved', 'root_qualification_execution_approved', 'root_training_approved')
    require(all(defaults[name] is False for name in flags) and defaults['intent'] == 'disabled', 'Actual callable defaults disabled')
    require(all(defaults[name] is None for name in ('source_seal_sha256', 'native_qualification_binding',
        'new_training_qualification_binding', 'prospective_quality_freeze_binding', 'study_binding', 'role_binding', 'full_view_binding')), 'No adopted release bindings')
    require(all(defaults[name] == () for name in ('body_seeds', 'adapter_seeds', 'member_rng_seeds')), 'No execution seed list supplied')
    require(assignment(contract, 'SPECS') == CONDITIONS and assignment(contract, 'MODULE_KEYS') == MODULE_KEYS, 'Exact ten condition specs and dependency keys')

    training = trees['joint_training.py']
    constructors = {node.name: node for node in training.body if isinstance(node, ast.FunctionDef) and node.name in CONDITIONS}
    require(set(constructors) == set(CONDITIONS), 'All ten named condition constructors')
    for name, node in constructors.items():
        default = node.args.defaults[-1]
        require(isinstance(default, ast.Call) and isinstance(default.func, ast.Name) and default.func.id == 'Config'
                and len(default.keywords) == 1 and default.keywords[0].arg == 'condition'
                and ast.literal_eval(default.keywords[0].value) == name, 'Named constructor keeps default-disabled Config')
    forbidden_calls = {'make_credit_plan', 'accumulate_replay', 'project_nonincrease_cone', 'try_private_step',
                       'build_private_direction', 'check_guard_values', 'correct', 'ScheduledSession'}
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
             for node in ast.walk(training) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))}
    require(not (calls & forbidden_calls), 'No old signed/guard/dose/scheduler machinery invoked')
    require({'collect_selected_logits', 'analyze', 'serve_full_input', 'scratch_native_state'} <= calls, 'Original readout/state authority reused')
    dtype_returns = [node for node in ast.walk(trees['joint_additive_adapter.py']) if isinstance(node, ast.Return)
                     and node.value and ast.unparse(node.value) == 'value + correction.to(dtype=value.dtype)']
    require(len(dtype_returns) == 2, 'Both additive boundaries preserve native output dtype')

    json_names = sorted(path.name for path in HERE.glob('*.json') if path.name not in {'MANIFEST.json', 'SEAL.json', 'STATIC_VERIFICATION.json'})
    documents = {name: read(HERE / name) for name in json_names}
    release, protocol = documents['RELEASE.disabled.json'], documents['PROTOCOL.json']
    require(all(release[name] is False for name in flags) and release['intent'] == 'disabled', 'Release template disabled')
    require(all(protocol[name] is False for name in ('enabled', 'runtime_authorized', 'qualification_execution_authorized', 'scientific_fit_authorized', 'prospective_quality_resource_custody_freeze')), 'Protocol is not an execution/freeze release')
    expected_specs = {name: dict(ownership=spec[0], parameterization=spec[1], rank=spec[2], credit=spec[3], copied_dictionary=spec[4]) for name, spec in CONDITIONS.items()}
    require(protocol['conditions'] == expected_specs and protocol['pairs'] == [[1, 1], [2, 2], [3, 3]], 'Original declared family retained')
    counts = documents['SCALAR_COUNTS.json']
    native = assignment(contract, 'NATIVE_SCALARS')
    fast = assignment(contract, 'FAST_RANK1_PER_MEMBER')
    require(native == counts['native_body_scalars'] == 83659532 and fast == 97536, 'Native/private symbolic constants')
    require(sum(site['groups'] * (site['inputs'] + site['outputs']) for site in counts['sites']) == fast, 'Symbolic six-site rank1 scalar accounting')
    for name, spec in CONDITIONS.items():
        bodies = 4 if spec[0] == 'independent' else 1
        paths = 1 if spec[0] == 'single' else 4
        require(counts['conditions'][name]['total_owned_scalars'] == native * bodies + fast * spec[2] * paths, 'Symbolic condition count: ' + name)
    scopes = documents['READ_SCOPES.json']
    require(scopes['new_packet_only_writes'] is True and scopes['subagents_used'] is False
            and scopes['GPU_or_numeric_execution'] is False, 'Source-only recorded scope')

    report = dict(schema='joint-proper-source-static-verification-v1', checks_passed=True,
        verification_kind='AST/JSON/source hashes/default-disabled/symbolic integer counts only',
        checked_python_files=python_rows, checked_payload_JSON_files=json_names,
        frozen_dependency_files=len(sources['files']), preserved_sealed_module_files=len(sources['modules']),
        all_ten_disabled_constructor_specs=True, both_native_output_dtype_casts_present=True,
        numerical_or_model_imports=False, runtime_or_qualification_or_fit_execution=False,
        actual_gradient_replay_restore_quality_or_resources_verified=False)
    if not unsealed:
        seal, manifest = read(HERE / 'SEAL.json'), read(HERE / 'MANIFEST.json')
        require(seal['source_only'] is True and seal['runtime_disabled'] is True
                and seal['qualification_execution_authorized'] is False and seal['scientific_fit_authorized'] is False, 'Final inactive seal')
        require(sha(HERE / 'MANIFEST.json') == seal['manifest_sha256'], 'Final manifest binding')
        check_rows(HERE, manifest['files'])
        actual = {str(path.relative_to(HERE)) for path in HERE.rglob('*') if path.is_file()
                  and path.name not in {'MANIFEST.json', 'SEAL.json'}}
        require(actual == {row['path'] for row in manifest['files']}, 'Complete source payload manifest')
        require(read(HERE / 'STATIC_VERIFICATION.json') == report, 'Saved deterministic source-only report')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unsealed', action='store_true', help='Source build phase only, before manifest/seal creation')
    parser.add_argument('--write-report', action='store_true')
    arguments = parser.parse_args()
    report = check(arguments.unsealed)
    if arguments.write_report:
        (HERE / 'STATIC_VERIFICATION.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(dict(checks_passed=True, source_only=True, sealed_payload_checked=not arguments.unsealed,
        checked_python_files=len(report['checked_python_files']), bound_dependency_files=report['frozen_dependency_files']), sort_keys=True))


if __name__ == '__main__':
    main()
