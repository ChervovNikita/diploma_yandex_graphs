"""Standard-library AST/JSON/hash checks only; no integration/provider import."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(root, values):
    require(len({value['path'] for value in values}) == len(values), 'Unique manifest paths')
    for value in values:
        require(not Path(value['path']).is_absolute(), 'Relative source path')
        path = (root / value['path']).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file() and path.stat().st_size == value['bytes']
                and sha(path) == value['sha256'], 'Exact frozen source bytes: ' + value['path'])


def check(unsealed=False):
    bindings = read(HERE / 'SOURCE_BINDINGS.json')
    rows(HERE.parent, bindings['files'])
    v2 = HERE.parent / bindings['joint_v2_packet']
    require(sha(v2 / 'SEAL.json') == bindings['joint_v2_seal_sha256']
            == 'c3fd927d4a7e7906fed60a389bab2c576ce31489d5b95a0a7e45800ff4643e05', 'Immutable V2 seal preserved')
    require(sha(v2 / 'MANIFEST.json') == read(v2 / 'SEAL.json')['manifest_sha256'], 'Original V2 manifest binding')
    rows(v2, read(v2 / 'MANIFEST.json')['files'])
    v1 = HERE.parent / 'joint_source_additive_sehgnn_training_source_20261009_v1'
    require(sha(v1 / 'SEAL.json') == 'b147739a9af48afbdba32cdf706a2b91373a1daebc37724feefb3c91fb333656'
            and sha(v1 / 'MANIFEST.json') == read(v1 / 'SEAL.json')['manifest_sha256'], 'Immutable V1 seal preserved')
    rows(v1, read(v1 / 'MANIFEST.json')['files'])
    require(sha(v2 / 'joint_training.py') == bindings['joint_training_sha256'], 'Exact injected V2 callable source')

    allowed = {'argparse', 'ast', 'dataclasses', 'gc', 'hashlib', 'json', 'pathlib', 'resource', 'sys', 'time', 'factual_contracts'}
    trees, python_files = {}, []
    for path in sorted(HERE.glob('*.py')):
        tree = ast.parse(path.read_text(), filename=str(path))
        trees[path.name] = tree
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                require(all(alias.name.split('.')[0] in allowed for alias in node.names), 'Only stdlib/local imports')
            elif isinstance(node, ast.ImportFrom):
                require(node.module and node.module.split('.')[0] in allowed, 'Only stdlib/local imports')
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                require(node.func.id not in {'exec', 'eval', '__import__'}, 'No dynamic source execution')
        python_files.append(dict(path=path.name, sha256=sha(path), lines=len(path.read_text().splitlines())))
    require(set(trees) == {'factual_contracts.py', 'factual_training.py', 'verify_static.py'}, 'Exact callable/checker files, no launcher')
    config = next(node for node in trees['factual_contracts.py'].body if isinstance(node, ast.ClassDef) and node.name == 'Config')
    defaults = {node.target.id: ast.literal_eval(node.value) for node in config.body
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
    flags = ('enabled', 'root_source_review_approved', 'root_qualification_execution_approved', 'root_training_approved')
    require(all(defaults[name] is False for name in flags) and defaults['intent'] == 'disabled', 'Actual config defaults disabled')
    require(all(defaults[name] == () for name in ('body_seeds', 'adapter_seeds', 'member_rng_seeds')), 'No runtime seed release supplied')
    require(all(defaults[name] is None for name in ('source_seal_sha256', 'joint_v2_source_review_binding',
        'native_qualification_binding', 'new_training_qualification_binding', 'prospective_quality_freeze_binding',
        'study_binding', 'role_binding', 'full_view_binding')), 'Qualification/freeze/adoption bindings pending')
    tree = trees['factual_training.py']
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    require({'prepare_factual_context', 'make_condition', 'factual_native_single_P', 'factual_independent_native4_P', 'fit_factual_condition'} <= names, 'Two callable native controls and factual setup/fit entries')
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
             for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))}
    require(not (calls & {'build_family_views', 'build_one', 'install_member_bank', 'shared_additive', 'install',
                         'grad', 'collect_selected_logits', 'analyze', 'make_credit_plan', 'try_private_step'}), 'No source views/adapters/source credit/diagnostic machinery invoked')
    require({'make_model', 'scratch_native_state', '_copy_buffers', 'backward', 'step', 'evaluate', 'snapshot', 'restore'} <= calls, 'Native live training and inherited factual state lifecycle present')
    backwards = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == 'backward']
    require(len(backwards) == 1 and ast.unparse(backwards[0].func.value) == 'own'
            and not backwards[0].args, 'Literal coefficient1 body BCE backward, no /4')
    require('fresh = make_condition(rt, modules, ctx, costs, config)' in ast.unparse(tree), 'Fresh factual reconstruction uses its own factory')
    require('super().forward' in ast.unparse(tree) and "token.source is None" in ast.unparse(tree), 'Inherited exact role guard with factual-only public forward')

    json_names = sorted(path.name for path in HERE.glob('*.json') if path.name not in {'MANIFEST.json', 'SEAL.json', 'STATIC_VERIFICATION.json'})
    docs = {name: read(HERE / name) for name in json_names}
    release, protocol, counts = (docs[name] for name in ('RELEASE.disabled.json', 'PROTOCOL.json', 'SCALAR_COUNTS.json'))
    require(all(release[name] is False for name in flags) and release['intent'] == 'disabled', 'Release disabled')
    require(all(protocol[name] is False for name in ('enabled', 'runtime_authorized', 'qualification_execution_authorized', 'scientific_fit_authorized')), 'Protocol source-only')
    require(set(protocol['conditions']) == {'factual_native_single_P', 'factual_independent_native4_P'}
            and protocol['combined_conditions_per_pair'] == 12 and protocol['all_original_ten_joint_conditions_preserved'], 'Two additions preserve original ten-condition family')
    require(counts['single_scalars'] == counts['native_scalars_per_body'] == 83659532
            and counts['independent4_scalars'] == 4 * counts['native_scalars_per_body'] == 334638128
            and counts['adapter_or_private_factor_scalars'] == 0, 'Symbolic full native budgets')
    report = dict(schema='factual-native-source-static-verification-v1', checks_passed=True,
        verification_kind='AST/JSON/default-disabled/source hashes/symbolic integer counts only',
        checked_python_files=python_files, checked_payload_JSON_files=json_names,
        bound_source_files=len(bindings['files']), immutable_V1_V2_payloads_and_seals_preserved=True,
        two_factual_control_entries_and_coefficient1_body_backward_present=True,
        source_view_adapter_source_credit_calls_absent=True, numeric_imports_or_execution=False,
        actual_BN_gradients_updates_restoration_resources_or_quality_verified=False)
    if not unsealed:
        seal, manifest = read(HERE / 'SEAL.json'), read(HERE / 'MANIFEST.json')
        require(seal['runtime_disabled'] is True and seal['qualification_execution_authorized'] is False
                and seal['scientific_fit_authorized'] is False and sha(HERE / 'MANIFEST.json') == seal['manifest_sha256'], 'Final disabled seal')
        rows(HERE, manifest['files'])
        require({path.name for path in HERE.iterdir() if path.is_file() and path.name not in {'MANIFEST.json', 'SEAL.json'}}
                == {value['path'] for value in manifest['files']}, 'Complete extension manifest')
        require(read(HERE / 'STATIC_VERIFICATION.json') == report, 'Saved deterministic source-only report')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unsealed', action='store_true')
    parser.add_argument('--write-report', action='store_true')
    arguments = parser.parse_args()
    report = check(arguments.unsealed)
    if arguments.write_report:
        (HERE / 'STATIC_VERIFICATION.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(dict(checks_passed=True, source_only=True, sealed_payload_checked=not arguments.unsealed,
        checked_python_files=len(report['checked_python_files']), bound_source_files=report['bound_source_files']), sort_keys=True))


if __name__ == '__main__':
    main()
