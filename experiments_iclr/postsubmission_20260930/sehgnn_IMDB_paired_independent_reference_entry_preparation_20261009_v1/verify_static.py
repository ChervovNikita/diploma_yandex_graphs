"""Stdlib AST/hash and disabled-default check; never execute the reference branch."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    read = lambda path: json.loads(path.read_text())
    for row in read(HERE / 'SOURCE_BINDINGS.json')['files']:
        path = (HERE.parent / row['path']).resolve(strict=True)
        assert path.is_relative_to(HERE.parent) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    text = (HERE / 'RUN.py').read_text()
    tree = ast.parse(text)
    providers = {'torch', 'numpy', 'dgl', 'torch_sparse', 'sklearn'}
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(alias.name.split('.')[0] in providers for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or '').split('.')[0] not in providers
        elif isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
            calls.append(name)
            assert name not in {'train', 'evaluate', 'evaluator', 'fit_body', 'fit', 'run_shared_family',
                'compare_independent_controls', 'compare_selected', 'Popen', 'kill', 'killpg', 'sleep'}
    assert calls.count('run_independent_family') == calls.count('runtime') == 1
    assert "cfg['native_and_independent_reference_competence_adopted'] is False" in text
    constructor = next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'PilotConfig')
    assert next(item.value.value for item in constructor.keywords if item.arg == 'native_and_independent_reference_competence_adopted') is False
    old = ast.parse((HERE.parent / 'sehgnn_independent4_full_input_qualification_activation_root_20261009_v1' / 'RUN.py').read_text())
    old_functions = {node.name: node for node in old.body if isinstance(node, ast.FunctionDef)}
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ('sha', 'read', 'load'):
        assert ast.dump(functions[name], include_attributes=False) == ast.dump(old_functions[name], include_attributes=False)
    release = read(HERE / 'RELEASE.disabled.json')
    assert release['enabled'] is release['root_source_review_approved'] is release['root_reference_fit_release_approved'] is False
    assert release['native_and_independent_reference_competence_adopted'] is False
    assert release['pairs'] == [[1,1],[2,2],[3,3]] and release['variants'] == ['plain_native','untied_same_six_factors']
    assert release['actual_body_fits'] == 24 and release['automatic_retry'] is release['comparative_outcome_publishing'] is False
    for key in ('native_qualification_receipt', 'integration_qualification_receipt', 'independent4_qualification_receipt'):
        assert release[key] == {'path':None,'bytes':None,'sha256':None}
    for path in HERE.glob('*.json'):
        read(path)
    result = subprocess.run([sys.executable, '-B', str(HERE / 'RUN.py')], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == {'inactive':True,'reference_fits_or_provider_imports':False} and not result.stderr
    return dict(status='passed', source_AST_hash_and_disabled_default_only=True, reused_bootstrap_helpers_AST_identical=True,
        existing_sealed_family_called_once=True, fixed_six_groups_24_real_body_fit_declaration=True,
        competence_adoption_false_and_pending=True, no_new_fitter_monitor_diagnostics_retry_or_comparative_publication=True,
        provider_GPU_server_data_scores_or_actual_qualification_receipts_read=False, actual_reference_or_resource_qualification=False)


if __name__ == '__main__':
    result = check()
    target = HERE / 'STATIC_VERIFICATION.json'
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))
