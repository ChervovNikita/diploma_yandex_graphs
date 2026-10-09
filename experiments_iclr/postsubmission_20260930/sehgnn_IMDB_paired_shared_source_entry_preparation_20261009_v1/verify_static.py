"""Stdlib AST/hash/default-disabled check; never execute the shared branch."""
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
            assert name not in {'train', 'evaluate', 'evaluator', 'fit_body', 'fit', 'run_independent_family',
                'compare_independent_controls', 'compare_selected', 'Popen', 'kill', 'killpg', 'sleep'}
    assert calls.count('run_shared_family') == calls.count('runtime') == 1
    old = ast.parse((HERE.parent / 'sehgnn_IMDB_paired_independent_reference_entry_preparation_20261009_v1' / 'RUN.py').read_text())
    old_functions = {node.name: node for node in old.body if isinstance(node, ast.FunctionDef)}
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ('sha', 'read', 'load', 'write', 'digest', 'bound_path', 'source_gate'):
        assert ast.dump(functions[name], include_attributes=False) == ast.dump(old_functions[name], include_attributes=False)
    release, adoption = read(HERE / 'RELEASE.disabled.json'), read(HERE / 'COMPETENCE_ADOPTION.disabled.json')
    assert release['enabled'] is release['root_source_review_approved'] is release['root_shared_fit_release_approved'] is False
    assert release['native_and_independent_reference_competence_adopted'] is False
    assert adoption['root_authorized'] is adoption['accuracy_reference_assessment_complete'] is adoption['native_and_independent_reference_competence_adopted'] is False
    assert release['study_binding'] == 'd0b30e8297bbedc6063a0bf0c553869157165759d8cd2a41e93e4fe4324d0ea4'
    assert release['actual_shared_fits'] == 18 and release['pairs'] == [[1,1],[2,2],[3,3]] and len(release['arms']) == 6
    assert release['automatic_retry'] is release['comparative_outcome_publishing'] is False
    for key in ('reference_family_receipt', 'native_reference_receipt', 'competence_adoption_receipt'):
        assert release[key] == {'path':None,'bytes':None,'sha256':None}
    assert "reference['completed_actual_body_fits'] == 24" in text and "native_reference['native_five_seed_recipe_completed'] is True" in text
    assert "adoption['accuracy_reference_assessment_complete'] is True" in text
    assert text.index("record['native_and_independent_reference_competence_adopted'] = True") > text.index("assert adoption['root_authorized'] is True")
    for path in HERE.glob('*.json'):
        read(path)
    default = subprocess.run([sys.executable, '-B', str(HERE / 'RUN.py')], check=True, capture_output=True, text=True)
    assert json.loads(default.stdout) == {'inactive':True,'shared_fits_or_provider_imports':False} and not default.stderr
    return dict(status='passed', source_AST_hash_and_disabled_default_only=True, reused_bootstrap_helpers_AST_identical=True,
        existing_sealed_shared_family_called_once=True, fixed_three_pairs_six_arms_18_fits=True,
        same_frozen_pilot_protocol_and_study=True, default_competence_false_and_actual_reference_plus_adoption_gate=True,
        no_new_fitter_monitor_diagnostics_retry_or_tiny_output_gate=True,
        provider_GPU_server_data_qualification_reference_outcomes_or_scores_read=False, actual_shared_fits_or_scientific_admission=False)


if __name__ == '__main__':
    result = check()
    target = HERE / 'STATIC_VERIFICATION.json'
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))
