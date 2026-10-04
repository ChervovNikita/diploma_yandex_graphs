"""Source-only diagnostic custody/AST checks; never import or execute targets."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
CANDIDATE = PHASE / 'pubmed_shared4_owned_continuation_diagnostic_source_20261004_v3'
V2 = PHASE / 'pubmed_shared4_owned_continuation_diagnostic_source_20261004_v2'
V1 = PHASE / 'pubmed_shared4_owned_continuation_diagnostic_source_20261004_v1'
EXPECTED = '93463ef2548914fa80e922959c83f7664ae09f94ec684ec51d801560dbac230a'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(root, declarations):
    result = []
    for row in declarations:
        path = root / row['path']
        assert path.suffix not in ('.pt', '.npy', '.npz', '.pkl'), path
        assert path.is_file() and not path.is_symlink(), path
        actual = {'path': str(path.relative_to(PHASE)), 'sha256': sha(path),
                  'bytes': path.stat().st_size}
        assert actual['sha256'] == row['sha256'] and actual['bytes'] == row['bytes'], path
        result.append({**actual, 'match': True})
    return result


def functions(path):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


def main():
    assert sha(CANDIDATE / 'MANIFEST.json') == EXPECTED
    manifest = read(CANDIDATE / 'MANIFEST.json')
    binding = read(CANDIDATE / 'SOURCE_BINDING.json')
    closure = read(CANDIDATE / 'METADATA_CLOSURE.json')
    plan = read(CANDIDATE / 'PLAN.json')
    candidate_rows = rows(CANDIDATE, manifest['files'])
    external_rows = rows(PHASE, binding['external_source_pins'])
    predecessor_rows = rows(V2, read(V2 / 'MANIFEST.json')['files'])
    prior_review = PHASE / 'pubmed_shared4_owned_continuation_diagnostic_fresh_review_20261004_v1'
    prior_review_rows = rows(prior_review, read(prior_review / 'MANIFEST.json')['files'])
    assert sha(prior_review / 'REVIEW.json') == '24177daf0474234ef30fc47b55b08855637e07a5d20d8fd02152fc6e7ae7fb93'
    assert read(prior_review / 'REVIEW.json')['status'] == 'PASS'
    assert closure['external_source_files'] == binding['external_source_pins']
    assert closure['owned_failed_attempt_metadata'] == binding['owned_failure_metadata']
    assert closure['scalar_prerequisite_metadata'] == binding['prerequisite_receipts']
    assert closure['additional_required_metadata'] == plan['input_authority']['feature_equivalence_receipt']
    prior_plan = read(V1 / 'PLAN.json')
    equal_profile = {key: plan[key] == prior_plan[key] for key in ('execution_profile', 'engineering_tolerance', 'input_authority')}
    assert all(equal_profile.values())
    assert plan['stages']['continuation_diagnostic']['caps'] == prior_plan['stages']['continuation_diagnostic']['caps']
    changed = [row['path'] for row in read(V2 / 'MANIFEST.json')['files']
               if (CANDIDATE / row['path']).read_bytes() != (V2 / row['path']).read_bytes()]
    assert changed == ['SOURCE_CHECK.json']
    old = read(V2 / 'SOURCE_CHECK.json')
    new = read(CANDIDATE / 'SOURCE_CHECK.json')
    delta = {key: {'v2': old.get(key, 'ABSENT'), 'v3': new.get(key, 'ABSENT')}
             for key in sorted(old.keys() | new.keys()) if old.get(key, 'ABSENT') != new.get(key, 'ABSENT')}
    assert set(delta) == {'UTC', 'all_comparison_collection_happens_after_both_native_epochs',
                          'original_fixed_comparator_and_final_endstate_comparisons_happen_after_both_native_epochs',
                          'prestate_saved_tree_alias_and_per_step_observations_are_collected_at_declared_boundaries'}
    common, old_common = functions(CANDIDATE / 'common.py'), functions(V1 / 'common.py')
    common_reuse = {key: common[key] == value for key, value in old_common.items() if key != 'gate'}
    assert all(common_reuse.values())
    diagnostic, old_diagnostic = functions(CANDIDATE / 'continuation_diagnostic.py'), functions(V1 / 'continuation_diagnostic.py')
    predicate_reuse = {key: diagnostic[key] == old_diagnostic[key] for key in ('exact', 'differences', 'step_aliases')}
    assert all(predicate_reuse.values())
    comparator = functions(PHASE / plan['comparator_source_path'])['compare']
    comparator_sha = hashlib.sha256(comparator.encode()).hexdigest()
    assert comparator_sha == '608c9d0030b2b2393608a7438f3c25653d33bab909e8700daee16d27e16871c9'
    python_rows = []
    for name in ('common.py', 'continuation_diagnostic.py', 'step_observer.py', 'supervise.py'):
        ast.parse((CANDIDATE / name).read_text())
        assert (CANDIDATE / name).read_bytes() == (V2 / name).read_bytes()
        python_rows.append({'path': name, 'AST_parse': True, 'byte_exact_v2': True})
    assert (CANDIDATE / 'supervise.py').read_bytes() == (V1 / 'supervise.py').read_bytes()
    observer = PHASE / plan['observer_source_path']
    assert observer.read_bytes() == (PHASE / 'pubmed_shared4_bridge_qualification_source_20261004_v2/native_observer.py').read_bytes()
    author = PHASE / 'pubmed_heart_available_inspector_native_adapter_20261004_v1'
    snapshot_rows = rows(author / 'public_author_code/HeaRT', read(author / 'PINNED_AUTHOR_SOURCE_AND_FUNCTIONS.json')['snapshot_files'])
    observer_tree = ast.parse((CANDIDATE / 'step_observer.py').read_text())
    hooks = [node for node in ast.walk(observer_tree) if isinstance(node, ast.FunctionDef) and node.name in ('pre', 'post')]
    assert len(hooks) == 2 and all(not any(isinstance(child, ast.Return) for child in ast.walk(node)) for node in hooks)
    receipt = {'schema': 'pubmed-shared4-owned-continuation-diagnostic-source-only-check-v3',
               'UTC': datetime.now(timezone.utc).isoformat(), 'candidate_manifest_sha256': EXPECTED,
               'candidate_payloads': candidate_rows, 'external_source_pins': external_rows,
               'v2_predecessor_payloads': predecessor_rows, 'prior_independent_v1_review_payloads': prior_review_rows,
               'author_snapshot_rows': snapshot_rows, 'closure_matches': True,
               'fixed_profile_tolerance_inputs_reused': equal_profile, 'caps_byte_semantics_reused': True,
               'changed_v2_inherited_payloads': changed, 'SOURCE_CHECK_delta': delta,
               'common_FunctionDefs_except_gate_AST_exact_v1': common_reuse,
               'original_diagnostic_predicates_AST_exact_v1': predicate_reuse,
               'comparator_FunctionDef_AST_sha256': comparator_sha,
               'target_Python_AST_checks': python_rows, 'supervise_byte_exact_v1': True,
               'native_observer_byte_exact_original_v2': True, 'optimizer_hooks_implicit_None': True,
               'target_import_compile_execution': False, 'numerical_imports': False,
               'actual_failed_execution_metadata_opened': False, 'state_score_data_array_payloads_accessed': False,
               'remote_access_or_dispatch': False, 'runtime_or_signal_probe': False}
    (HERE / 'INPUT_CHECKS.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'candidate_payloads': len(candidate_rows), 'candidate_bytes': sum(row['bytes'] for row in candidate_rows),
                      'external_pins': len(external_rows), 'author_snapshot_rows': len(snapshot_rows),
                      'v2_payloads_preserved': len(predecessor_rows), 'status': 'SOURCE_ONLY_CHECKS_PASS'}))


if __name__ == '__main__':
    main()
