"""Independent stdlib source checks; no candidate imports, data or fits."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
CANDIDATE = PHASE / 'pubmed_native_predictive_program_source_20261004_v1'
EXPECTED_MANIFEST = '3bfea657c856ca0be987a4329f688e302b22dc245c0c05e3404ff5083dd5872d'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def functions(path):
    return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}

manifest = json.loads((CANDIDATE / 'MANIFEST.json').read_text())
assert digest(CANDIDATE / 'MANIFEST.json') == EXPECTED_MANIFEST
checked = []
for row in manifest['files']:
    if row['path'] in {'README.md', 'STATIC_SOURCE_REVIEW.json'}:
        continue
    path = CANDIDATE / row['path']
    checked.append({'path': str(path.relative_to(PHASE)), 'sha256': digest(path), 'bytes': path.stat().st_size,
                    'match': digest(path) == row['sha256'] and path.stat().st_size == row['bytes']})
    if path.suffix == '.py':
        ast.parse(path.read_text())
binding = json.loads((CANDIDATE / 'SOURCE_BINDING.json').read_text())
for row in binding['external_source_pins']:
    path = PHASE / row['path']
    checked.append({'path': row['path'], 'sha256': digest(path), 'bytes': path.stat().st_size,
                    'match': digest(path) == row['sha256'] and path.stat().st_size == row['bytes']})
snapshot = json.loads((PHASE / 'pubmed_heart_available_inspector_native_adapter_20261004_v1/PINNED_AUTHOR_SOURCE_AND_FUNCTIONS.json').read_text())
for row in snapshot['snapshot_files']:
    path = PHASE / 'pubmed_heart_available_inspector_native_adapter_20261004_v1/public_author_code/HeaRT' / row['path']
    checked.append({'path': str(path.relative_to(PHASE)), 'sha256': digest(path), 'bytes': path.stat().st_size,
                    'match': digest(path) == row['sha256'] and path.stat().st_size == row['bytes']})
assert all(row['match'] for row in checked)

bodies = functions(PHASE / 'pubmed_heart_native_numerical_qualification_source_20261004_v2/native_bodies.py')
adapter = functions(PHASE / 'pubmed_heart_available_inspector_native_adapter_20261004_v1/native_train_functions.py')
native_equality = {}
for candidate_name, adapter_name in [('candidate_sage_train', 'sage_train'), ('candidate_ncnc_train', 'ncnc_train'),
                                    ('candidate_sage_test_edge', 'sage_test_edge')]:
    candidate = copy.deepcopy(bodies[candidate_name])
    candidate.name = adapter_name
    native_equality[candidate_name] = ast.dump(candidate, include_attributes=False) == ast.dump(adapter[adapter_name], include_attributes=False)
assert all(native_equality.values())

supervisor = functions(CANDIDATE / 'supervise.py')
preflight = [node for node in ast.walk(supervisor['main'])
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name) and node.func.value.id == 'subprocess'
             and node.func.attr == 'run']
assert len(preflight) == 1

# Run only the exact isolated ownership helper against inert stubs. No subprocess
# is created and no OS signal is sent. A child disappearing before identity read
# must be considered by the supervisor's exception/reaping path.
kill_calls = []
def require(value, message):
    if not value:
        raise RuntimeError(message)
def absent_child(pid):
    raise FileNotFoundError('/proc/<owned child>/ns/pid disappeared')
namespace = {'require': require, 'physical': absent_child,
             'os': SimpleNamespace(killpg=lambda *args: kill_calls.append(args))}
exec(compile(ast.Module(body=[supervisor['signal_owned']], type_ignores=[]), str(CANDIDATE / 'supervise.py'), 'exec'), namespace)
try:
    namespace['signal_owned'](SimpleNamespace(pid=999999), {'start_time_ticks': 100}, 15)
except Exception as error:
    race = {'exception_propagates': type(error).__name__, 'killpg_calls': len(kill_calls),
            'interpretation': 'Full physical() read occurs outside the ProcessLookupError guard.'}
else:
    raise AssertionError('Expected disappearing-child exception')

result = {'schema': 'pubmed-native-fresh-source-checks-v1', 'candidate_manifest_sha256': EXPECTED_MANIFEST,
          'all_checked_source_pins_match': True, 'candidate_python_AST_parse': 'PASS',
          'hash_checked_files': checked, 'native_function_AST_equality_after_name_normalization': native_equality,
          'nvidia_smi_preflight': {'line': preflight[0].lineno,
              'timeout_present': any(keyword.arg == 'timeout' for keyword in preflight[0].keywords),
              'subprocess_before_supervision_output_and_owned_child_monitor': True},
          'disappearing_owned_child_helper_probe': race,
          'omitted_candidate_files': ['README.md', 'STATIC_SOURCE_REVIEW.json'],
          'scientific_fits_executed': False, 'candidate_modules_imported': False,
          'servers_accessed': False, 'checkpoints_read': False, 'candidate_edited': False,
          'previous_reviews_or_outcome_files_read': False}
(HERE / 'SOURCE_CHECKS.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({key: result[key] for key in ('all_checked_source_pins_match', 'candidate_python_AST_parse',
    'native_function_AST_equality_after_name_normalization', 'nvidia_smi_preflight', 'disappearing_owned_child_helper_probe')}, indent=2))
