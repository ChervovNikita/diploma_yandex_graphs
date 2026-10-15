"""Pure source/compact-metadata checks; no subprocess, GPU, model or data calls."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import host_binding as host
from descriptor_binding import audit_runtime_paths, rebind_job
from run_gpu import command

HERE = Path(__file__).resolve().parent
E = HERE.parent
checks = []


def reject(name, call):
    try:
        call()
    except (ValueError, KeyError):
        checks.append(name)
        return
    raise AssertionError('Did not reject: ' + name)


def functions(source):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


for path in HERE.rglob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
checks.append('all_source_parses')
configs = {worker: host.load_config(host.config_path(worker))[0] for worker in host.WORKERS}
assert configs['allocation/gpu0']['host_enabled'] and not any(configs[w]['host_enabled'] for w in host.WORKERS[1:])
checks.append('allocation_only_host_enabled')
allocation = configs['allocation/gpu0']
host.check_observed_identity(allocation, 'anogena-2-0', [host.ALLOCATION_UUID])
checks.append('declared_identity_matches')
reject('wrong_hostname_rejected', lambda: host.check_observed_identity(allocation, 'elsewhere', [host.ALLOCATION_UUID]))
reject('wrong_GPU_rejected', lambda: host.check_observed_identity(allocation, 'anogena-2-0', ['GPU-other']))
reject('extra_GPU_rejected', lambda: host.check_observed_identity(allocation, 'anogena-2-0', [host.ALLOCATION_UUID, 'GPU-other']))

def forbidden_probe(*args, **kwargs):
    raise AssertionError('A disabled host attempted a probe')

old_hostname, old_subprocess = host.socket.gethostname, host.subprocess.check_output
host.socket.gethostname, host.subprocess.check_output = forbidden_probe, forbidden_probe
try:
    for worker in host.WORKERS[1:]:
        reject(worker + '_blocked_before_probe', lambda worker=worker: host.guard(configs[worker]))
finally:
    host.socket.gethostname, host.subprocess.check_output = old_hostname, old_subprocess

delta = json.loads((HERE / 'RUNTIME_DELTA.json').read_text())
for row in delta['files']:
    old, new = (E / row['source']).read_text(), (HERE / row['target']).read_text()
    assert hashlib.sha256(old.encode()).hexdigest() == row['source_sha256']
    assert hashlib.sha256(new.encode()).hexdigest() == row['target_sha256']
    before, after = functions(old), functions(new)
    allowed = {'require_allocation'} if row['source'].endswith('/common.py') else {'main'}
    for name, body in before.items():
        if name not in allowed:
            assert after[name] == body, (row['target'], name)
checks.append('native_training_selection_export_and_evaluator_functions_AST_identical')
before = functions((E / 'expanded_dispatcher_v2/qualified_python_entry.py').read_text())
after = functions((HERE / 'qualified_python_entry.py').read_text())
assert before['validate_runtime_environment'] == after['validate_runtime_environment']
checks.append('lexical_venv_actual_environment_validation_AST_identical')

representatives = []
for relative in ('qualification/runtime_v4_node_selected_mean_lifecycle_v1/JOB.json',
                 'qualification/molhiv_gin_vn_cuda_resource_v1/JOB.json',
                 'qualification/buddy_native_provider_input_gate_v2/JOB.json'):
    job = json.loads((E / relative).read_text())
    original = copy.deepcopy(job)
    target = configs['server77/gpu0']
    bound, receipt = rebind_job(job, host.ALLOCATION_ROOT, host.PHASE_RELATIVE,
                               target['project_root'], target['phase_relative'])
    restored, _ = rebind_job(bound, target['project_root'], target['phase_relative'], host.ALLOCATION_ROOT, host.PHASE_RELATIVE)
    assert original == job == restored and receipt['field_changes']
    assert job['callback_sha256'] == bound['callback_sha256'] and job['recipe']['native_fixed_config'] == bound['recipe']['native_fixed_config']
    identity, same = rebind_job(job, host.ALLOCATION_ROOT, host.PHASE_RELATIVE, host.ALLOCATION_ROOT, host.PHASE_RELATIVE)
    assert identity == job and not same['field_changes'] and same['canonical_runtime_job_hash'] == same['bound_runtime_job_hash']
    representatives.append({'path': relative, 'field_changes': len(receipt['field_changes']), 'roundtrip_exact': True})
checks.append('node_molecular_BUDDY_fieldwise_roundtrip_and_allocation_identity')
bad = copy.deepcopy(original); bad['unclassified_runtime_file'] = host.ALLOCATION_ROOT + '/file.json'
reject('unknown_absolute_runtime_field_rejected', lambda: audit_runtime_paths(bad, host.ALLOCATION_ROOT, host.PHASE_RELATIVE))
bad = copy.deepcopy(original); bad['data_root'] = host.ALLOCATION_ROOT + '/../outside'
reject('runtime_path_traversal_rejected', lambda: audit_runtime_paths(bad, host.ALLOCATION_ROOT, host.PHASE_RELATIVE))
bad = copy.deepcopy(original); bad['data_root'] = '/tmp/unrelated'
reject('runtime_path_outside_root_rejected', lambda: audit_runtime_paths(bad, host.ALLOCATION_ROOT, host.PHASE_RELATIVE))

environment = json.loads((E / 'expanded_dispatcher_v2/PYTHON_ENVIRONMENT.json').read_text())
spec = {'schema': 'portable-reviewed-allocation-run-v1', 'worker': 'allocation/gpu0', 'python_environment': environment,
        'dispatcher': {'path': host.ALLOCATION_ROOT + '/' + host.PHASE_RELATIVE + '/benchmark_expansion_20261015_v1/expanded_dispatcher_v2/dispatcher.py'}}
for key in ('release', 'go', 'readiness'):
    spec[key] = {'path': host.ALLOCATION_ROOT + '/' + key + '.json'}
argv = command(allocation, spec)
assert argv[0] == environment['lexical_executable'] and argv[0] != environment['resolved_executable']
checks.append('allocation_command_uses_lexical_venv_not_resolved_binary')
reject('77_never_delegated_to_allocation_dispatcher', lambda: command(configs['server77/gpu0'], spec))
assert not any(name in sys.modules for name in ('torch', 'numpy', 'scipy', 'ogb', 'torch_geometric'))
checks.append('no_scientific_imports')
result = {'schema': 'portable-source-only-checks-v1', 'status': 'PASS', 'checks': checks,
          'representative_metadata_rebindings': representatives, 'all6460_descriptor_rebinding_verified': False,
          'native_executions': 0, 'host_or_GPU_probes': 0, 'remote_connections': 0, 'data_or_tensor_reads': 0,
          'shared_host_scheduler_qualification': False, 'campaign_admission': False}
(HERE / 'CHECKS.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'checks': len(checks), 'native_executions': 0, 'host_or_GPU_probes': 0}))
