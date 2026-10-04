"""Narrow successor source review; exact helper ASTs, inert stubs only."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import signal
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
OLD = PHASE / 'pubmed_native_predictive_program_source_20261004_v1'
NEW = PHASE / 'pubmed_native_predictive_program_source_20261004_v3'
EXPECTED = 'cd360a31fc6feead174f1da3ea9478655aa3bd78f68aab548601cc9fb37090e5'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def require(value, message):
    if not value: raise RuntimeError(message)
assert sha(NEW / 'MANIFEST.json') == EXPECTED
immutable = {name: sha(OLD / name) == sha(NEW / name) for name in
             ('common.py', 'fit_cohort.py', 'replay_selected.py', 'COHORT.json', 'SOURCE_BINDING.json', 'ROOT_RELEASE_TEMPLATE.json')}
assert all(immutable.values())
manifest = json.loads((NEW / 'MANIFEST.json').read_text())
checked = []
for row in manifest['files']:
    if row['path'] in {'README.md', 'STATIC_SOURCE_REVIEW.json'}: continue
    path = NEW / row['path']
    checked.append({'path': row['path'], 'sha256': sha(path), 'bytes': path.stat().st_size,
                    'match': sha(path) == row['sha256'] and path.stat().st_size == row['bytes']})
assert all(row['match'] for row in checked)
tree = ast.parse((NEW / 'supervise.py').read_text())
fns = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
kill_calls = []
def vanished(pid): raise FileNotFoundError('inert disappearing-child stub')
ns = {'require': require, 'physical': vanished,
      'os': SimpleNamespace(killpg=lambda *args: kill_calls.append(args), WNOHANG=1),
      'time': SimpleNamespace(monotonic=lambda: 10, sleep=lambda value: None), 'subprocess': subprocess}
exec(compile(ast.Module(body=[fns['signal_owned'], fns['reap_owned']], type_ignores=[]), 'supervise.py', 'exec'), ns)
child = SimpleNamespace(pid=99, returncode=None, args=['inert'])
assert ns['signal_owned'](child, {'start_time_ticks': 100}, signal.SIGTERM) is False
assert kill_calls == []
ns['physical'] = lambda pid: {'start_time_ticks': 101}
try: ns['signal_owned'](child, {'start_time_ticks': 100}, signal.SIGTERM)
except RuntimeError: pass
else: raise AssertionError('PID reuse should refuse signalling')
assert kill_calls == []
usage = SimpleNamespace(ru_maxrss=100, ru_utime=2, ru_stime=1)
ns['os'].wait4 = lambda pid, option: (99, 0, usage)
ns['os'].waitstatus_to_exitcode = lambda status: 0
assert ns['reap_owned'](child, 5) is usage and child.returncode == 0

preflight = next(node for node in fns['main'].body if isinstance(node, ast.Try) and node.lineno == 74)
preflight_cases = []
for cap in (100, 12):
    writes, calls = [], []
    def timeout_query(argv, **kwargs):
        calls.append(kwargs['timeout'])
        raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
    context = {'time': SimpleNamespace(monotonic=lambda: 10), 'started': 0, 'caps': {'wall_seconds': cap},
               'require': require, 'subprocess': SimpleNamespace(run=timeout_query), 'gpu_rows': None,
               'output': HERE, 'args': SimpleNamespace(stage='fit_cohort'),
               'write': lambda path, receipt: writes.append(receipt)}
    try: exec(compile(ast.Module(body=[preflight], type_ignores=[]), 'supervise.py', 'exec'), context)
    except subprocess.TimeoutExpired: pass
    assert writes[0]['status'] == 'FAILED' and writes[0]['child_launched'] is False
    preflight_cases.append({'stage_wall_cap': cap, 'elapsed_before_preflight': 10, 'query_timeout': calls[0],
                            'failure_receipt': writes[0]})

owned_with = next(node for node in fns['main'].body if isinstance(node, ast.With))
cleanup_handler = owned_with.body[0].handlers[0]
def cleanup_case(identity, signal_fn, reap_fn):
    receipts = []
    context = {'child': SimpleNamespace(pid=99, returncode=None, args=['inert']), 'identity': identity, 'usage': None,
               'signal_owned': signal_fn, 'reap_owned': reap_fn, 'signal': signal, 'subprocess': subprocess, 'require': require,
               'write': lambda path, receipt: receipts.append({'path': path.name, 'receipt': receipt}),
               'output': HERE, 'time': SimpleNamespace(monotonic=lambda: 10), 'started': 0, 'observed_peak': 0}
    body = ast.parse("raise RuntimeError('inert initial failure')").body
    try: exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Try(body=body, handlers=[cleanup_handler], orelse=[], finalbody=[])], type_ignores=[])), 'supervise.py', 'exec'), context)
    except Exception as error: raised = type(error).__name__
    return {'propagated_exception': raised, 'written_receipts': receipts}
missing_identity = cleanup_case(None, ns['signal_owned'], ns['reap_owned'])
assert missing_identity['written_receipts'][-1]['receipt']['kernel_usage_available'] is True
assert missing_identity['written_receipts'][-1]['receipt']['child_exit_code'] == 0
def always_timeout(*args): raise subprocess.TimeoutExpired(['inert'], args[1])
escalation_timeout = cleanup_case({'start_time_ticks': 100}, lambda *args: True, always_timeout)
assert escalation_timeout['propagated_exception'] == 'RuntimeError'
assert [row['path'] for row in escalation_timeout['written_receipts']] == ['CLEANUP_FAILURE.json', 'SUPERVISOR_FAILURE.json']
assert escalation_timeout['written_receipts'][0]['receipt']['errors'][0]['action'] == 'wait4_after_escalation'
refused_signal = cleanup_case({'start_time_ticks': 100}, ns['signal_owned'], ns['reap_owned'])
assert [row['path'] for row in refused_signal['written_receipts']] == ['CLEANUP_FAILURE.json', 'SUPERVISOR_FAILURE.json']
assert refused_signal['written_receipts'][-1]['receipt']['kernel_usage_available'] is True
assert kill_calls == []

result = {'schema': 'pubmed-native-successor-delta-checks-v1', 'candidate_manifest_sha256': EXPECTED,
          'same_reviewer_successor_followup': True, 'immutable_native_payloads_equal_v1': immutable,
          'checked_candidate_payloads': checked, 'supervisor_AST_parse': 'PASS',
          'preflight_timeout_and_failure_receipt_probes': preflight_cases,
          'acquired_identity_disappearance_signal_noop': True, 'reused_PID_signal_refused': True,
          'wait4_reap_returns_usage_and_sets_returncode': True,
          'resolved_missing_identity_cleanup_probe': missing_identity,
          'resolved_escalation_timeout_cleanup_probe': escalation_timeout,
          'refused_signal_still_reaps_and_records_usage_probe': refused_signal,
          'scientific_fits_executed': False, 'servers_accessed': False, 'checkpoints_read': False,
          'candidate_imported_or_edited': False, 'real_subprocesses_or_signals_in_probes': False}
(HERE / 'SUCCESSOR_CHECKS.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({key: result[key] for key in ('immutable_native_payloads_equal_v1', 'supervisor_AST_parse',
    'acquired_identity_disappearance_signal_noop', 'wait4_reap_returns_usage_and_sets_returncode',
    'resolved_missing_identity_cleanup_probe', 'resolved_escalation_timeout_cleanup_probe',
    'refused_signal_still_reaps_and_records_usage_probe')}, indent=2))
