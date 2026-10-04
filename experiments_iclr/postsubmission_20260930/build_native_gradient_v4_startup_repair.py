"""Preserve the failed diagnostic; seal a narrow CUDA startup correction."""
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
OLD = P / 'graph_count_conditioned_pattern_minimal_gradient_preparation_20261004_v3'
NEW = P / 'graph_count_conditioned_pattern_minimal_gradient_preparation_20261004_v4'
EXEC = P / 'graph_count_conditioned_pattern_minimal_gradient_execution_root_20261004_v2'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')

old_manifest = json.loads((OLD / 'MANIFEST.json').read_text())
for row in old_manifest['files']:
    f = OLD / row['path']
    assert f.stat().st_size == row['bytes'] and sha(f) == row['sha256']
failure_root = P / 'graph_count_conditioned_pattern_minimal_gradient_execution_root_20261004_v1/owned_monitor01'
terminal_path = failure_root / 'supervision/run01/TERMINAL.json'
failure_path = failure_root / 'run01/FAILURE.json'
terminal = json.loads(terminal_path.read_text())
assert terminal['status'] == 'FAILED_NO_DIAGNOSTIC_ADOPTION'
assert terminal['physical_exit_code'] == 1 and terminal['physical_session_closed'] is True
assert terminal['source_manifest_sha256'] == sha(OLD / 'MANIFEST.json')
assert json.loads(failure_path.read_text())['condition'] == 'Invalid device argument '
NEW.mkdir()
for row in old_manifest['files']:
    if row['path'] in ('AUTHOR_SOURCE_CHECK.json', 'SOURCE_CHANGE.json'):
        continue
    (NEW / row['path']).write_bytes((OLD / row['path']).read_bytes())
diagnostic = (OLD / 'diagnostic.py').read_text()
early = '        torch.cuda.reset_peak_memory_stats(0)\n'
assert diagnostic.count(early) == 1
fixed = diagnostic.replace(early, '')
anchor = '        device, sampler = pilot_model.ordinary_runtime(context)\n'
assert fixed.count(anchor) == 1
fixed = fixed.replace(anchor, anchor +
    '        require(torch.cuda.is_initialized() and torch.cuda.current_device() == 0,\n'
    '                "Authenticated CUDA context must precede memory-stat reset")\n' + early)
(NEW / 'diagnostic.py').write_text(fixed)
plan = json.loads((OLD / 'PLAN.json').read_text())
prior_plan = json.loads(json.dumps(plan))
plan['execution_directory'] = EXEC.name
(NEW / 'PLAN.json').unlink()
write(NEW / 'PLAN.json', plan)
assert {k:v for k,v in plan.items() if k != 'execution_directory'} == {
    k:v for k,v in prior_plan.items() if k != 'execution_directory'}
diff = ''.join(difflib.unified_diff(diagnostic.splitlines(True), fixed.splitlines(True),
    fromfile=OLD.name+'/diagnostic.py', tofile=NEW.name+'/diagnostic.py'))
(NEW / 'V3_TO_V4.diff').write_text(diff)
pins = []
for f in (OLD / 'MANIFEST.json', OLD / 'SEAL.json', terminal_path, failure_path,
          failure_root / 'supervision/run01/STDERR.txt',
          P / 'graph_count_conditioned_pattern_minimal_gradient_independent_source_review_20261004_v3/REVIEW.json'):
    pins.append(dict(path=str(f.relative_to(P)), bytes=f.stat().st_size, sha256=sha(f)))
write(NEW / 'SOURCE_CHANGE.json', dict(schema='minimal-gradient-CUDA-startup-repair-v4',
    prior_inputs=pins, preserved_failed_execution=True, execution_authorized=False,
    change='Move memory-stat reset after qualified ordinary_runtime initializes/authenticates CUDA; assert initialized/current device0 first.',
    new_execution_directory=EXEC.name, unchanged_scientific_workload=True,
    automatic_retry=False, failure_before_data_or_model_loading=True))
(NEW / 'V4_REPAIR.md').write_text(
    '# CUDA startup correction\n\n'
    'The v3 worker failed before data/model loading because reset_peak_memory_stats(0) '
    'ran before Torch initialized CUDA. Its failed physical terminal is retained. '
    'This source moves the reset after the existing qualified ordinary_runtime, '
    'which authenticates the one visible GPU, calls set_device(0), and synchronizes. '
    'An explicit initialized/current-device check now precedes reset. No random '
    'draw, objective, batch, gradient, tolerance or cap changes. The inclusive '
    'wall timer still starts before the admission gate. A fresh execution directory '
    'prevents rewriting or restarting the failed attempt. Review and separate root '
    'admission are required; this preparation has not executed.\n')
with (NEW / 'README.md').open('a') as stream:
    stream.write('\n## V4 startup correction\n\nSee V4_REPAIR.md. Prior repair documents '
                 'and checks describe their own sealed versions. V4 changes CUDA startup '
                 'order and the fresh execution directory only. No numerical result exists.\n')
checks = []
for name in ('common.py', 'diagnostic.py', 'metrics.py', 'supervise.py'):
    f = NEW / name
    ast.parse(f.read_text())
    checks.append(dict(path=name, bytes=f.stat().st_size, sha256=sha(f),
                       AST_parse=True, byte_identical_to_v3=f.read_bytes()==(OLD/name).read_bytes()))
for row in plan['source_pins']:
    f = P / row['path']
    assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256']
write(NEW / 'AUTHOR_SOURCE_CHECK.json', dict(status='STATIC_SOURCE_CHECK_NOT_INDEPENDENT_PASS',
    files=checks, source_pins_verified=len(plan['source_pins']),
    target_imported=False, numerical_execution=False, server_access=False,
    fixed_plan_except_execution_directory=True, prior_versions_preserved=True))
rows = [dict(path=str(f.relative_to(NEW)), bytes=f.stat().st_size, sha256=sha(f))
        for f in sorted(NEW.iterdir()) if f.is_file()]
write(NEW / 'MANIFEST.json', dict(schema='minimal-gradient-source-manifest-v4',
    execution_authorized=False, source_only=True, files=rows))
write(NEW / 'SEAL.json', dict(schema='minimal-gradient-source-seal-v4',
    UTC=datetime.now(timezone.utc).isoformat(), manifest_sha256=sha(NEW/'MANIFEST.json'),
    payload_files=len(rows), payload_bytes=sum(r['bytes'] for r in rows),
    execution_authorized=False, independent_source_review_required=True,
    prior_failure_preserved=True, numerical_execution=False))
EXEC.mkdir()
old_exec = P / 'graph_count_conditioned_pattern_minimal_gradient_execution_root_20261004_v1'
for name in ('stage_admit_diagnostic.py', 'monitor_owned_diagnostic.py'):
    code = (old_exec/name).read_text().replace('count_native_gradient_v1_', 'count_native_gradient_v2_')
    ast.parse(code)
    (EXEC/name).write_text(code)
write(EXEC/'CLIENT_PREPARATION.json', dict(source=NEW.name, source_manifest_sha256=sha(NEW/'MANIFEST.json'),
    stage_or_release_or_launch=False, old_execution_preserved=True,
    only_client_change='unique transport identity v2; HERE supplies fresh directory'))
print(json.dumps(dict(candidate=NEW.name, manifest_sha256=sha(NEW/'MANIFEST.json'),
    fresh_execution=EXEC.name, execution=False, files=len(rows))))
