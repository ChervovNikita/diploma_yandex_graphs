"""Observe the original handle and retain a pinned projection of large JSON."""
from pathlib import Path
import argparse
import base64
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('owned_diagnostic_transport', HERE / 'stage_admit_monitor.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
parser = argparse.ArgumentParser()
parser.add_argument('--sequence', type=int, required=True)
args = parser.parse_args()
assert 1 <= args.sequence <= 99
out = HERE / ('compact_monitor%02d' % args.sequence)
out.mkdir()
expected = json.loads((HERE / 'DETACHED_LAUNCH.json').read_text())
code = driver.context() + 'expected=' + repr(expected) + '\n'
code += '''
proc=Path('/proc')/str(expected['supervisor_PID']);identity=None
if proc.exists():
 raw=(proc/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
 assert int(fields[19])==expected['supervisor_start_ticks']
 identity=dict(pid=expected['supervisor_PID'],start_ticks=int(fields[19]),state=fields[0])
files=[];large=[];projection=None
for relative in ['continuation_diagnostic/run01/DIAGNOSTIC.json','continuation_diagnostic/run01/FAILURE.json','continuation_diagnostic/run01/RESTORE_ALIAS_PROGRESS.json','continuation_diagnostic/run01/FINAL_CUSTODY.json','continuation_diagnostic/run01/STATUS.json','supervision/continuation_diagnostic/run01/SUPERVISOR_TERMINAL.json','supervision/continuation_diagnostic/run01/SUPERVISOR_STATUS.json','DETACHED_STDERR.txt']:
 path=root/relative
 if not path.is_file():continue
 assert not path.is_symlink() and path.stat().st_size<=64*1024**2
 raw=path.read_bytes();raw.decode('utf8');descriptor=dict(path=relative,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 if len(raw)<3_000_000:
  files.append(dict(descriptor,data=base64.b64encode(raw).decode()))
 else:large.append(descriptor)
 if relative.endswith('/DIAGNOSTIC.json'):
  d=json.loads(raw);e=d['diagnostic_evidence'];steps=e['per_step_observer']
  projection=dict(status=d['status'],source_manifest_sha256=d['source_manifest_sha256'],root_release_sha256=d['root_release_sha256'],progress=d['progress'],loss=e['loss'],saved_tree_mutated_after_first_epoch=e['saved_tree_mutated_after_first_epoch'],saved_tree_mutated_after_second_epoch=e['saved_tree_mutated_after_second_epoch'],native_streams_exact=e['native_streams_exact'],observed_final_continuation_parity_within_fixed_rule=e['observed_final_continuation_parity_within_fixed_rule'],first_divergence=steps['first_divergence'],hook_counts=steps['hook_counts'],hook_RNG_neutral_checks=steps['hook_RNG_neutral_checks'],peak_reference_tensor_payload_bytes=steps['peak_reference_tensor_payload_bytes'],retained_reference_tensor_payload_bytes=steps['retained_reference_tensor_payload_bytes'],unchanged_original_comparator_verdicts=e['unchanged_original_comparator_verdicts'],next_state_exact_blocks={k:dict(exact=v['exact'],within_fixed_numeric_rule=v['within_fixed_numeric_rule']) for k,v in e['next_state_exact_blocks'].items()},engineering_qualification_PASS=d['engineering_qualification_PASS'],scientific_fit_admitted=d['scientific_fit_admitted'],raw_diagnostic_descriptor=descriptor,complete_diagnostic_retained_on_server=True)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),supervisor_identity=identity,files=files,large_server_retained_metadata=large,diagnostic_projection=projection,metadata_only=True,signals_sent=False,restarts=False)))
'''
value = driver.transport('compact_monitor%02d' % args.sequence, code)
files = value.pop('files')
descriptors = []
terminal = None
for row in files:
    raw = base64.b64decode(row.pop('data'), validate=True)
    assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    path = out / row['path']
    assert path.resolve().is_relative_to(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)
    descriptors.append(row)
    if row['path'].endswith('SUPERVISOR_TERMINAL.json'):
        terminal = json.loads(raw)
value['fetched_descriptors'] = descriptors
driver.save(out / 'OBSERVATION.json', value)
driver.save(out / 'SUMMARY.json', dict(UTC=value['UTC'],supervisor_identity=value['supervisor_identity'],terminal_status=terminal.get('status') if terminal else None,terminal_exit_code=terminal.get('exit_code') if terminal else None,cap_violation=terminal.get('cap_violation') if terminal else None,diagnostic=value['diagnostic_projection']))
print(json.dumps(dict(UTC=value['UTC'],supervisor_identity=value['supervisor_identity'],terminal_status=terminal.get('status') if terminal else None,terminal_exit_code=terminal.get('exit_code') if terminal else None,diagnostic=value['diagnostic_projection']),indent=2))
