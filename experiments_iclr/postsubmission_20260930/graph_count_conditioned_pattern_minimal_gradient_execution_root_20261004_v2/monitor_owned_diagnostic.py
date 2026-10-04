"""Observe exact owned native diagnostic handles and scalar output metadata."""
from pathlib import Path
import argparse
import base64
import hashlib
import importlib.util
import json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('owned_native_gradient_transport', HERE / 'stage_admit_diagnostic.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
parser = argparse.ArgumentParser()
parser.add_argument('--sequence', type=int, required=True)
args = parser.parse_args()
assert 1 <= args.sequence <= 99
out = HERE / ('owned_monitor%02d' % args.sequence)
out.mkdir()
expected = json.loads((HERE / 'DETACHED_LAUNCH.json').read_text())
code = driver.context() + 'expected=' + repr(expected) + '\n'
code += '''
proc=Path('/proc')/str(expected['supervisor_PID']);identity=None
if proc.exists():
 raw=(proc/'stat').read_text();f=raw[raw.rfind(')')+2:].split()
 assert int(f[19])==expected['supervisor_start_ticks']
 argv=[v.decode() for v in (proc/'cmdline').read_bytes().split(bytes([0])) if v]
 assert argv==expected['command'] or f[0]=='Z'
 identity=dict(pid=expected['supervisor_PID'],start_ticks=int(f[19]),state=f[0])
files=[];large=[];diagnostic=None;terminal=None;progress=None
for relative in ['supervision/run01/STARTED.json','supervision/run01/PHYSICAL_TERMINAL.json','supervision/run01/TERMINAL.json','supervision/run01/SUPERVISOR_CUSTODY.json','supervision/run01/STDERR.txt','run01/PROGRESS.json','run01/DIAGNOSTIC.json','run01/FILE_CUSTODY.json','run01/FINAL_CUSTODY.json','run01/FAILURE.json','run01/RESOURCE_STOP.json','DETACHED_STDERR.txt']:
 p=root/relative
 if not p.is_file():continue
 assert not p.is_symlink() and p.stat().st_size<=64*1024**2
 b=p.read_bytes();b.decode('utf8');row=dict(path=relative,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 if len(b)<2_000_000:files.append(dict(row,data=base64.b64encode(b).decode()))
 else:large.append(row)
 if relative=='supervision/run01/TERMINAL.json':terminal=json.loads(b)
 if relative=='run01/PROGRESS.json':progress=json.loads(b)
 if relative=='run01/DIAGNOSTIC.json':
  q=json.loads(b)
  diagnostic={k:q[k] for k in ('status','source_manifest_sha256','release_sha256','core_manifest_sha256','workload','stream','losses','parameter_blocks','times_seconds','dispatch','inclusive_child_wall_seconds','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','optimizer_constructed','optimizer_updates','fits','VALID_TEST_reads','no_predictive_or_novelty_claim')}
  diagnostic['populations']={name:dict(census={k:v for k,v in value['census'].items() if k not in ('side_n_k_histograms','joint_n_k_histogram')},slot_gradients=value['slot_gradients'],native_target_logits=value['native_target_logits'],auxiliary_logits=value['auxiliary_logits'],aux_losses=value['aux_losses']) for name,value in q['populations'].items()}
  diagnostic['raw_descriptor']=row
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),supervisor_identity=identity,files=files,large_server_retained_metadata=large,progress=progress,diagnostic_summary=diagnostic,terminal=terminal,signals_sent=False,retries=False,metadata_only=True)))
'''
value = driver.transport('monitor%02d' % args.sequence, code)
rows = value.pop('files')
value['fetched_descriptors'] = []
for row in rows:
    raw = base64.b64decode(row.pop('data'), validate=True)
    assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    path = out / row['path']
    assert path.resolve().is_relative_to(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)
    value['fetched_descriptors'].append(row)
with (out / 'OBSERVATION.json').open('x') as stream:
    json.dump(value, stream, indent=2, allow_nan=False)
    stream.write('\n')
terminal = value['terminal']
summary = dict(UTC=value['UTC'], supervisor_identity=value['supervisor_identity'],
               progress=value['progress'], diagnostic=value['diagnostic_summary'],
               terminal_status=terminal.get('status') if terminal else None,
               physical_exit_code=terminal.get('physical_exit_code') if terminal else None,
               physical_session_closed=terminal.get('physical_session_closed') if terminal else None,
               stop=terminal.get('stop') if terminal else None,
               linked=terminal.get('linked') if terminal else None,
               adoption_eligible_only_after_original_supervisor_terminal=True)
with (out / 'SUMMARY.json').open('x') as stream:
    json.dump(summary, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({k:summary[k] for k in ('UTC','supervisor_identity','progress','terminal_status','physical_exit_code','stop')}, indent=2))
