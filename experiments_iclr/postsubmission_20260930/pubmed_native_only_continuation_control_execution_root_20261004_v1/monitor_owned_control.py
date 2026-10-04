"""Observe the original owned control and retain small hash-linked metadata."""
from pathlib import Path
import argparse
import base64
import hashlib
import importlib.util
import json

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('owned_native_control_driver',HERE/'stage_admit_control.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
parser=argparse.ArgumentParser();parser.add_argument('--sequence',type=int,required=True)
args=parser.parse_args();assert 1<=args.sequence<=99
out=HERE/('owned_monitor%02d'%args.sequence);out.mkdir()
expected=json.loads((HERE/'DETACHED_LAUNCH.json').read_text())
code=driver.context()+'expected='+repr(expected)+'\n'
code+='''
proc=Path('/proc')/str(expected['supervisor_PID']);identity=None
if proc.exists():
 raw=(proc/'stat').read_text();f=raw[raw.rfind(')')+2:].split()
 assert int(f[19])==expected['supervisor_start_ticks']
 argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]
 assert f[0]=='Z' or argv==expected['command']
 identity=dict(pid=expected['supervisor_PID'],start_ticks=int(f[19]),state=f[0])
files=[];large=[];projection=None;terminal=None;progress=None
names=['native_continuation_control/run01/DIAGNOSTIC.json','native_continuation_control/run01/FAILURE.json',
 'native_continuation_control/run01/PROGRESS.json','native_continuation_control/run01/TRAIN_INSPECTION.json',
 'native_continuation_control/run01/WARMUP.json','native_continuation_control/run01/RESTORE_ALIAS_PROGRESS.json',
 'native_continuation_control/run01/CUDA_PEAKS.json','native_continuation_control/run01/STEP_DIVERGENCE_PROGRESS.json',
 'native_continuation_control/run01/FINAL_CUSTODY.json','supervision/native_continuation_control/run01/TERMINAL.json',
 'supervision/native_continuation_control/run01/PHYSICAL_TERMINAL.json',
 'supervision/native_continuation_control/run01/SUPERVISOR_CUSTODY.json',
 'supervision/native_continuation_control/run01/STARTED.json',
 'supervision/native_continuation_control/run01/CHILD_STDERR.txt','DETACHED_STDERR.txt']
for relative in names:
 p=root/relative
 if not p.is_file():continue
 assert not p.is_symlink() and p.stat().st_size<=64*1024**2
 raw=p.read_bytes();raw.decode('utf8');descriptor=dict(path=relative,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 if len(raw)<2_000_000:files.append(dict(descriptor,data=base64.b64encode(raw).decode()))
 else:large.append(descriptor)
 if relative=='supervision/native_continuation_control/run01/TERMINAL.json':terminal=json.loads(raw)
 if relative=='native_continuation_control/run01/PROGRESS.json':progress=json.loads(raw)
 if relative.endswith('/DIAGNOSTIC.json'):
  d=json.loads(raw);e=d['diagnostic_evidence'];steps=e['per_step_observer']
  projection=dict(status=d['status'],source_manifest_sha256=d['source_manifest_sha256'],root_release_sha256=d['root_release_sha256'],
   progress=d['progress'],warmup=d['warmup'],continuation=d['continuation'],loss=e['loss'],
   saved_tree_mutated_after_first_epoch=e['saved_tree_mutated_after_first_epoch'],
   saved_tree_mutated_after_second_epoch=e['saved_tree_mutated_after_second_epoch'],
   native_streams_exact=e['native_streams_exact'],repeat_control_preconditions_exact=e['repeat_control_preconditions_exact'],
   observed_final_continuation_parity_within_fixed_rule=e['observed_final_continuation_parity_within_fixed_rule'],
   first_divergence=steps['first_divergence'],hook_counts=steps['hook_counts'],hook_RNG_neutral_checks=steps['hook_RNG_neutral_checks'],
   peak_reference_tensor_payload_bytes=steps['peak_reference_tensor_payload_bytes'],
   retained_reference_tensor_payload_bytes=steps['retained_reference_tensor_payload_bytes'],
   unchanged_original_comparator_verdicts=e['unchanged_original_comparator_verdicts'],
   next_state_exact_blocks={k:dict(exact=v['exact'],within_fixed_numeric_rule=v['within_fixed_numeric_rule']) for k,v in e['next_state_exact_blocks'].items()},
   engineering_qualification_PASS=d['engineering_qualification_PASS'],scientific_fit_admitted=d['scientific_fit_admitted'],
   state_file_loads=d['state_file_loads'],raw_diagnostic_descriptor=descriptor,complete_diagnostic_retained_on_server=True)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),supervisor_identity=identity,files=files,
 large_server_retained_metadata=large,diagnostic_projection=projection,terminal=terminal,progress=progress,
 metadata_only=True,signals_sent=False,restarts=False)))
'''
value=driver.transport('monitor%02d'%args.sequence,code)
value['fetched_descriptors']=[]
for row in value.pop('files'):
 raw=base64.b64decode(row.pop('data'),validate=True)
 assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 p=out/row['path'];assert p.resolve().is_relative_to(out);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as stream:stream.write(raw)
 value['fetched_descriptors'].append(row)
with (out/'OBSERVATION.json').open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
t=value['terminal']
summary=dict(UTC=value['UTC'],supervisor_identity=value['supervisor_identity'],progress=value['progress'],
 terminal_status=t.get('status') if t else None,stop=t.get('stop') if t else None,
 diagnostic=value['diagnostic_projection'],linked=t.get('linked') if t else None)
with (out/'SUMMARY.json').open('x') as stream:json.dump(summary,stream,indent=2);stream.write('\n')
print(json.dumps(summary,indent=2))
