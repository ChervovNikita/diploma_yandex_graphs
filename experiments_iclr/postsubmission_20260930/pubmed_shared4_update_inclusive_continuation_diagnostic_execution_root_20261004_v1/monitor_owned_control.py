"""Observe the exact original zero-update handle and compact scalar comparisons."""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pubmed_update_inclusive_owned_driver',HERE/'stage_admit_control.py')
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
 argv=[v.decode() for v in (proc/'cmdline').read_bytes().split(bytes([0])) if v]
 assert f[0]=='Z' or argv==expected['command']
 identity=dict(pid=expected['supervisor_PID'],start_ticks=int(f[19]),state=f[0])
files=[];large=[];diagnostic=None;terminal=None;progress=None
names=['update_inclusive/run01/DIAGNOSTIC.json','update_inclusive/run01/FAILURE.json',
 'update_inclusive/run01/PROGRESS.json','update_inclusive/run01/REPLAY_PROGRESS.json',
 'update_inclusive/run01/TRAIN_INSPECTION.json','update_inclusive/run01/CUDA_PEAKS.json',
 'update_inclusive/run01/FINAL_CUSTODY.json','supervision/update_inclusive/run01/STARTED.json',
 'supervision/update_inclusive/run01/PHYSICAL_TERMINAL.json',
 'supervision/update_inclusive/run01/TERMINAL.json',
 'supervision/update_inclusive/run01/SUPERVISOR_CUSTODY.json',
 'supervision/update_inclusive/run01/STDERR.txt','DETACHED_STDERR.txt']
for relative in names:
 p=root/relative
 if not p.is_file():continue
 assert not p.is_symlink() and p.resolve().is_relative_to(root) and p.stat().st_size<=64*1024**2
 b=p.read_bytes();b.decode('utf8');row=dict(path=relative,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 if len(b)<2_000_000:files.append(dict(row,data=base64.b64encode(b).decode()))
 else:large.append(row)
 if relative=='update_inclusive/run01/PROGRESS.json':progress=json.loads(b)
 if relative=='supervision/update_inclusive/run01/TERMINAL.json':terminal=json.loads(b)
 if relative=='update_inclusive/run01/DIAGNOSTIC.json':
  d=json.loads(b);e=d['diagnostic_evidence']
  diagnostic={k:d[k] for k in ('status','source_manifest_sha256','root_release_sha256','progress',
   'owned_engineering_state_sha256','owned_serialized_identity_sha256','inclusive_child_wall_seconds',
   'engineering_qualification_PASS','scientific_fit_admitted','state_donor_allowed',
   'VALID_files_opened','TEST_files_opened','score_files_opened','numeric_rule_changed',
   'restore_helper_changed','kernel_causality_established','interpretation')}
  diagnostic['decision']=e['decision']
  diagnostic['loss']=e['loss']
  diagnostic['observed_final_continuation_parity_within_fixed_rule']=e['observed_final_continuation_parity_within_fixed_rule']
  diagnostic['native_streams_exact']=e['native_streams_exact']
  diagnostic['original_comparator_verdicts']=e['unchanged_original_comparator_verdicts']
  diagnostic['selected_update_comparisons']=[{k:r[k] for k in ('step','same_state_interpretation_allowed','original_fixed_comparator_verdicts') if k in r} for r in e['selected_update_channels']['comparisons']]
  diagnostic['raw_descriptor']=row
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),supervisor_identity=identity,
 files=files,large_server_retained_metadata=large,progress=progress,diagnostic_projection=diagnostic,
 terminal=terminal,metadata_only=True,signals_sent=False,retries=False)))
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
t=value['terminal'];d=value['diagnostic_projection']
summary=dict(UTC=value['UTC'],supervisor_identity=value['supervisor_identity'],progress=value['progress'],
 terminal_status=t.get('status') if t else None,physical_exit_code=t.get('physical_exit_code') if t else None,
 physical_session_closed=t.get('physical_session_closed') if t else None,stop=t.get('stop') if t else None,
 diagnostic=d,linked=t.get('linked') if t else None)
with (out/'SUMMARY.json').open('x') as stream:json.dump(summary,stream,indent=2);stream.write('\n')
print(json.dumps(dict(UTC=summary['UTC'],supervisor_identity=summary['supervisor_identity'],
 terminal_status=summary['terminal_status'],physical_exit_code=summary['physical_exit_code'],
 decision=d['decision'] if d else None,loss=d['loss'] if d else None),indent=2))
