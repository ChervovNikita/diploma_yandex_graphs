"""Observe the exact original zero-update handle and compact scalar comparisons."""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pubmed_zero_update_owned_driver',HERE/'stage_admit_control.py')
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
names=['first_batch_repeat/run01/DIAGNOSTIC.json','first_batch_repeat/run01/FAILURE.json',
 'first_batch_repeat/run01/PROGRESS.json','first_batch_repeat/run01/REPLAY_PROGRESS.json',
 'first_batch_repeat/run01/TRAIN_INSPECTION.json','first_batch_repeat/run01/CUDA_PEAKS.json',
 'first_batch_repeat/run01/FINAL_CUSTODY.json','supervision/first_batch_repeat/run01/STARTED.json',
 'supervision/first_batch_repeat/run01/PHYSICAL_TERMINAL.json',
 'supervision/first_batch_repeat/run01/TERMINAL.json',
 'supervision/first_batch_repeat/run01/SUPERVISOR_CUSTODY.json',
 'supervision/first_batch_repeat/run01/STDERR.txt','DETACHED_STDERR.txt']
for relative in names:
 p=root/relative
 if not p.is_file():continue
 assert not p.is_symlink() and p.resolve().is_relative_to(root) and p.stat().st_size<=64*1024**2
 b=p.read_bytes();b.decode('utf8');row=dict(path=relative,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 if len(b)<2_000_000:files.append(dict(row,data=base64.b64encode(b).decode()))
 else:large.append(row)
 if relative=='first_batch_repeat/run01/PROGRESS.json':progress=json.loads(b)
 if relative=='supervision/first_batch_repeat/run01/TERMINAL.json':terminal=json.loads(b)
 if relative=='first_batch_repeat/run01/DIAGNOSTIC.json':
  d=json.loads(b);e=d['diagnostic_evidence']
  diagnostic={k:d[k] for k in ('status','source_manifest_sha256','root_release_sha256','progress',
   'owned_engineering_state_sha256','owned_serialized_identity_sha256','inclusive_child_wall_seconds',
   'engineering_qualification_PASS','scientific_fit_admitted','state_donor_allowed',
   'VALID_files_opened','TEST_files_opened','score_files_opened','numeric_rule_changed',
   'restore_helper_changed','kernel_causality_established','interpretation')}
  diagnostic['all_four_state_matching_valid']=e['all_four_state_matching_valid']
  diagnostic['saved_tree_unchanged']=e['saved_tree_unchanged']
  diagnostic['initial_restores_exact']={k:v['exact_native_checkpoint_state']['exact'] for k,v in e['initial_restores'].items()}
  diagnostic['prefixes']={k:{a:v[a] for a in ('pre_forward_state_sha256','pre_forward_gradients_None',
   'pre_forward_all_training_flags_true','unregistered_module_state_fully_inspected','unsupported_module_state',
   'saved_tree_unchanged','channel_identities')} for k,v in e['prefixes'].items()}
  diagnostic['comparisons']=[{k:r[k] for k in ('label','candidate','reference','state_matching_valid',
   'numerical_comparison_collected','status','native_streams_exact','initial_restores_exact',
   'saved_and_idle_exact','saved_and_cross_unit_step_alias_isolation_exact','original_fixed_comparator_verdicts')}
   |dict(complete_pre_forward_state=r['complete_pre_forward_state'],
         original_fixed_comparator_reports=r['original_fixed_comparator_reports']) for r in e['comparisons']]
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
 all_four_state_matching_valid=d['all_four_state_matching_valid'] if d else None,
 comparisons=[dict(label=r['label'],status=r['status'],
   passed_channels=[v['channel'] for v in r['original_fixed_comparator_verdicts'] if v['passed_original_fixed_predicate']],
   failed_channels=[v['channel'] for v in r['original_fixed_comparator_verdicts'] if not v['passed_original_fixed_predicate']])
   for r in d['comparisons']] if d else None),indent=2))
