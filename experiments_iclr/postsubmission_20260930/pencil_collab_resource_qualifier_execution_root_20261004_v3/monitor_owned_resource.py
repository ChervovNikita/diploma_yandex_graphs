"""Read only the exact owned PENCIL v3 execution's identity and resource receipts."""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pencil_v3_owned_resource_client', HERE / 'stage_admit_resource.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def monitor_code(expected):
    code = driver.authenticated_remote(expected['binding']) + 'expected=' + repr(expected) + '\n'
    code += '''assert expected['source_manifest_sha256']=='''+repr(driver.SOURCE_MANIFEST)+'''
assert json.loads((root/'DETACHED_LAUNCH.json').read_text())==expected
assert sha(root/'ROOT_RELEASE.json')==expected['release_sha256']
proc=Path('/proc')/str(expected['supervisor_PID']);identity=None
try:
 raw=(proc/'stat').read_text();f=raw[raw.rfind(')')+2:].split()
 assert int(f[19])==expected['supervisor_start_ticks']
 assert int(f[2])==expected['supervisor_group']==expected['supervisor_PID']
 assert int(f[3])==expected['supervisor_session']==expected['supervisor_PID']
 argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]
 assert argv==expected['command'] or f[0]=='Z'
 identity=dict(pid=expected['supervisor_PID'],start_ticks=int(f[19]),state=f[0],group=int(f[2]),session=int(f[3]))
except FileNotFoundError:
 assert not proc.exists()
files=[];large=[];values={};descriptors={};total=0
def observe(relative, fetch=True):
 global total
 p=root/relative
 if not p.exists():return None
 assert p.resolve().is_relative_to(root) and p.is_file() and not p.is_symlink()
 size=p.stat().st_size;assert size<=64*1024**2
 row=dict(path=relative,bytes=size,sha256=sha(p));descriptors[relative]=row
 if fetch and size<2_000_000 and total+size<=8*1024**2:
  raw=p.read_bytes();assert len(raw)==size and hashlib.sha256(raw).hexdigest()==row['sha256'];raw.decode('utf8')
  files.append(dict(row,data=base64.b64encode(raw).decode()));total+=size
  if p.suffix=='.json':values[relative]=json.loads(raw)
 else:large.append(row)
 return row
for relative in ('supervision/run01/STARTED.json','supervision/run01/LAUNCH.json','supervision/run01/STATUS.json',
                 'supervision/run01/PHYSICAL_TERMINAL.json','supervision/run01/TERMINAL.json',
                 'supervision/run01/SUPERVISOR_CUSTODY.json','run01/PROGRESS.json',
                 'supervision/run01/STDERR.txt','DETACHED_STDERR.txt'):
 observe(relative)
physical=values.get('supervision/run01/PHYSICAL_TERMINAL.json')
terminal=values.get('supervision/run01/TERMINAL.json')
custody=values.get('supervision/run01/SUPERVISOR_CUSTODY.json')
progress=values.get('run01/PROGRESS.json');resource=None;adoption=False;conditions=[]
# Child resource/final custody are collected only after physical evidence exists.
if physical is not None and terminal is not None:
 for value in (physical,terminal):
  assert value['source_manifest_sha256']==expected['source_manifest_sha256'] and value['release_sha256']==expected['release_sha256']
 assert terminal['physical_terminal_path']==str(root/'supervision/run01/PHYSICAL_TERMINAL.json')
 assert terminal['physical_terminal_sha256']==descriptors['supervision/run01/PHYSICAL_TERMINAL.json']['sha256']
 for relative in ('run01/RESOURCE.json','run01/FILE_CUSTODY.json','run01/FINAL_CUSTODY.json',
                  'run01/MONITOR_FAILURE.json','run01/FAILURE.json','run01/RESOURCE_STOP.json'):
  observe(relative)
 resource=values.get('run01/RESOURCE.json')
 if terminal['status']=='COMPLETE_RESOURCE_ONLY' and custody is not None and resource is not None:
  assert custody['status']==terminal['status']
  assert custody['terminal_path']==str(root/'supervision/run01/TERMINAL.json')
  assert custody['terminal_sha256']==descriptors['supervision/run01/TERMINAL.json']['sha256']
  assert physical['physical_session_closed'] is True and physical['direct_child_reaped'] is True
  assert physical['physical_exit_code']==0 and physical['stop'] is None
  assert terminal['physical_session_closed'] is True and terminal['direct_child_reaped'] is True
  assert terminal['physical_exit_code']==0 and terminal['stop'] is None
  budget=terminal['final_parent_budget_check']
  assert budget['status']=='PASS' and budget['last_completed_check']=='after_receipt_and_custody_publication'
  assert custody['final_parent_budget_check']==budget
  assert terminal['linked']['status']=='COLLECTED_COMPLETE_RESOURCE_ONLY'
  assert resource['status']=='COMPLETE_RESOURCE_ONLY' and resource['source_manifest_sha256']==expected['source_manifest_sha256']
  assert resource['release_sha256']==expected['release_sha256'] and resource['workload']==json.loads((root/'ROOT_RELEASE.json').read_text())['workload']
  assert resource['completed_native_epochs']==1 and resource['VALID_queries']==160084 and resource['VALID_batches']==157
  assert resource['VALID_sequential_indices'] is True and resource['all_observed_outputs_finite'] is True
  assert all(resource[k] is False for k in ('TEST_reads','predictive_metrics_computed','scores_saved','checkpoint_saved','state_donor','automatic_retry'))
  final=values['run01/FINAL_CUSTODY.json'];assert final['completed'] is True
  final_custody=values['run01/FILE_CUSTODY.json']
  assert final_custody['status']=='MATCH' and final_custody['source_manifest_sha256']==expected['source_manifest_sha256']
  assert final_custody['release_sha256']==expected['release_sha256']
  assert {r['path'] for r in final['files']}=={'PROGRESS.json','RESOURCE.json','FILE_CUSTODY.json'}
  assert {r['path'] for r in custody['child_output_files']}=={'PROGRESS.json','RESOURCE.json','FILE_CUSTODY.json','FINAL_CUSTODY.json'}
  assert final['file_custody_sha256']==descriptors['run01/FILE_CUSTODY.json']['sha256']
  for row in final['files']:
   actual=descriptors['run01/'+row['path']];assert actual['bytes']==row['bytes'] and actual['sha256']==row['sha256']
  for key,folder in (('child_output_files',root/'run01'),('supervisor_output_files',root/'supervision/run01')):
   for row in custody[key]:
    relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
    p=folder/relative;assert p.resolve().is_relative_to(folder) and p.is_file() and not p.is_symlink()
    assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
  assert custody['root_lock_sha256']==sha(root/'RESOURCE_ATTEMPT_SPENT.json')
  for name,key in (('RESOURCE.json','result_sha256'),('FINAL_CUSTODY.json','final_custody_sha256'),('FILE_CUSTODY.json','file_custody_sha256')):
   assert terminal['linked'][key]==descriptors['run01/'+name]['sha256']
  if identity is None or identity['state']=='Z':adoption=True
  else:conditions.append('Original supervisor still live; wait for physical supervisor exit before root adoption.')
elif terminal is not None:
 conditions.append('Terminal without matching physical receipt is incomplete.')
if not adoption and not conditions:conditions.append('Owned attempt is pending, failed, or incomplete; no automatic retry or resource adoption.')
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),supervisor_identity=identity,
 files=files,large_server_retained_metadata=large,progress=progress,resource=resource,
 terminal=terminal,physical_terminal=physical,adoption_eligible=adoption,conditions=conditions,
 signals_sent=False,retries=False,metadata_only=True,TEST_reads=False,state_transfer=False)))
'''
    # Publication is finite but not atomic across physical/terminal/custody
    # receipts. A partial snapshot remains observable and cannot be adopted.
    marker = '# Child resource/final custody are collected only after physical evidence exists.\n'
    prefix, validation = code.split(marker, 1)
    validation, publication = validation.split('print(json.dumps(dict(UTC=', 1)
    return (prefix + marker + 'try:\n' + '\n'.join(' ' + line for line in validation.splitlines())
            + '\nexcept (AssertionError,KeyError,TypeError,ValueError) as error:\n'
            + ' adoption=False;conditions.append("Receipt snapshot incomplete, changing, or different: "+type(error).__name__+": "+str(error))\n'
            + 'print(json.dumps(dict(UTC=' + publication)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=int, required=True)
    args = parser.parse_args()
    driver.require(1 <= args.sequence <= 99, 'Bounded unique monitor sequence required')
    expected = json.loads((HERE / 'DETACHED_LAUNCH.json').read_text())
    out = HERE / ('owned_monitor%02d' % args.sequence)
    out.mkdir()
    value = driver.transport('monitor%02d' % args.sequence, monitor_code(expected))
    rows = value.pop('files')
    value['fetched_descriptors'] = []
    for row in rows:
        raw = base64.b64decode(row.pop('data'), validate=True)
        driver.require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'Fetched receipt differs')
        path = out / row['path']
        driver.require(path.resolve().is_relative_to(out), 'Fetched path escaped observation')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
        value['fetched_descriptors'].append(row)
    with (out / 'OBSERVATION.json').open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    terminal = value['terminal']
    summary = dict(UTC=value['UTC'],supervisor_identity=value['supervisor_identity'],progress=value['progress'],
                   resource=value['resource'],terminal_status=terminal.get('status') if terminal else None,
                   physical_session_closed=terminal.get('physical_session_closed') if terminal else None,
                   physical_exit_code=terminal.get('physical_exit_code') if terminal else None,
                   adoption_eligible=value['adoption_eligible'],conditions=value['conditions'],
                   resource_only=True,predictive_quality_or_20_epoch_fit_qualified=False)
    with (out / 'SUMMARY.json').open('x') as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:summary[k] for k in ('UTC','supervisor_identity','progress','terminal_status','adoption_eligible','conditions')},indent=2))


if __name__ == '__main__':
    main()
