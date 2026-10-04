"""Observe exact queue/supervisor/child handles and compact owned receipt metadata."""
from pathlib import Path
import argparse
import base64
import hashlib
import json
from stage_admit_launch import (HERE,ROOT,REMOTE,REMOTE_CLIENT,SOURCE_SHA,context,
                                sha,save,transport,reviewed_local)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',type=int,required=True)
    args=parser.parse_args();assert 1<=args.sequence<=999
    plan,_=reviewed_local()
    assert (ROOT/'LOCAL_LAUNCH_ATTEMPT_SPENT.json').is_file()
    queue_sha=sha(ROOT/'QUEUE_RELEASE.json')
    expected=json.loads((ROOT/'DETACHED_LAUNCH.json').read_text()) if (ROOT/'DETACHED_LAUNCH.json').exists() else None
    out=ROOT/('owned_monitor%03d'%args.sequence);out.mkdir(exist_ok=False)
    code=context()+'import sys\nsys.path.insert(0,'+repr(str(REMOTE_CLIENT))+')\n'
    code+='assert hashlib.sha256((Path('+repr(str(REMOTE_CLIENT))+')/"remote_control.py").read_bytes()).hexdigest()=='+repr(sha(HERE/'remote_control.py'))+'\n'
    code+='from remote_control import bound_queue,physical\nq,source_plan=bound_queue('+repr(queue_sha)+')\nexpected='+repr(expected)+'\n'
    code+='''
launch_path=root/'DETACHED_LAUNCH.json'
launch=json.loads(launch_path.read_text()) if launch_path.is_file() else None
started_path=root/'queue/run01/STARTED.json'
queue_started=json.loads(started_path.read_text()) if started_path.is_file() else None
if launch is not None:
 assert launch['source_manifest_sha256']==q['source_manifest_sha256'] and launch['client_manifest_sha256']==q['client_manifest_sha256']
 assert launch['queue_release_sha256']==hashlib.sha256((root/'QUEUE_RELEASE.json').read_bytes()).hexdigest()
 if expected is not None:assert launch==expected
handles=[]
def observe(label,saved):
 if saved is None:return
 actual=physical(saved['PID'])
 if actual is not None:
  assert actual['start_time_ticks']==saved['start_time_ticks']
  if actual['state']!='Z':
   assert actual['argv']==saved['argv'] and actual['cwd']==saved['cwd'] and actual['exe']==saved['exe']
  assert actual['process_group']==saved['process_group'] and actual['session']==saved['session']
 handles.append(dict(label=label,expected=saved,actual=actual))
queue_identity=launch['queue_physical_identity'] if launch else (queue_started['physical_identity'] if queue_started else None)
observe('queue',queue_identity)
relative_files=['SOURCE_STAGE_RECEIPT.json','PRENUMERICAL_ADMISSION.json','ROOT_ADMISSION.json',
 'QUEUE_RELEASE.json','LAUNCH_ATTEMPT_SPENT.json','QUEUE_ATTEMPT_SPENT.json','DETACHED_LAUNCH.json',
 'queue/run01/STARTED.json','queue/run01/STATUS.json','queue/run01/TERMINAL.json']
progress=[];terminals=[]
for cell in q['queue_order']:
 name=cell['cell']; fit=root/name/'run01'; sup=root/'supervision'/name/'run01'
 queue_child=root/'queue/run01'/(name+'_STARTED.json')
 if queue_child.is_file():
  own=json.loads(queue_child.read_text());assert own['cell']==name
  observe(name+'_supervisor',own['physical_identity'])
 child_path=sup/'CHILD_STARTED.json'
 if child_path.is_file():
  child=json.loads(child_path.read_text());observe(name+'_fit_child',child['physical_identity'])
 for tail in ('STARTED.json','TERMINAL.json'):
  relative_files.append('queue/run01/'+name+'_'+tail)
 for filename in ('ATTEMPTS.json','STATUS.json','FAILED.json','COMPLETE.json','RUNTIME_PROFILE_TRANSITION.json'):
  relative_files.append(name+'/run01/'+filename)
 for filename in ('SUPERVISOR_STARTED.json','CHILD_STARTED.json','SUPERVISOR_STATUS.json',
                  'CHILD_CUDA_PEAKS.json','SUPERVISOR_FAILURE.json','SUPERVISOR_TERMINAL.json'):
  relative_files.append('supervision/'+name+'/run01/'+filename)
 status_path=fit/'STATUS.json'
 if status_path.is_file():
  a=json.loads(status_path.read_text());assert a['predictive_values_exposed'] is False
  assert a['unit']==cell['invocation']['unit'] and a['base_seed']==cell['invocation']['base_seed']
  progress.append(dict(cell=name,**{k:a[k] for k in ('status','phase','work','observed_wall_seconds')}))
 terminal=sup/'SUPERVISOR_TERMINAL.json'
 if terminal.is_file():
  t=json.loads(terminal.read_text())
  terminals.append(dict(cell=name,status=t['status'],child_exit_code=t['child_exit_code'],cap_violation=t['cap_violation']))
files=[];retained=[]
for relative in relative_files:
 p=root/relative
 if not p.is_file():continue
 assert not p.is_symlink() and p.resolve().is_relative_to(root) and p.stat().st_size<=16*1024**2
 raw=p.read_bytes();raw.decode('utf8')
 row=dict(path=relative,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 if len(raw)<2_000_000:files.append(dict(row,data=base64.b64encode(raw).decode()))
 else:retained.append(row)
stderr_tails=[]
for relative in ['DETACHED_STDERR.txt']+['queue/run01/'+c['cell']+'.stderr.txt' for c in q['queue_order']]+['supervision/'+c['cell']+'/run01/CHILD.stderr.log' for c in q['queue_order']]:
 p=root/relative
 if p.is_file():
  assert not p.is_symlink() and p.resolve().is_relative_to(root)
  with p.open('rb') as handle:
   handle.seek(max(0,p.stat().st_size-8192));raw=handle.read(8192)
  stderr_tails.append(dict(path=relative,tail=raw.decode('utf8',errors='replace'),file_bytes=p.stat().st_size))
queue_terminal=root/'queue/run01/TERMINAL.json'
terminal=json.loads(queue_terminal.read_text()) if queue_terminal.is_file() else None
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),queue_launch=launch,queue_started=queue_started,
 handles=handles,progress=progress,fit_terminals=terminals,queue_terminal=terminal,files=files,
 large_server_retained_metadata=retained,stderr_tails=stderr_tails,signals_sent=False,retries=False,
 private_selection_or_logits_read=False,TEST_access=False,metadata_only=True)))
'''
    result=transport('monitor%03d'%args.sequence,code)
    rows=result.pop('files');result['fetched_descriptors']=[]
    for row in rows:
        raw=base64.b64decode(row.pop('data'),validate=True)
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        path=out/row['path'];assert path.resolve().is_relative_to(out)
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as handle:handle.write(raw)
        result['fetched_descriptors'].append(row)
    save(out/'OBSERVATION.json',result)
    terminal=result['queue_terminal']
    summary={'UTC':result['UTC'],'queue_status':terminal['status'] if terminal else 'IN_PROGRESS_OR_LAUNCH_OUTCOME_UNKNOWN',
             'queue_handle':result['handles'][0] if result['handles'] else None,
             'progress':result['progress'],'fit_terminals':result['fit_terminals'],
             'queue_failure':terminal['failure'] if terminal else None,
             'unattempted_cells':terminal['unattempted_cells'] if terminal else None,
             'private_selection_or_logits_read':False,'signals_sent':False,'automatic_retry_or_resume':False}
    save(out/'SUMMARY.json',summary)
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':main()
