"""Prepare reviewed one-case commands; run only after separate exact review.

Default mode never imports Torch, launches a child or executes CUDA. Root owns
serialization across engineering queues. One external child is bounded600s,
with570s internal helper deadline and owned-PID cleanup reserve. No retries.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def route():
    if os.uname().nodename!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=[UUID]: raise ValueError('Excluded allocation route; project unopened')
def safe(root,rel):
    value=Path(rel); path=root/value
    if value.is_absolute() or '..' in value.parts or path.is_symlink() or not path.resolve().is_relative_to(root): raise ValueError('Confined project-relative path required')
    return path
def receipt(root,path):
    raw=path.read_bytes(); return {'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def bound(root,row):
    path=safe(root,row['path'])
    if receipt(root,path)!=row: raise ValueError('Reviewed source/manifest changed')
    return path
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(value,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n')
def module(path):
    spec=importlib.util.spec_from_file_location('reviewed_staged_source_helper',path); value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value
def identity(pid):
    p=Path('/proc')/str(pid)/'stat'
    if not p.exists(): return None
    a=p.read_text().rsplit(')',1)[1].split(); return {'pid':pid,'start_ticks':int(a[19]),'pgid':int(a[2]),'state':a[0]}
def same(current,birth): return current is not None and all(current.get(k)==birth[k] for k in ('pid','start_ticks','pgid'))


def command(root,roster,case_id,output):
    helper=bound(root,roster['helper']); bound(root,roster['manifest']); runtime=helper.parents[1]
    argv=[str(root.parents[1]/'.venv/bin/python'),'-B','-m','qa.staged_random_resource_helpers','--project-root',str(root),
        '--mode','execute-case','--manifest',roster['manifest']['path'],'--case-id',case_id,'--reviewed-helper-sha256',roster['helper']['sha256'],
        '--reviewed-manifest-sha256',roster['manifest']['sha256'],'--case-timeout-seconds','570','--output',str(output.relative_to(root))]
    return argv,runtime


def prepare_commands(root,roster,output):
    rows=[]
    for index,case in enumerate(roster['cases'],1):
        case_output=output/('case_'+str(index).zfill(2)+'_'+case['case_id']); argv,cwd=command(root,roster,case['case_id'],case_output)
        value={'schema':'bounded-staged-case-command-preparation-v1','case_id':case['case_id'],'command':argv,'cwd':str(cwd),
            'internal_helper_deadline_seconds':570,'external_owned_child_deadline_seconds':600,'no_retry':True,'execution_authorized':False,'executed':False,
            'scientific_fits':0,'CUDA_executions':0,'root_must_serialize_all_engineering_queues':True,'expected_helper':roster['helper'],'expected_manifest':roster['manifest']}
        path=output/('case_'+str(index).zfill(2)+'_COMMAND_PREPARED.json'); dump(path,value); rows.append(receipt(root,path))
    dump(output/'COMMANDS_PREPARED.json',{'case_commands':rows,'cases':len(rows),'all_unexecuted':True,'CUDA_executions':0,'jobs_started':0,'scientific_fits':0,'GO':False})
    return rows


def verify_result(result,case_id,helper_sha):
    if result.get('status')!='RANDOM_SOURCE_PROXY_COMPLETE' or result.get('case_id')!=case_id or result.get('helper_sha256')!=helper_sha: raise ValueError('Child result authority mismatch')
    for k in ('actual_optimizer_updates','actual_optimizer_kernel_calls','heldout_original_file_accesses','scores_emitted','accuracy_calls','scientific_fits'):
        if result.get(k)!=0: raise ValueError('Child violated no-fit/no-role/no-score constraint: '+k)
    for key in ('parameters','moments'):
        if result.get(key+'_sha256_before') is None or result[key+'_sha256_before']!=result.get(key+'_sha256_after'): raise ValueError('Child fixed-state equality missing')
    export=result.get('TRAIN_only_sequential_export',{})
    if export.get('scientific_parent_receipt') is not None or any(export.get(k)!=0 for k in ('accuracy_calls','scores_emitted','SELECT_CAL_evaluate_accesses')) or export.get('original_targets_or_labels_serialized') is not False or export.get('TRAIN_only_disposable_logit_serialization',{}).get('removed') is not True: raise ValueError('Sequential TRAIN export constraint failed')
    for key in ('parameters','moments'):
        if export.get(key+'_sha256_before') is None or export[key+'_sha256_before']!=export.get(key+'_sha256_after'): raise ValueError('Export changed fixed state')


def run_one(root,roster,case_id,output):
    matches=[r for r in roster['cases'] if r['case_id']==case_id]
    if len(matches)!=1: raise ValueError('One exact finite representative case required')
    helper=module(bound(root,roster['helper'])); manifest=json.loads(bound(root,roster['manifest']).read_text())
    custody=helper.physics_custody(root,manifest['Physics_owner_start'],manifest['Physics_original_watcher'])
    argv,cwd=command(root,roster,case_id,output/'child'); owner=identity(os.getpid())
    dump(output/'COMMAND_EXECUTION_REQUEST.json',{'command':argv,'cwd':str(cwd),'case_id':case_id,'root_runner_owner':owner,'physics_custody_before':custody,'no_retry':True})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1'); env.pop('PYTHONPATH',None); env.pop('PYTHONHOME',None)
    started=time.monotonic(); deadline=started+600; child=None; birth=None; failure=None; code=None; killed=False
    try:
        with (output/'engineering_child.log').open('xb') as log:
            child=subprocess.Popen(argv,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True)
            for _ in range(20):
                birth=identity(child.pid)
                if birth is not None or child.poll() is not None: break
                time.sleep(.01)
            if birth is None or birth['pgid']!=child.pid or birth['pid'] in (640426,640428,641081): raise ValueError('Fresh owned engineering child identity unavailable')
            dump(output/'OWNER_START.json',{'owner':owner,'child':birth,'case_id':case_id,'command':argv,'deadline_seconds':600,'no_retry':True})
            try: code=child.wait(timeout=max(.1,deadline-time.monotonic()-20))
            except subprocess.TimeoutExpired:
                failure='external_owned_child_deadline'; killed=True
                if same(identity(child.pid),birth): os.killpg(child.pid,signal.SIGTERM)
                try: code=child.wait(timeout=max(.1,min(10,deadline-time.monotonic()-10)))
                except subprocess.TimeoutExpired:
                    if same(identity(child.pid),birth): os.killpg(child.pid,signal.SIGKILL)
                    code=child.wait(timeout=max(.1,deadline-time.monotonic()))
        child_result=output/'child/RESULT.json'; failed=output/'child/TERMINAL_FAILURE.json'
        if code==0 and child_result.exists():
            result=json.loads(child_result.read_text()); verify_result(result,case_id,roster['helper']['sha256'])
            if matches[0]['arm']=='GF1' and matches[0]['prototype_bodies']==4:
                export=result['TRAIN_only_sequential_export']
                if export['prototype_bodies']!=4 or export['served_views']!=16 or export['maximum_simultaneously_live_full_graph_views']!=4 or export['full_graph_body_forwards']!=4: raise ValueError('Actual GFI16 sequential four-own/four-view export mismatch')
        else: failure=failure or ('child_failure_receipt' if failed.exists() else 'child_exit_without_complete_receipt')
    except BaseException as error: failure=failure or type(error).__name__
    finally:
        if child is not None and child.poll() is None:
            if birth is not None and same(identity(child.pid),birth): os.killpg(child.pid,signal.SIGKILL); killed=True
            try: code=child.wait(timeout=max(.1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired: failure=failure or 'owned_child_reap_timeout'
        absence=child is None or (birth is not None and not same(identity(child.pid),birth)) or (birth is None and child.poll() is not None)
        lifecycle_after=None
        try: lifecycle_after=helper.physics_custody(root,manifest['Physics_owner_start'],manifest['Physics_original_watcher'])
        except BaseException as error: failure=failure or ('Physics_lifecycle_'+type(error).__name__)
        dump(output/'OWNER_END.json',{'owner':owner,'child':birth,'case_id':case_id,'exit_code':code,'status':'ENGINEERING_CASE_CLOSED' if failure is None and absence else 'FAILED_RETAINED',
            'failure':failure,'direct_child_wait':child is not None,'old_owned_child_identity_absent':absence,'owned_child_killed':killed,'elapsed_wall_seconds':time.monotonic()-started,
            'physics_custody_after':lifecycle_after,'no_retry':True,'scientific_fits':0,'no_GO_written':True})
    if failure is not None or not absence: raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--project-root',required=True); p.add_argument('--roster',required=True); p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=('prepare-commands','run-one'),default='prepare-commands'); p.add_argument('--case-id'); p.add_argument('--reviewed-runner-sha256'); p.add_argument('--reviewed-roster-sha256'); args=p.parse_args()
    route(); root=Path(args.project_root).resolve(); roster_path=safe(root,args.roster); roster=json.loads(roster_path.read_text()); output=safe(root,args.output); output.mkdir(parents=True,exist_ok=False)
    if args.mode=='prepare-commands': prepare_commands(root,roster,output); print(json.dumps({'prepared_commands':len(roster['cases']),'CUDA_executions':0,'jobs_started':0})); return
    if args.reviewed_runner_sha256!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or args.reviewed_roster_sha256!=hashlib.sha256(roster_path.read_bytes()).hexdigest(): raise ValueError('Separate exact runner/roster review required; nothing launched')
    if not args.case_id: raise ValueError('Root selects one serialized engineering case; no queue launch mode')
    run_one(root,roster,args.case_id,output)


if __name__=='__main__': main()
