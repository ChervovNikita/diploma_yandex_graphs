"""Detached finite two-case engineering queue; stop on every failed case."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
CASE_IDS = ['559839ac8be8ae7769914054', '17ca2eaf16582d719c4df34a']


def guard():
    assert socket.gethostname() == 'anogena-2-0', 'Excluded route; project unopened'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],
                                   text=True,timeout=10).splitlines() == [GPU], 'Excluded GPU route; project unopened'


def identity(pid):
    try: fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError: return None
    return {'pid':pid,'pgid':int(fields[2]),'start_ticks':int(fields[19]),'state':fields[0]}


def same(actual, expected):
    return actual is not None and all(actual[k] == expected[k] for k in ('pid','pgid','start_ticks'))


def receipt(root,path):
    raw=path.read_bytes()
    return {'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def bound(root,row):
    relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
    path=root/relative
    assert not path.is_symlink() and path.resolve().is_relative_to(root) and receipt(root,path)==row
    return path


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.temporary')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(path)


def module(root,row,name):
    path=bound(root,row);spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def no_other_engineering(allowed=()):
    names={'practical_zero_update_resources.py','practical_modal_zero_update_resources.py',
           'practical_resource_guardian.py','practical_modal_resource_minimized_guardian.py',
           'staged_random_resource_helpers.py','staged_resource_review_runner.py',
           'private_source_zero_update_resources.py','private_source_resource_queue.py',
           'coauthor_resource_small_queue.py','route_recompute_queue.py'}
    modules={'qa.'+name[:-3] for name in names}
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) in {os.getpid(),*allowed}:continue
        try:tokens=(proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        for raw in tokens:
            token=raw.decode(errors='replace')
            if token.rsplit('/',1)[-1] in names or token in modules or token.endswith('/coauthor_resource_small_queue_20261012_v2/queue.py') or token.endswith('/coauthor_resource_larger_reservation_queue_20261012_v1/queue.py'):
                found.append(identity(int(proc.name)));break
    assert not found, 'Other engineering owner/child present: '+json.dumps(found)


def gpu_usage():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
                                 '--format=csv,noheader,nounits'],text=True,timeout=10)
    rows=[]
    for line in raw.splitlines():
        parts=[p.strip() for p in line.split(',')]
        if len(parts)==3:
            assert parts[1]==GPU
            rows.append({'pid':int(parts[0]),'gpu_uuid':parts[1],'used_memory_MiB':int(parts[2])})
    return rows


def preflight(root,decision,allowed=()):
    guard();no_other_engineering(allowed)
    h=module(root,decision['Physics_custody_helper'],'reviewed_route_Physics_custody')
    custody=h.physics_custody(root,decision['Physics_owner_start'],decision['Physics_original_watcher'])
    permitted=set()
    for expected in [*h.PHYSICS.values(),h.READOUT_WATCHER]:
        current=identity(expected['pid'])
        if same(current,expected) and current['state']!='Z':permitted.add(current['pid'])
    usage=gpu_usage()
    assert all(row['pid'] in permitted for row in usage), 'Unrelated CUDA owner present'
    closure_path=bound(root,decision['prior_larger_queue_closure']);closure=json.loads(closure_path.read_text())
    assert closure['status']=='CLOSED' and closure['passed']==0 and not same(identity(closure['owner']['pid']),closure['owner'])
    for case in closure['cases']:
        failure_path=root/case['result'];raw=failure_path.read_bytes()
        assert {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==case['result_receipt']
        failure=json.loads(raw);assert not same(identity(failure['owner']['pid']),failure['owner'])
    return {'Physics_custody':custody,'GPU_process_usage':usage,'no_other_engineering':True,
            'prior_75pct_owned_processes_absent':True}


def verify_success(value,case,birth):
    assert value['status']=='TRAIN_PROXY_COMPLETE' and value['case_id']==case['case_id'] and same(value['owner'],birth)
    assert all(value[k]==0 for k in ('actual_optimizer_updates','actual_optimizer_kernel_calls',
                                     'heldout_original_file_accesses','scores_emitted','accuracy_calls','scientific_fits'))
    for key in ('parameters','moments'):
        assert value[key+'_sha256_before']==value[key+'_sha256_after']
    assert value['allocator_cap']['maximum_fraction_of_total']==.4
    assert value['denied_optimizer_step_calls']==4
    expected=8 if case['run']['config']['train_views']==2 else 4
    assert len(value['route_recomputation'])==4
    for cycle in value['route_recomputation']:
        work=cycle['work'];counts=work['counts']
        assert cycle['completed'] and cycle['wrapper_restored'] and cycle['failure_type'] is None
        assert counts['logical_checkpoint_route_calls']==counts['completed_logical_route_forwards']==expected
        assert counts['recomputed_route_forward_attempts']==counts['completed_recomputed_route_forwards']==expected
        assert counts['bypassed_route_calls']==counts['failed_route_invocations']==0
        assert work['optimizer_actions']==work['token_bank_copies_created']==0


def terminate(child,birth):
    if child.poll() is not None:return {'TERM':False,'KILL':False,'reaped':True}
    assert same(identity(child.pid),birth),'Owned child identity changed; no signal sent'
    os.killpg(birth['pgid'],signal.SIGTERM);killed=False
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        assert same(identity(child.pid),birth),'Owned child identity changed; no KILL sent'
        os.killpg(birth['pgid'],signal.SIGKILL);killed=True;child.wait(timeout=5)
    return {'TERM':True,'KILL':killed,'reaped':child.poll() is not None}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config');args=p.parse_args()
    guard()
    config_path=Path(args.config).resolve();config=json.loads(config_path.read_text());root=Path(config['project_root']).resolve()
    assert config_path.is_relative_to(root)
    decision=json.loads(bound(root,config['decision']).read_text())
    assert decision['case_ids']==CASE_IDS and decision['failure_stops_before_next_execution'] is True
    assert config['helper']==decision['helper'] and config['manifest']==decision['manifest']
    helper=bound(root,config['helper']);manifest=json.loads(bound(root,config['manifest']).read_text())
    bound(root,decision['utility']);bound(root,decision['CPU_source_equivalence'])
    assert manifest['helper_sha256']==config['helper']['sha256'] and [c['case_id'] for c in manifest['cases']]==CASE_IDS
    runtime=root/decision['runtime'];assert runtime.is_dir() and runtime.resolve().is_relative_to(root)
    for case in manifest['cases']:
        study=json.loads(gzip.decompress(bound(root,case['study_binding']).read_bytes()))
        for row in study['source']['files']:
            if row['path'].endswith('.py'):bound(root,row)
    output=root/config['output'];assert output.resolve().is_relative_to(root) and not output.exists();output.mkdir()
    owner=identity(os.getpid());started=time.monotonic();closed=[];stop=None
    save(output/'OWNER_START.json',{'owner':owner,'config':receipt(root,config_path),'decision':config['decision'],
                                  'scientific_fits':0,'GO':False})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',
             MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(runtime))
    try:
        imports=subprocess.run([config['python'],'-B','-c','import prepare_node_data,run_study,engine'],
                               cwd=runtime,env=dict(env,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=60)
        save(output/'CPU_IMPORT_PREFLIGHT.json',{'exit_code':imports.returncode,'stdout':imports.stdout,'stderr':imports.stderr,
                                              'CUDA_executions':0,'scientific_fits':0})
        assert imports.returncode==0,'CPU import failed before CUDA child launch'
        for index,case in enumerate(manifest['cases'],1):
            assert time.monotonic()-started < 570,'Finite queue active ceiling reached'
            before=preflight(root,decision)
            destination=output/('case_%02d_%s'%(index,case['case_id']))
            argv=[config['python'],'-B',str(helper),'--project-root',str(root),'--output',str(destination.relative_to(root)),
                  '--mode','execute-node-case','--manifest',config['manifest']['path'],'--case-id',case['case_id'],
                  '--reviewed-helper-sha256',config['helper']['sha256'],'--warm-iterations','3','--case-timeout-seconds','240']
            save(output/('CASE_%02d_REQUEST.json'%index),{'case_id':case['case_id'],'command':argv,'custody_before':before,'no_retry':True})
            tick=time.monotonic();timeout=False;birth=None
            with (output/('case_%02d_stdout.log'%index)).open('xb') as stdout,(output/('case_%02d_stderr.log'%index)).open('xb') as stderr:
                child=subprocess.Popen(argv,cwd=runtime,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                       start_new_session=True,close_fds=True)
                birth=identity(child.pid);assert birth is not None and birth['pgid']==birth['pid']
                save(output/('CASE_%02d_START.json'%index),{'case_id':case['case_id'],'child':birth,'command':argv})
                try:
                    child.wait(timeout=250)
                except subprocess.TimeoutExpired:
                    timeout=True
                finally:
                    cleanup=terminate(child,birth)
            assert not same(identity(child.pid),birth) and child.pid not in {r['pid'] for r in gpu_usage()}
            result_path=destination/'RESULT.json'
            if not result_path.exists():result_path=destination/'TERMINAL_FAILURE.json'
            value=json.loads(result_path.read_text()) if result_path.exists() else None
            row={'case_id':case['case_id'],'child':birth,'exit_code':child.returncode,'timeout':timeout,
                 'cleanup':cleanup,'process_wall_seconds':time.monotonic()-tick,
                 'status':value['status'] if value else 'MISSING_TERMINAL_RESOURCE_RECEIPT',
                 'result':receipt(root,result_path) if value else None,
                 'owner_identity_absent':True,'owned_CUDA_PID_absent':True}
            closed.append(row);save(output/('CASE_%02d_PROCESS.json'%index),row)
            if child.returncode==0 and not timeout and value is not None:
                verify_success(value,case,birth)
            else:
                stop='Case failure retained; root report required before any next execution'
            after=preflight(root,decision)
            save(output/('CASE_%02d_CUSTODY_AFTER.json'%index),after)
            save(output/'PROGRESS.json',{'completed':len(closed),'expected':2,'cases':closed,'stop_reason':stop})
            if stop:break
    except BaseException as error:
        stop=type(error).__name__+': '+str(error)
    finally:
        save(output/'OWNER_END.json',{'status':'CLOSED' if stop is None and len(closed)==2 else 'HALTED_RETAINED',
          'owner':owner,'cases':closed,'unstarted_case_ids':[c for c in CASE_IDS if c not in {r['case_id'] for r in closed}],
          'passed':sum(r['status']=='TRAIN_PROXY_COMPLETE' and r['exit_code']==0 for r in closed),
          'stop_reason':stop,'elapsed_wall_seconds':time.monotonic()-started,'scientific_fits':0,'GO':False,
          'automatic_retries':0,'two_slot_concurrency_qualified':False})


if __name__=='__main__':main()
