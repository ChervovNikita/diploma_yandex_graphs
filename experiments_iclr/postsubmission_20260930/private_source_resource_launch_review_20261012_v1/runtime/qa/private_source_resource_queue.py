"""Prepared sequential12-case engineering queue; no execution without root review.

Stdlib only. The default validates metadata without creating CUDA children.
All source/role work belongs to the separately reviewed zero-update helper.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
CASE_SECONDS=600
CLEANUP_SECONDS=10
QUEUE_SECONDS=7380
QUEUE_ACTIVE_SECONDS=QUEUE_SECONDS-CLEANUP_SECONDS
LOG_BYTES=8*1024**2


def local(root,relative):
    value=Path(relative)
    if value.is_absolute() or '..' in value.parts:raise ValueError('Project-relative authorities only')
    path=root/value
    if path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError('Authority escapes project')
    return path
def receipt(root,path):
    raw=path.read_bytes();return {'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def bound(root,row):
    path=local(root,row['path'])
    if receipt(root,path)!=row:raise ValueError('Exact reviewed authority changed')
    return path
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.temporary')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(path)
def process(pid):
    path=Path('/proc')/str(pid)/'stat'
    try:fields=path.read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return None
    return {'pid':pid,'start_ticks':int(fields[19]),'pgid':int(fields[2]),'state':fields[0]}
def same(actual,expected):return actual is not None and all(actual.get(k)==expected.get(k) for k in ('pid','start_ticks','pgid'))
def own():
    result=process(os.getpid());result.update(hostname=socket.gethostname(),gpu_uuid=GPU);return result
def verify_allocation():
    if socket.gethostname()!='anogena-2-0':raise ValueError('Exact allocation hostname required')
    if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=[GPU]:raise ValueError('Exact sole GPU UUID required')
def module(root,row,name):
    path=bound(root,row);spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def preflight(root,selection):
    manifest=json.loads(bound(root,selection['full_manifest']).read_text())
    if manifest['helper']!=selection['helper'] or manifest['execution_authorized'] is not False:raise ValueError('Exact disabled helper manifest required')
    bound(root,selection['source_review']);bound(root,selection['interception_contract'])
    inventory=json.loads(bound(root,selection['helper_source_inventory']).read_text())
    # Every shared factory/bridge/controller dependency is rehashed, including
    # original v3 runtime paths used by actual factories. No role tensor hash.
    for row in inventory['exact_v3_runtime_copies']:
        bound(root,row['source']);bound(root,row['copy'])
    for row in inventory['owned_files']:bound(root,row)
    for packet in manifest['packets']:
        bound(root,packet['entry']);bound(root,packet['plan'])
    h=module(root,manifest['helper'],'reviewed_private_zero_update')
    physics=module(root,manifest['Physics_custody_helper'],'reviewed_private_Physics_lifecycle')
    custody=physics.physics_custody(root,manifest['Physics_owner_start'],manifest['Physics_original_watcher'])
    molbace=h.closed_molbace_custody(root,manifest['MolBACE_closed_queue']['binding'])
    allowed=set()
    for expected in [*physics.PHYSICS.values(),physics.READOUT_WATCHER]:
        p=process(expected['pid'])
        if same(p,expected) and p['state']!='Z':allowed.add(p['pid'])
    query=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=5)
    usage=[]
    for line in query.splitlines():
        values=[v.strip() for v in line.split(',')]
        if len(values)!=3 or values[1]!=GPU:continue
        pid=int(values[0]);used=int(values[2])*1024**2
        if pid not in allowed:raise ValueError('Unrelated CUDA process/engineering queue present; root must coordinate separately')
        usage.append({'pid':pid,'gpu_uuid':GPU,'used_bytes':used})
    return manifest,{'Physics_custody':custody,'MolBACE_closed_queue':molbace,'allowed_existing_GPU_processes':usage,'unrelated_GPU_processes_allowed':False}


def case_approval(root,selection,case,review_binding):
    review=json.loads(bound(root,review_binding).read_text())
    expected={'schema':'private-source-one-case-resource-review-v1','approved':True,'CUDA_engineering_execution_authorized':True,
        'helper':selection['helper'],'manifest':selection['full_manifest'],'case_id':case['case_id'],'scientific_fits_authorized':False,'MolBACE_queue_overlap_allowed':False}
    if review!=expected:raise ValueError('Exact root-approved engineering-only case receipt required')


def validate_success(value,case,owner):
    if value.get('status')!='RANDOM_SOURCE_ZERO_UPDATE_PROXY_COMPLETE' or value.get('case_id')!=case['case_id'] or not same(value.get('owner'),owner):raise ValueError('Owned exact-case completion identity differs')
    for key in ('actual_optimizer_updates','actual_optimizer_kernel_calls','scores_emitted','accuracy_calls','scientific_fits','heldout_original_file_accesses','denied_tensor_commit_calls'):
        if value.get(key)!=0:raise ValueError('Zero-update/no-role/no-score contract violated')
    if value.get('zero_update_equality_proof_complete') is not True or value.get('historical_CORE_stage2_activated') is not False:raise ValueError('Equality or historical boundary proof absent')
    for key in ('parameters','moments','buffers','immutable_GRAPH_TRAIN_stub','source_owned_state'):
        before=value.get(key+'_sha256_before')
        if not isinstance(before,str) or len(before)!=64 or before!=value.get(key+'_sha256_after'):raise ValueError('Complete source state equality proof differs')
    counts=case['source_operator_counts']
    if value.get('actual_source_operator_counts_by_cycle')!=[counts]*4:raise ValueError('Actual four source cycles differ from exact law')
    owners=counts['intercepted_optimizer_attempts']
    if value.get('optimizer_objects')!=owners or value.get('denied_optimizer_step_calls')!=4*owners or len(value.get('owners',[]))!=owners or any(v['denied_step_calls']!=4 for v in value['owners']):raise ValueError('Every source optimizer owner must be intercepted once each cycle')
    serial=value.get('disposable_serialization',{})
    if serial.get('removed') is not True or serial.get('never_selected_checkpoint') is not True or value.get('all_fit_ETA') is not None:raise ValueError('Disposable/unqualified serializer boundary differs')


def terminate_owned(child,identity):
    """At most10 cleanup seconds; never signal a reused/unrelated PID/group."""
    if child.poll() is not None:return {'TERM':False,'KILL':False,'reaped':True}
    if not same(process(child.pid),identity):raise ValueError('Cannot signal child: exact PID/start/PGID changed')
    os.killpg(identity['pgid'],signal.SIGTERM);killed=False
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if not same(process(child.pid),identity):raise ValueError('Cannot KILL child: exact PID/start/PGID changed')
        os.killpg(identity['pgid'],signal.SIGKILL);killed=True;child.wait(timeout=5)
    return {'TERM':True,'KILL':killed,'reaped':child.poll() is not None}


def run_one(root,selection,case,review_binding,queue,index,custody,deadline):
    case_approval(root,selection,case,review_binding)
    relative=str(queue.relative_to(root))+'/case_'+str(index).zfill(2)+'_'+case['case_id']
    if local(root,relative).exists():raise FileExistsError('Prior case output must be preserved')
    helper=bound(root,selection['helper']);runtime=helper.parents[1]
    command=[str(root.parents[1]/'.venv/bin/python'),'-B',str(helper),'--project-root',str(root),'--physics-helper',selection['Physics_custody_helper']['path'],
        '--mode','execute-case','--manifest',selection['full_manifest']['path'],'--case-id',case['case_id'],'--resource-review',review_binding['path'],
        '--reviewed-helper-sha256',selection['helper']['sha256'],'--reviewed-manifest-sha256',selection['full_manifest']['sha256'],
        '--case-timeout-seconds',str(CASE_SECONDS),'--output',relative]
    record={'case_id':case['case_id'],'command':command,'root_case_review':review_binding,'custody_before':custody,'actual_optimizer_updates_authorized':0,'no_retry':True}
    save(queue/('CASE_'+str(index).zfill(2)+'_COMMAND.json'),record)
    stdout_path=queue/'streams'/('case_'+str(index).zfill(2)+'_stdout.log');stderr_path=stdout_path.with_name('case_'+str(index).zfill(2)+'_stderr.log');stdout_path.parent.mkdir(exist_ok=True)
    child=None;birth=None;reason=None;cleanup={'reaped':True,'TERM':False,'KILL':False};started=time.monotonic()
    try:
        with stdout_path.open('x') as stdout,stderr_path.open('x') as stderr:
            launch_stops=[];previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
            for sig in previous:signal.signal(sig,lambda number,*a:launch_stops.append(number))
            try:
                child=subprocess.Popen(command,cwd=runtime,env=dict(os.environ,PYTHONPATH=str(runtime),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0'),
                    stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
                birth=process(child.pid)
                if birth is None:raise ValueError('Child birth identity unavailable before measurement')
                record['child']=birth;save(queue/('CASE_'+str(index).zfill(2)+'_COMMAND.json'),record)
            finally:
                for sig,handler in previous.items():signal.signal(sig,handler)
            if launch_stops:raise RuntimeError('Owned queue termination/deadline deferred until child birth receipt')
            while child.poll() is None:
                if time.monotonic()-started>=CASE_SECONDS:reason='EXTERNAL_600_SECOND_DEADLINE';break
                if time.monotonic()>=deadline:reason='FINITE_QUEUE_DEADLINE';break
                if stdout_path.stat().st_size>LOG_BYTES or stderr_path.stat().st_size>LOG_BYTES:reason='BOUNDED_LOG_SIZE_EXCEEDED';break
                time.sleep(.5)
            if not reason:child.wait(timeout=1)
            if stdout_path.stat().st_size>LOG_BYTES or stderr_path.stat().st_size>LOG_BYTES:reason='BOUNDED_LOG_SIZE_EXCEEDED'
    finally:
        # A second TERM or the queue alarm must not interrupt the owned TERM/KILL
        # cleanup. Replay a pending stop only after its process receipt exists.
        deferred=[];previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
        for sig in previous:signal.signal(sig,lambda number,*a:deferred.append(number))
        try:
            if child is not None and child.poll() is None:
                if birth is None:raise ValueError('Unreaped child without verified birth; stop queue for root')
                cleanup=terminate_owned(child,birth)
            save(queue/('CASE_'+str(index).zfill(2)+'_PROCESS.json'),{'child':birth,'exit_code':None if child is None else child.returncode,'external_stop_reason':reason,
                'cleanup':cleanup,'wall_seconds':time.monotonic()-started,'stdout':receipt(root,stdout_path) if stdout_path.exists() else None,'stderr':receipt(root,stderr_path) if stderr_path.exists() else None,'no_retry':True})
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
        if deferred:raise RuntimeError('Owned queue termination/deadline deferred until child cleanup receipt')
    directory=local(root,relative);result=directory/'RESULT.json';failure=directory/'TERMINAL_FAILURE.json'
    if result.exists() and child.returncode==0 and reason is None:
        value=json.loads(result.read_text());validate_success(value,case,birth)
        return {'case_id':case['case_id'],'status':'RANDOM_SOURCE_ZERO_UPDATE_PROXY_COMPLETE','result':receipt(root,result),'child':birth,'wall_seconds':time.monotonic()-started,'zero_update_equality_proof_complete':True}
    if failure.exists():
        value=json.loads(failure.read_text())
        if not same(value.get('owner'),birth):raise ValueError('Failure belongs to a different owner')
        if (value.get('status') not in ('RESOURCE_DEFERRED','ENGINEERING_FAILURE') or value.get('zero_update_equality_proof_complete') is not False
                or value.get('actual_optimizer_update_count_verified') is not None or value.get('scientific_fits')!=0
                or value.get('scores_emitted')!=0 or value.get('automatic_retry') is not False):raise ValueError('Failed case must retain incomplete/unknown proof and no scientific admission')
        return {'case_id':case['case_id'],'status':value['status'],'failure':receipt(root,failure),'child':birth,'wall_seconds':time.monotonic()-started,'zero_update_equality_proof_complete':False}
    value={'status':'RESOURCE_DEFERRED' if reason and 'DEADLINE' in reason else 'ENGINEERING_FAILURE','failure_type':reason or 'EXIT_BEFORE_TERMINAL_RECEIPT',
        'owner':birth,'wall_seconds':time.monotonic()-started,'zero_update_equality_proof_complete':False,'actual_optimizer_update_count_verified':None,'scientific_fits':0,'scores_emitted':0,'automatic_retry':False}
    path=queue/('CASE_'+str(index).zfill(2)+'_EXTERNAL_TERMINAL.json');save(path,value)
    return {'case_id':case['case_id'],'status':value['status'],'failure':receipt(root,path),'child':birth,'wall_seconds':value['wall_seconds'],'zero_update_equality_proof_complete':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project-root',required=True);parser.add_argument('--selection',required=True)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--queue-review');parser.add_argument('--reviewed-queue-sha256');parser.add_argument('--output')
    args=parser.parse_args();verify_allocation();root=Path(args.project_root).resolve();selection_path=local(root,args.selection);selection=json.loads(selection_path.read_text())
    if selection['schema']!='private-source-small-resource-roster-v1' or selection['execution_authorized'] is not False:raise ValueError('Exact disabled small roster required')
    manifest,custody=preflight(root,selection);cases={c['case_id']:c for c in manifest['cases']};ids=selection['case_ids']
    if len(ids)!=12 or len(set(ids))!=12 or any(i not in cases for i in ids):raise ValueError('Exact finite12-case source selection required')
    if not args.execute:
        print(json.dumps({'prepared_validation_complete':True,'selected_cases':12,'case_active_seconds':CASE_SECONDS,'case_cleanup_seconds':CLEANUP_SECONDS,'queue_maximum_seconds':7380,'CUDA_children_launched':0,'scientific_fits':0,'GO_created':False}));return
    if not all((args.queue_review,args.reviewed_queue_sha256,args.output)):raise ValueError('Exact separately root-reviewed queue and fresh output required')
    review_path=local(root,args.queue_review);review_binding=receipt(root,review_path)
    if review_binding['sha256']!=args.reviewed_queue_sha256:raise ValueError('Queue review bytes changed')
    review=json.loads(review_path.read_text())
    expected={'schema':'private-source-small-queue-review-v1','approved':True,'CUDA_engineering_execution_authorized':True,'runner':receipt(root,Path(__file__).resolve()),
        'selection':receipt(root,selection_path),'source_review':selection['source_review'],'helper':selection['helper'],'full_manifest':selection['full_manifest'],
        'case_ids':ids,'case_reviews':review.get('case_reviews'),'case_active_seconds':600,'case_cleanup_seconds':10,'queue_maximum_seconds':7380,
        'output':args.output,'one_engineering_queue_only':True,'scientific_fits_authorized':False,'GO_created':False}
    if review!=expected or not isinstance(review['case_reviews'],list) or len(review['case_reviews'])!=12:raise ValueError('Exact root-approved finite engineering queue receipt required')
    for identifier,case_review in zip(ids,review['case_reviews']):case_approval(root,selection,cases[identifier],case_review)
    queue=local(root,args.output);queue.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+QUEUE_ACTIVE_SECONDS;owner=own();closed=[];stop=None;active=None
    save(queue/'OWNER_START.json',{'owner':owner,'runner':expected['runner'],'root_queue_review':review_binding,'case_ids':ids,'queue_maximum_seconds':7380,'scientific_fits':0,'GO_created':False})
    def expired(*a):raise RuntimeError('Owned queue termination/deadline; preserve owned cleanup')
    signal.signal(signal.SIGTERM,expired);signal.signal(signal.SIGINT,expired);signal.signal(signal.SIGALRM,expired);signal.alarm(QUEUE_ACTIVE_SECONDS)
    try:
        for index,(identifier,case_review) in enumerate(zip(ids,review['case_reviews']),1):
            _,before=preflight(root,selection);active=identifier
            if time.monotonic()>=deadline:stop='FINITE_QUEUE_DEADLINE';break
            save(queue/'QUEUE_PROGRESS.json',{'owner':owner,'closed':closed,'active_case_id':active,'unexecuted_case_ids':[i for i in ids if i not in {r['case_id'] for r in closed}]})
            row=run_one(root,selection,cases[identifier],case_review,queue,index,before,deadline)
            _,after=preflight(root,selection);row['custody_after']=after;closed.append(row);active=None
            if row['status']=='ENGINEERING_FAILURE':stop='SOURCE_INTERFACE_OR_EQUALITY_FAILURE';break
            if time.monotonic()>=deadline:stop='FINITE_QUEUE_DEADLINE';break
    except BaseException as error:stop={'type':type(error).__name__,'message':str(error)}
    finally:
        signal.alarm(0);terminal={'status':'CLOSED' if len(closed)==12 and stop is None else 'STOPPED_FOR_ROOT_REVIEW','owner':owner,'closed':closed,
            'stop_reason':stop,'active_case_at_stop':active,'unexecuted_case_ids':[i for i in ids if i not in {r['case_id'] for r in closed}],
            'elapsed_wall_seconds':time.monotonic()-started,'automatic_retry':False,'scientific_fits':0,'GO_created':False,'whole_master_resource_qualified':False}
        save(queue/'OWNER_END.json',terminal);save(queue/'QUEUE_PROGRESS.json',terminal)


if __name__=='__main__':main()
