"""Exactly15 root-adopted Growth fits; unchanged run_fit per scientific child."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
CONDITIONS=('graph_growth','unfiltered_growth','unfiltered_top8_partition','capable_single_rank8','independent_graph_growth4')
ALL_GATES=set(CONDITIONS)|{'no_growth'}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def confined(relative):
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Phase-relative file required')
    path=(PHASE/rel).resolve(strict=True)
    if not path.is_relative_to(PHASE.resolve()) or not path.is_file():raise ValueError('File leaves phase')
    return path
def bound(record):
    path=confined(record['path'])
    if sha(path)!=record['sha256']:raise ValueError('Exact bound bytes changed: '+record['path'])
    return path
def load(name,record):
    path=bound(record);spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def physical():
    if Path.cwd().resolve()!=REPO or socket.gethostname()!='anogena-2-0':raise ValueError('Allocation host/cwd changed')
    if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).split()!=[GPU]:raise ValueError('Physical allocation GPU changed')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True);args=parser.parse_args();physical()
    if sha(args.config)!=args.config_sha256:raise ValueError('Root config changed')
    cfg=read(args.config)
    if any(cfg.get(k) is not True for k in ('execution_enabled','fits_authorized','VALID_values_access','full_pilot_adopted')) or cfg.get('TEST_access') is not False or cfg.get('retry') is not False or sha(__file__)!=cfg['adapter_sha256']:
        raise ValueError('Exact fixed full-pilot root adoption required')
    if cfg['warm1_custody_amendment']['approved'] is not True or cfg['warm1_custody_amendment']['process_exit_status']!='unknown':raise ValueError('Explicit preserved warm1 artifact custody amendment required')
    for row in cfg['bindings']:bound(row)
    plan=read(bound(cfg['plan']));bounds=plan['bounds'];source=PHASE/cfg['source_relative']
    if plan.get('root_adopted') is not True or plan.get('scientific_execution_authorized') is not True or plan.get('full_pilot_adopted') is not True or plan['TEST_closed'] is not True or tuple(plan['conditions'])!=CONDITIONS or (plan['warm_cycles'],plan['post_cycles'],plan['eval_every'])!=(20,60,5) or [b['seed'] for b in plan['blocks']]!=[0,1,2]:raise ValueError('Fixed15-fit scientific plan required')
    original_plan=read(source/'PLAN.json')
    for key in original_plan:
        if key not in ('root_adopted','scientific_execution_authorized','cost_gate') and plan.get(key)!=original_plan[key]:raise ValueError('Scientific plan changed: '+key)
    if plan['cost_gate']!=original_plan['cost_gate'].replace('each of four new arms','each of five new arms'):raise ValueError('Only the declared cost prose correction admitted')
    expected_limits={'combined_child_log_cap_bytes':bounds['log_per_child_bytes'],'minimum_fresh_GPU_free_bytes':bounds['minimum_free_GPU_bytes'],'own_fit_output_cap_bytes':bounds['output_per_child_bytes'],'owned_tree_GPU_memory_cap_bytes':bounds['GPU_bytes'],'owned_tree_RSS_cap_bytes':bounds['RSS_bytes'],'poll_interval_seconds':15,'resource_wait_seconds':3600,'telemetry_timeout_seconds':10}
    if cfg['resource_limits']!=expected_limits or len(cfg['templates'])!=15:raise ValueError('Original default bounds or15 cells changed')
    old=load('growth_existing_run_fit',cfg['existing_owner_loop']);helper=load('growth_existing_helper',cfg['existing_helper']);helper.GPU=GPU
    def absent_no_cuda(pid):
        return helper.identity(pid) is None and not any(len(parts)>1 and parts[1].strip()==str(pid) for parts in (row.split(',') for row in helper.query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],10)))
    def clean_terminal(receipt,job_hash,argv):
        child=receipt.get('raw_identity_observation')
        if receipt.get('exit_code')!=0 or receipt.get('reason') is not None or receipt.get('signals_sent') or receipt.get('terminal_wait_observed') is not True or receipt.get('job_sha256')!=job_hash or child is None or child['argv']!=argv or not absent_no_cuda(child['PID']):raise ValueError('Scientific child clean physical terminal required')
    quals={};costs={};warms={}
    for row in cfg['qualifications']:
        freeze=read(bound(row['FREEZE']));gates=read(bound(row['NATIVE_GATES']));seed=row['seed']
        if (freeze.get('phase'),freeze.get('success'),freeze.get('seed'),freeze.get('growth_source_manifest_sha256'),freeze.get('optimizer_updates'),freeze.get('VALID_TEST_access'))!=('qualify',True,seed,cfg['source_manifest_sha256'],0,False) or freeze['gates_sha256']!=row['NATIVE_GATES']['sha256'] or set(gates)!=ALL_GATES or not all(g.get('copied_native_forward_gradient_gate_passed') is True for g in gates.values()):raise ValueError('Actual same-seed all-six gate custody differs')
        quals[seed]=row['FREEZE']
    if set(quals)!={0,1,2}:raise ValueError('All3 exact seed gates required')
    for record in cfg['qualification_owner_receipts']:
        complete=read(bound(record))
        records=complete.get('completed')
        if records is None:
            terminal=complete['terminal']
            if complete.get('passed') is not True or complete.get('all_six_native_gates') is not True:raise ValueError('b0 qualification owner admission differs')
            clean_terminal(terminal,terminal['job_sha256'],terminal['raw_identity_observation']['argv'])
        else:
            for item in records:clean_terminal(item['terminal'],item['terminal']['job_sha256'],item['terminal']['raw_identity_observation']['argv'])
    for row in cfg['costs']:
        freeze=read(bound(row['FREEZE']));condition=row['condition'];expected=244 if condition=='independent_graph_growth4' else 61
        if (freeze['phase'],freeze['success'],freeze['seed'],freeze['condition'],freeze['complete_cycles'],freeze['growth_source_manifest_sha256'],freeze['VALID_TEST_access'],freeze['TEST_access'])!=('cost',True,0,condition,1,cfg['source_manifest_sha256'],False,False) or freeze['counters']!={'episodes':expected,'Adam_updates':3*expected,'complete_train_cycles_per_member':1}:raise ValueError('Actual complete same-arm b0 cost differs')
        family='independent_warm4' if condition=='independent_graph_growth4' else'single' if condition=='capable_single_rank8' else'shared_F4'
        if 60*freeze['cycle_seconds'][0]+freeze['inclusive_seconds']>=bounds[family]['soft_seconds']:raise ValueError('Measured complete cycles have no fixed-cap headroom for12VALID passes')
        costs[condition]=row['FREEZE']
    if set(costs)!=set(CONDITIONS):raise ValueError('All5 actual complete costs required')
    cost_owner=read(bound(cfg['cost_owner_complete']))
    if len(cost_owner['completed'])!=5 or cost_owner['fits']!=0:raise ValueError('Complete5-cost owner receipt required')
    for item in cost_owner['completed']:clean_terminal(item['terminal'],item['terminal']['job_sha256'],item['terminal']['raw_identity_observation']['argv'])
    for row in cfg['warms']:
        freeze_path=bound(row['FREEZE']);freeze=read(freeze_path);folder=freeze_path.parent;bound(row['checkpoint'])
        if (freeze['phase'],freeze['seed'],freeze['completed_cycles'],freeze['VALID_TEST_access'],freeze['TEST_access'],freeze['native_alias_restored'])!=('warm',row['seed'],20,False,False,True) or freeze['counters']!={'episodes':1220,'native_Adam_updates':3660} or freeze['checkpoint_sha256']!=row['checkpoint']['sha256'] or freeze['inputs']!=cfg['TRAIN_inputs'] or (folder/'FAILURE.json').exists():raise ValueError('Exact terminal20 TRAIN-only donor required')
        if freeze['runtime']!=cfg['runtime_versions'] or freeze['job_sha256']!=row['warm_job_sha256'] or freeze['plan_sha256']!=cfg['warm_plan_sha256']:raise ValueError('Original warm source/runtime/job/plan authority differs')
        for filename,digest in freeze['provenance_sha256'].items():
            if filename not in ('INITIAL_STATE.pt','FINAL_STATE.pt','CYCLE_DRAWS.jsonl.gz','HISTORY.jsonl') or sha(folder/filename)!=digest:raise ValueError('Warm state/RNG/draw provenance changed')
        if set(freeze['provenance_sha256'])!={'INITIAL_STATE.pt','FINAL_STATE.pt','CYCLE_DRAWS.jsonl.gz','HISTORY.jsonl'}:raise ValueError('All warm provenance required')
        if row['seed']==1 and not absent_no_cuda(row['worker_PID']):raise ValueError('Preserved warm1 worker is not absent')
        warms[row['seed']]=row
    if set(warms)!={0,1,2}:raise ValueError('Exactly3 retained warm donors required')
    ident=helper.identity(os.getpid())
    if ident is None or ident['sid']!=os.getpid() or ident['pgid']!=os.getpid():raise ValueError('Fresh detached fit owner session required')
    root=PHASE/cfg['execution_root_relative']
    if not root.resolve().is_relative_to(PHASE) or (root/'owner').exists():raise ValueError('Fresh one-shot fit root required')
    work=root/'owner';work.mkdir();(work/'logs').mkdir();(root/'jobs').mkdir();(root/'runs').mkdir()
    write(work/'START.json',{'identity':ident,'config_sha256':args.config_sha256,'full_fixed15_adopted':True,'new_warm_or_qualification_runs':0,'warm1_process_exit_status':'unknown','scores_read':False})
    entries=[];completed=[]
    try:
        for index,record in enumerate(cfg['templates']):
            job=copy.deepcopy(read(bound(record)));seed=index//5;condition=CONDITIONS[index%5];warm=warms[seed]
            if (job['phase'],job['seed'],job['condition'],job['program_sha256'],job['growth_source_manifest_sha256'])!=('fit',seed,condition,cfg['program_sha256'],cfg['source_manifest_sha256']) or job['factor_seed']!=plan['blocks'][seed]['factor_seed']:raise ValueError('Exact fixed job template differs')
            family='independent_warm4' if condition=='independent_graph_growth4' else'single' if condition=='capable_single_rank8' else'shared_F4'
            if job['soft_seconds']!=bounds[family]['soft_seconds']:raise ValueError('Original full-fit soft cap changed')
            cell='b'+str(seed)+'_'+condition;path=root/'jobs'/(cell+'.json');out=root/'runs'/cell
            if path.exists() or out.exists():raise ValueError('No existing output or retry')
            job.update(plan_relative=cfg['plan']['path'],plan_sha256=cfg['plan']['sha256'],warm_freeze_relative=warm['FREEZE']['path'],warm_freeze_sha256=warm['FREEZE']['sha256'],warm_checkpoint_sha256=warm['checkpoint']['sha256'],qualification_receipt=quals[seed],complete_cycle_cost_receipt=costs[condition],output_directory=str(out),source_review_approved=True,source_review={'approved':True,'evidence':cfg['source_review_evidence']},fits_authorized=True,VALID_values_access=True,optimizer_updates_authorized=True,external_hard_bound_confirmed=True,warm_process_exit_status='unknown' if seed==1 else'clean_exit0')
            if job['TEST_access'] is not False or job['retry'] is not False:raise ValueError('TESTclosed/noretry')
            write(path,job);path.chmod(0o444)
            entries.append({'cell_id':cell,'job_relative':str(path.relative_to(PHASE)),'job_sha256':sha(path),'hard_seconds':bounds[family]['hard_seconds'],'argv':[cfg['python_executable'],'-B',str(source/'run.py'),'--job',str(path),'--output',str(out)]})
        write(work/'BOUND_JOBS.json',{'entries':entries,'warms':cfg['warms'],'qualifications':cfg['qualifications'],'costs':cfg['costs'],'plan':cfg['plan'],'fixed60cycles_12VALIDpasses':True,'scores_read':False})
        environment=dict(os.environ,**cfg['environment']);environment.pop('PYTHONHOME',None)
        context=SimpleNamespace(REPO=REPO,SOURCE=source,SOURCE_SHA=cfg['source_manifest_sha256'],GPU_UUID=GPU,GPU_UUIDS=(GPU,),phase_file=confined,physical_host=physical,sha=sha,write=write)
        for entry in entries:
            physical();bound({'path':entry['job_relative'],'sha256':entry['job_sha256']});out=Path(entry['argv'][6])
            if out.exists():raise ValueError('No fit overwrite/retry')
            terminal=old.run_fit(helper,work,entry,{'resource_limits':cfg['resource_limits']},environment,out,context)
            clean_terminal(terminal,entry['job_sha256'],entry['argv']);freeze=read(out/'FREEZE.json');job=read(confined(entry['job_relative']))
            count=244 if job['condition']=='independent_graph_growth4' else 61
            if (freeze['phase'],freeze['success'],freeze['seed'],freeze['condition'],freeze['complete_cycles'],freeze['growth_source_manifest_sha256'],freeze['TEST_access'])!=('fit',True,job['seed'],job['condition'],60,cfg['source_manifest_sha256'],False) or freeze['counters']!={'episodes':60*count,'Adam_updates':180*count,'complete_train_cycles_per_member':60} or (out/'FAILURE.json').exists():raise ValueError('Complete fixed fit custody differs')
            completed.append({'cell_id':entry['cell_id'],'FREEZE':{'path':str((out/'FREEZE.json').relative_to(PHASE)),'sha256':sha(out/'FREEZE.json')},'terminal':terminal})
            write(work/'PROGRESS.json',{'completed':completed,'total':15,'scores_read':False})
        write(work/'COMPLETE.json',{'completed':completed,'fits':15,'new_warm_or_qualifications':0,'TEST_access':False,'retry':False,'scores_read':False})
    except BaseException as error:
        write(work/'FAILURE.json',{'error':type(error).__name__+': '+str(error),'completed':completed,'preserve_outputs':True,'retry':False,'scores_read':False});raise

if __name__=='__main__':main()
