"""Fixed remaining14 initializer fits; direct scientific children only."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from types import SimpleNamespace

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
ORDER=('random_signs_all','tabm_first_normal','warm_identity','graph_covariance','feature_covariance','single','independent_warm4')

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
    if not path.is_relative_to(PHASE.resolve()) or not path.is_file():raise ValueError('Input leaves phase')
    return path
def bound(record):
    path=confined(record['path'])
    if sha(path)!=record['sha256']:raise ValueError('Bound file changed: '+record['path'])
    return path
def load(name,record):
    path=bound(record);spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def physical():
    if Path.cwd().resolve()!=REPO or socket.gethostname()!='anogena-2-0':raise ValueError('Allocation host/cwd changed')
    if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).split()!=[GPU]:raise ValueError('Physical GPU changed')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True);args=parser.parse_args()
    physical()
    if sha(args.config)!=args.config_sha256:raise ValueError('Root config changed')
    cfg=read(args.config)
    if cfg.get('execution_enabled') is not True or cfg.get('fits_authorized') is not True or cfg.get('VALID_values_access') is not True or cfg.get('TEST_access') is not False or cfg.get('retry') is not False or sha(__file__)!=cfg['adapter_sha256']:
        raise ValueError('Exact root remaining-fit adoption required')
    if cfg['warm1_custody_amendment']['approved'] is not True or cfg['warm1_custody_amendment']['process_exit_status']!='unknown':
        raise ValueError('Root must explicitly admit artifact completion with unknown warm1 process exit')
    for binding in cfg['bindings']:bound(binding)
    plan=read(bound(cfg['plan']));bounds=plan['bounds']
    if not plan['root_adopted'] or plan['TEST_closed'] is not True or tuple(plan['conditions'])!=ORDER or (plan['warm_cycles'],plan['post_cycles'],plan['eval_every'])!=(20,60,5):raise ValueError('Original fixed plan changed')
    expected_limits={'combined_child_log_cap_bytes':bounds['log_per_child_bytes'],'minimum_fresh_GPU_free_bytes':bounds['minimum_free_GPU_bytes'],'own_fit_output_cap_bytes':bounds['output_per_child_bytes'],'owned_tree_GPU_memory_cap_bytes':bounds['GPU_bytes'],'owned_tree_RSS_cap_bytes':bounds['RSS_bytes'],'poll_interval_seconds':15,'resource_wait_seconds':3600,'telemetry_timeout_seconds':10}
    if cfg['resource_limits']!=expected_limits:raise ValueError('Original bounds changed')
    if [row['seed'] for row in cfg['warms']]!=[1,2] or len(cfg['templates'])!=14:raise ValueError('Fixed remaining two seeds/fourteen cells required')
    old=load('flat_existing_run_fit',cfg['existing_owner_loop']);helper=load('flat_existing_helper',cfg['existing_helper']);helper.GPU=GPU
    owner=helper.identity(os.getpid())
    if owner is None or owner['sid']!=os.getpid() or owner['pgid']!=os.getpid():raise ValueError('Detached flat owner required')
    root=PHASE/cfg['execution_root_relative']
    if not root.resolve().is_relative_to(PHASE) or (root/'owner').exists():raise ValueError('Fresh flat execution root required')
    work=root/'owner';work.mkdir();(work/'logs').mkdir();(root/'jobs').mkdir();(root/'runs').mkdir()
    write(work/'START.json',{'identity':owner,'config_sha256':args.config_sha256,'warm1_process_exit_status':'unknown','b0_retained':True,'new_warm_fits':0,'nested_owner_children':0,'scores_read':False})
    completed=[]

    def no_cuda(pid):
        return not any(len(parts)>1 and parts[1].strip()==str(pid) for parts in (line.split(',') for line in helper.query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],10)))

    def clean_terminal(receipt,expected_job,argv):
        ident=receipt.get('raw_identity_observation')
        if receipt.get('exit_code')!=0 or receipt.get('reason') is not None or receipt.get('signals_sent') or receipt.get('terminal_wait_observed') is not True or receipt.get('job_sha256')!=expected_job or ident is None or ident['argv']!=argv or helper.identity(ident['PID']) is not None or not no_cuda(ident['PID']):raise ValueError('Scientific child lacks clean physical terminal closure')

    def warm_binding(row):
        jobpath=bound(row['job']);job=read(jobpath);block=next(b for b in plan['blocks'] if b['seed']==row['seed'])
        if (job['phase'],job['seed'],job['factor_seed'])!=('warm',row['seed'],block['factor_seed']) or any(job[k] is not False for k in ('TEST_access','VALID_values_access','fits_authorized','retry')) or job['plan_relative']!=cfg['plan']['path'] or job['plan_sha256']!=cfg['plan']['sha256'] or job['program_sha256']!=cfg['scientific_program_sha256'] or job['initialization_sha256']!=cfg['initialization_sha256']:raise ValueError('Original warm authority differs')
        folder=PHASE/row['output_relative'];freeze_path=folder/'FREEZE.json'
        if row['seed']==2:
            warmcfg=read(bound(row['owner_config']));owner_folder=PHASE/row['owner_relative'];started=time.monotonic()
            while True:
                physical();receipt_path=owner_folder/'TERMINAL.json';complete_path=owner_folder/'WARM_COMPLETE.json'
                if (folder/'FAILURE.json').exists():raise ValueError('Warm2 scientific failure preserved')
                if receipt_path.is_file():
                    receipt=read(receipt_path)
                    if receipt.get('exit_code')!=0 or receipt.get('reason') is not None or receipt.get('signals_sent'):raise ValueError('Warm2 failed terminal')
                    if complete_path.is_file() and helper.identity(row['owner_PID']) is None:break
                elif helper.identity(row['owner_PID']) is None:raise ValueError('Prescribed warm2 owner ended without terminal authority')
                if time.monotonic()-started>=row['wait_seconds']:raise TimeoutError('Fixed warm2 receipt wait exhausted')
                write(work/'PROGRESS.json',{'phase':'await_prescribed_warm2','completed':0,'wait_seconds':time.monotonic()-started,'scores_read':False});time.sleep(15)
            identity=read(owner_folder/'OWNER_STARTED.json')['identity']
            if (identity['PID'],identity['start_ticks'])!=(row['owner_PID'],row['owner_start_ticks']):raise ValueError('Warm2 owner identity changed')
            clean_terminal(receipt,row['job']['sha256'],warmcfg['argv']);closure=read(complete_path)
            if closure['child_absent'] is not True or closure['freeze_sha256']!=sha(freeze_path):raise ValueError('Warm2 complete authority differs')
        else:
            if helper.identity(row['worker_PID']) is not None or not no_cuda(row['worker_PID']):raise ValueError('Preserved warm1 worker is not physically absent')
            if sha(freeze_path)!=row['freeze_sha256']:raise ValueError('Admitted warm1 FREEZE changed')
        freeze=read(freeze_path)
        if (folder/'FAILURE.json').exists() or (freeze['phase'],freeze['seed'],freeze['completed_cycles'])!=('warm',row['seed'],20) or freeze['counters']!={'episodes':1220,'native_Adam_updates':3660} or len(freeze['cycle_seconds'])!=20 or freeze['VALID_TEST_access'] is not False or freeze['TEST_access'] is not False or freeze['selected_cycle'] is not None or freeze['selected_VALID_MRR'] is not None or freeze['native_alias_restored'] is not True:raise ValueError('Complete TRAIN-only terminal20 artifact required')
        if freeze['job_sha256']!=row['job']['sha256'] or freeze['plan_sha256']!=cfg['plan']['sha256'] or freeze['runtime']!=job['runtime_versions'] or freeze['inputs']!=cfg['TRAIN_inputs']:raise ValueError('Warm state source/runtime/input custody differs')
        if sha(folder/'warm_checkpoint.pt')!=freeze['checkpoint_sha256']:raise ValueError('Warm checkpoint hash changed')
        for name,digest in freeze['provenance_sha256'].items():
            if name not in ('CYCLE_DRAWS.jsonl.gz','INITIAL_STATE.pt','FINAL_STATE.pt','HISTORY.jsonl') or sha(folder/name)!=digest:raise ValueError('Warm RNG/state/draw provenance changed')
        if set(freeze['provenance_sha256'])!={'CYCLE_DRAWS.jsonl.gz','INITIAL_STATE.pt','FINAL_STATE.pt','HISTORY.jsonl'}:raise ValueError('Complete state/RNG/draw provenance required')
        if row['seed']==1 and freeze['checkpoint_sha256']!=row['checkpoint_sha256']:raise ValueError('Admitted orphan checkpoint changed')
        result={'seed':row['seed'],'job':row['job'],'FREEZE':{'path':str(freeze_path.relative_to(PHASE)),'sha256':sha(freeze_path)},'checkpoint_sha256':freeze['checkpoint_sha256'],'provenance_sha256':freeze['provenance_sha256'],'warm_process_exit_status':'unknown' if row['seed']==1 else 'clean_exit0','source_completion_verified':True,'worker_absent_no_CUDA':True}
        write(work/('WARM_'+str(row['seed'])+'_BOUND.json'),result);return result,job

    try:
        prior=read(PHASE/cfg['preserved_b0_block_freeze_relative'])
        if prior['block']!='b0' or prior['selected_block_complete'] is not True or prior['physical_postfits']!=7 or prior['TEST_access'] is not False:raise ValueError('Original b0 seven fits not complete')
        donors={}
        for row in cfg['warms']:donors[row['seed']]=warm_binding(row)
        entries=[]
        for index,binding in enumerate(cfg['templates']):
            template=read(bound(binding));seed=1+index//7;condition=ORDER[index%7];warm,warmjob=donors[seed];job=copy.deepcopy(template)
            if (job['phase'],job['seed'],job['condition'],job['program_sha256'],job['initialization_sha256'])!=('fit',seed,condition,cfg['scientific_program_sha256'],cfg['initialization_sha256']):raise ValueError('Exact fixed postfit template differs')
            for key in ('repository','physical_gpu_uuid','runtime_versions','source_manifest_sha256','runtime_qualification','feature_authority','negative_pool_authority'):
                if job[key]!=warmjob[key]:raise ValueError('Original warm/fit source/runtime/input authority differs')
            if job['factor_seed']!=warmjob['factor_seed']:raise ValueError('Original factor seed changed')
            family=condition if condition in ('single','independent_warm4') else 'shared_F4'
            if job['soft_seconds']!=bounds[family]['soft_seconds']:raise ValueError('Original fit bound changed')
            cell='b'+str(seed)+'_'+condition;path=root/'jobs'/(cell+'.json');out=root/'runs'/cell
            if path.exists() or out.exists():raise ValueError('No existing fit output or retry')
            job.update(plan_relative=cfg['plan']['path'],plan_sha256=cfg['plan']['sha256'],warm_freeze_relative=warm['FREEZE']['path'],warm_freeze_sha256=warm['FREEZE']['sha256'],output_directory=str(out),fits_authorized=True,VALID_values_access=True,source_review_approved=True,source_review=copy.deepcopy(warmjob['source_review']),external_hard_bound_confirmed=True,warm_custody={'process_exit_status':warm['warm_process_exit_status'],'root_config_sha256':args.config_sha256,'artifact_completion_verified':True})
            write(path,job);path.chmod(0o444)
            entries.append({'cell_id':cell,'job_relative':str(path.relative_to(PHASE)),'job_sha256':sha(path),'hard_seconds':bounds[family]['hard_seconds'],'argv':[cfg['python_executable'],'-B',str(PHASE/cfg['scientific_source_relative']/'run.py'),'--job',str(path),'--output',str(out)]})
        write(work/'BOUND_JOBS.json',{'entries':entries,'warms':[donors[s][0] for s in (1,2)],'fits':14,'b0_retained':True,'new_warm_fits':0,'scores_read':False})
        environment=dict(os.environ,**cfg['environment']);environment.pop('PYTHONHOME',None)
        context=SimpleNamespace(REPO=REPO,SOURCE=PHASE/cfg['scientific_source_relative'],SOURCE_SHA=cfg['scientific_program_sha256'],GPU_UUID=GPU,GPU_UUIDS=(GPU,),phase_file=confined,physical_host=physical,sha=sha,write=write)
        for entry in entries:
            physical();bound({'path':entry['job_relative'],'sha256':entry['job_sha256']})
            out=Path(entry['argv'][6])
            if out.exists():raise ValueError('No fit overwrite or retry')
            terminal=old.run_fit(helper,work,entry,{'resource_limits':cfg['resource_limits']},environment,out,context)
            clean_terminal(terminal,entry['job_sha256'],entry['argv'])
            freeze=read(out/'FREEZE.json');expected=read(confined(entry['job_relative']))
            if (freeze['phase'],freeze['seed'],freeze['condition'],freeze['completed_cycles'],freeze['job_sha256'],freeze['plan_sha256'],freeze['TEST_access'])!=('fit',expected['seed'],expected['condition'],60,entry['job_sha256'],cfg['plan']['sha256'],False) or (out/'FAILURE.json').exists():raise ValueError('Complete original fit custody differs')
            completed.append({'cell_id':entry['cell_id'],'FREEZE':{'path':str((out/'FREEZE.json').relative_to(PHASE)),'sha256':sha(out/'FREEZE.json')},'terminal':terminal})
            write(work/'PROGRESS.json',{'completed':completed,'total':14,'scores_read':False})
        write(work/'COMPLETE.json',{'completed':completed,'fits':14,'new_warm_fits':0,'b0_retained':True,'warm1_process_exit_status':'unknown','TEST_access':False,'retry':False,'scores_read':False})
    except BaseException as error:
        write(work/'FAILURE.json',{'error':type(error).__name__+': '+str(error),'completed':completed,'preserve_outputs':True,'retry':False,'scores_read':False});raise

if __name__=='__main__':main()
