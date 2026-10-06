"""One b0 qualification binding to unchanged reviewed run_fit/ownership helper."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,importlib.util,json,os,socket,subprocess
REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');PHASE=REPO/'experiments_iclr/postsubmission_20260930'
PREP=PHASE/'citeseer_nonlinear_preaggregation_growth_b0_allocation_activation_preparation_20261007_v1';EXEC=PHASE/'citeseer_nonlinear_preaggregation_growth_b0_allocation_qualification_execution_root_20261007_v1'
OLD=PHASE/'citeseer_known_ranking_control_matched_reference_owned_preparation_20261006_v2/owner.py'
OLD_SHA='1b1edf43a895600515bbf340fbc813b5dbc48e325ef73603fc5118146d4d5fdf'
HELPER=PHASE/'shared_private_transfer_gpu77_qualification_preparation_20261005_v3/ownership_helpers.py'
HELPER_SHA='e71503c87865546319cddbbf7a4f9f15d13cdf9e4875e406d65de21e64e047fd'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def phase_file(rel):
    value=Path(rel)
    if value.is_absolute() or '..' in value.parts:raise ValueError('Require phase-relative input')
    p=(PHASE/value).resolve(strict=True)
    if not p.is_relative_to(PHASE.resolve()) or not p.is_file():raise ValueError('Input leaves phase')
    return p
def load(name,p,expected):
    if sha(p)!=expected:raise ValueError('Pinned primitive changed: '+name)
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release',type=Path,required=True);args=parser.parse_args()
    r=read(args.release)
    if args.release.resolve(strict=True)!=EXEC/'ROOT_RELEASE.json' or r.get('execution_enabled') is not True or r.get('qualification_authorized') is not True:
        raise ValueError('Exact root qualification release required')
    if any(r.get(k) is not False for k in ('fits_authorized','complete_cycle_costs_authorized','VALID_values_access','TEST_access','optimizer_updates_authorized','retry')):
        raise ValueError('Qualification-only zero-update scope required')
    if sha(__file__)!=r['owner_program_sha256'] or sha(PREP/'OWNER_CONFIG_DISABLED.json')!=r['owner_config_sha256']:
        raise ValueError('Root-bound owner/config changed')
    cfg=read(PREP/'OWNER_CONFIG_DISABLED.json');GPU=cfg['physical_gpu_uuid']
    def physical():
        if Path.cwd().resolve()!=REPO.resolve() or socket.gethostname()!=cfg['expected_hostname']:raise ValueError('Allocation hostname/repository changed')
        if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).split()!=[GPU]:raise ValueError('Allocation physical GPU changed')
    physical()
    source=PHASE/cfg['source_relative']
    if sha(source/'SOURCE_MANIFEST.json')!=cfg['source_manifest_sha256'] or sha(source/'run.py')!=cfg['program_sha256'] or sha(phase_file(cfg['review_relative']))!=cfg['review_sha256']:
        raise ValueError('Growth source/review changed')
    for row in read(source/'SOURCE_MANIFEST.json')['files']:
        p=(source/row['path']).resolve(strict=True)
        if not p.is_relative_to(source) or sha(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise ValueError('Sealed source bytes changed')
    if sha(PREP/'ROOT_ADOPTED_PLAN.json')!=cfg['plan_sha256'] or sha(PREP/'B0_JOB_DISABLED.json')!=cfg['job_preview_sha256']:
        raise ValueError('Root-bound plan/job preview changed')
    preview=read(PREP/'B0_JOB_DISABLED.json')
    frozen=phase_file(preview['warm_freeze_relative']);freeze=read(frozen)
    if sha(frozen)!=preview['warm_freeze_sha256'] or sha(frozen.parent/'warm_checkpoint.pt')!=preview['warm_checkpoint_sha256'] or freeze['checkpoint_sha256']!=preview['warm_checkpoint_sha256']:
        raise ValueError('Retained common warm state changed')
    if sha(Path(cfg['argv'][0]).resolve(strict=True))!=cfg['python_binary_sha256']:raise ValueError('Allocation Python binary changed')
    old=load('growth_existing_owner',OLD,OLD_SHA);h=load('growth_existing_helper',HELPER,HELPER_SHA);h.GPU=GPU
    owner=h.identity(os.getpid())
    if owner is None or owner['pgid']!=os.getpid() or owner['sid']!=os.getpid():raise ValueError('Fresh detached owner session required')
    root=EXEC/'owner';jobpath=EXEC/'jobs/b0_qualify.json';output=Path(cfg['argv'][6])
    if root.exists() or jobpath.exists() or output.exists():raise ValueError('Fresh one-shot owner/job/output only; no retry')
    root.mkdir();(root/'logs').mkdir();write(root/'ROOT_ADOPTION.json',r)
    write(root/'OWNER_STARTED.json',{'UTC':h.now(),'identity':owner,'root_release_sha256':sha(args.release),'TRAIN_only':True,'optimizer_updates':0,'VALID_TEST_values_access':False})
    try:
        job=dict(preview);job.update(execution_enabled=True,source_review_approved=True,external_hard_bound_confirmed=True)
        job['source_review']={**job['source_review'],'approved':True};write(jobpath,job)
        if sha(jobpath)!=cfg['job_active_sha256']:raise ValueError('Activated job differs from exact preview transformation')
        env=dict(os.environ,**cfg['environment']);env.pop('PYTHONHOME',None)
        context=SimpleNamespace(REPO=REPO,SOURCE=source,SOURCE_SHA=cfg['source_manifest_sha256'],GPU_UUID=GPU,GPU_UUIDS=(GPU,),phase_file=phase_file,physical_host=physical,sha=sha,write=write)
        entry={'cell_id':'b0_qualify','job_relative':cfg['job_relative'],'job_sha256':sha(jobpath),'argv':cfg['argv'],'hard_seconds':cfg['child_hard_seconds']}
        terminal=old.run_fit(h,root,entry,{'resource_limits':cfg['resource_limits']},env,output,context)
        write(root/'TERMINAL.json',terminal);identity=terminal['raw_identity_observation']
        if identity is None:raise ValueError('Owned child identity not observed')
        absent=h.identity(identity['PID']) is None
        rows=h.query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],10)
        no_cuda=not any(len(parts)>1 and parts[1].strip()==str(identity['PID']) for parts in (row.split(',') for row in rows))
        write(root/'PHYSICAL_TERMINAL.json',{'PID':identity['PID'],'start_ticks':identity['start_ticks'],'owned_PID_absent':absent,'owned_PID_no_CUDA_rows':no_cuda,'terminal_wait_observed':terminal['terminal_wait_observed']})
        if terminal['exit_code']!=0 or terminal['reason'] is not None or terminal['signals_sent'] or not absent or not no_cuda or not terminal['terminal_wait_observed'] or (output/'FAILURE.json').exists():
            raise ValueError('Qualification failed clean completion; preserve paid failure')
        freeze=read(output/'FREEZE.json');gates=read(output/'NATIVE_GATES.json')
        required={'no_growth','graph_growth','unfiltered_growth','unfiltered_top8_partition','capable_single_rank8','independent_graph_growth4'}
        if freeze.get('phase')!='qualify' or freeze.get('success') is not True or freeze.get('seed')!=0 or freeze.get('growth_source_manifest_sha256')!=cfg['source_manifest_sha256'] or freeze.get('optimizer_updates')!=0 or freeze.get('VALID_TEST_access') is not False:
            raise ValueError('Qualification completion custody differs')
        if set(gates)!=required or not all(row.get('copied_native_forward_gradient_gate_passed') is True for row in gates.values()) or sha(output/'NATIVE_GATES.json')!=freeze['gates_sha256']:
            raise ValueError('All six actual native gates required')
        write(root/'COMPLETE.json',{'passed':True,'phase':'qualify','seed':0,'all_six_native_gates':True,'freeze_sha256':sha(output/'FREEZE.json'),'gates_sha256':sha(output/'NATIVE_GATES.json'),'terminal':terminal,'fits':0,'optimizer_updates':0,'VALID_TEST_values_access':False,'retry':False})
    except (Exception,KeyboardInterrupt) as error:
        write(root/'FAILURE.json',{'error':type(error).__name__+': '+str(error),'partial_outputs_preserved':True,'retry':False,'fits':0});raise

if __name__=='__main__':main()
