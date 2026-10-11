"""Finite one-shot: full cohort -> frozen90 -> export -> join -> CAL -> TEST.

Imports stdlib only. No partial predictive metric or target array is decoded by
the watcher/binders. Every numerical operation belongs to pinned stage children.
"""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from common import R,H,UUID,CORE,BLOCKS,require,scoped,item,checked,read,write,sha,freeze_recheck,pipeline_gate

def now():return datetime.now(timezone.utc).isoformat()

def interrupted(signum,frame):
    raise InterruptedError('Finite pipeline deadline or owner termination; preserve failure without continuation')

def identity(pid):
    path=Path('/proc')/str(pid)/'stat'
    if not path.exists():return None
    try:parts=path.read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return None
    return {'pid':pid,'state':parts[0],'start_ticks':int(parts[19])}

def gpu_pids():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,timeout=15)
    require(result.returncode==0,'GPU cleanup observation failed; no next phase')
    return {int(line.split(',')[0].strip()) for line in result.stdout.splitlines() if UUID in line}

def child_stage(cfg,metadata,stage,command,output):
    cap=cfg['seconds'][stage];before=time.monotonic();timed_out=False
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    child=None;original=None;stage_error=None;code=None
    with (metadata/(stage+'.log')).open('xb') as log:
        try:
            child=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            original=identity(child.pid)
            write(metadata/(stage+'_OWNER_START.json'),{'UTC':now(),'child':original,'command':command,'cap_seconds':cap})
            try:code=child.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                timed_out=True
        except BaseException as error:
            stage_error={'type':type(error).__name__,'message':str(error)}
        finally:
            # Control only the direct child launched by this Popen object.
            if child is not None and child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait(timeout=15)
            if child is not None:code=child.returncode
    absent=child is None or identity(child.pid) is None
    cuda_absent=child is None or child.pid not in gpu_pids()
    grace=time.monotonic()+10
    while not (absent and cuda_absent) and time.monotonic()<grace:
        time.sleep(.2);absent=identity(child.pid) is None;cuda_absent=child.pid not in gpu_pids()
    result={'UTC':now(),'stage':stage,'exit_code':code,'timed_out':timed_out,'direct_child_wait':True,
            'child':original,'child_pid_absent':absent,'owned_cuda_pid_absent':cuda_absent,
            'seconds':time.monotonic()-before,'output':str(output),'retry':False,'stage_error':stage_error,
            'complete':code==0 and not timed_out and absent and cuda_absent and stage_error is None}
    write(metadata/(stage+'_OWNER_END.json'),result)
    require(result['complete'],'Fixed stage failure retained; no retry, survivor or next phase: '+stage)
    return item(metadata/(stage+'_OWNER_END.json'))

def close_cohort(cfg):
    start=read(checked(cfg['cohort_owner_start']));owner=read(cfg['cohort_owner_end'])
    require(owner.get('status')=='BOTH_COMPLETE_ZERO_EXIT' and owner.get('complete') is True
            and owner.get('banks')==60 and owner.get('fits')==90
            and set(owner.get('children',{}))=={'core','author'}
            and all(v['exit_code']==0 for v in owner['children'].values()),'Both complete direct zero exits; no survivor cohort')
    require(all(owner['children'][k]['pid']==start['children'][k]['pid'] for k in ('core','author')),
            'Exact initially bound owner/child roster')
    require(all(identity(v['pid']) is None and v['pid'] not in gpu_pids() for v in owner['children'].values()),
            'Both acquisition children absent/clean before state or prediction exports')
    require(owner['core_complete_receipt']==item(cfg['core_complete'])
            and owner['author_complete_receipt']==item(cfg['author_complete']),'Owner-bound complete metadata hashes')
    core=read(checked(owner['core_complete_receipt']));author=read(checked(owner['author_complete_receipt']))
    require(core['complete'] and core['full_same_runtime_stage_complete'] and core['groups']==50 and core['fit_units']==80
            and core['fixed_seed_roster']==list(BLOCKS) and core['fixed_arm_roster']==list(CORE)
            and len(core['results'])==50 and {(r['seed'],r['arm']) for r in core['results']}=={(b,a) for b in BLOCKS for a in CORE},
            'Fixed50-bank/80-fit complete core before any partial score use')
    require(author['complete'] and author['banks']==10 and author['fits']==10 and len(author['records'])==10
            and {r['seed'] for r in author['records']}==set(BLOCKS)
            and all(r['arm']=='author_meanSAGE_M1' and r['status']=='SELECTED_COMPLETE' for r in author['records']),
            'Fixed10 native source-context banks, all outcomes retained')
    states=[];archives=[]
    def pinned(value,**tags):
        checked(value);return dict(value,**tags)
    for bank in core['results']:
        require(len(bank['fits'])==(4 if bank['arm']=='equipped_I4' else 1),'Core body ownership unchanged')
        for member,fit in enumerate(bank['fits']):
            states.append(pinned(fit['selected_state_receipt'],block=bank['seed'],arm=bank['arm'],member=member))
        archives.append(pinned(bank['selected_VALID_archive'],block=bank['seed'],arm=bank['arm'],role='SELECT'))
    for bank in author['records']:
        require(len(bank['fits'])==1,'Native context is one complete single')
        states.append(pinned(bank['fits'][0]['selected_state_receipt'],block=bank['seed'],arm=bank['arm'],member=0))
        for role in ('SELECT','CAL','CONFIRM'):
            archives.append(pinned(bank['archives'][role],block=bank['seed'],arm=bank['arm'],role=role))
    require(len(states)==90 and len({r['path'] for r in states})==90 and len(archives)==80,
            'All90 genuine state hashes and80 existing selected archives, no arrays decoded')
    value={'schema':'Physics_all90_selected_freeze_v1','terminal':True,'complete':True,'banks':60,'body_fit_units':90,
           'core_complete':owner['core_complete_receipt'],'author_complete':owner['author_complete_receipt'],
           'cohort_owner_end':item(cfg['cohort_owner_end']),'selected_states':states,'selected_archives':archives,
           'protocol':cfg['protocol'],'core_integration_spec':cfg['core_integration_spec'],
           'partial_scores_opened_or_compared':False,'target_arrays_decoded':False,'UTC':now()}
    return value,core,author,owner

def bind_stages(cfg,metadata,freeze_path,core,author,owner,export_owner,core_predictions):
    prepared=read(checked(cfg['prepared_receipt']))
    require(prepared['complete'] and prepared['dataset']=='CoauthorPhysics' and set(prepared['profiles'])=={str(b) for b in BLOCKS},
            'Root-admitted complete provider roles; no role changes')
    readout=Path(cfg['readout_directory']);sources=dict(cfg['source_receipts'])
    sources.update(protocol=cfg['protocol'],pipeline_seal=item(Path(__file__).with_name('SEAL.json')),
                   selected_freeze=item(freeze_path),prepared=cfg['prepared_receipt'],core_complete=item(cfg['core_complete']),
                   author_complete=item(cfg['author_complete']),cohort_owner=item(cfg['cohort_owner_end']))
    costs=[{'scope':'Custodial preprocessing','receipt':cfg['prepared_receipt'],'seconds':prepared['seconds']},
           {'scope':'Core entire acquisition/preparation','complete':item(cfg['core_complete']),
            'preparation':core.get('preparation'),'seconds':core.get('entire_family_observed_seconds')},
           {'scope':'Author context entire acquisition','complete':item(cfg['author_complete']),
            'seconds':author.get('entire_context_wall_seconds'),'operation_counts':author.get('operation_counts')},
           {'scope':'Concurrent cohort owner wall; not summed isolated latency','seconds':owner['owner_seconds']},
           {'scope':'Selected core export owner','owner':export_owner,'seconds':read(checked(export_owner))['seconds']}]
    costs.extend({'scope':'Earlier preparation/qualification/failed attempt retained','receipt':v} for v in cfg['prior_cost_receipts'])
    join_cfg={'enabled':True,'protocol_sha256':cfg['protocol']['sha256'],'core_predictions':core_predictions,
              'author_context':item(cfg['author_complete']),'source_bindings':sources,'cost_records':costs}
    write(metadata/'JOIN_CONFIG.json',join_cfg)
    join_adm={'admitted':True,'stage':'join_predictions','config_sha256':sha(metadata/'JOIN_CONFIG.json'),
              'source_seal_sha256':sha(readout/'SEAL.json'),'root_validated_all90_selected_state_custody':True,
              'all_native_owners_closed_and_clean':True,'CAL_truth_access':False,'TEST_truth_access':False}
    write(metadata/'JOIN_ADMISSION.json',join_adm)
    roles={str(b):{key:prepared['files'][name] for key,name in {
        'cal_ids':f'block_{b}/cal_ids.npy','confirm_ids':f'block_{b}/confirm_ids.npy',
        'cal_truth':f'calibration_only/block_{b}_cal_targets.npz',
        'confirm_truth':f'scorer_only/block_{b}_confirm_targets.npz'}.items()} for b in BLOCKS}
    return readout,roles

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--admission',type=Path,required=True);args=parser.parse_args()
    cfg,adm,cfg_receipt,adm_receipt=pipeline_gate(args.config,args.admission)
    metadata=H/'after_fit_pipeline_v1';metadata.mkdir(exist_ok=False)
    began=time.monotonic();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    total_cap=sum(cfg['seconds'].values())+300  # Includes bounded owner-cleanup grace.
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.alarm(total_cap)
    write(metadata/'PIPELINE_START.json',{'UTC':now(),'config':cfg_receipt,'admission':adm_receipt,'boot_id':boot,
                                        'wait_cap_seconds':cfg['seconds']['watch'],'whole_pipeline_cap_seconds':total_cap,
                                        'partial_scores_read':False})
    stage='wait_full_cohort'
    try:
        while not Path(cfg['cohort_owner_end']).exists():
            require(time.monotonic()-began<cfg['seconds']['watch'],'Finite cohort wait expired; no next phase')
            require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,'Boot changed; retain excluded wait')
            time.sleep(30)
        freeze,core,author,owner=close_cohort(cfg)
        freeze_path=metadata/'ALL90_SELECTED_FREEZE.json';write(freeze_path,freeze)
        stage='export';export_path=H/'selected_core_export_v1'
        release={'admitted':True,'stage':'selected_core_export','pipeline_config':cfg_receipt,'pipeline_admission':adm_receipt,
                 'freeze':item(freeze_path),'pipeline_seal_sha256':sha(Path(__file__).with_name('SEAL.json')),
                 'CAL_truth_access':False,'TEST_truth_access':False}
        write(metadata/'EXPORT_RELEASE.json',release)
        python=str(R/'.venv/bin/python')
        export_owner=child_stage(cfg,metadata,'export',[python,'-B',str(Path(__file__).with_name('export_stage.py')),
            '--config',str(args.config),'--admission',str(args.admission),'--freeze',str(freeze_path),
            '--release',str(metadata/'EXPORT_RELEASE.json'),'--output',str(export_path)],export_path)
        core_predictions=item(export_path/'COMPLETE_SELECTED_CORE_PREDICTIONS.json')
        require(read(checked(core_predictions))['complete'],'Entire core selected-prediction export before join')
        freeze_recheck(freeze)
        readout,roles=bind_stages(cfg,metadata,freeze_path,core,author,owner,export_owner,core_predictions)
        stage='join';joined=H/'COMPLETE_SELECTED_PREDICTIONS_V1.json'
        join_owner=child_stage(cfg,metadata,'join',[python,'-B',str(readout/'join_predictions.py'),
            '--config',str(metadata/'JOIN_CONFIG.json'),'--admission',str(metadata/'JOIN_ADMISSION.json'),
            '--output',str(joined)],joined)
        closure=item(joined);freeze_recheck(freeze)
        readout_cfg={'enabled':True,'protocol_sha256':cfg['protocol']['sha256'],'blocks':list(BLOCKS),
            'arms':list(CORE)+['published_mean_SAGE_source_port_M1'],'node_count':34493,'classes':5,
            'role_counts':{'CAL':150,'CONFIRM':34093},'prediction_closure':closure,'proper_roles_admitted':True,
            'roles_disjoint_and_complete':True,'TEST_truth_in_predictions':False,'roles':roles,
            'provider_modules':cfg['provider_modules'],'stage_seconds':{k:cfg['seconds'][k] for k in ('calibrate','test')}}
        write(metadata/'READOUT_CONFIG.json',readout_cfg)
        base={'admitted':True,'config_sha256':sha(metadata/'READOUT_CONFIG.json'),'source_seal_sha256':sha(readout/'SEAL.json'),
              'prediction_closure':closure,'all60_banks_90_bodies_closed':True,'pipeline_config':cfg_receipt,
              'pipeline_admission':adm_receipt,'freeze':item(freeze_path),'join_owner':join_owner}
        write(metadata/'CAL_ADMISSION.json',dict(base,stage='calibrate',CAL_truth_access=True,TEST_truth_access=False))
        stage='calibrate';cal_path=H/'calibrate_v1'
        cal_owner=child_stage(cfg,metadata,'calibrate',[python,'-B',str(readout/'readout.py'),
            '--config',str(metadata/'READOUT_CONFIG.json'),'--admission',str(metadata/'CAL_ADMISSION.json'),
            '--stage','calibrate','--output',str(cal_path)],cal_path)
        cal=item(cal_path/'COMPLETE_CAL.json');value=read(checked(cal))
        require(value['terminal'] and value['scalar_endpoints']==60 and len(value['records'])==60,
                'Every fixed CAL attempt closed; finite/nonfinite outcomes retained without veto/subset')
        freeze_recheck(freeze)
        write(metadata/'TEST_ADMISSION.json',dict(base,stage='test',TEST_truth_access=True,one_complete_TEST_truth_release=True,
            no_prior_TEST_score_in_scope=True,calibration_terminal=cal,calibration_owner_closed_and_clean=True,
            calibration_owner=cal_owner,all60_fixed_endpoints_closed=True))
        stage='test';test_path=H/'test_v1'
        test_owner=child_stage(cfg,metadata,'test',[python,'-B',str(readout/'readout.py'),
            '--config',str(metadata/'READOUT_CONFIG.json'),'--admission',str(metadata/'TEST_ADMISSION.json'),
            '--stage','test','--output',str(test_path)],test_path)
        complete=item(test_path/'COMPLETE_TEST.json');value=read(checked(complete))
        require(value['terminal'] and value['one_complete_TEST_truth_release'] and len(value['rows'])==60,
                'One full60-bank raw/cal TEST report, no partial interpretation')
        freeze_recheck(freeze)
        write(metadata/'PIPELINE_END.json',{'UTC':now(),'status':'COMPLETE_FIXED_PIPELINE','terminal':True,
            'freeze':item(freeze_path),'prediction_closure':closure,'calibration':cal,'complete_TEST':complete,
            'owners':{'export':export_owner,'join':join_owner,'calibrate':cal_owner,'test':test_owner},
            'seconds':time.monotonic()-began,'extra_fit_updates':0,'one_TEST_release':True,'retry':False,
            'no_survivor_subset_or_recipe_change':True,'manuscript_verdict':None})
    except BaseException as error:
        write(metadata/'PIPELINE_FAILURE.json',{'UTC':now(),'terminal':True,'status':'FAILED_RETAINED','stage':stage,
            'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),
            'seconds':time.monotonic()-began,'retry':False,'next_phase_admitted':False,'survivor_subset':False})
        raise
    finally:
        signal.alarm(0)

if __name__=='__main__':main()
