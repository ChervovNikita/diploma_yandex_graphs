"""Finite root-authorized MolBACE zero-update queue around the frozen helper.

Only stdlib is used by this guardian. Scientific outputs are neither opened nor
interpreted. Physics is never modified or signaled. No case retry is possible.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
HELPER='practical_modal_resource_qualifier_source_20261012_v1/runtime/qa/practical_modal_zero_update_resources.py'
HELPER_SHA='ce4f1298a0ef62ca308f23266337fd2909e1c193bd564da6f2dce6dcca802cdc'
BLUEPRINT='practical_resource_extension_blueprint_20261012_v2/BLUEPRINT.json'
BLUEPRINT_SHA='b188a3f3ba9a01e064df9545b1f5213e7b43efc3278b0237c301b9f49244891a'
PHYSICS_START_SHA='f939f7833b2a1ffcabf1e6fc836c11e5a9650c339003110f30487b03b36d50f3'
PHYSICS={'owner':{'pid':640426,'start_ticks':6055039154,'pgid':640426},'child':{'pid':640428,'start_ticks':6055039165,'pgid':640428}}
WATCHER={'pid':641081,'start_ticks':6055370992,'pgid':641081}
CASES=('2deb57d73a0194d8c3607ef8','ec8d8b2038646ac40533c365','3488c5be9984e3db5daa30c6','0f72ece7ac64212461b8473a','53e6dad380b5a7503757a849','25d56d3f32f86b997df0cdba','6a9c83519cf833b2bcdf8d34')
OUTPUT='practical_modal_resource_minimized_molbace_20261012_v1'
TIMEOUT=180

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def save(path,value):
    temporary=path.with_name(path.name+'.temporary');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def binding(path):
    raw=path.read_bytes();return {'path':str(path.relative_to(ROOT)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def identity(pid):
    raw=(Path('/proc')/str(pid)/'stat').read_text();fields=raw.rsplit(')',1)[1].split()
    return {'pid':pid,'start_ticks':int(fields[19]),'pgid':int(fields[2]),'state':fields[0]}
def same_identity(actual,expected):return all(actual[k]==v for k,v in expected.items())

def physics_custody():
    directory=ROOT/'private_GRU_acquisition_root_20261011_v1/owner_v1'
    start=directory/'OWNER_START.json';assert binding(start)['sha256']==PHYSICS_START_SHA
    initial=json.loads(start.read_text());assert all(initial[k]==v for k,v in PHYSICS.items())
    end=directory/'OWNER_END.json'
    if end.exists():
        terminal=json.loads(end.read_text())
        assert all(terminal.get(k)==v for k,v in PHYSICS.items())
        assert terminal.get('command')==initial['command']
        assert terminal.get('status') in {'COMPLETE_EXIT_0','FAILED_RETAINED'}
        assert terminal.get('child_pid_absent') is True and terminal.get('owned_cuda_pid_absent') is True
        retained={}
        for name,expected in PHYSICS.items():
            try:current=identity(expected['pid'])
            except FileNotFoundError:retained[name]={'original_identity':expected,'process_present':False}
            else:
                assert not (same_identity(current,expected) and current['state']!='Z')
                retained[name]={'original_identity':expected,'process_present':True,'same_original_identity':same_identity(current,expected)}
        watch_directory=ROOT/'private_GRU_readout_after_whole_acquisition_root_20261011_v1'
        launch=watch_directory/'LAUNCH.json';watch_start=watch_directory/'watch_v1/WATCH_START.json'
        launch_data=json.loads(launch.read_text());watch_data=json.loads(watch_start.read_text())
        assert same_identity(launch_data['watcher_observed'],WATCHER) and watch_data['pid']==WATCHER['pid']
        assert watch_data['acquisition_owner_start']['sha256']==PHYSICS_START_SHA
        watch_end=watch_directory/'watch_v1/WATCH_END.json'
        if watch_end.exists():
            assert json.loads(watch_end.read_text()).get('pid')==WATCHER['pid']
            watcher_state={'mode':'retained_terminal_watcher_metadata','end':binding(watch_end)}
        else:
            watcher_current=identity(WATCHER['pid']);assert same_identity(watcher_current,WATCHER) and watcher_current['state']!='Z'
            watcher_state={'mode':'exact_original_watcher_alive','identity':watcher_current}
        return {'mode':'root_authorized_exact_terminal_ownership_receipt_only','start':binding(start),'end':binding(end),'original_acquisition_identities':retained,'original_watcher':dict(watcher_state,launch=binding(launch),start=binding(watch_start)),'scientific_outcomes_interpreted':False,'terminal_failure_never_authorizes_restart':True}
    current={}
    for name,expected in PHYSICS.items():
        observed=identity(expected['pid']);assert same_identity(observed,expected) and observed['state']!='Z'
        current[name]=observed
    return {'mode':'both_exact_Physics_processes_alive','identities':current,'start':binding(start),'scientific_outcomes_interpreted':False}

def no_other_GPU_queue():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    records=[]
    for line in raw.splitlines():
        if not line.strip():continue
        pid,uuid,memory=[x.strip() for x in line.split(',')]
        assert uuid=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
        record={'pid':int(pid),'gpu_uuid':uuid,'used_MiB':memory};records.append(record)
        if int(pid)!=PHYSICS['child']['pid']:raise RuntimeError('Another CUDA process prevents zero-overlap modal queue admission')
    return records

def validate_success(value):
    assert value['status']=='TRAIN_PROXY_COMPLETE' and value['family']=='molecular'
    assert value['helper_sha256']==HELPER_SHA and value['blueprint_sha256']==BLUEPRINT_SHA
    assert value['zero_update_equality_proof_complete'] is True
    assert value['parameters_sha256_before']==value['parameters_sha256_after'] and value['moments_sha256_before']==value['moments_sha256_after']
    for key in ['actual_optimizer_updates','actual_optimizer_kernel_calls','heldout_original_file_accesses','scientific_fits','accuracy_calls','scores_emitted']:assert value[key]==0
    assert value['denied_optimizer_step_calls']>0
    cap=value['allocator_cap'];assert cap['allocator_cap_bytes']==min(cap['fresh_free_bytes']-4*2**30,math.floor(.4*cap['total_bytes']))
    assert max(value['CUDA_peak_allocated_bytes'],value['CUDA_peak_reserved_bytes'])<=cap['allocator_cap_bytes']
    assert value['original_selector_or_endpoint_qualification'] is False and value['all_fit_ETA'] is None

def main():
    assert subprocess.check_output(['hostname'],text=True).strip()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
    assert binding(ROOT/HELPER)['sha256']==HELPER_SHA and binding(ROOT/BLUEPRINT)['sha256']==BLUEPRINT_SHA
    directory=ROOT/OUTPUT;directory.mkdir(exist_ok=False)
    started=time.monotonic();owner=identity(os.getpid());source=binding(Path(__file__))
    save(directory/'OWNER_START.json',{'owner':owner,'source':source,'case_ids':list(CASES),'timeout_seconds_per_case':TIMEOUT,'maximum_case_wall_seconds':len(CASES)*TIMEOUT,'root_authorized_exact_cases':True,'scientific_fits':0,'scientific_launch_enabled':False})
    closed=[];state='RUNNING';stop=None
    save(directory/'QUEUE_PROGRESS.json',{'status':state,'owner':owner,'closed':closed,'next_case':CASES[0]})
    try:
        for index,case_id in enumerate(CASES,1):
            before=physics_custody();usage=no_other_GPU_queue()
            output=f'{OUTPUT}/case_{index:02}_{case_id}'
            command=[str(ROOT.parents[1]/'.venv/bin/python'),'-B',str(ROOT/HELPER),'--project-root',str(ROOT),'--output',output,'--case-id',case_id,'--reviewed-helper-sha256',HELPER_SHA,'--case-timeout-seconds',str(TIMEOUT)]
            receipt={'case_id':case_id,'command':command,'Physics_custody_before':before,'GPU_processes_before':usage,'automatic_retry':False,'source':source,'scientific_fits':0}
            path=directory/f'CASE_{index:02}_COMMAND_RECEIPT.json';save(path,receipt)
            child=subprocess.Popen(command,cwd=(ROOT/HELPER).parents[1],env=dict(os.environ,PYTHONPATH=str((ROOT/HELPER).parents[1]),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0'),stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
            child_identity=identity(child.pid);receipt['child']=child_identity;save(path,receipt)
            save(directory/'QUEUE_PROGRESS.json',{'status':state,'owner':owner,'closed':closed,'active_case_id':case_id,'active_child':child_identity,'active_case_index':index})
            case_started=time.monotonic();external_timeout=False
            try:stdout,stderr=child.communicate(timeout=TIMEOUT)
            except subprocess.TimeoutExpired:
                external_timeout=True
                if (Path('/proc')/str(child.pid)/'stat').exists():
                    current=identity(child.pid);assert same_identity(current,{k:child_identity[k] for k in ['pid','pgid','start_ticks']})
                    os.killpg(child_identity['pgid'],signal.SIGKILL)
                stdout,stderr=child.communicate(timeout=10)
            after=physics_custody()
            completion={'case_id':case_id,'child':child_identity,'exit_code':child.returncode,'external_timeout':external_timeout,'wall_seconds':time.monotonic()-case_started,'stdout':stdout,'stderr':stderr,'Physics_custody_after':after,'automatic_retry':False}
            save(directory/f'CASE_{index:02}_PROCESS_COMPLETION.json',completion)
            result=ROOT/output/'RESULT.json';failure=ROOT/output/'TERMINAL_FAILURE.json'
            if result.exists():
                value=json.loads(result.read_text());validate_success(value);assert value['case_id']==case_id and child.returncode==0 and not external_timeout
                assert same_identity(value['owner'],{k:child_identity[k] for k in ['pid','pgid','start_ticks']})
                row={'case_id':case_id,'status':'TRAIN_PROXY_COMPLETE','result':binding(result),'wall_seconds':completion['wall_seconds'],'Physics_before_and_after_verified':True,'zero_update_parameter_and_moment_equality_verified':True}
            elif failure.exists():
                value=json.loads(failure.read_text());row={'case_id':case_id,'status':value['status'],'failure':binding(failure),'wall_seconds':completion['wall_seconds'],'Physics_before_and_after_verified':True,'zero_update_equality_proof_complete':False}
                if value['status']!='RESOURCE_DEFERRED':stop='ENGINEERING_INTERFACE_OR_CONTRACT_FAILURE'
            else:
                value={'status':'RESOURCE_DEFERRED' if external_timeout else 'ENGINEERING_FAILURE','case_id':case_id,'owner':child_identity,'failure_type':'External180SecondDeadlineBeforeRESULT' if external_timeout else 'OwnedProcessExitedBeforeRESULT','zero_update_equality_proof_complete':False,'actual_optimizer_updates_verified':None,'wall_seconds':completion['wall_seconds'],'scientific_fits':0,'scores_emitted':0,'automatic_retry':False}
                failure.parent.mkdir(exist_ok=True);save(failure,value);row={'case_id':case_id,'status':value['status'],'failure':binding(failure),'wall_seconds':completion['wall_seconds'],'Physics_before_and_after_verified':True,'zero_update_equality_proof_complete':False}
                if not external_timeout:stop='ENGINEERING_INTERFACE_OR_CONTRACT_FAILURE'
            closed.append(row);save(directory/'QUEUE_PROGRESS.json',{'status':'STOPPING' if stop else state,'owner':owner,'closed':closed,'next_case':None if stop or index==len(CASES) else CASES[index]})
            if stop:break
        state='CLOSED' if not stop else 'STOPPED_FOR_ENGINEERING_REVIEW'
    except BaseException as error:
        state='STOPPED_FOR_CUSTODY_OR_CONTRACT_REVIEW';stop={'type':type(error).__name__,'message':str(error)}
    finally:
        terminal={'status':state,'owner':owner,'source':source,'closed':closed,'stop_reason':stop,'unexecuted_case_ids':[x for x in CASES if x not in {v['case_id'] for v in closed}],'guardian_wall_seconds':time.monotonic()-started,'automatic_retry':False,'new_scientific_fits':0,'scientific_launch_enabled':False,'all_fit_ETA':None,'individual_molecular_scope':'at most8of18 actual cases including the prior MolHIV singleton; no initializer/dataset transfer or whole-roster qualification'}
        save(directory/'OWNER_END.json',terminal);save(directory/'QUEUE_PROGRESS.json',terminal)

if __name__=='__main__':main()
