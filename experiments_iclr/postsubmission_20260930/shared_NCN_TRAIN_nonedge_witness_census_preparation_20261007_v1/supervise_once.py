#!/usr/bin/env python3
"""CPU census binding to existing reviewed run_fit/ownership primitives."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

REPO=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
PREP=PHASE/'shared_NCN_TRAIN_nonedge_witness_census_preparation_20261007_v1'
EXEC=PHASE/'shared_NCN_TRAIN_nonedge_witness_census_execution_root_20261007_v1'
OLD=PHASE/'citeseer_known_ranking_control_matched_reference_owned_preparation_20261006_v2/owner.py'
OLD_SHA='1b1edf43a895600515bbf340fbc813b5dbc48e325ef73603fc5118146d4d5fdf'
HELPER=PHASE/'shared_private_transfer_gpu77_qualification_preparation_20261005_v3/ownership_helpers.py'
HELPER_SHA='e71503c87865546319cddbbf7a4f9f15d13cdf9e4875e406d65de21e64e047fd'
GPU='GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release',type=Path,required=True);args=parser.parse_args()
    r=read(args.release)
    if args.release.resolve(strict=True)!=EXEC/'ROOT_RELEASE.json' or r.get('execution_enabled') is not True or r.get('source_review_approved') is not True:
        raise ValueError('Exact root-approved CPU census release required')
    if r.get('VALID_TEST_values_access') is not False or r.get('fits')!=0 or r.get('retry') is not False:raise ValueError('TRAIN-only zero-fit one-child scope required')
    if sha(__file__)!=r['supervisor_sha256'] or sha(OLD)!=OLD_SHA or sha(HELPER)!=HELPER_SHA or sha(PREP/'SOURCE_MANIFEST.json')!=r['source_manifest_sha256']:
        raise ValueError('Reviewed source/primitives changed')
    for row in read(PREP/'SOURCE_MANIFEST.json')['files']:
        p=(PREP/row['path']).resolve(strict=True)
        if not p.is_relative_to(PREP) or sha(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise ValueError('Sealed census file changed')
    spec=importlib.util.spec_from_file_location('census_existing_owner',OLD);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.physical_host();h,_=old.lane(GPU);owner=h.identity(os.getpid())
    if owner is None or owner['pgid']!=os.getpid() or owner['sid']!=os.getpid():raise ValueError('Fresh detached supervisor session required')
    contract=read(PREP/'COMMANDS_DISABLED.json')
    if sha(PREP/'COMMANDS_DISABLED.json')!=r['command_sha256'] or r['resource_limits']!=contract['resource_limits'] or r['external_hard_seconds']!=60:
        raise ValueError('CPU/output/telemetry budget changed')
    preview=PREP/'JOB_DISABLED.json'
    if sha(preview)!=contract['job_preview_sha256']:raise ValueError('Root scientific preview changed')
    root=EXEC/'owner'
    if root.exists():raise ValueError('No prior owner or retry')
    root.mkdir();(root/'logs').mkdir()
    write(root/'ROOT_ADOPTION.json',r);write(root/'OWNER_STARTED.json',{'identity':owner,'root_release_sha256':sha(args.release),'CPU_only':True,'fits':0})
    try:
        job=copy.deepcopy(read(preview));job.update(execution_enabled=True,source_review_approved=True)
        jobpath=EXEC/'JOB.json';output=EXEC/'result'
        if jobpath.exists() or output.exists():raise ValueError('Fresh census job/output only')
        write(jobpath,job)
        context=SimpleNamespace(REPO=REPO,SOURCE=PREP,SOURCE_SHA=r['source_manifest_sha256'],GPU_UUID=GPU,GPU_UUIDS=old.GPU_UUIDS,
            phase_file=old.phase_file,physical_host=old.physical_host,sha=sha,write=write)
        command=contract['argv'];entry={'cell_id':'TRAIN_nonedge_census','argv':command,'job_relative':str(jobpath.relative_to(PHASE)),
            'job_sha256':sha(jobpath),'hard_seconds':60}
        env=dict(os.environ,**contract['env']);env.pop('PYTHONHOME',None)
        terminal=old.run_fit(h,root,entry,{'resource_limits':r['resource_limits']},env,output,context)
        identity=terminal['raw_identity_observation']
        if identity is None:raise ValueError('Fresh census child identity not observed')
        absent=h.identity(identity['PID']) is None
        rows=h.query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],5)
        no_cuda=not any(len(p)>1 and p[1].strip()==str(identity['PID']) for p in (row.split(',') for row in rows))
        write(root/'PHYSICAL_TERMINAL.json',{'PID':identity['PID'],'owned_PID_absent':absent,'owned_PID_no_CUDA_rows':no_cuda,'terminal_wait_observed':terminal['terminal_wait_observed']})
        if terminal['exit_code']!=0 or terminal['reason'] is not None or terminal['signals_sent'] or not absent or not no_cuda or not terminal['terminal_wait_observed'] or (output/'FAILURE.json').exists():
            raise ValueError('Census child failed clean completion; retain paid failure')
        result=read(output/'CENSUS.json')
        if result.get('complete') is not True or result.get('source_manifest_sha256')!=r['source_manifest_sha256'] or result.get('job_sha256')!=sha(jobpath) or result.get('VALID_TEST_values_access') is not False:
            raise ValueError('Complete census custody differs')
        write(root/'COMPLETE.json',{'passed':True,'census_sha256':sha(output/'CENSUS.json'),'terminal':terminal,'fits':0,'VALID_TEST_values_access':False})
    except (Exception,KeyboardInterrupt) as error:
        write(root/'FAILURE.json',{'error':type(error).__name__+': '+str(error),'partial_outputs_preserved':True,'retry':False,'fits':0})
        raise

if __name__=='__main__':main()
