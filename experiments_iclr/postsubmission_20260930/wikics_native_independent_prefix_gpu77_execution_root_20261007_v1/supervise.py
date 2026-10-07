#!/usr/bin/env python3
"""Thin detached invocation of the existing reviewed run_fit/helper for two fixed banks."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib, importlib.util, json, os, socket, subprocess, time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
QUEUE=json.loads((ROOT/'QUEUE.json').read_text())
REPO=Path(QUEUE['repository']);PHASE=REPO/'experiments_iclr/postsubmission_20260930'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n')
def phase_file(relative):
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Exact phase-relative file required')
    p=(PHASE/rel).resolve(strict=True)
    if not p.is_relative_to(PHASE.resolve()) or not p.is_file():raise ValueError('File leaves authorized phase')
    return p
def physical_host():
    if Path.cwd().resolve()!=REPO.resolve() or socket.gethostname()!=QUEUE['hostname']:raise ValueError('Exact authorized GPU77 host/repository required')
    if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()!=QUEUE['physical_gpu_inventory']:raise ValueError('Exact full GPU77 inventory required')
def load_module(name,binding):
    p=phase_file(binding['path'])
    if sha(p)!=binding['sha256']:raise ValueError('Existing reviewed supervision dependency changed')
    spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def worker(entry,owner):
    helper=load_module('wikics_prefix_helper_'+str(entry['seed']),QUEUE['existing_ownership_helper']);helper.GPU=entry['physical_gpu_uuid']
    source=PHASE/QUEUE['source_directory'];job=phase_file(entry['job_relative'])
    if sha(source/'MANIFEST.json')!=QUEUE['source_manifest_sha256'] or sha(source/'run.py')!=QUEUE['program_sha256'] or sha(job)!=entry['job_sha256']:raise ValueError('Exact root/source/job binding changed')
    output=Path(entry['output_directory'])
    if output.exists():raise ValueError('Fresh output; no retry')
    physical_host();context=SimpleNamespace(REPO=REPO,SOURCE=source,SOURCE_SHA=QUEUE['source_manifest_sha256'],GPU_UUID=entry['physical_gpu_uuid'],GPU_UUIDS=tuple(QUEUE['physical_gpu_inventory']),phase_file=phase_file,physical_host=physical_host,sha=sha,write=write)
    env=dict(os.environ,**QUEUE['environment'],CUDA_VISIBLE_DEVICES=entry['physical_gpu_uuid']);env.pop('PYTHONHOME',None)
    receipt=owner.run_fit(helper,ROOT,entry,QUEUE,env,output,context)
    identity=receipt.get('raw_identity_observation');absent=identity is not None and helper.identity(identity['PID']) is None
    rows=helper.query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],QUEUE['resource_limits']['telemetry_timeout_seconds'])
    no_cuda=identity is not None and not any(len(p)>=2 and p[1].strip()==str(identity['PID']) for p in (r.split(',') for r in rows))
    physical={'UTC':helper.now(),'cell_id':entry['cell_id'],'child_identity':identity,'owned_PID_absent':absent,'owned_PID_no_CUDA_rows':no_cuda,'physical_gpu_uuid':entry['physical_gpu_uuid'],'terminal_wait_observed':receipt['terminal_wait_observed']};write(ROOT/'logs'/(entry['cell_id']+'.PHYSICAL_TERMINAL.json'),physical)
    ok=receipt['exit_code']==0 and receipt['reason'] is None and not receipt['signals_sent'] and receipt['terminal_wait_observed'] and absent and no_cuda and not (output/'FAILURE.json').exists()
    summary={'cell_id':entry['cell_id'],'exit_code':receipt['exit_code'],'reason':receipt['reason'],'physical_terminal_complete':ok,'quality_values_read':False}
    if ok:
        freeze_path=output/'DONOR_FREEZE.json';freeze=json.loads(freeze_path.read_text())
        if freeze.get('complete') is not True or len(freeze['members'])!=4 or freeze['arm']!='DONOR_INDEPENDENT' or freeze['seed']!=entry['seed']:raise ValueError('Incomplete native prefix bank')
        for m,binding in enumerate(freeze['members']):
            f=phase_file(binding['path']);record=json.loads(f.read_text())
            if sha(f)!=binding['sha256'] or record['epochs']!=1100 or record['seed']!=entry['seed']+1009*m or record['source_manifest_sha256']!=QUEUE['source_manifest_sha256']:raise ValueError('Native prefix member custody differs')
        summary.update(complete=True,members=4,epochs_each=1100,donor_freeze={'path':str(freeze_path.relative_to(PHASE)),'sha256':sha(freeze_path)})
    else:summary['complete']=False
    write(ROOT/'logs'/(entry['cell_id']+'.SUMMARY.json'),summary);return summary
if __name__=='__main__':
    try:
        physical_host()
        for row in json.loads((ROOT/'ACTIVATION_MANIFEST.json').read_text())['files']:
            p=ROOT/row['path']
            if sha(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise ValueError('Immutable root activation changed')
        owner=load_module('existing_WikiCS_prefix_run_fit',QUEUE['existing_run_fit'])
        write(ROOT/'OWNER_STARTED.json',{'PID':os.getpid(),'UTC':time.time(),'fixed_cells':[e['cell_id'] for e in QUEUE['entries']],'source_manifest_sha256':QUEUE['source_manifest_sha256'],'quality_values_read':False})
        results=[]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(worker,e,owner):e for e in QUEUE['entries']}
            for future in as_completed(futures):
                try:results.append(future.result())
                except Exception as error:results.append({'cell_id':futures[future]['cell_id'],'complete':False,'error':type(error).__name__+': '+str(error),'quality_values_read':False})
        write(ROOT/'OWNER_COMPLETE.json',{'complete':all(r.get('complete') is True for r in results),'results':results,'retry':False,'quality_values_read':False})
    except BaseException as error:
        write(ROOT/'OWNER_FAILURE.json',{'error':type(error).__name__+': '+str(error),'retry':False,'quality_values_read':False});raise
