"""Set only this worker's allocator cap, then run the sealed public CLI unchanged."""
import argparse
import os
from pathlib import Path
import resource
import runpy
import signal
import sys
import threading
import time
import controller as owner


def expired(signum, frame): raise TimeoutError('Frozen owned-fit active deadline')


def main():
    began=time.monotonic(); parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True); parser.add_argument('--job-sha256',required=True)
    args=parser.parse_args(); owner.require(owner.sha(args.job)==args.job_sha256,'Exact controller worker job')
    job=owner.read(args.job); pins=owner.read(owner.HERE/'SOURCE_BINDINGS.json'); limits=pins['limits']
    owner.require(job['schema']=='canonical-SupCon-full3-owned-worker-v1' and job['seed'] in owner.SEEDS,
                  'Fixed full3 worker roster')
    owner.seal(owner.HERE,job['controller_manifest_sha256'])
    parent=owner.proc(os.getppid())
    owner.require(parent is not None and owner.identity(parent)==job['parent_owner'],'Actual admitted controller parent')
    owner.require(owner.sha(job['admission_path'])==job['admission_sha256'],'Root admission changed')
    admission=owner.read(job['admission_path'])
    owner.require(admission['controller_manifest_sha256']==job['controller_manifest_sha256']
                  and admission['root_scientific_launch_authorized'] is True,'Exact root scientific launch')
    owner.require(job['protocol_sha256']==pins['protocol']['sha256'],'Frozen protocol binding')
    actual_providers=owner.verify_setup(pins)
    seed=job['seed']; gpu=pins['seed_GPU_map'][str(seed)]
    owner.require(job['physical_gpu_uuid']==gpu and os.environ.get('CUDA_VISIBLE_DEVICES')==gpu,
                  'Original per-seed physical GPU')
    owner.require(owner.free_bytes(gpu)>=limits['minimum_fresh_free_GPU_bytes'],'Fresh28GiB headroom before allocator setup')
    output=Path(job['output']); entry=Path(job['entry_directory'])
    family=owner.inside(admission['output_directory'])
    owner.require(output==family/'cells'/('supcon_eq2_'+str(seed)) and entry==family/'entry'
                  and not output.exists() and entry.is_dir(),'Fresh exact worker output paths')
    active_deadline=job['active_deadline_monotonic']; hard_deadline=job['hard_deadline_monotonic']
    owner.require(abs(hard_deadline-job['fit_started_monotonic']-limits['fit_hard_seconds'])<1e-6
                  and abs(active_deadline-job['fit_started_monotonic']-limits['fit_active_seconds'])<1e-6,
                  'Per-fit active/cleanup/hard envelope')
    owner.require(time.monotonic()<active_deadline,'Fit active deadline already expired')
    # Controller cleanup reserves10seconds. A self-only hard timer also bounds an orphaned entry.
    hard_timer=threading.Timer(max(0,hard_deadline-time.monotonic()),lambda:os.kill(os.getpid(),signal.SIGKILL))
    hard_timer.daemon=True; hard_timer.start()
    signal.signal(signal.SIGALRM,expired); signal.signal(signal.SIGTERM,expired)
    signal.setitimer(signal.ITIMER_REAL,max(.001,active_deadline-time.monotonic()))
    code=1; error=None; torch=None
    try:
        import torch
        owner.require(str(torch.__version__)==pins['runtime']['torch']
                      and torch.cuda.is_available() and torch.cuda.device_count()==1,'Actual original Torch and one mapped GPU')
        torch.cuda.set_device(0); total=torch.cuda.get_device_properties(0).total_memory
        cap=limits['owned_GPU_allocator_cap_bytes']; owner.require(cap<total,'Own allocator cap fits GPU')
        torch.cuda.set_per_process_memory_fraction(cap/total,0); torch.cuda.reset_peak_memory_stats(0)
        actual=owner.proc(os.getpid())
        owner.write(entry/('seed'+str(seed)+'_STARTED.json'),dict(seed=seed,owner=owner.identity(actual),
            parent_owner=job['parent_owner'],job_sha256=args.job_sha256,physical_gpu_uuid=gpu,
            allocator_cap_bytes=cap,providers=actual_providers,torch=str(torch.__version__),
            active_deadline_monotonic=active_deadline,hard_deadline_monotonic=hard_deadline,
            seed_or_session_modified=False,source_manifest_sha256=pins['supcon_source']['manifest_sha256']),fresh=True)
        source=owner.inside(pins['supcon_source']['path']); public=owner.inside(pins['public_source']['path'])
        sys.argv=[str(source/'train.py'),'--condition','supcon_eq2','--seed',str(seed),'--device','cuda:0',
            '--public-interface',str(public),'--polynormer',str(owner.bound(pins['inputs']['polynormer'])),
            '--train',str(owner.bound(pins['inputs']['train'])),'--valid',str(owner.bound(pins['inputs']['development'])),
            '--output',str(output)]
        runpy.run_path(str(source/'train.py'),run_name='__main__')
        code=0
    except BaseException as failure:
        error=type(failure).__name__+': '+str(failure)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        usage=resource.getrusage(resource.RUSAGE_SELF)
        allocated=int(torch.cuda.max_memory_allocated(0)) if torch is not None and torch.cuda.is_initialized() else None
        reserved=int(torch.cuda.max_memory_reserved(0)) if torch is not None and torch.cuda.is_initialized() else None
        if reserved is not None and reserved>limits['owned_GPU_allocator_cap_bytes']:
            code=1; error=error or 'Allocator cap exceeded'
        record=dict(seed=seed,exit_code=code,error=error,job_sha256=args.job_sha256,
            controller_manifest_sha256=job['controller_manifest_sha256'],protocol_sha256=job['protocol_sha256'],
            physical_gpu_uuid=gpu,allocator_cap_bytes=limits['owned_GPU_allocator_cap_bytes'],
            peak_CUDA_allocated_bytes=allocated,peak_CUDA_reserved_bytes=reserved,
            CPU_user_seconds=usage.ru_utime,CPU_system_seconds=usage.ru_stime,peak_RSS_bytes=int(usage.ru_maxrss*1024),
            inclusive_entry_seconds=time.monotonic()-began,seed_or_session_modified=False,
            original_public_CLI_run_unchanged=True,quality_printed=False,automatic_retry=False)
        owner.write(entry/('seed'+str(seed)+'_TERMINAL.json'),record,fresh=True)
        hard_timer.cancel()
    print(owner.json.dumps(dict(event='entry_terminal',seed=seed,exit_code=code,
                               allocator_cap_bytes=limits['owned_GPU_allocator_cap_bytes'],quality_printed=False)),flush=True)
    return code


if __name__=='__main__': sys.exit(main())
