"""One admitted timeout-supervised lifecycle; no retries or source repair."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
import traceback
from support import guard, inside, receipt, require, load_admission, sources, ROOT, atomic_json


def process(pid):
    path=Path('/proc')/str(pid)
    try:
        fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        # fields[0] is kernel field3; starttime is field22.
        if fields[0]=='Z':return None
        return {'pid':int(pid),'start_ticks':fields[19],
            'rss_bytes':int(fields[21])*os.sysconf('SC_PAGE_SIZE'),
            'command':(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')}
    except (FileNotFoundError,ProcessLookupError):return None


def gpu_state():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
    row=[v.strip() for v in raw.strip().split(',')]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    processes=[]
    for line in apps.splitlines():
        if not line.strip():continue
        pid,memory=[v.strip() for v in line.split(',')]
        require(pid.isdigit() and memory.isdigit(),'Unmeasured GPU process usage')
        p=process(int(pid));require(p is not None,'GPU process could not be authenticated')
        processes.append({**p,'gpu_memory_bytes':int(memory)*1024**2})
    return {'uuid':row[0],'total_bytes':int(row[1])*1024**2,'used_bytes':int(row[2])*1024**2,
            'free_bytes':int(row[3])*1024**2,'processes':processes}


def preflight(admission):
    c=Path('/sys/fs/cgroup');current=int((c/'memory.current').read_text());maximum=(c/'memory.max').read_text().strip()
    stats=dict(line.split() for line in (c/'memory.stat').read_text().splitlines())
    host=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    hard=host if maximum=='max' else max(int(maximum)-current,0)
    reclaim=max(int(stats.get('inactive_file',0))-int(stats.get('shmem',0)),0)
    available=min(host,hard+reclaim)
    quota=(c/'cpu.max').read_text().split()
    capacity=len(os.sched_getaffinity(0)) if quota[0]=='max' else int(quota[0])/int(quota[1])
    peers=[];host_growth=0
    for pin in admission['peers']:
        p=process(pin['pid'])
        require(p is not None and p['start_ticks']==str(pin['start_ticks']),'Admitted peer PID changed; refresh admission')
        require(type(pin['peak_host_RSS_bytes']) is int and pin['peak_host_RSS_bytes']>=p['rss_bytes'],'Peer host envelope below observed RSS')
        growth=pin['peak_host_RSS_bytes']-p['rss_bytes'];host_growth+=growth
        peers.append({**p,'peak_host_RSS_bytes':pin['peak_host_RSS_bytes'],'host_growth_reserve_bytes':growth})
    after=available-host_growth
    disk=os.statvfs(ROOT);free_disk=disk.f_bavail*disk.f_frsize
    require(capacity>=2 and after>=24*1024**3 and free_disk>=2*1024**3,
        'Admission requires CPU2,24GiB cache-aware host headroom after peer growth,and2GiB disk')
    out={'CPU_capacity':capacity,'host_available_bytes':host,'cgroup_current_bytes':current,'cgroup_max':maximum,
        'hard_headroom_bytes':hard,'inactive_file_reclaimable_bytes':reclaim,'cache_aware_available_bytes':available,
        'host_peer_growth_reserve_bytes':host_growth,'available_after_peer_growth_bytes':after,
        'minimum_host_available_bytes':24*1024**3,'own_host_RSS_reservation_bytes':20*1024**3,
        'free_disk_bytes':free_disk,'peers':peers,'memory_events':(c/'memory.events').read_text()}
    if admission['device']=='cuda:0':
        evidence=admission['NCNC_CUDA_resource_evidence'];evidence_path=inside(evidence['path'])
        require(receipt(evidence_path)==evidence,'Pinned NCNC CUDA resource evidence changed')
        require(admission.get('NCNC_CUDA_resource_evidence_applicable') is True,'Root must explicitly admit applicability of native NCNC CUDA evidence')
        need=admission['own_GPU_reservation_bytes']
        require(type(need) is int and need>0,'Measured/admitted GPU memory envelope required')
        state=gpu_state(); require(len(state['processes'])<=1,'Never add a third GPU model worker')
        pinned={row['pid']:row for row in admission['peers']};growth=0
        for row in state['processes']:
            require(row['pid'] in pinned and str(pinned[row['pid']]['start_ticks'])==row['start_ticks'],'Unadmitted active GPU peer')
            peak=pinned[row['pid']]['peak_GPU_bytes']
            require(type(peak) is int and peak>=row['gpu_memory_bytes'],'Peer GPU envelope below observed memory')
            growth+=peak-row['gpu_memory_bytes']
        require(state['free_bytes']-growth>=need,'GPU free headroom after peer growth below admitted NCNC reservation')
        out.update(GPU=state,GPU_peer_growth_reserve_bytes=growth,own_GPU_reservation_bytes=need,
            GPU_available_after_growth_bytes=state['free_bytes']-growth,resource_evidence=evidence)
    return out


def main():
    guard()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--admission',required=True)
    args=parser.parse_args();here=inside(Path(__file__).parent)
    output=inside(args.output)
    require(not output.exists(),'One fresh output; preserve all earlier attempts')
    output.mkdir(parents=True)
    started=time.monotonic();child=None
    result={'schema':'runtime-link-v6-native-lifecycle-supervisor-v1','status':'preflight',
        'supervisor_pid':os.getpid(),'script':receipt(__file__),'native_child_started':False,
        'automatic_retries':0,'TEST_accessed':False,'scientific_fits':0,'full_campaign_GO':False}
    def write():
        result['wall_seconds']=time.monotonic()-started
        atomic_json(output/'SUPERVISOR.json',result)
    write()
    try:
        admission,job_path=load_admission(args.admission,here)
        require(inside(admission['output'])==output,'Admission output differs')
        result.update(admission=receipt(args.admission),job=receipt(job_path),device=admission['device'],
            publication_commit=admission['publication_commit'],source_receipts=sources(here,admission['driver_manifest_sha256']))
        pre=preflight(admission);result['preflight']=pre
        # Exclusive campaign identity prevents a second device/attempt, even with a new output.
        claim=here.parent/'runtime_link_v6_native_lifecycle_ATTEMPT_CLAIM.json'
        with claim.open('x') as stream:
            json.dump({'admission':receipt(args.admission),'output':str(output),'supervisor_pid':os.getpid(),
                'device':admission['device'],'publication_commit':admission['publication_commit']},stream,indent=2)
        result['one_attempt_claim']=receipt(claim)
        shutil_copy=output/'ADMISSION.json'
        with shutil_copy.open('xb') as stream:stream.write(Path(args.admission).read_bytes())
        watchdog=5400 if admission['device']=='cpu' else 1800
        command=[str(ROOT/'.venv/bin/python'),'-B','-u',str(here/'qualify.py'),
                 '--output',str(output/'qualification'),'--admission',str(shutil_copy)]
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='' if admission['device']=='cpu' else '0',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        with (output/'child.log').open('xb') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        result.update(status='running',native_child_started=True,child_pid=child.pid,
            child_identity=process(child.pid),command=command,watchdog_seconds=watchdog)
        launched=time.monotonic();deadline=launched+watchdog;write()
        peak_rss=0;peak_GPU=0
        while child.poll() is None and time.monotonic()<deadline:
            current=process(child.pid)
            if current:peak_rss=max(peak_rss,current['rss_bytes'])
            if admission['device']=='cuda:0':
                state=gpu_state();own=[r['gpu_memory_bytes'] for r in state['processes'] if r['pid']==child.pid]
                peak_GPU=max([peak_GPU]+own)
            result.update(observed_child_peak_RSS_bytes=peak_rss,observed_child_peak_GPU_bytes=peak_GPU,
                child_elapsed_seconds=time.monotonic()-launched)
            write()
            try:child.wait(timeout=min(10,max(.01,deadline-time.monotonic())))
            except subprocess.TimeoutExpired:pass
        timed_out=child.poll() is None
        if timed_out:
            result['timeout']=True
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait();result['forced_KILL_after_TERM']=True
        result.update(child_exit_code=child.returncode,child_wall_seconds=time.monotonic()-launched)
        child_result=output/'qualification/RESULT.json'
        numerical=json.loads(child_result.read_text()) if child_result.exists() else None
        result['child_result_receipt']=receipt(child_result) if child_result.exists() else None
        result['last_child_report']=numerical
        result['status']='passed' if not timed_out and child.returncode==0 and numerical and numerical['status']=='passed' else 'failed'
        result['source_receipts_after']=sources(here,admission['driver_manifest_sha256'])
        require(result['source_receipts']==result['source_receipts_after'],'Source bytes changed during attempt')
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        result.update(status='failed',failure={'type':type(error).__name__,'message':str(error),
            'traceback':traceback.format_exc(limit=16)})
    finally:
        result['memory_events_after']=Path('/sys/fs/cgroup/memory.events').read_text()
        usage=resource.getrusage(resource.RUSAGE_CHILDREN)
        result['child_os_cost']={'user_seconds':usage.ru_utime,'system_seconds':usage.ru_stime,'peak_RSS_KiB':usage.ru_maxrss,'scope':'supervisor children including read-only GPU probes'}
        write()
        report_path=output/'qualification/RESULT.json'
        last=json.loads(report_path.read_text()) if report_path.exists() else {}
        cost={key:last.get(key) for key in ('native_constructions_attempted','native_constructions_completed','TRAIN_epochs_started',
            'TRAIN_epochs_completed','updates_attempted','updates_completed','optimizer_attempts','optimizer_completed',
            'reverse_attempts','reverse_completed','extra_reverse_attempts','VALID_attempts','VALID_completed',
            'mean_exports_attempted','mean_exports_completed','encoder_forwards','elapsed_seconds','process_peak_RSS_KiB',
            'cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','invocations','failure')}
        cost.update(schema='runtime-link-v6-native-lifecycle-attempt-cost-v1',status=result['status'],
            native_child_started=result['native_child_started'],supervisor_wall_seconds=result['wall_seconds'],
            child_os_cost=result['child_os_cost'],observed_child_peak_RSS_bytes=result.get('observed_child_peak_RSS_bytes'),
            observed_child_peak_GPU_bytes=result.get('observed_child_peak_GPU_bytes'),
            timeout=result.get('timeout',False),forced_KILL_after_TERM=result.get('forced_KILL_after_TERM',False),
            work_counts_are_last_durable_observations=bool(last and last.get('status')=='running'),
            unrecorded_inflight_work_possible=bool(result.get('forced_KILL_after_TERM') or (last and last.get('status')=='running')),
            automatic_retries=0,TEST_accessed=False,scientific_fits=0)
        atomic_json(output/'COST.json',cost)
    print(json.dumps({key:result.get(key) for key in ('status','native_child_started','child_pid','wall_seconds','failure')}),flush=True)
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    sys.exit(main())
