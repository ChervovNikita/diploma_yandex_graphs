"""Original qualification owner pattern, finite complete18 scientific child."""
import argparse
from pathlib import Path
import datetime
import json
import os
import signal
import subprocess
import time
from ENTRY import HERE,PHASE,GPU,route,frozen,admitted,sha

R=PHASE.parent.parent
A=HERE


def write(path,value):
    with path.open('x') as output:json.dump(value,output,indent=2);output.write('\n')


def identity(pid):
    value=Path('/proc',str(pid),'stat').read_text();fields=value[value.rfind(')')+2:].split()
    return dict(pid=pid,start_ticks=int(fields[19]),group=int(fields[2]),session=int(fields[3]),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def check_owned(saved):assert identity(saved['pid'])==saved,'Never signal an unowned or reused process'


def output_size(folder):
    total=0
    for path in Path(folder).rglob('*'):
        try:
            if path.is_file():total+=path.stat().st_size
        except FileNotFoundError:pass  # Original atomic selected-state replacement.
    return total


def owned_memory(pid,timeout):
    try:
        rows=Path('/proc',str(pid),'status').read_text().splitlines()
    except FileNotFoundError:return None
    rss=next(int(row.split()[1])*1024 for row in rows if row.startswith('VmRSS:'))
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True,timeout=timeout)
    device=0
    for row in raw.splitlines():
        parts=[part.strip() for part in row.split(',')]
        if len(parts)==2 and parts[0].isdigit() and int(parts[0])==pid:device+=int(parts[1].split()[0])*1024**2
    return dict(RSS_bytes=rss,device_resident_bytes=device)


def main():
    route()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args();release=A/'RELEASE.json'
    assert sha(release)==args.release_sha256
    assert not (A/'LAUNCH.json').exists() and not (A/'TERMINAL.json').exists()
    cfg=json.loads(release.read_text());b,manifest_sha=frozen();admitted(cfg,b,manifest_sha)
    limits=b['finite_bounds'];output=cfg['output_directory']
    env=dict(os.environ,**cfg['runtime_environment'],PYTHONDONTWRITEBYTECODE='1',DGLBACKEND='pytorch')
    argv=[cfg['python_executable'],'-B',str(A/'ENTRY.py'),'--release-sha256',args.release_sha256]
    source_commit=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True,timeout=10).strip()
    started=time.monotonic();active_deadline=started+limits['active_seconds']
    child=saved=None;peak=dict(RSS_bytes=0,device_resident_bytes=0)
    stop_reason=None;cleanup_errors=[];directly_waited=False;cleanup_started=None
    with (A/'worker.stdout').open('x') as stdout,(A/'worker.stderr').open('x') as stderr:
        try:
            child=subprocess.Popen(argv,cwd=R,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            saved=identity(child.pid);assert saved['group']==child.pid and saved['session']==child.pid
            launch=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),child=saved,parent=identity(os.getpid()),argv=argv,
                        source_commit=source_commit,launcher_sha256=sha(__file__),release_sha256=args.release_sha256,
                        activation_source_manifest_sha256=manifest_sha,finite_bounds=limits,automatic_retry=False,
                        model_work='All18 full fixed candidate-condition/role fits; comparison awaits complete24 references',
                        normal_host_execution=True,filesystem_namespace_or_mount_changes=False)
            write(A/'LAUNCH.json',launch);print(json.dumps(dict(launched=launch)),flush=True)
            while child.poll() is None:
                remaining=active_deadline-time.monotonic()
                if remaining<=0:stop_reason='owned reference reached its declared12h active bound';break
                check_owned(saved);memory=owned_memory(child.pid,min(5,remaining))
                if memory is not None:
                    for key in peak:peak[key]=max(peak[key],memory[key])
                    if memory['RSS_bytes']>limits['host_RSS_bytes']:stop_reason='owned reference exceeded host RSS budget'
                    if memory['device_resident_bytes']>limits['GPU_bytes']:stop_reason='owned reference exceeded device budget'
                if (A/'worker.stdout').stat().st_size+(A/'worker.stderr').stat().st_size>limits['log_bytes']:stop_reason='owned reference exceeded log budget'
                if output_size(output)>limits['output_bytes']:stop_reason='owned reference exceeded output budget'
                if time.monotonic()>=active_deadline:stop_reason=stop_reason or 'owned reference reached active bound'
                if stop_reason:break
                time.sleep(min(2,max(0,active_deadline-time.monotonic())))
        except BaseException as error:stop_reason=type(error).__name__+': '+str(error)
        finally:
            # Popen and every later bookkeeping/monitor failure reach owned cleanup.
            cleanup_started=time.monotonic();cleanup_deadline=cleanup_started+limits['cleanup_seconds']
            if child is not None:
                try:
                    if child.poll() is None:
                        if saved is None:saved=identity(child.pid)
                        check_owned(saved);os.killpg(saved['group'],signal.SIGTERM)
                        try:child.wait(timeout=min(5,max(.001,cleanup_deadline-time.monotonic())))
                        except subprocess.TimeoutExpired:
                            check_owned(saved);os.killpg(saved['group'],signal.SIGKILL)
                    child.wait(timeout=max(.001,cleanup_deadline-time.monotonic()))
                    directly_waited=True
                except BaseException as error:
                    cleanup_errors.append(type(error).__name__+': '+str(error));stop_reason=stop_reason or 'owned cleanup/direct wait unconfirmed'
            cuda_absent=None
            try:
                remaining=cleanup_deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('cleanup cap exhausted before owned CUDA absence')
                raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=min(2,remaining))
                cuda_absent=child is not None and str(child.pid) not in [q.strip() for q in raw.splitlines()]
            except BaseException as error:cleanup_errors.append(type(error).__name__+': '+str(error))
            process_absent=child is not None and not Path('/proc',str(child.pid)).exists()
    if output_size(output)>limits['output_bytes']:stop_reason=stop_reason or 'owned output budget exceeded at completion'
    if (A/'worker.stdout').stat().st_size+(A/'worker.stderr').stat().st_size>limits['log_bytes']:stop_reason=stop_reason or 'owned log budget exceeded at completion'
    code=None if child is None else child.returncode
    good=code==0 and stop_reason is None and directly_waited and process_absent and cuda_absent and not cleanup_errors
    terminal=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete=bool(good),exit_code=code,
                  seconds=time.monotonic()-started,sampled_peak=peak,stop_reason=stop_reason,child_reaped=directly_waited,
                  original_pid_absent=process_absent,owned_CUDA_absence_verified=cuda_absent,cleanup_errors=cleanup_errors,
                  cleanup_seconds=time.monotonic()-cleanup_started,finite_bounds=limits,argv=argv,child=saved,
                  activation_source_manifest_sha256=manifest_sha,release_sha256=args.release_sha256,
                  all18_fixed_fits_required=True,comparison_waits_for_both_whole_families=True,other_processes_or_jobs_changed=False,comparative_scores_opened=False,
                  automatic_retry=False,normal_host_execution=True)
    if not good:write(A/'MONITOR_FAILURE.json',dict(reason=stop_reason or 'child/absence/cleanup unsuccessful',child=saved,cleanup_errors=cleanup_errors))
    write(A/'TERMINAL.json',terminal);print(json.dumps(dict(terminal=terminal)),flush=True)
    raise SystemExit(0 if good else 1)


if __name__=='__main__':main()
