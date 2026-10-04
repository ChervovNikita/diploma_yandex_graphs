"""Two sequential owned graph children. No retries or other-job operations."""
import os
import resource
import signal
import subprocess
import time
import traceback
from common import *


def group_alive(pgid):
    try:
        os.killpg(pgid,0)
        return True
    except ProcessLookupError:
        return False


def close_owned_group(process, timed_out, identity):
    receipt = dict(identity,cap_exceeded=timed_out,signals=[])
    for sig in (signal.SIGTERM,signal.SIGKILL):
        if not group_alive(process.pid):
            break
        try:
            os.killpg(process.pid,sig)
        except ProcessLookupError:
            break
        receipt['signals'].append(sig.name)
        deadline = time.monotonic()+GRACE_SECONDS
        while time.monotonic() < deadline and group_alive(process.pid):
            process.poll()
            time.sleep(.1)
    try:
        process.wait(timeout=GRACE_SECONDS)
        reaped = True
    except subprocess.TimeoutExpired:
        reaped = False
    receipt.update(parent_reaped=reaped,exit_code=process.returncode,group_absent=not group_alive(process.pid))
    return receipt


def main():
    gpu_snapshot()
    guard = json.loads((EXECUTION/'START_GUARD.json').read_text())
    write_new(EXECUTION/'EXECUTION_START.json',dict(supervisor_pid=os.getpid(),started_unix=time.time(),
        supervisor_identity=process_identity(os.getpid(),[str(PYTHON),'-B',str(HERE/'supervise.py')]),
        sequence=['Squirrel17','Photo17'],per_graph_cap_seconds=CAP_SECONDS,
        predictive_continuation=False,source_check=verify_sources()))
    started, results = time.monotonic(), []
    current = None
    current_identity = None
    finish = dict(status='UNRESOLVED',graphs=[])
    try:
        for graph in ('Squirrel','Photo'):
            verify_sources()
            gpu = gpu_snapshot()
            floor = guard['memory_floors_MiB'][graph]
            row = dict(graph=graph,seed=17,preflight_gpu=gpu,required_free_MiB=floor)
            results.append(row)
            write_new(EXECUTION/(graph+'_PREFLIGHT.json'),row)
            if gpu['free_MiB'] < floor:
                row.update(status='PREFLIGHT_MEMORY_FLOOR_NOT_MET',child_started=False)
                break
            argv = [str(PYTHON),'-B',str(HERE/'child.py'),graph]
            wall = time.monotonic()
            usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
            with (EXECUTION/(graph+'_STDOUT.log')).open('x') as log:
                current = subprocess.Popen(argv,cwd=str(PHASE),env=child_environment(),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                current_identity = dict(pid=current.pid,pgid=current.pid,argv=argv,start_ticks=None,
                    identity_recording='UNRESOLVED_IF_PROC_METADATA_READ_FAILS')
                current_identity = process_identity(current.pid,argv)
                row.update(current_identity)
                row.update(child_started=True,started_unix=time.time(),
                    memory_snapshot_age_seconds=time.time()-gpu['observed_unix'])
                write_new(EXECUTION/(graph+'_PID.json'),row)
                timed_out = False
                try:
                    current.wait(timeout=max(0,CAP_SECONDS-(time.monotonic()-wall)))
                except subprocess.TimeoutExpired:
                    timed_out = True
                closure = close_owned_group(current,timed_out,current_identity)
                write_new(EXECUTION/(graph+'_PHYSICAL_CLOSURE.json'),closure)
                row.update(closure=closure,cap_exceeded=timed_out,wall_seconds=time.monotonic()-wall)
                usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
                row.update(reaped_child_cpu_seconds=usage_after.ru_utime+usage_after.ru_stime-usage_before.ru_utime-usage_before.ru_stime,
                    children_peak_rss_bytes=usage_after.ru_maxrss*1024,
                    rss_scope='cumulative reaped-child high water; native report records own process/phase costs')
                current = None
                current_identity = None
            report_path = EXECUTION/graph/'QUALIFICATION_REPORT.json'
            row['qualification_status'] = json.loads(report_path.read_text())['status'] if report_path.exists() else None
            row['status'] = 'CAP_EXCEEDED' if timed_out else ('CHILD_COMPLETE' if closure['exit_code'] == 0 else 'CHILD_FAILED')
            row['sources_unchanged_after'] = verify_sources()
            write_new(EXECUTION/(graph+'_FINISH.json'),row)
            if not closure['group_absent'] or not closure['parent_reaped']:
                finish['status'] = 'OWNED_CHILD_CLOSURE_UNRESOLVED'
                break
        else:
            finish['status'] = 'TWO_ENGINEERING_CHILDREN_FINISHED'
        if finish['status'] == 'UNRESOLVED':
            finish['status'] = 'SEQUENCE_STOPPED'
    except BaseException as error:
        finish.update(status='SUPERVISOR_FAILED',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
    finally:
        if current is not None:
            finish['exception_closure'] = close_owned_group(current,False,current_identity)
        finish.update(graphs=results,finished_unix=time.time(),wall_seconds=time.monotonic()-started,
            supervisor_pid=os.getpid(),automatic_retries=0,predictive_continuation=False,
            all_engineering_reports_passed=len(results) == 2 and all(
                row.get('qualification_status') == 'PASSED_ACTUAL_NATIVE_WARM_ENGINEERING_ONLY'
                and row.get('status') == 'CHILD_COMPLETE' for row in results))
        write_new(EXECUTION/'EXECUTION_FINISH.json',finish)


if __name__ == '__main__':
    main()
