"""Future explicit launch only. One marker; one owned detached supervisor."""
import os
import subprocess
import time
from common import *


def main():
    snapshot = gpu_snapshot()  # Before project data/receipt operations.
    floors = dict(Squirrel=MINIMUM_FREE_MIB,Photo=MINIMUM_FREE_MIB)
    require(MINIMUM_FREE_MIB <= snapshot['total_MiB'], 'GPU smaller than reviewed headroom floor')
    sources = verify_sources()
    EXECUTION.mkdir(exist_ok=True)
    write_new(EXECUTION/'START_GUARD.json',dict(launch_pid=os.getpid(),launch_unix=time.time(),
        memory_floors_MiB=floors,per_graph_cap_seconds=CAP_SECONDS,closure_grace_seconds=GRACE_SECONDS,
        initial_gpu=snapshot,source_check=sources,no_automatic_retries=True))
    if snapshot['free_MiB'] < floors['Squirrel']:
        blocked = dict(status='PREFLIGHT_MEMORY_FLOOR_NOT_MET',graphs_started=0,
            physical_children_created=False,gpu=snapshot,once_marker_consumed=True)
        write_new(EXECUTION/'EXECUTION_FINISH.json',blocked)
        print(json.dumps(blocked))
        return 2
    for path in ('tmp','cache','cache/torch','cache/torch_extensions','cache/cuda'):
        (EXECUTION/path).mkdir(parents=True,exist_ok=True)
    argv = [str(PYTHON),'-B',str(HERE/'supervise.py')]
    with (EXECUTION/'SUPERVISOR.log').open('x') as log:
        process = subprocess.Popen(argv,cwd=str(PHASE),env=child_environment(),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt = dict(pid=process.pid,pgid=process.pid,launcher_pid=os.getpid(),argv=argv,
            detached=True,started_unix=time.time(),memory_floors_MiB=floors)
        write_new(EXECUTION/'SUPERVISOR_PID.json',receipt)
    print(json.dumps(receipt))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
