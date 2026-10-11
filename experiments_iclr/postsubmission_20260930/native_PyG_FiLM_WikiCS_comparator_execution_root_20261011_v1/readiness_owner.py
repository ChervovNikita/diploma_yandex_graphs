"""Own one disposable FiLM update with finite wait and direct-child cleanup."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [UUID]
    h = Path(__file__).resolve().parent
    assert h.is_relative_to(R) and Path.cwd() == R
    s = h.parent/'native_PyG_FiLM_WikiCS_comparator_source_20261011_v1'
    side = h/'readiness_owner_v1'
    side.mkdir(exist_ok=False)
    output = h/'readiness_v1'
    assert not output.exists()
    cmd = [str(R/'.venv/bin/python'), '-B', str(s/'readiness.py'), '--config', str(h/'READINESS_CONFIG_V1.json'), '--output', str(output)]
    start = time.monotonic()
    with (side/'stdout.log').open('x') as out, (side/'stderr.log').open('x') as err:
        child = subprocess.Popen(cmd, cwd=R, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
        write(side/'OWNER_START.json', dict(UTC=datetime.now(timezone.utc).isoformat(), owner_pid=os.getpid(), child_pid=child.pid, wall_seconds=300, scientific_fit_or_selection=False))
        timed_out = False
        try:
            code = child.wait(timeout=300)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait(timeout=15)
    result_path = output/'READINESS_RESULT.json'
    value = json.loads(result_path.read_text()) if result_path.exists() else {}
    absent = not Path('/proc', str(child.pid)).exists()
    cuda_absent = str(child.pid) not in subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()
    successful = code == 0 and value.get('complete') is True and absent and cuda_absent and not timed_out
    result = dict(UTC=datetime.now(timezone.utc).isoformat(), exit_code=code, timed_out=timed_out,
        direct_child_wait=True, child_pid=child.pid, child_absent=absent, owned_cuda_pid_absent=cuda_absent,
        seconds=time.monotonic()-start, successful_readiness=successful, scientific_fit_or_selection=False,
        TEST_access=False, science_launch_admitted=False,
        overlap='Existing Physics and Penn94 on the same sole GPU; no isolated speed claim')
    write(side/'OWNER_END.json', result)
    return 0 if successful else 1


if __name__ == '__main__':
    raise SystemExit(main())
