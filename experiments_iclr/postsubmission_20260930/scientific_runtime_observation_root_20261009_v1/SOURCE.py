"""Read only: confirm exact owned allocation handles without opening scores."""
from pathlib import Path
import datetime
import json
import socket
import subprocess

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert socket.gethostname() == 'anogena-2-0'
assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [GPU]

def process(pid, expected_birth):
    p = Path('/proc', str(pid))
    try:
        stat = (p/'stat').read_text()
    except FileNotFoundError:
        return dict(pid=pid, present=False, expected_birth=expected_birth)
    f = stat[stat.rfind(')')+2:].split()
    birth = int(f[19])
    assert birth == expected_birth, 'PID reused; do not attribute process'
    command = (p/'cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert str(R) in command or str(pid) == '523400'
    return dict(pid=pid, present=True, start_ticks=birth, group=int(f[2]), session=int(f[3]), state=f[0], command=command)

d = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), host=socket.gethostname(),
         boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
         head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),
         scores_opened=False, TEST_truth_accessed=False, other_processes_changed=False)
d['processes'] = [process(543521,6032159607),process(523400,6019318952)]
d['GPU'] = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).splitlines()
q = P/'pre_sigmoid_qk_distributed_F_execution_root_20261009_v1/allocation_a100'
d['qk_directory_present'] = q.exists()
if q.exists():
    d['qk_root_names'] = sorted(x.name for x in q.iterdir())
    d['qk_completed_endpoint_names'] = [str(x.relative_to(q)) for x in sorted(q.glob('**/COMPLETE.json'))]
    d['qk_handles'] = []
    for path in sorted(q.glob('handles/*.json')):
        value=json.loads(path.read_text())
        d['qk_handles'].append({k:value[k] for k in ('parent','child','spec','output') if k in value})
print(json.dumps(d,indent=2))
