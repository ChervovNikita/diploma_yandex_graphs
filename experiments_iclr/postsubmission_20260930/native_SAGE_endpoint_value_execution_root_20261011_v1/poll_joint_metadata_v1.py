"""Observe the existing two studies without opening predictive outputs."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shlex

H = Path(__file__).resolve().parent
P = H.parent
REMOTE = r'''
import base64,json,socket,subprocess,sys,zlib
from datetime import datetime,timezone
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
E=P/'native_SAGE_endpoint_value_execution_root_20261011_v1'
H=P/'native_SAGE_Penn94_data_preparation_root_20261011_v1'
def j(p):
    assert p.resolve().is_relative_to(P)
    return json.loads(p.read_text()) if p.exists() else None
def process(pid):
    q=Path('/proc')/str(pid)/'stat'
    if not q.exists():return dict(PID=pid,present=False)
    b=q.read_text().split()
    return dict(PID=pid,present=True,state=b[2],parent_PID=int(b[3]),start_ticks=int(b[21]))
r=dict(UTC=datetime.now(timezone.utc).isoformat(),partial_predictive_metrics_read=False,
    processes={n:process(p) for n,p in [('VALUE_fit_child',626848),('VALUE_reader_watcher',626868),('Penn94_fit_child',626532),('Penn94_pipeline_watcher',627089)]},
    VALUE=dict(owner_end=j(E/'fit_owner_v1/OWNER_END.json'),progress=j(E/'actual_family_v1/PROGRESS.json'),reader_end=j(E/'READER_WATCH_END_V1.json')),
    Penn94=dict(owner_end=j(H/'native_acquisition_v1_owner/OWNER_END.json'),banks=sorted(q.name for q in (H/'native_acquisition_v1').iterdir() if q.is_dir()),pipeline_end=j(H/'PIPELINE_WATCH_END_V1.json')),
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).splitlines())
print(json.dumps(r))
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--version',type=int,required=True)
    a=ap.parse_args()
    output=H/f'JOINT_SCIENTIFIC_OBSERVATION_V{a.version}.json'
    transport=H/f'JOINT_OBSERVATION_TRANSPORT_V{a.version}.json'
    assert not output.exists() and not transport.exists()
    spec=importlib.util.spec_from_file_location('transport',P/'publication/publish_exact_inventory_v19.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,'metadata','literal-route','existing-processes'])]
    import base64,zlib
    r=mod.terminal_transport(argv,input=base64.b64encode(zlib.compress(b'{}')).decode(),capture_output=True,text=True,timeout=60)
    transport.write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
    assert r.returncode==0,r.stderr[-1500:]
    obj=json.loads(r.stdout);output.write_text(json.dumps(obj,indent=2)+'\n');print(json.dumps(obj))

if __name__=='__main__':main()
