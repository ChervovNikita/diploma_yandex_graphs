"""Launch once from exact committed root bindings on the authorized sole GPU."""
import base64
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

H = Path(__file__).resolve().parent
P = H.parent
R = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
REMOTE = r'''
import base64,zlib,json,sys,socket,subprocess,pathlib,hashlib,os,datetime
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==[sys.argv[2]]
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
repo=pathlib.Path(sys.argv[1]);phase=repo/'experiments_iclr/postsubmission_20260930';H=phase/'native_SAGE_Physics_confirmation_root_20261011_v1'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==payload['commit']
spec_path=H/'ROOT_INTEGRATION_V1.json'
assert hashlib.sha256(spec_path.read_bytes()).hexdigest()==payload['spec_sha256']
bindings=json.loads((H/'FROZEN_BINDINGS_V1.json').read_text())
owner_source=H/'complete_cohort_outer_owner_v1.py'
bound=[v for v in bindings['bound_files'] if v['path']==str(owner_source)];assert len(bound)==1
assert hashlib.sha256(owner_source.read_bytes()).hexdigest()==bound[0]['sha256']
assert not (H/'external_cohort_owner_v1').exists() and not (H/'SCIENTIFIC_LAUNCH_V1.json').exists()
os.chdir(repo)
with (H/'scientific_cohort_outer.log').open('x') as log:
    owner=subprocess.Popen([str(repo/'.venv/bin/python'),'-B','-u',str(owner_source)],cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
record=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='FIXED_COHORT_LAUNCHED',owner_PID=owner.pid,commit=payload['commit'],spec_sha256=payload['spec_sha256'],banks=60,optimizer_bodies=90,paired_masks=10,CAL_CONFIRM_truth_scoring=False,external_wall_seconds=172920,no_retry=True)
(H/'SCIENTIFIC_LAUNCH_V1.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''


def main():
    commit = json.loads((P / 'publication/Physics_complete_cohort_admission_20261011_v1/COMMIT_RECEIPT.json').read_text())['commit']
    receipt = json.loads((H / 'COMPLETE_COHORT_BIND_RECEIPT_V1.json').read_text())['spec_receipt']
    m_spec = importlib.util.spec_from_file_location('raw_terminal_transport', P / 'publication/publish_exact_inventory_v19.py')
    m = importlib.util.module_from_spec(m_spec); m_spec.loader.exec_module(m)
    args = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no', 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
            shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', REMOTE, R, UUID, 'fixed-Physics-scientific-launch'])]
    payload = dict(commit=commit, spec_sha256=receipt['sha256'])
    r = m.terminal_transport(args, input=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode(), capture_output=True, text=True, timeout=45)
    (H / 'SCIENTIFIC_LAUNCH_TRANSPORT_V1.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(), exit_code=r.returncode, stdout=r.stdout, stderr=r.stderr), indent=2) + '\n')
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:]
    value = json.loads(r.stdout)
    (H / 'SCIENTIFIC_LAUNCH_V1.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value))


if __name__ == '__main__': main()
