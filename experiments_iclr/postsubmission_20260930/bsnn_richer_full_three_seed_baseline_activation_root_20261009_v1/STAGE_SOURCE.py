from pathlib import Path
import json,sys,base64,hashlib,subprocess,socket,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,stdin=subprocess.DEVNULL).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True,stdin=subprocess.DEVNULL).strip()=='2e913233f2ffac3e1b43d12f5559b47a95ab0471'
for pid in [564820,564827]:assert not Path('/proc',str(pid)).exists(), 'Prior BSNN PID present; revalidate identity before proceeding'
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,stdin=subprocess.DEVNULL).splitlines()
assert not {'564820','564827'} & {v.strip() for v in apps}
rows=json.loads(sys.stdin.read());prepared=[]
for row in rows:
 path=P/row['path'];assert path.resolve().is_relative_to(P.resolve()) and path.parent.name=='bsnn_richer_full_three_seed_baseline_activation_root_20261009_v1' and not path.exists()
 data=base64.b64decode(row['data']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'];prepared.append((path,data))
for path,data in prepared:
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(data)
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),staged=len(prepared),predecessor_groups_and_cuda_absent=True,source_head='2e913233f2ffac3e1b43d12f5559b47a95ab0471',numerical_work=False)))
