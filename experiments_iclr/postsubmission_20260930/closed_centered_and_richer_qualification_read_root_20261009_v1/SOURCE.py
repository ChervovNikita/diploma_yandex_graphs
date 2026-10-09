from pathlib import Path
import json,subprocess,hashlib,base64,socket,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,stdin=subprocess.DEVNULL).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
pids={564508,564511,564820,564827}
assert all(not Path('/proc',str(pid)).exists() for pid in pids)
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:s=(p/'stat').read_text()
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 f=s[s.rfind(')')+2:].split();assert int(f[2]) not in pids,'Owned process group still present'
cu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,stdin=subprocess.DEVNULL).splitlines()
assert not pids & {int(v.strip()) for v in cu if v.strip().isdigit()}
paths=['centered_prior_minimal_full_input_qualification_execution_root_20261009_v2/QUALIFICATION.json','centered_prior_minimal_full_input_qualification_activation_root_20261009_v2/TERMINAL.json','bsnn_richer_full_input_qualification_execution_root_20261009_v1/RESULT.json','bsnn_richer_full_input_qualification_execution_root_20261009_v1/COMPLETE.json','bsnn_richer_full_input_qualification_activation_root_20261009_v1/TERMINAL.json']
rows=[]
for rel in paths:
 p=P/rel;b=p.read_bytes();assert len(b)<1000000;d=json.loads(b)
 assert d.get('qualification_passed',d.get('complete',True)) is True
 rows.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_owned_parent_worker_group_CUDA_absent=True,engineering_only=True,scientific_comparative_scores_opened=False,files=rows)),flush=True)
