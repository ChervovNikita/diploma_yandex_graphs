from pathlib import Path
import json,socket,subprocess,hashlib,base64,zlib,datetime
assert socket.gethostname()=='anogena-2-0'
U='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[U]
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
A=P/'molhiv_closed_roster_selected_readout_activation_root_20261009_v1'
Q=P/'molhiv_closed17_failed1_selected_readout_execution_root_20261009_v1'
r=json.loads((A/'EXECUTION_RECEIPT.json').read_text());assert r['collector_exit_code']==1 and r['reaped'] and not r['timed_out']
h=[r['child'],json.loads((A/'ATTEMPT.json').read_text())['owner']]
for v in h:
 p=Path('/proc',str(v['pid']),'stat')
 if p.exists():
  f=p.read_text().rsplit(') ',1)[1].split();assert int(f[19])!=v['start_ticks']
cuda=[]
for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True).splitlines():
 if row.split(',')[0].strip().isdigit() and int(row.split(',')[0]) in [v['pid'] for v in h]:cuda.append(row)
assert not cuda
files=[]
for p in sorted((Q/'compact').iterdir()):
 if not p.is_file():continue
 assert p.suffix=='.json' and p.stat().st_size<2000000
 b=p.read_bytes();files.append(dict(path=str(p.relative_to(P)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(zlib.compress(b,9)).decode()))
for name in ['ATTEMPT.json','CHILD_STARTED.json','EXECUTION_RECEIPT.json','FIXTURE_RECEIPT.json','FIXTURE.log','READOUT.log']:
 p=A/name;assert p.is_file() and p.stat().st_size<2000000
 b=p.read_bytes();files.append(dict(path=str(p.relative_to(P)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(zlib.compress(b,9)).decode()))
for name in ['PUSH_RECEIPT.json','PUSH_TRANSPORT_RECOVERY.json']:
 p=P/'publication/complete_IMDB_decision_and_molecular_readout_release_20261009_v1'/name
 if p.exists():
  b=p.read_bytes();files.append(dict(path=str(p.relative_to(P)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(zlib.compress(b,9)).decode()))
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),readout_original_handles_absent=True,owned_CUDA=cuda,files=files,raw_arrays_fetched=False)))