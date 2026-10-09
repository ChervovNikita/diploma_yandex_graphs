
import socket,subprocess,json,hashlib,os,datetime,sys,runpy
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';os.chdir(R)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
a=P/'pre_sigmoid_qk_distributed_F_activation_root_20261009_v1/allocation_a100';o=P/'pre_sigmoid_qk_distributed_F_execution_root_20261009_v1/allocation_a100'
close=json.loads((o/'CLOSURE.json').read_text());assert close['complete'] and len(close['cells'])==12
if not (o/'ABSENCE_RECEIPT.json').exists():
 source=P/'pre_sigmoid_qk_route_lane_owner_source_20261009_v2/attest.py';sys.path.insert(0,str(source.parent));sys.argv=[str(source),'--release',str(a/'RELEASE.json'),'--release-sha256',sha(a/'RELEASE.json'),'--authorized'];runpy.run_path(str(source),run_name='__main__')
paths=[p for p in o.rglob('*') if p.is_file()]+[a/'RELEASE.json']
plan=json.loads((P/'pre_sigmoid_qk_distributed_F_activation_root_20261009_v1/GLOBAL_PLAN.json').read_text());adopt=plan['qualification_adoptions']['allocation_a100'];paths +=[P/adopt['report']['path']]+[P/b['path'] for b in adopt['legacy_allocation_evidence'].values()]
rows=[]
for p in sorted(set(paths)):
 assert not p.is_symlink() and p.resolve().is_relative_to(P)
 rel='experiments_iclr/postsubmission_20260930/'+str(p.relative_to(P));rows.append(dict(source_relative=rel,target_relative=rel,bytes=p.stat().st_size,sha256=sha(p)))
assert sum(r['bytes'] for r in rows)<8*1024**3
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),all12_complete=True,files=rows,total_bytes=sum(r['bytes'] for r in rows),absence_receipt=next(r for r in rows if r['source_relative'].endswith('/ABSENCE_RECEIPT.json')),scores_opened=False,payloads_decoded=False)))
