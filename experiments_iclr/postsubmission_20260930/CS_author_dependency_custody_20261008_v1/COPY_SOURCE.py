from pathlib import Path
import sys,json,base64,hashlib,socket,subprocess,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
rows=json.loads(sys.stdin.read());out=[]
for r in rows:
 f=P/r['path'];assert f.resolve().is_relative_to(P) and r['path'].startswith('continuous_method_gap_search_v1/round15_graph_route_initialization/primary/author_source/')
 data=base64.b64decode(r['data']);assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
 existed=f.exists()
 if existed:assert f.read_bytes()==data
 else:f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(data)
 out.append({k:r[k] for k in ('path','sha256','bytes')}|{'existed_identically':existed})
D=P/'CS_author_dependency_custody_20261008_v1';D.mkdir(exist_ok=True);x={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'route_verified':True,'source_only':True,'code_executed':False,'raw_author_source_kept_server_only':True,'files':out}
with (D/'ALLOCATION_COPY_RECEIPT.json').open('x') as h:json.dump(x,h,indent=2);h.write('\n')
print(json.dumps(x))
