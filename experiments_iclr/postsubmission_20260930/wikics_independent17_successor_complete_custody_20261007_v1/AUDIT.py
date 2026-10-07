import json,hashlib,socket,subprocess,base64
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
D=P/'wikics_independent17_successor_execution_root_20261007_v2';owner=json.loads((D/'COMPLETE.json').read_text());assert owner['complete'] and len(owner['results'])==2
rows=[]
for r in owner['results']:
 t=r['terminal'];assert r['complete'] and t['exit_code']==0 and t['reason'] is None and not t['signals_sent'] and t['terminal_wait_observed'];p=P/r['freeze']['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['freeze']['sha256'];f=json.loads(p.read_text());assert f['complete'] and f['seed']==17 and f['TEST_access'] is False
 rows.append({'cell_id':r['cell_id'],'freeze':r['freeze'],'clean_owned_terminal':True,'elapsed_seconds':t['elapsed_seconds'],'peak_GPU_bytes':t['max_sampled_owned_GPU_bytes'],'peak_RSS_bytes':t['max_sampled_owned_RSS_bytes'],'source_manifest_sha256':f['source_manifest_sha256'],'no_quality_export':True})
p=D/'COMPLETE.json';blob=p.read_bytes();print(json.dumps({'rows':rows,'complete':True,'quality_values_read':False,'owner_complete_bytes':base64.b64encode(blob).decode(),'owner_complete_sha256':hashlib.sha256(blob).hexdigest()}))
