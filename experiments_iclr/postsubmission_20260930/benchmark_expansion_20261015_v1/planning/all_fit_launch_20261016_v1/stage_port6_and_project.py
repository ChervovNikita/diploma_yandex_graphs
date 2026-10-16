"""Stage frozen repair sources on both hosts, then project unchanged scientific tasks."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys,json,hashlib,subprocess
HERE=Path(__file__).resolve().parent;E=HERE.parents[1];sys.path.insert(0,str(HERE));import stage_batch
packages=['qualification/source_rng_host_binding_20261016_v1','server51_campaign_port_v6','qualification/all_fit_smoke_20261016_v6','campaign_execution_20261016_v7','server51_campaign_projection_cache_v6'];paths=[]
for name in packages:
 p=E/name;m=json.loads((p/'MANIFEST.json').read_text());paths.append(p/'MANIFEST.json')
 for row in m['files']:
  f=p/row['path'];raw=f.read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];paths.append(f)
def stage(host):
 r=stage_batch.stage(host,paths);(HERE/('PORT6_RUNTIME_STAGE_'+host+'.json')).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'host':host,'files':len(r['files'])}),flush=True);return host
with ThreadPoolExecutor(max_workers=2) as pool:
 fs=[pool.submit(stage,h) for h in ('allocation','server51')];project=None
 for f in as_completed(fs):
  if f.result()=='allocation':project=subprocess.Popen([sys.executable,'-B',str(HERE/'prepare_port6.py')])
 assert project is not None
 code=project.wait();raise SystemExit(code)
