"""Stage fixed successor source on both hosts; project metadata after allocation bytes pass."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys,json,hashlib,subprocess
HERE=Path(__file__).resolve().parent;E=HERE.parents[1];sys.path.insert(0,str(HERE));import stage_batch
packages=['server51_campaign_port_v4','qualification/all_fit_smoke_20261016_v4','campaign_execution_20261016_v5','server51_campaign_projection_cache_v4'];paths=[]
for name in packages:
 p=E/name;m=json.loads((p/'MANIFEST.json').read_text());paths.append(p/'MANIFEST.json')
 for row in m['files']:
  f=p/row['path'];raw=f.read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];paths.append(f)
def stage(host):
 r=stage_batch.stage(host,paths);(HERE/('PORT4_RUNTIME_STAGE_'+host+'.json')).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'host':host,'files':len(r['files'])}),flush=True);return host
with ThreadPoolExecutor(max_workers=2) as pool:
 fs=[pool.submit(stage,h) for h in ('allocation','server51')];project=None
 for f in as_completed(fs):
  if f.result()=='allocation':project=subprocess.Popen([sys.executable,'-B',str(HERE/'prepare_port4.py')])
 assert project is not None
 code=project.wait();raise SystemExit(code)
