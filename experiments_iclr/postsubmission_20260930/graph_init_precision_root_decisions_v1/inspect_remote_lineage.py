"""Read only the authorized predecessor text inventory; never read arrays or models."""
import hashlib,json,shlex,subprocess
from pathlib import Path
from datetime import datetime,timezone
from read_authorized_resource import SSH,REPO,UUID
PHASE=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
lineage=json.loads((PHASE/'graph_init_precision_execution_root_v1/LINEAGE_BINDINGS.json').read_text())
CODE="""import hashlib,json,pathlib,subprocess,sys
repo=pathlib.Path(sys.argv[1]);base=repo/'experiments_iclr/postsubmission_20260930'
assert pathlib.Path.cwd().resolve()==repo
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==[sys.argv[2]]
v=json.load(sys.stdin)
ext={'.json','.py','.md','.csv','.txt','.sh','.toml','.yaml','.yml','.log','.jsonl','.patch','.diff','.html'}
rows=[]
for root in v['preserved_roots']:
 for p in sorted((base/root).rglob('*')):
  if p.is_file() and p.suffix in ext and '__pycache__' not in p.parts:
   assert not p.is_symlink() and p.resolve().is_relative_to(base)
   data=p.read_bytes();rows.append(dict(path=str(p),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
expected={r['path']:r for r in v['all_bound_text_files']};actual={r['path']:r for r in rows}
print(json.dumps(dict(expected_count=len(expected),actual_count=len(actual),missing=sorted(set(expected)-set(actual)),extra=[actual[k] for k in sorted(set(actual)-set(expected))],changed=[dict(expected=expected[k],actual=actual[k]) for k in sorted(set(actual)&set(expected)) if actual[k]['sha256']!=expected[k]['sha256']])))
"""
command='cd '+shlex.quote(REPO)+' && .venv/bin/python -c '+shlex.quote(CODE)+' '+shlex.join([REPO,UUID])
r=subprocess.run(SSH+[command],input=json.dumps(lineage),capture_output=True,text=True,timeout=60)
v=dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stderr=r.stderr,inventory=json.loads(r.stdout) if r.returncode==0 else None)
p=HERE/'REMOTE_LINEAGE_INVENTORY_v1.json'
with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
print(json.dumps(v))
raise SystemExit(r.returncode)
