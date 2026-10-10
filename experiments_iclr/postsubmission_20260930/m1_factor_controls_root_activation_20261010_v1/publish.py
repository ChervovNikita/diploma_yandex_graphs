"""Commit explicit text evidence and current decisions on the authorized route."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,subprocess

H=Path(__file__).resolve().parent;P=H.parent
packet='publication/actual_M1_six_launch_and_IMDB_confidence_complete_20261010_v1'
selected={P/n for n in ['RESEARCH_STATE.md','PUBLIC_STATUS.md','research_ledger.json','RESEARCH_WORKFLOW.md']}
directories=[H,P/'imdb_confidence_bins_root_execution_20261010_v1',P/'neighbor_dispersion_competence_gap_scout_20261010_v1']
for directory in directories:
 for q in directory.rglob('*'):
  if q.is_file() and not q.is_symlink() and q.suffix in {'.md','.json','.py','.txt'} and q.name not in {'TRANSPORT.json','GPU77_WRAPPER_OUTPUT.json'}:selected.add(q)
selected.add(P/'combination_shared_backbone_scope_assessment_20261010_v1.json')
payload={}
for q in sorted(selected):
 data=q.read_bytes();data.decode('utf-8');assert len(data)<4*1024**2 and b'\0' not in data
 payload[str(q.relative_to(P))]=dict(data=base64.b64encode(data).decode(),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
old=json.loads((H/'PREVIOUS_CANONICAL_HASHES.json').read_text())
inventory={n:{k:r[k] for k in ['sha256','bytes']} for n,r in payload.items()}
(H/'PUBLICATION_INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
remote=r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,socket,subprocess
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
PAYLOAD=__PAYLOAD__;OLD=__OLD__;packet=__PACKET__
def git(*args):return subprocess.check_output(['git','-C',str(R),*args],text=True)
assert git('branch','--show-current').strip()=='codex/postsubmission-research-20260930'
assert git('rev-parse','HEAD').strip()=='1fe4cf807474a4b5d18eb50809e6f2507eabe6e6'
assert not git('diff','--cached','--name-only')
receipt=P/packet/'COMMIT_RECEIPT.json';assert not receipt.exists()
for name,row in PAYLOAD.items():
 q=P/name;assert q.resolve().is_relative_to(P)
 if q.exists():
  got=hashlib.sha256(q.read_bytes()).hexdigest()
  assert got==row['sha256'] or (name in OLD and got==OLD[name]),name+' differs from prior observed state'
for name,row in PAYLOAD.items():
 q=P/name;q.parent.mkdir(parents=True,exist_ok=True)
 data=base64.b64decode(row['data']);assert hashlib.sha256(data).hexdigest()==row['sha256']
 if not q.exists() or q.read_bytes()!=data:q.write_bytes(data)
paths=['experiments_iclr/postsubmission_20260930/'+name for name in sorted(PAYLOAD)]
subprocess.run(['git','-C',str(R),'add','-f','--',*paths],check=True)
staged=git('diff','--cached','--name-only').splitlines();assert staged and set(staged)<=set(paths)
for name,row in PAYLOAD.items():
 blob=subprocess.check_output(['git','-C',str(R),'show',':experiments_iclr/postsubmission_20260930/'+name])
 assert hashlib.sha256(blob).hexdigest()==row['sha256']
prior=git('rev-parse','HEAD').strip()
subprocess.run(['git','-C',str(R),'commit','-m','Record actual M1 factor-control launch and complete IMDB confidence analysis'],capture_output=True,check=True)
commit=git('rev-parse','HEAD').strip();assert not git('diff','--cached','--name-only')
value=dict(commit=commit,prior_head=prior,branch='codex/postsubmission-research-20260930',changed_paths=staged,
 UTC=datetime.now(timezone.utc).isoformat(),binary_artifacts_included=False,credentials_included=False,
 original_paper_scores_unchanged=True,push_verified=False)
receipt.parent.mkdir(parents=True,exist_ok=True)
with receipt.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value),flush=True)
'''.replace('__PAYLOAD__',repr(payload)).replace('__OLD__',repr(old)).replace('__PACKET__',repr(packet))
argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes',
 '-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no',
 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -']
# Source is streamed without a PTY to avoid echoing the payload.
out=P/packet;out.mkdir(exist_ok=False)
res=subprocess.run(argv,input=remote,capture_output=True,text=True,timeout=60)
(out/'COMMIT_TRANSPORT.json').write_text(json.dumps(dict(exit_code=res.returncode,stdout=res.stdout,stderr=res.stderr),indent=2)+'\n')
if res.returncode:print(res.stderr);raise SystemExit(res.returncode)
value=json.loads(res.stdout);(out/'COMMIT_RECEIPT.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(commit=value['commit'],changed_files=len(value['changed_paths']))))
