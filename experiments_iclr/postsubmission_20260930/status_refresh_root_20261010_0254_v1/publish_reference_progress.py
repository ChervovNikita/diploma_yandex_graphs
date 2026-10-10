"""Publish explicit current text and completed evidence without binary copies."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,subprocess
H=Path(__file__).resolve().parent;P=H.parent
packet='publication/capable_reference_actual_launch_and_history_update_20261010_v1'
canonical=['RESEARCH_STATE.md','PUBLIC_STATUS.md','research_ledger.json','RESEARCH_WORKFLOW.md']
selected={P/q for q in canonical}
for directory in [H,P/'pubmed_complete9_error_diagnostic_root_20261010_v1',P/'masked_context_pubmed_stage1_complete9_comparison_execution_20261010_v1',P/'pubmed_complete9_stored_diagnostic_source_assessment_20261010_v1',P/'historical_pubmed_complete9_followup_20261010_v1']:
    for q in directory.rglob('*'):
        if q.is_file() and q.suffix in {'.md','.json','.py','.txt'}:selected.add(q)
# Preserve immutable source and previous terminal receipts; no logits/checkpoints/data.
selected.add(P/'masked_context_pubmed_stage1_root_execution_20261010_v1/WHOLE_FAMILY_CLOSURE_OBSERVATION.json')
payload={}
for q in sorted(selected):
    data=q.read_bytes();data.decode('utf-8');assert len(data)<2*1024**2 and b'\0' not in data
    payload[str(q.relative_to(P))]=dict(data=base64.b64encode(data).decode(),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
old=json.loads((H/'PRE_REFERENCE_LAUNCH_CANONICAL_HASHES.json').read_text())
remote=r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,socket,subprocess
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
PAYLOAD=__PAYLOAD__;OLD=__OLD__;packet=__PACKET__
def git(*args):return subprocess.check_output(['git','-C',str(R),*args],text=True)
assert git('branch','--show-current').strip()=='codex/postsubmission-research-20260930'
assert git('rev-parse','HEAD').strip()=='82c1ab8a117d9611b64e214c3f11bbb470ba8acd'
assert not git('diff','--cached','--name-only')
receipt=P/packet/'COMMIT_RECEIPT.json';assert not receipt.exists()
for name,row in PAYLOAD.items():
    q=P/name;assert q.resolve().is_relative_to(P)
    if q.exists():
        got=hashlib.sha256(q.read_bytes()).hexdigest()
        assert got==row['sha256'] or (name in OLD and got==OLD[name]),name+' differs from reviewed prior state'
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
subprocess.run(['git','-C',str(R),'commit','-m','Record actual capable-reference launch and completed-history synthesis'],capture_output=True,check=True)
commit=git('rev-parse','HEAD').strip();assert not git('diff','--cached','--name-only')
value=dict(commit=commit,prior_head=prior,branch='codex/postsubmission-research-20260930',changed_paths=staged,UTC=datetime.now(timezone.utc).isoformat(),binary_artifacts_included=False,credentials_included=False,original_paper_scores_unchanged=True,push_verified=False)
receipt.parent.mkdir(parents=True,exist_ok=True)
with receipt.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value),flush=True)
'''.replace('__PAYLOAD__',repr(payload)).replace('__OLD__',repr(old)).replace('__PACKET__',repr(packet))
argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -']
# A PTY would echo the streamed input, so disable PTY for source transport.
argv.remove('-tt')
res=subprocess.run(argv,input=remote,capture_output=True,text=True,timeout=60)
out=P/packet;out.mkdir(exist_ok=False)
(out/'COMMIT_TRANSPORT.json').write_text(json.dumps(dict(exit_code=res.returncode,stdout=res.stdout,stderr=res.stderr),indent=2)+'\n')
if res.returncode:print(res.stderr);raise SystemExit(res.returncode)
value=json.loads(res.stdout);(out/'COMMIT_RECEIPT.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(commit=value['commit'],changed_files=len(value['changed_paths']))))
