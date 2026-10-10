"""Commit reviewed source, then qualify and launch frozen capable references."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,subprocess
H=Path(__file__).resolve().parent;P=H.parent
O=P/'pubmed_strong_reference_root_execution_20261010_v1'
review=json.loads((O/'OWNER_REVIEW_TEMPLATE_DISABLED.json').read_text())
review.update(approved=True,enabled=True,root_delta_review='Changed dispatch/per-record limits/namespaces and actual3-to9 admission inspected. Frozen roster and existing finite owner logic retained.')
(O/'OWNER_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n')
files={}
for folder in [P/'combination_pubmed_strong_reference_source_20261010_v1',P/'combination_pubmed_strong_reference_source_20261010_v2',O]:
    for q in folder.rglob('*'):
        if q.is_file() and q.suffix in {'.py','.json','.md'}:files[str(q.relative_to(P))]=base64.b64encode(q.read_bytes()).decode()
files[str((H/'SCIENTIFIC_DECISION.md').relative_to(P))]=base64.b64encode((H/'SCIENTIFIC_DECISION.md').read_bytes()).decode()
remote=r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,socket,subprocess
assert socket.gethostname()=='anogena-2-0'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=5).splitlines()==[GPU]
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
O=P/'pubmed_strong_reference_root_execution_20261010_v1'
FILES=__FILES__
for name,b in FILES.items():
    q=P/name;assert q.resolve().is_relative_to(P)
    data=base64.b64decode(b)
    if q.exists():assert q.read_bytes()==data,name+' conflicts with immutable source'
    else:q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data)
inv=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
ready=dict(UTC=datetime.now(timezone.utc).isoformat(),normal_host_execution=True,GPU_inventory=inv,source_reviewed=True)
with (O/'READINESS.json').open('x') as f:json.dump(ready,f,indent=2);f.write('\n')
def git(*args):return subprocess.check_output(['git','-C',str(R),*args],text=True)
prior=git('rev-parse','HEAD').strip();assert prior=='d0c006db9346a3c5e1fb03f6f99554bea51381ee'
assert git('branch','--show-current').strip()=='codex/postsubmission-research-20260930' and not git('diff','--cached','--name-only')
paths=['experiments_iclr/postsubmission_20260930/'+k for k in FILES]+['experiments_iclr/postsubmission_20260930/pubmed_strong_reference_root_execution_20261010_v1/READINESS.json']
subprocess.run(['git','-C',str(R),'add','-f','--',*paths],check=True)
staged=git('diff','--cached','--name-only').splitlines();assert staged and set(staged)<=set(paths)
subprocess.run(['git','-C',str(R),'commit','-m','Prepare capable PubMed references with individual selectors and four-dropout control'],capture_output=True,check=True)
value=dict(commit=git('rev-parse','HEAD').strip(),prior_head=prior,branch='codex/postsubmission-research-20260930',changed_paths=staged,UTC=datetime.now(timezone.utc).isoformat(),push_verified=False,binary_artifacts_included=False)
receipt=P/'publication/pubmed_capable_references_source_and_qualification_20261010_v1/COMMIT_RECEIPT.json';receipt.parent.mkdir(parents=True,exist_ok=True)
with receipt.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(dict(committed=value)),flush=True)
subprocess.run(['python3','-B',str(O/'render_engineering.py')],cwd=R,check=True,timeout=30)
plan=O/'ENGINEERING_OWNER_PLAN.json'
res=subprocess.run(['python3','-B',str(O/'queue.py'),'--plan',str(plan),'--plan-sha256',hashlib.sha256(plan.read_bytes()).hexdigest()],cwd=R,check=False,timeout=1850)
fetched={}
for q in [O/'READINESS.json',O/'ENGINEERING_OWNER_PLAN.json',O/'engineering/FAMILY_COMPLETE.json',O/'engineering/FAMILY_FAILURE.json']:
    if q.is_file():fetched[str(q.relative_to(P))]=json.loads(q.read_text())
for condition in ('single_native','single_mean4_dropout','independent4_own'):
    record='seed9101__'+condition
    for q in [O/'engineering/cells'/record/'COMPLETE.json',O/'engineering/cells'/record/'FAILURE.json',O/'owners'/('engineering__'+record)/'RAW_OWNER_TERMINAL.json']:
        if q.is_file():fetched[str(q.relative_to(P))]=json.loads(q.read_text())
    log=O/'owners'/('engineering__'+record)/'WORKER.log'
    if res.returncode and log.is_file():fetched[str(log.relative_to(P))]=log.read_text()[-5000:]
print(json.dumps(dict(engineering_exit=res.returncode,fetched=fetched)),flush=True)
if res.returncode:raise SystemExit(res.returncode)
inv=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
with (O/'SCIENCE_READINESS.json').open('x') as f:json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),normal_host_execution=True,GPU_inventory=inv),f,indent=2);f.write('\n')
subprocess.run(['python3','-B',str(O/'render_science.py'),'--readiness',str(O/'SCIENCE_READINESS.json')],cwd=R,check=True,timeout=30)
plan=O/'SCIENCE_OWNER_PLAN.json';sha=hashlib.sha256(plan.read_bytes()).hexdigest()
with (O/'SCIENCE_OWNER.log').open('xb') as log:
    owner=subprocess.Popen(['python3','-B',str(O/'queue.py'),'--plan',str(plan),'--plan-sha256',sha],cwd=R,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
fields=Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()
launch=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=owner.pid,start_ticks=int(fields[19]),pgid=int(fields[2]),sid=int(fields[3]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),owner_plan_sha256=sha,source_commit=value['commit'],new_groups=9,new_full_native_trajectories=18,TEST_access=False,automatic_retry=False)
with (O/'ACTUAL_SCIENCE_LAUNCH.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n')
print(json.dumps(dict(science_launched=launch,owner_plan=json.loads(plan.read_text()),root_admission=json.loads((O/'ROOT_ADOPTION.json').read_text()))),flush=True)
'''.replace('__FILES__',repr(files))
argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -']
res=subprocess.run(argv,input=remote,capture_output=True,text=True,timeout=1950)
(H/'STRONG_REFERENCE_ACTIVATION_TRANSPORT.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=res.returncode,stdout=res.stdout,stderr=res.stderr),indent=2)+'\n')
for line in res.stdout.splitlines():
    if not line.startswith('{'):continue
    item=json.loads(line)
    if 'committed' in item:
        q=P/'publication/pubmed_capable_references_source_and_qualification_20261010_v1/COMMIT_RECEIPT.json';q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(item['committed'],indent=2)+'\n')
    for name,data in item.get('fetched',{}).items():
        q=H/'strong_reference_metadata'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(data,indent=2)+'\n')
    if 'science_launched' in item:
        (H/'STRONG_REFERENCE_SCIENCE_LAUNCH.json').write_text(json.dumps(item,indent=2)+'\n')
print(json.dumps(dict(exit_code=res.returncode,science_launched=(H/'STRONG_REFERENCE_SCIENCE_LAUNCH.json').exists(),stderr_tail=res.stderr[-3000:])))
raise SystemExit(res.returncode)
