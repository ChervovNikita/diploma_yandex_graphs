"""Bind actual complete18 custody to its previously prepared comparison."""
from pathlib import Path
import json,shlex,subprocess
HERE=Path(__file__).resolve().parent
REMOTE=r'''
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,shutil,socket,subprocess
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
H=P/'private_hop_credit_pubmed_fullfit_source_20261010_v1';A=P/'closed_pubmed_readouts_root_20261010_v1'
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
def save(q,v):
 q.parent.mkdir(parents=True,exist_ok=True)
 with q.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def bind(q):return dict(path=str(q),sha256=sha(q))
manifest='d565db9ecde277a1753a4ec040067974a9a5533dfb4a802775ab5f90bdbaad8d'
assert sha(H/'SOURCE_MANIFEST.json')==manifest
for row in json.loads((H/'SOURCE_MANIFEST.json').read_text())['files']:
 q=(H/row['path']).resolve(strict=True);assert q.is_relative_to(H) and sha(q)==row['sha256']
start=json.loads((P/'private_hop_full18_root_activation_20261010_v1/STARTER.json').read_text());assert not Path('/proc',str(start['PID'])).exists()
family=H/'science/FAMILY_COMPLETE.json';closed=json.loads(family.read_text());assert closed['complete'] and len(closed['records'])==18 and not (H/'science/FAMILY_FAILURE.json').exists()
mem={r.split(':')[0]:int(r.split(':')[1].strip().split()[0])*1024 for r in Path('/proc/meminfo').read_text().splitlines() if ':' in r}
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip())*1024**2
assert free>=32*1024**3 and mem['MemAvailable']>=16*1024**3 and shutil.disk_usage(P).free>=1024**3
readiness=H/'COMPARISON_READINESS.json';save(readiness,dict(UTC=datetime.now(timezone.utc).isoformat(),GPU_free_bytes=free,MemAvailable_bytes=mem['MemAvailable'],normal_host_execution=True,other_jobs_changed=False))
review=H/'COMPARISON_OWNER_REVIEW.json';v=json.loads((H/'COMPARISON_OWNER_REVIEW_TEMPLATE_DISABLED.json').read_text());v.update(approved=True,UTC=datetime.now(timezone.utc).isoformat(),reviewer='root author readout review',specialized_owner_main_not_invoked=True);save(review,v)
contract=H/'COMPARISON_OWNER_RELEASE.json';v=json.loads((H/'COMPARISON_OWNER_RELEASE_TEMPLATE_DISABLED.json').read_text());v.update(enabled=True,owner_review=bind(review));save(contract,v)
release=H/'COMPARISON_RELEASE.json';v=json.loads((H/'COMPARISON_RELEASE_TEMPLATE_DISABLED.json').read_text())
v.update(enabled=True,root_authorized=True,source_review_approved=True,all_eighteen_closed=True,finite_owner_verified=True,ordinary_runtime_confirmed=True,fresh_resource_readiness_confirmed=True,VALID_access=True,source_manifest_sha256=manifest,root_review=bind(H/'ROOT_REVIEW.json'),resource_readiness_evidence=bind(readiness),external_owner_release=bind(contract),family_complete=bind(family))
assert [r['record_id'] for r in v['records']]==closed['records']
for row in v['records']:
 for key in ('complete','terminal','raw_terminal'):
  q=Path(row[key]['path']);assert q.is_relative_to(H);row[key]=bind(q)
assert not Path(v['output']).exists();save(release,v)
plan=json.loads((H/'COMPARISON_OWNER_ROW_TEMPLATE_DISABLED.json').read_text());plan.update(enabled=True,source_manifest_sha256=manifest)
plan['records'][0]['release_sha256']=sha(release);pp=H/'COMPARISON_OWNER_PLAN.json';save(pp,plan)
with (A/'full18_owner.stdout').open('xb') as out,(A/'full18_owner.stderr').open('xb') as err:
 process=subprocess.Popen([v['runtime']['python']['path'],'-B','-P',str(A/'owner_full18.py'),sha(pp)],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 f=Path('/proc',str(process.pid),'stat').read_text().rsplit(')',1)[1].split()
 starter=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=process.pid,start_ticks=int(f[19]),pgid=int(f[2]),sid=int(f[3]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_commit=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),source_manifest_sha256=manifest,scientific_fits=0,new_model_forwards=0,VALID_access=True,TEST_access=False,automatic_retry=False,owner_wrapper=bind(A/'owner_full18.py'))
 assert starter['pgid']==starter['sid']==process.pid;save(A/'FULL18_STARTER.json',starter)
kept=[readiness,review,contract,release,pp,A/'FULL18_STARTER.json']
print(json.dumps(dict(started=starter,retained={str(q.relative_to(P)):json.loads(q.read_text()) for q in kept})))
'''
assert not (HERE/'FULL18_TRANSPORT.json').exists()
argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(REMOTE)]
r=subprocess.run(argv,capture_output=True,text=True,timeout=60)
(HERE/'FULL18_TRANSPORT.json').write_text(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
assert r.returncode==0,r.stdout[-2000:]+r.stderr
v=json.loads(r.stdout)
for name,data in v['retained'].items():
 q=HERE/'fetched'/name;assert q.resolve().is_relative_to((HERE/'fetched').resolve());q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(v['started']))
