"""Launch the fixed, qualified 18-fit private-hop study once."""
from pathlib import Path
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE = r'''
from datetime import datetime, timezone
from pathlib import Path
import hashlib,json,shutil,socket,subprocess
assert socket.gethostname()=='anogena-2-0'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==[GPU]
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
H=P/'private_hop_credit_pubmed_fullfit_source_20261010_v1';A=P/'private_hop_full18_root_activation_20261010_v1'
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
def save(q,v):
 q.parent.mkdir(parents=True,exist_ok=True)
 with q.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def binding(q):return dict(path=str(q),sha256=sha(q))
manifest='d565db9ecde277a1753a4ec040067974a9a5533dfb4a802775ab5f90bdbaad8d'
assert sha(H/'SOURCE_MANIFEST.json')==manifest and not (A/'STARTER.json').exists()
for row in json.loads((H/'SOURCE_MANIFEST.json').read_text())['files']:
 q=(H/row['path']).resolve(strict=True);assert q.is_relative_to(H) and sha(q)==row['sha256']
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip())*1024**2
mem={r.split(':')[0]:int(r.split(':')[1].strip().split()[0])*1024 for r in Path('/proc/meminfo').read_text().splitlines() if ':' in r}
assert free>=32*1024**3 and mem['MemAvailable']>=16*1024**3 and shutil.disk_usage(P).free>=10*1024**3
save(H/'READINESS.json',dict(UTC=datetime.now(timezone.utc).isoformat(),GPU_free_bytes=free,MemAvailable_bytes=mem['MemAvailable'],normal_host_execution=True,overlapping_acquisition=True,other_jobs_changed=False))
review=json.loads((H/'ROOT_REVIEW.json').read_text());assert review['approved'] and review['source_manifest_sha256']==manifest
owner=json.loads((H/'OWNER_REVIEW.json').read_text());assert owner['approved'] and owner['owner_sha256']==sha(H/'queue.py')
plan=json.loads((H/'OWNER_PLAN_TEMPLATE_DISABLED.json').read_text())
plan.update(enabled=True,source_manifest_sha256=manifest)
expected=[f'seed{s}__{c}' for s in (9101,9203,9307) for c in ('shared4_own','private_missinghop','full_aux','common_nonfull','allblock_missinghop','factor1_allview')]
assert [row['record_id'] for row in plan['records']]==expected
kept=[H/'READINESS.json',H/'ROOT_REVIEW.json',H/'OWNER_REVIEW.json']
for row in plan['records']:
 record=row['record_id'];release=json.loads((H/'releases_disabled'/(record+'.json')).read_text())
 contract=json.loads((H/'EXTERNAL_OWNER_RELEASE_TEMPLATE_DISABLED.json').read_text())
 contract.update(enabled=True,record_id=record,owner_source=binding(H/'queue.py'),owner_review=binding(H/'OWNER_REVIEW.json'))
 contract_path=H/'contracts'/(record+'.json');save(contract_path,contract)
 release.update(enabled=True,root_authorized=True,source_review_approved=True,complete_roster_frozen=True,qualification_verified=True,finite_owner_verified=True,fresh_resource_readiness_confirmed=True,ordinary_runtime_confirmed=True,VALID_access=True,science_enabled=True,source_manifest_sha256=manifest,root_review=binding(H/'ROOT_REVIEW.json'),resource_readiness_evidence=binding(H/'READINESS.json'),external_owner_release=binding(contract_path))
 assert release['TEST_access'] is False and release['HPO'] is False and release['automatic_retry'] is False
 assert release['output']==str(H/'science'/'cells'/record) and not Path(release['output']).exists()
 path=H/'releases'/(record+'.json');save(path,release)
 row.update(release=str(path.relative_to(P)),release_sha256=sha(path))
 kept.extend((path,contract_path))
save(H/'OWNER_PLAN.json',plan);kept.append(H/'OWNER_PLAN.json')
with (A/'owner.stdout').open('xb') as out,(A/'owner.stderr').open('xb') as err:
 process=subprocess.Popen([release['runtime']['python']['path'],'-B','-P',str(H/'queue.py'),'--plan',str(H/'OWNER_PLAN.json'),'--plan-sha256',sha(H/'OWNER_PLAN.json')],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 f=Path('/proc',str(process.pid),'stat').read_text().rsplit(')',1)[1].split()
 starter=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=process.pid,start_ticks=int(f[19]),pgid=int(f[2]),sid=int(f[3]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_commit=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),source_manifest_sha256=manifest,records=expected,scientific_fits=18,VALID_access=True,TEST_access=False,automatic_retry=False,normal_host_execution=True,other_jobs_changed=False)
 assert starter['pgid']==starter['sid']==process.pid
 save(A/'STARTER.json',starter);kept.append(A/'STARTER.json')
print(json.dumps(dict(started=starter,retained={str(q.relative_to(P)):json.loads(q.read_text()) for q in kept})))
'''

def main():
    assert not (HERE/'TRANSPORT.json').exists()
    argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(REMOTE)]
    r=subprocess.run(argv,capture_output=True,text=True,timeout=60)
    with (HERE/'TRANSPORT.json').open('x') as f:json.dump(dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),f,indent=2)
    if r.returncode:
        print(r.stderr);raise SystemExit(r.returncode)
    v=json.loads(r.stdout)
    for name,row in v['retained'].items():
        q=HERE/'fetched'/name;assert q.resolve().is_relative_to((HERE/'fetched').resolve());q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(v['started']))

if __name__=='__main__':main()
