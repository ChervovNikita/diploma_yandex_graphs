"""Start one fixed normalization-control stage with the existing finite owner."""
import argparse,json,shlex,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
REMOTE=r'''
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,shutil,socket,subprocess,sys
assert socket.gethostname()=='anogena-2-0'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==[GPU]
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
H=P/'pubmed_own4_M_normalization_source_20261010_v1';A=P/'pubmed_own4_M_normalization_root_activation_20261010_v1'
stage=sys.argv[1];assert stage in ('qualification','science','comparison')
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
def save(q,v):
 q.parent.mkdir(parents=True,exist_ok=True)
 with q.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def binding(q):return dict(path=str(q),sha256=sha(q))
manifest='c6cdb03c639866a8ce0dff9ad52e2cb7a0b6fd207d40b83750c8b9d5857c2046'
assert sha(H/'SOURCE_MANIFEST.json')==manifest
for row in json.loads((H/'SOURCE_MANIFEST.json').read_text())['files']:
 q=(H/row['path']).resolve(strict=True);assert q.is_relative_to(H) and sha(q)==row['sha256']
target=A/stage;target.mkdir(exist_ok=False)
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip())*1024**2
mem={r.split(':')[0]:int(r.split(':')[1].strip().split()[0])*1024 for r in Path('/proc/meminfo').read_text().splitlines() if ':' in r}
assert (stage=='comparison' or free>=32*1024**3) and mem['MemAvailable']>=16*1024**3 and shutil.disk_usage(P).free>=3*1024**3
readiness=H/('READINESS_'+stage+'.json');save(readiness,dict(UTC=datetime.now(timezone.utc).isoformat(),GPU_free_bytes=free,MemAvailable_bytes=mem['MemAvailable'],normal_host_execution=True,overlapping_acquisition=True,other_jobs_changed=False))
review=json.loads((H/'ROOT_REVIEW.json').read_text());assert review['approved'] and review['source_manifest_sha256']==manifest
owner=json.loads((H/'OWNER_REVIEW.json').read_text());assert owner['approved'] and owner['owner_sha256']==sha(H/'queue.py')
qualification=None
if stage!='qualification':
 family=json.loads((H/'qualification/FAMILY_COMPLETE.json').read_text());assert family['complete'] and not (H/'qualification/FAMILY_FAILURE.json').exists()
 starter=json.loads((A/'qualification/STARTER.json').read_text());assert not Path('/proc',str(starter['PID'])).exists()
 record='seed9101__own4_M_normalization_TRAIN_qualification'
 qualification=(H/'qualification/cells'/record/'COMPLETE.json',H/'owners'/('qualification__'+record)/'RAW_OWNER_TERMINAL.json')
if stage=='comparison':
 assert json.loads((H/'science/FAMILY_COMPLETE.json').read_text())['complete']
 assert json.loads((P/'private_hop_credit_pubmed_fullfit_source_20261010_v1/science/FAMILY_COMPLETE.json').read_text())['complete']
plan=json.loads((H/(stage.upper()+'_OWNER_PLAN_TEMPLATE_DISABLED.json')).read_text())
plan.update(enabled=True,source_manifest_sha256=manifest)
kept=[readiness,H/'ROOT_REVIEW.json',H/'OWNER_REVIEW.json']
for row in plan['records']:
 rid=row['record_id'];spec=json.loads((H/'releases_disabled'/(stage+'__'+rid+'.json')).read_text())
 contract=json.loads((H/(stage.upper()+'_OWNER_RELEASE_TEMPLATE_DISABLED.json')).read_text())
 contract.update(enabled=True,record_id=rid,owner_source=binding(H/'queue.py'),owner_review=binding(H/'OWNER_REVIEW.json'))
 cp=H/'contracts'/(stage+'__'+rid+'.json');save(cp,contract)
 spec.update(enabled=True,root_authorized=True,source_review_approved=True,complete_roster_frozen=True,native_shared_groups_unchanged=True,fresh_resource_readiness_confirmed=True,ordinary_runtime_confirmed=True,external_hard_bound_confirmed=True,source_manifest_sha256=manifest,root_review=binding(H/'ROOT_REVIEW.json'),resource_readiness_evidence=binding(readiness),external_owner_release=binding(cp))
 if qualification:spec.update(qualification_complete=binding(qualification[0]),qualification_terminal=binding(qualification[1]))
 if stage=='comparison':spec.update(science_owner_plan=binding(H/'SCIENCE_OWNER_PLAN.json'),full18_owner_plan=binding(P/'private_hop_credit_pubmed_fullfit_source_20261010_v1/OWNER_PLAN.json'))
 assert spec['TEST_access'] is False and spec['HPO'] is False and not Path(spec['output']).exists()
 rp=H/'releases'/(stage+'__'+rid+'.json');save(rp,spec);row.update(release=str(rp.relative_to(P)),release_sha256=sha(rp));kept.extend((cp,rp))
pp=H/(stage.upper()+'_OWNER_PLAN.json');save(pp,plan);kept.append(pp)
with (target/'owner.stdout').open('xb') as out,(target/'owner.stderr').open('xb') as err:
 process=subprocess.Popen([spec['runtime']['python']['path'],'-B','-P',str(H/'queue.py'),'--plan',str(pp),'--plan-sha256',sha(pp)],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 f=Path('/proc',str(process.pid),'stat').read_text().rsplit(')',1)[1].split()
 starter=dict(UTC=datetime.now(timezone.utc).isoformat(),stage=stage,PID=process.pid,start_ticks=int(f[19]),pgid=int(f[2]),sid=int(f[3]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_commit=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),source_manifest_sha256=manifest,scientific_fits=3 if stage=='science' else 0,VALID_access=stage!='qualification',TEST_access=False,automatic_retry=False,other_jobs_changed=False)
 assert starter['pgid']==starter['sid']==process.pid;save(target/'STARTER.json',starter);kept.append(target/'STARTER.json')
print(json.dumps(dict(started=starter,retained={str(q.relative_to(P)):json.loads(q.read_text()) for q in kept})))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=('qualification','science','comparison'),required=True);a=p.parse_args()
 receipt=HERE/(a.stage+'_TRANSPORT.json');assert not receipt.exists()
 argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(REMOTE)+' '+shlex.quote(a.stage)]
 r=subprocess.run(argv,capture_output=True,text=True,timeout=60);receipt.write_text(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
 assert r.returncode==0,r.stdout[-5000:]+r.stderr
 v=json.loads(r.stdout)
 for name,row in v['retained'].items():
  q=HERE/'fetched'/name;assert q.resolve().is_relative_to((HERE/'fetched').resolve());q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(row,indent=2)+'\n')
 print(json.dumps(v['started']))
if __name__=='__main__':main()
