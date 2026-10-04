"""Admit one reviewed Pubmed engineering attempt; no scientific fit."""
from datetime import datetime,timezone
from pathlib import Path
import base64
import hashlib
import json
from stage_source_metadata import HERE,PHASE,REPO,REMOTE_PHASE,REMOTE,SOURCE,run

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')

observation=json.loads((HERE/'SOURCE_PREREQUISITE_RESOURCE_STAGE_RECEIPT.json').read_text())
assert observation['status']=='PASS_EXACT_SOURCE_AND_PREREQUISITE_METADATA'
review_path=PHASE/'pubmed_shared4_bridge_fresh_source_review_20261004_v2/REVIEW.json'
assert sha(review_path)=='51c95a8e6700059f1ed61e5019cad6ca6ab85ced68b99827973c33a7238d6086'
review=json.loads(review_path.read_text())
manifest_sha=sha(SOURCE/'MANIFEST.json')
assert review['status']=='PASS' and review['candidate_manifest_sha256']==manifest_sha and review['execution_authorized'] is False
for row in json.loads((SOURCE/'MANIFEST.json').read_text())['files']:
    f=SOURCE/row['path'];assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256']
plan=json.loads((SOURCE/'PLAN.json').read_text())
admission=dict(schema='pubmed-shared4-root-engineering-admission-v1',UTC=datetime.now(timezone.utc).isoformat(),status='APPROVED',
               source_manifest_sha256=manifest_sha,independent_review_sha256=sha(review_path),
               root_source_review='Native puregcn/activeJK, original TRAIN bodies/observers, geometry, owned complete-epoch replay, explicit frozen tolerances and ordinary owned supervisor inspected.',
               scope='One fresh 504-update engineering qualifier; 14 full native epochs/four complete VALID. No TEST/science/donor.',
               resource_metadata=observation,dispatch_recheck_required=True,existing_qualified_states_read=False,
               geometry_donors_for_resource_or_science=False,original_paper_scores_changed=False)
write(HERE/'ROOT_ADMISSION.json',admission)
release=json.loads((SOURCE/'ROOT_RELEASE_TEMPLATE.json').read_text())
release.update(status='APPROVED',root_authorization_reference=str(REMOTE/'ROOT_ADMISSION.json'),root_source_review_approved=True,
               authorized_stages=['qualification'],caps=plan['stages']['qualification']['caps'],
               invocation=plan['stages']['qualification']['invocation'],input_authority=plan['input_authority'],
               existing_qualification_reused=True,source_manifest_sha256=manifest_sha,plan_sha256=sha(SOURCE/'PLAN.json'),
               independent_source_review_path=str(REMOTE_PHASE/review_path.relative_to(PHASE)),independent_source_review_sha256=sha(review_path))
for name,row in release['prerequisites'].items():
    actual=observation['prerequisites'][name]
    assert actual['verified_by_root'] is True and actual['path']==row['path'] and (not row['sha256'] or row['sha256']==actual['sha256'])
    row.update(sha256=actual['sha256'],verified_by_root=True)
write(HERE/'ROOT_RELEASE_qualification.json',release)
files=[]
for local,remote in ((HERE/'ROOT_ADMISSION.json',REMOTE/'ROOT_ADMISSION.json'),
                     (HERE/'ROOT_RELEASE_qualification.json',REMOTE/'ROOT_RELEASE_qualification.json'),
                     (review_path,REMOTE_PHASE/review_path.relative_to(PHASE))):
    raw=local.read_bytes();files.append(dict(path=str(remote),bytes=len(raw),sha256=sha(local),data=base64.b64encode(raw).decode()))
environment=dict(plan['execution_profile']['environment'],GNNM_SSH_DESTINATION='shmelev@192.168.18.77',PYTHONDONTWRITEBYTECODE='1')
command=[plan['execution_profile']['interpreter_path'],'-B',str(REMOTE_PHASE/SOURCE.name/'supervise.py'),'--stage','qualification',
         '--root-release',str(REMOTE/'ROOT_RELEASE_qualification.json'),'--release-sha256',sha(HERE/'ROOT_RELEASE_qualification.json')]
code='from pathlib import Path\nfrom datetime import datetime,timezone\nimport base64,hashlib,json,os,subprocess\n'
code+='repo=Path('+repr(str(REPO))+');phase=Path('+repr(str(REMOTE_PHASE))+');root=Path('+repr(str(REMOTE))+')\n'
code+='assert Path.cwd()==repo and os.uname().nodename=="peptide"\nrows='+repr(files)+'\n'
code+="for r in rows:\n p=Path(r['path']);assert p.resolve().is_relative_to(phase);b=base64.b64decode(r['data'],validate=True);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']\n if p.exists():assert not p.is_symlink() and p.read_bytes()==b\n else:\n  p.parent.mkdir(parents=True,exist_ok=True)\n  with p.open('xb') as f:f.write(b)\n"
code+="assert not (root/'qualification/run01').exists() and not (root/'supervision/qualification/run01').exists()\n"
code+="g=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15);assert g.returncode==0\n"
code+="gpu_rows=[[v.strip() for v in line.split(',')] for line in g.stdout.splitlines()];assert int(next(r[1] for r in gpu_rows if r[0]=='GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'))>=73728\n"
code+="available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'));assert available>=40*1024**3\n"
code+='environment='+repr(environment)+';command='+repr(command)+'\n'
code+="with (root/'DETACHED_STDOUT.txt').open('xb') as out,(root/'DETACHED_STDERR.txt').open('xb') as err:\n proc=subprocess.Popen(command,cwd=repo,env=dict(os.environ,**environment),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)\n"
code+="raw=(Path('/proc')/str(proc.pid)/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()\n"
code+="result=dict(UTC=datetime.now(timezone.utc).isoformat(),status='ENGINEERING_QUALIFIER_LAUNCHED',supervisor_PID=proc.pid,supervisor_start_ticks=int(fields[19]),command=command,environment=environment,GPU_rows=gpu_rows,host_MemAvailable_bytes=available,science_fits=0,TEST_access=False,own_engineering_updates_planned=504)\n"
code+="with (root/'DETACHED_LAUNCH.json').open('x') as f:json.dump(result,f,indent=2);f.write('\\n')\nprint(json.dumps(result))\n"
result=run('pubmed_bridge_qual_v2_ordinary_launch_20261004',code)
write(HERE/'DETACHED_LAUNCH.json',result)
print(json.dumps({k:result[k] for k in ('UTC','status','supervisor_PID','supervisor_start_ticks','science_fits','TEST_access','own_engineering_updates_planned')}))
