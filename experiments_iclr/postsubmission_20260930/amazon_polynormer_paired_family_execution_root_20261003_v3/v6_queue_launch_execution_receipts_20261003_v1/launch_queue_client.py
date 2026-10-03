"""Publish one exact queue and detach its unchanged ordinary supervisor."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import json
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
PHASE=HERE.parents[1]
RUNNER='''from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'amazon_polynormer_paired_family_execution_root_20261003_v3'
launch=root/'v6_queue_detached_launch_20261003_v1'
release=root/'v6_releases/all_v1.json'
output=root/'v6_full_schedule_v1'
source=phase/'amazon_polynormer_paired_family_source_preparation_20261003_v6'
def identity(pid):
 p=Path('/proc')/str(pid);raw=(p/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(v[19]),ppid=int(v[1]),pgid=int(v[2]),sid=int(v[3]),state=v[0],argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x])
def write(p,v):
 with p.open('x') as h:json.dump(v,h,indent=2,allow_nan=False);h.write('\\n');h.flush();os.fsync(h.fileno())
def desc(p):
 b=p.read_bytes();return dict(path=str(p.relative_to(phase)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
started=time.perf_counter()
write(launch/'RUNNER_STARTED.json',dict(UTC=datetime.now(timezone.utc).isoformat(),identity=identity(os.getpid()),automatic_retry=False))
command=['/usr/bin/python3','-B',str(source/'supervise.py'),'--kind','all','--release',str(release),'--output',str(output)]
with (launch/'SUPERVISOR_STDOUT.txt').open('x') as out,(launch/'SUPERVISOR_STDERR.txt').open('x') as err:
 child=subprocess.Popen(command,cwd=phase,stdin=subprocess.DEVNULL,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
 write(launch/'SUPERVISOR_LAUNCH.json',dict(UTC=datetime.now(timezone.utc).isoformat(),supervisor_identity=identity(child.pid),runner_identity=identity(os.getpid()),argv=command,cwd=str(phase),release=desc(release),ordinary_unchanged_supervisor=True,outer_timeout_or_retry=False,detached_runner_survives_SSH=True))
 code=child.wait()
write(launch/'SUPERVISOR_TERMINAL.json',dict(UTC=datetime.now(timezone.utc).isoformat(),status='success' if code==0 else 'failed',physical_supervisor_exit_code=code,whole_runner_wall_seconds=time.perf_counter()-started,release=desc(release),supervisor_stdout=desc(launch/'SUPERVISOR_STDOUT.txt'),supervisor_stderr=desc(launch/'SUPERVISOR_STDERR.txt'),automatic_retry=False))
raise SystemExit(code)
'''
CODE='''from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,subprocess,time
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'amazon_polynormer_paired_family_execution_root_20261003_v3'
source=phase/'amazon_polynormer_paired_family_source_preparation_20261003_v6'
candidate=root/'v6_admitted_resource_fit_release_preparation_20261003_v1/ALL_FITS_RELEASE_DISABLED.json'
release=root/'v6_releases/all_v1.json'
output=root/'v6_full_schedule_v1'
launch=root/'v6_queue_detached_launch_20261003_v1'
os.chdir(repo)
g=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
assert Path.cwd()==repo and g.returncode==0 and len(g.stdout.strip().splitlines())==1
gpu=[v.strip() for v in g.stdout.strip().split(',')];assert gpu[0]=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
route=dict(repository=str(repo),GPU_UUID=gpu[0])
memory={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
cg={}
for n in ('memory.max','memory.current'):
 p=Path('/sys/fs/cgroup')/n
 if p.is_file():cg[n]=p.read_text().strip()
headroom=int(cg['memory.max'])-int(cg['memory.current']) if cg.get('memory.max','max').isdigit() and cg.get('memory.current','').isdigit() else None
fs=os.statvfs(phase)
resources=dict(UTC=datetime.now(timezone.utc).isoformat(),GPU_free_bytes=int(gpu[1])*2**20,GPU_used_bytes=int(gpu[2])*2**20,GPU_utilization_percent=int(gpu[3]),host_MemAvailable_bytes=memory['MemAvailable'],cgroup=cg,cgroup_available_bytes=headroom,filesystem_available_bytes=fs.f_bavail*fs.f_frsize,quota_status='UNKNOWN_UNCERTIFIED',physical_allocation_expiry_status='UNKNOWN_UNCERTIFIED')
assert resources['GPU_free_bytes']>=75*2**30 and memory['MemAvailable']>=32*2**30 and (headroom is None or headroom>=32*2**30) and resources['filesystem_available_bytes']>=32*2**30
def desc(p):
 assert p.suffix in ('.json','.jsonl','.py','.txt')
 b=p.read_bytes();b.decode('utf8');return dict(path=str(p.relative_to(phase)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def verify(r):
 p=phase/r['path'];assert not Path(r['path']).is_absolute() and '..' not in p.parts and not p.is_symlink()
 assert desc(p)==r;return p
def write(p,value):
 with p.open('x') as h:json.dump(value,h,indent=2,allow_nan=False);h.write('\\n');h.flush();os.fsync(h.fileno())
 return desc(p)
def identity(pid):
 p=Path('/proc')/str(pid);raw=(p/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(v[19]),ppid=int(v[1]),pgid=int(v[2]),sid=int(v[3]),state=v[0],argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x])
assert desc(candidate)==dict(path=str(candidate.relative_to(phase)),sha256='f2d04782d47f629c709dfde7fe0098a1fbf85e376c5aedceb58e490a6d8fea4d',bytes=81078)
q=json.loads(candidate.read_text());assert q['execution_authorized'] is False and q['kind']=='all' and q['device']=='cuda:0'
assert q['self_path']==str(release.relative_to(phase)) and q['output']==str(output.relative_to(phase))
assert q['caps']==dict(wall_seconds=673200,rss_bytes=32*2**30,cuda_peak_allocated_bytes=75*2**30,cuda_peak_reserved_bytes=75*2**30)
assert q['automatic_retry_authorized'] is False and q['test_labels_authorized'] is False
assert not release.exists() and not output.exists() and not launch.exists()
registry=json.loads(verify(q['registry']).read_text());assert len(registry['physical_fits'])==len(q['fit_releases'])==15
assert not any((phase/r[k]).exists() for r in registry['physical_fits'] for k in ('output','claim_path'))
assert not (root/'v6_closure_v1').exists()
resource=json.loads(verify(q['resource_admission']).read_text());assert resource['execution_authorized'] is True and resource['full_15_fit_schedule_authorized'] is True and resource['source']==q['source'] and resource['registry']==q['registry'] and resource['qualification_freeze']==q['qualification_freeze']
assert resource['quota_status']==resource['physical_allocation_expiry_status']=='UNKNOWN_UNCERTIFIED' and resource['prospective_queue_time_budget_seconds']==691200 and resource['prospective_storage_budget_bytes']==32*2**30
verify(q['attempt_registry']);verify(q['consumer_release']);verify(q['runtime_receipt']);verify(q['source_review']);verify(q['qualification_freeze'])
manifest=json.loads(verify(q['source']['manifest']).read_text());assert json.loads(verify(q['source']['seal']).read_text())['manifest']==q['source']['manifest']
for row in manifest['payload']:
 actual=desc(source/row['path']);assert (actual['sha256'],actual['bytes'])==(row['sha256'],row['bytes'])
for r,row in zip(registry['physical_fits'],q['fit_releases']):
 body=json.loads(verify(row).read_text());assert body['execution_authorized'] is True and body['kind']=='fit' and body['fit_id']==r['id'] and body['self_path']==r['release_path'] and body['output']==r['output'] and body['registered_claim_path']==r['claim_path']
 assert body['source']==q['source'] and body['registry']==q['registry'] and body['resource_admission']==q['resource_admission'] and body['qualification_freeze']==q['qualification_freeze'] and body['attempt_registry']==q['attempt_registry'] and body['test_labels_authorized'] is False and body['automatic_retry_authorized'] is False
assert json.loads(verify(q['closure_release']).read_text())['execution_authorized'] is True
launch.mkdir()
preflight=write(launch/'PREFLIGHT.json',dict(UTC=datetime.now(timezone.utc).isoformat(),route=route,resources=resources,disabled_queue=desc(candidate),all15_fit_outputs_claims_and_queue_release_output_absent=True,root_explicit_authorization='One exact original fixed15 queue, exact source/registry/reviews/resource/qualification and actual15fit/closure hashes, ordinary detached unchanged supervision, no TEST/control scoring, retry, replacement cohort or new registry.',queue_cap_seconds=673200,prospective_budget_seconds=691200))
published=write(release,dict(q,execution_authorized=True));assert json.loads(release.read_text())==dict(q,execution_authorized=True)
runner_source=RUNNER_SOURCE
runner=launch/'run_queue_detached.py'
with runner.open('x') as h:h.write(runner_source);h.flush();os.fsync(h.fileno())
compile(runner_source,str(runner),'exec')
with (launch/'RUNNER_STDOUT.txt').open('x') as out,(launch/'RUNNER_STDERR.txt').open('x') as err:
 process=subprocess.Popen(['/usr/bin/python3','-B',str(runner)],cwd=phase,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,close_fds=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
 runner_identity=identity(process.pid)
detached=write(launch/'DETACHED_LAUNCH.json',dict(UTC=datetime.now(timezone.utc).isoformat(),route=route,published_queue=published,runner_source=desc(runner),runner_identity=runner_identity,ordinary_detached_supervised_execution=True,automatic_retry=False,only_queue_execution_authorized_changed=True))
time.sleep(0.3)
files=[candidate,release,launch/'PREFLIGHT.json',runner,launch/'DETACHED_LAUNCH.json']
for name in ('RUNNER_STARTED.json','SUPERVISOR_LAUNCH.json','SUPERVISOR_TERMINAL.json'):
 p=launch/name
 if p.is_file():files.append(p)
new_metadata=[dict(descriptor=desc(p),base64=base64.b64encode(p.read_bytes()).decode()) for p in files]
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),route=route,resources_before_launch=resources,disabled_queue=desc(candidate),published_queue=published,changed_fields=['execution_authorized'],runner_identity=runner_identity,runner_source=desc(runner),remote_launch_metadata_root=str(launch.relative_to(phase)),ordinary_detached_supervised_execution=True,automatic_retry=False,fit_TEST_control_scoring_or_new_registration_beyond_exact_queue=False,new_metadata=new_metadata)))
'''
CODE=CODE.replace('runner_source=RUNNER_SOURCE','runner_source='+repr(RUNNER))
with (HERE/'DETACHED_RUNNER_SOURCE.py').open('x') as h:h.write(RUNNER)
with (HERE/'LAUNCH_REMOTE_CODE.py.txt').open('x') as h:h.write(CODE)
ssh=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']
started=datetime.now(timezone.utc).isoformat()
r=subprocess.run([*ssh,shlex.join(['/usr/bin/python3','-I','-S','-B','-c',CODE])],capture_output=True,text=True)
transport=dict(start_UTC=started,terminal_UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stderr=r.stderr,stdout_sha256=hashlib.sha256(r.stdout.encode()).hexdigest(),client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),remote_code_sha256=hashlib.sha256(CODE.encode()).hexdigest(),private_key_contents_read=False,automatic_retry=False)
with (HERE/'LAUNCH_TRANSPORT.json').open('x') as h:json.dump(transport,h,indent=2);h.write('\n')
if r.stdout:
    value=json.loads(r.stdout);files=value.pop('new_metadata');value['fetched_descriptors']=[]
    for entry in files:
        row=entry['descriptor'];raw=base64.b64decode(entry['base64'],validate=True)
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        p=PHASE/row['path'];assert p.resolve().is_relative_to(PHASE)
        if p.exists():assert p.read_bytes()==raw
        else:
            p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as h:h.write(raw)
        value['fetched_descriptors'].append(row)
    with (HERE/'LAUNCH_RESULT.json').open('x') as h:json.dump(value,h,indent=2);h.write('\n')
    print(json.dumps(value,indent=2))
else:print(json.dumps(transport,indent=2))
raise SystemExit(r.returncode)
