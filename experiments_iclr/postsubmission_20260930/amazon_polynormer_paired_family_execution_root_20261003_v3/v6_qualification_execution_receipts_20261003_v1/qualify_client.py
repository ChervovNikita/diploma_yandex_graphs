"""One root-authorized V6 qualification; exact publication and ordinary supervision."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
COMMON = '''from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,subprocess,sys
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'amazon_polynormer_paired_family_execution_root_20261003_v3'
source=phase/'amazon_polynormer_paired_family_source_preparation_20261003_v6'
packet=root/'v6_execution_metadata_preparation_v1/after_runtime_disabled_v1'
candidate=packet/'disabled_releases/qualification.json'
release=root/'v6_releases/qualification_v1.json'
output=root/'v6_qualification_block0_cuda0_v1'
os.chdir(repo)
g=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
assert Path.cwd()==repo and g.returncode==0 and g.stdout.strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
route=dict(repository=str(repo),GPU_UUID=g.stdout.strip())
def desc(p):
 b=p.read_bytes();return dict(path=str(p.relative_to(phase)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def verify(row):
 p=phase/row['path'];assert not Path(row['path']).is_absolute() and '..' not in p.parts
 assert p.is_relative_to(phase) and not any(q.is_symlink() for q in (p,*p.parents) if q.is_relative_to(phase))
 assert p.suffix not in ('.pt','.npz','.npy','.pth') and desc(p)==row
 return p
def reviewed_candidate():
 assert desc(candidate)==dict(path=str(candidate.relative_to(phase)),sha256='5ddd62e790d75d9586fabb1d5b3a6177e5f568eb47a4e6c00f43d73715278945',bytes=54470)
 q=json.loads(candidate.read_text());assert q['execution_authorized'] is False and q['kind']=='qualify'
 assert q['self_path']==str(release.relative_to(phase)) and q['output']==str(output.relative_to(phase))
 assert q['device']=='cuda:0' and q['automatic_retry_authorized'] is False and q['test_labels_authorized'] is False
 assert q['caps']==dict(wall_seconds=3600,rss_bytes=32*2**30,cuda_peak_allocated_bytes=75*2**30,cuda_peak_reserved_bytes=75*2**30)
 assert q['source']['manifest']['sha256']=='d4160d8aeec2b770274875f9a1fb137facc9f93efd4eca6ad91e58802521ff44'
 m=json.loads(verify(q['source']['manifest']).read_text());assert json.loads(verify(q['source']['seal']).read_text())['manifest']==q['source']['manifest']
 for row in m['payload']:
  actual=desc(source/row['path']);assert (actual['sha256'],actual['bytes'])==(row['sha256'],row['bytes'])
 for key in ('source_review','registry','runtime_receipt','consumer_release','attempt_registry'):verify(q[key])
 review=json.loads(verify(q['source_review']).read_text());assert review['status']=='passed' and review['source']==q['source']
 assert review['review_outcome']=='PASS_SOURCE_ONLY' and review['execution_authorized'] is False
 verify(review['independent_review_report'])
 for name in ('v6_runtime_cpu_v1','v6_runtime_cuda0_v1'):
  d=root/name;f=json.loads((d/'FREEZE.json').read_text());t=json.loads((d/'TERMINAL.json').read_text())
  assert f['status']=='success' and f['source']==q['source'] and t['status']=='success' and t['physical_exit_code']==0
  for r in f['files']:verify(r['descriptor'])
 for row in q['custody_inputs']:
  if Path(row['path']).suffix in ('.pt','.npz','.npy','.pth'):
   p=phase/row['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes']
  else:verify(row)
 registry=json.loads(verify(q['registry']).read_text());assert len(registry['physical_fits'])==15 and len(registry['families'])==9
 cohort=json.loads((source/'COHORT_SOURCE_BINDING.json').read_text());assert q['registry']==cohort['registry'] and registry['source']==cohort['registered_source']
 verify(cohort['master_source_claim'])
 assert not any((phase/r[k]).exists() for r in registry['physical_fits'] for k in ('output','release_path','claim_path'))
 return q
def resources():
 r=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
 assert r.returncode==0 and len(r.stdout.strip().splitlines())==1
 v=[s.strip() for s in r.stdout.strip().split(',')];assert v[0]==route['GPU_UUID']
 memory={k:int(val.split()[0])*1024 for k,val in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
 fs=os.statvfs(phase)
 cg={}
 for n in ('memory.max','memory.current','memory.high'):
  p=Path('/sys/fs/cgroup')/n
  if p.is_file():cg[n]=p.read_text().strip()
 cg_available=None
 if cg.get('memory.max','max').isdigit() and cg.get('memory.current','').isdigit():cg_available=int(cg['memory.max'])-int(cg['memory.current'])
 return dict(UTC=datetime.now(timezone.utc).isoformat(),GPU=dict(uuid=v[0],name=v[1],total_bytes=int(v[2])*2**20,free_bytes=int(v[3])*2**20,used_bytes=int(v[4])*2**20,utilization_percent=int(v[5])),host=dict(MemTotal_bytes=memory['MemTotal'],MemAvailable_bytes=memory['MemAvailable']),cgroup=cg,cgroup_available_bytes=cg_available,filesystem=dict(path=str(phase),available_bytes=fs.f_bavail*fs.f_frsize,total_bytes=fs.f_blocks*fs.f_frsize),RSS_cap_currently_available=memory['MemAvailable']>=32*2**30 and (cg_available is None or cg_available>=32*2**30),CUDA_cap_currently_available=int(v[3])*2**20>=75*2**30)
def require_resources(r):
 assert r['RSS_cap_currently_available'] and r['CUDA_cap_currently_available'],r
def small_files(paths):
 result=[];total=0
 for p in paths:
  assert p.is_file() and p.suffix in ('.json','.jsonl','.txt') and p.stat().st_size<=2*2**20
  b=p.read_bytes();b.decode('utf8');total+=len(b);assert total<=8*2**20
  result.append(dict(descriptor=desc(p),base64=base64.b64encode(b).decode()))
 return result
'''
ACTIONS = {
    'preflight': '''q=reviewed_candidate();assert not release.exists() and not output.exists()
r=resources()
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),route=route,candidate=desc(candidate),source=q['source'],resources=r,release_output_absent=True,all15fit_paths_absent=True,read_only=True,checkpoint_arrays_stat_only=True)))
require_resources(r)
''',
    'publish': '''q=reviewed_candidate();assert not release.exists() and not output.exists()
r=resources();require_resources(r)
published=dict(q,execution_authorized=True)
assert [k for k in published if published[k]!=q[k]]==['execution_authorized']
with release.open('x') as h:json.dump(published,h,indent=2,allow_nan=False);h.write('\\n');h.flush();os.fsync(h.fileno())
assert json.loads(release.read_text())==published
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),route=route,resources=r,candidate=desc(candidate),published=desc(release),changed_fields=['execution_authorized'],root_explicit_authorization='One exact V6 qualification only, caps 3600 s / 32 GiB RSS / 75 GiB CUDA allocated and reserved; no fits, TEST, controls, scoring, retries or new registration.',files_overwritten=False,numerical_launch=False)))
''',
    'run': '''q=reviewed_candidate();assert json.loads(release.read_text())==dict(q,execution_authorized=True) and not output.exists()
r=resources();require_resources(r)
command=['/usr/bin/python3','-B',str(source/'supervise.py'),'--kind','qualify','--release',str(release),'--output',str(output)]
s=subprocess.run(command,cwd=phase,capture_output=True,text=True,stdin=subprocess.DEVNULL)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),route=route,resources_before_launch=r,release=desc(release),supervisor_command=command,physical_supervisor_exit_code=s.returncode,supervisor_stdout=s.stdout,supervisor_stderr=s.stderr,ordinary_supervision=True,automatic_retry=False,predictive_fits_TEST_control_scoring_or_new_registration=False)))
sys.exit(s.returncode)
''',
    'fetch': '''q=reviewed_candidate();paths=[release]
for n in ('FREEZE.json','LAUNCH.json','READY.json','RESULT.json','TERMINAL.json','FAILURE.json','OUTER_TRACEBACK.txt','stdout.txt','stderr.txt','INITIALIZATION_PAIRING.json'):
 p=output/n
 if p.is_file():paths.append(p)
for p in sorted(output.glob('PARITY_*.json')):paths.append(p)
for d in sorted(output.iterdir()):
 if d.is_dir():
  for n in ('FORM.json','RETIREMENT_PROBE.jsonl'):
   p=d/n
   if p.is_file():paths.append(p)
data=small_files(paths)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),route=route,read_only=True,large_model_or_array_states_read=False,files=data)))
''',
    'after-qualification': '''q=reviewed_candidate();t=json.loads((output/'TERMINAL.json').read_text());assert t['status']=='success' and t['physical_exit_code']==0
helper=root/'v6_execution_metadata_preparation_v1/prepare_disabled_metadata.py'
assert desc(helper)['sha256']=='44e2727b933bedcddd5158738f19e68d9b223d49eddf276e9730412db806429b'
target=helper.parent/'after_qualification_disabled_v1';assert not target.exists()
command=['/usr/bin/python3','-I','-S','-B',str(helper),'--stage','after-qualification','--packet-name','after_qualification_disabled_v1','--review','amazon_polynormer_v6_bounded_retention_forecast_independent_source_review_20261003_v1/REVIEW.json','--runtime-gate-review','amazon_polynormer_v6_source_review_runtime_compatibility_receipt_20261003_v1/REVIEW.json','--consumer-release','amazon_polynormer_paired_family_execution_root_20261003_v3/V6_CONSUMER_RELEASE_v1.json']
s=subprocess.run(command,cwd=phase,capture_output=True,text=True,stdin=subprocess.DEVNULL)
data=small_files(sorted(p for p in target.rglob('*') if p.is_file())) if s.returncode==0 else []
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),route=route,command=command,exit_code=s.returncode,stdout=s.stdout,stderr=s.stderr,metadata_only=True,files=data,execution_authorized=False,predictive_fits_TEST_control_scoring_or_new_registration=False)))
sys.exit(s.returncode)
'''
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=ACTIONS)
    args = parser.parse_args()
    paths = {key: HERE / (args.action + '_' + key + '.json') for key in ('TRANSPORT', 'RESULT')}
    assert not any(p.exists() for p in paths.values()), 'Once-only action receipt exists'
    code = COMMON + ACTIONS[args.action]
    code_path = HERE / (args.action + '_REMOTE_CODE.py.txt')
    with code_path.open('x') as handle:
        handle.write(code)
    ssh = ['ssh', '-T', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
           '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
           '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2', LOGIN]
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run([*ssh, shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', code])],
                            capture_output=True, text=True)
    transport = dict(start_UTC=started, terminal_UTC=datetime.now(timezone.utc).isoformat(),
                     destination=LOGIN, port=2222, action=args.action, exit_code=result.returncode,
                     stderr=result.stderr, stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
                     client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     remote_code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                     ordinary_supervision=True, automatic_retry=False, private_key_contents_inspected=False)
    with paths['TRANSPORT'].open('x') as handle:
        json.dump(transport, handle, indent=2); handle.write('\n')
    if result.stdout:
        value = json.loads(result.stdout)
        files = value.pop('files', [])
        value['fetched_descriptors'] = []
        for entry in files:
            row = entry['descriptor']; raw = base64.b64decode(entry['base64'], validate=True)
            assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
            target = PHASE / row['path']
            assert not Path(row['path']).is_absolute() and target.resolve().is_relative_to(PHASE)
            if target.exists():
                assert target.read_bytes() == raw
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as handle: handle.write(raw)
            value['fetched_descriptors'].append(row)
        value['local_existing_files_overwritten'] = False
        with paths['RESULT'].open('x') as handle:
            json.dump(value, handle, indent=2); handle.write('\n')
        print(json.dumps(value, indent=2), flush=True)
    else:
        print(json.dumps(transport, indent=2), flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
