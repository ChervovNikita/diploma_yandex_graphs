"""Deploy exact small bindings, then launch one detached admitted pilot."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import base64
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
EXECUTION = 'amazon_learnability_responsibility_strict_scientific_execution_root_20261006_v2'
RELEASE = 'amazon_learnability_responsibility_sequential_train_only_release_root_20261006_v2'
RUNNER = 'amazon_learnability_responsibility_strict_process_scientific_runner_preparation_20261006_v3'
REVIEW = 'amazon_learnability_responsibility_scientific_v3_and_strict_runner_v3_independent_source_review_20261006_v1'
RUNNER_SHA = '65c63b0b62f49bc852bc4a1db3a47196d5c46c1355a328b8ef838397dc79a6bb'
RUNNER_REVIEW_SHA = 'ccdc219e2b3c332b419609250d32a5c0b5ffe145484c6f6e36dd2bedf4a6330f'
ADMISSION_SHA = '5499d852064b08c064487b537bdb2e3d7476276e558d9924c78da5a969cdee14'

REMOTE = r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,importlib.util,json,os,socket,subprocess,sys
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
request=json.load(sys.stdin)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def confined(relative):
 r=Path(relative);assert not r.is_absolute() and '..' not in r.parts
 p=phase/r;assert p.resolve().is_relative_to(phase)
 cur=phase
 for part in r.parts:cur/=part;assert not cur.is_symlink()
 return p
prepared=[]
for row in request['files']:
 p=confined(row['path']);b=base64.b64decode(row['data'])
 assert len(b)==row['bytes']<2_000_000 and hashlib.sha256(b).hexdigest()==row['sha256']
 if p.exists():assert p.is_file() and p.read_bytes()==b,'Never overwrite differing source/evidence: '+row['path']
 prepared.append((p,b))
for p,b in prepared:
 if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 p.chmod(0o444)
runner=confined(request['runner']);assert sha(runner)==request['runner_sha256']
spec=importlib.util.spec_from_file_location('root_strict_metadata_only',runner);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
w=r.load_worker(confined(request['worker']))
b=w._read(w.PACKET/'SOURCE_BINDINGS.json');q=w._read(w.PACKET/'QUEUE.json')
a,c=w._prerequisites(phase,b,q,'cuda:0')
assert sha(w.PACKET/'PREREQUISITES.json')==request['admission_sha256']
strict,evidence=r.strict_evidence(w,phase,a,c,(phase/r.PUBLIC_B_RELATIVE).resolve())
assert a['external_watchdog_seconds']==14520>a['scientific_resource_limits']['max_elapsed_seconds']==14400
assert not any(x=='torch' or x.startswith('torch.') or x=='numpy' for x in sys.modules)
memory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).strip()
reply={'UTC':datetime.now(timezone.utc).isoformat(),'status':'PASS_REMOTE_STDLIB_METADATA_PREFLIGHT','operation':request['operation'],'host':socket.gethostname(),'GPU_record':memory,'deployed_files':len(prepared),'admission_sha256':sha(w.PACKET/'PREREQUISITES.json'),'numeric_imports':False,'W_access':False,'A_scoring':False}
if request['operation']=='launch':
 release=request['execution_release'];assert release['external_watchdog_seconds']==14520
 supervisor=confined(request['supervisor']);assert sha(supervisor)==release['supervisor_sha256']
 assert sha(confined(release['independent_supervisor_review']['path']))==release['independent_supervisor_review']['sha256']
 assert int(memory.split(',')[-1].strip())>=42000,'Insufficient free memory for the qualified native scope'
 out=confined(request['execution']);assert not out.exists(),'Never restart an observed or unobserved execution identity'
 out.mkdir()
 p=out/'EXECUTION_RELEASE.json';p.write_text(json.dumps(release,indent=2)+'\n');p.chmod(0o444)
 argv=['/usr/bin/python3','-I','-S','-B',str(supervisor)]
 with (out/'supervisor.stdout.log').open('xb') as stdout,(out/'supervisor.stderr.log').open('xb') as stderr:
  proc=subprocess.Popen(argv,cwd=repo,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,close_fds=True)
  raw=(Path('/proc')/str(proc.pid)/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
  owner={'PID':proc.pid,'start_ticks':int(fields[19]),'argv':argv,'cwd':str(repo),'UTC':datetime.now(timezone.utc).isoformat()}
  p=out/'SUPERVISOR_LAUNCH.json';p.write_text(json.dumps(owner,indent=2)+'\n');p.chmod(0o444)
 reply.update(status='DETACHED_SUPERVISOR_LAUNCHED_ONCE',owned_supervisor=owner,execution_directory=str(out),scientific_outcome_pending=True)
elif request['operation']!='deploy':raise ValueError('Unknown operation')
print(json.dumps(reply),flush=True)
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('deploy', 'launch'))
    parser.add_argument('--supervisor-review')
    parser.add_argument('--supervisor-review-sha')
    args = parser.parse_args()
    names = set()
    def add(relative, expected=None, size=None):
        p = PHASE / relative
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        assert p.resolve().is_relative_to(PHASE) and p.is_file() and not p.is_symlink()
        assert p.stat().st_size < 2_000_000
        b = p.read_bytes()
        if expected is not None:
            assert digest(b) == expected and (size is None or len(b) == size)
        names.add(relative)
    for packet in (RELEASE, RUNNER, REVIEW):
        for path in (PHASE / packet).iterdir():
            if path.is_file():
                add(str(path.relative_to(PHASE)))
    bindings = json.loads((PHASE / RELEASE / 'SOURCE_BINDINGS.json').read_text())
    for row in bindings['files'].values():
        add(row['path'], row['sha256'], row['bytes'])
    admission = json.loads((PHASE / RELEASE / 'PREREQUISITES.json').read_text())
    assert digest((PHASE / RELEASE / 'PREREQUISITES.json').read_bytes()) == ADMISSION_SHA
    def metadata_descriptors(value):
        if isinstance(value, dict):
            if {'path', 'sha256', 'bytes'} <= set(value):
                add(value['path'], value['sha256'], value['bytes'])
            else:
                for child in value.values():
                    metadata_descriptors(child)
        elif isinstance(value, list):
            for child in value:
                metadata_descriptors(child)
    metadata_descriptors(admission)
    certificate = json.loads((PHASE / admission['all_six_full_certificate']['path']).read_text())
    for row in certificate['raw_result_evidence']:
        add(row['path'], row['sha256'], row['bytes'])
    add(RUNNER + '/run_scientific.py', RUNNER_SHA)
    add(REVIEW + '/REPORT.md', RUNNER_REVIEW_SHA)
    add(str(HERE.relative_to(PHASE)) + '/supervise_once.py', '58516322fdc0dd89eaddfdc582f2cc5adf8033cf4042cd8740d969d25f90a612')
    add(str(HERE.relative_to(PHASE)) + '/SUPERVISOR_MANIFEST.json')
    files = []
    for relative in sorted(names):
        b = (PHASE / relative).read_bytes()
        files.append({'path': relative, 'bytes': len(b), 'sha256': digest(b), 'data': base64.b64encode(b).decode()})
    request = {'operation': args.operation, 'files': files, 'runner': RUNNER + '/run_scientific.py',
               'runner_sha256': RUNNER_SHA, 'worker': RELEASE + '/six_arm_worker.py', 'admission_sha256': ADMISSION_SHA,
               'supervisor': str(HERE.relative_to(PHASE)) + '/supervise_once.py', 'execution': EXECUTION}
    if args.operation == 'launch':
        assert args.supervisor_review and args.supervisor_review_sha
        add(args.supervisor_review, args.supervisor_review_sha)
        b = (PHASE / args.supervisor_review).read_bytes()
        files.append({'path': args.supervisor_review, 'bytes': len(b), 'sha256': digest(b), 'data': base64.b64encode(b).decode()})
        assert not (PHASE / EXECUTION).exists(), 'Local execution custody exists; do not relaunch'
        release = {'UTC': datetime.now(timezone.utc).isoformat(), 'root_explicit_fit_launch': True,
                   'runner_sha256': RUNNER_SHA, 'scientific_worker_sha256': bindings['files']['worker']['sha256'],
                   'admission_sha256': ADMISSION_SHA, 'supervisor_sha256': digest((HERE / 'supervise_once.py').read_bytes()),
                   'independent_runner_review': {'path': REVIEW + '/REPORT.md', 'sha256': RUNNER_REVIEW_SHA},
                   'independent_supervisor_review': {'path': args.supervisor_review, 'sha256': args.supervisor_review_sha},
                   'external_watchdog_seconds': 14520, 'scientific_limits': admission['scientific_resource_limits'],
                   'fixed_warm_updates': 400, 'fixed_episodes_per_arm': 16,
                   'fixed_controls': ['live', 'uniform', 'margins', 'graph_free', 'permuted', 'stop_q'],
                   'fitting_forward_bill': 5712, 'eventual_total_forward_bill': 5740,
                   'A_scoring': False, 'VALID_TEST_access': False, 'automatic_retry': False,
                   'transport_source_sha256': digest(Path(__file__).read_bytes()), 'remote_deployment_source_sha256': digest(REMOTE.encode())}
        request['execution_release'] = release
        out = PHASE / EXECUTION
        out.mkdir()
        (out / 'EXECUTION_RELEASE.json').write_text(json.dumps(release, indent=2) + '\n')
    else:
        out = HERE
    receipt_path = out / ('DEPLOY_RECEIPT.json' if args.operation == 'deploy' else 'LAUNCH_TRANSPORT_RECEIPT.json')
    assert not receipt_path.exists()
    command = ['ssh', '-T', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
               '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
               '-o', 'UpdateHostKeys=no', '-o', 'ConnectTimeout=20', LOGIN,
               'cd ' + shlex.quote(REPO) + ' && exec /usr/bin/python3 -I -S -B -c ' + shlex.quote(REMOTE)]
    try:
        result = subprocess.run(command, input=json.dumps(request), capture_output=True, text=True, timeout=60)
        receipt = {'UTC': datetime.now(timezone.utc).isoformat(), 'operation': args.operation,
                   'exit_code': result.returncode, 'stderr': result.stderr, 'stdout': result.stdout,
                   'remote_source_sha256': digest(REMOTE.encode())}
        if result.returncode == 0:
            receipt['result'] = json.loads(result.stdout)
    except BaseException as error:
        receipt = {'UTC': datetime.now(timezone.utc).isoformat(), 'operation': args.operation,
                   'observation_failure': True, 'error_type': type(error).__name__, 'error': str(error),
                   'retry_authorized': False}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    receipt_path.chmod(0o444)
    print(json.dumps(receipt.get('result', receipt)), flush=True)


if __name__ == '__main__':
    main()
