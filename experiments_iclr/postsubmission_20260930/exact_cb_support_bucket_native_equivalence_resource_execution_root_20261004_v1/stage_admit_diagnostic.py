"""Root client for one reviewed native direct/bucket TRAIN diagnostic; run only when directed."""
from datetime import datetime, timezone
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import zlib

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
REMOTE_PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
REMOTE = REMOTE_PHASE / HERE.name
SOURCE_NAME = 'exact_cb_support_bucket_native_equivalence_resource_preparation_20261004_v1'
REVIEW_NAME = 'exact_cb_support_bucket_native_equivalence_resource_independent_source_review_20261004_v1'
SOURCE_MANIFEST_SHA256 = '8b8adcbb90366098c2482220612b07868d9fe2bb86e7e879fa928e0c0537d98c'
REVIEW_SHA256 = '649a6830ea1adc10240bb12ad6feeae1be50a5ba982c764a75ecd404326eac7c'
REVIEW_MANIFEST_SHA256 = 'e4620e4feec236694561d5ac883a141902ce4800faec65da9d51332165dc2027'
REVIEW_SEAL_SHA256 = 'e4b4a9f0adde7efb794a0ce40149fab24e0337969ac9269a5c405e339200683b'
SOURCE = PHASE / SOURCE_NAME
REVIEW = PHASE / REVIEW_NAME


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def transport(identity, code):
    path = PHASE / 'ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py'
    spec = importlib.util.spec_from_file_location('bucket_native_project_transport', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    return module.run('bucket_native_equivalence_v1_' + identity + '_20261004', code)


def context():
    return ('from pathlib import Path\nimport base64,hashlib,json,os,subprocess\n'
            'from datetime import datetime,timezone\n'
            'repo=Path(' + repr(str(REPO)) + ');phase=Path(' + repr(str(REMOTE_PHASE)) + ');root=Path(' + repr(str(REMOTE)) + ')\n'
            'assert Path.cwd()==repo and os.uname().nodename=="peptide"\n')


def packet(folder):
    paths = []
    for row in json.loads((folder / 'MANIFEST.json').read_text())['files']:
        path = folder / row['path']
        assert path.resolve().is_relative_to(folder) and not path.is_symlink()
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        paths.append(path)
    seal = json.loads((folder / 'SEAL.json').read_text())
    assert seal['manifest_sha256'] == sha(folder / 'MANIFEST.json') and seal['sealed'] is True
    return paths + [folder / 'MANIFEST.json', folder / 'SEAL.json']


def reviewed_plan():
    assert SOURCE == PHASE / SOURCE_NAME and REVIEW == PHASE / REVIEW_NAME
    assert sha(SOURCE / 'MANIFEST.json') == SOURCE_MANIFEST_SHA256
    assert sha(REVIEW / 'REVIEW.json') == REVIEW_SHA256
    assert sha(REVIEW / 'MANIFEST.json') == REVIEW_MANIFEST_SHA256
    assert sha(REVIEW / 'SEAL.json') == REVIEW_SEAL_SHA256
    packet(SOURCE)
    packet(REVIEW)
    review = json.loads((REVIEW / 'REVIEW.json').read_text())
    assert review['status'] == 'PASS' and not review.get('blocking_findings')
    assert review['candidate_manifest_sha256'] == sha(SOURCE / 'MANIFEST.json')
    assert review['execution_authorized'] is False
    plan = json.loads((SOURCE / 'PLAN.json').read_text())
    assert plan['execution_directory'] == HERE.name and plan['repository'] == str(REPO)
    assert plan['workload'] == dict(full_mask_batches=1, gradient_evaluations=4, heldout_or_scores=False,
                                    native_forward_graphs=1, negative_queries=65536, optimizer_updates=0,
                                    positive_queries=65536, warm_fit_epochs=0)
    assert plan['batch_size'] == 65536 and plan['bank_size'] == 4
    assert plan['fixed_gradient_agreement_rule'] == dict(atol=128 * 2**-23, rtol=128 * 2**-23)
    assert plan['GPU_UUID'] == 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'
    return plan


def stage():
    plan = reviewed_plan()
    files = packet(SOURCE) + packet(REVIEW)
    for row in plan['source_pins']:
        path = PHASE / row['path']
        assert path.resolve().is_relative_to(PHASE) and not path.is_symlink()
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        files.append(path)
    rows = []
    for path in dict.fromkeys(files):
        raw = path.read_bytes()
        assert len(raw) < 2_000_000 and path.suffix in ('.py', '.json', '.md', '.diff', '.patch')
        rows.append(dict(path=str(path.relative_to(PHASE)), bytes=len(raw), sha256=sha(path), data=base64.b64encode(raw).decode()))
    save('SOURCE_STAGE_INVENTORY.json', [{k:v for k,v in row.items() if k != 'data'} for row in rows])
    packed = zlib.compress(json.dumps(rows).encode(), 9)
    encoded = base64.b64encode(packed).decode()
    chunks = [encoded[n:n+40000] for n in range(0, len(encoded), 40000)]
    for number, chunk in enumerate(chunks, 1):
        code = context() + 'staging=root/"source_chunks";staging.mkdir(parents=True,exist_ok=True)\nraw=' + repr(chunk) + '.encode()\n'
        code += 'with (staging/' + repr('part%03d.b64' % number) + ').open("xb") as s:s.write(raw)\n'
        code += 'print(json.dumps(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())))\n'
        value = transport('chunk%03d' % number, code)
        assert value['sha256'] == hashlib.sha256(chunk.encode()).hexdigest()
    code = context() + 'import zlib\npacked=base64.b64decode(b"".join((root/"source_chunks"/("part%03d.b64"%n)).read_bytes() for n in range(1,' + str(len(chunks)+1) + ')),validate=True)\n'
    code += 'assert hashlib.sha256(packed).hexdigest()==' + repr(hashlib.sha256(packed).hexdigest()) + '\nrows=json.loads(zlib.decompress(packed))\n'
    code += 'for row in rows:\n p=phase/row["path"];assert p.resolve().is_relative_to(phase);raw=base64.b64decode(row["data"],validate=True);assert len(raw)==row["bytes"] and hashlib.sha256(raw).hexdigest()==row["sha256"]\n if p.exists():assert p.is_file() and not p.is_symlink() and p.read_bytes()==raw\n else:\n  p.parent.mkdir(parents=True,exist_ok=True)\n  with p.open("xb") as s:s.write(raw)\n'
    code += 'print(json.dumps(dict(status="EXACT_REVIEWED_NATIVE_BUCKET_DIAGNOSTIC_SOURCE_STAGED",files=len(rows),bytes=sum(r["bytes"] for r in rows),numerical_execution=False)))\n'
    save('SOURCE_STAGE_RECEIPT.json', transport('join', code))
    print('Reviewed diagnostic source and prerequisite metadata staged; no numerical execution.')


def admit():
    plan = reviewed_plan()
    assert (HERE / 'SOURCE_STAGE_RECEIPT.json').exists()
    runtime = json.loads((PHASE / plan['runtime_authority']).read_text())
    environment = dict(CUDA_VISIBLE_DEVICES=plan['GPU_UUID'], OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       PYTHONPATH=runtime['project_PYTHONPATH'], GNNM_SSH_DESTINATION='shmelev@192.168.18.77')
    code = context() + 'prerequisites=' + repr(dict(plan['CPU_prerequisites'], **plan['BUCKET_CPU_prerequisites'])) + '\n'
    code += 'for key,row in prerequisites.items():\n p=Path(row["path"]);assert p.resolve().is_relative_to(phase) and p.is_file() and not p.is_symlink() and p.stat().st_size==row["bytes"];assert hashlib.sha256(p.read_bytes()).hexdigest()==row["sha256"]\n'
    code += 'interpreter=Path(' + repr(runtime['interpreter_path']) + ');assert hashlib.sha256(interpreter.read_bytes()).hexdigest()==' + repr(runtime['interpreter_sha256']) + '\n'
    code += 'g=subprocess.run(["nvidia-smi","--query-gpu=uuid,name,memory.total,memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=15)\nrows=[dict(zip(("uuid","name","total_MiB","free_MiB"),[x.strip() for x in line.split(",")])) for line in g.stdout.splitlines()];assert {r["uuid"] for r in rows}=={"GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998","GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced"}\n'
    code += 'selected=next(r for r in rows if r["uuid"]==' + repr(plan['GPU_UUID']) + ');assert int(selected["free_MiB"])>=24576,"Insufficient current free memory for native diagnostic admission"\n'
    code += 'print(json.dumps(dict(status="EXACT_NATIVE_RUNTIME_PREREQUISITES_AND_CURRENT_GPU_OBSERVED",UTC=datetime.now(timezone.utc).isoformat(),selected_GPU=selected,interpreter_sha256=' + repr(runtime['interpreter_sha256']) + ',numerical_execution=False,unrelated_jobs_modified=False)))\n'
    resource = transport('resources', code)
    save('RESOURCE_AND_PREREQUISITE_RECEIPT.json', resource)
    release = json.loads((SOURCE / 'ROOT_RELEASE_TEMPLATE.json').read_text())
    review = REVIEW / 'REVIEW.json'
    release.update(status='APPROVED', authorized_stage='one_native_TRAIN_bucket_equivalence_resource_diagnostic',
                   source_manifest_sha256=sha(SOURCE / 'MANIFEST.json'), plan_sha256=sha(SOURCE / 'PLAN.json'),
                   root_authorization_reference=str(REMOTE / 'ROOT_ADMISSION.json'),
                   independent_source_review=dict(path=str(REMOTE_PHASE / REVIEW.name / 'REVIEW.json'),
                                                  bytes=review.stat().st_size, sha256=sha(review)))
    save('ROOT_RELEASE.json', release)
    save('ROOT_ADMISSION.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
         status='APPROVED_ONE_NATIVE_TRAIN_BUCKET_SAME_STATE_EQUIVALENCE_RESOURCE_DIAGNOSTIC',
         source_manifest_sha256=sha(SOURCE / 'MANIFEST.json'), workload=plan['workload'],
         candidate_manifest_sha256=plan['candidate_manifest_sha256'], core_manifest_sha256=plan['core_manifest_sha256'],
         independent_review_sha256=REVIEW_SHA256,
         equivalence_verdict_requires_original_fixed_rule=True, physical_completion_is_equivalence_PASS=False,
         same_batch_not_favorable_selection=True, optimizer_updates=0, scientific_fits=0,
         VALID_TEST_access=False, automatic_retry=False,
         limitations=['One initialization and fixed TRAIN batch cannot establish predictive transfer.',
                     'Four reverse evaluations and coexisting auxiliary graphs do not qualify an ordinary fit/VALID schedule.',
                     'Observed free GPU memory and sampled RSS are not resource reservations.']))
    payload = []
    for name in ('ROOT_RELEASE.json', 'ROOT_ADMISSION.json'):
        f = HERE / name
        payload.append(dict(path=str(REMOTE / name), sha256=sha(f), data=base64.b64encode(f.read_bytes()).decode()))
    code = context() + 'rows=' + repr(payload) + '\nfor row in rows:\n p=Path(row["path"]);assert p.resolve().is_relative_to(root);raw=base64.b64decode(row["data"],validate=True);assert hashlib.sha256(raw).hexdigest()==row["sha256"]\n with p.open("xb") as s:s.write(raw)\nprint(json.dumps(dict(status="EXACT_NATIVE_BUCKET_DIAGNOSTIC_RELEASE_STAGED",numerical_execution=False)))\n'
    save('RELEASE_STAGE_RECEIPT.json', transport('release', code))
    remote_source = REMOTE_PHASE / SOURCE.name
    gate_code = 'import sys;from pathlib import Path;sys.path.insert(0,' + repr(str(remote_source)) + ');from common import gate;gate(Path(' + repr(str(REMOTE / 'ROOT_RELEASE.json')) + '),' + repr(sha(HERE / 'ROOT_RELEASE.json')) + ');print("PASS_NATIVE_BUCKET_DIAGNOSTIC_STDLIB_GATE")'
    gate_command = [runtime['interpreter_path'], '-B', '-c', gate_code]
    code = context() + 'r=subprocess.run(' + repr(gate_command) + ',cwd=repo,env=dict(os.environ,**' + repr(environment) + '),capture_output=True,text=True,timeout=60)\nprint(json.dumps(dict(status="PASS" if r.returncode==0 else "FAILED",exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,numerical_execution=False)))\n'
    gate = transport('gate', code)
    save('PRENUMERICAL_GATE.json', gate)
    assert gate['status'] == 'PASS', gate
    command = [runtime['interpreter_path'], '-B', str(remote_source / 'supervise.py'),
               '--release', str(REMOTE / 'ROOT_RELEASE.json'), '--release-sha256', sha(HERE / 'ROOT_RELEASE.json')]
    code = context() + 'command=' + repr(command) + '\nassert not (root/"run01").exists() and not (root/"supervision/run01").exists() and not (root/"DIAGNOSTIC_ATTEMPT_SPENT.json").exists()\n'
    code += 'with (root/"DETACHED_STDOUT.txt").open("xb") as out,(root/"DETACHED_STDERR.txt").open("xb") as err:\n child=subprocess.Popen(command,cwd=repo,env=dict(os.environ,**' + repr(environment) + '),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)\n'
    code += 'raw=(Path("/proc")/str(child.pid)/"stat").read_text();f=raw[raw.rfind(")")+2:].split();v=dict(UTC=datetime.now(timezone.utc).isoformat(),status="OWNED_NATIVE_TRAIN_BUCKET_EQUIVALENCE_DIAGNOSTIC_LAUNCHED",supervisor_PID=child.pid,supervisor_start_ticks=int(f[19]),command=command,environment=' + repr(environment) + ',optimizer_updates=0,scientific_fits=0,VALID_TEST_access=False)\n'
    code += 'with (root/"DETACHED_LAUNCH.json").open("x") as s:json.dump(v,s,indent=2);s.write("\\n")\nprint(json.dumps(v))\n'
    launch = transport('launch', code)
    save('DETACHED_LAUNCH.json', launch)
    print(json.dumps({k:launch[k] for k in ('UTC', 'status', 'supervisor_PID', 'supervisor_start_ticks')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('stage', 'admit', 'run'))
    parser.add_argument('--source', default=SOURCE_NAME)
    parser.add_argument('--review', default=REVIEW_NAME)
    args = parser.parse_args()
    SOURCE = PHASE / args.source
    REVIEW = PHASE / args.review
    assert all(f.resolve().is_relative_to(PHASE) and not f.is_symlink() for f in (SOURCE, REVIEW))
    if args.mode in ('stage', 'run'):
        stage()
    if args.mode in ('admit', 'run'):
        admit()
