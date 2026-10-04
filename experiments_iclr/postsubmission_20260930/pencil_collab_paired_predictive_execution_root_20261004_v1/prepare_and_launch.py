"""Root client for the three fixed PENCIL scientific fits.

Reuse the admitted resource client's exact packet/overlay transport helpers.
No package installation, resource-only fit, checkpoint donor or TEST release.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE = PHASE / 'pencil_collab_paired_predictive_preparation_20261004_v1'
REMOTE_PHASE = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930')
REMOTE = REMOTE_PHASE / HERE.name
SOURCE_PIN = '1a22966029e9a1b4b4affad7e5cc24f6b3d795245a8d3fc8d0ac12af4b2983c2'
SEAL_PIN = 'ca3099d5116e3aa8a06dfe87678dc0dc370a9a2ed27c709ac12bc7c0c95818f8'
PLAN_PIN = '423a1d15f1a2b6caafe572f3e0e8f20eb802165662dc172bf7a08e20a698c7ee'
BASE_PATH = PHASE / 'pencil_collab_resource_qualifier_execution_root_20261004_v3/stage_admit_resource.py'
BASE_PIN = '5725c2af90f5728e87882ed15cf7efc6ff67e4a1c2d0908be385bd9781d85073'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


assert not BASE_PATH.is_symlink() and sha(BASE_PATH) == BASE_PIN
spec = importlib.util.spec_from_file_location('pencil_admitted_root_client_helpers', BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.HERE = HERE
base.SOURCE = SOURCE
base.REMOTE = REMOTE
base.REMOTE_SOURCE = REMOTE_PHASE / SOURCE.name
base.SOURCE_MANIFEST = SOURCE_PIN
base.SOURCE_SEAL = SEAL_PIN
base.PLAN_SHA = PLAN_PIN


def transport(identity, code):
    for row in base.TRANSPORT_PINS:
        base.pin(PHASE / row['path'], row)
    spec = importlib.util.spec_from_file_location('pencil_scientific_root_transport', PHASE / base.TRANSPORT_PINS[0]['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    return module.run('pencil_scientific_' + identity + '_20261004', code)


base.transport = transport
inherited_stage_join_code = base.stage_join_code


def scientific_stage_join_code(binding, packed_sha, chunk_count):
    return inherited_stage_join_code(binding, packed_sha, chunk_count).replace(
        'EXACT_REVIEWED_PENCIL_V3_SOURCE_STAGED', 'EXACT_REVIEWED_PENCIL_SCIENTIFIC_SOURCE_STAGED')


base.stage_join_code = scientific_stage_join_code


def reviewed_plan(review, review_seal, verdict_name):
    base.packet(SOURCE, SEAL_PIN, SOURCE_PIN)
    assert sha(SOURCE / 'PLAN.json') == PLAN_PIN
    base.packet(review, review_seal)
    assert verdict_name == 'REVIEW.json'
    verdict = json.loads((review / verdict_name).read_text())
    seal = json.loads((review / 'SEAL.json').read_text())
    assert seal['candidate_manifest_sha256'] == SOURCE_PIN
    assert verdict['candidate_manifest_sha256'] == SOURCE_PIN
    assert verdict['status'] == 'PASS' and not verdict['blocking_findings']
    assert verdict['execution_authorized'] is False
    row = dict(path=str(REMOTE_PHASE / review.name / verdict_name), bytes=(review / verdict_name).stat().st_size,
               sha256=sha(review / verdict_name))
    binding = dict(source_manifest_sha256=SOURCE_PIN, source_seal_sha256=SEAL_PIN,
                   review_directory=review.name, review_seal_sha256=review_seal,
                   review_manifest_sha256=sha(review / 'MANIFEST.json'), verdict=row,
                   reviewer_context=verdict.get('reviewer_context'), review_is_manuscript_acceptance=False)
    plan = json.loads((SOURCE / 'PLAN.json').read_text())
    assert plan['execution_directory'] == HERE.name and REMOTE.name == HERE.name
    assert plan['seeds'] == [0, 1, 2] and plan['workload']['native_epochs'] == 20
    assert plan['workload']['TEST_reads'] is False and plan['state_donor'] is False
    assert plan['automatic_retry'] is False
    return plan, binding


base.reviewed_plan = reviewed_plan


def stage(review, review_seal):
    import contextlib
    import io
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        base.stage(review, review_seal, 'REVIEW.json')
    assert captured.getvalue().strip() == 'Exact sealed v3 source, review, and 22 bound external source pins staged; no numerical execution.'
    print('Exact sealed scientific source, independent review, and 16 bound external source pins staged; no numerical execution.')
    # The helper stages the immutable source/review/external closure first.
    # The additional root-owned queue files are bounded plain source only.
    rows = []
    import base64
    for name in ('owned_queue.py', 'ROOT_AUTHORIZATION.md'):
        p = HERE / name
        rows.append(dict(path=name, sha256=sha(p), bytes=p.stat().st_size,
                         data=base64.b64encode(p.read_bytes()).decode()))
    code = base.context() + 'rows=' + repr(rows) + '\n'
    code += 'for row in rows:\n p=root/row["path"];b=base64.b64decode(row["data"],validate=True);assert len(b)==row["bytes"] and hashlib.sha256(b).hexdigest()==row["sha256"]\n with p.open("xb") as s:s.write(b)\n'
    code += 'print(json.dumps(dict(status="ROOT_SCIENTIFIC_QUEUE_SOURCE_STAGED",files=len(rows),numerical_execution=False)))\n'
    save('QUEUE_SOURCE_STAGE.json', transport('queue_source', code))


def admit(review, review_seal):
    plan, binding = reviewed_plan(review, review_seal, 'REVIEW.json')
    assert json.loads((HERE / 'SOURCE_STAGE_RECEIPT.json').read_text())['binding'] == binding
    code = base.admission_code(binding).replace('ROOT_ADMITTED_FOR_RESOURCE_ONLY', 'ROOT_ADMITTED_FOR_NATIVE_PENCIL_PREDICTIVE_FITS')
    code = code.replace('FOR_RESOURCE_ONLY', 'FOR_PREDICTIVE_FITS')
    observation = transport('dependencies', code)
    save('DEPENDENCY_ADMISSION_RECEIPT.json', observation)
    release = json.loads((SOURCE / 'ROOT_RELEASE_TEMPLATE.json').read_text())
    release.update(status='APPROVED', source_manifest_sha256=SOURCE_PIN, plan_sha256=PLAN_PIN,
                   root_authorization_reference=str(REMOTE / 'ROOT_AUTHORIZATION.md'),
                   independent_source_review=binding['verdict'], dependency_admission=observation['dependency_admission'])
    save('ROOT_RELEASE.json', release)
    save('ROOT_ADMISSION.json', dict(UTC=datetime.now(timezone.utc).isoformat(), binding=binding,
         status='APPROVED_THREE_FRESH_PENCIL_20_EPOCH_VALID_FITS', seeds=plan['seeds'],
         workload=plan['workload'], caps=plan['caps'], dependency_admission=observation['dependency_admission'],
         TEST_reads=False, historical_Collab_TEST_consumed=True, resource_donor=False, automatic_retry=False))
    import base64
    rows = [dict(name=name, data=base64.b64encode((HERE / name).read_bytes()).decode(), sha256=sha(HERE / name))
            for name in ('ROOT_RELEASE.json', 'ROOT_ADMISSION.json')]
    code = base.authenticated_remote(binding) + 'rows=' + repr(rows) + '\n'
    code += 'for row in rows:\n b=base64.b64decode(row["data"],validate=True);assert hashlib.sha256(b).hexdigest()==row["sha256"]\n with (root/row["name"]).open("xb") as s:s.write(b)\n'
    text = ('import sys;from pathlib import Path;sys.path.insert(0,' + repr(str(base.REMOTE_SOURCE)) + ');'
            'from common import gate;[gate(Path(' + repr(str(REMOTE / 'ROOT_RELEASE.json')) + '),'
            + repr(sha(HERE / 'ROOT_RELEASE.json')) + ',seed) for seed in (0,1,2)];print("PASS_THREE_SCIENTIFIC_STDLIB_GATES")')
    command = [json.loads((SOURCE / 'metadata/RUNTIME_AUTHORITY.json').read_text())['interpreter_path'], '-B', '-c', text]
    proposed = json.loads((SOURCE / 'PROPOSED_COMMAND.json').read_text())
    code += 'r=subprocess.run(' + repr(command) + ',cwd=' + repr(proposed['cwd']) + ',env=dict(os.environ,**' + repr(proposed['environment']) + '),capture_output=True,text=True,timeout=60)\n'
    code += 'assert r.returncode==0,(r.returncode,r.stdout,r.stderr)\nprint(json.dumps(dict(status="THREE_SCIENTIFIC_STDLIB_GATES_PASS",stdout=r.stdout,stderr=r.stderr,numerical_execution=False,TEST_reads=False)))\n'
    save('PRENUMERICAL_GATE.json', transport('gate', code))


def launch(review, review_seal):
    plan, binding = reviewed_plan(review, review_seal, 'REVIEW.json')
    assert (HERE / 'PRENUMERICAL_GATE.json').exists()
    proposed = json.loads((SOURCE / 'PROPOSED_COMMAND.json').read_text())
    release_sha = sha(HERE / 'ROOT_RELEASE.json')
    code = base.authenticated_remote(binding)
    code += 'assert sha(root/"ROOT_RELEASE.json")==' + repr(release_sha) + '\n'
    code += 'assert sha(root/"owned_queue.py")==' + repr(sha(HERE / 'owned_queue.py')) + '\n'
    code += 'assert not any((root/x).exists() for x in ("QUEUE_ATTEMPT_SPENT.json","DETACHED_LAUNCH.json","ACTIVE_FIT.json","seed0","seed1","seed2"))\n'
    code += 'q=subprocess.run(["nvidia-smi","-i",' + repr(plan['GPU_UUID']) + ',"--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=15);g=q.stdout.strip().split(",");assert g[0].strip()==' + repr(plan['GPU_UUID']) + ' and int(g[1])>=74*1024\n'
    code += 'save(root/"QUEUE_ATTEMPT_SPENT.json",dict(UTC=datetime.now(timezone.utc).isoformat(),release_sha256=' + repr(release_sha) + ',automatic_retry=False))\n'
    command = [proposed['commands'][0]['argv'][0], '-B', str(REMOTE / 'owned_queue.py'), '--release-sha256', release_sha]
    code += 'command=' + repr(command) + '\n'
    code += 'with (root/"QUEUE_STDOUT.txt").open("xb") as out,(root/"QUEUE_STDERR.txt").open("xb") as err:\n p=subprocess.Popen(command,cwd=' + repr(proposed['cwd']) + ',env=dict(os.environ,**' + repr(proposed['environment']) + '),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)\n'
    code += 'raw=(Path("/proc")/str(p.pid)/"stat").read_text();f=raw[raw.rfind(")")+2:].split();assert int(f[2])==int(f[3])==p.pid\n'
    code += 'v=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=p.pid,start_ticks=int(f[19]),group=int(f[2]),session=int(f[3]),command=command,source_manifest_sha256=' + repr(SOURCE_PIN) + ',release_sha256=' + repr(release_sha) + ');save(root/"DETACHED_LAUNCH.json",v);print(json.dumps(v))\n'
    save('DETACHED_LAUNCH.json', transport('launch', code))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('stage', 'admit', 'launch'))
    parser.add_argument('--review', required=True)
    parser.add_argument('--review-seal', required=True)
    args = parser.parse_args()
    review = PHASE / args.review
    assert review.resolve().is_relative_to(PHASE)
    {'stage': stage, 'admit': admit, 'launch': launch}[args.operation](review, args.review_seal)
