"""Close the unchanged complete initializer study; no new fits or heldout score."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
AUDIT = PHASE / 'graph_init_closed_family_independent_audit_20261003_v1'
DEST = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE_PHASE = REPO + '/experiments_iclr/postsubmission_20260930'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as h:
        json.dump(value, h, indent=2, allow_nan=False)
        h.write('\n')


def main():
    assert sha(AUDIT / 'MANIFEST.json') == '70c9b452ea4d6f137ab093f581bad5a35bf71b543cad08da471cefce5604526e'
    assert sha(AUDIT / 'SEAL.json') == 'bb52e123244dbd0841fefdaa784fdc77ba8cd1dc818ebf4dcd0f10852408d2c0'
    manifest = json.loads((AUDIT / 'MANIFEST.json').read_text())
    for row in manifest['payload']:
        relative = Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        p = AUDIT / relative
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
    candidate = json.loads((AUDIT / 'ROOT_RELEASE_CANDIDATE_CLOSED.json').read_text())
    assert candidate['metadata_family_closure_complete'] is True
    assert candidate['original_last_process_exit_verified'] is True
    assert candidate['remaining_claims'] == []
    draft = AUDIT / 'COMPARISON_ADMISSION_CLOSED_DRAFT.json'
    assert sha(draft) == candidate['admission_draft']['sha256']
    admission = json.loads(draft.read_text())
    assert set(admission) == set(candidate['admission_schema_fields'])
    assert admission['execution_authorized'] is False and admission['authorized_phase'] == 'compare'
    assert len(admission['expected_contexts']) == 6 and len(admission['fit_freezes']) == 30
    assert len({x['path'] for x in admission['fit_freezes']}) == 30
    assert {(x['graph'], x['seed'], x['source_split_index']) for x in admission['expected_contexts']} == {
        (graph, seed, split) for graph in ('Squirrel', 'Photo') for split, seed in enumerate((17, 29, 43))}
    admission['execution_authorized'] = True
    save(HERE / 'ROOT_COMPARISON_ADMISSION.json', admission)
    release = dict(schema='root-full-initializer-source-comparison-release-v1',
                   UTC=datetime.now(timezone.utc).isoformat(), approved=True,
                   source_candidate_sha256=sha(AUDIT / 'ROOT_RELEASE_CANDIDATE_CLOSED.json'),
                   immutable_audit_manifest_sha256=sha(AUDIT / 'MANIFEST.json'),
                   admission_sha256=sha(HERE / 'ROOT_COMPARISON_ADMISSION.json'),
                   complete_contexts=6, complete_fits=30, complete_phases=72,
                   heldout_or_final_report_authorized=False,
                   checkpoint_inference_replay_established=False,
                   independent_numerical_audit_required_before_quality_claims=True,
                   unchanged_comparator=candidate['comparator'],
                   no_new_fit_retry_restart_or_job_signal=True)
    save(HERE / 'ROOT_RELEASE.json', release)
    encoded = {name: base64.b64encode((HERE / name).read_bytes()).decode()
               for name in ('ROOT_COMPARISON_ADMISSION.json', 'ROOT_RELEASE.json')}
    remote = '''from pathlib import Path
import base64,datetime,hashlib,json,os,subprocess
repo=Path(REPO_VALUE);phase=repo/'experiments_iclr/postsubmission_20260930'
os.chdir(repo)
assert subprocess.run(['git','rev-parse','--show-toplevel'],capture_output=True,text=True,check=True).stdout.strip()==str(repo)
actual=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()
assert actual==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
candidate=CANDIDATE_VALUE
for descriptor in (candidate['comparator'],candidate['attempt_registry']):
 assert hashlib.sha256(Path(descriptor['path']).read_bytes()).hexdigest()==descriptor['sha256']
packet=Path(candidate['comparator']['path']).parents[1]
assert hashlib.sha256((packet/'MANIFEST.json').read_bytes()).hexdigest()==candidate['comparator_packet_manifest_sha256']
assert hashlib.sha256((packet/'SEAL.json').read_bytes()).hexdigest()==candidate['comparator_packet_seal_sha256']
out=Path(candidate['canonical_output'])
assert not out.exists() and not (out.parent/'SOURCE_COMPARISON_CLAIM.json').exists()
root=phase/'graph_init_closed_source_comparison_release_20261003_v1'
root.mkdir(exist_ok=False)
for name,value in PAYLOAD_VALUE.items():
 with (root/name).open('xb') as h:h.write(base64.b64decode(value,validate=True))
env=os.environ.copy();env.update(GNNM_PHASE_ROOT=str(phase),GNNM_SSH_DESTINATION=DEST_VALUE,PYTHONDONTWRITEBYTECODE='1')
argv=[str(repo/'.venv/bin/python'),candidate['comparator']['path'],'compare','--admission',str(root/'ROOT_COMPARISON_ADMISSION.json'),'--output',str(out)]
with (root/'COMPARE_STDOUT.txt').open('x') as stdout,(root/'COMPARE_STDERR.txt').open('x') as stderr:
 result=subprocess.run(argv,cwd=repo,env=env,stdout=stdout,stderr=stderr)
artifacts=[]
for p in sorted(out.glob('*')) if out.is_dir() else []:
 if p.is_file():artifacts.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
receipt=dict(schema='root-unchanged-full-initializer-comparison-terminal-v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=result.returncode,argv=argv,artifacts=artifacts,heldout_authorized=False,ordinary_host_execution=True)
with (root/'TERMINAL.json').open('x') as h:json.dump(receipt,h,indent=2);h.write('\\n')
print(json.dumps(receipt))
raise SystemExit(result.returncode)
'''.replace('REPO_VALUE', repr(REPO)).replace('CANDIDATE_VALUE', repr(candidate)).replace('PAYLOAD_VALUE', repr(encoded)).replace('DEST_VALUE', repr(DEST))
    (HERE / 'REMOTE_EXECUTION_SOURCE.py').write_text(remote)
    argv = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
            '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
            '-o', 'StrictHostKeyChecking=yes', DEST, '/usr/bin/python3 -B -']
    result = subprocess.run(argv, input=remote, capture_output=True, text=True, timeout=1800)
    save(HERE / 'TRANSPORT_TERMINAL.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
         exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
         destination=DEST, execution_source_sha256=sha(HERE / 'REMOTE_EXECUTION_SOURCE.py'),
         no_automatic_retry=True))
    print(json.dumps(dict(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
