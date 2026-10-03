"""Root-reviewed exact-source diagnostic dispatch preparation, stdlib only."""
import ast
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent
REMOTE = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930')
NORMAL = 'graph_ncNC_structural_pattern_parity_diagnostic_normal_supervision_preparation_20261004_v1'
CORE = 'graph_ncNC_structural_pattern_parity_diagnostic_source_preparation_20261004_v1'
REVIEW = 'graph_ncNC_structural_pattern_parity_diagnostic_independent_source_review_20261004_v1'
OUT = 'graph_ncNC_structural_pattern_parity_diagnostic_execution_root_20261004_v1'
EXPECTED = {
    NORMAL: 'c306301e1cec9801151f0c7789a4c0b38a3668f2d60078f437c0643c6e14d8f6',
    CORE: 'b665affd187593875a2be81e30dbe7a3c8e3ce231f7e344d791d67a800331ebe',
    REVIEW: 'c09384131c4f378337591ba3f29655911598ce32a887160a58e180b818402bd2',
}

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def write_exclusive(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as handle:
        handle.write(raw)
    return raw

def main():
    plan = json.loads((LOCAL / NORMAL / 'PLAN.json').read_text())
    payload = {}
    verified = []
    all_packets = dict(EXPECTED)
    all_packets.update({row['root']: row['manifest_sha256'] for row in plan['source_packets']})
    for name, pin in all_packets.items():
        root = LOCAL / name
        raw = (root / 'MANIFEST.json').read_bytes()
        assert sha(raw) == pin, name
        manifest = json.loads(raw)
        for row in manifest['files']:
            path = (root / row['path']).resolve()
            assert path.is_relative_to(root.resolve())
            content = path.read_bytes()
            assert len(content) == row.get('bytes', row.get('size')) and sha(content) == row['sha256'], str(path)
            verified.append({'path': str(path.relative_to(LOCAL)), 'bytes': len(content), 'sha256': sha(content)})
            if name in EXPECTED:
                payload[str(path.relative_to(LOCAL))] = base64.b64encode(content).decode()
        if name in EXPECTED:
            payload[name + '/MANIFEST.json'] = base64.b64encode(raw).decode()
            seal = root / 'SEAL.json'
            if seal.exists():
                payload[name + '/SEAL.json'] = base64.b64encode(seal.read_bytes()).decode()
    for row in plan['authority_files']:
        raw = (LOCAL / row['path']).read_bytes()
        assert sha(raw) == row['sha256'] and len(raw) == row['bytes']
        verified.append(row)
    review = json.loads((LOCAL / REVIEW / 'REVIEW.json').read_text())
    assert review['verdict'] == 'PASS' and review['candidate_manifest_sha256'] == EXPECTED[CORE]
    out = LOCAL / OUT
    out.mkdir(exist_ok=False)
    root_review = {
        'schema': 'root-parity-diagnostic-source-review-v1',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'verdict': 'APPROVED_FOR_ONE_DIAGNOSTIC_ONLY',
        'source_manifest_pins': EXPECTED,
        'root_read_scope': ['complete diagnostic_run.py', 'complete diagnostic_child.py', 'complete supervise_diagnostic.py', 'complete parity_diagnostic.py', 'both source PLAN.json files'],
        'checks': {
            'exact_plain_scalar_writer_no_model_hooks_or_frame_inspection': True,
            'all_29_callbacks_including_final_check_RNG_and_runtime': True,
            'one_common_fresh_state_12_native_steps_48_member_steps': True,
            'complete_real_TRAIN_graph_same_fixed_probe': True,
            'V3_repeat_control_and_V3_V4_discrepancies_keep_original_threshold': True,
            'no_VALID_or_TEST_scoring_or_fitted_donor': True,
            'ordinary_host_only_owned_child_session_caps': True,
            'fresh_GPU1_and_host_headroom_gate': True,
            'GPU0_other_jobs_not_signaled': True,
        },
        'verified_local_payloads': verified,
        'qualification_PASS_or_scientific_fit_admitted': False,
        'prior_failures_preserved': True,
    }
    review_raw = write_exclusive(out / 'ROOT_SOURCE_REVIEW.json', root_review)
    admission = plan['required_admission_fields']
    admission['normal_supervision_manifest_sha256'] = EXPECTED[NORMAL]
    admission['root_authorization_reference'] = 'User-authorized project research. Exact root source review ' + str(REMOTE / OUT / 'ROOT_SOURCE_REVIEW.json') + ' sha256=' + sha(review_raw)
    admission['UTC'] = datetime.now(timezone.utc).isoformat()
    admission_raw = write_exclusive(out / 'ROOT_DIAGNOSTIC_ADMISSION.json', admission)
    for name, raw in [('ROOT_SOURCE_REVIEW.json', review_raw), ('ROOT_DIAGNOSTIC_ADMISSION.json', admission_raw)]:
        payload[OUT + '/' + name] = base64.b64encode(raw).decode()
    inventory = [{'path': name, 'bytes': len(base64.b64decode(data)), 'sha256': sha(base64.b64decode(data))} for name, data in sorted(payload.items())]
    packed = gzip.compress(json.dumps(payload, sort_keys=True).encode(), mtime=0)
    encoded = base64.b64encode(packed).decode()
    remote_script = '''import base64,gzip,hashlib,json,os,subprocess,sys
from pathlib import Path
repo=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert repo.is_dir() and phase.resolve().is_relative_to(repo.resolve())
os.chdir(repo)
data=json.loads(gzip.decompress(base64.b64decode(PAYLOAD)))
assert len(data)<100 and sum(len(v) for v in data.values())<2000000
for name,encoded in sorted(data.items()):
 p=(phase/name).resolve()
 assert p.is_relative_to(phase.resolve()) and '..' not in Path(name).parts
 raw=base64.b64decode(encoded,validate=True)
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  assert p.is_file() and p.read_bytes()==raw, 'Existing source differs: '+name
 else:
  with p.open('xb') as f:f.write(raw)
execution=phase/OUT
admission=execution/'ROOT_DIAGNOSTIC_ADMISSION.json'
assert hashlib.sha256(admission.read_bytes()).hexdigest()==ADMISSION_SHA
assert not (execution/'diagnostic/run01').exists() and not (execution/'supervision/run01').exists()
python=Path('/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12')
assert hashlib.sha256(python.read_bytes()).hexdigest()=='14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2'
env=os.environ.copy()
env.update(GNNM_SSH_DESTINATION='shmelev@192.168.18.77',PYTHONDONTWRITEBYTECODE='1')
argv=[str(python),'-B',str(phase/NORMAL/'supervise_diagnostic.py'),'--root-admission',str(admission),'--admission-sha256',ADMISSION_SHA,'--output',str(execution/'supervision/run01')]
with (execution/'SUPERVISOR.stdout.log').open('x') as stdout,(execution/'SUPERVISOR.stderr.log').open('x') as stderr:
 child=subprocess.Popen(argv,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
 stat=(Path('/proc')/str(child.pid)/'stat').read_text()
 fields=stat[stat.rfind(')')+2:].split()
 record={'status':'LAUNCHED_ONCE','PID':child.pid,'start_time_ticks':int(fields[19]),'argv':argv,'admission_sha256':ADMISSION_SHA,'qualified_python_sha256':'14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2','scientific_fit_admitted':False,'other_jobs_signaled':False}
 with (execution/'ROOT_LAUNCH.json').open('x') as f:json.dump(record,f,indent=2)
 print(json.dumps(record))
'''.replace('ADMISSION_SHA', repr(sha(admission_raw))).replace('NORMAL', repr(NORMAL)).replace('OUT', repr(OUT)).replace('PAYLOAD', repr(encoded))
    ast.parse(remote_script)
    command = '/usr/bin/python3 -c ' + shlex.quote(remote_script)
    command_path = HERE / 'diagnostic_stage_launch_20261004_v1.txt'
    with command_path.open('x') as f:
        f.write(command)
    write_exclusive(out / 'ROOT_DISPATCH_REVIEW.json', {
        'schema': 'root-reviewed-diagnostic-dispatch-v1',
        'command_path': str(command_path), 'command_sha256': sha(command.encode()),
        'compressed_payload_bytes': len(packed), 'payload_inventory': inventory,
        'admission_sha256': sha(admission_raw),
        'source_changes': False, 'automatic_retry': False,
        'remote_targets_inside_authorized_repo': True,
        'launches_only_fixed_diagnostic_supervisor_once': True,
        'no_credentials_in_payload': True,
    })
    print(json.dumps({'command': str(command_path), 'command_bytes': len(command.encode()), 'admission_sha256': sha(admission_raw), 'payload_files': len(payload)}))

if __name__ == '__main__':
    main()
