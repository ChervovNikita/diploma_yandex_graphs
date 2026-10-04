"""Bind actual V5 qualifications and prepare two separately dispatched fixed fits."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import base64
import hashlib
import json
import lzma
import shlex

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
REMOTE_REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
REMOTE_PHASE = REMOTE_REPO / 'experiments_iclr/postsubmission_20260930'
REMOTE_ROOT = REMOTE_PHASE / ROOT.name
SOURCE = PHASE / 'graph_ncNC_structural_pattern_fit_normal_supervision_preparation_20261004_v1'
SOURCE_SHA = '14f85415638ad6e879c9269e2503ca4c5d14f7cae43e20195403af10ad00411a'
DRIVER_SHA = '9fc539b8f92d7d4e883224c3b0aa85ae583b64c70f2a701a4a648fd818aa32a1'
QUALIFICATIONS = {
    'numerical': ('graph_ncNC_structural_pattern_numerical_execution_root_20261004_v5',
                  '3763623932f76739960b57452b0fd8ea784a465d91b946843a23ba7b62c20212',
                  '483327fe60e73f5a6baa5acbcf9f3ca34dfaed2bf1c79804a519fae707b22bba'),
    'full_graph': ('graph_ncNC_structural_pattern_full_graph_execution_root_20261004_v2',
                   '2b9c0c18f81a1717d1af1d4e2357d32937db1b3abd83ca94bf6db183e206ef50',
                   '63a2403a5b972c109a13e0cb6aec9b656ed9df4b49608ca1b448bab1efd1c60e'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


assert sha(SOURCE / 'MANIFEST.json') == SOURCE_SHA
manifest = json.loads((SOURCE / 'MANIFEST.json').read_text())
for row in manifest['files']:
    path = SOURCE / row['path']
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    if path.suffix == '.py':
        ast.parse(path.read_text())
qualified = {}
for stage, (name, qualification_sha, terminal_sha) in QUALIFICATIONS.items():
    fetched = PHASE / name / 'remote_receipts_observation01'
    qpath = fetched / (stage + '/run01/QUALIFICATION.json')
    tpath = fetched / 'supervision/run01/SUPERVISOR_TERMINAL.json'
    assert sha(qpath) == qualification_sha and sha(tpath) == terminal_sha
    q = json.loads(qpath.read_text())
    terminal = json.loads(tpath.read_text())
    assert q['status'] == 'PASS' and q['stage'] == stage
    assert q['identity']['driver_manifest_sha256'] == DRIVER_SHA
    assert q['state_donor'] is False and q['test_file_opened'] is False
    assert q['activation_checkpoint_parity']['status'] == 'PASS'
    transition = q['runtime_profile_transition']
    assert transition['RNG_exactly_unchanged'] is True
    assert transition['actual_profile_matches_exact_transition'] is True
    assert transition['actual_profile_before']['deterministic_algorithms'] is False
    assert transition['actual_profile_after']['deterministic_algorithms'] is True
    assert q['runtime_profile_final']['actual_runtime_profile'] == transition['actual_profile_after']
    assert q['runtime_profile_final']['runtime_profile_exact'] is True
    assert terminal['status'] == 'COMPLETE' and terminal['child_exit_code'] == 0
    assert terminal['child_signal'] is None and terminal['cap_violation'] is None
    assert terminal['qualification_receipt']['sha256'] == qualification_sha
    assert terminal['other_jobs_signaled'] is False
    qualified[stage] = q
assert qualified['numerical']['identity'] == qualified['full_graph']['identity']
full = qualified['full_graph']
assert full['full_TRAIN_epochs'] == 2 and full['full_VALID_evaluations'] == 2
assert full['full_graph_numerical_and_serialized_replay'] is True
assert full['project_metric_computed'] is False
assert {r['arm'] for r in full['records']} == {'J', 'F'}
for row in full['records']:
    assert row['TRAIN']['full_batches'] == row['TRAIN']['optimizer_steps'] == 17
    assert row['VALID']['positive_queries'] == 60084 and row['VALID']['negative_queries'] == 100000
    assert row['VALID']['route_count'] == 5
review = dict(schema='root-fixed-J-F-fit-source-and-prerequisite-review-v1',
              UTC=datetime.now(timezone.utc).isoformat(), status='SOURCE_AND_PREREQUISITES_PASS',
              fit_source_manifest_sha256=SOURCE_SHA, driver_manifest_sha256=DRIVER_SHA,
              root_read_complete_supervisor_and_child=True, sealed_payload_count=len(manifest['files']),
              actual_qualification_bindings=QUALIFICATIONS,
              observed_full_TRAIN_epoch_seconds={r['arm']: r['TRAIN']['wall_seconds'] for r in full['records']},
              proposed_wall_cap_adopted_seconds=86400,
              wall_cap_reason='Observed full native TRAIN epochs are about 62 seconds. The 24-hour ceiling leaves ample headroom for fixed 100-epoch fits, selectors, serialization and ordinary shared-host variability.',
              science_budget_unchanged=True, fresh_seed0_pair_only=True,
              engineering_PASS_is_predictive_gain=False, manuscript_acceptance_established=False)
save(ROOT / 'ROOT_FIXED_FIT_SOURCE_REVIEW.json', review)
save(ROOT / 'ROOT_FIXED_PAIR_ADMISSION.json', dict(
    schema='root-fixed-J-F-predictive-pair-admission-v1', UTC=review['UTC'],
    authorizing_agent='/root', authorized_fits=['J_seed0', 'F_seed0'],
    epochs_each=100, optimizer_updates_each=1700, validation_selectors_each=100,
    total_complete_VALID_traversals_each=102, lambda_value=1,
    first_quality_disclosure_after_both_fits_and_frozen_pair_closure=True,
    J_GPU='GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998',
    F_GPU='GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced',
    J_dispatch_waits_until_exact_owned_fixed_diagnostic_terminal=True,
    TEST_authorized=False, automatic_retry_or_resume=False,
    original_paper_scores_and_failed_attempts_preserved=True))

for arm in ('J', 'F'):
    release = json.loads((SOURCE / ('ROOT_FIT_' + arm + '_RELEASE_EXAMPLE.json')).read_text())
    invocation = release.pop('example_invocations_NOT_AUTHORIZED')[0]
    release.pop('example_supervisor_argv_NOT_AUTHORIZED')
    release.update(example_only_NOT_AUTHORIZATION=False, execution_enabled=True,
                   authorized_stages=['fit'], authorized_invocations=[invocation],
                   root_authorization_reference=str(REMOTE_ROOT / 'ROOT_FIXED_PAIR_ADMISSION.json'),
                   fit_supervision_manifest_sha256=SOURCE_SHA, TEST_authorized=False,
                   fit_receipts={}, diagnostics_receipt={})
    release['fit_wall_ceiling_proposal']['status'] = 'ADOPTED_BY_ROOT_FROM_ACTUAL_FULLGRAPH_TIMING'
    for stage, (name, qualification_sha, terminal_sha) in QUALIFICATIONS.items():
        release['qualification'][stage]['sha256'] = qualification_sha
        release[stage + '_supervision_terminal']['sha256'] = terminal_sha
    release_path = ROOT / ('ROOT_FIT_' + arm + '_RELEASE.json')
    save(release_path, release)
    paths = [SOURCE / r['path'] for r in manifest['files']] + [SOURCE / 'MANIFEST.json', SOURCE / 'SEAL.json',
             ROOT / 'ROOT_FIXED_FIT_SOURCE_REVIEW.json', ROOT / 'ROOT_FIXED_PAIR_ADMISSION.json', release_path]
    rows = [dict(path=str(p.relative_to(PHASE)), bytes=p.stat().st_size, sha256=sha(p)) for p in paths]
    metadata = json.dumps(rows).encode()
    envelope = len(metadata).to_bytes(8, 'little') + metadata + b''.join(p.read_bytes() for p in paths)
    packed = base64.b64encode(lzma.compress(envelope, preset=9)).decode()
    code = '''from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,lzma,os,socket,subprocess,time
repo=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='peptide'
assert subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip()==str(repo)
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='9c7ed8a6192405c64743206f80c13ad0f2a3dcd7'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify_packet(folder,wanted):
 assert sha(folder/'MANIFEST.json')==wanted
 for row in json.loads((folder/'MANIFEST.json').read_text())['files']:
  p=(folder/row['path']).resolve();assert p.is_relative_to(folder.resolve()) and p.stat().st_size==row.get('bytes',row.get('size')) and sha(p)==row['sha256']
def identity(pid):
 p=Path('/proc')/str(pid);raw=(p/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
 return dict(PID=pid,start_time_ticks=int(v[19]),session=int(v[3]),process_group=int(v[2]),argv=(p/'cmdline').read_bytes().decode().split('\\0')[:-1],cwd=str((p/'cwd').resolve()),exe=str((p/'exe').resolve()))
'''
    code += 'root=phase/' + repr(ROOT.name) + '\nsource=phase/' + repr(SOURCE.name) + '\narm=' + repr(arm) + '\npacked=' + repr(packed) + '\n'
    code += '''assert not (root/('DISPATCH_'+arm+'.json')).exists() and not (root/('DETACHED_'+arm+'.log')).exists()
assert not (root/('fit_'+arm+'/run01')).exists() and not (root/('supervision/fit_'+arm+'/run01')).exists()
data=lzma.decompress(base64.b64decode(packed,validate=True));assert len(data)<2*1024**2
length=int.from_bytes(data[:8],'little');assert length<1024**2
rows=json.loads(data[8:8+length]);offset=8+length
assert len(data)==offset+sum(r['bytes'] for r in rows)
blobs={}
for row in rows:
 p=(phase/row['path']).resolve();assert p.is_relative_to(source) or p.is_relative_to(root)
 raw=data[offset:offset+row['bytes']];offset+=row['bytes']
 assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if p.exists():assert p.is_file() and not p.is_symlink() and p.read_bytes()==raw
 blobs[p]=raw
release_path=root/('ROOT_FIT_'+arm+'_RELEASE.json');release=json.loads(blobs[release_path])
python=Path('/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12')
assert sha(python)=='14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2'
GPU=release['cuda_visible_devices']
if arm=='J':
 diagnostic=phase/'ncnc_fixed_fabricated_f4_repeatability_diagnostic_execution_root_20261004_v1'
 terminal=json.loads((diagnostic/'PHYSICAL_TERMINAL.json').read_text())
 for handle in (terminal['runner_identity'],terminal['child_identity']):
  assert not (Path('/proc')/str(handle['pid'])).exists(),'Owned diagnostic handle still present; do not overlap J fit'
for stage in ('numerical','full_graph'):
 pin=release['qualification'][stage];assert sha(Path(pin['path']))==pin['sha256']
 pin=release[stage+'_supervision_terminal'];assert sha(Path(pin['path']))==pin['sha256']
driver=Path(release['fit_driver_path']).parent;verify_packet(driver,release['driver_manifest_sha256'])
q=subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(',')
assert q[0].strip()==GPU and int(q[1])>=77824,'Available physical GPU memory below conservative dispatch headroom'
apps=subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=15).strip()
assert not apps,'Selected GPU occupied; no other job is changed'
available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
assert available>=64*1024**3
for p,raw in blobs.items():
 if not p.exists():
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as handle:handle.write(raw)
verify_packet(source,release['fit_supervision_manifest_sha256'])
argv=[str(python),'-B',str(source/'supervise_fit.py'),'--arm',arm,'--root-release',str(release_path),'--output',str(root/('supervision/fit_'+arm+'/run01'))]
env=os.environ.copy();env.update(GNNM_SSH_DESTINATION='shmelev@192.168.18.77',PYTHONDONTWRITEBYTECODE='1')
with (root/('DETACHED_'+arm+'.log')).open('xb') as log:
 process=subprocess.Popen(['nohup',*argv],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(.3)
owned=identity(process.pid) if (Path('/proc')/str(process.pid)).exists() else None
record=dict(schema='root-one-fixed-V5-fit-dispatch-v1',UTC=datetime.now(timezone.utc).isoformat(),arm=arm,seed=0,physical_identity=owned,requested_argv=argv,GPU_UUID=GPU,free_MiB_before_dispatch=int(q[1]),host_MemAvailable_bytes=available,release_sha256=sha(release_path),staged_files=len(rows),ordinary_host_execution=True,automatic_retry_or_resume=False,other_jobs_signaled=False,TEST_authorized=False)
with (root/('DISPATCH_'+arm+'.json')).open('x') as handle:json.dump(record,handle,indent=2);handle.write('\\n')
print(json.dumps(record,indent=2))
assert owned is not None,'Supervisor absent after dispatch. Preserve failed attempt, no retry.'
assert owned['argv']==argv and owned['cwd']==str(repo) and owned['session']==process.pid and owned['process_group']==process.pid
'''
    ast.parse(code)
    command = 'cd ' + shlex.quote(str(REMOTE_REPO)) + '\n' + shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', code]) + '\n'
    assert len(command.encode()) < 100000
    target = PHASE / 'gpu77_connection_recovery_v1' / ('ncnc_V5_fixed_fit_' + arm + '_launch_20261004_v1_command.txt')
    with target.open('x') as handle:
        handle.write(command)
    save(ROOT / ('ROOT_' + arm + '_COMMAND_PREPARATION.json'), dict(
        command_path=str(target.relative_to(PHASE)), command_bytes=len(command.encode()),
        command_sha256=sha(target), payloads=rows, no_remote_execution=True))
    print(json.dumps(dict(arm=arm, command_path=str(target), command_bytes=len(command.encode()), payloads=len(rows))))
