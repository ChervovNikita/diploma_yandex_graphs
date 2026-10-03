"""Build an exact source custody command; does not run project code."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
TRANSPORT = PHASE / 'gpu77_connection_recovery_v1'


def pin(relative):
    content = (PHASE / relative).read_bytes()
    return {'path': relative, 'bytes': len(content),
            'sha256': hashlib.sha256(content).hexdigest()}


packets = []
for name in [
    'graph_ncNC_structural_pattern_pilot_preparation_20261003_v4',
    'graph_ncNC_structural_pattern_pilot_preparation_20261003_v3',
    'graph_ncNC_collab_predictive_pilot_design_20261003_v1',
    'graph_ncNC_member_completion_qualification_preparation_20261003_v2',
    'graph_ncNC_collab_resource_epoch_preparation_20261003_v4',
    'graph_ncNC_collab_predictive_driver_preparation_20261003_v1',
    'ncnc_structural_ambiguity_diversity_gap_assessment_20261003_v1',
    'graph_ncNC_structural_pattern_normal_supervision_preparation_20261003_v2',
    'graph_ncNC_structural_pattern_normal_supervision_preparation_20261003_v3',
    'graph_ncNC_structural_pattern_v4_numerical_execution_metadata_preparation_20261003_v1',
    'graph_ncNC_structural_pattern_v4_independent_source_review_20261003_v1',
]:
    row = {'path': name, 'manifest': pin(name + '/MANIFEST.json')}
    if (PHASE / name / 'SEAL.json').is_file():
        row['seal'] = pin(name + '/SEAL.json')
    packets.append(row)

metadata = PHASE / 'graph_ncNC_structural_pattern_v4_numerical_execution_metadata_preparation_20261003_v1'
review = PHASE / 'graph_ncNC_structural_pattern_v4_independent_source_review_20261003_v1'
extras = json.loads((ROOT / 'SOURCE_STAGE_FILE_PINS.json').read_text())['files'][:]
extras += json.loads((metadata / 'SOURCE_REVIEW_INTERPRETER_BINDINGS.json').read_text())['inputs']
extras += list(json.loads((review / 'INPUT_BINDINGS.json').read_text())['inputs'].values())
extras.append(pin('graph_ncNC_valid_data_authority_root_20261003_v1/DATA_AUTHORITY.json'))
history = json.loads((metadata / 'PRIOR_ATTEMPTS_AND_COST_CUSTODY.json').read_text())
originals = []
for attempt in history['attempts']:
    if attempt['kind'] == 'numerical':
        rows = [attempt['terminal'], attempt['original_release']]
        if 'historical_qualification' in attempt:
            rows.append(attempt['historical_qualification'])
    else:
        rows = [row['original'] for row in attempt['evidence']]
    for row in rows:
        originals.append({**row, 'path': row['path'].replace('/remote_receipts_run01/', '/')
                          .replace('/remote_receipts_observation01/', '/'),
                          'recorded_local_original_path': row['path']})

code = '''from pathlib import Path
import json,hashlib,datetime,subprocess,os
repo=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git').resolve()
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'graph_ncNC_structural_pattern_numerical_execution_root_20261003_v4'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='db0302db9991bbb5e167a94948ff6fe5f4199551'
def verify(x):
    p=(phase/x['path']).resolve();assert p.is_relative_to(phase)
    b=p.read_bytes();assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256'],'Custody differs: '+x['path']
    return p
'''
code += 'packets=' + repr(packets) + '\nextras=' + repr(extras) + '\noriginals=' + repr(originals) + '\n'
code += '''summary=[]
for x in packets:
    manifest=verify(x['manifest']);d=phase/x['path']
    if 'seal' in x:verify(x['seal'])
    rows=json.loads(manifest.read_text())['files']
    for row in rows:
        p=(d/row['path']).resolve();assert p.is_relative_to(d.resolve())
        verify({'path':str(p.relative_to(phase)), 'bytes':row.get('bytes',row.get('size')), 'sha256':row['sha256']})
    summary.append({**x,'payloads_verified':len(rows)})
for x in extras:verify(x)
for x in originals:verify(x)
assert json.loads((phase/'graph_ncNC_structural_pattern_v4_numerical_helper_root_review_20261003_v1/REVIEW.json').read_text())['status']=='passed'
assert json.loads((phase/'graph_ncNC_structural_pattern_v4_independent_source_review_20261003_v1/REVIEW.json').read_text())['verdict']=='NO_IDENTIFIED_MATERIAL_SOURCE_BLOCKER_IN_SCOPED_CHANGE'
for name in ('SOURCE_CUSTODY.json','ROOT_NUMERICAL_ADMISSION.json','ROOT_RELEASE_NUMERICAL.json','DETACHED_SUPERVISOR.log','DISPATCH.json','supervision/run01','numerical/run01'):
    assert not (root/name).exists(),'Nonfresh execution path: '+name
value={'schema':'V4_numerical_exact_remote_source_custody_v1',
       'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'hostname':subprocess.check_output(['hostname'],text=True).strip(),
       'repository':str(repo),'git_HEAD':'db0302db9991bbb5e167a94948ff6fe5f4199551',
       'verified_packets':summary,'extra_exact_file_checks':len(extras),
       'prior_original_remote_receipts_verified':originals,
       'both_full_graph_failures_preserved':True,'historical_numerical_attempts_preserved':True,
       'historical_supervisor_inclusive_wall_sum_seconds':96.37820457667112,
       'source_only_no_numerical_execution':True,'source_overwrites':False,
       'candidate_release_sha256':'76bb72c5dec9883d8511a31ad911c3937315009a91ba9092f0676b65232d5654',
       'model_review_sha256':'7682474372bee7162f969710ad4f4651cdb06cb1bbb507529d4a5826402f5322',
       'helper_root_review_sha256':'84b94d7e1850d48de979598eafc958d4d07e1faac3b00d426ee34751cf8c327c'}
root.mkdir(mode=0o700,exist_ok=False)
b=(json.dumps(value,indent=2,allow_nan=False)+'\\n').encode()
with (root/'SOURCE_CUSTODY.json').open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
print(b.decode())
'''
command = "set -e\ncd /disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git\n/usr/bin/python3 - <<'PYREMOTE'\n" + code + 'PYREMOTE\n'
path = TRANSPORT / 'NCNC_V4_NUMERICAL_FINAL_SOURCE_CUSTODY_20261003_v1.txt'
with path.open('x') as handle:
    handle.write(command)
print(json.dumps({'command_file': str(path), 'bytes': len(command.encode()),
                  'packets': len(packets), 'extra_checks': len(extras),
                  'original_receipts': len(originals)}))
