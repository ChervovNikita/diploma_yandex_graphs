"""Operational scalar-only preparation; never launches or imports checkpoints."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE = PHASE / 'pubmed_native_predictive_program_source_20261004_v3'
ADMISSION = PHASE / 'pubmed_native_cohort_root_admission_20261004_v1'
SOURCE_SHA = 'cd360a31fc6feead174f1da3ea9478655aa3bd78f68aab548601cc9fb37090e5'
FIT_RELEASE_SHA = 'b1c8a054a8ae1acd4bf75fa0fb3d75d52b513474f028e943adb5a53ee2ca4a6e'
EXPECTED_SUPERVISOR = {'PID': 3198977, 'start_time_ticks': 1723410112}
EXPECTED_FITS = ['SAGE_seed0', 'SAGE_seed1', 'SAGE_seed2', 'NCNC_seed0', 'NCNC_seed1', 'NCNC_seed2']

def require(value, message):
    if not value: raise RuntimeError(message)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--wrapper-receipt', required=True, type=Path)
parser.add_argument('--observation-id', required=True)
args = parser.parse_args()
require(args.observation_id.isdecimal(), 'Numeric observation identity required')
require(sha(SOURCE / 'MANIFEST.json') == SOURCE_SHA, 'Existing scientific source manifest changed')
require(sha(ADMISSION / 'ROOT_RELEASE_fit_cohort.json') == FIT_RELEASE_SHA, 'Original fit release changed')
wrapper_bytes = args.wrapper_receipt.read_bytes()
wrapper = json.loads(wrapper_bytes)
require(wrapper['exit_code'] == 0 and wrapper['target'] == 'shmelev@192.168.18.77' and wrapper['credential_value_recorded'] is False, 'Read-only wrapper receipt invalid')
observation = json.loads(wrapper['stdout'])
require(observation['source_manifest_sha256'] == SOURCE_SHA and observation['original_fit_release_sha256'] == FIT_RELEASE_SHA, 'Operational source/release binding differs')
for key, expected in EXPECTED_SUPERVISOR.items():
    actual = observation['supervisor'].get(key, observation['supervisor'].get('expected', {}).get(key))
    require(actual == expected, 'Exact-owned supervisor identity differs')
require(observation['no_process_or_configuration_mutation'] is True and observation['optimizer_updates_performed'] == 0
        and observation['checkpoint_deserialization'] is False and observation['replay_or_scoring_performed'] is False
        and observation['TEST_accessed'] is False, 'Read-only preparation scope differs')
require(not observation['failure_files'], 'Owned fit failure detected; root action required')
(HERE / ('REMOTE_OBSERVATION_' + args.observation_id + '_WRAPPER.json')).write_bytes(wrapper_bytes)
write(HERE / ('REMOTE_OBSERVATION_' + args.observation_id + '.json'), observation)

cohort = json.loads((SOURCE / 'COHORT.json').read_text())
original = json.loads((ADMISSION / 'ROOT_RELEASE_fit_cohort.json').read_text())
candidate = copy.deepcopy(original)
candidate.update(status='DISABLED_TEMPLATE', authorized_stages=[], scientific_fit_admitted=False,
                 root_authorization_reference=None, caps=cohort['stages']['replay_selected']['caps'],
                 invocation=cohort['stages']['replay_selected']['invocation'],
                 fit_cohort_freeze_sha256=None, fit_supervisor_terminal_sha256=None)
state = {'schema': 'pubmed-native-selected-replay-operational-preparation-v1', 'UTC': datetime.now(timezone.utc).isoformat(),
         'label': 'Operational follow-up, not a new independent source review', 'status': 'PENDING_COMPLETE_FIT_CUSTODY',
         'source_manifest_sha256': SOURCE_SHA, 'original_fit_release_sha256': FIT_RELEASE_SHA,
         'exact_supervisor_identity': EXPECTED_SUPERVISOR, 'latest_remote_observation_UTC': observation['UTC'],
         'latest_completed_fits': observation['progress']['completed_fits'],
         'wrapper_receipt_sha256': hashlib.sha256(wrapper_bytes).hexdigest(), 'command_sha256': wrapper['command_sha256'],
         'replay_optimizer_updates': 0, 'replay_authorized': False, 'replay_or_scoring_executed': False,
         'TEST_supported': False, 'automatic_retry': False, 'fit_attempt_changed': False,
         'scientific_source_or_original_fit_modified': False}
if observation['readiness'] == 'READY_FOR_DISABLED_REPLAY_RELEASE_CANDIDATE':
    require(observation['all_six_completed'] is True and observation['all_scalar_completion_files_exist'] is True,
            'All six completion custody absent')
    expected_names = {'COHORT_FREEZE.json', 'FINAL_CUSTODY.json', 'SUPERVISOR_TERMINAL.json'}
    require(set(observation['artifacts']) == expected_names, 'Only three scalar completion artifacts are permitted')
    artifacts = {}
    for name, row in observation['artifacts'].items():
        raw = row['text'].encode('utf-8')
        require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'Scalar transfer custody differs')
        artifacts[name] = json.loads(raw)
        (HERE / name).write_bytes(raw)
    freeze, custody, terminal = (artifacts[name] for name in ('COHORT_FREEZE.json', 'FINAL_CUSTODY.json', 'SUPERVISOR_TERMINAL.json'))
    require(freeze['status'] == 'COMPLETE' and [row['fit_id'] for row in freeze['fits']] == EXPECTED_FITS
            and all(row['status'] == 'COMPLETE' for row in freeze['fits']), 'Six native fits incomplete')
    require(freeze['source_manifest_sha256'] == SOURCE_SHA and freeze['root_release_sha256'] == FIT_RELEASE_SHA, 'Fit provenance differs')
    require(terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0 and terminal['cap_violation'] is None,
            'Fit supervisor did not close normally')
    require({key: terminal['supervisor_identity'][key] for key in EXPECTED_SUPERVISOR} == EXPECTED_SUPERVISOR,
            'Original supervisor terminal identity differs')
    require(terminal['source_manifest_sha256'] == SOURCE_SHA and terminal['release_sha256'] == FIT_RELEASE_SHA
            and terminal['ordinary_host_execution'] is True, 'Terminal source/release/normal physics differs')
    require(custody['completed'] is True and custody['stage'] == 'fit_cohort', 'Owned final fit custody incomplete')
    require(terminal['result']['sha256'] == sha(HERE / 'COHORT_FREEZE.json')
            and terminal['final_custody']['sha256'] == sha(HERE / 'FINAL_CUSTODY.json'), 'Terminal scalar custody binding differs')
    require(freeze['cohort_sha256'] == sha(SOURCE / 'COHORT.json'), 'Completed cohort differs from reviewed policy')
    file_rows = {row['path']: row for row in custody['files']}
    selected = []
    for row in freeze['fits']:
        fit_id = row['fit_id']
        for name, key in [('selected_state.pt', 'selected_state_sha256'), ('selected_VALID_scores.pt', 'selected_VALID_scores_sha256')]:
            require(file_rows[fit_id + '/' + name]['sha256'] == row[key], 'Selected artifact hash inventory differs')
        selected.append({key: row[key] for key in ('fit_id', 'model', 'seed', 'literal_seed_id', 'selected_epoch',
                                                  'selected_state_sha256', 'selected_VALID_scores_sha256')})
    candidate.update(fit_cohort_freeze_sha256=sha(HERE / 'COHORT_FREEZE.json'),
                     fit_supervisor_terminal_sha256=sha(HERE / 'SUPERVISOR_TERMINAL.json'))
    state.update(status='DISABLED_REPLAY_RELEASE_CANDIDATE_READY',
                 fit_cohort_freeze_sha256=candidate['fit_cohort_freeze_sha256'],
                 fit_supervisor_terminal_sha256=candidate['fit_supervisor_terminal_sha256'],
                 fit_final_custody_sha256=sha(HERE / 'FINAL_CUSTODY.json'), selected_artifact_scalar_identities=selected,
                 supervisor_inclusive_wall_seconds=terminal['inclusive_supervisor_wall_seconds'],
                 fit_optimizer_updates=freeze['progress']['Adam_calls_completed'],
                 complete_VALID_serves=freeze['progress']['complete_VALID_serves'],
                 scientific_selected_replay_status='UNEXECUTED')
write(HERE / 'ROOT_RELEASE_replay_selected.json', candidate)
state['disabled_replay_release_candidate_sha256'] = sha(HERE / 'ROOT_RELEASE_replay_selected.json')
write(HERE / 'PREPARATION_STATE.json', state)
print(json.dumps({key: state[key] for key in ('status', 'latest_remote_observation_UTC', 'latest_completed_fits',
    'disabled_replay_release_candidate_sha256', 'replay_authorized')}, indent=2))
