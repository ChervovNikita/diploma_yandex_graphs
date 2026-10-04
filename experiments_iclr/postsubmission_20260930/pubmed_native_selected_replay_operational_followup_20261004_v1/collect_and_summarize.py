"""Authenticate completed scalar receipts and summarize native VALID baselines."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
PREPARATION=PHASE/'pubmed_native_selected_replay_root_preparation_20261004_v1'
ADMISSION=PHASE/'pubmed_native_selected_replay_root_admission_20261004_v1'
TRANSPORT=PHASE/'gpu77_connection_recovery_v1/commands'
SOURCE_SHA='cd360a31fc6feead174f1da3ea9478655aa3bd78f68aab548601cc9fb37090e5'
RELEASE_SHA='561444463580d2de7eb6039b0ad779cc5ceb7a42532fdce9587db80f92df45e8'
EXPECTED={'PID':3201522,'start_time_ticks':1723585972}
FIT_IDS=['SAGE_seed0','SAGE_seed1','SAGE_seed2','NCNC_seed0','NCNC_seed1','NCNC_seed2']
def require(value,message):
    if not value:raise RuntimeError(message)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

completion_receipt=TRANSPORT/'pubmed_selected_replay_completion_20261004_001/RECEIPT.json'
wrapper=json.loads(completion_receipt.read_text())
require(wrapper['exit_code']==0 and wrapper['target']=='shmelev@192.168.18.77' and wrapper['credential_value_recorded'] is False,'Authenticated transport receipt differs')
observation=json.loads(wrapper['stdout'])
require(observation['readiness']=='COMPLETE_PASS' and not observation['failure_files'],'Owned replay did not complete normally')
require(observation['source_manifest_sha256']==SOURCE_SHA and observation['replay_release_sha256']==RELEASE_SHA,'Completed replay binding differs')
require(observation['all_six_replays_completed'] is True and observation['all_completion_files_exist'] is True,'Six completed replays absent')
require(observation['no_process_or_configuration_mutation'] is True and observation['TEST_accessed'] is False
    and observation['observer_replay_or_scoring_performed'] is False and observation['checkpoint_deserialization_by_observer'] is False,'Observer scope differs')
(HERE/'COMPLETION_WRAPPER_RECEIPT.json').write_bytes(completion_receipt.read_bytes())
write(HERE/'COMPLETE_OBSERVATION.json',observation)
for name,row in observation['artifacts'].items():
    require(name in {'REPLAY.json','FINAL_CUSTODY.json','SUPERVISOR_TERMINAL.json'},'Unexpected artifact')
    raw=row['text'].encode()
    require(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],'Scalar artifact transfer differs')
    (HERE/name).write_bytes(raw)
replay=json.loads((HERE/'REPLAY.json').read_text())
terminal=json.loads((HERE/'SUPERVISOR_TERMINAL.json').read_text())
custody=json.loads((HERE/'FINAL_CUSTODY.json').read_text())
require(replay['status']=='PASS' and replay['optimizer_updates']==0 and [r['fit_id'] for r in replay['reports']]==FIT_IDS,'Complete zero-update replay PASS absent')
require(replay['source_manifest_sha256']==SOURCE_SHA and replay['root_release_sha256']==RELEASE_SHA,'Replay source/release differs')
require(terminal['status']=='COMPLETE' and terminal['exit_code']==0 and terminal['cap_violation'] is None,'Replay supervisor did not complete within cap')
require({k:terminal['supervisor_identity'][k] for k in EXPECTED}==EXPECTED and terminal['ordinary_host_execution'] is True,'Owned supervisor/normal physics differs')
require(terminal['source_manifest_sha256']==SOURCE_SHA and terminal['release_sha256']==RELEASE_SHA,'Terminal provenance differs')
require(terminal['result']['sha256']==sha(HERE/'REPLAY.json') and terminal['final_custody']['sha256']==sha(HERE/'FINAL_CUSTODY.json'),'Terminal scalar custody differs')
require(custody['completed'] is True and custody['stage']=='replay_selected','Replay final custody incomplete')
require(next(row for row in custody['files'] if row['path']=='REPLAY.json')['sha256']==sha(HERE/'REPLAY.json'),'Replay inventory binding differs')
release_path=ADMISSION/'ROOT_RELEASE_replay_selected.json'
require(sha(release_path)==RELEASE_SHA,'Local admitted release differs')
release=json.loads(release_path.read_text())
require(sha(PREPARATION/'COHORT_FREEZE.json')==release['fit_cohort_freeze_sha256']==replay['fit_cohort_freeze_sha256'],'Selected fit freeze binding differs')
(HERE/'COHORT_FREEZE.json').write_bytes((PREPARATION/'COHORT_FREEZE.json').read_bytes())
freeze=json.loads((HERE/'COHORT_FREEZE.json').read_text())
cohort=json.loads((PHASE/'pubmed_native_predictive_program_source_20261004_v3/COHORT.json').read_text())
require(replay['rule']==cohort['replay_rule'],'Prospective replay tolerance changed')

dispatch=TRANSPORT/'pubmed_selected_replay_dispatch_20261004_v1/RECEIPT.json'
launch_wrapper=json.loads(dispatch.read_text())
require(launch_wrapper['exit_code']==0 and launch_wrapper['target']=='shmelev@192.168.18.77','Authenticated dispatch receipt differs')
require(sha(dispatch.parent/'COMMAND.txt')==launch_wrapper['command_sha256'],'Dispatch command custody differs')
launch=json.loads(launch_wrapper['stdout'])
require(launch['runner_handle']['pid']==EXPECTED['PID'] and launch['runner_handle']['start_ticks']==EXPECTED['start_time_ticks']
    and launch['root_release_sha256']==RELEASE_SHA and launch['optimizer_updates']==0,'Authenticated launch identity differs')
(HERE/'LAUNCH_WRAPPER_RECEIPT.json').write_bytes(dispatch.read_bytes())
raw_row=observation['raw_launch_descriptor'];raw=base64.b64decode(raw_row['b64'])
require(len(raw)==raw_row['bytes'] and hashlib.sha256(raw).hexdigest()==raw_row['sha256'],'Raw launch descriptor custody differs')
(HERE/'REPLAY_DETACHED_LAUNCH.raw').write_bytes(raw)
require(json.loads(raw)==launch,'Raw descriptor and authenticated launch stdout differ')
write(HERE/'NORMALIZED_REPLAY_DETACHED_LAUNCH.json',launch)
launch_note={'schema':'pubmed-native-replay-launch-descriptor-advisory-v1','advisory':'Root suspected a literal backslash-n suffix; the authenticated fetched bytes do not confirm this defect',
    'advisory_resolution':'Raw descriptor is valid JSON and ends in an actual newline; no malformed suffix, failure or repair claim',
    'raw_sha256':raw_row['sha256'],'raw_bytes':len(raw),'raw_last_bytes_hex':raw[-6:].hex(),
    'ends_with_actual_newline':raw.endswith(b'\n'),'ends_with_literal_backslash_n':raw.endswith(b'\\n'),
    'raw_JSON_parse':'PASS','normalized_from':'Authenticated launch wrapper stdout, compared equal to valid raw JSON payload',
    'normalized_descriptor_sha256':sha(HERE/'NORMALIZED_REPLAY_DETACHED_LAUNCH.json'),
    'original_descriptor_altered':False,'launch_rerun':False}
write(HERE/'LAUNCH_DESCRIPTOR_ADVISORY.json',launch_note)

cells=[]
for selected,report in zip(freeze['fits'],replay['reports']):
    require(selected['fit_id']==report['fit_id'] and selected['model']==report['model'] and selected['seed']==report['seed']
        and selected['literal_seed_id']==report['literal_seed_id'] and selected['selected_epoch']==report['selected_epoch'],'Selected replay cell identity differs')
    require(selected['selected_state_sha256']==report['selected_state_sha256'] and selected['selected_VALID_scores_sha256']==report['selected_VALID_scores_sha256'],'Selected replay artifact hashes differ')
    require(report['status']=='PASS' and report['optimizer_updates']==0 and report['state_restore_tolerance']==0
        and report['full_postserve_state_and_RNG_tolerance']==0 and report['all_per_query_and_rounded_metric_tolerance']==0,'Exact replay conditions differ')
    require(report['scores_atol']==cohort['replay_rule']['atol'] and report['scores_rtol']==cohort['replay_rule']['rtol'],'Cell prospective score tolerance differs')
    cells.append({'fit_id':selected['fit_id'],'model':selected['model'],'seed':selected['seed'],'literal_seed_id':selected['literal_seed_id'],
        'selected_native_VALID_MRR_rounded4':selected['selected_VALID_MRR'],'selected_epoch':selected['selected_epoch'],
        'last_training_epoch':selected['last_training_epoch'],'native_Adam_updates':selected['native_Adam_updates'],
        'native_stop_reason':selected['stop_reason'],'selected_state_sha256':selected['selected_state_sha256'],
        'selected_VALID_scores_sha256':selected['selected_VALID_scores_sha256'],'serialized_selected_replay_status':report['status'],
        'observed_max_absolute_score_difference':report['observed_max_absolute_score_difference']})
summary={'schema':'pubmed-six-cell-native-VALID-baseline-summary-v1','status':'COMPLETE_WITH_SERIALIZED_SELECTED_REPLAY_PASS',
    'baseline_only':True,'new_GNNM_result':False,'novelty_claim':False,'TEST_supported':False,
    'VALID_positive_queries':2216,'negative_candidates_per_query':500,'metric':'Native HeaRT rounded-four-decimal MRR',
    'selector':'First rounded VALID maximum at the fixed five-epoch evaluation cadence',
    'feature_semantics':release['feature_authority'],'negative_pool_semantics':release['negative_pool_authority'],
    'fit_cohort_freeze_sha256':sha(HERE/'COHORT_FREEZE.json'),'replay_sha256':sha(HERE/'REPLAY.json'),
    'replay_terminal_sha256':sha(HERE/'SUPERVISOR_TERMINAL.json'),'source_manifest_sha256':SOURCE_SHA,
    'original_fit_release_sha256':freeze['root_release_sha256'],'replay_release_sha256':RELEASE_SHA,
    'fit_optimizer_updates':freeze['progress']['Adam_calls_completed'],'fit_complete_VALID_serves':freeze['progress']['complete_VALID_serves'],
    'replay_optimizer_updates':0,'inclusive_replay_supervisor_wall_seconds':terminal['inclusive_supervisor_wall_seconds'],
    'cells':cells,'limitations':['TRAIN/VALID selected native baseline experiment on one released graph/pool','No TEST, transfer or new GNNM experiment','No original-paper score amendment or novelty claim','Native pool retains replacement duplicates and other-VALID positive collisions','Terminal receipt-write tail is explicitly unmeasured']}
write(HERE/'NATIVE_VALID_BASELINE_SUMMARY.json',summary)
lookup={(row['model'],row['seed']):row for row in cells}
lines=['Native Pubmed TRAIN/VALID baselines — six cells, 2026-10-04','','All six selected scientific states passed serialized replay; its supervisor completed with zero optimizer updates and no cap breach. These are native baseline results.','','| Native model | Seed 0 VALID MRR | Seed 1 VALID MRR | Seed 2 VALID MRR |','|---|---:|---:|---:|']
for model in ('SAGE','NCNC'):
    lines.append('| '+model+' | '+' | '.join(f"{lookup[(model,seed)]['selected_native_VALID_MRR_rounded4']:.4f}" for seed in (0,1,2))+' |')
lines+=['','MRR is the native rounded four-decimal metric over 2,216 VALID queries with 500 ordered candidates per query. Each cell is the first rounded maximum under the fixed five-epoch VALID cadence.','','| Fit | Selected epoch | Final native epoch | Serialized replay | Max absolute score difference |','|---|---:|---:|---|---:|']
for row in cells:lines.append(f"| {row['fit_id']} | {row['selected_epoch']} | {row['last_training_epoch']} | PASS | {row['observed_max_absolute_score_difference']:g} |")
lines+=['',f"The native fits incurred {summary['fit_optimizer_updates']:,} Adam calls and {summary['fit_complete_VALID_serves']} VALID serves. Selected replay incurred zero optimizer updates and completed in {terminal['inclusive_supervisor_wall_seconds']:.2f} inclusive supervisor seconds. It checked exact restored/postserve state, RNG and per-query/rounded metrics under the prospectively fixed score tolerance.",'','The feature values are the released raw Planetoid/PyG values without NormalizeFeatures. The native negative candidate pool preserves replacement duplicates and other-VALID positives. This is one TRAIN/VALID baseline cohort: it establishes no TEST performance, transfer result, new GNNM result or novelty. Original-paper scores were not amended.','','Launch metadata advisory: the suspected literal backslash-n suffix was unconfirmed. The preserved raw descriptor is valid JSON and ends in an actual newline; a separate normalized copy was reconstructed from authenticated stdout. The original was not edited, and launch was not repeated.','']
(HERE/'NATIVE_VALID_BASELINE_SUMMARY.md').write_text('\n'.join(lines))
write(HERE/'OPERATIONAL_COMPLETION_RECEIPT.json',{'schema':'pubmed-native-selected-replay-operational-completion-receipt-v1','UTC':datetime.now(timezone.utc).isoformat(),
    'operational_followup':True,'fresh_independent_source_review':False,'status':'COMPLETE_PASS','complete_replay_cells':6,
    'source_manifest_sha256':SOURCE_SHA,'replay_release_sha256':RELEASE_SHA,'supervisor_identity':EXPECTED,
    'child_identity':{'PID':terminal['child_identity']['PID'],'start_time_ticks':terminal['child_identity']['start_time_ticks']},
    'replay_sha256':sha(HERE/'REPLAY.json'),'final_custody_sha256':sha(HERE/'FINAL_CUSTODY.json'),'supervisor_terminal_sha256':sha(HERE/'SUPERVISOR_TERMINAL.json'),
    'fit_cohort_freeze_sha256':sha(HERE/'COHORT_FREEZE.json'),'native_VALID_baseline_summary_sha256':sha(HERE/'NATIVE_VALID_BASELINE_SUMMARY.json'),
    'supervisor_status':'COMPLETE','supervisor_exit_code':0,'cap_violation':None,'replay_optimizer_updates':0,
    'observed_max_absolute_score_differences':[row['observed_max_absolute_score_difference'] for row in cells],
    'launch_advisory_sha256':sha(HERE/'LAUNCH_DESCRIPTOR_ADVISORY.json'),'raw_launch_descriptor_sha256':sha(HERE/'REPLAY_DETACHED_LAUNCH.raw'),
    'source_or_tolerance_changed':False,'original_fit_changed':False,'retry_or_resume':False,'TEST_accessed':False,
    'observer_performed_replay_or_scoring':False,'observer_opened_checkpoints_or_score_files':False,'process_signals_sent':False,
    'original_launch_descriptor_altered':False,'launch_rerun':False,'baseline_only':True,'new_GNNM_or_novelty_claim':False})
files=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name not in {'MANIFEST.json','SEAL.json'}]
write(HERE/'MANIFEST.json',{'schema':'pubmed-native-selected-replay-operational-followup-manifest-v1','source_manifest_sha256':SOURCE_SHA,
    'replay_release_sha256':RELEASE_SHA,'files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in files]})
seal={'schema':'pubmed-native-selected-replay-operational-followup-seal-v1','status':'COMPLETE_PASS','manifest_sha256':sha(HERE/'MANIFEST.json'),
    'completion_receipt_sha256':sha(HERE/'OPERATIONAL_COMPLETION_RECEIPT.json'),'summary_sha256':sha(HERE/'NATIVE_VALID_BASELINE_SUMMARY.json'),
    'replay_sha256':sha(HERE/'REPLAY.json'),'supervisor_terminal_sha256':sha(HERE/'SUPERVISOR_TERMINAL.json'),'baseline_only':True}
write(HERE/'SEAL.json',seal)
print(json.dumps({'seal':seal,'native_VALID_MRR':[{k:row[k] for k in ('fit_id','selected_native_VALID_MRR_rounded4','selected_epoch','observed_max_absolute_score_difference')} for row in cells]},indent=2))
