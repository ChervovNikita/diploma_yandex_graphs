"""Read immutable source/JSON evidence and reported scalars; write this audit only.

No packet imports, array/model/checkpoint reads, network, or numerical reruns.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import itertools
import json
import math
import re
import statistics

OUT = Path(__file__).resolve().parent
P = OUT.parent
R = P / 'ncnc_heldout_wrapper_qualification_execution_root_20261004_v1'
S = P / 'ncnc_frozen_all25_heldout_source_preparation_20261004_v2'
D = P / 'ncnc_frozen_all25_heldout_release_preparation_20261004_v2'
M = R / 'heldout_owned_monitor01'
V = P / 'ncnc_all25_heldout_fresh_source_review_20261004_v2'
OLD_AUDIT = P / 'ncnc_v4_all25_audit_owned_monitoring_20261004_v1/observation_0001/receipts'
LOCK = P / 'ncnc_complete_family_lock_execution_receipts_20261004_v1/graph_ncNC_predictive_family_execution_root_20261003_v1/family_lock/FAMILY_LOCK.json'
REMOTE_P = '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930'
inputs = {}
checks = []


def file_record(path, expected=None, binding='audit evidence'):
    path = Path(path)
    assert path.resolve().is_relative_to(P), path
    assert path.is_file() and not path.is_symlink(), path
    assert path.suffix in {'.json', '.py', '.md', '.txt', '.patch', '.log'} or path.name == '.gitignore', path
    assert not any(piece.startswith('PRIVATE_') and piece != 'PRIVATE_SOURCE_PINS.json' for piece in path.parts), path
    raw = path.read_bytes()
    raw.decode('utf-8')  # text/source only; never a tensor/checkpoint payload
    row = dict(local_path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), binding=binding)
    if expected:
        row.update(expected_sha256=expected['sha256'], sha256_matches=row['sha256'] == expected['sha256'])
        assert row['sha256_matches'], path
        if 'bytes' in expected:
            row.update(expected_bytes=expected['bytes'], bytes_match=row['bytes'] == expected['bytes'])
            assert row['bytes_match'], path
    inputs[str(path)] = row
    return row


def read(path, expected=None, binding='JSON evidence content reviewed'):
    file_record(path, expected, binding)
    return json.loads(Path(path).read_text())


def check(label, condition, evidence):
    assert condition, label
    checks.append(dict(check=label, observed_match=True, evidence=evidence))


def local_descriptor(path, remote=None):
    row = file_record(path)
    return dict(path=str(remote or REMOTE_P + '/' + str(Path(path).relative_to(P))), bytes=row['bytes'], sha256=row['sha256'])


def write_json(name, data):
    with (OUT / name).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


# Reuse prior static review, and recheck its exact text/source inputs for mutation.
review = read(V / 'SOURCE_REVIEW.json', {'sha256': 'd134482ee99afbac03c5426e6c0cdc9267016e4da9896bdb731fd63281ef4858', 'bytes': 7325})
prior = read(V / 'INPUT_HASH_RECEIPT.json', review['input_hash_receipt'])
for row in prior['inputs']:
    file_record(row['local_path'], row, 'previously reviewed text/source input; hash rechecked only')
check('158 prior source/metadata inputs still match immutable v2 review', len(prior['inputs']) == 158, str(V / 'INPUT_HASH_RECEIPT.json'))

enabled_path = R / 'ROOT_HELDOUT_RELEASE_ENABLED_ROOT_V1.json'
enabled = read(enabled_path, {'sha256': 'dbc38c9559d11012c910cd39b8064debb0a5bb7df42830499428b3ff2b6f5cb0', 'bytes': 34800})
release_hash = inputs[str(enabled_path)]['sha256']
adoption = read(R / 'ROOT_QA_ADOPTION.json', enabled['heldout_wrapper_qualification_root_adoption'])
admission = read(R / 'ROOT_HELDOUT_RESOURCE_ADMISSION.json', enabled['runtime_resource_admission'])
qa_release = read(R / 'ROOT_QA_RELEASE.json', enabled['heldout_wrapper_qualification_root_release'])
qa_release_hash = inputs[str(R / 'ROOT_QA_RELEASE.json')]['sha256']
qa = read(R / 'owned_monitor01/QUALIFICATION.json', enabled['heldout_wrapper_fabricated_qualification'])
qa_physical = read(R / 'owned_monitor01/physical/SUPERVISOR_TERMINAL.json', enabled['heldout_wrapper_qualification_physical_terminal'])
dispatch = read(R / 'ROOT_FINAL_DISPATCH_REVIEW.json')
gate = read(R / 'PRE_TEST_STDLIB_GATE.json', {'sha256': dispatch['pre_TEST_stdlib_gate_sha256']})
launch = read(R / 'HELDOUT_DETACHED_LAUNCH.json')
for name in ['ROOT_QA_ADMISSION.json', 'ROOT_PREHELDOUT_RESOURCE_OBSERVATION.json', 'SOURCE_STAGE_RECEIPT.json', 'DETACHED_LAUNCH.json', 'admit_and_dispatch_heldout.py', 'physical_qa_supervisor.py', 'stage_and_launch_qa.py']:
    file_record(R / name)
monitor = read(M / 'MONITOR_RECEIPT.json')
for row in monitor['files']:
    file_record(M / row['path'], row, 'immutable heldout monitor pin')
result = read(M / 'heldout/run01/HELDOUT_RESULT.json')
physical = read(M / 'supervision_run01/PHYSICAL_TERMINAL.json')
child_started = read(M / 'supervision_run01/CHILD_STARTED.json')
claim = read(M / 'ONE_TIME_TEST_CLAIM.json')
status = read(M / 'heldout/run01/STATUS.json')
lock = read(LOCK, enabled['family_lock'])
old = read(OLD_AUDIT / 'audit/run01/AUDIT_RESULT.json', enabled['v4_audit_result'])
old_physical = read(OLD_AUDIT / 'supervision_run01/PHYSICAL_TERMINAL.json', enabled['v4_audit_physical_terminal'])
old_release = read(P / 'ncnc_v4_scientific_audit_root_admission_20261004_v1/ROOT_ORDINARY_AUDIT_RELEASE.json', enabled['v4_audit_release'])
runtime = read(P / 'graph_ncNC_predictive_runtime_authority_root_20261003_v1/RUNTIME_AUTHORITY.json', enabled['runtime_authority'])
data_authority = read(P / 'graph_ncNC_valid_data_authority_root_20261003_v1/DATA_AUTHORITY.json', enabled['TRAIN_VALID_data_authority'])
test_authority = read(D / 'TEST_DATA_AUTHORITY_METADATA_CANDIDATE.json', enabled['TEST_data_authority'])
pilot = read(P / 'graph_ncNC_collab_predictive_pilot_design_20261003_v1/PILOT_PLAN.json')
manifest = read(S / 'MANIFEST.json', {'sha256': enabled['heldout_source_manifest_sha256']})
for row in manifest['files']:
    file_record(S / row['path'], row, 'current sealed heldout source manifest')
plan = read(S / 'FABRICATED_QUALIFICATION_PLAN.json')
for pin in [review['supervisor_source'], review['dispatcher_source']]:
    file_record(D / Path(pin['path']).name, pin, 'sealed one-time supervisor/dispatcher source')
file_record(R / 'physical_qa_supervisor.py', qa_physical['physical_supervisor'])

check('identity and policy inherited without substitution', result['identity'] == enabled['identity'] == lock['identity'] == old['identity'] == qa['original_family_identity'] == review['identity'] and result['policy'] == enabled['policy'] == review['policy'] == qa['policy'] == qa_release['policy'] == admission['policy'], 'root release, frozen lock, v4 audit, raw QA, v2 source review, result')
check('v4 lock and full selected unit custody retained', old_release['unit_custody'] == enabled['unit_custody'] and old['family_lock'] == result['family_lock'] == enabled['family_lock'], 'v4 release and authenticated all25 result')
check('actual QA release bound to logical and successful physical records', qa['status'] == 'PASS' and qa['root_release_sha256'] == qa_physical['root_release_sha256'] == qa_release_hash and qa_physical['status'] == 'PHYSICALLY_COMPLETE' and qa_physical['exit_code'] == 0 and qa_physical['error'] is None and qa_physical['bounded_stop_reason'] is None and qa_physical['signal_events'] == [] and qa_physical['release_and_supervisor_after_sha256_match'] is True, 'ROOT_QA_RELEASE, QUALIFICATION, SUPERVISOR_TERMINAL')
check('QA physical closure refers to exact adopted logical bytes', qa_physical['logical_result'] == enabled['heldout_wrapper_fabricated_qualification'] == adoption['actual_logical_QA'] and qa_physical['physical_supervisor'] == adoption['physical_QA_supervisor'] and enabled['root_authorization_reference'] == enabled['heldout_wrapper_qualification_root_adoption']['path'], 'QA physical logical-result descriptor and root authorization reference')
check('QA enabled for fabricated scope only', qa_release['execution_enabled'] is True and qa_release['fabricated_inputs_only'] is True and qa_release['TEST_access_authorized'] is False and qa_release['study_checkpoint_access_authorized'] is False and qa['fabricated_inputs_only'] is True and qa['TEST_opened'] is False and qa['study_data_or_selected_checkpoint_accessed'] is False and qa['training_updates'] == 0 and qa['automatic_retry'] is False, 'QA release and result')
check('exact 18 QA case identities PASS and literal 40/52 counts', qa['case_count'] == len(qa['cases']) == 18 and {x['case'] for x in qa['cases']} == set(plan['cases']) and all(x['status'] == 'PASS' for x in qa['cases']) and qa['actual_original_scorer_calls'] == sum(x.get('original_scorer_calls', 0) for x in qa['cases']) == 40 and qa['actual_stub_scorer_entries'] == sum(x.get('stub_scorer_entries', 0) for x in qa['cases']) == 52, 'fabricated plan and actual QA cases')
check('QA source entry, runtime, GPU/stage/output, supervisor, dispatcher match', qa['source_entry'] == qa_physical['source_entry'] == review['qualification_source_entry'] and qa['runtime_authority'] == qa_release['runtime_authority'] == enabled['runtime_authority'] and qa['invocation'] == qa_physical['invocation'] == qa_release['authorized_invocations'][0] == review['qualification_invocation'] and qa['heldout_source_manifest_sha256'] == qa_physical['heldout_source_manifest_sha256'] == enabled['heldout_source_manifest_sha256'] and qa_release['supervisor_source'] == review['supervisor_source'] and qa_release['dispatcher_source'] == review['dispatcher_source'], 'authenticated source and QA/root JSON')
check('root adoption actually pins exact QA release/logical/physical/source records', adoption['status'] == 'PASS' and adoption['actual_enabled_fabricated_QA_release'] == enabled['heldout_wrapper_qualification_root_release'] and adoption['actual_logical_QA'] == enabled['heldout_wrapper_fabricated_qualification'] and adoption['actual_successful_physical_QA_terminal'] == enabled['heldout_wrapper_qualification_physical_terminal'] and adoption['entry'] == qa['source_entry'] and adoption['supervisor'] == review['supervisor_source'] and adoption['dispatcher'] == review['dispatcher_source'] and adoption['runtime_authority'] == enabled['runtime_authority'] and adoption['invocation'] == qa['invocation'] and adoption['identity'] == enabled['identity'] and adoption['policy'] == enabled['policy'], 'actual root adoption content; all referenced local text pins verified')
check('adoption pinned in TEST release and resource admission and reviewed dispatch', admission['root_QA_adoption'] == dispatch['root_QA_adoption'] == enabled['heldout_wrapper_qualification_root_adoption'] and dispatch['root_release']['sha256'] == release_hash and dispatch['dispatcher'] == review['dispatcher_source'] and dispatch['status'] == 'APPROVED_ONCE' and enabled['execution_enabled'] is True and enabled['TEST_access_authorized'] is True and enabled['evaluation_once'] is True, 'root enabled release, admission and final dispatch review')
check('pre-TEST gate hash/release PASS and frozen dispatch criteria', gate['status'] == 'PASS' and gate['exit_code'] == 0 and gate['root_release_sha256'] == release_hash and gate['TEST_file_opened_or_hashed'] is False and dispatch['primary_contrast'] == 'private_minus_pooled' and dispatch['new_threshold'] is None and dispatch['training_updates'] == 0 and dispatch['retry_refit_reselection_calibration'] is False and dispatch['all25_no_subset_selection'] is True and dispatch['no_direct_child_invocation'] is True, 'PRE_TEST_STDLIB_GATE and exact dispatch record')
expected_release_remote = REMOTE_P + '/' + D.name + '/' + enabled_path.name
check('reviewed exact isolated-stdlib dispatcher command', dispatch['command'] == ['/usr/bin/python3', '-I', '-S', '-B', REMOTE_P + '/' + D.name + '/dispatch_heldout_once.py', '--root-release', expected_release_remote, '--release-sha256', release_hash], 'ROOT_FINAL_DISPATCH_REVIEW command')
check('single-use persistent claim precedes owned child spawn', claim['UTC'] < child_started['UTC'] and claim['root_release_sha256'] == launch['root_release_sha256'] == child_started['root_release_sha256'] == physical['root_release_sha256'] == result['root_release_sha256'] == release_hash and claim['TEST_access_once'] is True and claim['no_retry'] is True and claim['output_directory'] == enabled['authorized_invocations'][0]['output_directory'], 'one-time claim, launch, child start, physical and logical records; source x/open claim retained')
check('owned launch/claim/child identities match', all(launch['runner_handle'][k] == physical['runner_identity'][k] == child_started['runner_identity'][k] == claim['supervisor_identity'][k] for k in ['pid', 'start_ticks', 'pgid', 'sid']) and child_started['identity'] == physical['child_identity'] and child_started['argv'] == physical['argv'], 'owned process receipts')
check('successful physical heldout closure retains exact logical result', physical['status'] == 'PHYSICALLY_COMPLETE' and physical['exit_code'] == physical['raw_wait_status'] == 0 and physical['bounded_stop_reason'] is None and physical['error'] is None and physical['cleanup_errors'] == [] and physical['own_session_signal_events'] == [] and physical['unreaped_owned_child'] is None and physical['logical_closure']['action'] == 'CHILD_SUCCESS_RETAINED' and physical['logical_closure']['logical_terminal'] == local_descriptor(M / 'heldout/run01/HELDOUT_RESULT.json', enabled['authorized_invocations'][0]['output_directory'] + '/HELDOUT_RESULT.json'), 'PHYSICAL_TERMINAL fields and independent result bytes/hash')
check('actual heldout physical scope is authorized official serving without retry', physical['fabricated_inputs_only'] is False and physical['TEST_access_authorized'] is True and physical['original_TRAIN_VALID_lock_selected_checkpoint_access_authorized'] is True and physical['scientific_fit_updates_authorized'] == 0 and physical['no_retry'] is True and physical['other_jobs_signaled'] is False and physical['namespace_isolation_used'] is False and physical['runtime_hooks_added'] is False, 'owned physical terminal scope flags')
check('terminal status/source and status mirror match', result['status'] == status['status'] == 'ALL25_FROZEN_HELDOUT_CONFIRMATION_COMPLETE' and result['heldout_source_manifest_sha256'] == physical['sidecar_manifest_sha256'] == enabled['heldout_source_manifest_sha256'] and result['failures'] == [] and status['authoritative_result_receipt'] == physical['logical_closure']['logical_terminal'], 'terminal and mirror')
check('resource admission caps and concrete root observation match', admission['status'] == 'PASS' and admission['dispatch_recheck_required'] is True and admission['minimum_GPU_free_MiB'] == 24576 and admission['minimum_host_MemAvailable_bytes'] == 17179869184 and admission['maximum_child_wall_seconds'] == physical['timeout_seconds'] == 1800 and admission['maximum_sampled_owned_session_RSS_bytes'] == physical['maximum_sampled_owned_session_RSS_bytes'] == 34359738368 and admission['actual_resource_observation']['GPU'][0] == enabled['cuda_visible_devices'] and int(admission['actual_resource_observation']['GPU'][1]) >= 24576 and admission['actual_resource_observation']['host_MemAvailable_bytes'] >= 17179869184 and adoption['sentinel_QA_does_not_establish_production_memory'] is True, 'root resource observation/admission plus physical bounds; successful pinned supervisor path enforces immediate recheck')

arms = ['native_single_64', 'independent_native_4', 'factorized_private_4', 'factorized_pooled_after_clamp_4', 'native_single_70']
names = dict(zip(arms, ['Native64', 'Independent native4', 'Factorized private4', 'Factorized pooled-after-clamp4', 'Native70']))
cells = result['cells']
cell_map = {(c['arm'], c['base_seed']): c for c in cells}
lock_map = {(c['arm'], c['base_seed']): c for c in lock['cells']}
expected_cells = set(itertools.product(arms, range(5)))
check('all 25 identities complete with no subset, nulls or duplicated identities', len(cells) == len(cell_map) == 25 and set(cell_map) == set(lock_map) == expected_cells and all(c['status'] == 'PASS' and c['phase'] == 'cell_complete' and type(c['TEST_hits50']) is float and math.isfinite(c['TEST_hits50']) and 0 <= c['TEST_hits50'] <= 1 and all(c[k] is True for k in ['official_metric_attempted', 'official_metric_returned', 'official_metric_validated']) for c in cells), 'independent arm/seed grid and each terminal cell')
check('each selection and selected checkpoint metadata exactly equals frozen lock', all(c['selection'] == lock_map[key]['selection'] and all(c['checkpoint'][k] == lock_map[key]['checkpoint'][k] for k in ['path', 'bytes', 'sha256']) for key, c in cell_map.items()), '25 terminal cells compared field-for-field to FAMILY_LOCK metadata; no checkpoint bytes opened')
slots = [(c['arm'], c['base_seed'], i, digest) for c in cells for i, digest in enumerate(c['selected_state_digests'])]
check('40 scorer slots with arm-specific member cardinality', len(slots) == 40 and all(len(c['selected_state_digests']) == (4 if c['arm'] == 'independent_native_4' else 1) for c in cells), '5 native64 +20 independent members +5 private +5 pooled +5 native70')
duplicates = [[dict(arm=a, base_seed=s, member=i) for a,s,i,d in slots if d == digest] for digest in sorted({d for *_,d in slots}) if sum(d == digest for *_,d in slots) > 1]
check('39 distinct snapshots due disclosed seed4 donor reuse', len({d for *_,d in slots}) == 39 and duplicates == [[dict(arm='native_single_64', base_seed=4, member=0), dict(arm='independent_native_4', base_seed=4, member=0)]], '40 selected digest slots; seed4 native64 epoch17 is member0 in individually selected native4 bank')
work = result['work']
check('literal planned/attempted/entered/returned/validated scorer count40', all(work[k] == 40 for k in ['scorer_calls_planned', 'attempted', 'entered_original_scorer', 'returned', 'completed_validated']), 'public work ledger and mandatory pre-call/post-return source accounting')
check('all25 distinct official-metric/helper phases and cell counts', all(work[k] == 25 for k in ['cells_planned','cells_completed','official_metric_calls_planned','official_metric_calls','official_metric_attempted','official_metric_returned','official_metric_validated','metric_helper_attempted','metric_helper_returned']), 'public work ledger and per-cell official metric flags')
check('35 unique original fits and 59500 original updates; heldout zero updates', sum(x['unique_fits_completed'] for x in lock['unit_costs']) == lock['unique_fits_completed'] == result['frozen_unique_training_fits'] == 35 and pilot['work_budget']['full_family_unique_fits'] == 35 and pilot['epochs'] * pilot['data_graph']['train_batches_per_epoch'] * 35 == pilot['work_budget']['core_plus_auxiliary_update_steps'] == result['original_optimizer_updates'] == 59500 and work['training_updates'] == result['training_updates'] == 0 and result['automatic_retry'] is False and result['new_checkpoint_selection'] is False and result['new_calibration'] is False and result['no_success_only_subset_summary'] is True, 'frozen fit-cost metadata, original 100epoch/17full-batch schedule, heldout fields')
graph = result['actual_graph_receipt']
query = result['official_TEST_query_receipt']
check('TRAIN plus VALID topology with no TEST additions or target removal', graph['actual_graph'] == 'native_TRAIN_plus_VALID' and graph['train_records'] == 1179052 and graph['VALID_positive_records'] == 60084 and graph['graph_records'] == 1239136 and graph['TEST_rows_added'] is False and graph['weights_used'] is False and graph['query_target_removal'] is False and graph['original_label_is_not_actual_topology_authority'] is True, 'actual graph receipt and authenticated adapt_TEST_queries source; tensor topology not independently recomputed')
check('complete official TEST geometry and typed/order pins match authority', query['positive_rows'] == 46329 and query['negative_rows'] == 100000 and query['year'] == 2019 and query['complete_original_order'] is True and query['typed_query_digests'] == test_authority['typed_query_digests'] and query['official_TEST_file'] == test_authority['official_TEST_file'], 'official TEST receipt versus adopted metadata authority; TEST bytes not read/hashed')
check('final custody and strict serving profile retained', all(result['final_input_custody'][k] == 'PASS' for k in ['original_family_source_runtime_TRAIN_VALID_TEST_file_custody','loaded_canonical_graph_and_TEST_rows','fixed_strict_profile']) and result['policy']['profile'] == 'True2' and result['policy']['eval_batch_size'] == 131072 and result['policy']['serving_pool'] == 'mean_raw_logits', 'mandatory final custody immediately before public serialization and scorer restore/profile/RNG guards')

scores = {a: [cell_map[a,s]['TEST_hits50'] for s in range(5)] for a in arms}
arm_stats = {a: dict(TEST_hits50=v, mean=statistics.mean(v), sample_sd=statistics.stdev(v)) for a,v in scores.items()}
for row in result['summary']['arm_summaries']:
    check('independent terminal arm summary equals cells: ' + row['arm'], row['TEST_hits50'] == scores[row['arm']] and row['mean'] == arm_stats[row['arm']]['mean'] and row['sample_sd'] == arm_stats[row['arm']]['sample_sd'], 'statistics from all five scalar terminal cells')


def contrast(a, b, scope):
    ds = [x-y for x,y in zip(scores[a], scores[b])]
    mean = statistics.mean(ds)
    sd = statistics.stdev(ds)
    se = sd / math.sqrt(5)
    critical = 2.7764451051977987
    sign_means = [sum(sign*d for sign,d in zip(signs,ds))/5 for signs in itertools.product([-1,1], repeat=5)]
    extreme = sum(abs(x) >= abs(mean)-1e-15 for x in sign_means)
    return dict(candidate=a, control=b, scope=scope, base_seeds=list(range(5)), paired_TEST_hits50_differences=ds, mean=mean, sample_sd=sd, paired_standard_error=se, paired_t_statistic=mean/se, degrees_of_freedom=4, approximate_95_percent_paired_t_interval=[mean-critical*se,mean+critical*se], t_critical=critical, range=[min(ds),max(ds)], sign_count=dict(positive=sum(d>0 for d in ds),zero=sum(d==0 for d in ds),negative=sum(d<0 for d in ds)), exact_sign_flip_two_sided_p=extreme/32, exact_sign_flip_extreme_combinations=extreme, exact_sign_flip_combinations=32, leave_one_seed_out_means=[statistics.mean(ds[:i]+ds[i+1:]) for i in range(5)])


primary = contrast(arms[2], arms[3], 'frozen primary; separately VALID-selected arms; five training-seed blocks conditional on one graph/time split')
reported = result['summary']['primary_frozen_private_minus_pooled']
check('independent primary paired summary equals terminal cells', primary['paired_TEST_hits50_differences'] == reported['paired_TEST_hits50_differences'] and primary['mean'] == reported['mean'] and primary['sample_sd'] == reported['sample_sd'] and primary['range'] == reported['range'] and primary['sign_count'] == reported['sign_count'] and reported['new_quality_threshold'] is None, 'all five private minus pooled scores; no monitor summary used')
exploratory_pairs = [(arms[2],arms[0]),(arms[2],arms[1]),(arms[2],arms[4]),(arms[3],arms[0]),(arms[3],arms[1]),(arms[3],arms[4]),(arms[1],arms[0]),(arms[4],arms[0]),(arms[1],arms[4])]
exploratory = [contrast(a,b,'exploratory descriptive; no multiplicity adjustment or confirmatory superiority conclusion') for a,b in exploratory_pairs]

limits = [
    'This authenticates local text/source copies against immutable pins and reconciles reported scalar metrics; it is not an independent numerical rerun.',
    'No TEST/TRAIN/VALID arrays, model/checkpoint tensors, runtime binary bytes, private numerical evidence or private scorer/cell receipt contents were opened, hashed or deserialized. Selected state/RNG/profile/graph proof therefore uses authenticated mandatory source guards, terminal custody fields and inherited v4/lock evidence.',
    'The standalone immediate RESOURCE_DISPATCH_RECHECK.json was not among the locally copied monitor files. Its successful enforcement is evidenced by the authenticated supervisor path before claim/spawn and the clean child result; immediate free-memory figures cannot be independently quoted from a fetched recheck receipt.',
    'The QA physical schema has no explicit cleanup_errors or unreaped_child fields. Its exit0, no error/stop/signals, kernel wait4 RSS accounting and authenticated monitor/finally source establish the successful reaped route; the TEST physical schema records those flags explicitly.',
    'R1 remains an automatic-gate limitation: heldout_gate.py108–115 does not itself authenticate the exact QA root release or successful physical terminal, and raw QA PASS is written before process exit. Concrete root adoption, exact immutable release pins and reviewed dispatch satisfy the obligation for this run; raw PASS alone remains insufficient for another run.',
    'Five seed blocks measure training randomness conditional on this one graph and official time split. The paired t interval assumes approximately independent seed differences and suitable mean-distribution behavior; n5 cannot validate those assumptions.',
    'The exact 32-combination sign-flip calculation is conditional on symmetric/exchangeable paired differences under a zero-centered null. It is not a randomized allocation experiment. With five nonzero pairs its minimum two-sided p is 0.0625.',
    'Query rows and the common negative pool are not independent inferential replicates. No row bootstrap, node/edge population interval or cross-dataset generalization is claimed.',
    'The primary TEST estimate is small and mixed-sign, with uncertainty covering both directions. It does not establish superiority, equivalence, noninferiority, a minimal practical gain, SOTA or novelty. No acceptance threshold was introduced.',
    'The frozen comparison includes separate VALID checkpoint selections and the full training/serving procedures. It does not isolate a pure within-checkpoint architectural causal effect. Development VALID was used for selection and is optimistic for that comparison.',
    'All nine baseline contrasts are exploratory; multiple comparisons were not adjusted and intervals are assumption-dependent descriptive summaries. Independent native4 has the highest mean among these frozen five arms.',
    'This successful serving run is one workload observation, not a guaranteed production memory bound or fresh benchmark comparison. Host RSS, CUDA peaks and overlapping wall intervals have distinct scopes and must not be summed.',
    'The terminal explicitly retains old_v2_exact_replay_failure_repaired=false. Successful frozen TEST serving does not retroactively repair the old exact-replay failure or justify broader qualification claims.',
]
audit = dict(schema='ncnc-all25-independent-heldout-evidence-audit-v1', UTC=datetime.now(timezone.utc).isoformat(), scope='focused independent evidence audit; no requested verdict or manuscript acceptance judgment', evidence_finding='The pinned completed all25/40 TEST record is consistent with the frozen source/selection/policy and clean owned physical closure. Root explicit adoption satisfies the concrete R1 obligation for this dispatch; scientific effect remains uncertain.', checks=checks, identity=result['identity'], root_release_sha256=release_hash, heldout_source_manifest_sha256=result['heldout_source_manifest_sha256'], logical_result=physical['logical_closure']['logical_terminal'], physical_result=local_descriptor(M/'supervision_run01/PHYSICAL_TERMINAL.json'), QA_actual_release_sha256=qa_release_hash, root_adoption_sha256=enabled['heldout_wrapper_qualification_root_adoption']['sha256'], selected_custody=dict(frozen_cells=25, snapshot_slots=40, distinct_state_digests=39, duplicate_slots=duplicates, checkpoint_and_selection_metadata_exactly_match_lock=True, checkpoint_bytes_reinspected=False), cohort_counts=dict(unique_training_fits=35, original_optimizer_updates=59500, new_heldout_optimizer_updates=0, scorer_slots=40, metric_cells=25), work=work, graph_receipt=graph, TEST_query_receipt=query, serving_profile='True2', runtime_authority_sha256=enabled['runtime_authority']['sha256'], prior_source_inputs_rehashed=158, current_source_manifest_files=len(manifest['files']), arm_statistics=arm_stats, frozen_primary_contrast=primary, exploratory_baseline_contrasts=exploratory, limitations=limits, resource_observations=dict(root_GPU_free_MiB=int(admission['actual_resource_observation']['GPU'][1]),root_host_MemAvailable_bytes=admission['actual_resource_observation']['host_MemAvailable_bytes'],cuda_peak_allocated_bytes=result['cuda_peak_allocated_bytes'],cuda_peak_reserved_bytes=result['cuda_peak_reserved_bytes'],owned_child_wait4_peak_RSS_bytes=physical['owned_child_wait4_peak_RSS_bytes'],sampled_peak_owned_session_RSS_bytes=physical['sampled_peak_owned_session_RSS_bytes'],physical_wrapper_wall_seconds=physical['physical_wrapper_wall_seconds'],child_spawn_through_wait4_wall_seconds=physical['child_spawn_through_wait4_wall_seconds'],inclusive_child_accounting_wall_seconds=result['inclusive_wall_seconds_through_accounting'],terminal_write_tail_measured=False,overlapping_intervals_added=False), restrictions_observed=dict(project_only=True,packet_imports=False,target_numerical_execution=False,server_access=False,arrays_or_model_checkpoint_reads_or_hashes=False,private_numerical_payload_reads_or_hashes=False,existing_artifacts_modified=False), manuscript_acceptance_judgment=None)

write_json('EVIDENCE_AUDIT.json', audit)
write_json('PAIRED_STATISTICS.json', dict(source='HELDOUT_RESULT.json cells only', source_sha256=inputs[str(M/'heldout/run01/HELDOUT_RESULT.json')]['sha256'], metric='official ogbl-collab Hits@50', unit='proportion; percentage points =100*proportion difference', arm_statistics=arm_stats, primary=primary, exploratory=exploratory, uncertainty_assumptions=limits[5:8]))
with (OUT / 'ALL_SEED_SCORES.csv').open('x', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['base_seed'] + arms + ['private_minus_pooled'])
    for seed in range(5):
        writer.writerow([seed] + [scores[a][seed] for a in arms] + [primary['paired_TEST_hits50_differences'][seed]])

rows = ['# Independent frozen all25 heldout evidence audit', '', f"Audit sealed: {audit['UTC']}. Scope: source/JSON evidence and official terminal scalar Hits@50 only. No requested verdict or manuscript acceptance judgment.", '', '## Evidence finding', '', audit['evidence_finding'], '', f"All {len(checks)} evidence consistency checks matched. The 158 prior v2 source/metadata inputs were rehashed without repeating the static review; all matched. The current sealed heldout source manifest has {len(manifest['files'])} files and all matched.", '', '## Custody and concrete root obligations', '', '| Evidence | Independent reconciliation |', '|---|---|', f"| Actual enabled QA release | `{qa_release_hash}`; fabricated only, no TEST/checkpoint access |", '| Actual QA logical/physical pairing | PASS, exact release/source/entry/runtime/GPU/stage/output/policy identity; physical exit0, no error/stop/signals; 18 exact cases, 40 original scorer calls and 52 stub entries |', f"| Root adoption | `{enabled['heldout_wrapper_qualification_root_adoption']['sha256']}`; actual QA release/result/physical pins included in enabled TEST release, resource admission and reviewed dispatcher |", f"| Exact enabled TEST release | `{release_hash}`; final review authorizes exactly one isolated stdlib dispatcher route; no subset, new threshold, training update, refit, reselection, calibration or retry |", f"| Logical terminal | `{inputs[str(M/'heldout/run01/HELDOUT_RESULT.json')]['sha256']}`; immutable result bytes155656 match monitor and physical closure |", f"| Physical TEST terminal | `{inputs[str(M/'supervision_run01/PHYSICAL_TERMINAL.json')]['sha256']}`; exit0/rawwait0, no stop/error/signals/cleanup failure/unreaped child; exact child logical result retained |", '| Single-use route | Persistent claim precedes child start; launch, claim, child and physical runner identity/release/source match; pinned source uses exclusive creation and fixed fresh output |', '| Frozen selected custody | 25 exact arm/seed selection and checkpoint metadata matches; 40 snapshot slots with 39 distinct state digests. Seed4 native64 epoch17 is the reused member0 state in the native4 individual-best bank. Training fits remain35 |', '', 'R1 is preserved: the automatic heldout gate checks raw QA PASS/source/runtime/identity/policy/cases/counts, but does not itself authenticate QA’s exact root release and physical success. QA can write PASS before process exit. This root used explicit admission that checked the actual matching release and successful physical terminal, pinned them into the exact TEST release, and reviewed the exact dispatcher/release hash. These concrete records satisfy the obligation for this run; raw PASS remains insufficient by itself.', '', 'The QA physical schema omits explicit cleanup/unreaped flags. Its no-error exit0 terminal, kernel wait4 accounting, and the authenticated monitor source (wait4 followed by child.returncode assignment) establish the successful reaped route. The heldout physical schema explicitly supplies empty cleanup_errors and null unreaped_owned_child.', '', '## All25/40 accounting, graph and runtime', '', 'The cohort is five arms × seeds0–4:25 PASS cells;40 scorer planned/attempted/entered/returned/validated events;25 official metric attempted/returned/validated events and25 helper attempted/returned events. No failures or null scores. Unique original fits35; frozen original schedule100epochs ×17full minibatches ×35fits =59,500 optimizer updates. Heldout optimizer updates0. Native64 reuses the native4 member0 fit; it is not five new fits.', '', f"The actual topology receipt is complete native TRAIN+VALID:1,179,052 TRAIN records followed by60,084 VALID positives =1,239,136 graph records. Canonical pair digest `{graph['canonical_pairs_sha256']}`. No TEST rows, weights or query-target removal. The legacy scorer label `complete_TRAIN_only` is explicitly an alias-era label, not the actual topology authority.", '', f"TEST uses all46,329 positives and the shared100,000 negatives, year2019, in complete official order. Positive/negative/order digests match the adopted metadata authority. TEST file descriptor is `{query['official_TEST_file']['sha256']}`. These are authenticated receipt checks; no array bytes were independently loaded or hashed.", '', 'Serving is fixed True2, float32, eval batch131072, mean raw logits for the native4 bank, official OGB Hits@50. Authenticated source requires exact restored model/Adam/flags/RNG state, unchanged RNG/model/optimizer after each scorer, complete finite score geometry, strict profile and graph/query custody before final publication. Private per-slot receipt contents were not locally fetched, so this is proof through mandatory source guards and authenticated completion/custody, not raw state reinspection.', '', '## All seed TEST scores', '', 'Scores below are percentages (100×Hits@50). Full original precision is retained in ALL_SEED_SCORES.csv and PAIRED_STATISTICS.json.', '', '| Seed | Native64 | Independent native4 | Private4 | Pooled-after-clamp4 | Native70 | Private−pooled (pp) |', '|---:|---:|---:|---:|---:|---:|---:|']
for seed in range(5):
    rows.append('| ' + str(seed) + ' | ' + ' | '.join(f'{100*scores[a][seed]:.6f}' for a in arms) + f" | {100*primary['paired_TEST_hits50_differences'][seed]:+.6f} |")
rows += ['', '| Arm | Mean (%) | Sample SD (pp) |', '|---|---:|---:|']
for a in arms:
    rows.append(f"| {names[a]} | {100*arm_stats[a]['mean']:.6f} | {100*arm_stats[a]['sample_sd']:.6f} |")
lo, hi = primary['approximate_95_percent_paired_t_interval']
rows += ['', '## Frozen primary private-minus-pooled contrast', '', f"Mean difference **{100*primary['mean']:+.6f} percentage points**, paired sample SD {100*primary['sample_sd']:.6f}pp, paired SE {100*primary['paired_standard_error']:.6f}pp. Three positive differences, two negative, no ties. Range [{100*primary['range'][0]:+.6f}, {100*primary['range'][1]:+.6f}]pp.", '', f"An approximate two-sided95% paired Student t interval (df4, critical2.776445) is **[{100*lo:+.6f}, {100*hi:+.6f}]pp**. This estimates mean training-seed variation conditional on one graph/time split and assumes approximately independent seed differences with suitable mean-distribution behavior. Five pairs cannot validate those assumptions. The interval includes differences in either direction; it is not evidence of equivalence or a guaranteed small effect.", '', f"An exact sign-flip enumeration over all32 sign combinations gives a two-sided tail probability {primary['exact_sign_flip_extreme_combinations']}/32 = **{primary['exact_sign_flip_two_sided_p']:.4f}** for the absolute paired mean. This relies on symmetric/exchangeable paired differences under a zero-centered null; it is not a randomized experiment. A sign-only binomial test for3wins/5 gives two-sided p1.0. No new acceptance threshold is introduced.", '', f"Seed3 contributes the largest positive difference ({100*primary['paired_TEST_hits50_differences'][3]:+.6f}pp); a descriptive leave-seed3-out mean is {100*primary['leave_one_seed_out_means'][3]:+.6f}pp. The frozen estimate always retains all five seeds; this sensitivity is not a replacement subset analysis.", '', 'The development VALID private-minus-pooled mean was+0.843153pp with5/5 positive differences; TEST is+0.223618pp with3/5. VALID selected checkpoints, so its development contrast is not an unbiased heldout estimate. The two graph contexts also differ by the native TRAIN+VALID TEST policy.', '', '## Exploratory baseline contrasts', '', 'Every row uses all five matching seed indices. Intervals are unadjusted, assumption-dependent paired t summaries. These nine comparisons support descriptive comparison, with no multiplicity-adjusted confirmatory superiority conclusion.', '', '| Candidate − control | Mean (pp) | Paired SD (pp) | Approx.95% t interval (pp) | Wins / losses |', '|---|---:|---:|---:|---:|']
for c in exploratory:
    low,high = c['approximate_95_percent_paired_t_interval']
    rows.append(f"| {names[c['candidate']]} − {names[c['control']]} | {100*c['mean']:+.6f} | {100*c['sample_sd']:.6f} | [{100*low:+.6f}, {100*high:+.6f}] | {c['sign_count']['positive']} / {c['sign_count']['negative']} |")
rows += ['', 'Independent native4 has the highest observed mean (67.629778%). Private4 is0.338881pp below it and wins1/5 seed comparisons. Private4 exceeds Native64 in all five paired scores, but that exploratory finding is not a broad baseline-superiority or SOTA claim. Shared training-fit reuse also means arms are correlated; those comparisons are paired descriptions, not independent-arm experiments.', '', '## Resource observations and physical limits', '', f"Root admission observed81,149MiB GPU free of81,920MiB, host MemAvailable123,088,461,824bytes. Required immediate recheck was≥24,576MiB GPU and≥16GiB host; caps were1,800seconds and32GiB sampled owned-session RSS. The locally fetched monitor contains no standalone RESOURCE_DISPATCH_RECHECK.json. The authenticated supervisor requires a passing recheck before claim and spawn; a clean completed child evidences that route, while immediate recheck memory figures are not independently quoted here.", '', f"Observed CUDA peak allocated {result['cuda_peak_allocated_bytes']:,}bytes and reserved {result['cuda_peak_reserved_bytes']:,}bytes; owned-child wait4 RSS peak {physical['owned_child_wait4_peak_RSS_bytes']:,}bytes; sampled owned-session RSS peak {physical['sampled_peak_owned_session_RSS_bytes']:,}bytes. CUDA/host scopes differ. wait4 RSS is not an aggregate simultaneous sum; sampled session RSS may double-count shared pages. No arithmetic sum of those peaks is meaningful.", '', f"Physical wrapper wall {physical['physical_wrapper_wall_seconds']:.6f}s; child spawn through wait4 {physical['child_spawn_through_wait4_wall_seconds']:.6f}s; inclusive child accounting {result['inclusive_wall_seconds_through_accounting']:.6f}s. These overlap and are not added. Terminal write tail is unmeasured. Sentinel QA did not guarantee production memory; the observed serving success is one concrete workload, not a general deployment bound.", '', '## Practical claim limits', '']
rows += ['- ' + x for x in limits]
rows += ['', '## Seal and reproduction', '', 'AUDIT_BUILD.py uses Python standard library only. It reads the listed text/JSON evidence and scalar terminal cells, performs no target execution, and writes only this new audit directory. INPUT_HASH_RECEIPT.json records every accessed/hash-checked source or metadata file with expected pins. EVIDENCE_AUDIT.json contains all consistency checks, exact scores, selections/counts, resource observations and limits. MANIFEST.json seals all audit outputs except itself. Earlier reviews and source/result packets were preserved.', '']
report_text = '\n'.join(rows)
for before, after in [
    ('bytes155656', 'bytes 155656'), ('exit0/rawwait0', 'exit 0 / raw wait 0'),
    ('exit0', 'exit 0'), ('Seed4', 'Seed 4'), ('remain35', 'remain 35'),
    ('seeds0–4:25', 'seeds 0–4: 25'), (';40', '; 40'), (';25', '; 25'),
    ('and25', 'and 25'), ('fits35', 'fits 35'), ('schedule100epochs', 'schedule 100 epochs'),
    ('×17full', '× 17 full'), ('×35fits', '× 35 fits'), ('updates0', 'updates 0'),
    ('TRAIN+VALID:1,179', 'TRAIN+VALID: 1,179'), ('by60,084', 'by 60,084'),
    ('all46,329', 'all 46,329'), ('shared100,000', 'shared 100,000'), ('year2019', 'year 2019'),
    ('batch131072', 'batch 131072'), ('two-sided95%', 'two-sided 95%'),
    ('df4', 'df 4'), ('critical2.776445', 'critical 2.776445'), ('all32', 'all 32'),
    ('for3wins/5', 'for 3 wins / 5'), ('p1.0', 'p 1.0'), ('Seed3', 'Seed 3'),
    ('leave-seed3-out mean', 'mean after omitting seed 3'), ('was+', 'was +'),
    ('with5/5', 'with 5/5'), ('is+', 'is +'), ('with3/5', 'with 3/5'),
    ('Approx.95%', 'Approx. 95%'), ('is0.338881', 'is 0.338881'), ('wins1/5', 'wins 1/5'),
    ('observed81,149', 'observed 81,149'), ('of81,920', 'of 81,920'),
    ('MemAvailable123,088', 'MemAvailable 123,088'), ('was≥', 'was ≥'),
    ('were1,800', 'were 1,800'), ('and32GiB', 'and 32 GiB'),
    ('heldout_gate.py108–115', 'heldout_gate.py:108–115'), ('n5', 'n=5'),
]:
    report_text = report_text.replace(before, after)
report_text = re.sub(r'(\d)(pp|bytes|MiB|GiB|seconds|epochs)\b', r'\1 \2', report_text)
with (OUT / 'REPORT.md').open('x') as stream:
    stream.write(report_text)
write_json('INPUT_HASH_RECEIPT.json', dict(schema='ncnc-all25-independent-heldout-evidence-input-receipt-v1',UTC=datetime.now(timezone.utc).isoformat(),project_root=str(P.parent),input_count=len(inputs),prior_v2_inputs_rehashed=158,all_expected_pins_match=True,inputs=sorted(inputs.values(),key=lambda x:x['local_path']),restrictions_observed=audit['restrictions_observed']))
files = []
for path in sorted(OUT.iterdir()):
    if path.is_file() and path.name != 'MANIFEST.json':
        raw = path.read_bytes()
        files.append(dict(path=path.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
write_json('MANIFEST.json', dict(schema='ncnc-all25-independent-heldout-evidence-output-manifest-v1',UTC=datetime.now(timezone.utc).isoformat(),files=files,existing_packets_preserved=True,no_numerical_rerun=True))
seal = (OUT/'MANIFEST.json').read_bytes()
print(json.dumps(dict(folder=str(OUT),input_count=len(inputs),consistency_checks=len(checks),manifest_sha256=hashlib.sha256(seal).hexdigest(),primary_mean_pp=100*primary['mean'],primary_paired_t95_pp=[100*lo,100*hi],sign_flip_two_sided_p=primary['exact_sign_flip_two_sided_p'])))
