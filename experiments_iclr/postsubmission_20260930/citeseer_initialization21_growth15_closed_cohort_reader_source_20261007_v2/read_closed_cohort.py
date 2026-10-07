"""One-shot server CPU reader of complete, source-selected Citeseer families.

No model imports, checkpoint deserialization, forward pass, dataset payload,
training, TEST, selector changes, process signals, launches, or polling.
The requested whole-family metadata gate precedes all HISTORY/logit access.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
PYTHON = str(PHASE / 'native_ncn_runtime_20261005_v1/.venv/bin/python')
INIT_SOURCE = 'citeseer_gnnm_initialization_pilot_preparation_20261006_v1'
GROWTH_SOURCE = 'citeseer_nonlinear_preaggregation_growth_pilot_source_20261007_v2'
INIT_ROOT = 'citeseer_gnnm_initialization_pilot_execution_root_20261006_v2'
INIT_REMAINING = 'citeseer_initialization_remaining_flat_execution_root_20261007_v1'
GROWTH_ROOT = 'citeseer_growth_full15_flat_execution_root_20261007_v1'
OUTPUTS = {
    ('initialization',): 'citeseer_initialization21_closed_cohort_readout_root_20261007_v2',
    ('growth',): 'citeseer_growth15_closed_cohort_readout_root_20261007_v2',
    ('initialization', 'growth'): 'citeseer_initialization21_growth15_combined36_closed_cohort_readout_root_20261007_v2'}
INIT_SHA = 'b319ab6b2d523aeee60858c16fe1b5d97a71861fd135fc5cc013526d630d071b'
GROWTH_SHA = '718887b7360583b6c9032ae433b27899de1cf7760a8687348be500a1b353da46'
INITIALIZER_SHA = '64c3a064336d523e1dabd968e83a4938c0e5cfaf263cb4e3b6536ceb36adc7bb'
SHARED_MANIFEST_SHA = 'db7102df30491be8809ea4295b9ce0d5c48f3608129f09dcfef74b8f7aa2233f'
GROWTH_MANIFEST_SHA = '22d9ae9f53daeb5f47cc17a732f98ccd58ff75a239b95b8322eb64be13b0b7b5'
AVAILABLE = {'train_pos.txt', 'gnn_feature', 'valid_pos.txt', 'heart_valid_samples.npy'}
ACQUISITION = {'path': 'citeseer_heart_official_acquisition_server_20261005_v1/AVAILABLE_MANIFEST.json',
               'sha256': '1b9c8bb57278d91b0f6212136225afcfd6b067c6b17e0dfed7ee36dd6316efdc'}
CONDITIONS = {
    'initialization': ('random_signs_all', 'tabm_first_normal', 'warm_identity', 'graph_covariance',
                       'feature_covariance', 'single', 'independent_warm4'),
    'growth': ('graph_growth', 'unfiltered_growth', 'unfiltered_top8_partition',
               'capable_single_rank8', 'independent_graph_growth4')}
CONTRASTS = {
    'initialization': [('graph_covariance', c) for c in
                       ('feature_covariance', 'warm_identity', 'tabm_first_normal',
                        'random_signs_all', 'single', 'independent_warm4')],
    'growth': [('graph_growth', c) for c in
               ('unfiltered_growth', 'unfiltered_top8_partition',
                'capable_single_rank8', 'independent_graph_growth4')]}
OWNERS = {
    'b0': (INIT_ROOT + '/b0_fit_owner', 'OWNER_STARTED.json', 491048, 6012817974,
           {'path': INIT_ROOT + '/B0_BLOCK_OWNER_CONFIG.json',
            'sha256': 'e5513450f6a46609438dab6ae4128571bf97b7ca354e159d22a2fd6680be55b7'},
           'citeseer_gnnm_initialization_pilot_block_queue_preparation_20261006_v1/block_owner.py'),
    'initialization': (INIT_REMAINING + '/owner', 'START.json', 495028, 6013629570,
                       {'path': 'citeseer_initialization_remaining_flat_activation_root_20261007_v1/CONFIG_ACTIVATED.json',
                        'sha256': '9dd4a045d1d5f077f9aae91833d5931938a1a82c1f1a31c26c3049fed65e7cbe'},
                       'citeseer_initialization_remaining_flat_preparation_20261007_v1/execute.py'),
    'growth': (GROWTH_ROOT + '/owner', 'START.json', 495546, 6013797508,
               {'path': 'citeseer_growth_full15_flat_activation_root_20261007_v1/CONFIG_ACTIVATED.json',
                'sha256': '33d2f986c631f413c7c815ec19c077f34f613c7a0acef33feff025fcde2cc145'},
               'citeseer_growth_full15_flat_preparation_20261007_v1/execute.py')}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file(relative):
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Phase-relative path required')
    path = (PHASE / rel).resolve(strict=True)
    require(path.is_relative_to(PHASE.resolve()) and path.is_file(), 'File leaves phase')
    return path


def bound(record):
    path = file(record['path'])
    require(sha(path) == record['sha256'], 'Bound bytes changed: ' + record['path'])
    if 'bytes' in record:
        require(path.stat().st_size == record['bytes'], 'Bound size changed')
    return path


def read(path):
    return json.loads(Path(path).read_text())


def receipt(path):
    return {'path': str(path.relative_to(PHASE)), 'sha256': sha(path)}


def absent(pid, cuda_pids):
    require(type(pid) is int and pid > 0, 'Positive actual PID required')
    # Same strict absence rule as the existing owners: PID reuse also refuses.
    require(not (Path('/proc') / str(pid)).exists() and pid not in cuda_pids,
            'Original owner or child is not physically absent')


def clean_terminal(value, entry, cuda_pids):
    child = value.get('raw_identity_observation')
    require(value.get('exit_code') == 0 and value.get('reason') is None
            and not value.get('signals_sent') and value.get('terminal_wait_observed') is True
            and value.get('job_sha256') == entry['job_sha256'] and child is not None
            and child['argv'] == entry['argv'] and value.get('child_identity') == child
            and value.get('attempts') == 1 and value.get('retry') is False,
            'Existing owner clean terminal contract differs')
    require(child['start_ticks'] > 0 and child['pgid'] == child['PID'] == child['sid'],
            'Scientific child identity differs')
    absent(child['PID'], cuda_pids)


def source_manifest(record):
    path = bound(record)
    value = read(path)
    for row in value['files']:
        rel = str((path.parent / row['path']).relative_to(PHASE))
        bound(dict(row, path=rel))
    for rel, digest in value.get('dependencies', {}).items():
        dependency = {'path': rel, 'sha256': digest}
        if rel.endswith('/SOURCE_MANIFEST.json'):
            source_manifest(dependency)
        else:
            bound(dependency)


def owner(name, cuda_pids):
    relative, started_name, pid, ticks, config_record, adapter = OWNERS[name]
    work = PHASE / relative
    absent(pid, cuda_pids)
    require(not (work / 'FAILURE.json').exists() and not (work / 'QUEUE_FAILURE.json').exists(),
            'Requested owner failure remains')
    cfg = read(bound(config_record))
    bound({'path': adapter, 'sha256': cfg['adapter_sha256']})
    started_path = file(relative + '/' + started_name)
    started = read(started_path)
    ident = started['identity']
    expected_argv = ['/usr/bin/python3', '-I', '-S', '-B', str(PHASE / adapter),
                     '--config', str(PHASE / config_record['path']),
                     '--config-sha256', config_record['sha256']]
    require((ident['PID'], ident['start_ticks'], ident['pgid'], ident['sid']) == (pid, ticks, pid, pid)
            and ident['argv'] == expected_argv and started['config_sha256'] == config_record['sha256'],
            'Actual owner start/config identity differs')
    require(cfg['TEST_access'] is False and cfg['retry'] is False, 'Owner TEST/retry authority differs')
    for row in cfg.get('bindings', []):
        bound(row)
    for row in cfg.get('templates', cfg.get('job_templates', [])):
        bound(row)
    plan = read(bound(cfg['plan']))
    family = 'growth' if name == 'growth' else 'initialization'
    require(plan.get('root_adopted') is True and plan['TEST_closed'] is True
            and tuple(plan['conditions']) == CONDITIONS[family]
            and (plan['warm_cycles'], plan['post_cycles'], plan['eval_every']) == (20, 60, 5)
            and [b['seed'] for b in plan['blocks']] == [0, 1, 2], 'Fixed scientific plan differs')
    if name != 'b0':
        require(started['warm1_process_exit_status'] == 'unknown'
                and cfg['warm1_custody_amendment']['approved'] is True
                and cfg['warm1_custody_amendment']['process_exit_status'] == 'unknown',
                'Preserved warm1 unknown-exit amendment differs')
    closure_path = file(relative + ('/BLOCK_FREEZE.json' if name == 'b0' else '/COMPLETE.json'))
    closure = read(closure_path)
    require(closure['TEST_access'] is False and closure['retry'] is False
            and closure['scores_read'] is False, 'Operational closure authority differs')
    if name == 'b0':
        require(closure['block'] == 'b0' and closure['selected_block_complete'] is True
                and closure['physical_postfits'] == 7 and closure['plan'] == cfg['plan'],
                'Preserved b0 actual block completion differs')
        seeds = (0,)
    elif name == 'initialization':
        require(closure['fits'] == 14 and closure['new_warm_fits'] == 0
                and closure['b0_retained'] is True and closure['warm1_process_exit_status'] == 'unknown',
                'Remaining14 actual completion differs')
        seeds = (1, 2)
    else:
        require(closure['fits'] == 15 and closure['new_warm_or_qualifications'] == 0,
                'Growth15 actual completion differs')
        seeds = (0, 1, 2)
    expected = ['b' + str(s) + '_' + c for s in seeds for c in CONDITIONS[family]]
    jobs_path = file(relative + '/BOUND_JOBS.json')
    entries = read(jobs_path)['entries']
    require([e['cell_id'] for e in entries] == expected
            and [e['cell_id'] for e in closure['completed']] == expected,
            'Whole requested roster/actual completion differs')
    return cfg, plan, entries, closure, {'owner': name, 'PID': pid, 'start_ticks': ticks,
        'absent_no_CUDA': True, 'START': receipt(started_path), 'config': config_record,
        'closure': receipt(closure_path), 'BOUND_JOBS': receipt(jobs_path)}


def warm(job, seed, expected_train, cuda_pids):
    path = bound({'path': job['warm_freeze_relative'], 'sha256': job['warm_freeze_sha256']})
    value = read(path)
    require((value['phase'], value['seed'], value['completed_cycles']) == ('warm', seed, 20)
            and value['VALID_TEST_access'] is False and value['TEST_access'] is False
            and value['native_alias_restored'] is True and value['retry'] is False
            and value['selected_cycle'] is None and value['selected_VALID_MRR'] is None
            and value['counters'] == {'episodes': 1220, 'native_Adam_updates': 3660}
            and len(value['cycle_seconds']) == 20 and value['inputs'] == expected_train
            and value['runtime'] == job['runtime_versions'] and not (path.parent / 'FAILURE.json').exists(),
            'Complete bound TRAIN-only donor metadata differs')
    warm_job_path = file(INIT_ROOT + '/jobs/b' + str(seed) + '_warm.json')
    warm_job = read(warm_job_path)
    require(sha(warm_job_path) == value['job_sha256'] and warm_job['phase'] == 'warm'
            and warm_job['seed'] == seed and warm_job['factor_seed'] == job['factor_seed']
            and warm_job['program_sha256'] == INIT_SHA and warm_job['initialization_sha256'] == INITIALIZER_SHA
            and warm_job['plan_sha256'] == value['plan_sha256']
            and warm_job['TEST_access'] is False and warm_job['VALID_values_access'] is False,
            'Warm source/job/plan custody differs')
    require(value['plan_sha256'] == '1e50c1bb7657499bf9e6adf1f983d29aba2b09b30578b70f43357f59eadbe7a2',
            'Original warm plan changed')
    if seed == 1:
        absent(493938, cuda_pids)
        require(value['checkpoint_sha256'] == '9f9114c5f8ada4fad999fb7a083fb09a89174de7b9e9b808ad381914772a3485'
                and sha(path) == '182cd68d9a992041e4aa2f5c2da47ee91b5f1eba6f30c8676d3498806da229a7',
                'Admitted warm1 orphan artifact changed')
        terminal_authority = {'process_exit_status': 'unknown',
                              'artifact_completion_authority': 'Root admitted source terminal20 orphan artifacts'}
    else:
        warm_cfg_record = ({'path': INIT_ROOT + '/WARM_OWNER_CONFIG.json',
                            'sha256': '69015875e10bba9d314261e929836f037b66fcd8ff73433cc91b9bc761272276'}
                           if seed == 0 else
                           {'path': 'allocation_initializer_warm2_direct_activation_root_20261007_v1/ROOT_WARM2_CONFIG.json',
                            'sha256': '48d2c60a09f1e14c25c1c4eab67aaa98b306400ce27a026aae13751a5b5bff8d'})
        warm_cfg = read(bound(warm_cfg_record))
        pid, ticks = (490954, 6012736369) if seed == 0 else (494276, 6013552180)
        absent(pid, cuda_pids)
        work = INIT_ROOT + '/warm_b' + str(seed) + '_owner'
        identity = read(file(work + '/OWNER_STARTED.json'))['identity']
        require((identity['PID'], identity['start_ticks'], identity['pgid'], identity['sid']) == (pid, ticks, pid, pid),
                'Actual donor owner identity differs')
        argv = [PYTHON, '-B', str(PHASE / INIT_SOURCE / 'run.py'), '--job', str(warm_job_path),
                '--output', str(path.parent)]
        require(warm_cfg['argv'] == argv and warm_cfg['job_sha256'] == value['job_sha256'],
                'Actual donor child source/job argv differs')
        terminal_path = file(work + '/TERMINAL.json')
        clean_terminal(read(terminal_path), {'argv': argv, 'job_sha256': value['job_sha256']}, cuda_pids)
        complete_path = file(work + '/WARM_COMPLETE.json'); complete = read(complete_path)
        require(complete['freeze_sha256'] == sha(path) and complete['checkpoint_sha256'] == value['checkpoint_sha256']
                and complete['TEST_closed'] is True and complete['child_absent'] is True,
                'Actual donor terminal20 completion differs')
        terminal_authority = {'process_exit_status': 'clean_exit0', 'TERMINAL': receipt(terminal_path),
                              'WARM_COMPLETE': receipt(complete_path)}
    return {'seed': seed, 'FREEZE': receipt(path), 'inclusive_seconds': value['inclusive_seconds'],
            'cycle_seconds': value['cycle_seconds'], 'checkpoint_sha256': value['checkpoint_sha256'],
            **terminal_authority}


def preflight(families):
    """Metadata only. Never hash/open HISTORY, scores, or any state/checkpoint here."""
    cuda_lines = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines()
    cuda_pids = {int(row.split(',')[1].strip()) for row in cuda_lines if row.strip()}
    manifest = read(bound(ACQUISITION))
    require(set(manifest['files']) == AVAILABLE and manifest['TEST_available_to_loader'] is False,
            'Exactly retained TRAIN/VALID acquisition metadata required')
    identities = {name: {k: manifest['files'][name][k] for k in ('sha256', 'bytes')} for name in AVAILABLE}
    for name in AVAILABLE:
        require(tuple(Path(manifest['files'][name]['relative_path']).parts) == ('available', 'citeseer', name),
                'Input role/path identity differs')
    expected_train = {k: identities[k] for k in ('train_pos.txt', 'gnn_feature')}
    bound({'path': INIT_SOURCE + '/run.py', 'sha256': INIT_SHA})
    bound({'path': INIT_SOURCE + '/initialize.py', 'sha256': INITIALIZER_SHA})
    source_manifest({'path': 'shared_backbone_private_transfer_training_source_20261005_v2/SOURCE_MANIFEST.json',
                     'sha256': SHARED_MANIFEST_SHA})
    if 'growth' in families:
        source_manifest({'path': GROWTH_SOURCE + '/SOURCE_MANIFEST.json', 'sha256': GROWTH_MANIFEST_SHA})
        for row in read(file(GROWTH_SOURCE + '/INPUT_BINDINGS.json'))['files']:
            bound(row)
    metadata, states, donors, paid = [], [], {}, {'qualifications': [], 'cost_runs': []}
    # All owner COMPLETE/absence/rosters are checked before any per-cell payload.
    names = (['b0', 'initialization'] if 'initialization' in families else []) + (['growth'] if 'growth' in families else [])
    owners = [(name, owner(name, cuda_pids)) for name in names]
    for name, (cfg, plan, entries, closure, observation) in owners:
        metadata.append(observation)
        family = 'growth' if name == 'growth' else 'initialization'
        root = GROWTH_ROOT if family == 'growth' else INIT_ROOT if name == 'b0' else INIT_REMAINING
        source = GROWTH_SOURCE if family == 'growth' else INIT_SOURCE
        for entry, item in zip(entries, closure['completed']):
            cell = entry['cell_id']; seed = int(cell[1]); condition = cell[3:]
            job_path = bound({'path': entry['job_relative'], 'sha256': entry['job_sha256']})
            require(entry['job_relative'] == root + '/jobs/' + cell + '.json', 'Exact actual job path differs')
            folder = PHASE / root / 'runs' / cell
            expected_argv = [PYTHON, '-B', str(PHASE / source / 'run.py'), '--job', str(job_path), '--output', str(folder)]
            require(entry['argv'] == expected_argv, 'Pinned source/executable/output argv differs')
            terminal = item if name == 'b0' else item['terminal']
            clean_terminal(terminal, entry, cuda_pids)
            exit_path = file(str((PHASE / OWNERS[name][0] / 'logs' / (cell + '.EXIT.json')).relative_to(PHASE)))
            actual_exit = read(exit_path)
            require(all(terminal.get(k) == v for k, v in actual_exit.items()), 'Embedded terminal/actual EXIT differs')
            clean_terminal(actual_exit, entry, cuda_pids)
            frozen_record = ({'path': item['freeze_relative'], 'sha256': item['freeze_sha256']}
                             if name == 'b0' else item['FREEZE'])
            require(frozen_record['path'] == str((folder / 'FREEZE.json').relative_to(PHASE)), 'Frozen role path differs')
            freeze_path = bound(frozen_record); freeze = read(freeze_path); job = read(job_path)
            require(not (folder / 'FAILURE.json').exists() and job['TEST_access'] is False and job['retry'] is False
                    and job['fits_authorized'] is True and job['VALID_values_access'] is True
                    and job['source_review_approved'] is True and job['external_hard_bound_confirmed'] is True
                    and (job['phase'], job['seed'], job['condition'], job['repository'], job['physical_gpu_uuid'])
                        == ('fit', seed, condition, str(REPO), GPU)
                    and job['output_directory'] == str(folder) and job['plan_relative'] == cfg['plan']['path']
                    and job['plan_sha256'] == cfg['plan']['sha256']
                    and job['factor_seed'] == next(b['factor_seed'] for b in plan['blocks'] if b['seed'] == seed)
                    and job['source_manifest_sha256'] == SHARED_MANIFEST_SHA
                    and job['program_sha256'] == (GROWTH_SHA if family == 'growth' else INIT_SHA)
                    and job['available_manifest_relative'] == ACQUISITION['path']
                    and job['available_manifest_sha256'] == ACQUISITION['sha256'], 'Actual source/job/plan/input custody differs')
            require((freeze['phase'], freeze['seed'], freeze['condition']) == ('fit', seed, condition)
                    and freeze['TEST_access'] is False and freeze['VALID_TEST_access'] is True and freeze['retry'] is False
                    and freeze['inputs'] == identities and freeze['runtime'] == job['runtime_versions']
                    and len(freeze['cycle_seconds']) == 60 and freeze['selected_cycle'] in range(5, 61, 5),
                    'Actual60 endpoint/selector metadata differs')
            independent = condition in ('independent_warm4', 'independent_graph_growth4')
            episodes = 14640 if independent else 3660
            if family == 'initialization':
                require(freeze['completed_cycles'] == 60 and freeze['job_sha256'] == entry['job_sha256']
                        and freeze['plan_sha256'] == cfg['plan']['sha256'] and freeze['native_alias_restored'] is True
                        and job['initialization_sha256'] == INITIALIZER_SHA
                        and freeze['counters'] == {'episodes': episodes, 'native_Adam_updates': 3 * episodes},
                        'Initialization full60 counter/source endpoint differs')
            else:
                require(freeze['complete_cycles'] == 60 and freeze['success'] is True
                        and freeze['growth_source_manifest_sha256'] == job['growth_source_manifest_sha256'] == GROWTH_MANIFEST_SHA
                        and freeze['warm_freeze_sha256'] == job['warm_freeze_sha256']
                        and freeze['counters'] == {'episodes': episodes, 'Adam_updates': 3 * episodes,
                                                   'complete_train_cycles_per_member': 60}
                        and freeze['calibration_and_basis_generation_paid'] is True
                        and freeze['warm20_cost_must_be_charged_from_bound_warm_receipt'] is True,
                        'Growth full60 counter/source/paid endpoint differs')
                for key, phase in [('qualification_receipt', 'qualify'), ('complete_cycle_cost_receipt', 'cost')]:
                    paid_path = bound(job[key]); paid_value = read(paid_path)
                    require(paid_value['phase'] == phase and paid_value['success'] is True
                            and paid_value['VALID_TEST_access'] is False
                            # Actual qualify FREEZE omits TEST_access; its pinned source/job forbids TEST.
                            and paid_value.get('TEST_access', False) is False
                            and paid_value['growth_source_manifest_sha256'] == GROWTH_MANIFEST_SHA,
                            'Paid qualification/cost custody differs')
                    require(paid_value['seed'] == seed if phase == 'qualify' else
                            paid_value['condition'] == condition and paid_value['complete_cycles'] == 1,
                            'Same-seed qualification or same-arm full-cycle cost required')
            donor = warm(job, seed, expected_train, cuda_pids)
            if seed in donors:
                require(donors[seed] == donor, 'Same-seed common donor differs')
            donors[seed] = donor
            states.append({'family': family, 'seed': seed, 'condition': condition, 'cell_id': cell,
                           'folder': folder, 'freeze': freeze, 'job': job, 'FREEZE': frozen_record,
                           'job_binding': receipt(job_path), 'terminal_binding': receipt(exit_path)})
        if family == 'growth':
            # Reuse the same retained qualification/cost closure checks as execute.py.
            admission_records = [*cfg['qualification_owner_receipts'], cfg['cost_owner_complete']]
            for admission_record in admission_records:
                admission = read(bound(admission_record))
                terminals = ([admission['terminal']] if 'terminal' in admission else
                             [r['terminal'] for r in admission['completed']])
                for terminal in terminals:
                    argv = terminal['raw_identity_observation']['argv']
                    require(len(argv) == 7 and argv[:3] == [PYTHON, '-B', str(PHASE / GROWTH_SOURCE / 'run.py')]
                            and argv[3] == '--job' and argv[5] == '--output', 'Paid child scientific argv differs')
                    job_rel = str(Path(argv[4]).relative_to(PHASE))
                    paid_job = read(bound({'path': job_rel, 'sha256': terminal['job_sha256']}))
                    require(paid_job['phase'] in ('qualify', 'cost') and paid_job['TEST_access'] is False
                            and paid_job['VALID_values_access'] is False
                            and paid_job['growth_source_manifest_sha256'] == GROWTH_MANIFEST_SHA,
                            'Paid TRAIN-only child custody differs')
                    clean_terminal(terminal, {'argv': argv, 'job_sha256': terminal['job_sha256']}, cuda_pids)
            require(len(read(bound(cfg['cost_owner_complete']))['completed']) == 5, 'All5 paid cost closures required')
            for row in cfg['qualifications']:
                qualified = read(bound(row['FREEZE'])); gates = read(bound(row['NATIVE_GATES']))
                require(qualified['seed'] == row['seed'] and qualified['optimizer_updates'] == 0
                        and qualified['gates_sha256'] == row['NATIVE_GATES']['sha256']
                        and set(gates) == set(CONDITIONS['growth']) | {'no_growth'}
                        and all(g['copied_native_forward_gradient_gate_passed'] is True for g in gates.values()),
                        'Actual all-six same-seed qualification gates differ')
            for key, target in [('qualifications', 'qualifications'), ('costs', 'cost_runs')]:
                for row in cfg[key]:
                    p = bound(row['FREEZE']); value = read(p)
                    paid[target].append({'FREEZE': receipt(p), 'seed': value['seed'],
                        'condition': value.get('condition'), 'inclusive_seconds': value['inclusive_seconds'],
                        'calibration_seconds': value.get('calibration_seconds'), 'cycle_seconds': value.get('cycle_seconds')})
    for family in families:
        require([(s['seed'], s['condition']) for s in states if s['family'] == family]
                == [(seed, c) for seed in (0, 1, 2) for c in CONDITIONS[family]], 'Complete requested roster gate failed')
    return states, {'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'families': families,
        'whole_requested_roster_complete_before_predictive_open': True, 'owner_observations': metadata,
        'actual60_endpoint_count': len(states), 'input_identities': identities, 'acquisition_manifest': ACQUISITION,
        'TEST_access': False, 'predictive_files_opened_during_gate': False,
        'warm1_process_exit_status': 'unknown', 'donors': [donors[s] for s in (0, 1, 2)], 'paid': paid}


def metric(rank):
    return {'MRR': float((1 / rank.float()).mean().item()),
            'Hits10': float((rank <= 10).float().mean().item())}


def valid_metric(value):
    require(set(value) == {'MRR', 'Hits10'} and all(type(x) in (int, float) and math.isfinite(x)
            and 0 <= x <= 1 and round(x, 4) == x for x in value.values()), 'Source rounded4 metric differs')


def normalize(torch, state, gate):
    folder = state['folder']; freeze = state['freeze']
    history_path = file(str((folder / 'HISTORY.jsonl').relative_to(PHASE)))
    require(sha(history_path) == freeze['provenance_sha256']['HISTORY.jsonl'], 'Closed HISTORY bytes changed')
    history = [json.loads(line) for line in history_path.read_text().splitlines()]
    require([r['cycle'] for r in history] == list(range(1, 61)), 'Exactly60 complete HISTORY rows required')
    member_count = 1 if state['condition'] in ('single', 'capable_single_rank8') else 4
    exposures = []
    for row in history:
        require(row['cycle_seconds'] == freeze['cycle_seconds'][row['cycle'] - 1]
                and math.isfinite(row['cycle_seconds']) and row['cycle_seconds'] > 0, 'Cycle cost endpoint differs')
        episodes = row['cycle'] * (244 if state['condition'] in ('independent_warm4', 'independent_graph_growth4') else 61)
        expected_counters = ({'episodes': episodes, 'native_Adam_updates': 3 * episodes}
                             if state['family'] == 'initialization' else
                             {'episodes': episodes, 'Adam_updates': 3 * episodes,
                              'complete_train_cycles_per_member': row['cycle']})
        require(row['counters'] == expected_counters, 'Actual complete-cycle HISTORY counters differ')
        require(('complete_VALID' in row) == (row['cycle'] % 5 == 0)
                and ('members' in row) == (row['cycle'] % 5 == 0), 'Exactly12 source exposures required')
        if 'complete_VALID' in row:
            valid_metric(row['complete_VALID'])
            require(len(row['members']) == member_count, 'All same-bank members required')
            for value in row['members']:
                valid_metric(value)
            exposures.append({'cycle': row['cycle'], 'pooled': row['complete_VALID'], 'members': row['members']})
    require(history[-1]['counters'] == freeze['counters'], 'HISTORY actual terminal counters differ')
    # Source rule: strict improvement of the already rounded4 pooled MRR.
    best = max(e['pooled']['MRR'] for e in exposures)
    selected = next(e for e in exposures if e['pooled']['MRR'] == best)
    require(selected['cycle'] == freeze['selected_cycle'] and best == freeze['selected_VALID_MRR'],
            'Frozen checkpoint is not source first strict rounded4 maximum')
    checkpoint = file(str((folder / 'selected_checkpoint.pt').relative_to(PHASE)))
    checkpoint_hash = sha(checkpoint)  # Bytes only; never deserialize state.
    if state['family'] == 'initialization':
        require(checkpoint_hash == freeze['checkpoint_sha256'], 'Selected checkpoint hash differs')
    logits_path = file(str((folder / 'selected_VALID_logits.pt').relative_to(PHASE)))
    bank = torch.load(logits_path, map_location='cpu', weights_only=True)
    require(set(bank) == {'member_pos', 'member_neg', 'mean_pos', 'mean_neg', 'inputs',
                          'selected_cycle', 'checkpoint_sha256'}
            and bank['inputs'] == gate['input_identities'] == freeze['inputs']
            and bank['selected_cycle'] == selected['cycle'] and bank['checkpoint_sha256'] == checkpoint_hash,
            'Existing selected VALID bank metadata differs')
    shapes = {'member_pos': (227, member_count), 'member_neg': (227, 500, member_count),
              'mean_pos': (227,), 'mean_neg': (227, 500)}
    for name, shape in shapes.items():
        value = bank[name]
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.layout == torch.strided
                and value.dtype == torch.float32 and tuple(value.shape) == shape and bool(torch.isfinite(value).all()),
                'Finite original CPU score tensor shape/dtype differs')
    # Use saved served means. A CPU member reduction could change ties.
    p, n = bank['mean_pos'], bank['mean_neg']
    ranks = 1 + .5 * ((n >= p[:, None]).sum(1) + (n > p[:, None]).sum(1))
    p, n = bank['member_pos'], bank['member_neg']
    member_ranks = 1 + .5 * ((n >= p[:, None, :]).sum(1) + (n > p[:, None, :]).sum(1))
    cpu_metrics = [metric(ranks)] + [metric(member_ranks[:, m]) for m in range(member_count)]
    source_metrics = [selected['pooled']] + selected['members']
    for cpu, source in zip(cpu_metrics, source_metrics):
        require(all(abs(cpu[k] - source[k]) <= .000051 for k in ('MRR', 'Hits10')),
                'Stored bank does not match selected source metrics within rounding4 tolerance')
    all_wrong = (member_ranks > 1).all(1)
    shared_slots = (n >= p[:, None, :]).all(2)
    require(bool((~shared_slots.any(1) | all_wrong).all()), 'Shared-slot competitor must imply all-member error')
    donor = next(d for d in gate['donors'] if d['seed'] == state['seed'])
    record = {k: state[k] for k in ('family', 'seed', 'condition', 'cell_id', 'FREEZE', 'job_binding', 'terminal_binding')}
    record.update(completed_cycles=60, selected_cycle=selected['cycle'], MRR=best,
        Hits10=selected['pooled']['Hits10'], member_MRR=[v['MRR'] for v in selected['members']],
        member_Hits10=[v['Hits10'] for v in selected['members']], CPU_raw_metrics=cpu_metrics,
        all12_VALID_exposures=exposures, rounded4_maximum_tie_cycles=[e['cycle'] for e in exposures if e['pooled']['MRR'] == best],
        HISTORY=receipt(history_path), selected_VALID_logits=receipt(logits_path), selected_checkpoint_sha256=checkpoint_hash,
        logit_hash_authority='Hash recorded by this closed reader; saved packet is bound to actual checkpoint bytes',
        common_top1_error_queries=int(all_wrong.sum()), common_negative_slot_queries=int(shared_slots.any(1).sum()),
        pool_top1_correct_on_all_member_wrong=int((all_wrong & (ranks == 1)).sum()),
        costs={'warm20_inclusive_seconds': donor['inclusive_seconds'], 'fit60_inclusive_seconds': freeze['inclusive_seconds'],
               'standalone_warm_plus_fit_seconds': donor['inclusive_seconds'] + freeze['inclusive_seconds'],
               'all60_cycle_seconds': freeze['cycle_seconds'], 'calibration_seconds': freeze.get('calibration_seconds'),
               'calibration_included_in_fit_inclusive_seconds': state['family'] == 'growth',
               'peak_CUDA_allocated_bytes': freeze['peak_CUDA_allocated_bytes'],
               'peak_CUDA_reserved_bytes': freeze['peak_CUDA_reserved_bytes'],
               'peak_RSS_bytes': freeze.get('peak_RSS_bytes'), 'parameter_count': freeze.get('parameter_count')})
    queries = [{'query_index0': q, 'pool_rank': float(ranks[q]), 'member_ranks': member_ranks[q].tolist(),
                'all_members_top1_wrong': bool(all_wrong[q]),
                'common_negative_slot_indices0': shared_slots[q].nonzero().flatten().tolist()}
               for q in range(227)]
    return record, queries


def describe(values):
    # Reuses the existing paired_summary.py descriptive df2 calculation.
    mean = statistics.mean(values); sd = statistics.stdev(values)
    half = 4.302652729911275 * sd / math.sqrt(3)
    return {'values_in_seed_order_0_1_2': values, 'mean': mean, 'sample_SD': sd,
            'range': [min(values), max(values)], 'positive_seeds': sum(v > 0 for v in values),
            'negative_seeds': sum(v < 0 for v in values), 'zero_seeds': sum(v == 0 for v in values),
            'exploratory_t95_interval_df2': [mean - half, mean + half]}


def summarize(records, families):
    keyed = {(r['family'], r['seed'], r['condition']): r for r in records}
    result = {}
    for family in families:
        conditions, contrasts = {}, {}
        for condition in CONDITIONS[family]:
            rows = [keyed[family, s, condition] for s in (0, 1, 2)]
            conditions[condition] = {k: describe([r[k] for r in rows]) for k in ('MRR', 'Hits10')}
            for k in ('MRR', 'Hits10'):
                conditions[condition]['mean_member_' + k] = describe([statistics.mean(r['member_' + k]) for r in rows])
                conditions[condition]['minimum_member_' + k] = describe([min(r['member_' + k]) for r in rows])
                conditions[condition]['pool_minus_mean_member_' + k] = describe([r[k] - statistics.mean(r['member_' + k]) for r in rows])
            conditions[condition]['selected_cycles'] = [r['selected_cycle'] for r in rows]
        for a, b in CONTRASTS[family]:
            contrasts[a + ' minus ' + b] = {k: describe([keyed[family, s, a][k] - keyed[family, s, b][k]
                                                         for s in (0, 1, 2)]) for k in ('MRR', 'Hits10')}
        result[family] = {'conditions': conditions, 'paired_contrasts': contrasts}
    if set(families) == set(CONDITIONS):
        result['growth_vs_retained_warm_identity'] = {k: describe([
            keyed['growth', s, 'graph_growth'][k] - keyed['initialization', s, 'warm_identity'][k]
            for s in (0, 1, 2)]) for k in ('MRR', 'Hits10')}
    return result


def flows(query_records, families):
    comparisons = [(f, a, f, b) for f in families for a, b in CONTRASTS[f]]
    if set(families) == set(CONDITIONS):
        comparisons.append(('growth', 'graph_growth', 'initialization', 'warm_identity'))
    result, csv_rows = [], []
    for af, a, bf, b in comparisons:
        per_seed = []
        for seed in (0, 1, 2):
            candidate = query_records[af, seed, a]; baseline = query_records[bf, seed, b]
            cohort_indices = {'full227': list(range(227)),
                'baseline_all_members_top1_wrong': [q for q in range(227) if baseline[q]['all_members_top1_wrong']],
                'baseline_common_negative_slot': [q for q in range(227) if baseline[q]['common_negative_slot_indices0']]}
            groups = {}
            for cohort, indices in cohort_indices.items():
                counts = {k: 0 for k in ('wrong_to_correct', 'correct_to_wrong', 'wrong_to_wrong', 'correct_to_correct')}
                rr = []
                for q in indices:
                    br, ar = baseline[q]['pool_rank'], candidate[q]['pool_rank']
                    transition = ('wrong' if br > 1 else 'correct') + '_to_' + ('wrong' if ar > 1 else 'correct')
                    counts[transition] += 1; rr.append(1 / ar - 1 / br)
                groups[cohort] = {'query_count': len(indices), 'query_indices0': indices, **counts,
                    'net_top1_repairs': counts['wrong_to_correct'] - counts['correct_to_wrong'],
                    'paired_RR_delta_sum': sum(rr), 'paired_RR_delta_mean': statistics.mean(rr) if rr else None,
                    'RR_improved_queries': sum(v > 0 for v in rr), 'RR_worsened_queries': sum(v < 0 for v in rr),
                    'RR_unchanged_queries': sum(v == 0 for v in rr)}
            for q in range(227):
                br, ar = baseline[q]['pool_rank'], candidate[q]['pool_rank']
                csv_rows.append({'candidate_family': af, 'candidate_condition': a, 'baseline_family': bf,
                    'baseline_condition': b, 'seed': seed, 'query_index0': q, 'baseline_pool_rank': br,
                    'candidate_pool_rank': ar, 'paired_RR_delta': 1 / ar - 1 / br,
                    'baseline_all_members_top1_wrong': baseline[q]['all_members_top1_wrong'],
                    'baseline_common_negative_slot': bool(baseline[q]['common_negative_slot_indices0']),
                    'wrong_to_correct': br > 1 and ar == 1, 'correct_to_wrong': br == 1 and ar > 1})
            per_seed.append({'seed': seed, 'cohorts': groups})
        result.append({'candidate': {'family': af, 'condition': a}, 'baseline': {'family': bf, 'condition': b},
            'per_seed': per_seed, 'equal_seed_full227_RR_delta': describe([
                r['cohorts']['full227']['paired_RR_delta_mean'] for r in per_seed]),
            'net_top1_repairs_by_seed': [r['cohorts']['full227']['net_top1_repairs'] for r in per_seed]})
    return result, csv_rows


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args()
    require(sha(args.config) == args.config_sha256, 'Reader config changed')
    cfg = read(args.config)
    require(cfg.get('read_enabled') is True and cfg.get('VALID_values_access') is True
            and cfg.get('TEST_access') is False and cfg.get('reader_sha256') == sha(__file__),
            'Explicit exact reader activation required; default disabled')
    families = cfg['requested_families']
    require(families in (['initialization'], ['growth'], ['initialization', 'growth']), 'Whole requested families only')
    output_relative = OUTPUTS[tuple(families)]
    require(cfg.get('output_relative') == output_relative, 'Fixed requested-family output mapping required')
    # Physical singleton route, verified before chdir; no GPU numerical context.
    require(socket.gethostname() == 'anogena-2-0', 'Server-only exact host required')
    require(subprocess.check_output(['/usr/bin/nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                    text=True, timeout=10).splitlines() == [GPU], 'Sole physical GPU identity changed')
    os.chdir(REPO)
    require(Path.cwd().resolve() == REPO, 'Server repository changed')
    output = PHASE / output_relative
    require(not output.exists(), 'Fresh readout only; no overwrite or reader rerun')
    states, gate = preflight(families)
    # First predictive read is below. Both requested families have passed above.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '2'; os.environ['MKL_NUM_THREADS'] = '2'
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    require(torch.__version__ == '2.1.2+cu118', 'Pinned source reader torch runtime required')
    records, query_records = [], {}
    with torch.no_grad():
        for state in states:
            record, queries = normalize(torch, state, gate)
            records.append(record); query_records[state['family'], state['seed'], state['condition']] = queries
    summaries = summarize(records, families)
    repair_flows, csv_rows = flows(query_records, families)
    measured = {'donors_once_seconds': sum(d['inclusive_seconds'] for d in gate['donors']),
        'requested_full60_fits_seconds': sum(r['costs']['fit60_inclusive_seconds'] for r in records),
        'qualification_runs_seconds': sum(r['inclusive_seconds'] for r in gate['paid']['qualifications']),
        'complete_cycle_cost_runs_seconds': sum(r['inclusive_seconds'] for r in gate['paid']['cost_runs'])}
    measured['sum_of_listed_source_inclusive_seconds'] = sum(measured.values())
    measured['scope'] = 'Retained donor once plus requested fits and listed paid admissions; standalone warm+fit costs per row.'
    measured['unmeasured_upstream_failed_or_orphan_owner_and_recovery_walltime'] = True
    limits = [
        'Three optimizer seeds on one fixed graph/split. No query iid bootstrap, graph-population interval, or significance claim.',
        'All12 full VALID exposures, first strict maximum of rounded4 pooled MRR, ties and selector optimism remain.',
        'Every member is evaluated at the pooled source-selected bank; no member/cycle/seed reselection or recombination.',
        'Top1 wrong means tie-aware native rank>1. All members wrong and one aligned negative slot wrong for all are separate diagnostics.',
        'Query indices0 and negative-slot indices0 inherit exact frozen input hashes/order; no labels, datasets, or new predictions are opened.',
        'Stored served mean logits are used. CPU raw metrics are diagnostics; source rounded4 HISTORY metrics remain authoritative.',
        'Growth checkpoint has no FREEZE checkpoint hash field; existing logits pointer is compared with actual checkpoint bytes. Logit hash is first recorded by this reader.',
        'Feature covariance uses graph-derived H; added4096 growth parameters match added single rank8 capacity, not total capacity.',
        'Copied independent controls share one warm donor and a synchronous pooled selector; they are not four independently acquired donors.',
        'Warm1 process exit remains unknown under the artifact-custody amendment. Operational recovery artifacts and unmeasured losses remain outside listed source times.',
        'Combined36 reopens the original source-selected banks. Reopened banks add no fits, independent evidence, or selector exposures; they retain the same checkpoint and selection.',
        'Engineering completion does not establish positive effect, novelty, or ensemble necessity. TEST remains closed.']
    # Publish only after every requested bank and analysis is validated.
    output.mkdir()
    gate.update(reader_config_sha256=args.config_sha256, reader_sha256=sha(__file__),
                CPU_only=True, torch_runtime=torch.__version__, output_relative=output_relative,
                source_selected_bank_reuse={
                    'combined36_read': families == ['initialization', 'growth'],
                    'reopened_individual_family_banks': [
                        {'family': family, 'bank_count': 3 * len(CONDITIONS[family]),
                         'prior_individual_readout_manifest_path': str(PHASE / OUTPUTS[(family,)] / 'MANIFEST.json')}
                        for family in families
                        if families == ['initialization', 'growth']
                        and (PHASE / OUTPUTS[(family,)] / 'MANIFEST.json').is_file()],
                    'prior_manifest_presence_is_metadata_disclosure_not_extra_completion_gate': True,
                    'same_original_source_selected_banks_no_extra_fits_independent_evidence_or_selection': True})
    write(output / 'METADATA_GATE.json', gate)
    write(output / 'NORMALIZED.json', {'families': families, 'records': records, 'TEST_access': False, 'limitations': limits})
    write(output / 'PAIRED_SUMMARIES.json', {'summaries': summaries, 'measured_costs': measured,
                                          'TEST_access': False, 'limitations': limits})
    write(output / 'QUERY_RANKS_AND_COMMON_ERRORS.json', {'input_identities': gate['input_identities'],
        'query_order': 'Original fixed VALID nonself order; zero-based indices0; all500 aligned negative slots retained',
        'banks': [{'family': f, 'seed': s, 'condition': c, 'queries': q} for (f, s, c), q in query_records.items()],
        'TEST_access': False})
    write(output / 'QUERY_FLOWS.json', {'contrasts': repair_flows, 'TEST_access': False, 'limitations': limits})
    with (output / 'QUERY_FLOWS.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    write(output / 'MANIFEST.json', {'files': [receipt(p) for p in sorted(output.iterdir()) if p.is_file()],
        'complete_requested_readout': True, 'TEST_access': False, 'models_loaded': False,
        'new_predictions': False, 'jobs_launched_or_signalled': False})
    print(json.dumps({'status': 'complete_closed_family_readout', 'families': families,
                      'records': len(records), 'output_relative': output_relative, 'TEST_access': False}))


if __name__ == '__main__':
    main()
