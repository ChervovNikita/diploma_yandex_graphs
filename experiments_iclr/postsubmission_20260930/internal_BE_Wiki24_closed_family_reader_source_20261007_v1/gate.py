"""Saved engineering metadata gate. No framework or scientific source imports."""
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
PYTHON = PHASE / 'native_ncn_runtime_20261005_v1/.venv/bin/python'
SUITE = 'learnable_internal_be_contrastive_multitask_suite_20261007_v4'
DRIVER = 'learnable_internal_be_WikiCS_scientific_family_driver_source_20261007_v1'
QUALIFIER = 'learnable_internal_be_resource_qualifier_source_20261007_v1'
SUITE_SHA = '76de82781e7fd496a5a3382b3a71a5781ea023dfe3777469cc005a2b19afbfce'
DRIVER_SHA = '9aeb3cd2f37e64fc082de800cd83af55b1c53fc45f313f7d084b133c0598ec9d'
QUALIFIER_SHA = '3668747a8e27ab7eaa3754f56744697d6a71cb1e2e3a05284175362e64833f50'
ADOPTION_SHA = 'b940cd222ea4befe9fbc0132b4bf5a7f53ab3bf11989dffef0c2043db3eed0cc'
CONFIG_SHA = 'aa30418d95bd4b44a0955aa334776f4ffb67b5920c29e4961998e703742a152f'
ARMS = ['single', 'single_contrastive', 'independent4', 'independent4_contrastive',
        'be_unit', 'be_init', 'be_unit_contrastive', 'be_init_contrastive']
SEEDS = [6101, 6203, 6307]
STATUSES = {'complete', 'retained_resource_failure', 'retained_fit_failure',
            'retained_memory_admission_failure', 'retained_source_or_custody_failure',
            'not_launched_driver_failure'}
OWNER_PID, OWNER_TICKS = 510850, 6015502511


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def file(relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Phase-relative file required')
    path = (PHASE / relative).resolve(strict=True)
    require(path.is_relative_to(PHASE.resolve()) and path.is_file(), 'File leaves phase')
    return path


def binding(path):
    path = Path(path)
    return {'path': str(path.relative_to(PHASE)), 'sha256': sha(path), 'bytes': path.stat().st_size}


def bound(row):
    path = file(row['path'])
    require(sha(path) == row['sha256'], 'Bound bytes changed: ' + row['path'])
    if 'bytes' in row:
        require(path.stat().st_size == row['bytes'], 'Bound size changed')
    return path


def seal(root, expected=None):
    root = Path(root).resolve()
    manifest_path = root / 'MANIFEST.json'
    if expected is not None:
        require(sha(manifest_path) == expected, 'Exact frozen source manifest required')
    for row in read(manifest_path)['files']:
        path = (root / row['path']).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file(), 'Source seal path')
        require(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Source bytes changed')
    return sha(manifest_path)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def identity(job):
    return dict(task='wikics', arm=job['arm'], source_manifest_sha256=SUITE_SHA,
                config=job['config'], data_manifest=job['data_manifest'],
                runtime_pin_sha256=sha(file(SUITE + '/RUNTIME_PIN.json')),
                hostname='anogena-2-0', physical_gpu_uuid='GPU-44039938-fd82-41d2-fefd-de71514e2fac',
                members=1 if job['arm'].startswith('single') else 4, own_views=2)


def base_job(job, cell, adoption):
    require(job.get('task') == 'wikics' and job.get('arm') == cell['arm'] and job.get('seed') == cell['seed'], 'Exact job cell')
    for key in ('config', 'data_manifest', 'data_export_review'):
        require(job.get(key) == adoption[key], 'Exact job ' + key)
    require(job.get('source_manifest_sha256') == SUITE_SHA and job.get('TEST_access') is False
            and job.get('automatic_retry') is False, 'Exact closed job source/scope')


def resource_metadata(receipt_path, job_path, cell, adoption, passed_required=False):
    """Recheck saved original receipt custody, without qualification work."""
    receipt, job = read(receipt_path), read(job_path)
    base_job(job, cell, adoption)
    require(receipt.get('schema') == 'internal-be-resource-qualification-v2'
            and receipt.get('cell_identity') == identity(job) and receipt.get('seed') == cell['seed']
            and receipt.get('job_sha256') == sha(job_path)
            and receipt.get('qualifier_manifest_sha256') == QUALIFIER_SHA, 'Saved same-arm/seed resource identity')
    live_path = file(job['supervisor_receipt_path'])
    terminal_path = live_path.with_name(live_path.stem + '_TERMINAL.json')
    live, terminal = read(live_path), read(terminal_path)
    require(receipt.get('live_receipt_sha256') == sha(live_path)
            and receipt.get('terminal_receipt_sha256') == sha(terminal_path)
            and terminal.get('live_receipt_sha256') == sha(live_path), 'Saved resource live/terminal hashes')
    qidentity = dict(cell_identity=identity(job), seed=cell['seed'], qualifier_manifest_sha256=QUALIFIER_SHA,
                     suite_manifest_sha256=SUITE_SHA, scope='committed_representative_update_complete_VALID',
                     data_export_review=adoption['data_export_review'])
    require(receipt.get('qualifier_identity') == qidentity and terminal.get('qualifier_identity') == qidentity
            and live.get('qualifier_identity') == qidentity and live.get('job_sha256') == sha(job_path)
            and live.get('supervisor_source_sha256') == sha(file(QUALIFIER + '/supervise.py')), 'Saved resource parent/source custody')
    require(finite(receipt.get('inclusive_seconds')) and finite(terminal.get('inclusive_seconds')), 'Actual finite resource cost')
    if passed_required:
        require(receipt.get('passed') is True and receipt.get('status') == 'complete' and receipt.get('exit_code') == 0
                and terminal.get('schema') == 'internal-be-resource-supervisor-terminal-v1'
                and live.get('schema') == 'internal-be-resource-live-supervisor-v1'
                and terminal.get('owned_worker_completed') is True and terminal.get('reap_observed') is True
                and terminal.get('cap_exceeded') is False and terminal.get('status') == 'complete'
                and terminal.get('exit_code') == 0 and receipt['inclusive_seconds'] <= job['hard_seconds']
                and terminal.get('absolute_hard_seconds') == job['hard_seconds'], 'Saved passing resource terminal')
        require(type(receipt.get('peak_GPU_bytes')) is int and 0 < receipt['peak_GPU_bytes'] <= 80 * 1024**3
                and 0 < receipt['inclusive_seconds'] <= 46800, 'Original domain-valid resource measurements')
        candidate_path = Path(job['output_directory']) / 'WORKER_RESOURCE.json'
        require(candidate_path.resolve().is_relative_to(PHASE.resolve()), 'Resource output leaves phase')
        candidate = read(candidate_path)
        require(sha(candidate_path) == receipt.get('worker_resource_sha256')
                and candidate.get('worker_completed') is True and candidate.get('passed') is False
                and candidate.get('qualifier_identity') == qidentity
                and candidate.get('finite_parameters_gradients_optimizer_outputs') is True
                and candidate.get('worker_pid') == terminal.get('child_pid')
                and candidate.get('worker_start_ticks') == terminal.get('child_start_ticks'), 'Saved original finite resource worker')
        work = receipt['work']
        require(work == candidate['work'] and work.get('modes') == ['local', 'global']
                and work.get('members') == identity(job)['members'] and work.get('own_views') == 2,
                'Unchanged resource scope')
        for key in ('two_view_TRAIN_backward_Adam', 'complete_VALID_evaluation', 'checkpoint_serialization', 'predictive_scores_closed'):
            require(work.get(key) is True, 'Complete saved resource work')
    return dict(receipt=binding(receipt_path), job_binding=binding(job_path), live_binding=binding(live_path),
                terminal_binding=binding(terminal_path), metadata=receipt, terminal=terminal)


def preflight(activation_path, activation_sha256, stage):
    """Returns only after all24 saved endpoints/failures pass; values stay closed."""
    require(sha(activation_path) == activation_sha256, 'Exact root activation bytes')
    cfg = read(activation_path)
    require(cfg.get('schema') == 'internal-be-Wiki24-reader-activation-v1'
            and cfg.get('stage') == stage and cfg.get('stage_enabled') is True
            and cfg.get('root_comparative_opening_authorized') is True
            and cfg.get('source_review_approved') is True and cfg.get('TEST_access') is False,
            'Explicit root opening/source authority required')
    require(socket.gethostname() == 'anogena-2-0' and Path.cwd().resolve() == REPO
            and str(Path(sys.executable).absolute()) == str(PYTHON) and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'Exact server repository/interpreter and hidden CUDA required')
    require(ROOT.parent == PHASE, 'Source must be staged in exact phase')
    reader_sha = seal(ROOT, cfg['reader_manifest_sha256'])
    approval = read(bound(cfg['root_reader_inspection']))
    require(approval.get('approved') is True and approval.get('reader_manifest_sha256') == reader_sha, 'Root reader source inspection')
    seal(PHASE / DRIVER, DRIVER_SHA); seal(PHASE / SUITE, SUITE_SHA); seal(PHASE / QUALIFIER, QUALIFIER_SHA)
    release_path = bound(cfg['family_release']); release = read(release_path)
    require(release.get('driver_manifest_sha256') == DRIVER_SHA
            and release.get('TEST_access') is False and release.get('automatic_retry') is False
            and release.get('predictive_opening_authorized') is False, 'Original closed family release')
    for key in ('root_scientific_fit_authorized', 'root_resource_execution_authorized', 'fixed_family_adopted', 'root_driver_source_approved'):
        require(release.get(key) is True, 'Original family authority')
    require(release.get('adoption') == {'path': DRIVER + '/ADOPTION_PROSPECTIVE.json', 'sha256': ADOPTION_SHA}, 'Fixed family adoption')
    adoption = read(bound(release['adoption']))
    config = read(bound(adoption['config']))
    require(adoption['config'] == {'path': SUITE + '/configs/wikics.json', 'sha256': CONFIG_SHA}
            and config['arms'] == ARMS and config['pilot_seeds'] == SEEDS
            and config['training']['epochs'] == 1100 and config['training']['local_epochs'] == 100, 'Exact declared full horizon/roster')
    roster = [dict(arm=a, seed=s, cell=a + '_' + str(s)) for s in SEEDS for a in ARMS]
    require(adoption['cells'] == roster and adoption['source_manifest_sha256'] == SUITE_SHA, 'Whole24 prospectively fixed cells')
    # Only role manifest/approval JSON, never dataset payloads.
    role = read(bound(adoption['data_manifest'])); data_approval = read(bound(adoption['data_export_review']))
    require(role.get('schema') == 'internal-be-official-role-projection-v2' and role.get('format') == 'NPZ_numeric_only'
            and role.get('task') == 'wikics' and role.get('split_index') == 0
            and role.get('train_count') == 580 and role.get('valid_count') == 5274
            and role.get('TEST_values_in_payload') is False and role.get('official_split_preserved') is True
            and data_approval.get('approved') is True and data_approval.get('data_manifest') == adoption['data_manifest']
            and data_approval.get('source_manifest_sha256') == SUITE_SHA, 'Exact official split0 role authority')
    execution_relative = release['execution_directory']
    execution = (PHASE / execution_relative).resolve(strict=True)
    require(execution.is_relative_to(PHASE.resolve()), 'Family execution leaves phase')
    closure_path = bound(cfg['family_closure'])
    require(closure_path == execution / 'FAMILY_CLOSURE.json', 'This released family closure only')
    closure = read(closure_path); owner_path = execution / 'OWNER.json'; owner = read(owner_path)
    require(owner == closure.get('owner') and owner.get('PID') == OWNER_PID and owner.get('start_ticks') == OWNER_TICKS
            and owner.get('release_sha256') == sha(release_path) and owner.get('driver_manifest_sha256') == DRIVER_SHA
            and owner.get('adoption') == release['adoption'], 'Exact actual driver owner/release custody')
    require(closure.get('schema') == 'internal-be-WikiCS-family-closure-v1' and closure.get('closed') is True
            and closure.get('adoption') == release['adoption'] and closure.get('predictive_values_opened') is False
            and closure.get('TEST_access') is False and closure.get('automatic_retry') is False
            and closure.get('all_full_endpoints_or_retained_failure') is True, 'Authoritative closed family required')
    rows = closure['cells']
    require(len(rows) == 24 and [{k: r[k] for k in ('arm', 'seed', 'cell')} for r in rows] == roster,
            'Exact all24 rows in fixed roster order')
    require(closure.get('complete_cells') == sum(r.get('status') == 'complete' for r in rows)
            and closure.get('failed_or_not_launched_cells') == sum(r.get('status') != 'complete' for r in rows)
            and finite(closure.get('inclusive_family_driver_seconds')), 'Closure counts/actual family elapsed')
    states = []
    for row in rows:
        cell = row['cell']; status = row['status']
        require(status in STATUSES and row.get('predictive_values_opened') is False, 'Retained terminal cell status')
        if status == 'not_launched_driver_failure':
            failure = read(execution / 'DRIVER_FAILURE.json')
            require(failure.get('rows') == rows and failure.get('automatic_retry') is False, 'Original fatal/unlaunched accounting')
        else:
            require(read(execution / 'receipts' / (cell + '_CELL.json')) == row, 'Immutable per-cell closure row')
            require(finite(row.get('inclusive_cell_driver_seconds')), 'Actual cell driver cost')
        state = {k: row[k] for k in ('arm', 'seed', 'cell', 'status')}
        state['closure_row'] = row
        if 'immutable_fit_job' in row:
            job_path = bound(row['immutable_fit_job']); job = read(job_path); base_job(job, row, adoption)
            require(job_path == execution / 'fit_jobs' / (cell + '.json')
                    and job.get('schema') == 'internal-be-predictive-cell-v1'
                    and job.get('root_execution_authorized') is True and job.get('source_review_approved') is True
                    and job.get('fixed_protocol_adopted') is True and job.get('external_hard_bound_confirmed') is True
                    and job.get('source_review_evidence') == adoption['source_review_evidence']
                    and job.get('family_release_sha256') == sha(release_path) and job.get('family_adoption') == release['adoption']
                    and job.get('resource_checkpoints_reused') is False and job.get('same_seed_resource_binding_enforced_by_driver') is True,
                    'Original immutable fit/adoption/resource identity')
            output = Path(job['output_directory'])
            require(output == execution / 'fits/outputs' / cell, 'Original fit output only')
            state.update(job_binding=binding(job_path), job=job)
            supervised = row.get('scientific_supervisor')
            if supervised is not None:
                require(finite(supervised.get('inclusive_driver_seconds')), 'Actual fit parent elapsed')
                parent_log = execution / 'receipts' / (cell + '_FIT_PARENT.log')
                require(sha(parent_log) == supervised['closed_parent_log_sha256']
                        and read(parent_log.with_suffix('.OWNER.json')) == supervised['supervisor_owner'], 'Saved fit parent owner/log custody')
            custody = row.get('fit_custody', {})
            if 'terminal' in custody:
                terminal_path = bound(custody['terminal']); terminal = read(terminal_path)
                state.update(terminal_binding=binding(terminal_path), terminal=terminal)
            if status == 'complete':
                require(supervised is not None and supervised.get('exit_code') == 0, 'Complete supervised fit')
                live_path = file(job['supervisor_receipt_path']); live = read(live_path)
                require(live_path == execution / 'fits/receipts' / (cell + '_LIVE.json')
                        and terminal_path == live_path.with_name(live_path.stem + '_TERMINAL.json')
                        and terminal.get('schema') == 'internal-be-supervisor-terminal-v2'
                        and terminal.get('status') == 'complete' and terminal.get('exit_code') == 0
                        and terminal.get('admission_success') is True and terminal.get('reap_observed') is True
                        and terminal.get('cap_exceeded') is False and terminal.get('live_receipt_sha256') == sha(live_path)
                        and terminal.get('cell_identity') == identity(job) and live.get('cell_identity') == identity(job)
                        and live.get('job_sha256') == sha(job_path), 'Complete actual fit live/terminal custody')
                recorded_owner = supervised['supervisor_owner']
                require(live.get('supervisor_pid') == recorded_owner['pid']
                        and live.get('supervisor_start_ticks') == recorded_owner['start_ticks']
                        and recorded_owner['job_sha256'] == sha(job_path)
                        and live.get('supervisor_source_sha256') == recorded_owner['source_supervisor_sha256']
                        == sha(file(SUITE + '/supervise.py')), 'Actual native fit supervisor identity')
                for key, value in dict(soft_seconds=28800, hard_seconds=32400, active_compute_seconds=32390, cleanup_grace_seconds=10).items():
                    require(job.get(key) == value, 'Full unchanged fit budget')
                    if key != 'soft_seconds':
                        require(live.get(key) == value, 'Live unchanged full fit cap')
                require(terminal.get('absolute_admission_cap_seconds') == 32400
                        and terminal.get('active_compute_seconds') == 32390 and terminal.get('cleanup_grace_seconds') == 10
                        and finite(terminal.get('inclusive_seconds')) and terminal['inclusive_seconds'] <= 32400, 'Actual full fit in cap')
                freeze_path = bound(custody['freeze']); freeze = read(freeze_path)
                require(freeze_path == output / 'FREEZE.json' and freeze.get('complete') is True
                        and freeze.get('epochs') == 1100 and freeze.get('steps') == 1100
                        and freeze.get('task') == 'wikics' and freeze.get('arm') == row['arm'] and freeze.get('seed') == row['seed']
                        and freeze.get('source_manifest_sha256') == SUITE_SHA and freeze.get('TEST_access') is False
                        and freeze.get('scores_closed') is True and finite(freeze.get('seconds')), 'Actual full native scientific endpoint')
                selected = output / 'selected.pt'
                require(sha(selected) == freeze.get('selected_sha256') == custody.get('selected_sha256'), 'Official selected checkpoint digest')
                require(row.get('same_arm_and_seed_resource_binding') is True
                        and job.get('resource_qualification_evidence') == [row['resource_evidence']], 'Exact selected-cell resource evidence')
                state.update(live_binding=binding(live_path), live=live, freeze_binding=binding(freeze_path), freeze=freeze,
                             selected_checkpoint=binding(selected))
        require(status != 'complete' or 'selected_checkpoint' in state, 'Complete cell requires all endpoint custody')
        states.append(state)
    # Whole24 engineering closure is established before recording auxiliary hashes.
    for state in states:
        if state['status'] == 'complete' and state['arm'] == 'independent4':
            output = Path(state['job']['output_directory'])
            state['own_checkpoints'] = [binding(output / ('own_best_' + str(m) + '.pt')) for m in range(4)]
            state['own_bank_metric_json'] = binding(output / 'CLOSED_OWN_BEST_BANK.json')
    # Actual resource history is engineering JSON only. Failed previous attempts remain visible.
    origins = adoption['prior_resource_origins'] + [dict(execution_directory=execution_relative + '/resources', job_directory=execution_relative + '/resource_jobs')]
    resource_history = []
    for origin in origins:
        for cell in roster:
            receipt_path = PHASE / origin['execution_directory'] / 'receipts' / (cell['cell'] + '_LIVE_RESOURCE.json')
            if receipt_path.is_file():
                job_relative = origin['job_directory'] + '/' + cell['cell'] + '.json'
                try:
                    job_path = file(job_relative)
                    resource_history.append(resource_metadata(receipt_path, job_path, cell, adoption))
                except (ValueError, KeyError, FileNotFoundError) as error:
                    # An incomplete historical attempt is preserved as failure
                    # evidence, never promoted to the passing receipt for a fit.
                    job_path = PHASE / job_relative
                    resource_history.append(dict(receipt=binding(receipt_path),
                        job_binding=binding(job_path) if job_path.is_file() else None,
                        missing_job_path=None if job_path.is_file() else job_relative,
                        metadata=read(receipt_path), custody_status='incomplete_or_unverified',
                        custody_error_type=type(error).__name__, custody_error=str(error)))
    for state in states:
        if 'resource_evidence' in state['closure_row']:
            evidence_path = bound(state['closure_row']['resource_evidence'])
            found = next((h for h in resource_history if h['receipt']['path'] == str(evidence_path.relative_to(PHASE))), None)
            require(found is not None and found['job_binding'] is not None, 'Declared resource evidence has original job in preserved history')
            resource_metadata(evidence_path, bound(found['job_binding']), state, adoption, passed_required=True)
    supersession_path = bound(release['resource_queue_supersession'])
    gate = dict(schema='internal-be-Wiki24-saved-closure-gate-v1', passed=True, family_closed=True,
                closure=binding(closure_path), family_release=binding(release_path), owner=binding(owner_path),
                owner_metadata=owner, reader_manifest_sha256=reader_sha, activation=binding(Path(activation_path).resolve()),
                source_manifest_sha256=SUITE_SHA, config_binding=adoption['config'], config=config,
                data_manifest_binding=adoption['data_manifest'], data_export_review_binding=adoption['data_export_review'],
                complete_cells=closure['complete_cells'], failed_or_not_launched_cells=closure['failed_or_not_launched_cells'],
                inclusive_family_driver_seconds=closure['inclusive_family_driver_seconds'], cells=states,
                resource_history=resource_history, resource_queue_supersession=binding(supersession_path),
                resource_queue_supersession_metadata=read(supersession_path), predictive_values_opened_at_gate=False,
                process_observations='Saved authoritative closure/reap metadata only; no live polling')
    return cfg, gate


def fresh_output(cfg):
    relative = Path(cfg['output_directory'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Phase-relative fresh output')
    output = (PHASE / relative).resolve()
    require(output.is_relative_to(PHASE.resolve()) and output.parent.is_dir() and not output.exists(), 'Fresh phase-owned output only')
    output.mkdir(mode=0o700)
    return output


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.chmod(path, 0o600)
