"""Source-only v7: exact occupied-recovery lineage and saved-logit VALID audit.

No Torch import, model construction, checkpoint deserialization, or heldout
label/report access. Execution requires a separate exact root admission and
independent source-review receipt. This module has not been numerically run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REMOTE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
STUDY = PHASE / 'graph_init_precision_execution_root_v2/study_v2'
REGISTRY = STUDY / 'GRAPH_INIT_ATTEMPT_REGISTRY.json'
REGISTRY_SHA = '715c361c82b543d4c436a565ccaa86470a9204433998d410954f2f268300307f'
COMPARISON = STUDY / 'comparison/COMPARISON_FREEZE.json'
COMPARISON_SHA = '954bcc7b37a6bfeebabdc0bfe8d5c2c65ffc4f8b7dde874d8f06665711907956'
ORIGINAL = PHASE / 'graph_init_precision_continuation_v3_cap_binding/coordinator_run_v3'
RECOVERY = PHASE / 'graph_init_precision_continuation_v3_cap_binding/coordinator_run_v4_occupied_recovery'
ROOT_RECOVERY = PHASE / 'graph_init_occupied_launch_recovery_root_v1/ROOT_RECOVERY_DECISION.json'
ROOT_RECOVERY_SHA = '5757ebfba14c50598cbfeaf57c1ce9a1f066b618784c5282631f70835e682425'
PRIOR_ROOT_SHA = '501bfb7084ae24c9b2fd89f4a5624882737a35d863cf5e3b968dbc697afe3e28'
PRIOR_FAILURE_SHA = 'cf4589fd455afd2e738b5e7ca3bf77df54f28dec85f575f0ceb6c3ff7e904f38'
PRECHILD_SHA = '56bdf120c66bd106249839b941b37d54751c9c994572ffec911a8732b1899bcc'
V4_SHA = 'cb4adb577ab52ae3b7e7dfda1693705ef10968f14ffaac928d8ee4139beb8a8f'
DRIVER = PHASE / 'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v3_precision/prototype/graph_init_driver.py'
DRIVER_SHA = '76873c66454fce606c3b54f609036706bc640676a300430be841e33fbb0a6e4f'
GPU_UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
ARMS = ('graph', 'common_only', 'random_tangent', 'topology_permuted', 'warm_copy')
SETTINGS = ('Squirrel', 'Photo')
SEEDS = (17, 29, 43)
OUTPUT = PHASE / 'graph_init_analysis_companion_v7_source_valid_execution_root_v1/run_v1'
NATIVE_NLL_ATOL = 1e-6


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def confined(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'Absolute project-confined path required')
    require(path.is_relative_to(PHASE), 'Path escapes exact project phase')
    require(not any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(PHASE)),
            'Symlink in project path')
    return path


def sha(path):
    digest = hashlib.sha256()
    with confined(path).open('rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def descriptor(path):
    return {'path': str(confined(path)), 'sha256': sha(path)}


def read(path):
    path = confined(path)
    require(path.suffix == '.json', 'Metadata reader opens JSON only')
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def bound(record, suffix='.json'):
    require(isinstance(record, dict) and set(record) in ({'path', 'sha256'}, {'path', 'sha256', 'bytes'}),
            'Exact path/hash descriptor required')
    path = confined(record['path'])
    require(path.suffix == suffix and isinstance(record['sha256'], str) and len(record['sha256']) == 64,
            'Descriptor type differs')
    require(sha(path) == record['sha256'], 'Descriptor hash differs: ' + str(path))
    if 'bytes' in record:
        require(type(record['bytes']) is int and record['bytes'] >= 0 and path.stat().st_size == record['bytes'],
                'Descriptor length differs')
    return path


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    with confined(path).open('x') as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write('\n')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def load_source(path, name, expected):
    require(sha(path) == expected, 'Imported source identity differs')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def admission_guard(path, output):
    require(PHASE == REMOTE, 'Execution is confined to the exact remote repository phase')
    admission = read(path)
    fields = {'schema', 'execution_authorized', 'authorized_phase', 'attempt_registry',
              'comparison_freeze', 'lineage', 'source_manifest', 'source_seal', 'independent_source_review',
              'output_root', 'cpu_only', 'Tensor_GPU_execution_authorized', 'heldout_access_authorized',
              'final_report_access_authorized', 'serialized_checkpoint_replay_authorized'}
    require(set(admission) == fields and admission['schema'] == 'graph-init-companion-v7-source-valid-admission-v1'
            and admission['execution_authorized'] is True and admission['authorized_phase'] == 'source_VALID_audit',
            'Exact separately admitted source VALID audit required')
    require(admission['cpu_only'] is True and all(admission[k] is False for k in
            ('Tensor_GPU_execution_authorized', 'heldout_access_authorized', 'final_report_access_authorized',
             'serialized_checkpoint_replay_authorized')), 'No Tensor/GPU/heldout/report/checkpoint replay authority')
    require(confined(output) == OUTPUT and admission['output_root'] == str(OUTPUT), 'Canonical new v7 audit output required')
    require(bound(admission['attempt_registry']) == REGISTRY and admission['attempt_registry']['sha256'] == REGISTRY_SHA,
            'Exact registered study required')
    require(bound(admission['comparison_freeze']) == COMPARISON
            and admission['comparison_freeze']['sha256'] == COMPARISON_SHA, 'Exact root-closed comparison required')
    manifest_path = bound(admission['source_manifest'])
    seal_path = bound(admission['source_seal'])
    require(manifest_path == HERE / 'MANIFEST.json' and seal_path == HERE / 'SEAL.json', 'Exact v7 source packet required')
    manifest, seal = read(manifest_path), read(seal_path)
    require(seal['manifest_sha256'] == sha(manifest_path) and seal['source_only'] is True
            and seal['execution_authorized'] is False, 'Frozen source-only seal required')
    names = []
    for record in manifest['payload']:
        relative = Path(record['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source payload')
        file = confined(HERE / relative)
        require(sha(file) == record['sha256'] and file.stat().st_size == record['bytes'], 'Source payload changed')
        names.append(str(relative))
    require(len(names) == len(set(names)), 'Duplicate source payload')
    require({str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()
             and p.name not in ('MANIFEST.json', 'SEAL.json')} == set(names), 'Source packet file inventory changed')
    review = read(bound(admission['independent_source_review']))
    require(set(review) == {'schema', 'approved', 'source_manifest', 'source_seal', 'source_VALID_only',
                            'serialized_checkpoint_replay_missing_disclosed'}
            and review['schema'] == 'graph-init-companion-v7-independent-source-review-v1'
            and review['approved'] is True and review['source_manifest'] == admission['source_manifest']
            and review['source_seal'] == admission['source_seal'] and review['source_VALID_only'] is True
            and review['serialized_checkpoint_replay_missing_disclosed'] is True,
            'Exact independent source review required before execution')
    require(sha(HERE / 'legacy_v4_contract.py') == V4_SHA, 'Preserved v4 contract changed')
    return admission


def lineage_guard(lineage, registry):
    fields = {'prior_root_decision', 'prior_failed_coordinator', 'pre_child_proof', 'recovery_root_decision',
              'prior_start', 'recovery_start', 'recovery_completed'}
    require(set(lineage) == fields, 'Exact occupied-recovery lineage descriptors required')
    paths = {key: bound(record) for key, record in lineage.items()}
    require(paths['prior_start'] == ORIGINAL / 'START.json'
            and paths['prior_failed_coordinator'] == ORIGINAL / 'FAILED.json'
            and paths['recovery_start'] == RECOVERY / 'START.json'
            and paths['recovery_completed'] == RECOVERY / 'COMPLETED.json'
            and paths['recovery_root_decision'] == ROOT_RECOVERY, 'Exact two coordinator namespaces required')
    require(lineage['prior_root_decision']['sha256'] == PRIOR_ROOT_SHA
            and lineage['prior_failed_coordinator']['sha256'] == PRIOR_FAILURE_SHA
            and lineage['pre_child_proof']['sha256'] == PRECHILD_SHA
            and lineage['recovery_root_decision']['sha256'] == ROOT_RECOVERY_SHA, 'Preserved root/failure/proof identity differs')
    require(not (ORIGINAL / 'COMPLETED.json').exists() and not (RECOVERY / 'FAILED.json').exists(),
            'Original failure/recovery success namespaces changed')
    prior, failure, proof_receipt, decision, start, completed = (read(paths[name]) for name in
            ('prior_root_decision', 'prior_failed_coordinator', 'pre_child_proof', 'recovery_root_decision',
             'recovery_start', 'recovery_completed'))
    prior_start = read(paths['prior_start'])
    for root in (prior, decision):
        require(root['schema'] == 'graph-init-finite-continuation-root-decision-v1' and root['approved'] is True
                and root['scientific_phases_authorized'] is True and root['fixed_native_protocol_unchanged'] is True
                and root['automatic_retry_authorized'] is False and root['compare_authorized'] is False
                and root['report_authorized'] is False and root['final_labels_authorized'] is False
                and root['attempt_registry'] == descriptor(REGISTRY), 'Original scientific authority changed')
    require(prior['coordinator_run_root'] == str(ORIGINAL) and decision['coordinator_run_root'] == str(RECOVERY),
            'Root decision namespace changed')
    require(prior_start['schema'] == 'graph-init-finite-coordinator-start-v1'
            and prior_start['root_decision'] == lineage['prior_root_decision']
            and prior_start['attempt_registry'] == descriptor(REGISTRY) and prior_start['attempt_count'] == 72
            and prior_start['automatic_retry'] is False, 'Original coordinator START differs')
    require(failure['schema'] == 'graph-init-finite-coordinator-failure-v1' and failure['automatic_retry'] is False
            and failure['this_coordinator_run_blocked'] is True and failure['closure_blocked'] is True,
            'The preserved v3 failure must remain a real blocking failure')
    recovery = decision['administrative_recovery']
    require(recovery['approved'] is True and recovery['pre_child_refusal_only'] is True
            and recovery['scientific_retry_authorized'] is False and recovery['completed_prefix_count'] == 64
            and recovery['source_method_parameters_seeds_caps_and_selector_unchanged'] is True
            and recovery['completed_keys_not_retried'] is True and recovery['TEST_closed'] is True
            and recovery['prior_root_decision'] == lineage['prior_root_decision']
            and recovery['prior_failed_coordinator'] == lineage['prior_failed_coordinator']
            and recovery['pre_child_proof'] == lineage['pre_child_proof'], 'Exact pre-child recovery authority required')
    proof = proof_receipt['proof']
    require(proof_receipt['exit_code'] == 0 and proof['failed_key'] == recovery['failed_key']
            and proof['failed_phase'] == 'fit' and proof['GPU_UUIDs'] == [GPU_UUID]
            and proof['GPU_compute_processes'] == [] and proof['test_payload_opened'] is False,
            'Failed-key pre-child proof differs')
    require(all(proof[key] is False for key in ('claim_exists', 'phase_terminal_exists', 'science_output_exists',
            'inner_supervisor_exists', 'root_terminal_exists')), 'Scientific attempt may not be retried')
    rejected = proof['outer_terminal']
    require(rejected['complete'] is False and rejected['child_exit_code'] == 1
            and rejected['root_request_unchanged'] is True and rejected['timed_out'] is False
            and rejected['TERM_sent'] is False and rejected['KILL_sent'] is False
            and all(value is None for value in rejected['inner_evidence'].values()), 'Pre-child refusal contract differs')
    require(hashlib.sha256(proof['original_log'].encode()).hexdigest() == proof['original_log_sha256']
            == rejected['stdout_stderr_sha256'] and 'Authorized GPU is occupied; no child started' in proof['original_log'],
            'Preserved refusal evidence differs')
    plan = decision['finite_plan']
    require(plan == prior['finite_plan'] and object_hash(plan) == decision['finite_plan_sha256']
            == prior['finite_plan_sha256'] == prior_start['finite_plan_sha256'], 'Exact fixed finite plan changed')
    expected = {row['key']: row for row in registry['attempts']}
    require(len(plan) == len(expected) == 72 and {row['key'] for row in plan} == set(expected), 'Finite plan is not all72')
    for row in plan:
        require(all(row[field] == expected[row['key']][field] for field in ('key', 'context_sha256', 'phase', 'arm', 'output')),
                'Plan row differs from fixed registry')
    prefix, suffix = [row['key'] for row in plan[:64]], [row['key'] for row in plan[64:]]
    require(len(set(prefix)) == 64 and len(set(suffix)) == 8 and set(prefix).isdisjoint(suffix)
            and all(row['phase'] == 'fit' for row in plan[64:]) and suffix[0] == recovery['failed_key'],
            'Exact64 completed prefix and8 untouched fit suffix required')
    failed_key = recovery['failed_key']
    failed_launch_path = ORIGINAL / (failed_key + '_LAUNCH_CLAIM.json')
    failed_launch = read(failed_launch_path)
    failed_request_path = bound(failed_launch['request'])
    require(failed_request_path == ORIGINAL / 'requests' / failed_key / 'REQUEST.json'
            and failed_launch['attempt'] == plan[64] and failed_launch['automatic_retry'] is False,
            'Preserved rejected launch identity differs')
    failed_request = read(failed_request_path)
    failed_outer = ORIGINAL / 'supervision' / (failed_key + '_outer')
    failed_start = read(failed_outer / 'START.json')
    require(read(failed_outer / 'TERMINAL.json') == rejected
            and rejected['START_sha256'] == sha(failed_outer / 'START.json')
            and failed_start['root_request']['sha256'] == sha(failed_request_path)
            and confined(PHASE / failed_start['root_request']['path']) == failed_request_path
            and failed_request['root_decision'] == lineage['prior_root_decision']
            and failed_request['attempt'] == plan[64]
            and failed_request['outer_supervisor_directory'] == str(failed_outer)
            and failed_start['outer_output'] == str(failed_outer)
            and failed_start['ssh_destination'] == LOGIN,
            'Actual pre-child refusal terminal/request custody differs from preserved proof')
    require(set(start) == {'UTC', 'decision', 'completed_prefix', 'remaining_keys', 'scientific_source_changed',
                           'scientific_retry', 'final_labels_authorized'}
            and start['decision'] == lineage['recovery_root_decision'] and start['completed_prefix'] == prefix
            and start['remaining_keys'] == suffix and start['scientific_source_changed'] is False
            and start['scientific_retry'] is False and start['final_labels_authorized'] is False,
            'Recovery START serialization or exact prefix/suffix differs')
    require(set(completed) == {'UTC', 'completed', 'registered_successful_terminals', 'seconds',
                               'compare_or_final_labels_executed'}
            and completed['completed'] is True and completed['registered_successful_terminals'] == 72
            and completed['compare_or_final_labels_executed'] is False and finite(completed['seconds'])
            and completed['seconds'] >= 0, 'Exact successful recovery coordinator serializer required')
    return prefix, suffix, plan, {'preserved_failure': lineage['prior_failed_coordinator'],
            'preserved_pre_child_proof': lineage['pre_child_proof'],
            'preserved_rejected_launch': descriptor(failed_launch_path),
            'preserved_rejected_request': descriptor(failed_request_path),
            'preserved_rejected_outer_terminal': descriptor(failed_outer / 'TERMINAL.json'),
            'refusal_outer_seconds': rejected['whole_supervised_seconds'],
            'prior_coordinator_seconds_includes_successful_prefix': failure['seconds'],
            'failure_costs_are_not_fabricated_as_zero': True,
            'cost_exclusions': '72 successful registered-phase outer costs exclude acquisition/setup, prior failures/refusal, final scoring and this audit; failure receipts remain separate, and coordinator totals must not be double-counted.'}


def whole_cost(run, key, launch_claim, record, root_decision):
    """Retains v4 physical checks; recovery receipt differences are exact and explicit."""
    fields = {'UTC', 'phase_terminal', 'whole_terminal', 'whole_supervised_seconds', 'whole_cap_seconds'}
    if run == ORIGINAL:
        fields.add('whole_process_costs_charged')
        require(record.get('whole_process_costs_charged') is True, 'Original cost-charge statement missing')
    require(set(record) == fields, 'Unexpected completion receipt serializer')
    outer = run / 'supervision' / (key + '_outer')
    whole_path = bound(record['whole_terminal'])
    require(whole_path == outer / 'TERMINAL.json', 'Canonical per-attempt outer terminal required')
    whole, start_path = read(whole_path), outer / 'START.json'
    require(whole.get('schema') == 'gnnm-whole-process-bound-terminal-v1'
            and whole.get('START_sha256') == sha(start_path), 'Whole terminal changed START identity')
    start = read(start_path)
    request_path = bound(launch_claim['request'])
    require(request_path == run / 'requests' / key / 'REQUEST.json', 'Canonical launch request required')
    request = read(request_path)
    phase_admission_path = bound(request['phase_admission'])
    phase_admission = read(phase_admission_path)
    source_claim = read(STUDY / 'claims' / (key + '.json'))
    require(phase_admission_path == run / 'requests' / key / 'ADMISSION.json'
            and phase_admission['schema'] == 'graph-init-phase-admission-v1'
            and phase_admission['execution_authorized'] is True
            and phase_admission['authorized_phase'] == launch_claim['attempt']['phase']
            and phase_admission['arm'] == launch_claim['attempt']['arm']
            and object_hash(phase_admission['context']) == launch_claim['attempt']['context_sha256']
            and phase_admission['attempt_registry'] == descriptor(REGISTRY)
            and source_claim['admission'] == request['phase_admission'], 'Launch phase-admission custody differs')
    relative = Path(start['root_request']['path'])
    require(not relative.is_absolute() and '..' not in relative.parts
            and confined(PHASE / relative) == request_path and start['root_request']['sha256'] == sha(request_path),
            'START does not bind this launch request')
    inner = run / 'supervision' / (key + '_inner')
    claim = launch_claim['attempt']
    require(start.get('schema') == 'gnnm-whole-process-bound-start-v1' and start.get('ssh_destination') == LOGIN
            and start['outer_output'] == request['outer_supervisor_directory'] == str(outer)
            and start['inner_supervisor_directory'] == request['supervisor_directory'] == str(inner)
            and request['root_admitted'] is True and request['attempt'] == claim
            and request['action'] == claim['phase'] and request['output'] == claim['output'],
            'Whole-process source request identity differs')
    cap = claim['whole_cap_seconds']
    require(finite(cap) and cap > 0 and cap == record['whole_cap_seconds'] == start['whole_cap_seconds']
            == request['whole_cap_seconds'], 'Whole-process cap binding differs')
    require(whole.get('complete') is True and whole.get('within_whole_cap') is True
            and whole.get('root_request_unchanged') is True and whole.get('timed_out') is False
            and whole.get('child_exit_code') == 0, 'Unverified whole-process completion')
    seconds = record['whole_supervised_seconds']
    require(finite(seconds) and 0 <= seconds <= cap and seconds == whole['whole_supervised_seconds'],
            'Cost disagrees with physical whole terminal')
    require(launch_claim['automatic_retry'] is False and request['automatic_retry'] is False
            and request['root_decision'] == root_decision and request['heldout_scoring_admitted'] is False
            and request['compare_admitted'] is False and request['final_labels_accessible'] is False,
            'Attempt authority/scope differs')
    root_dir = run / 'root_receipts' / key
    require(request['receipt_directory'] == str(root_dir), 'Root receipt namespace differs')
    root_start, terminal = read(root_dir / 'START.json'), read(root_dir / 'TERMINAL.json')
    require(root_start['request_sha256'] == terminal['request_sha256'] == sha(request_path)
            and root_start['root_decision'] == root_decision and root_start['attempt'] == claim
            and root_start['gpu_uuid'] == GPU_UUID and root_start['whole_cap_seconds'] == cap
            and root_start['outer_start'] == descriptor(start_path) and root_start['automatic_retry'] is False
            and terminal['completed'] is True and terminal['child_exit_code'] == 0
            and terminal['phase_terminal'] == record['phase_terminal'] and terminal['final_labels_read'] is False
            and terminal['compare_or_report_executed'] is False and terminal['automatic_retry'] is False,
            'Root terminal/start/authority links differ')
    command, completion = read(inner / 'command.json'), read(inner / 'completion.json')
    require(completion['exit_code'] == 0 and completion['command_sha256'] == sha(inner / 'command.json')
            and whole['inner_evidence']['command.json']['sha256'] == sha(inner / 'command.json')
            and whole['inner_evidence']['completion.json']['sha256'] == sha(inner / 'completion.json')
            and completion['environment_sha256'] == whole['inner_evidence']['environment.json']['sha256']
            and completion['log_sha256'] == whole['inner_evidence']['stdout_stderr.log']['sha256']
            and completion['environment_sha256'] == sha(inner / 'environment.json')
            and completion['log_sha256'] == sha(inner / 'stdout_stderr.log')
            and command['argv'] == start['child_argv'], 'Inner completion/command custody differs')
    return seconds, {'whole_terminal': record['whole_terminal'], 'START': descriptor(start_path),
            'request': descriptor(request_path), 'root_START': descriptor(root_dir / 'START.json'),
            'root_terminal': descriptor(root_dir / 'TERMINAL.json'), 'inner_command': descriptor(inner / 'command.json'),
            'inner_completion': descriptor(inner / 'completion.json'),
            'whole_process_costs_charged_by_this_audit_after_physical_verification': True,
            'original_record_cost_charge_field_present': run == ORIGINAL}


def closed_cohort(admission, driver):
    registry, source_terminals = driver.registry_closure(admission['attempt_registry'])
    prefix, suffix, plan, failure_disclosure = lineage_guard(admission['lineage'], registry)
    expected = {row['key']: row for row in registry['attempts']}
    contexts = {object_hash(ctx): ctx for ctx in registry['contexts']}
    require(len(contexts) == 6, 'Exact six context identities required')
    require({p.stem for p in (STUDY / 'claims').glob('*.json')} == set(expected)
            and {p.stem for p in (STUDY / 'terminals').glob('*.json')} == set(expected),
            'Unknown/missing canonical claim or terminal blocks v6 closure')
    rows, bindings = [], []
    for run, keys, root in ((ORIGINAL, prefix, admission['lineage']['prior_root_decision']),
                           (RECOVERY, suffix, admission['lineage']['recovery_root_decision'])):
        receipts = sorted(run.glob('*_COMPLETED.json'))
        launch_keys = {p.name.removesuffix('_LAUNCH_CLAIM.json') for p in run.glob('*_LAUNCH_CLAIM.json')}
        permitted_launch_keys = set(keys) | ({suffix[0]} if run == ORIGINAL else set())
        require(launch_keys == permitted_launch_keys, 'Unknown/missing launch claim or hidden failed attempt')
        require({p.name.removesuffix('_COMPLETED.json') for p in receipts} == set(keys)
                and len(receipts) == len(keys), 'Exact disjoint successful receipt namespace required')
        for path in receipts:
            key = path.name.removesuffix('_COMPLETED.json')
            launch_path = run / (key + '_LAUNCH_CLAIM.json')
            launch_claim, record, attempt = read(launch_path), read(path), expected[key]
            require(launch_claim['attempt'] == next(row for row in plan if row['key'] == key), 'Launch differs from exact plan')
            terminal_path = bound(record['phase_terminal'])
            require(terminal_path == STUDY / 'terminals' / (key + '.json'), 'Canonical registered terminal required')
            terminal = read(terminal_path)
            require(terminal['schema'] == 'graph-init-attempt-terminal-v1' and terminal['completed'] is True
                    and terminal['final_labels_read'] is False and bound(terminal['claim']) == STUDY / 'claims' / (key + '.json'),
                    'Canonical phase terminal/claim differs')
            freeze, directory = driver.verify_freeze(terminal['freeze'], attempt['phase'], contexts[attempt['context_sha256']])
            require(directory == confined(attempt['output']) and freeze.get('arm') == attempt['arm'], 'Canonical arm/output differs')
            seconds, physical = whole_cost(run, key, launch_claim, record, root)
            rows.append({'setting': contexts[attempt['context_sha256']]['graph'], 'phase': attempt['phase'],
                         'arm': attempt['arm'], 'key': key, 'coordinator': str(run), 'outer_supervised_seconds': seconds})
            bindings.append({'key': key, 'completion': descriptor(path), 'launch': descriptor(launch_path),
                             'phase_terminal': record['phase_terminal'], **physical})
    require(len(rows) == 72 and len({row['key'] for row in rows}) == 72
            and Counter(row['phase'] for row in rows) == {'qualify': 6, 'warm': 6, 'initialize': 30, 'fit': 30},
            'All72 fixed phases required')
    require(len({b['whole_terminal']['path'] for b in bindings}) == 72, 'Physical outer terminals must be distinct')
    for setting in SETTINGS:
        require(Counter(r['phase'] for r in rows if r['setting'] == setting)
                == {'qualify': 3, 'warm': 3, 'initialize': 15, 'fit': 15}, 'Setting phase counts differ')
        for arm in ARMS:
            require(Counter(r['phase'] for r in rows if r['setting'] == setting and r['arm'] == arm)
                    == {'initialize': 3, 'fit': 3}, 'Arm phase counts differ')
    return registry, rows, bindings, failure_disclosure, source_terminals


def trace_selector(path, backbone, selection):
    cap, patience, offset, midpoint = (1950, 250, 50, 950) if backbone == 'polyformer_mono' else (950, None, 250, 450)
    require(backbone in ('polyformer_mono', 'polynormer_r'), 'Unknown frozen backbone')
    regular, events = [], []
    with confined(path).open() as file:
        for line in file:
            require(line.endswith('\n') and line.strip(), 'Partial/blank trace record')
            row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite trace')))
            if 'event' in row:
                require(set(row) == {'event', 'continuation_epoch', 'native_stage_epoch', 'actual_update', 'stage'}
                        and row['event'] == 'native_midpoint_saved' and row['continuation_epoch'] == midpoint
                        and len(regular) == midpoint + 1 and row['actual_update'] == midpoint + offset
                        and row['native_stage_epoch'] == (1000 if patience else 500)
                        and row['stage'] == ('native' if patience else 'global'), 'Changed/late midpoint event')
                events.append(row)
            else:
                epoch = row['continuation_epoch']
                require(type(epoch) is int and epoch == len(regular) and finite(row['validation_nll'])
                        and row['validation_nll'] >= 0, 'Trace epoch/score contract differs')
                if epoch == 0:
                    require(set(row) == {'continuation_epoch', 'validation_nll', 'initializer_checkpoint_eligible'}
                            and row['initializer_checkpoint_eligible'] is True, 'Epoch0 selector eligibility differs')
                else:
                    require(set(row) == {'continuation_epoch', 'actual_update', 'stage', 'train_ce', 'validation_nll',
                                        'nonzero_gradient_tensors'} and row['actual_update'] == epoch + offset
                            and row['stage'] == ('native' if patience else 'global') and finite(row['train_ce'])
                            and row['train_ce'] >= 0
                            and type(row['nonzero_gradient_tensors']) is int and row['nonzero_gradient_tensors'] > 0,
                            'Native trace update/stage contract differs')
                regular.append(row)
    completed = len(regular) - 1
    require(1 <= completed <= cap, 'Incomplete/over-budget continuation trace')
    best_epoch, best_value, first_stop = 0, regular[0]['validation_nll'], None
    for row in regular[1:]:
        epoch = row['continuation_epoch']
        require(first_stop is None, 'Trace continued beyond frozen first patience stop')
        if row['validation_nll'] < best_value:
            best_epoch, best_value = epoch, row['validation_nll']
        if patience is not None and epoch - best_epoch >= patience:
            first_stop = epoch
    require(completed == cap or (patience is not None and first_stop == completed), 'Premature continuation stop')
    require(len(events) == int(completed >= midpoint), 'Missing/duplicate midpoint event')
    require(selection['continuation_updates_completed'] == completed and selection['update_cap'] == cap
            and selection['patience'] == patience and selection['selected_continuation_epoch'] == best_epoch
            and selection['selected_actual_update'] == best_epoch + offset
            and selection['primary_validation_nll'] == best_value and selection['configuration'] == 0
            and selection['members'] == 4 and selection['global_only'] is (backbone == 'polynormer_r')
            and selection['selection'] == 'pooled mean raw-logit validation NLL; earliest strict tie; epoch0 eligible'
            and selection['native_midpoint_saved'] is (completed >= midpoint)
            and selection['native_midpoint_continuation_epoch'] == midpoint
            and selection['native_midpoint_stage_epoch'] == (1000 if patience else 500)
            and selection['native_midpoint_actual_update'] == midpoint + offset
            and selection['native_midpoint_absent_reason'] == (None if completed >= midpoint else
                'native early stopping before frozen absolute native-stage midpoint'), 'Saved selector differs from complete trace')
    return {'trace': descriptor(path), 'regular_records': len(regular), 'midpoint_events': len(events),
            'completed_updates': completed, 'selected_epoch': best_epoch, 'selected_actual_update': best_epoch + offset,
            'selected_trace_NLL': best_value, 'earliest_strict_minimum_verified': True,
            'historical_epoch_scores_independently_recomputed': False}


def full_logit_guard(np, row, context):
    meta = read(bound(context['graph_input']))
    path = bound(row['selected_logits'], '.npy')
    require(path == bound(row['fit_freeze']).parent / 'selected_member_logits.npy', 'Selected-logit path differs')
    array = np.load(path, allow_pickle=False, mmap_mode='r')
    require(array.dtype == np.float32 and array.shape == (4, meta['num_nodes'], meta['num_classes'])
            and np.isfinite(array).all(), 'Complete finite float32 K4 logit contract required')
    return {'logits': row['selected_logits'], 'shape': list(array.shape), 'dtype': str(array.dtype),
            'full_array_finite': True, 'graph_input': context['graph_input']}


def validation_pack(np, context):
    role_path = bound(context['role_freeze'])
    role = read(role_path)
    meta = read(bound(context['graph_input']))
    require(role['labels_read'] is False, 'Exact pre-label role freeze required')
    ids, node_bindings = {}, {}
    for name in ('train', 'validation', 'pool'):
        matches = [row for row in role['payload'] if row['path'] == name + '_nodes.npy']
        require(len(matches) == 1, 'Exact frozen role identity array required')
        record = matches[0]
        node_bindings[name] = {'path': str(role_path.parent / record['path']),
                               'sha256': record['sha256'], 'bytes': record['bytes']}
        ids[name] = np.load(bound(node_bindings[name], '.npy'), allow_pickle=False)
        nodes = ids[name]
        require(nodes.dtype == np.int64 and nodes.ndim == 1 and len(nodes) > 0
                and np.all(nodes[1:] > nodes[:-1]) and nodes.min() >= 0 and nodes.max() < meta['num_nodes']
                and len(nodes) == role['source_counts'][name], 'Sorted unique complete role IDs required')
    require(len(np.unique(np.concatenate(list(ids.values())))) == sum(len(nodes) for nodes in ids.values()),
            'Train/validation/final-pool node identities overlap')
    nodes, role_nodes = ids['validation'], node_bindings['validation']
    pack_path = bound(context['source_labels']['validation'], '.npz')
    with np.load(pack_path, allow_pickle=False) as pack:
        require(set(pack.files) == {'nodes', 'labels'}, 'Exact compact VALID fields required')
        packed_nodes, labels = pack['nodes'], pack['labels']
        require(packed_nodes.dtype == labels.dtype == np.int64 and np.array_equal(nodes, packed_nodes)
                and labels.shape == nodes.shape and ((labels >= 0) & (labels < meta['num_classes'])).all(),
                'Compact VALID pack differs from committed role IDs/classes')
        labels = labels.copy()
    return nodes, labels, {'role_freeze': context['role_freeze'], 'validation_nodes': role_nodes,
                           'all_role_node_identity_bindings': node_bindings,
                           'validation_labels': context['source_labels']['validation'], 'count': len(nodes)}


def stable_nll(np, pooled, labels):
    values = pooled.astype(np.float64)
    shifted = values - values.max(axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    nll = float(-logp[np.arange(len(labels)), labels].mean())
    require(finite(nll) and nll >= 0, 'Nonfinite/invalid recomputed VALID NLL')
    return nll


def numerical_audit(admission, driver, comparison, registry):
    import numpy as np  # Deliberately after root admission, source review and full closure.
    for context in registry['contexts']:
        require(np.__version__ == context['environment']['numpy']
                and platform.python_version() == context['environment']['python'], 'Frozen NumPy/Python runtime differs')
    contexts = {(ctx['graph'], ctx['seed']): ctx for ctx in registry['contexts']}
    expected = {(graph, seed, arm) for graph in SETTINGS for seed in SEEDS for arm in ARMS}
    require(len(comparison['rows']) == 30
            and {(r['graph'], r['seed'], r['arm']) for r in comparison['rows']} == expected, 'All30 exact source outcomes required')
    # Complete-array validation for ALL30 occurs before the first VALID label read.
    array_guards = [full_logit_guard(np, row, contexts[(row['graph'], row['seed'])]) for row in comparison['rows']]
    packs = {key: validation_pack(np, context) for key, context in contexts.items()}
    results = []
    for row in comparison['rows']:
        context = contexts[(row['graph'], row['seed'])]
        fit, directory = driver.verify_freeze(row['fit_freeze'], 'fit', context)
        require(fit['complete_declared_continuation'] is True and fit['report_eligible'] is True
                and fit['selected_checkpoint'] == row['selected_checkpoint']
                and fit['selected_checkpoint'] == descriptor(directory / 'selected_checkpoint.pt'), 'Selected checkpoint custody differs')
        selection_meta = read(directory / 'SELECTION.json')
        require(selection_meta['selection'] == fit['selection'] == row['selection']
                and row['primary_validation_nll'] == fit['selection']['primary_validation_nll']
                and selection_meta['selected_checkpoint'] == fit['selected_checkpoint']
                and selection_meta['final_pool_labels_read'] is False and selection_meta['secondary_pool_selected'] is False,
                'Saved selection metadata differs')
        selector = trace_selector(directory / 'continuation_trace.jsonl', context['backbone'], fit['selection'])
        nodes, labels, pack_binding = packs[(row['graph'], row['seed'])]
        path = bound(row['selected_logits'], '.npy')
        logits = np.load(path, allow_pickle=False, mmap_mode='r')
        z = logits[:, nodes]
        native_pool = z.mean(axis=0, dtype=np.float32)
        nll = stable_nll(np, native_pool, labels)
        diagnostic_nll = stable_nll(np, z.astype(np.float64).mean(axis=0), labels)
        saved = selector['selected_trace_NLL']
        require(abs(nll - saved) <= NATIVE_NLL_ATOL, 'Recomputed selected VALID NLL differs from native trace/selection')
        accuracy = float((native_pool.argmax(axis=-1) == labels).mean())
        # Confirm selected files/label channels remained bound during reads.
        bound(row['selected_logits'], '.npy')
        bound(context['source_labels']['validation'], '.npz')
        require(sha(directory / 'continuation_trace.jsonl') == selector['trace']['sha256'], 'Trace changed during numerical audit')
        results.append({'graph': row['graph'], 'seed': row['seed'], 'arm': row['arm'],
                        'fit_freeze': row['fit_freeze'], 'selected_logits': row['selected_logits'],
                        'selected_checkpoint': row['selected_checkpoint'], 'validation_pack': pack_binding,
                        'recomputed_native_FP32_member_mean_stable_FP64_NLL': nll,
                        'native_saved_NLL': saved, 'absolute_difference': abs(nll - saved),
                        'diagnostic_FP64_member_mean_NLL': diagnostic_nll,
                        'recomputed_native_pool_accuracy': accuracy, 'selector': selector,
                        'serialized_checkpoint_inference_replayed': False,
                        'initialization_status': row['initialization_status'], 'fallback_reason': row['fallback_reason']})
    require(len(results) == 30, 'No source arm may be pruned')
    return results, array_guards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    admission_record = descriptor(args.admission)
    admission = admission_guard(args.admission, args.output)
    driver = load_source(DRIVER, 'unchanged_graph_init_driver_v3_precision', DRIVER_SHA)
    registry, costs, bindings, failure_disclosure, source_terminals = closed_cohort(admission, driver)
    comparison, _ = driver.verify_comparison(admission['comparison_freeze'])
    require(comparison['terminal_closure'] == source_terminals and comparison['expected_contexts'] == registry['contexts'],
            'Comparison source closure differs')
    require(descriptor(args.admission) == admission_record, 'Admission changed during source guards')
    out = confined(args.output)
    require(not out.exists(), 'Once-only new v7 audit output required')
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'AUDIT_STARTED.json', {'schema': 'graph-init-companion-v7-source-valid-start-v1', 'UTC': utc(),
          'admission': admission_record, 'comparison_freeze': admission['comparison_freeze'], 'retry_allowed': False,
          'heldout_labels_authorized': False, 'Tensor_GPU_execution_authorized': False})
    started = time.monotonic()
    freeze_path = out / 'AUDIT_FREEZE.json'
    pending_freeze = out / '.AUDIT_FREEZE.pending.json'
    try:
        metrics, array_guards = numerical_audit(admission, driver, comparison, registry)
        # Repeat all admission/source/review and physical cohort checks after reads.
        require(descriptor(args.admission) == admission_record, 'Admission changed during numerical audit')
        final_admission = admission_guard(args.admission, args.output)
        require(final_admission == admission, 'Admission content changed during numerical audit')
        final_cohort = closed_cohort(final_admission, driver)
        require(final_cohort == (registry, costs, bindings, failure_disclosure, source_terminals),
                'Registry, lineage, source terminals or physical cost custody changed during audit')
        final_comparison, _ = driver.verify_comparison(final_admission['comparison_freeze'])
        require(final_comparison == comparison and final_comparison['terminal_closure'] == final_cohort[4]
                and final_comparison['expected_contexts'] == final_cohort[0]['contexts'],
                'Comparison content or final source closure changed during audit')
        require(descriptor(args.admission) == admission_record, 'Admission changed during final revalidation')
        result = {'schema': 'graph-init-companion-v7-source-valid-audit-v1', 'UTC': utc(), 'completed': True,
                  'admission': admission_record, 'source_manifest': admission['source_manifest'],
                  'comparison_freeze': admission['comparison_freeze'], 'attempt_registry': admission['attempt_registry'],
                  'source_cells': 30, 'registered_phases': 72, 'metrics': metrics, 'complete_array_guards': array_guards,
                  'phase_cost_rows': costs, 'physical_cost_bindings': bindings, 'preserved_failure': failure_disclosure,
                  'cohort_outer_supervised_seconds': sum(row['outer_supervised_seconds'] for row in costs),
                  'native_score_tolerance': {'absolute': NATIVE_NLL_ATOL, 'relative': 0.0},
                  'primary_pooling': 'softmax(mean raw member logits)',
                  'source_VALID_is_the_selection_channel': True, 'outcome_aware_exploratory_cfg0': True,
                  'new_originality_superiority_or_unseen_quality_claim': False,
                  'serialized_checkpoint_inference_replayed': False, 'historical_epoch_scores_independently_recomputed': False,
                  'original_fit_in_process_selected_state_replay': 'Best in-memory state reload and VALID replay precede selected-logit save; saved checkpoint was serialized subsequently. This audit does not reopen or infer from that checkpoint.',
                  'heldout_labels_or_final_report_opened': False, 'Torch_imported_or_GPU_used': False,
                  'source_scientific_recipes_changed': False, 'scientific_restarts_or_retries': False,
                  'audit_seconds': time.monotonic() - started,
                  'audit_seconds_scope': 'Numerical body plus post-read revalidation, excluding pre-body admission/source/lineage guards and final JSON serialization; a separate external whole-process receipt is required for complete audit wall cost.'}
        write(out / 'SOURCE_VALID_AUDIT.json', result)
        # Stage complete JSON, then atomically publish with a no-overwrite hard link.
        write(pending_freeze, {'schema': 'graph-init-companion-v7-source-valid-freeze-v1',
              'completed': True, 'audit': descriptor(out / 'SOURCE_VALID_AUDIT.json'),
              'admission': admission_record, 'comparison_freeze': admission['comparison_freeze'],
              'source_cells': 30, 'heldout_labels_opened': False, 'serialized_checkpoint_inference_replayed': False,
              'payload': [descriptor(out / name) for name in ('AUDIT_STARTED.json', 'SOURCE_VALID_AUDIT.json')]})
        with pending_freeze.open('rb') as file:
            os.fsync(file.fileno())
        os.link(pending_freeze, freeze_path)  # Atomic success commit; refuses any existing destination.
    except BaseException as error:
        # The committed freeze is authoritative inside this process. A later
        # physical exit failure remains a wrapper failure, never a second audit terminal.
        if not freeze_path.exists():
            write(out / 'AUDIT_FAILED.json', {'schema': 'graph-init-companion-v7-source-valid-failure-v1',
                  'UTC': utc(), 'completed': False, 'error_type': type(error).__name__, 'message': str(error),
                  'admission': admission_record, 'automatic_retry': False, 'heldout_labels_opened': False,
                  'Tensor_GPU_execution_authorized': False, 'seconds': time.monotonic() - started})
        raise
    # No stdout operation can change a committed success into an audit failure.
    try:
        pending_freeze.unlink()
    except OSError:
        pass  # An identical staging copy is non-authoritative if cleanup fails.


if __name__ == '__main__':
    main()
