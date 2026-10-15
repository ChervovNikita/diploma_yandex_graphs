"""TEST scalar normalizer and accepted reporter bridge; JSON metadata only.

The VALID-only extract_runtime/extract_i4 functions are deliberately unused.
The accepted report() function receives explicit TEST rows. All expected rows,
failed attempts, absent results and unresolved costs remain visible.
"""
import copy
from pathlib import Path
from .common import EXPANSION, digest, pin, require, source_module, verify


def _json(path, root):
    return verify(pin(path, root), root, json_metadata=True)


def _observer(task_id, final_root, raw, root):
    folder = Path(final_root) / 'control/tasks' / digest(task_id) / 'attempts'
    matches = []
    for path in sorted(folder.glob('*/SCORING_OBSERVATION.json')):
        observation = _json(path, root)
        require(observation['task_id'] == task_id and observation['producer'] == pin(EXPANSION / 'final_scoring_integration_v1/runtime_entry.py', root), 'Final observer source/task differs')
        if observation.get('scalar_RESULT') == raw and observation.get('RESULT'):
            verify(observation['RESULT'], root, json_metadata=True)
            matches.append(observation)
    require(len(matches) <= 1, 'Multiple final observations require explicit disposition, never hidden retry selection')
    return matches[0] if matches else None


def _base(identity, raw, training, spec):
    row = copy.deepcopy(identity)
    row['declared_bindings'] = row.pop('bindings', {})
    row.update(schema='exploratory-predictor-row-v1', role='TEST', scope='expanded_exploratory',
               bindings={}, status='not_started' if raw is None else raw['status'],
               metadata_placeholder=raw is None, selected_served_metric=None, native_proper_loss=None,
               selected_epoch=None if raw is None else raw.get('selected_epoch'),
               selected_checkpoint=None if raw is None else raw.get('selected_checkpoint'),
               selected_export=None if raw is None else raw.get('TEST_export'),
               trained_epochs=(training or {}).get('completed_epochs', 0), stop_reason=(training or {}).get('stop_reason'),
               convergence_status='source_horizon_separate' if identity['track'] == 'source_reference' else
                   'declared_matched_patience_satisfied' if (training or {}).get('stop_reason') == 'patience' else 'unresolved',
               own_selection=False, descriptor_hash=spec.get('job_hash'),
               runtime_binding=(training or {}).get('runtime_binding'),
               raw_final_result=copy.deepcopy(raw), raw_training_result=copy.deepcopy(training),
               costs={'physical_fit_count': 1, 'additional_final_training_fits': 0,
                      'training_cost_reference': identity['fit_id'], 'lifetime_total': None})
    return row


def normalize_fit(spec, task, raw, training, observer, root, go_pin, seal_pin):
    row = _base(spec['identity'], raw, training, spec)
    if raw is None:
        return row
    require(raw['schema'] == 'final-selected-fit-scoring-result-v2' and raw['task_id'] == task['id']
            and raw['identity'] == spec['identity'] and raw['job_hash'] == spec['job_hash'] and raw['role'] == 'TEST'
            and raw['campaign_GO'] == go_pin and raw['training_seal'] == seal_pin
            and raw['physical_training_fits'] == raw['optimizer_updates'] == 0, 'Final fit scalar authority differs')
    if raw['status'] != 'complete':
        return row
    require(training is not None and training['status'] == 'complete' and training['TEST_accessed'] is False
            and training['job_id'] == task['training_fit_id'] and training['job_hash'] == spec['job_hash']
            and raw['selected_checkpoint'] == training['selected_checkpoint'] and raw['selected_epoch'] == training['selected_epoch']
            and raw['TEST_accessed'] is True and raw['role_contract'] == task['role_contract'], 'Final score changed its selected state/role')
    qualification = verify(raw['VALID_qualification'], root, json_metadata=True)
    require(qualification['passed'] is True and qualification['role'] == 'VALID' and qualification['TEST_accessed'] is False
            and qualification['selected_checkpoint'] == raw['selected_checkpoint']
            and qualification['recorded_VALID_export'] == training['selected_VALID_export'], 'Own complete VALID qualification differs')
    row.update(selected_served_metric=raw['metrics'].get(row['endpoint']['name']), native_proper_loss=raw['diagnostic_loss'],
               own_selection=True, final_role_contract=task['role_contract'], VALID_qualification=raw['VALID_qualification'])
    if observer is not None:
        row['bindings'] = copy.deepcopy(spec['bindings'])
        row['binding_evidence'] = {'final_observer': observer['producer'], 'request': observer['request'],
                                   'role_contract': task['role_contract'], 'selected_TEST_export': raw['TEST_export']}
    return row


def normalize_i4(spec, task, raw, components, observer, go_pin, seal_pin):
    row = _base(spec['identity'], raw, None, spec)
    row.update(components=copy.deepcopy(components), M1_exact_member0_fit_id=spec['component_fit_ids'][0],
               aggregation='mean_raw_logits_after_four_own_selectors', extra_training_fits=0,
               selected_checkpoint=None, selected_epoch=None, stop_reason='four_own_selected_components',
               costs={'physical_fit_count': 0, 'additional_final_training_fits': 0, 'aggregation_cost': None})
    require([part['fit_id'] for part in components] == spec['component_fit_ids'], 'I4 frozen component order differs')
    if raw is None or raw['status'] != 'complete':
        return row
    require(raw['schema'] == 'final-I4-TEST-scoring-result-v2' and raw['task_id'] == task['id'] and raw['role'] == 'TEST'
            and raw['identity'] == spec['identity'] and raw['role_contract'] == task['role_contract']
            and raw['campaign_GO'] == go_pin and raw['training_seal'] == seal_pin and raw['TEST_accessed'] is True
            and raw['physical_training_fits'] == raw['optimizer_updates'] == raw['native_forwards'] == 0
            and raw['selected_epochs'] == [part['selected_epoch'] for part in components]
            and all(part['status'] == 'complete' for part in components), 'I4 final scalar components/selection differ')
    require(raw['components'] == [part['final_RESULT'] for part in components], 'I4 references different final component results')
    row.update(selected_served_metric=raw['metrics'].get(row['endpoint']['name']), native_proper_loss=raw['diagnostic_loss'],
               own_selection=True, final_role_contract=task['role_contract'])
    if observer:
        for field in ('native_source', 'data', 'controller', 'evaluator'):
            values = [part['bindings'].get(field) for part in components]
            require(all(value is not None and value == values[0] for value in values), 'I4 component common bindings unresolved')
        row['bindings'] = copy.deepcopy(spec['identity']['bindings'])
        row['binding_evidence'] = {'final_observer': observer['producer'], 'request': observer['request'],
                                   'role_contract': task['role_contract'], 'selected_TEST_export': raw['TEST_export']}
    return row


def collect_final_costs(tasks, final_root, root):
    ledger = []
    for task in tasks:
        folder = Path(final_root) / 'control/tasks' / digest(task['id']) / 'attempts'
        for start in sorted(folder.glob('*/START.json')):
            observation, terminal = start.parent / 'SCORING_OBSERVATION.json', start.parent / 'TERMINAL.json'
            obs = _json(observation, root) if observation.is_file() else None
            term = _json(terminal, root) if terminal.is_file() else None
            source = pin(observation if obs else terminal if term else start, root)
            raw = (obs or {}).get('scalar_RESULT')
            ledger.append({'attempt_id': 'final/' + digest(pin(start, root)), 'fit_id': task.get('training_fit_id', task.get('training_I4_id')),
                'final_task_id': task['id'], 'physical_fit_count': 0,
                'worker_wall_seconds': obs['worker_wall_seconds'] if obs else (term or {}).get('observed_launch_to_reap_seconds'),
                'status': (raw or term or {}).get('status', 'unresolved_no_terminal'), 'source': source,
                'recorded_native_worker_wall_seconds': (raw or {}).get('worker_wall_seconds'),
                'native_wall_is_contained_or_overlapping_not_added_twice': True,
                'unknown_crashed_or_uncovered_cost': None if term else 'unknown, retained; never zero',
                'TEST_accessed': (raw or term or {}).get('TEST_accessed')})
    return ledger


def collect_profile(profile, roster, training_root, final_root, root, go_pin, seal_pin, *, training_cost_ledger=None, host=None):
    """Return all expected role rows and report; caller has authenticated sources.

    Pass the accepted TRAIN collector's complete cost ledger when available.
    Missing TRAIN cost custody is marked unresolved rather than assumed zero.
    """
    tasks = {row['id']: row for row in roster['tasks']}
    fits, rows = {}, []
    for fit_id, spec in profile['fit_specs'].items():
        task = tasks['TEST/' + fit_id]
        result_path = Path(final_root) / task['output_relative'] / 'RESULT.json'
        training_path = Path(training_root) / spec['output_relative'] / 'RESULT.json'
        raw = _json(result_path, root) if result_path.is_file() else None
        training = _json(training_path, root) if training_path.is_file() else None
        observer = _observer(task['id'], final_root, raw, root) if raw else None
        row = normalize_fit(spec, task, raw, training, observer, root, go_pin, seal_pin)
        row['final_RESULT'] = pin(result_path, root) if raw else None
        fits[fit_id] = row
        if spec['identity']['scientific_method_id'] == 'native_M1' or spec['identity']['runtime_procedure_id'] != 'native_single':
            rows.append(row)
        if 'same_bank_mean_identity' in spec:
            diagnostic = copy.deepcopy(row)
            diagnostic.update(spec['same_bank_mean_identity'])
            diagnostic['declared_bindings'] = diagnostic.pop('bindings', {})
            diagnostic['bindings'] = {}
            mean = (raw or {}).get('same_selected_bank_mean')
            if mean:
                require(mean['identity'] == spec['same_bank_mean_identity'] and mean['selected_checkpoint'] == row['selected_checkpoint']
                        and mean['selected_epoch'] == row['selected_epoch'] and mean['diagnostic_only'] is True
                        and mean['extra_forward_calls'] == 0 and mean['selection_changed'] is False, 'Mean diagnostic changed own selected state/forwards')
                diagnostic['selected_served_metric'] = mean['metrics'].get(row['endpoint']['name']) if raw['status'] == 'complete' else None
                diagnostic['native_proper_loss'] = mean['diagnostic_loss']
                if row['bindings']:
                    diagnostic['bindings'] = copy.deepcopy(spec['same_bank_mean_identity']['bindings'])
            else:
                diagnostic.update(status='missing_same_bank_diagnostic', selected_served_metric=None, native_proper_loss=None, own_selection=False)
            diagnostic.update(diagnostic_only=True, costs={'physical_fit_count': 0, 'same_physical_fit_cost_reference': fit_id})
            rows.append(diagnostic)
    for task_id, spec in profile['I4_specs'].items():
        task = tasks['TEST/' + task_id]
        result_path = Path(final_root) / task['output_relative'] / 'RESULT.json'
        raw = _json(result_path, root) if result_path.is_file() else None
        observer = _observer(task['id'], final_root, raw, root) if raw else None
        components = [fits[key] for key in spec['component_fit_ids']]
        row = normalize_i4(spec, task, raw, components, observer, go_pin, seal_pin)
        row['final_RESULT'] = pin(result_path, root) if raw else None
        rows.append(row)
    require(len(rows) == len(profile['registry']['expected_predictors']), 'Do not omit missing expected role rows')
    reporter = source_module(EXPANSION / 'benchmark_reporting_implementation_v1/reporter.py', '_accepted_final_scalar_reporter')
    final_tasks = [task for task in tasks.values() if task['preset_id'] == profile['registry']['presets'][0]['id']]
    costs = copy.deepcopy(training_cost_ledger or []) + collect_final_costs(final_tasks, final_root, root)
    if training_cost_ledger is None:
        costs.append({'attempt_id': 'TRAIN-cost-custody/' + profile['registry']['presets'][0]['id'], 'fit_id': None,
                      'worker_wall_seconds': None, 'status': 'complete_TRAIN_cost_ledger_not_supplied',
                      'source': None, 'unknown_cost_is_not_zero': True})
    report = reporter.report(profile['registry'], rows, costs, profile['contrast_plan'])
    by_key = {reporter.row_key(row): row for row in rows}
    for value in report['all_expected_predictor_statuses']:
        actual = by_key[reporter.row_key(value['identity'])]
        value['observed'] = actual['final_RESULT'] is not None
        value['explicit_role_status'] = actual['status']
    report.update(schema='final-TEST-scalar-report-v1', scored_role='TEST', own_selection_role='VALID',
                  report_reads_JSON_only=True, TEST_tensor_access_by_collector=False,
                  TEST_scores_are_exploratory_not_confirmation=True, source_reporter=pin(reporter.__file__, root),
                  complete_TRAIN_cost_ledger_supplied=training_cost_ledger is not None, host=host)
    return report


def merge_profile_reports(report_pins, ownership, reference_profile, root):
    """Compact metadata-only reporting merge; never relocates model state.

    Each real row comes from its immutable TRAIN/final-serving host. Other
    hosts' explicit placeholders are not interpreted as observed failures.
    The accepted reporter recomputes all contrasts from the complete row set.
    """
    owners = {row['id']: row['worker'].split('/')[0] for row in ownership['assignments']}
    required = set(owners.values())
    require(set(report_pins) == required, 'Every nonempty host report required')
    reports = {host: verify(value, root, json_metadata=True) for host, value in report_pins.items()}
    reporter = source_module(EXPANSION / 'benchmark_reporting_implementation_v1/reporter.py', '_accepted_merged_final_scalar_reporter')
    indexed = {}
    for host, report in reports.items():
        require(report['schema'] == 'final-TEST-scalar-report-v1' and report['host'] == host
                and report['scored_role'] == 'TEST' and report['own_selection_role'] == 'VALID', 'Wrong host role report')
        indexed[host] = {reporter.row_key(row): row for row in report['raw_rows']}
        require(len(indexed[host]) == len(reference_profile['registry']['expected_predictors']), 'Host report omitted expected role rows')
    rows, costs = [], []
    for identity in reference_profile['registry']['expected_predictors']:
        host = owners[identity['fit_id']]
        row = copy.deepcopy(indexed[host][reporter.row_key(identity)])
        if identity['scientific_method_id'] == 'native_I4':
            require(all(owners[part['fit_id']] == host for part in row['components']), 'I4 cannot move TEST tensors across hosts')
        row['physical_owner_host'] = host
        rows.append(row)
    for host, report in reports.items():
        for record in report['cost_ledger']:
            if record.get('fit_id') is not None and owners[record['fit_id']] != host:
                continue
            item = copy.deepcopy(record)
            item['origin_attempt_id'] = item['attempt_id']
            item['attempt_id'] = host + '/' + item['attempt_id']
            item['physical_owner_host'] = host
            costs.append(item)
    merged = reporter.report(reference_profile['registry'], rows, costs, reference_profile['contrast_plan'])
    lookup = {reporter.row_key(row): row for row in rows}
    for value in merged['all_expected_predictor_statuses']:
        row = lookup[reporter.row_key(value['identity'])]
        value.update(observed=row['final_RESULT'] is not None, explicit_role_status=row['status'])
    merged.update(schema='offline-merged-final-TEST-scalar-report-v1', scored_role='TEST', own_selection_role='VALID',
                  host_reports=report_pins, TEST_tensor_access_by_collector=False, checkpoint_or_logit_transfer=False,
                  complete_owner_partition=True, live_coordinator_used=False,
                  TEST_scores_are_exploratory_not_confirmation=True)
    return merged
