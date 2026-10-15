"""One named NCN source repair after a pinned zero-work failure; no launch or retry."""
from __future__ import annotations
import json
from pathlib import Path
import sys
from host_binding import ALLOCATION_ROOT, HERE, guard, require, verify

ZERO_COUNTERS = (
    'TRAIN_epochs_started', 'TRAIN_epochs_completed', 'TRAIN_source_iterator_records',
    'TRAIN_task_query_records', 'VALID_attempts', 'VALID_completed',
    'native_VALID_attempts', 'native_VALID_completed', 'native_constructions_attempted',
    'native_constructions_completed', 'optimizer_attempts', 'optimizer_completed',
    'updates_attempted', 'updates_completed', 'reverse_attempts', 'reverse_completed',
    'extra_reverse_attempts', 'provider_TRAIN_hydration_calls',
    'provider_VALID_hydration_calls', 'provider_sampler_calls', 'mean_exports_attempted',
    'mean_exports_completed', 'update_cost_ledger_rows', 'automatic_retries', 'scientific_fits',
)
EMPTY_LISTS = (
    'VALID_receipts', 'epoch_receipts', 'export_receipts', 'global_RNG_restore_checks',
    'invocations', 'restore_checks', 'snapshots',
)


def validate_prior_ncn_report(supervisor, child, contract):
    """Pure validation of actual saved metadata, also used by local source checks."""
    require(supervisor['schema'] == 'portablev3-link-queued-single-case-supervisor-v1', 'Wrong prior supervisor schema')
    require(child['schema'] == 'portablev3-link-queued-native-lifecycle-result-v1', 'Wrong prior child schema')
    require(supervisor['case_id'] == child['case_id'] == 'NCN', 'Only the prior NCN case is repairable')
    require(supervisor['status'] == child['status'] == 'failed', 'Preserve the failed predecessor outcome')
    require(supervisor['native_child_started'] is True and type(supervisor['child_exit_code']) is int
            and supervisor['child_exit_code'] == 1, 'Prior child must have started and failed with exit 1')
    require(child['stage'] == 'imports' and child['failure'] == contract['failure'], 'Different failure is not covered by this source repair')
    require(supervisor['publication_commit'] == contract['prior_publication_commit'], 'Wrong failed publication')
    for report in (supervisor, child):
        require(report['interpreter']['Torch_imported'] is False, 'Prior Torch import is outside this repair')
        require(report['TEST_accessed'] is False and report['full_campaign_GO'] is False, 'Prior scope differs')
        require(type(report['scientific_fits']) is int and report['scientific_fits'] == 0
                and type(report['automatic_retries']) is int and report['automatic_retries'] == 0, 'Prior fit/retry count differs')
        require(report['job'] == contract['prior_job'], 'Prior JOB custody differs')
    require(supervisor['script'] == contract['prior_supervisor_script']
            and child['script'] == contract['prior_child_script'], 'Prior executable source differs')
    require(supervisor['one_attempt_claim'] == contract['prior_attempt_claim']
            and supervisor['child_result_receipt'] == contract['prior_child_result'], 'Prior claim/result custody differs')
    require(supervisor['last_child_report'] == child, 'Preserved child terminal differs from supervisor evidence')
    require(supervisor['source_receipts'] == supervisor['source_receipts_after'] == child['sources_before'],
            'Prior source closure changed during the failed attempt')
    for pin in contract['prior_source_receipts']:
        require(pin in child['sources_before'], 'Missing exact prior runtime/source receipt')
    require(child['limits'] == contract['limits'], 'Prior work bounds differ')
    for key in ZERO_COUNTERS:
        require(type(child[key]) is int and child[key] == 0, 'Prior native work is not zero: ' + key)
    for key in EMPTY_LISTS:
        require(child[key] == [], 'Prior native evidence is nonempty: ' + key)
    for key in ('native_forward_attempts', 'native_forwards'):
        require(child[key] == {'TRAIN': 0, 'VALID': 0}
                and all(type(v) is int for v in child[key].values()), 'Prior native forwards are not zero')
    require(child['update_cost_chain_sha256'] == '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
            'Prior update-cost ledger is not empty')
    return {'repair_attempt_id': contract['repair_attempt_id'], 'prior_child_started': True,
            'prior_native_work': 0, 'prior_outcome': 'failed',
            'prior_attempt_claim': contract['prior_attempt_claim'], 'prior_child_result': contract['prior_child_result'],
            'prior_observation': contract['prior_observation'], 'automatic_retries': 0}


def validate_repair_admission(admission, contract):
    """Supplement the driver's existing publication, resources and finite-case checks."""
    require(admission.get('case_id') == 'NCN', 'Repair admission is only for NCN')
    require(admission.get('repair_attempt_id') == contract['repair_attempt_id'], 'Explicit named repair identity required')
    require(admission.get('prior_failed_attempt') == contract['admission_prior_failed_attempt'], 'Exact prior failure lineage required')
    require(admission.get('output') == contract['repair_output'], 'Use the distinct one-shot repair output')
    require(admission.get('attempt_claim') == contract['repair_attempt_claim'], 'Use the distinct one-shot repair claim')
    require(admission.get('job_sha256') == contract['prior_job']['sha256'] and admission.get('limits') == contract['limits'],
            'Same original queued NCN JOB and work bounds required')
    require(type(admission.get('maximum_attempts')) is int and admission['maximum_attempts'] == 1
            and type(admission.get('automatic_retries')) is int and admission['automatic_retries'] == 0,
            'One repair attempt; no automatic retry')


def verify_prior_ncn_failure(root):
    """Guard first; read only pinned JSON/source, preserving every predecessor file."""
    require(str(root) == ALLOCATION_ROOT, 'This repair applies only to the fixed allocation')
    require('torch' not in sys.modules, 'Verify repair authority before tensor imports')
    guard()
    contract = json.loads((HERE / 'NCN_REPAIR_CONTRACT.json').read_text())
    observation = verify(contract['prior_observation'], root)
    child = verify(contract['prior_child_result'], root)
    verify(contract['prior_attempt_claim'], root)
    verify(contract['prior_job'], root)
    for pin in contract['prior_source_receipts']:
        verify(pin, root, parse_json=False)
    result = validate_prior_ncn_report(observation['supervisor'], child, contract)
    return contract, result
