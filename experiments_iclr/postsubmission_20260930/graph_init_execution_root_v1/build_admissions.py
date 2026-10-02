"""Build exact unapproved metadata drafts; finalize only from separate root decisions.

No scientific module, array, label pack or checkpoint is opened or imported.
Outputs are exclusive. Initial allowed actions are register and Squirrel17 qualify.
"""
import argparse
import copy
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from admission_support import (HERE, PHASE, REMOTE_PHASE, REMOTE_REPO, R17_REL, MODERN_REL,
    PREPARATION_SHA, CELLS, SEEDS, DISCLOSURE, require, confined, mirror, sha, read, remote,
    descriptor, bound, write, source_descriptors, verify_sources, expected_attempts, object_hash, bindings)

CERT_ROOT = PHASE/'modern_teacher_execution_root_v1/certification_v3/issued_v1'
STUDY_ID = 'graph_init_cfg0_outcome_aware_v1'
ANCHOR = str(REMOTE_PHASE/'graph_init_execution_root_v1/study_v1')
GIB = 1024**3
CAPS = {'Squirrel': {'qualify': 1800, 'warm': 900, 'initialize': 1800, 'fit': 6000},
        'Photo': {'qualify': 3600, 'warm': 3600, 'initialize': 3600, 'fit': 15000}}


def new_output(path):
    path = mirror(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def forecast(graph):
    caps = CAPS[graph]
    return {'whole_graph_peak_bytes': (60 if graph == 'Squirrel' else 75)*GIB,
        'wall_seconds_by_phase': copy.deepcopy(caps),
        'total_budget_seconds': caps['qualify']+caps['warm']+5*(caps['initialize']+caps['fit']),
        'available_memory_bytes': 75*GIB}


def check_forecast(value):
    require(set(value) == {'whole_graph_peak_bytes', 'wall_seconds_by_phase', 'total_budget_seconds',
                           'available_memory_bytes'}, 'Exact forecast keys required')
    require(set(value['wall_seconds_by_phase']) == {'qualify', 'warm', 'initialize', 'fit'}, 'Exact phase caps required')
    scalars = [value[k] for k in ('whole_graph_peak_bytes', 'total_budget_seconds', 'available_memory_bytes')]
    scalars += list(value['wall_seconds_by_phase'].values())
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in scalars), 'Positive finite forecasts required')
    require(value['whole_graph_peak_bytes'] <= value['available_memory_bytes'] <= 75*GIB,
            'Declared usable-memory budget cannot exceed75GiB')
    require(all(v <= 28800 for v in value['wall_seconds_by_phase'].values()), 'Each whole-phase cap must fit bounded supervisor')
    c = value['wall_seconds_by_phase']
    require(c['qualify']+c['warm']+5*(c['initialize']+c['fit']) <= value['total_budget_seconds'],
            'Full five-arm cell schedule must fit admitted budget')


def protected_metadata():
    records = source_descriptors()
    for relative in ('graph_init_execution_root_v1/graph_init_root_entry.py',
                     'graph_init_execution_root_v1/admission_support.py',
                     'protocols/launch_modern_root_v1.py', 'protocols/bounded_run_v1.py',
                     'protocols/run_authorized_v2.py', 'protocols/run_logged.py', 'protocols/repo_env.sh',
                     'protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json',
                     'modern_teacher_execution_root_v1/certification_v3/issued_v1/INDEX.json',
                     'modern_teacher_execution_root_v1/certification_v3/issued_v1/ISSUE_TERMINAL.json',
                     'modern_teacher_execution_root_v1/source_pack_audit_run03/SOURCE_LABEL_BINDING.json'):
        records.append(descriptor(PHASE/relative))
    # Preserve protected metadata from all six original native qualification requests.
    index = read(CERT_ROOT/'INDEX.json')
    for row in index['rows']:
        records.append(row['modern_qualification_certificate'])
        if row['seed'] != 17:
            continue
        certificate = read(bound(row['modern_qualification_certificate']))
        original = read(bound(certificate['partial_run_linkage']['binding']['original_root_request']))
        records.extend(original['protected_files'])
    unique = {}
    for record in records:
        bound(record)
        require(record['path'] not in unique or unique[record['path']] == record, 'Contradictory protected descriptor')
        unique[record['path']] = record
    return [unique[key] for key in sorted(unique)]


def decision_base(schema):
    return {'schema': schema, 'approved': False, 'approved_by': '', 'approved_utc': None,
        'r17_source_manifest': source_descriptors()[0],
        'entry_source': descriptor(HERE/'graph_init_root_entry.py'),
        'support_source': descriptor(HERE/'admission_support.py'),
        'independent_source_audit_accepted': False,
        'independent_source_audit': {'path': None, 'sha256': None},
        'full_fit_authorized': False, 'heldout_scoring_authorized': False,
        'registration_authorized': False, 'cold_qualification_authorized': False}


def launch_request(action, payload, decision, outdir, authorized=False):
    # Root receipts and outer/inner supervision are separate from driver output.
    base = REMOTE_PHASE/'graph_init_execution_root_v1'
    if action == 'register':
        output = str(Path(ANCHOR)/'GRAPH_INIT_ATTEMPT_REGISTRY.json')
        cap = 600
        name = 'register_v1'
    else:
        output = str(Path(ANCHOR)/'qualify/Squirrel_seed17')
        cap = 1800
        name = 'qualify_Squirrel_seed17_v1'
    return {'schema': 'graph-init-root-launch-request-v1', 'root_admitted': authorized, 'action': action,
        'full_fit_admitted': False, 'heldout_scoring_admitted': False,
        'source_packet': str(REMOTE_PHASE/R17_REL), 'source_manifest': source_descriptors()[0],
        'source_seals': source_descriptors(), 'protected_files': protected_metadata(),
        'entry_script': str(base/'graph_init_root_entry.py'),
        'driver_script': str(REMOTE_PHASE/R17_REL/'prototype/graph_init_driver.py'),
        'allocation_route': descriptor(PHASE/'protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json'),
        'root_decision': descriptor(decision), 'phase_payload': descriptor(payload),
        'output': output, 'receipt_directory': str(base/'root_receipts'/name),
        'supervisor_directory': str(base/'supervision'/(name+'_inner')),
        'outer_supervisor_directory': str(base/'supervision'/(name+'_outer')),
        'local_launch_receipt': str(base/'supervision'/(name+'_LAUNCH.json')),
        'whole_cap_seconds': cap, 'deterministic': True,
        'heldout_release_requires_all72_completed_and_separate_admission': True}


def lineage(contexts, evidence, approved=False):
    source, protocol = bindings()
    return {'schema': 'graph-init-version-lineage-authorization-v1', 'approved': approved,
        'study_id': STUDY_ID, 'anchor_directory': ANCHOR, 'contexts_sha256': object_hash(contexts),
        'source_bindings': source, 'protocol': protocol, 'prior_attempt_registries': [],
        'predecessor_history_complete': approved, 'evidence': evidence}


def registry_request(contexts, lineage_path, approved=False):
    return {'schema': 'graph-init-registry-request-v1', 'registration_authorized': approved,
        'study_id': STUDY_ID, 'contexts': contexts, 'anchor_directory': ANCHOR,
        'prior_attempt_registries': [], 'lineage_authorization': descriptor(lineage_path)}


def qualify_admission(context, registry_record, approved=False):
    return {'schema': 'graph-init-phase-admission-v1', 'execution_authorized': approved,
        'authorized_phase': 'qualify', 'context': context, 'dependencies': {}, 'arm': None,
        'attempt_registry': registry_record}


def prepare(output):
    verify_sources()
    out = new_output(output)
    index = read(CERT_ROOT/'INDEX.json')
    require(index['schema'] == 'modern-issued-certificate-index-v3' and len(index['rows']) == 18,
            'Exact issued18-certificate index required')
    costs = []
    evidence = [descriptor(CERT_ROOT/'INDEX.json'), descriptor(CERT_ROOT/'ISSUE_TERMINAL.json'),
        descriptor(PHASE/'modern_teacher_execution_root_v1/SIX_QUALIFICATION_AUDIT_v1.json'),
        descriptor(PHASE/R17_REL/'COSTS_AND_FAILURES.md'),
        descriptor(PHASE/R17_REL/'prototype/graph_init_training_adapter.py')]
    for row in index['rows']:
        if row['seed'] != 17:
            continue
        certificate = read(bound(row['modern_qualification_certificate']))
        partial = bound(certificate['partial_receipt'])
        cost_path = partial.parent/'COSTS.json'
        cost = read(cost_path)
        terminal = certificate['partial_run_linkage']['binding']['root_terminal']
        cell = certificate['partial_run_linkage']['binding']['teacher_cell_freeze']
        bound(terminal); bound(cell)
        evidence.extend([descriptor(cost_path), terminal, cell])
        costs.append({'backbone': row['backbone'], 'family': row['family'], 'tested_seed': 17,
            'tested_config': 0, 'native_qualification_updates': 3 if row['backbone'] == 'polyformer_mono' else 4,
            'costs': descriptor(cost_path), 'root_terminal': terminal, 'teacher_cell_freeze': cell,
            'measured_operations': cost})
    forecast_evidence = {'schema': 'graph-init-prospective-resource-evidence-v1',
        'status': 'UNADMITTED_PROSPECTIVE_RESERVATION', 'native_measured_costs': costs,
        'forecast_by_graph': {g: forecast(g) for g, _ in CELLS},
        'usable_memory_bytes_semantics': 'Proposed75GiB allocation budget; not a measurement of current free memory.',
        'peak_semantics': 'Conservative proposed60GiB/75GiB reservations; source-copy review and cold measurements remain required.',
        'time_semantics': 'Whole-phase supervised caps, including guards/preprocessing/copies/AD and receipts; no measured R17 stage rates.',
        'source_copy_scope': 'native disposable state plus K1 and K4 clones, populated Adam snapshots, frozen/local handoff copies, functional closure AD/JVP/VJP/SpMM/Armijo trial tensors and installation; all charged.',
        'full_schedule_updates': {'Squirrel_warm': 50, 'Photo_warm': '200local+50global',
            'Squirrel_each_K4_continuation': 1950, 'Photo_each_K4_global_continuation': 950,
            'arms_per_context': 5, 'contexts': 6},
        'native_short_costs_establish_R17_AD_cost': False, 'full_schedule_feasibility_passed': False,
        'root_source_copy_and_memory_review_required': True, 'measured_AD_cost_or_low_cost_claim': False,
        'evidence': evidence}
    write(out/'PROSPECTIVE_RESOURCE_EVIDENCE.json', forecast_evidence)
    bundle_path = PHASE/'modern_teacher_execution_root_v1/source_pack_audit_run03/SOURCE_LABEL_BINDING.json'
    bundle = read(bundle_path)
    source, protocol = bindings()
    contexts = []
    for graph, backbone in CELLS:
        for split, seed in enumerate(SEEDS):
            rows = [row for row in index['rows'] if (row['backbone'], row['family'], row['seed'], row['config']) ==
                    (backbone, 'gnnm_boundary_4', seed, 0)]
            require(len(rows) == 1, 'One exact GNNM certificate per context required')
            row = rows[0]
            certificate = read(bound(row['modern_qualification_certificate']))
            identity = certificate['input_identity']
            require(certificate['schema'] == 'modern-teacher-qualification-certificate-v2' and
                    certificate['admission_eligible'] is True and certificate['report_eligible'] is False and
                    certificate['family'] == 'gnnm_boundary_4' and certificate['qualified_config'] == 0 and
                    certificate['numerical_tested_seeds'] == [17] and
                    certificate['numerical_tests_on_target_seed'] is (seed == 17) and
                    certificate['numerical_tested_configurations'] == [0] and
                    certificate['registered_configuration_scope'] == [0, 1, 2, 3] and
                    identity == certificate['authorized_target_input'] == row['input_identity'],
                    'Exact eligible certificate identity/scope required')
            pair = [pair for pair in bundle['pairs'] if pair['backbone'] == backbone and pair['seed'] == seed]
            require(len(pair) == 1 and pair[0]['independently_extracted_and_row_verified'] is True,
                    'Independent verified source pair required')
            for key in ('role_freeze', 'graph_input', 'source_labels', 'provider_row_identity', 'extraction_provenance'):
                require(identity[key] == pair[0][key], 'Preserved original source descriptors differ')
            role = read(bound(identity['role_freeze']))
            graph_meta = read(bound(identity['graph_input']))
            bound(identity['provider_row_identity']); bound(identity['extraction_provenance'])
            require(role['labels_read'] is False and role['role_derivation_version'] == 'derived_roles_v2' and
                    role['preparation_driver_sha256'] == PREPARATION_SHA and role['source_split_index'] == split and
                    role['graph'] == graph and role['seed'] == seed and role['graph_input'] == identity['graph_input'],
                    'Label-blind role/source split differs')
            require(identity['source_label_binding'] == descriptor(bundle_path) and graph_meta['graph'] == graph,
                    'Exact graph/bundle binding required')
            context = {'schema': 'graph-init-cell-context-v1', 'graph': graph, 'backbone': backbone,
                'seed': seed, 'source_split_index': split, 'config': 0,
                'role_freeze': copy.deepcopy(identity['role_freeze']),
                'graph_input': copy.deepcopy(identity['graph_input']),
                'source_labels': copy.deepcopy(identity['source_labels']),
                'environment': copy.deepcopy(certificate['environment']), 'device': certificate['environment']['device'],
                'source_bindings': source, 'protocol': protocol,
                'source_label_binding': copy.deepcopy(identity['source_label_binding']),
                'modern_qualification_certificate': copy.deepcopy(row['modern_qualification_certificate']),
                'roles_frozen_before_label_extraction': True, 'prior_exposure_disclosure': DISCLOSURE,
                'independent_of_stage1_outcomes': True,
                'resource_forecast': {'admitted': False, 'forecast': forecast(graph),
                    'evidence': [descriptor(out/'PROSPECTIVE_RESOURCE_EVIDENCE.json')]+evidence}}
            check_forecast(context['resource_forecast']['forecast'])
            contexts.append(context)
    write(out/'CONTEXTS_DRAFT.json', contexts)
    inventory = {'schema': 'graph-init-prospective-attempt-inventory-v1', 'registration_authorized': False,
        'draft_contexts_sha256': object_hash(contexts), 'anchor_directory': ANCHOR,
        'attempts': expected_attempts(contexts, ANCHOR),
        'identities_are_draft': True,
        'finalization_note': 'Admitting resource_forecast changes context hashes. Finalize once before registry creation; recompute all72keys. Output names remain fixed. No hidden retry.'}
    write(out/'PROSPECTIVE_72_ATTEMPTS_DRAFT.json', inventory)
    history = {'schema': 'graph-init-source-only-predecessor-history-v1', 'disclosure': DISCLOSURE,
        'prior_R17_numerical_attempt_registries': [], 'R17_numerical_attempts_claimed': 0,
        'predecessor_history_complete': False,
        'root_independent_lineage_review_required': True,
        'evidence': [descriptor(PHASE/R17_REL/'SOURCE_BINDINGS.json'),
            descriptor(PHASE/'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v1/MANIFEST.json'),
            descriptor(PHASE/R17_REL/'review_evidence/FINDINGS.json'),
            descriptor(PHASE/'modern_teacher_execution_root_v1/certification_v3/prepared_v1/HISTORY_BINDINGS.json')]}
    write(out/'PREDECESSOR_HISTORY_DRAFT.json', history)
    evidence_lineage = [descriptor(out/'PREDECESSOR_HISTORY_DRAFT.json'), descriptor(CERT_ROOT/'INDEX.json'),
                       descriptor(CERT_ROOT/'ISSUE_TERMINAL.json')]+source_descriptors()
    write(out/'LINEAGE_DRAFT.json', lineage(contexts, evidence_lineage))
    write(out/'REGISTRY_REQUEST_DRAFT.json', registry_request(contexts, out/'LINEAGE_DRAFT.json'))
    registration = decision_base('graph-init-root-registration-decision-v1')
    registration.update({'study_id': STUDY_ID, 'anchor_directory': ANCHOR,
        'prepared_contexts': descriptor(out/'CONTEXTS_DRAFT.json'), 'draft_contexts_sha256': object_hash(contexts),
        'resource_forecasts_accepted': False, 'source_copy_memory_review_accepted': False,
        'predecessor_history_complete': False, 'predecessor_history': descriptor(out/'PREDECESSOR_HISTORY_DRAFT.json'),
        'accepted_forecasts': [{'graph': c['graph'], 'seed': c['seed'], 'forecast': c['resource_forecast']['forecast']} for c in contexts],
        'root_review_note': 'Fresh independent sealed-R17-v2 source audit, exact descriptors, source-copy memory and prospective costs require root acceptance. No R17 AD/full-fit feasibility claimed.'})
    write(out/'ROOT_REGISTRATION_DECISION_TEMPLATE.json', registration)
    first = contexts[0]
    registry_record = {'path': str(Path(ANCHOR)/'GRAPH_INIT_ATTEMPT_REGISTRY.json'), 'sha256': None}
    write(out/'SQUIRREL17_QUALIFY_ADMISSION_DRAFT.json', qualify_admission(first, registry_record))
    qualification = decision_base('graph-init-root-cold-qualification-decision-v1')
    qualification.update({'study_id': STUDY_ID, 'attempt_registry': registry_record,
        'context_sha256': object_hash(first), 'output': str(Path(ANCHOR)/'qualify/Squirrel_seed17'),
        'whole_cap_seconds': 1800, 'context_hash_must_be_replaced_after_registration': True,
        'scope': 'TRAIN-only disposable full-graph cold qualification; no useful checkpoint, no continuation or report admission.'})
    write(out/'ROOT_COLD_QUALIFICATION_DECISION_TEMPLATE.json', qualification)
    write(out/'ROOT_REGISTER_REQUEST_DRAFT.json', launch_request('register', out/'REGISTRY_REQUEST_DRAFT.json',
          out/'ROOT_REGISTRATION_DECISION_TEMPLATE.json', out))
    write(out/'ROOT_SQUIRREL17_QUALIFY_REQUEST_DRAFT.json', launch_request('qualify', out/'SQUIRREL17_QUALIFY_ADMISSION_DRAFT.json',
          out/'ROOT_COLD_QUALIFICATION_DECISION_TEMPLATE.json', out))
    write(out/'PREPARE_RECEIPT.json', {'schema': 'graph-init-metadata-prepare-receipt-v1',
        'UTC': datetime.now(timezone.utc).isoformat(), 'contexts': 6, 'prospective_attempts': 72,
        'draft_contexts_sha256': object_hash(contexts), 'scientific_import_or_execution': False,
        'arrays_labels_or_checkpoints_opened': False, 'all_execution_and_resource_approvals_false': True,
        'source_only_independence_asserted_under_outcome_aware_disclosure': True})
    return out


def approved_decision(path, schema):
    value = read(path)
    require(value['schema'] == schema and value['approved'] is True and value['approved_by'].strip() and
            value['approved_utc'] and value['independent_source_audit_accepted'] is True and
            value['r17_source_manifest'] == source_descriptors()[0] and
            value['entry_source'] == descriptor(HERE/'graph_init_root_entry.py') and
            value['support_source'] == descriptor(HERE/'admission_support.py') and
            value['full_fit_authorized'] is False and value['heldout_scoring_authorized'] is False,
            'Explicit signed root source/initial-action decision required')
    bound(value['independent_source_audit'])
    return value


def finalized_contexts(decision):
    contexts = read(bound(decision['prepared_contexts']))
    require(object_hash(contexts) == decision['draft_contexts_sha256'] and len(contexts) == 6,
            'Signed exact draft context array required')
    accepted = {(r['graph'], r['seed']): r['forecast'] for r in decision['accepted_forecasts']}
    require(len(decision['accepted_forecasts']) == len(accepted) == 6 and
            set(accepted) == {(g, s) for g, _ in CELLS for s in SEEDS}, 'Six exact admitted forecasts required')
    final = copy.deepcopy(contexts)
    for context in final:
        value = accepted[(context['graph'], context['seed'])]
        check_forecast(value)
        require(context['resource_forecast']['admitted'] is False, 'Only original unapproved drafts accepted')
        context['resource_forecast']['admitted'] = True
        context['resource_forecast']['forecast'] = value
        context['resource_forecast']['evidence'] += [decision['independent_source_audit']]
    return final


def finalize_registration(decision_path, output):
    verify_sources()
    decision = approved_decision(decision_path, 'graph-init-root-registration-decision-v1')
    require(decision['registration_authorized'] is True and decision['cold_qualification_authorized'] is False and
            decision['resource_forecasts_accepted'] is True and decision['source_copy_memory_review_accepted'] is True and
            decision['predecessor_history_complete'] is True and decision['study_id'] == STUDY_ID and
            decision['anchor_directory'] == ANCHOR, 'Signed exact source/resource/lineage registry admission required')
    bound(decision['predecessor_history'])
    contexts = finalized_contexts(decision)
    # Root decision descriptor is additional fixed evidence, itself immutable.
    for context in contexts:
        context['resource_forecast']['evidence'].append(descriptor(decision_path))
    out = new_output(output)
    write(out/'CONTEXTS.json', contexts)
    evidence = [descriptor(decision_path), decision['independent_source_audit'], decision['predecessor_history'],
        descriptor(CERT_ROOT/'INDEX.json'), descriptor(CERT_ROOT/'ISSUE_TERMINAL.json')]+source_descriptors()
    write(out/'LINEAGE_AUTHORIZATION.json', lineage(contexts, evidence, True))
    write(out/'REGISTRY_REQUEST.json', registry_request(contexts, out/'LINEAGE_AUTHORIZATION.json', True))
    write(out/'PROSPECTIVE_72_ATTEMPTS.json', {'schema': 'graph-init-prospective-attempt-inventory-v1',
        'registration_authorized': True, 'contexts_sha256': object_hash(contexts),
        'anchor_directory': ANCHOR, 'attempts': expected_attempts(contexts, ANCHOR), 'identities_are_draft': False})
    write(out/'ROOT_REGISTER_REQUEST.json', launch_request('register', out/'REGISTRY_REQUEST.json', decision_path, out, True))
    # This fresh template has the admitted context hash; registry SHA remains pending.
    q = decision_base('graph-init-root-cold-qualification-decision-v1')
    q.update({'study_id': STUDY_ID, 'attempt_registry': {'path': str(Path(ANCHOR)/'GRAPH_INIT_ATTEMPT_REGISTRY.json'), 'sha256': None},
        'context_sha256': object_hash(contexts[0]), 'output': str(Path(ANCHOR)/'qualify/Squirrel_seed17'),
        'whole_cap_seconds': 1800, 'scope': 'TRAIN-only disposable cold qualifier; separate post-registration root approval.'})
    q['independent_source_audit'] = decision['independent_source_audit']
    write(out/'ROOT_COLD_QUALIFICATION_DECISION_TEMPLATE.json', q)
    return out


def make_qualify(decision_path, output):
    verify_sources()
    decision = approved_decision(decision_path, 'graph-init-root-cold-qualification-decision-v1')
    require(decision['cold_qualification_authorized'] is True and decision['registration_authorized'] is False and
            decision['study_id'] == STUDY_ID and decision['whole_cap_seconds'] == 1800,
            'Separate first cold qualifier root approval required')
    registry_path = bound(decision['attempt_registry'])
    registry = read(registry_path)
    source, protocol = bindings()
    require(registry['schema'] == 'graph-init-attempt-registry-v1' and registry['study_id'] == STUDY_ID and
            registry['anchor_directory'] == ANCHOR and registry['source_bindings'] == source and
            registry['protocol'] == protocol and registry['silent_retry_forbidden'] is True and
            registry['failure_blocks_this_study'] is True and
            registry['attempts'] == expected_attempts(registry['contexts'], ANCHOR), 'Exact realized registry required')
    bound(registry['request']); bound(registry['lineage_authorization'])
    first = [c for c in registry['contexts'] if (c['graph'], c['seed']) == ('Squirrel', 17)]
    require(len(first) == 1 and object_hash(first[0]) == decision['context_sha256'] and
            decision['output'] == str(Path(ANCHOR)/'qualify/Squirrel_seed17'), 'Exact first registered context/output required')
    out = new_output(output)
    write(out/'SQUIRREL17_QUALIFY_ADMISSION.json', qualify_admission(first[0], decision['attempt_registry'], True))
    write(out/'ROOT_SQUIRREL17_QUALIFY_REQUEST.json', launch_request('qualify', out/'SQUIRREL17_QUALIFY_ADMISSION.json',
          decision_path, out, True))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output', type=Path, default=HERE/'prepared_v1')
    for command in ('finalize-registration', 'make-qualify'):
        p = sub.add_parser(command)
        p.add_argument('--decision', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        out = prepare(args.output)
    elif args.action == 'finalize-registration':
        out = finalize_registration(args.decision, args.output)
    else:
        out = make_qualify(args.decision, args.output)
    print(str(out))


if __name__ == '__main__':
    main()
