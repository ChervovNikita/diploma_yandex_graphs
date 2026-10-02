"""Stdlib AST, sealed-text and exact metadata draft checks; no scientific imports."""
import ast
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from admission_support import (HERE, PHASE, R17_REL, CELLS, SEEDS, ARMS, require, read, sha,
    bound, object_hash, expected_attempts, verify_sources, source_descriptors, bindings, write)


def main():
    checks = []
    allowed = {'argparse', 'ast', 'copy', 'datetime', 'hashlib', 'json', 'math', 'os', 'pathlib',
               'subprocess', 'sys', 'time', 'traceback', 'admission_support'}
    for path in sorted(HERE.glob('*.py')):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) else []
            require(all(name in allowed for name in names), 'Non-stdlib scientific import in '+path.name)
        checks.append({'check': 'AST and import allowlist: '+path.name, 'passed': True, 'sha256': sha(path)})
    verify_sources()
    checks.append({'check': 'Full sealed R17v2/modernv3 text payload hash verification', 'passed': True})
    prepared = HERE/'prepared_v1'
    contexts = read(prepared/'CONTEXTS_DRAFT.json')
    expected_keys = set(read(PHASE/R17_REL/'templates/CELL_CONTEXT_TEMPLATE.json'))
    require(len(contexts) == 6 and [(c['graph'], c['backbone'], c['seed'], c['source_split_index']) for c in contexts] ==
            [(g, b, s, i) for g, b in CELLS for i, s in enumerate(SEEDS)], 'Exact ordered cohort required')
    for context in contexts:
        require(set(context) == expected_keys and context['config'] == 0 and
                context['resource_forecast']['admitted'] is False and
                context['roles_frozen_before_label_extraction'] is True and
                context['independent_of_stage1_outcomes'] is True and
                'outcome-aware' in context['prior_exposure_disclosure'] and
                'source-pack custody only' in context['prior_exposure_disclosure'], 'Draft context/provenance shape differs')
        cert = read(bound(context['modern_qualification_certificate']))
        identity = cert['authorized_target_input']
        for key in ('role_freeze', 'graph_input', 'source_labels', 'source_label_binding'):
            require(context[key] == identity[key], 'Original descriptor must remain exact, including bytes')
        require(context['environment'] == cert['environment'], 'Certified environment must remain exact')
        require(context['source_bindings'] == bindings()[0] and context['protocol'] == bindings()[1], 'Exact remote v2 bindings required')
        require(set(context['source_labels']) == {'train', 'validation'}, 'Final labels forbidden in contexts')
        for record in context['resource_forecast']['evidence']:
            bound(record)
    checks.append({'check': 'Six exact eligible GNNM source/runtime descriptors with optional bytes preserved', 'passed': True})
    inventory = read(prepared/'PROSPECTIVE_72_ATTEMPTS_DRAFT.json')
    require(inventory['attempts'] == expected_attempts(contexts, inventory['anchor_directory']) and
            inventory['draft_contexts_sha256'] == object_hash(contexts), 'Canonical prospective inventory differs')
    rows = inventory['attempts']
    counts = {phase: sum(row['phase'] == phase for row in rows) for phase in ('qualify', 'warm', 'initialize', 'fit')}
    require(counts == {'qualify': 6, 'warm': 6, 'initialize': 30, 'fit': 30}, 'Wrong phase inventory')
    checks.append({'check': '72 unique canonical attempt/output identities', 'passed': True, 'counts': counts})
    lineage = read(prepared/'LINEAGE_DRAFT.json')
    registry = read(prepared/'REGISTRY_REQUEST_DRAFT.json')
    require(set(lineage) == set(read(PHASE/R17_REL/'templates/VERSION_LINEAGE_AUTHORIZATION_TEMPLATE.json')) and
            set(registry) == set(read(PHASE/R17_REL/'templates/REGISTRY_REQUEST_TEMPLATE.json')) and
            lineage['approved'] is False and lineage['predecessor_history_complete'] is False and
            registry['registration_authorized'] is False and registry['contexts'] == contexts and
            registry['prior_attempt_registries'] == lineage['prior_attempt_registries'] == [], 'Draft lineage/registry shape differs')
    require(lineage['contexts_sha256'] == object_hash(contexts), 'Lineage context digest differs')
    for record in lineage['evidence']:
        bound(record)
    admission = read(prepared/'SQUIRREL17_QUALIFY_ADMISSION_DRAFT.json')
    require(set(admission) == set(read(PHASE/R17_REL/'templates/QUALIFY_ADMISSION_TEMPLATE.json')) and
            admission['context'] == contexts[0] and admission['execution_authorized'] is False and
            admission['authorized_phase'] == 'qualify' and admission['dependencies'] == {} and
            admission['arm'] is None and admission['attempt_registry']['sha256'] is None,
            'Initial qualifier must remain unapproved pending exact realized registry')
    for name in ('ROOT_REGISTER_REQUEST_DRAFT.json', 'ROOT_SQUIRREL17_QUALIFY_REQUEST_DRAFT.json'):
        request = read(prepared/name)
        require(request['root_admitted'] is False and request['full_fit_admitted'] is False and
                request['heldout_scoring_admitted'] is False and request['source_seals'] == source_descriptors() and
                not Path(request['receipt_directory']).is_relative_to(Path(registry['anchor_directory'])), 'Draft launch scope/receipt separation differs')
        for record in request['protected_files']:
            bound(record)
        bound(request['root_decision']); bound(request['phase_payload'])
    for name in ('ROOT_REGISTRATION_DECISION_TEMPLATE.json', 'ROOT_COLD_QUALIFICATION_DECISION_TEMPLATE.json'):
        value = read(prepared/name)
        require(value['approved'] is False and value['registration_authorized'] is False and
                value['cold_qualification_authorized'] is False and value['independent_source_audit_accepted'] is False,
                'Root decision template must remain unapproved')
    costs = read(prepared/'PROSPECTIVE_RESOURCE_EVIDENCE.json')
    require(len(costs['native_measured_costs']) == 6 and costs['native_short_costs_establish_R17_AD_cost'] is False and
            costs['full_schedule_feasibility_passed'] is False, 'No invented AD/full-schedule cost result allowed')
    checks.append({'check': 'Exact draft lineage/admission and false root/resource/phase approvals', 'passed': True})
    checks.append({'check': 'Actual six native COSTS aggregates; no AD or full-schedule feasibility claim', 'passed': True})
    result = {'schema': 'graph-init-root-builder-static-check-v1', 'passed': True, 'checks': checks,
        'scope': 'AST/hash/JSON only; entry and R17 driver not executed; no arrays/models/checkpoints/scientific modules opened.',
        'independent_R17_source_audit_still_root_required': True, 'root_decisions_still_false': True}
    write(HERE/'STATIC_CHECKS.json', result)
    print(json.dumps({'passed': True, 'checks': len(checks), 'contexts': 6, 'prospective_attempts': 72}))


if __name__ == '__main__':
    main()
