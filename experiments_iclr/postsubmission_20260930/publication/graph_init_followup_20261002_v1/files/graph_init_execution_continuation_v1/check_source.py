"""Stdlib source and exact finite-plan checks. No wrapper/coordinator is run."""
import ast
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from continuation_support import (HERE, PHASE, PHASES, FIRST_CELL, TEXT_EXTENSIONS, require,
    read, sha, write, descriptor, object_hash, verify_sources, registry_guard, finite_plan,
    row_for, completed_phase, scan_state, decision_guard, deployment_ancillary)
from build_continuation import template, REGISTRY_PATH


def main():
    allowed = {'argparse', 'ast', 'copy', 'datetime', 'hashlib', 'io', 'json', 'os', 'pathlib',
        'shlex', 'subprocess', 'sys', 'tarfile', 'time', 'traceback', 'continuation_support', 'build_continuation'}
    ast_sources = []
    for path in sorted(HERE.glob('*.py')):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) else []
            require(all(n in allowed for n in names), 'Only stdlib/source metadata imports allowed')
        ast_sources.append(descriptor(path))
    verify_sources()
    record = descriptor(REGISTRY_PATH)
    registry = registry_guard(record)
    plan = finite_plan(registry)
    counts = {p: sum(r['phase'] == p for r in plan) for p in PHASES}
    require(counts == {'qualify': 5, 'warm': 6, 'initialize': 30, 'fit': 30}, 'Exactly71remaining attempts required')
    phase_indices = [PHASES.index(r['phase']) for r in plan]
    require(phase_indices == sorted(phase_indices), 'Remaining qualifications must all precede warm/init/fit')
    require(len({r['key'] for r in plan}) == len({r['output'] for r in plan}) == 71, 'No retries/new output identities')
    for row in plan:
        context = [c for c in registry['contexts'] if object_hash(c) == row['context_sha256']][0]
        require(row['whole_cap_seconds'] == context['resource_forecast']['forecast']['wall_seconds_by_phase'][row['phase']],
                'Every cap must equal the immutable registered forecast')
    first = row_for(registry, [c for c in registry['contexts'] if (c['graph'], c['seed']) == FIRST_CELL][0], 'qualify')
    first_freeze, first_terminal, _ = completed_phase(record, registry, first)
    completed = scan_state(record, registry)
    require(set(completed) == {first['key']}, 'Only the root-owned initial qualification may already be completed')
    decision = template()
    require(decision['approved'] is False and decision['scientific_phases_authorized'] is False and
            decision['compare_authorized'] is False and decision['report_authorized'] is False and
            decision['final_labels_authorized'] is False and decision['automatic_retry_authorized'] is False and
            decision['finite_plan'] == plan and decision['finite_plan_sha256'] == object_hash(plan) and
            decision['initial_qualification_freeze'] == first_freeze and decision['initial_qualification_terminal'] == first_terminal,
            'One exact signed-false finite decision required')
    write(HERE/'ROOT_FINITE_SCIENCE_DECISION_TEMPLATE.json', decision)
    try:
        decision_guard(HERE/'ROOT_FINITE_SCIENCE_DECISION_TEMPLATE.json')
    except ValueError:
        pass
    else:
        raise ValueError('Unsigned template must be rejected')
    write(HERE/'FINITE_71_PLAN_DRAFT.json', {'schema': 'graph-init-finite-plan-draft-v1',
        'scientific_execution_authorized': False, 'attempt_registry': record,
        'plan_sha256': object_hash(plan), 'attempts': plan,
        'remaining_whole_cap_budget_seconds': sum(r['whole_cap_seconds'] for r in plan)})
    ancillary = deployment_ancillary()
    write(HERE/'DEPLOYMENT_DEPENDENCIES.json', {'schema': 'graph-init-continuation-deployment-dependencies-v1',
        'required_explicit_ancillary_text_files': ancillary,
        'already_required_R17_v2_and_modern_v3_sealed_payloads': decision['source_seals'],
        'initial_and_current_registry_metadata_closure': decision['protected_files'],
        'whole_phase_or_results_directory_upload_allowed': False,
        'array_model_or_checkpoint_upload_download_by_coordinator_allowed': False})
    write(HERE/'STATIC_CHECKS.json', {'schema': 'graph-init-continuation-source-check-v1', 'passed': True,
        'AST_sources': ast_sources, 'exact_current_registry': record, 'remaining_attempt_counts': counts,
        'remaining_plan_sha256': object_hash(plan), 'all71caps_equal_registered_forecasts': True,
        'unchanged_Round15_deployment_dependency_verified': ancillary,
        'initial_qualification_reference_is_root_owned_completed_JSON_evidence': True,
        'unsigned_decision_rejected': True, 'scientific_phase_approval_flags_false': True,
        'source_sealed_texts_JSON_AST_only': True, 'scientific_import_or_execution': False,
        'arrays_labels_checkpoints_opened': False, 'wrapper_coordinator_SSH_or_GPU_executed': False,
        'no_Photo_actual_warm_AD_or_full_schedule_feasibility_claim': True,
        'root_continuation_source_review_still_required': True})
    print(json.dumps({'passed': True, 'AST_sources': len(ast_sources), 'exact_remaining_attempts': 71, 'counts': counts}))


if __name__ == '__main__':
    main()
