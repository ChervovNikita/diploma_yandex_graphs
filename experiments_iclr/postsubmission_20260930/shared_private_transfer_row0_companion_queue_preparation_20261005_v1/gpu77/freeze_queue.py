#!/usr/bin/env python3
"""Metadata-only F1 companion release adapted from the reviewed pilot freeze."""
import argparse
import copy
from pathlib import Path
import pilot_common as c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    args=parser.parse_args();release=c.read(args.release)
    c.bind_block(release['execution_blocks'])
    if release.get('root_provider_block_approved') is not True:raise ValueError('Separate exact-provider F1 block release remains pending')
    if not args.release.resolve(strict=True).is_relative_to(c.PHASE.resolve(strict=True)):
        raise ValueError('Root release must stay in the authorized project phase')
    if release.get('root_numeric_cohort_approved') is not True or release.get('pilot_source_review_approved') is not True:
        raise ValueError('Root must review the companion queue and freeze bounds after actual F1 costs')
    c.verify_packet(release['pilot_source_manifest_sha256']);c.verify_training()
    if not release.get('root_review_evidence'):raise ValueError('Exact root source-review receipt required')
    for row in release['root_review_evidence']:
        if c.sha(c.phase_file(row['path']))!=row['sha256']:raise ValueError('Root source-review receipt changed')
    spec=c.read(c.ROOT/'STUDY_SPEC.json');base=c.read(c.ROOT/'QUALIFIED_BASE_JOB.json')
    if c.sha(c.ROOT/'STUDY_SPEC.json')!=c.COMPANION_SPEC_SHA or release.get('study_spec_sha256')!=c.COMPANION_SPEC_SHA:
        raise ValueError('Exact sealed nine-cell companion design changed')
    if release.get('original_plan_binding')!=spec['original_plan_binding'] or c.sha(c.phase_file(spec['original_plan_binding']['path']))!=spec['original_plan_binding']['sha256']:
        raise ValueError('Original thirty-fit plan changed')
    if release.get('original_promotion_gate_changed') is not False or release.get('original_promotion_gate_binding')!=c.ORIGINAL_PROMOTION_BINDING:
        raise ValueError('Original promotion gate must retain its exact binding')
    if c.sha(c.phase_file(c.ORIGINAL_PROMOTION_BINDING['path']))!=c.ORIGINAL_PROMOTION_BINDING['sha256']:
        raise ValueError('Original promotion release changed')
    if base.get('source_review_approved') is not True or base.get('source_review',{}).get('approved') is not True:
        raise ValueError('Exact F1 source review remains pending for this provider')
    if base.get('source_manifest_sha256')!=c.SOURCE_SHA or base.get('training_step_gate')!={'approved':True,'path':c.GATE_RELATIVE,'sha256':c.GATE_SHA}:
        raise ValueError('Exact-provider F1 base/gate binding differs')
    numeric=release['root_numeric_decision']
    if not numeric.get('sha256') or c.sha(c.phase_file(numeric['path']))!=numeric['sha256']:
        raise ValueError('Actual root F1 cost/bounds adoption remains pending')
    adopted=c.read(c.phase_file(numeric['path']))
    horizon=release.get('max_cycles')
    if type(horizon) is not int or horizon!=60 or horizon!=adopted['max_complete_cycles'] or any(row['schedule']!={'max_cycles':60,'eval_every_cycles':5,'validation_miss_limit':11} for row in spec['cells']):
        raise ValueError('Require the sealed sixty-cycle/five-cycle/eleven-miss schedule')
    if release.get('cost_completeness_approved') is not True or len(release.get('complete_cycle_cost_evidence',[]))!=3:
        raise ValueError('All three actual F1 complete-cycle costs must precede release')
    if release['complete_cycle_cost_evidence']!=adopted['complete_cycle_cost_evidence']:
        raise ValueError('Require exact root-reviewed F1 cost receipts')
    for row in release['complete_cycle_cost_evidence']:
        if c.sha(c.phase_file(row['path']))!=row['sha256']:raise ValueError('Exact F1 cost evidence changed')
    if release.get('selection_budget_fairness_approved') is not True or not release.get('paid_budget_description'):
        raise ValueError('Explicit companion exposure/selection budget approval required')
    c.check_environment(release['environment_overrides'])
    if not Path(release['python_executable']).is_file():raise ValueError('Reviewed qualified Python executable missing')
    root=c.fresh_phase_directory(c.PHASE/c.relative_path(release['execution_directory_relative']))
    if type(release.get('queue_hard_seconds')) is not int or release['queue_hard_seconds']<1:
        raise ValueError('Whole companion block hard bound remains pending')
    blocks=spec['blocks'];cells=spec['cells'];order=spec['execution_order']
    expected_ids={row['cell_id'] for row in cells}
    if len(blocks)!=3 or len(cells)!=9 or len(order)!=9 or len(expected_ids)!=9 or set(order)!=expected_ids or any(sum(row['cell_id'].startswith(block['block']+'_') for row in cells)!=3 for block in blocks):
        raise ValueError('Require exactly the sealed three-block three-cell companion')
    execution_blocks=release.get('execution_blocks')
    if execution_blocks not in (['b1'],['b2']):raise ValueError('Peptide companion admits complete b1 or b2 only')
    if release.get('resource_assignment')!=spec['resource_assignment']:
        raise ValueError('Prospective whole-block provider assignment changed')
    budgets=release['fit_bounds_by_cell']
    if set(budgets)!={row['cell'] for row in cells} or budgets!=adopted['fit_bounds_by_cell']:
        raise ValueError('Require actual adopted bounds for all three companion rules')
    for cell in cells:
        bounds=budgets[cell['cell']]
        if any(type(bounds.get(k)) is not int or bounds[k]<1 for k in ('soft_seconds','hard_seconds')) or bounds['hard_seconds']<=bounds['soft_seconds']:
            raise ValueError('Positive explicit soft/hard bounds required for '+cell['cell'])
    selected=[row for row in cells if row['cell_id'].split('_',1)[0] in execution_blocks]
    minimum_queue=sum(budgets[row['cell']]['hard_seconds'] for row in selected)
    overhead=release.get('queue_overhead_seconds');limits=release.get('resource_limits',{})
    required_limits=('minimum_fresh_GPU_free_bytes','owned_tree_GPU_memory_cap_bytes','owned_tree_RSS_cap_bytes',
                     'combined_child_log_cap_bytes','own_fit_output_cap_bytes','resource_wait_seconds',
                     'telemetry_timeout_seconds','poll_interval_seconds')
    if limits!=spec['resource_limits'] or any(type(limits.get(key)) is not int or limits[key]<1 for key in required_limits) or limits['poll_interval_seconds']>15 or limits['telemetry_timeout_seconds']>30:
        raise ValueError('Exact reviewed own-resource/output/wait bounds required')
    selected_count=len(selected)
    if selected_count!=3 or type(overhead) is not int or overhead<1 or release['queue_hard_seconds']<minimum_queue+selected_count*limits['resource_wait_seconds']+overhead:
        raise ValueError('Whole block bound must cover three fit bounds, resource waits and cleanup')
    admission=c.read(c.ROOT/'PROVIDER_ADMISSION_TEMPLATE.json')
    if admission.get('F1_exact_provider_gate_admitted') is not True or admission.get('provider_source_manifest',{}).get('sha256')!=c.SOURCE_SHA or admission.get('training_step_gate')!=base['training_step_gate']:
        raise ValueError('Exact-provider F1 admission remains pending')
    root.mkdir();(root/'jobs').mkdir();(root/'runs').mkdir();(root/'logs').mkdir()
    c.write(root/'ROOT_RELEASE.json',release)
    cohort=copy.deepcopy(spec)
    cohort.update(schema='root_frozen_nine_cell_row0_companion_v1',root_adopted_after_TRAIN_cost=True,
        new_fit_release=True,selection_budget_fairness_approved=True,
        paid_budget_description=release['paid_budget_description'],complete_cycle_cost_evidence=release['complete_cycle_cost_evidence'],
        root_numeric_decision=numeric,max_cycles=horizon,fit_bounds_by_cell=budgets,
        fit_bounds_status='actual_F1_costs_and_root_adoption',companion_spec_sha256=c.COMPANION_SPEC_SHA,
        original_promotion_gate_binding=c.ORIGINAL_PROMOTION_BINDING)
    c.write(root/'COHORT_PLAN.json',cohort)
    plan_relative=str((root/'COHORT_PLAN.json').relative_to(c.PHASE));plan_sha=c.sha(root/'COHORT_PLAN.json')
    entries={}
    for row in selected:
        job=copy.deepcopy(base)
        job['physical_gpu_uuid']=c.GPU_UUID
        job['source_review']['evidence']+=release['root_review_evidence']
        job.update(row);job.update(purpose='TRAIN_VALID_prospective_private_transfer_fit',fits_authorized=True,
            VALID_values_access=True,program_sha256=c.sha(c.SOURCE/'run.py'),source_manifest_sha256=c.SOURCE_SHA,
            cohort_plan_relative=plan_relative,cohort_plan_sha256=plan_sha,
            output_directory=str(root/'runs'/row['cell_id']),soft_seconds=budgets[row['cell']]['soft_seconds'],
            external_hard_bound_confirmed=True,scope='Fixed-row0 F1 companion only; original30/promotion untouched; no TEST/retry.')
        path=root/'jobs'/(row['cell_id']+'.json');c.write(path,job)
        entries[row['cell_id']]={'cell_id':row['cell_id'],'job_relative':str(path.relative_to(c.PHASE)),
            'job_sha256':c.sha(path),'output_relative':str((root/'runs'/row['cell_id']).relative_to(c.PHASE)),
            'hard_seconds':budgets[row['cell']]['hard_seconds']}
    queue={'schema':'root_frozen_outcome_free_row0_companion_queue_v1','execution_directory_relative':str(root.relative_to(c.PHASE)),
        'pilot_source_manifest_sha256':release['pilot_source_manifest_sha256'],'training_source_manifest_sha256':c.SOURCE_SHA,
        'cohort_plan_sha256':plan_sha,'root_release_sha256':c.sha(root/'ROOT_RELEASE.json'),
        'python_executable':release['python_executable'],'environment_overrides':release['environment_overrides'],
        'queue_hard_seconds':release['queue_hard_seconds'],'queue_overhead_seconds':overhead,
        'resource_limits':limits,'execution_blocks':execution_blocks,'resource_assignment':spec['resource_assignment'],
        'entries':[entries[i] for i in order if i.split('_',1)[0] in execution_blocks],
        'retry':False,'stop_on_operational_failure':True,'read_scores_or_change_family':False,'TEST_access':False}
    admission.update(schema='root_admitted_peptide_row0_companion_provider_v1',approved=True,
        admitted_before_provider_first_fit=True,GPU_UUID=c.GPU_UUID,
        root_source_review_evidence=base['source_review']['evidence']+release['root_review_evidence'])
    c.write(root/'PROVIDER_ADMISSION.json',admission)
    c.write(root/'QUEUE.json',queue)
    c.write(root/'DONOR_REGISTRATION.json',{'block':execution_blocks[0],'provider':'authorized_18.77',
        'hostname':'peptide','GPU_UUID':c.GPU_UUID,'donor_directory_relative':str(root.relative_to(c.PHASE)),
        'replica_directory_relative':str(root.relative_to(c.PHASE)),
        'provider_admission':{'path':str((root/'PROVIDER_ADMISSION.json').relative_to(c.PHASE)),
                              'sha256':c.sha(root/'PROVIDER_ADMISSION.json')},
        'queue_sha256':c.sha(root/'QUEUE.json'),'root_release_sha256':c.sha(root/'ROOT_RELEASE.json'),
        'registered_before_provider_first_fit':True,'fallback_activated':False})
    print('Frozen exact nine-cell companion and three whole-block jobs without starting fits: '+str(root))


if __name__=='__main__':main()
