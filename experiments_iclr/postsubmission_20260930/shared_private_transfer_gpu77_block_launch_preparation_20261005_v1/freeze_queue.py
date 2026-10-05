#!/usr/bin/env python3
"""Metadata-only root release: freeze exact cohort/jobs/queue, never start fits."""
import argparse
import copy
from pathlib import Path
import pilot_common as c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    args=parser.parse_args();release=c.read(args.release);c.bind_block(release['execution_blocks'])
    if release.get('root_provider_block_approved') is not True:
        raise ValueError('Peptide accuracy fits require a separate root block release')
    if not args.release.resolve(strict=True).is_relative_to(c.PHASE.resolve(strict=True)):
        raise ValueError('Root release must stay in the authorized project phase')
    if release.get('root_numeric_cohort_approved') is not True or release.get('pilot_source_review_approved') is not True:
        raise ValueError('Root must review the queue and freeze horizons/bounds after measured costs')
    c.verify_packet(release['pilot_source_manifest_sha256']);c.verify_training()
    if not release.get('root_review_evidence'):raise ValueError('Exact root source-review receipt required')
    for row in release['root_review_evidence']:
        if c.sha(c.phase_file(row['path']))!=row['sha256']:raise ValueError('Root source-review receipt changed')
    spec=c.read(c.ROOT/'STUDY_SPEC.json');base=c.read(c.ROOT/'QUALIFIED_BASE_JOB.json')
    if release.get('science_contract_sha256')!=c.sha(c.ROOT/'SCIENCE_CONTRACT.json'):
        raise ValueError('Immutable cross-provider numerical science contract changed')
    if c.sha(c.ROOT/'STUDY_SPEC.json')!=release['study_spec_sha256'] or c.sha(c.ROOT/'EXTERNAL_ANCHORS.json')!=release['external_anchors_sha256']:
        raise ValueError('Prospective cells/seeds or original fixed anchors changed')
    numeric=spec['root_numeric_decision']
    if release.get('root_numeric_decision')!=numeric or c.sha(c.phase_file(numeric['path']))!=numeric['sha256']:
        raise ValueError('Exact root cost-based numeric decision changed')
    adopted=c.read(c.phase_file(numeric['path']))
    horizon=release.get('max_cycles')
    if type(horizon) is not int or horizon!=spec['numeric_horizon'] or horizon!=adopted['max_complete_cycles']:
        raise ValueError('Require the exact root-adopted sixty-complete-cycle horizon')
    if release.get('cost_completeness_approved') is not True or not release.get('complete_cycle_cost_evidence'):
        raise ValueError('Remaining complete-cycle costs must be reviewed before freeze')
    if release['complete_cycle_cost_evidence']!=adopted['complete_cycle_cost_evidence']:
        raise ValueError('Require all exact root-reviewed complete-cycle cost receipts')
    for row in release['complete_cycle_cost_evidence']:
        if c.sha(c.phase_file(row['path']))!=row['sha256']: raise ValueError('Exact cost evidence changed')
    if release.get('selection_budget_fairness_approved') is not True or not release.get('paid_budget_description'):
        raise ValueError('Explicit competent-reference exposure/selection budget approval required')
    c.check_environment(release['environment_overrides'])
    if not Path(release['python_executable']).is_file(): raise ValueError('Reviewed qualified Python executable missing')
    root=c.fresh_phase_directory(c.PHASE/c.relative_path(release['execution_directory_relative']))
    if type(release.get('queue_hard_seconds')) is not int or release['queue_hard_seconds']<1:
        raise ValueError('Whole queue hard bound remains pending')
    policy=release.get('analysis_policy',{})
    if policy!=c.read(c.ROOT/'ROOT_RELEASE_TEMPLATE.json')['analysis_policy']:
        raise ValueError('Require the complete exact pre-outcome quality/attribution analysis policy')
    if type(policy.get('minimum_mean_MRR_gain')) not in (float,int) or policy['minimum_mean_MRR_gain']!=0 or policy.get('minimum_higher_blocks')!=2 or policy.get('strict_positive_mean') is not True:
        raise ValueError('Root must freeze exploratory lead definitions before outcomes')
    blocks=spec['blocks'];cells=spec['cells']
    expected_ids={b['block']+'_'+row['cell'] for b in blocks for row in cells}
    if len(blocks)!=3 or len(cells)!=10 or len(spec['execution_order'])!=30 or len(expected_ids)!=30 or set(spec['execution_order'])!=expected_ids:
        raise ValueError('Require the exact representative three-block ten-cell family')
    execution_blocks=release.get('execution_blocks')
    if not isinstance(execution_blocks,list) or not execution_blocks or len(set(execution_blocks))!=len(execution_blocks) or set(execution_blocks)-{b['block'] for b in blocks}:
        raise ValueError('Execution subset must contain distinct complete prespecified blocks')
    if release.get('resource_assignment')!=spec['resource_assignment']:
        raise ValueError('Prospective whole-block provider assignment/fallback changed')
    if release.get('singleton_fallback_activated_for_blocks')!=[] or release.get('fallback_activation_evidence')!=[]:
        raise ValueError('This launcher admits only the fixed primary peptide provider')
    budgets=release['fit_bounds_by_cell']
    if set(budgets)!={r['cell'] for r in cells}: raise ValueError('All thirty fits require pre-outcome cell budget bounds')
    for cell in cells:
        limits=budgets[cell['cell']]
        if any(type(limits.get(k)) is not int or limits[k]<1 for k in ('soft_seconds','hard_seconds')) or limits['hard_seconds']<=limits['soft_seconds']:
            raise ValueError('Positive explicit soft/hard bounds required for '+cell['cell'])
    minimum_queue=sum(budgets[row['cell']]['hard_seconds'] for row in cells)*len(execution_blocks)
    overhead=release.get('queue_overhead_seconds')
    limits=release.get('resource_limits',{})
    required_limits=('minimum_fresh_GPU_free_bytes','owned_tree_GPU_memory_cap_bytes','owned_tree_RSS_cap_bytes',
                     'combined_child_log_cap_bytes','own_fit_output_cap_bytes','resource_wait_seconds',
                     'telemetry_timeout_seconds','poll_interval_seconds')
    if any(type(limits.get(key)) is not int or limits[key]<1 for key in required_limits) or limits['poll_interval_seconds']>15 or limits['telemetry_timeout_seconds']>30:
        raise ValueError('Positive reviewed own-resource/output/wait bounds and short bounded telemetry required')
    selected_count=10*len(execution_blocks)
    if type(overhead) is not int or overhead<1 or release['queue_hard_seconds']<minimum_queue+selected_count*limits['resource_wait_seconds']+overhead:
        raise ValueError('Whole block queue must cover every selected hard fit/resource wait and explicit cleanup overhead')
    root.mkdir();(root/'jobs').mkdir();(root/'runs').mkdir();(root/'logs').mkdir()
    c.write(root/'ROOT_RELEASE.json',release)
    canonical=c.ROOT/'COHORT_PLAN.json'
    if c.sha(canonical)!=c.CANONICAL_PLAN_SHA:
        raise ValueError('Canonical full thirty-cell raw plan changed')
    cohort=c.read(canonical)
    for key,expected in (('blocks',blocks),('execution_order',spec['execution_order']),('max_cycles',horizon),
                         ('analysis_policy',policy),('fit_bounds_by_cell',budgets),('resource_limits',limits),
                         ('resource_assignment',spec['resource_assignment']),('complete_cycle_cost_evidence',release['complete_cycle_cost_evidence'])):
        if cohort[key]!=expected:raise ValueError('Canonical plan/release differs: '+key)
    # Preserve original plan source_manifest as scientific lineage. Actual provider
    # source/runtime/qualifier bindings live in jobs, queue and provider admission.
    (root/'COHORT_PLAN.json').write_bytes(canonical.read_bytes())
    (root/'EXTERNAL_ANCHORS.json').write_bytes((c.ROOT/'EXTERNAL_ANCHORS.json').read_bytes())
    plan_relative=str((root/'COHORT_PLAN.json').relative_to(c.PHASE));plan_sha=c.sha(root/'COHORT_PLAN.json')
    entries={}
    for row in cohort['cells']:
        if row['cell_id'].split('_',1)[0] not in execution_blocks:continue
        job=copy.deepcopy(base)
        job['physical_gpu_uuid']=c.GPU_UUID
        job['source_review']['evidence']+=release['root_review_evidence']
        selected=row['cell_id'].split('_',1)[0] in execution_blocks
        job.update(row);job.update(purpose='TRAIN_VALID_prospective_private_transfer_fit',fits_authorized=selected,
            VALID_values_access=selected,program_sha256=c.sha(c.SOURCE/'run.py'),source_manifest_sha256=c.SOURCE_SHA,
            cohort_plan_relative=plan_relative,cohort_plan_sha256=plan_sha,
            output_directory=str(root/'runs'/row['cell_id']),soft_seconds=budgets[row['cell']]['soft_seconds'],
            external_hard_bound_confirmed=True,scope='Prospective paired pilot; no original-anchor rerun, no TEST, no retry.')
        path=root/'jobs'/(row['cell_id']+'.json');c.write(path,job)
        entries[row['cell_id']]={'cell_id':row['cell_id'],'job_relative':str(path.relative_to(c.PHASE)),
            'job_sha256':c.sha(path),'output_relative':str((root/'runs'/row['cell_id']).relative_to(c.PHASE)),
            'hard_seconds':budgets[row['cell']]['hard_seconds']}
    queue={'schema':'root_frozen_outcome_free_paired_queue_v1','execution_directory_relative':str(root.relative_to(c.PHASE)),
        'pilot_source_manifest_sha256':release['pilot_source_manifest_sha256'],'training_source_manifest_sha256':c.SOURCE_SHA,
        'cohort_plan_sha256':plan_sha,'root_release_sha256':c.sha(root/'ROOT_RELEASE.json'),
        'python_executable':release['python_executable'],'environment_overrides':release['environment_overrides'],
        'queue_hard_seconds':release['queue_hard_seconds'],'queue_overhead_seconds':overhead,
        'resource_limits':limits,'execution_blocks':execution_blocks,'resource_assignment':spec['resource_assignment'],
        'entries':[entries[i] for i in spec['execution_order'] if i.split('_',1)[0] in execution_blocks],
        'retry':False,'stop_on_operational_failure':True,'read_scores_or_change_family':False,'TEST_access':False}
    admission=c.read(c.ROOT/'PROVIDER_ADMISSION_TEMPLATE.json')
    admission.update(schema='root_admitted_peptide_private_transfer_provider_v1',approved=True,
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
    print('Frozen exact canonical plan and '+str(selected_count)+' complete-block jobs without starting fits: '+str(root))


if __name__=='__main__':main()
