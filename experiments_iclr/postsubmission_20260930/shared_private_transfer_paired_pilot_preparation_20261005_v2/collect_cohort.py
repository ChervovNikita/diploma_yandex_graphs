#!/usr/bin/env python3
"""Authenticate all three immutable donor blocks without analyzing quality."""
import argparse
import copy
from pathlib import Path
import pilot_common as c

PLAN_KEYS=('TEST_closed','selection','blocks','max_cycles','include_stale_commit_ablation',
           'execution_order','cells','analysis_policy','root_numeric_decision',
           'complete_cycle_cost_evidence','selection_budget_fairness_approved')


def binding(reference):
    path=c.phase_file(reference['path'])
    if c.sha(path)!=reference['sha256']:raise ValueError('Frozen metadata binding changed: '+reference['path'])
    return path


def evidence(rows):
    if not isinstance(rows,list) or not rows:raise ValueError('Explicit pre-fit provider qualification/review evidence required')
    for row in rows:binding(row)


def replica(donor,original):
    original=c.relative_path(original);prefix=c.relative_path(donor['donor_directory_relative'])
    if not original.is_relative_to(prefix):raise ValueError('Donor artifact is outside its registered execution directory')
    return c.phase_file(str(c.relative_path(donor['replica_directory_relative'])/original.relative_to(prefix)))


def provider_admission(donor,science):
    path=binding(donor['provider_admission']);admission=c.read(path)
    if admission.get('approved') is not True or admission.get('admitted_before_provider_first_fit') is not True:
        raise ValueError('Provider source/runtime admission must precede its first fit')
    for key in ('provider','hostname','GPU_UUID'):
        if admission.get(key)!=donor[key]:raise ValueError('Registered provider identity differs: '+key)
    if any(admission.get(key) is not False for key in ('TEST_access','original_score_recalculation','numerical_science_changes')):
        raise ValueError('Provider admits only unchanged TRAIN/VALID science')
    if admission['science_contract_sha256']!=c.sha(c.ROOT/'SCIENCE_CONTRACT.json') or admission['available_manifest_sha256']!=science['available_manifest_sha256']:
        raise ValueError('Provider scientific/data contract differs')
    source_manifest=binding(admission['provider_source_manifest']);program=binding(admission['program'])
    if program.parent!=source_manifest.parent or program.name!='run.py' or admission['program']['sha256']!=science['program_sha256']:
        raise ValueError('Provider program is not the exact unchanged qualified scientific run')
    for row in c.read(source_manifest)['files']:
        file=(source_manifest.parent/c.relative_path(row['path'])).resolve(strict=True)
        if not file.is_relative_to(source_manifest.parent) or c.sha(file)!=row['sha256'] or file.stat().st_size!=row['bytes']:
            raise ValueError('Provider source bytes changed')
    expected=science['numerical_science_files'];actual=admission['numerical_science_files']
    if set(actual)!=set(expected):raise ValueError('Complete numerical science/dependency set differs')
    for name,reference in actual.items():
        if reference['sha256']!=expected[name]['sha256']:raise ValueError('Provider changed numerical science: '+name)
        file=binding(reference)
        if name in ('run.py','models.py','private_adam.py','transfer_step.py') and file!=program.parent/name:
            raise ValueError('Numerical implementation is not beside its admitted run.py')
    evidence(admission['qualification_evidence']);evidence(admission['root_source_review_evidence'])
    if not isinstance(admission.get('runtime_versions'),dict) or not admission['runtime_versions'] or not admission.get('python_executable'):
        raise ValueError('Actual reviewed provider runtime must be retained')
    expected_repo={'authorized_one_GPU_allocation':str(c.REPO),
                   'authorized_18.77':'/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'}
    if admission['authorized_repository']!=expected_repo.get(admission['provider']):
        raise ValueError('Provider repository is outside an authorized project checkout')
    return admission,path


def job_authorities(job,admission):
    for name in ('runtime_qualification','feature_authority','negative_pool_authority','source_review'):
        authority=job.get(name,{})
        if authority.get('approved') is not True:raise ValueError('Provider fit authority missing: '+name)
        evidence(authority['evidence'])
    gate=job.get('training_step_gate',{})
    if gate.get('approved') is not True:raise ValueError('Provider training-step qualification missing')
    gate_path=binding({'path':gate['path'],'sha256':gate['sha256']})
    gate_value=c.read(gate_path)
    if gate_value.get('passed') is not True or gate_value.get('source_manifest_sha256')!=admission['provider_source_manifest']['sha256']:
        raise ValueError('Qualification did not admit this actual provider source')
    if job.get('runtime_versions')!=admission['runtime_versions']:raise ValueError('Job runtime differs from provider admission')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();release=c.read(args.release)
    if not args.release.resolve(strict=True).is_relative_to(c.PHASE.resolve(strict=True)):
        raise ValueError('Collection contract leaves the authorized project phase')
    if release.get('root_collection_approved') is not True or release.get('collection_contract_frozen_before_any_new_pilot_score_analysis') is not True:
        raise ValueError('Separate pre-analysis frozen root collection approval required')
    for key in ('TEST_access','fits_authorized','original_score_recalculation'):
        if release.get(key) is not False:raise ValueError('Metadata collection admits no fit, TEST or original-score replay')
    if release.get('no_donor_block_or_fit_selection') is not True or release.get('all_30_required_before_analysis') is not True:
        raise ValueError('All prespecified donors/fits must be retained')
    c.verify_packet(release['pilot_source_manifest_sha256']);evidence(release['source_review_evidence'])
    science=c.read(c.ROOT/'SCIENCE_CONTRACT.json')
    if c.sha(c.ROOT/'SCIENCE_CONTRACT.json')!=release['science_contract_sha256']:raise ValueError('Frozen common science contract changed')
    plan_path=binding(release['common_plan']);plan=c.read(plan_path)
    if {key:plan[key] for key in PLAN_KEYS}!=science['common_scientific_plan']:
        raise ValueError('Full canonical thirty-cell science differs from prospective contract')
    anchor_path=binding(release['external_anchors'])
    if c.sha(anchor_path)!=c.sha(c.ROOT/'EXTERNAL_ANCHORS.json'):raise ValueError('Original verbatim anchors changed')
    donors=release['donor_blocks']
    if len(donors)!=3 or {d['block'] for d in donors}!={'b0','b1','b2'}:raise ValueError('Three distinct whole-block donors required')
    records={};donor_receipts=[];common_inputs=None
    for donor in donors:
        block=donor['block'];assigned=science['resource_assignment'][block]
        if donor.get('registered_before_provider_first_fit') is not True:raise ValueError('Donor custody must be registered before its first fit')
        if donor.get('fallback_activated') is True:
            if block=='b0' or donor['provider']!='authorized_one_GPU_allocation':raise ValueError('Only predeclared entire-block singleton fallback is permitted')
            if donor['hostname']!=science['resource_assignment']['b0']['hostname'] or donor['GPU_UUID']!=science['resource_assignment']['b0']['GPU_UUID']:
                raise ValueError('Fallback must use the exact authorized singleton host/GPU')
            evidence(donor.get('fallback_activation_evidence'))
        elif any(donor[key]!=assigned[key] for key in ('provider','hostname','GPU_UUID') if key in assigned):
            raise ValueError('Prospective provider assignment changed')
        # Study calls the provider field primary_provider; retain it exactly.
        if not donor.get('fallback_activated') and donor['provider']!=assigned['primary_provider']:
            raise ValueError('Prospective whole-block provider differs')
        admission,admission_path=provider_admission(donor,science)
        prefix=str(c.relative_path(donor['donor_directory_relative']))
        queue_path=replica(donor,prefix+'/QUEUE.json');queue=c.read(queue_path)
        release_path=replica(donor,prefix+'/ROOT_RELEASE.json');donor_release=c.read(release_path)
        if c.sha(queue_path)!=donor['queue_sha256'] or c.sha(release_path)!=donor['root_release_sha256'] or queue['root_release_sha256']!=donor['root_release_sha256']:
            raise ValueError('Pre-fit donor queue/root release binding changed')
        donor_plan=replica(donor,prefix+'/COHORT_PLAN.json');donor_anchors=replica(donor,prefix+'/EXTERNAL_ANCHORS.json')
        if c.sha(donor_plan)!=release['common_plan']['sha256'] or queue['cohort_plan_sha256']!=release['common_plan']['sha256'] or c.sha(donor_anchors)!=release['external_anchors']['sha256']:
            raise ValueError('Every provider must retain byte-identical canonical full plan/anchors')
        expected=[identity for identity in plan['execution_order'] if identity.startswith(block+'_')]
        if queue.get('execution_blocks')!=[block] or [r['cell_id'] for r in queue['entries']]!=expected or len(expected)!=10:
            raise ValueError('Provider queue is not exactly its fixed ten-cell whole block')
        if queue['training_source_manifest_sha256']!=admission['provider_source_manifest']['sha256'] or queue['python_executable']!=admission['python_executable']:
            raise ValueError('Queue actual source/runtime differs from admitted provider')
        block_path=replica(donor,prefix+'/BLOCK_FREEZE.json');block_freeze=c.read(block_path)
        if block_freeze.get('selected_blocks_complete') is not True or block_freeze.get('execution_blocks')!=[block] or block_freeze.get('physical_fits')!=10:
            raise ValueError('Whole-block completion is partial or differently scoped')
        if any(block_freeze.get(key) is not False for key in ('TEST_access','comparative_scoring_performed','retry')):
            raise ValueError('Donor block scope permits no prior comparative score analysis')
        if block_freeze['queue_sha256']!=donor['queue_sha256'] or block_freeze['cohort_plan_sha256']!=release['common_plan']['sha256'] or block_freeze['external_anchors_sha256']!=release['external_anchors']['sha256']:
            raise ValueError('Whole-block freeze custody changed')
        completions=block_freeze['completed']
        if [r['cell_id'] for r in completions]!=expected:raise ValueError('Missing, duplicate, selected or reordered completed cell')
        queue_records={r['cell_id']:r for r in queue['entries']}
        for receipt in completions:
            identity=receipt['cell_id'];entry=queue_records[identity]
            if identity in records:raise ValueError('Duplicate cross-provider fit')
            if receipt.get('terminal_wait_observed') is not True or receipt.get('exit_code_authority')!='subprocess.Popen.wait/poll' or receipt.get('exit_code')!=0 or receipt.get('reason') is not None or receipt.get('signals_sent')!=[] or receipt.get('attempts')!=1 or receipt.get('retry') is not False:
                raise ValueError('Fit lacks successful authoritative owned Popen completion')
            if receipt.get('identity_admitted_for_signals') is False:raise ValueError('Rejected initial child identity cannot admit a fit')
            owner=receipt.get('child_identity')
            if not isinstance(owner,dict) or owner.get('pgid')!=owner.get('PID') or owner.get('sid')!=owner.get('PID') or not owner.get('start_ticks') or not owner.get('argv'):
                raise ValueError('Fresh child/session identity missing')
            original_phase=admission['authorized_repository']+'/experiments_iclr/postsubmission_20260930/'
            program_absolute=original_phase+admission['program']['path']
            command=[admission['python_executable'],program_absolute,'--job',original_phase+entry['job_relative'],
                     '--output',original_phase+entry['output_relative']]
            if owner['argv'] not in (command,[command[0],'-B',*command[1:]]):
                raise ValueError('Fresh child argv does not match its pre-fit admitted program/job/output')
            if receipt.get('observed_cwd')!=admission['authorized_repository']:
                raise ValueError('Actual child working directory differs from its admitted repository')
            job_path=replica(donor,entry['job_relative']);job=c.read(job_path)
            if c.sha(job_path)!=entry['job_sha256'] or receipt['job_sha256']!=entry['job_sha256']:raise ValueError('Donor job custody differs')
            cell=next(row for row in plan['cells'] if row['cell_id']==identity)
            for key in ('cell_id','cell','paired_seed_block','seed','factor_seed','arm','rule','geometry','outer_size','inner_size','schedule'):
                if job.get(key)!=cell[key]:raise ValueError('Donor scientific cell differs: '+key)
            if job.get('purpose')!='TRAIN_VALID_prospective_private_transfer_fit' or job.get('fits_authorized') is not True or job.get('VALID_values_access') is not True or job.get('TEST_access') is not False or job.get('retry') is not False:
                raise ValueError('Donor job scope differs')
            if job['source_manifest_sha256']!=admission['provider_source_manifest']['sha256'] or job['program_sha256']!=science['program_sha256'] or job['available_manifest_sha256']!=science['available_manifest_sha256']:
                raise ValueError('Donor source/program/data identity differs')
            if job['cohort_plan_sha256']!=release['common_plan']['sha256']:raise ValueError('Job canonical plan differs')
            if job['soft_seconds']!=plan['fit_bounds_by_cell'][cell['cell']]['soft_seconds'] or entry['hard_seconds']!=plan['fit_bounds_by_cell'][cell['cell']]['hard_seconds']:
                raise ValueError('Donor operational fit horizon changed')
            job_authorities(job,admission)
            value_path=replica(donor,receipt['freeze_relative'])
            if value_path!=replica(donor,entry['output_relative']+'/FREEZE.json') or c.sha(value_path)!=receipt['freeze_sha256']:
                raise ValueError('Donor fit freeze custody differs')
            value=c.read(value_path)
            # Access only custody/scope/identity metadata; quality fields are not used or emitted.
            if value.get('scope')!='TRAIN_VALID_freeze' or value.get('TEST_access') is not False or value.get('complete_VALID_scoring') is not True or value['job_sha256']!=entry['job_sha256'] or value['source_manifest_sha256']!=admission['provider_source_manifest']['sha256'] or value['cohort_plan_sha256']!=release['common_plan']['sha256']:
                raise ValueError('Provider selected-state freeze is incomplete/different')
            for key in ('arm','rule','geometry','seed','paired_seed_block'):
                if value[key]!=cell[key]:raise ValueError('Selected-state donor identity differs: '+key)
            inputs=value['input_identities']
            if common_inputs is None:common_inputs=inputs
            elif inputs!=common_inputs:raise ValueError('Actual dataset/feature/negative-pool identities differ across providers')
            tensor_directory=value_path.parent
            if not (tensor_directory/'selected_checkpoint.pt').is_file() or not (tensor_directory/'selected_VALID_logits.pt').is_file():
                raise ValueError('Selected-state artifacts missing from authenticated donor replica')
            records[identity]={'cell_id':identity,'block':block,'provider':donor['provider'],'hostname':donor['hostname'],'GPU_UUID':donor['GPU_UUID'],
                'provider_admission_relative':str(admission_path.relative_to(c.PHASE)),'provider_admission_sha256':c.sha(admission_path),
                'source_manifest_sha256':admission['provider_source_manifest']['sha256'],'program_sha256':science['program_sha256'],
                'runtime_versions':admission['runtime_versions'],'original_donor_directory_relative':prefix,
                'job_relative':str(job_path.relative_to(c.PHASE)),'job_sha256':entry['job_sha256'],
                'freeze_relative':str(value_path.relative_to(c.PHASE)),'freeze_sha256':receipt['freeze_sha256'],
                'selected_checkpoint_relative':str((tensor_directory/'selected_checkpoint.pt').relative_to(c.PHASE)),
                'selected_VALID_logits_relative':str((tensor_directory/'selected_VALID_logits.pt').relative_to(c.PHASE)),
                'checkpoint_sha256':value['checkpoint_sha256'],'VALID_logits_sha256':value['VALID_logits_sha256'],
                'donor_execution_receipt':copy.deepcopy(receipt)}
        donor_receipts.append({'block':block,'provider_admission_relative':str(admission_path.relative_to(c.PHASE)),
            'provider_admission_sha256':c.sha(admission_path),'queue_relative':str(queue_path.relative_to(c.PHASE)),
            'queue_sha256':c.sha(queue_path),'root_release_relative':str(release_path.relative_to(c.PHASE)),
            'root_release_sha256':c.sha(release_path),'block_freeze_relative':str(block_path.relative_to(c.PHASE)),
            'block_freeze_sha256':c.sha(block_path),'actual_source_manifest_sha256':admission['provider_source_manifest']['sha256']})
    if len(records)!=30 or set(records)!=set(plan['execution_order']):raise ValueError('Authenticated family lacks exactly all thirty unique fits')
    output=c.fresh_phase_directory(args.output);output.mkdir()
    (output/'COHORT_PLAN.json').write_bytes(plan_path.read_bytes());(output/'EXTERNAL_ANCHORS.json').write_bytes(anchor_path.read_bytes())
    (output/'COLLECTION_RELEASE.json').write_bytes(args.release.read_bytes())
    result={'schema':'authenticated_complete_mixed_provider_paired_pilot_collection_v1','complete':True,'physical_fits':30,
        'completed':[records[i] for i in plan['execution_order']],'donor_blocks':donor_receipts,
        'collection_release_sha256':c.sha(args.release),'pilot_source_manifest_sha256':release['pilot_source_manifest_sha256'],
        'science_contract_sha256':release['science_contract_sha256'],'cohort_plan_sha256':release['common_plan']['sha256'],
        'external_anchors_sha256':release['external_anchors']['sha256'],'input_identities':common_inputs,
        'comparative_scoring_performed':False,'quality_fields_accessed_or_emitted':False,'selected_prediction_payloads_opened':False,
        'TEST_access':False,'fits_authorized':False,'original_score_recalculation':False,'retry':False,
        'provider_provenance_retained_without_single_source_relabeling':True}
    c.write(output/'COLLECTION_FREEZE.json',result)
    print('Authenticated all thirty immutable donor fits without numerical analysis: '+str(output))


if __name__=='__main__':main()
