#!/usr/bin/env python3
"""Authenticate fixed full39 with reviewed custody helpers; never parse fit outcomes."""
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
import protocol as p

PHASE = p.HERE.parent
SOURCE = {'original30':('shared_backbone_private_transfer_training_source_20261005_v2',
                       'db7102df30491be8809ea4295b9ce0d5c48f3608129f09dcfef74b8f7aa2233f',
                       '6d7e75f9bae93ef88b2873f55f4f449ae52b9a0b6768fa808fae391da50ed524'),
          'companion9':('shared_private_transfer_row0_single_companion_preparation_20261005_v1',
                        '67fab0016a144fdfda639e19cf1cf7feae4d10cbec04f1eaf8187917aa975ea8',
                        'd2c7518bf1904aa7c7612ea5165a514ac7d5b11ddb0357937a56015cdb161ce6')}


def helpers():
    """Load hash-bound stdlib metadata helpers, never any numerical source."""
    refs = p.read(p.HERE/'REUSED_CUSTODY_HELPERS.json')
    paths = {name:p.binding(PHASE,ref) for name,ref in refs.items()}
    sys.path.insert(0,str(paths['collect_companion.py'].parent))
    import collect_companion as c
    p.require(Path(c.__file__).resolve() == paths['collect_companion.py'].resolve(), 'Wrong custody helper module')
    spec = importlib.util.spec_from_file_location('reviewed_full39_metadata_helpers',paths['inventory_full39_histories.py'])
    i = importlib.util.module_from_spec(spec);spec.loader.exec_module(i)
    p.require(i.c is c and c.PHASE.resolve() == PHASE.resolve(), 'Custody helpers use the actual collection phase')
    return c,i


def new_source_custody(c,donor,family,packet_sha):
    directory,source_sha,program_sha = SOURCE[family];custody = donor['source_custody']
    training = c.source_copy_manifest(custody['training_manifest'],directory+'/SOURCE_MANIFEST.json',source_sha,donor)
    program = p.file_in(PHASE,str((training.parent/'run.py').relative_to(PHASE)))
    p.require(p.sha(program) == program_sha, 'Actual executed numerical source differs')
    p.require(set(custody['external_dependencies']) == set(c.EXTERNAL_TRAINING),'All external numerical sources required')
    for original,expected in c.EXTERNAL_TRAINING.items():
        ref = custody['external_dependencies'][original]
        if p.relative(original).name == 'SOURCE_MANIFEST.json':
            c.source_copy_manifest(ref,original,expected,donor)
        else:
            c.source_copy(ref,original,expected,donor)
    operational = c.source_copy_manifest(custody['operational_manifest'],
        p.HERE.name+'/MANIFEST.json',packet_sha,donor)
    names = {r['path'] for r in c.read_metadata(operational)['files']}
    p.require({family+'/run_queue.py',family+'/pilot_common.py','protocol.py','freeze_all26.py','launch_next_once.py'} <= names,
              'Actual reviewed replication queue/helper/launcher source custody required')
    sup = c.SUPERVISOR[p.PROVIDER]
    c.source_copy(custody['supervisor'],sup['path'],sup['sha256'],donor)
    p.require(custody['operational_review_evidence'],'Operational review custody required')
    for ref in custody['operational_review_evidence']:
        c.binding(ref)
    return training,program,operational


def new_prefit(c,donor,a,release,q,root_release):
    family,block = donor['family'],donor['block']
    descriptor = next(r for r in a['queues'] if r['family'] == family and r['block'] == block)
    p.require(donor['physical_attempt_id'] == descriptor['attempt_id']
              and donor['donor_directory_relative'] == descriptor['execution_directory_relative']
              and all(q[k] == v for k,v in descriptor.items()), 'Prospective chosen attempt differs')
    p.require(root_release['execution_authorized'] is True and root_release['explicit_replication_exception_approved'] is True
              and root_release['source_review_approved'] is True and root_release['all26_selected_before_any_score'] is True
              and root_release['full39_collection_contract_approved'] is True
              and root_release['operational_manifest_sha256'] == release['collector_manifest_sha256']
              and root_release['amendment_sha256'] == release['amendment_sha256']
              and root_release['attempt_history_sha256'] == release['attempt_history_sha256'], 'Actual replication root release differs')
    for refs in (root_release['root_review_evidence'],root_release['independent_review_evidence']):
        p.require(refs,'Independent/root review authority required')
        for ref in refs:c.binding(ref)
    launch = c.read_metadata(c.binding(donor['launch_receipt']))
    p.require(launch['launch_intent']['path'] == donor['donor_directory_relative']+'/LAUNCH_INTENT.json',
              'Emitter-fixed launch intent path differs')
    intent = c.read_metadata(c.donor_binding(donor,launch['launch_intent']))
    review = c.read_metadata(c.binding(launch['root_job_review']))
    auth = c.read_metadata(c.binding(review['frozen_authentication']))
    registration = c.read_metadata(c.donor_binding(donor,auth['queues'][descriptor['queue_id']]['registration']))
    p.require(launch['queue_id'] == q['queue_id'] == descriptor['queue_id']
              and launch['attempt_id'] == descriptor['attempt_id'] and launch['detached_launches'] == 1
              and launch['identity_admitted_for_signals'] is True and launch['attempts'] == 1
              and all(launch[k] is False for k in ('scores_read','TEST_access','retry'))
              and intent['root_job_review'] == launch['root_job_review']
              and intent['serial_index'] == a['serial_queue_order'].index(q['queue_id'])
              and intent['all26_job_hashes_reviewed_before_first_fit'] is True
              and review['all26_generated_jobs_reviewed'] is True and review['full39_collection_contract_reviewed'] is True
              and review['explicit_replication_exception_reviewed'] is True
              and review['generated_job_hashes'] == auth['job_hashes']
              and auth['scientific_children_started'] == 0 and auth['physical_fits'] == 26
              and auth['amendment_sha256'] == release['amendment_sha256']
              and auth['attempt_history_sha256'] == release['attempt_history_sha256'], 'Actual all26 prefit review->launch authority differs')
    expected_ids = {r['cell_id'] for r in a['new_selected_attempts']}
    p.require(set(auth['job_hashes']) == expected_ids and set(auth['queues']) == set(a['serial_queue_order'])
              and all(review['queues'][key] == auth['queues'][key]['queue'] for key in a['serial_queue_order']),
              'Review must retain all26 fixed donors before scores')
    p.require(registration['registered_before_first_new_fit'] is True and registration['scientific_children_started'] == 0
              and registration['replication_exception_explicitly_approved'] is True
              and registration['old_attempts_preserved_current_status_unknown'] is True
              and registration['chosen_cells'] == descriptor['ordered_cell_ids']
              and registration['job_hashes'] == {k:auth['job_hashes'][k] for k in descriptor['ordered_cell_ids']}
              and registration['attempt_history_sha256'] == release['attempt_history_sha256'], 'Genuine fresh physical attempt registration required')
    p.require(q['retry'] is False and q['read_scores_or_change_family'] is False and q['TEST_access'] is False
              and q['python_executable'] == a['python_executable'] and q['environment_overrides'] == a['environment_overrides']
              and q['resource_limits'] == a['resource_limits'] and q['training_source_manifest_sha256'] == SOURCE[family][1]
              and launch['source_manifest_sha256'] == SOURCE[family][1]
              and launch['operational_manifest_sha256'] == release['collector_manifest_sha256'], 'Actual source/runtime/resources differ')
    p.require(launch['queue_sha256'] == donor['queue_binding']['sha256']
              == auth['queues'][q['queue_id']]['queue']['sha256'] == registration['queue']['sha256']
              and launch['root_release_sha256'] == donor['root_release_binding']['sha256'] == registration['root_release']['sha256'],
              'Pre-fit queue/release identity differs')
    p.require(len(intent['predecessors']) == intent['serial_index'], 'Every preceding queue custody required')
    for predecessor,descriptor_before in zip(intent['predecessors'],a['queues'][:intent['serial_index']]):
        selected_predecessors = [d for d in release['donors']
            if d['family'] == descriptor_before['family'] and d['block'] == descriptor_before['block']]
        p.require(len(selected_predecessors) == 1, 'Exactly one predetermined predecessor donor required')
        chosen_predecessor = selected_predecessors[0]
        p.require(chosen_predecessor['donor_directory_relative'] == descriptor_before['execution_directory_relative']
                  and predecessor['block_freeze']['path'] == descriptor_before['execution_directory_relative']+'/BLOCK_FREEZE.json'
                  and predecessor['block_freeze'] == chosen_predecessor['block_freeze_binding'],
                  'Emitter-fixed predecessor reference must match its chosen donor freeze binding')
        before = c.read_metadata(c.donor_binding(chosen_predecessor,predecessor['block_freeze']))
        p.require(predecessor['queue_id'] == descriptor_before['queue_id']
                  and predecessor['prior_queue_identity_no_longer_present'] is True
                  and before['selected_blocks_complete'] is True
                  and [r['cell_id'] for r in before['completed']] == descriptor_before['ordered_cell_ids'],
                  'Fixed serial whole-block predecessor evidence differs')
    training,program,operational = new_source_custody(c,donor,family,release['collector_manifest_sha256'])
    return launch,auth['job_hashes'],training,program,operational


def authenticate39(c,i,release,a):
    donors = release['donors']
    p.require([(d['family'],d['block']) for d in donors] ==
              [('original30','b0'),('original30','b1'),('original30','b2'),('companion9','b0'),('companion9','b1'),('companion9','b2')],
              'Exactly six predetermined whole-block donors required')
    records = [];common_inputs = None
    for donor in donors:
        family,block = donor['family'],donor['block'];prefix = donor['donor_directory_relative']
        p.require(donor['provider'] == p.PROVIDER and donor['hostname'] == a['hostname'] and donor['GPU_UUID'] == p.GPU_UUID
                  and donor['repository'] == str(p.REMOTE_REPO), 'Actual allocation donor identity differs')
        if block == 'b0':p.require(prefix == a['chosen_b0_donors'][family], 'Predetermined b0 donor changed')
        plan_path = c.donor_binding(donor,donor['cohort_plan_binding']);plan = c.read_metadata(plan_path)
        p.require(p.sha(plan_path) == a['family_plans'][family]['sha256']
                  and plan['cells'] == [r['scientific_configuration'] for r in a['logical_cells39'] if r['family'] == family],
                  'Exact original full-family scientific plan required')
        for key in ('complete_cycle_cost_evidence','root_numeric_decision','fit_bounds_by_cell'):
            p.require(plan[key] == a['family_metadata'][family][key], 'Canonical scientific cost/bounds authority differs')
        for ref in plan['complete_cycle_cost_evidence']+[plan['root_numeric_decision']]:c.binding(ref)
        p.require(plan['root_adopted_after_TRAIN_cost'] is True and plan['TEST_closed'] is True
                  and plan['selection'] == 'first_maximum_complete_VALID_MRR_rounded4'
                  and plan['selection_budget_fairness_approved'] is True, 'Canonical selector/adoption differs')
        q = c.read_metadata(c.donor_binding(donor,donor['queue_binding']))
        root_release_path = c.donor_binding(donor,donor['root_release_binding']);root_release = c.read_metadata(root_release_path)
        block_path = c.donor_binding(donor,donor['block_freeze_binding']);complete = c.read_metadata(block_path)
        start_path = c.replica(donor,prefix+'/QUEUE_START.json');start = c.read_metadata(start_path)
        order = [cell for cell in plan['execution_order'] if cell.startswith(block+'_')]
        p.require(q['execution_blocks'] == complete['execution_blocks'] == start['execution_blocks'] == [block]
                  and complete['selected_blocks_complete'] is True and complete['physical_fits'] == len(order)
                  and [e['cell_id'] for e in q['entries']] == [r['cell_id'] for r in complete['completed']] == order
                  and q['cohort_plan_sha256'] == complete['cohort_plan_sha256'] == donor['cohort_plan_binding']['sha256']
                  and start['queue_sha256'] == complete['queue_sha256'] == donor['queue_binding']['sha256']
                  and q['root_release_sha256'] == donor['root_release_binding']['sha256']
                  and all(complete[k] is False for k in ('TEST_access','comparative_scoring_performed','retry'))
                  and not (block_path.parent/'QUEUE_FAILURE.json').exists(), 'Complete successful exact whole-block custody required')
        p.require(start['physical_fits'] == len(order)
                  and start['full_scientific_cohort_fits'] == (39 if block != 'b0' else (30 if family == 'original30' else 9))
                  and all(start[k] is False for k in ('TEST_access','comparative_scoring_performed','retry')),
                  'Actual queue start scientific scope differs')
        admission = None
        if block != 'b0':
            launch,job_hashes,training,program,operational = new_prefit(c,donor,a,release,q,root_release)
            queue_program = p.HERE.name+'/'+family+'/run_queue.py';observed_cwd = launch['observed_cwd']
        elif family == 'companion9':
            expected,auth,launch = c.prefit(donor,c.read_metadata(c.binding(c.SPEC)))
            job_hashes = expected['job_sha256_by_cell_id'];observed_cwd = launch['observed_cwd']
            training,program,operational = c.authenticate_source_custody(donor,donor['source_custody'],
                SOURCE[family][0],SOURCE[family][1],SOURCE[family][2],c.COMPANION_OPERATIONAL[p.PROVIDER])
            queue_program = str(p.relative(c.COMPANION_OPERATIONAL[p.PROVIDER]['path']).parent/'run_queue.py')
            p.require(root_release['complete_cycle_cost_evidence'] == plan['complete_cycle_cost_evidence']
                      and q['resource_limits'] == a['resource_limits'] and q['queue_hard_seconds'] == 95700
                      and q['pilot_source_manifest_sha256'] == c.COMPANION_OPERATIONAL[p.PROVIDER]['sha256']
                      and q['training_source_manifest_sha256'] == SOURCE[family][1], 'Actual qualified b0 F1 source/cost/resources differ')
        else:
            p.require(donor['registered_before_provider_first_fit'] is True and donor['fallback_activated'] is False,
                      'Preserve real original b0 registration')
            supplemental = donor['original_operational_custody']
            p.require(supplemental['source_custody'] == donor['source_custody'], 'Retain one actual b0 source custody inventory')
            start,training,program = i.original_queue_custody(donor,q,root_release,donor['cohort_plan_binding']['sha256'],supplemental)
            launch = c.read_metadata(c.binding(supplemental['launch_receipt']));observed_cwd = launch['queue_identity']['cwd']
            job_hashes = {e['cell_id']:e['job_sha256'] for e in q['entries']}
            queue_program = str(p.relative(i.ORIGINAL_OPERATIONAL[p.PROVIDER]['path']).parent/'run_queue.py')
            admission = c.read_metadata(c.binding(donor['provider_admission']))
            operational = c.binding(donor['source_custody']['operational_manifest'])
        queue_argv = [q['python_executable'],'-B',donor['repository']+'/experiments_iclr/postsubmission_20260930/'+queue_program,
                      '--queue',donor['repository']+'/experiments_iclr/postsubmission_20260930/'+prefix+'/QUEUE.json']
        c.identity(launch['queue_identity'],queue_argv,observed_cwd,donor['repository'])
        c.identity(start['supervisor_identity'],queue_argv,observed_cwd,donor['repository'])
        p.require(all(start['supervisor_identity'][k] == launch['queue_identity'][k] for k in ('PID','start_ticks','pgid','sid','argv')),
                  'Actual queue launch/start process differs')
        for entry,receipt in zip(q['entries'],complete['completed']):
            c.terminal(receipt);cell = next(row for row in plan['cells'] if row['cell_id'] == entry['cell_id'])
            job_path = c.replica(donor,entry['job_relative']);job = c.read_metadata(job_path)
            p.require(p.sha(job_path) == entry['job_sha256'] == receipt['job_sha256'] == job_hashes[cell['cell_id']]
                      and all(job[k] == v for k,v in cell.items()) and job['cohort_plan_sha256'] == p.sha(plan_path)
                      and job['source_manifest_sha256'] == SOURCE[family][1] and job['program_sha256'] == SOURCE[family][2]
                      and job['fits_authorized'] is True and job['VALID_values_access'] is True
                      and job['TEST_access'] is False and job['retry'] is False
                      and job['soft_seconds'] == a['family_metadata'][family]['fit_bounds_by_cell'][cell['cell']]['soft_seconds']
                      and entry['hard_seconds'] == a['family_metadata'][family]['fit_bounds_by_cell'][cell['cell']]['hard_seconds'],
                      'Actual exact scientific job/source/bounds required')
            if block != 'b0':
                selected = next(r for r in a['new_selected_attempts'] if r['cell_id'] == cell['cell_id'])
                p.require(job == p.job_from_preview(PHASE,a,selected), 'Predetermined new physical attempt job differs')
            for role in ('source_review','runtime_qualification','feature_authority','negative_pool_authority'):
                p.require(job[role]['approved'] is True and job[role]['evidence'],'Actual authority required')
                for ref in job[role]['evidence']:c.binding(ref)
            gate = job['training_step_gate'];p.require(gate['approved'] is True,'Exact-source gate required')
            c.binding({'path':gate['path'],'sha256':gate['sha256']})
            p.require(gate['sha256'] == p.read(p.HERE/('BASE_JOB_'+family+'.json'))['training_step_gate']['sha256'], 'Exact qualified gate differs')
            versions = c.qualified_runtime(p.PROVIDER,job)
            available = c.read_metadata(c.binding({'path':job['available_manifest_relative'],'sha256':job['available_manifest_sha256']}))
            p.require(job['available_manifest_sha256'] == c.AVAILABLE_SHA and available['TEST_available_to_loader'] is False
                      and set(available['files']) == c.INPUT_ROLES, 'Exact fixed non-TEST input authority required')
            inputs = {name:{k:available['files'][name][k] for k in ('sha256','bytes')} for name in c.INPUT_ROLES}
            if common_inputs is None:common_inputs = inputs
            p.require(inputs == common_inputs,'Fixed scientific inputs differ across all39')
            if admission is not None:
                original_row = {'provider_admission_relative':donor['provider_admission']['path'],
                    'provider_admission_sha256':donor['provider_admission']['sha256'],'source_manifest_sha256':job['source_manifest_sha256'],
                    'program_sha256':job['program_sha256'],'runtime_versions':versions,
                    **{k:donor[k] for k in ('provider','hostname','GPU_UUID')}}
                i.original_admission(donor,original_row,job,c.read_metadata(c.binding(i.SCIENCE)),training,program,donor['source_custody'])
            output = c.replica(donor,receipt['freeze_relative']).parent
            p.require(receipt['freeze_relative'] == entry['output_relative']+'/FREEZE.json'
                      and p.sha(output/'FREEZE.json') == receipt['freeze_sha256'] and not (output/'FAILURE.json').exists(),
                      'Owned successful terminal lacks authentic source FREEZE bytes')
            child_path = c.replica(donor,prefix+'/logs/'+cell['cell_id']+'.CHILD_STARTED.json')
            exit_path = c.replica(donor,prefix+'/logs/'+cell['cell_id']+'.EXIT.json')
            child,exit_record = c.read_metadata(child_path),c.read_metadata(exit_path)
            p.require(exit_record == {k:v for k,v in receipt.items() if k not in ('freeze_relative','freeze_sha256')}, 'Actual EXIT receipt differs')
            argv = [q['python_executable'],'-B',donor['repository']+'/experiments_iclr/postsubmission_20260930/'+SOURCE[family][0]+'/run.py',
                    '--job',donor['repository']+'/experiments_iclr/postsubmission_20260930/'+entry['job_relative'],
                    '--output',donor['repository']+'/experiments_iclr/postsubmission_20260930/'+entry['output_relative']]
            c.identity(receipt['child_identity'],argv,receipt['observed_cwd'],donor['repository'])
            p.require(child['identity'] == child['raw_identity_observation'] == receipt['child_identity'] == receipt['raw_identity_observation']
                      and child['identity_admitted_for_signals'] is True and child['argv'] == argv
                      and child['observed_cwd'] == receipt['observed_cwd'] == donor['repository']
                      and child['limits'] == dict(q['resource_limits'],external_hard_seconds_per_cell=entry['hard_seconds'])
                      and receipt['child_identity']['ppid'] == start['supervisor_identity']['PID']
                      and receipt['child_identity']['start_ticks'] >= start['supervisor_identity']['start_ticks']
                      and c.timestamp(start['UTC']) <= c.timestamp(child['UTC']) <= c.timestamp(receipt['UTC']),
                      'Actual owned queue->child->EXIT chronology differs')
            checkpoint = c.phase_file(str((output/'selected_checkpoint.pt').relative_to(PHASE)))
            logits = c.phase_file(str((output/'selected_VALID_logits.pt').relative_to(PHASE)))
            records.append({'family':family,'cell':cell,'job':job,'donor':donor,'receipt':receipt,'output':output,
                'job_path':job_path,'checkpoint':c.reference(checkpoint),'logits':c.reference(logits),
                'queue_start':c.reference(start_path),'child_start':c.reference(child_path),'exit_receipt':c.reference(exit_path),
                'block_freeze':c.reference(block_path),'root_release':c.reference(root_release_path),
                'source_custody':donor['source_custody'],'qualified_runtime_versions':versions})
    p.require(len(records) == len({r['cell']['cell_id'] for r in records}) == 39
              and [r['cell']['cell_id'] for r in records] == a['selected39_order'], 'Exactly all fixed39 custody required')
    return records,common_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args();release_path = p.file_in(PHASE,str(args.release.resolve(strict=True).relative_to(PHASE)))
    release = p.read(release_path)
    p.require(release['root_collection_approved'] is True and release['full39_custody_before_score_history_parsing'] is True
              and release['no_old_new_outcome_selection'] is True
              and all(release[k] is False for k in ('fits_authorized','TEST_access','scores_read')),
              'Disabled until separate reviewed complete39 metadata release')
    p.verify_packet(release['collector_manifest_sha256'])
    p.require(p.sha(p.HERE/'PROSPECTIVE_AMENDMENT.json') == release['amendment_sha256']
              and p.sha(p.HERE/'ATTEMPT_HISTORY.json') == release['attempt_history_sha256'], 'Prospective donor/history bytes changed')
    p.require(release['source_review_evidence'],'Actual collector review required')
    for ref in release['source_review_evidence']:p.binding(PHASE,ref)
    a = p.read(p.HERE/'PROSPECTIVE_AMENDMENT.json');history = p.read(p.HERE/'ATTEMPT_HISTORY.json')
    for old in history['old77_attempt_registration_and_observation']:
        p.binding(PHASE,old['launch_receipt']);p.binding(PHASE,old['queue'])
        for key in ('actual_original_donor_registration','actual_original_provider_admission','original_root_release'):
            p.binding(PHASE,old[key])
        p.require(old['latest_terminal_status'] == 'UNKNOWN_AFTER_WITHDRAWAL', 'Do not infer withdrawn attempt outcomes')
    p.binding(PHASE,history['old77_saved_observation'])
    c,i = helpers()
    records,inputs = authenticate39(c,i,release,a)
    # This point is reached only after all39 metadata/source/terminal/artifact checks.
    # No history bytes, score-bearing FREEZE or CONFIG are observed by this collector.
    output = c.fresh_output(args.output);output.mkdir()
    (output/'COLLECTION_RELEASE.json').write_bytes(release_path.read_bytes())
    (output/'ATTEMPT_HISTORY.json').write_bytes((p.HERE/'ATTEMPT_HISTORY.json').read_bytes())
    completed = []
    for record in records:
        donor,job,cell,receipt = record['donor'],record['job'],record['cell'],record['receipt']
        descriptor_path = output/('provider_custody_'+cell['cell_id']+'.json')
        descriptor = {'schema':'authenticated_allocation_replication_actual_custody_descriptor_v1',
            'descriptor_is_pre_fit_admission_object':False,'collector_authenticated_real_pre_fit_evidence':True,
            'provider':donor['provider'],'hostname':donor['hostname'],'GPU_UUID':donor['GPU_UUID'],
            'actual_donor':donor,'source_manifest_sha256':job['source_manifest_sha256'],'program_sha256':job['program_sha256'],
            'declared_runtime_versions':job['runtime_versions'],'qualified_runtime_versions':record['qualified_runtime_versions'],
            'CONFIG_runtime_observation_deferred':True,'retrospective_executable_source_custody':record['source_custody'],
            'chronology':{key:record[key] for key in ('queue_start','child_start','exit_receipt','block_freeze','root_release')},
            'cross_host_wall_clock_order_assumed':False,'attempt_history_binding':p.reference(PHASE,output/'ATTEMPT_HISTORY.json')}
        c.write(descriptor_path,descriptor)
        completed.append({'cell_id':cell['cell_id'],'family':record['family'],'block':donor['block'],
            'provider':donor['provider'],'hostname':donor['hostname'],'GPU_UUID':donor['GPU_UUID'],
            'physical_attempt_id':job.get('physical_attempt_id'),
            'physical_attempt_id_was_source_emitted':donor['block'] != 'b0',
            'retained_actual_attempt_identity':{'donor_directory_relative':donor['donor_directory_relative'],
                'job_sha256':receipt['job_sha256'],'child_identity':receipt['child_identity']},
            'provider_custody_binding':c.reference(descriptor_path),'source_manifest_sha256':job['source_manifest_sha256'],
            'program_sha256':job['program_sha256'],'runtime_versions':job['runtime_versions'],
            'original_donor_directory_relative':donor['donor_directory_relative'],'job_relative':str(record['job_path'].relative_to(PHASE)),
            'job_sha256':receipt['job_sha256'],'freeze_relative':str((record['output']/'FREEZE.json').relative_to(PHASE)),
            'freeze_sha256':receipt['freeze_sha256'],'selected_checkpoint_relative':record['checkpoint']['path'],
            'checkpoint_sha256':record['checkpoint']['sha256'],'selected_VALID_logits_relative':record['logits']['path'],
            'VALID_logits_sha256':record['logits']['sha256'],'donor_execution_receipt':receipt})
    c.write(output/'COLLECTION_FREEZE.json',{'schema':'authenticated_complete_prospective_allocation_replication39_v1',
        'complete':True,'selected_logical_scientific_fits':39,'selected_physical_fits':39,'new_physical_fits':26,
        'physical_fits_lower_bound_including_known_old77_starts':45,'old77_final_physical_fits_and_costs_unknown':True,
        'completed':completed,'input_identities':inputs,'collection_release_sha256':p.sha(release_path),
        'collector_manifest_sha256':release['collector_manifest_sha256'],'amendment_sha256':release['amendment_sha256'],
        'attempt_history_binding':p.reference(PHASE,output/'ATTEMPT_HISTORY.json'),
        'full39_terminal_source_artifact_custody_passed':True,'comparative_scoring_performed':False,
        'TEST_access':False,'fits_authorized':False,'history_bytes_observed':False,'history_JSON_parsed':False,
        'FREEZE_JSON_parsed':False,'CONFIG_JSON_parsed':False,'selected_prediction_payloads_deserialized':False,
        'quality_fields_accessed_or_emitted':False,'no_old_new_outcome_selection':True,'automatic_retry':False,
        'next_gate':'Separate root-reviewed full39 history byte inventory then D2 semantic custody/scoring adapter for this explicit amended registry.'})
    print('Authenticated all fixed39; retained both attempt histories; no scores/history/CONFIG parsed.')


if __name__ == '__main__':
    main()
