#!/usr/bin/env python3
"""Disabled retrospective full39 history byte inventory; never parse outcomes."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import collect_companion as c

INVENTORY_SOURCE_REVIEWED = False
SCIENCE = {'path':'shared_private_transfer_paired_pilot_preparation_20261005_v2/SCIENCE_CONTRACT.json',
           'sha256':'cdca2fe2ca5e2bfcd29e6fdee515d37570d21aaa246fca870f6a5105534d7445'}
ORIGINAL_SOURCE_SHA = {'authorized_one_GPU_allocation':'db7102df30491be8809ea4295b9ce0d5c48f3608129f09dcfef74b8f7aa2233f',
                       'authorized_18.77':'7f274c09bb317e6976d0c8b8636e779dc3ffcd2f2b76493d00f008b8bd08cebc'}
ORIGINAL_PROGRAM_SHA = '6d7e75f9bae93ef88b2873f55f4f449ae52b9a0b6768fa808fae391da50ed524'
SCHEDULE = {'max_cycles':60,'eval_every_cycles':5,'validation_miss_limit':11}


def evidence(refs):
    c.require(isinstance(refs,list) and refs, 'Actual bound review/qualification evidence required')
    for ref in refs:
        c.binding(ref)


def populations(release):
    """Finish both full ordered registry headers before per-fit metadata custody."""
    original = c.read_metadata(c.binding(c.ORIGINAL_PLAN))
    companion = c.read_metadata(c.binding(c.SPEC))
    c.binding(c.PROMOTION);c.binding(SCIENCE)
    c.require(release['original_plan_binding'] == c.ORIGINAL_PLAN and release['companion_spec_binding'] == c.SPEC
              and release['original_promotion_binding'] == c.PROMOTION, 'Immutable full39 study/promotion differs')
    joined, common_inputs = [], None
    c.require(set(release['registries']) == {'original30','companion9'}, 'Exact two unchanged registries required')
    for family, spec, size, schema in (
        ('original30',original,30,'authenticated_complete_mixed_provider_paired_pilot_collection_v1'),
        ('companion9',companion,9,'authenticated_complete_row0_companion_collection_v1')):
        path = c.binding(release['registries'][family]);registry = c.read_metadata(path)
        c.require(registry['schema'] == schema and registry['complete'] is True and registry['physical_fits'] == size,
                  'Actual complete registry required: '+family)
        rows = registry['completed']
        c.require(len(spec['cells']) == len(rows) == size and len(set(spec['execution_order'])) == size
                  and [r['cell_id'] for r in rows] == spec['execution_order'], 'No incomplete/reordered/subset registry')
        for key in ('comparative_scoring_performed','quality_fields_accessed_or_emitted','selected_prediction_payloads_opened',
                    'TEST_access','fits_authorized','original_score_recalculation','retry'):
            c.require(registry[key] is False, 'Original registry scope changed: '+key)
        if common_inputs is None:
            common_inputs = registry['input_identities']
        c.require(registry['input_identities'] == common_inputs and set(common_inputs) == c.INPUT_ROLES,
                  'Exact common TRAIN/VALID/feature/pool declarations required')
        root = path.parent
        collection_path = c.phase_file(str((root/'COLLECTION_RELEASE.json').relative_to(c.PHASE)))
        c.require(c.sha(collection_path) == registry['collection_release_sha256'], 'Actual collection release bytes differ')
        collection = c.read_metadata(collection_path)
        c.require(collection['root_collection_approved'] is True and collection['fits_authorized'] is False
                  and collection['TEST_access'] is False and collection['original_score_recalculation'] is False,
                  'Actual separately approved metadata collection required')
        evidence(collection['source_review_evidence'])
        plan_path = c.phase_file(str((root/'COHORT_PLAN.json').relative_to(c.PHASE)))
        c.require(c.sha(plan_path) == registry['cohort_plan_sha256'], 'Collected full plan bytes changed')
        plan = c.read_metadata(plan_path)
        c.require(all(plan[k] == spec[k] for k in ('cells','blocks','execution_order','resource_assignment','selection')),
                  'Original full ordered population/science differs')
        packet_ref = release['collector_source_manifests'][family]
        packet = c.verify_manifest(packet_ref)
        if family == 'original30':
            c.require(registry['cohort_plan_sha256'] == c.ORIGINAL_PLAN['sha256'] == collection['common_plan']['sha256']
                      and registry['science_contract_sha256'] == SCIENCE['sha256'] == collection['science_contract_sha256']
                      and packet_ref['sha256'] == registry['pilot_source_manifest_sha256'] == collection['pilot_source_manifest_sha256']
                      and collection['collection_contract_frozen_before_any_new_pilot_score_analysis'] is True
                      and collection['all_30_required_before_analysis'] is True and collection['no_donor_block_or_fit_selection'] is True,
                      'Actual original30 collection source/plan/science authority differs')
            anchors = c.phase_file(str((root/'EXTERNAL_ANCHORS.json').relative_to(c.PHASE)))
            c.require(c.sha(anchors) == registry['external_anchors_sha256'] == collection['external_anchors']['sha256'],
                      'Verbatim original anchors changed')
            c.binding(collection['common_plan']);c.binding(collection['external_anchors'])
        else:
            c.require(registry['companion_spec_sha256'] == c.SPEC['sha256'] and registry['original_plan_binding'] == c.ORIGINAL_PLAN
                      and registry['original_promotion_binding'] == c.PROMOTION and collection['all9_required'] is True
                      and collection['collection_contract_frozen_before_companion_score_analysis'] is True
                      and packet_ref['sha256'] == registry['collector_source_manifest_sha256'] == collection['collector_source_manifest_sha256']
                      and c.sha(packet.parent/'collect_companion.py') == c.sha(c.HERE/'collect_companion.py'),
                      'Actual separately reviewed exact9 collector identity differs')
        joined.append({'family':family,'spec':spec,'path':path,'registry':registry,'rows':rows,'collection':collection})
    c.require(len({r['cell_id'] for group in joined for r in group['rows']}) == 39, 'Exactly all39 distinct identities required')
    return joined, common_inputs


def artifacts(row):
    """Authenticate bytes only; source FREEZE semantics are later D2 work."""
    freeze = c.binding({'path':row['freeze_relative'],'sha256':row['freeze_sha256']})
    checkpoint = c.binding({'path':row['selected_checkpoint_relative'],'sha256':row['checkpoint_sha256']})
    logits = c.binding({'path':row['selected_VALID_logits_relative'],'sha256':row['VALID_logits_sha256']})
    c.require(freeze.name == 'FREEZE.json' and checkpoint.name == 'selected_checkpoint.pt'
              and logits.name == 'selected_VALID_logits.pt' and freeze.parent == checkpoint.parent == logits.parent
              and not (freeze.parent/'FAILURE.json').exists(), 'Actual complete retained source-fixed artifact set required')
    return freeze


def original_admission(donor, row, job, science):
    ref = {'path':row['provider_admission_relative'],'sha256':row['provider_admission_sha256']}
    c.require(ref == donor['provider_admission'], 'Keep actual original provider-admission identity')
    admission = c.read_metadata(c.binding(ref))
    c.require(admission['approved'] is True and admission['admitted_before_provider_first_fit'] is True
              and donor['registered_before_provider_first_fit'] is True
              and all(admission[k] == row[k] == donor[k] for k in ('provider','hostname','GPU_UUID'))
              and all(admission[k] is False for k in ('TEST_access','original_score_recalculation','numerical_science_changes'))
              and admission['authorized_repository'] == c.REPOSITORY[donor['provider']]
              and admission['science_contract_sha256'] == SCIENCE['sha256']
              and admission['available_manifest_sha256'] == c.AVAILABLE_SHA
              and admission['runtime_versions'] == row['runtime_versions'] == job['runtime_versions'],
              'Actual original pre-fit provider/source/runtime admission differs')
    c.require(row['source_manifest_sha256'] == job['source_manifest_sha256'] == admission['provider_source_manifest']['sha256']
              == ORIGINAL_SOURCE_SHA[donor['provider']] and row['program_sha256'] == job['program_sha256']
              == admission['program']['sha256'] == ORIGINAL_PROGRAM_SHA, 'Exact original provider source/program differs')
    c.source_packet(admission['provider_source_manifest'],admission['program'],ORIGINAL_SOURCE_SHA[donor['provider']],ORIGINAL_PROGRAM_SHA)
    expected, numerical = science['numerical_science_files'], admission['numerical_science_files']
    c.require(set(numerical) == set(expected), 'Complete original numerical source set required')
    for name, ref in numerical.items():
        c.require(ref['sha256'] == expected[name]['sha256'], 'Original numerical source changed: '+name)
        c.binding(ref)
    evidence(admission['qualification_evidence']);evidence(admission['root_source_review_evidence'])
    for role in ('source_review','runtime_qualification','feature_authority','negative_pool_authority'):
        c.require(job[role]['approved'] is True, 'Original job authority differs')
        evidence(job[role]['evidence'])
    gate = job['training_step_gate']
    c.require(gate['approved'] is True, 'Original source gate authority differs')
    c.binding({'path':gate['path'],'sha256':gate['sha256']})
    return admission


def fixed_inputs(job, inputs):
    c.require(job['available_manifest_sha256'] == c.AVAILABLE_SHA, 'Source-fixed acquisition manifest changed')
    available = c.read_metadata(c.binding({'path':job['available_manifest_relative'],'sha256':job['available_manifest_sha256']}))
    c.require(available['TEST_available_to_loader'] is False and set(available['files']) == c.INPUT_ROLES
              and {name:{k:available['files'][name][k] for k in ('sha256','bytes')} for name in c.INPUT_ROLES} == inputs,
              'All39 declared fixed input identities differ')


def original_custody(group, inputs):
    """Retain/recheck original30 registry identity without re-running its collector."""
    collection, spec, registry = group['collection'], group['spec'], group['registry']
    science = c.read_metadata(c.binding(SCIENCE))
    donors = collection['donor_blocks'];c.require(len(donors) == 3 and {d['block'] for d in donors} == {'b0','b1','b2'}, 'Exact original donors required')
    rows_by_id = {r['cell_id']:r for r in group['rows']};seen = set()
    for donor in donors:
        block = donor['block'];prefix = donor['donor_directory_relative'];assigned = science['resource_assignment'][block]
        if donor['fallback_activated'] is True:
            c.require(block != 'b0' and donor['provider'] == 'authorized_one_GPU_allocation'
                      and donor['hostname'] == science['resource_assignment']['b0']['hostname']
                      and donor['GPU_UUID'] == science['resource_assignment']['b0']['GPU_UUID'], 'Only original declared whole-block fallback allowed')
            evidence(donor['fallback_activation_evidence'])
        else:
            c.require(donor['provider'] == assigned['primary_provider'] and all(donor[k] == assigned[k] for k in ('hostname','GPU_UUID')),
                      'Original actual whole-block assignment changed')
        queue_path = c.replica(donor,prefix+'/QUEUE.json');q = c.read_metadata(queue_path)
        release_path = c.replica(donor,prefix+'/ROOT_RELEASE.json');root_release = c.read_metadata(release_path)
        plan_path = c.replica(donor,prefix+'/COHORT_PLAN.json');anchors_path = c.replica(donor,prefix+'/EXTERNAL_ANCHORS.json')
        block_path = c.replica(donor,prefix+'/BLOCK_FREEZE.json');complete = c.read_metadata(block_path)
        c.require(c.sha(queue_path) == donor['queue_sha256'] == complete['queue_sha256']
                  and c.sha(release_path) == donor['root_release_sha256'] == q['root_release_sha256']
                  and c.sha(plan_path) == registry['cohort_plan_sha256'] == complete['cohort_plan_sha256'] == q['cohort_plan_sha256']
                  and c.sha(anchors_path) == registry['external_anchors_sha256'] == complete['external_anchors_sha256'],
                  'Original donor queue/release/plan/anchor bytes differ')
        expected = [cell for cell in spec['execution_order'] if cell.startswith(block+'_')]
        c.require(q['execution_blocks'] == complete['execution_blocks'] == [block] and complete['selected_blocks_complete'] is True
                  and complete['physical_fits'] == len(expected) == 10 and [r['cell_id'] for r in complete['completed']] == expected
                  and [r['cell_id'] for r in q['entries']] == expected
                  and all(complete[k] is False for k in ('TEST_access','comparative_scoring_performed','retry'))
                  and not (block_path.parent/'QUEUE_FAILURE.json').exists(), 'Original exact ten-cell owned terminal block required')
        saved = next(d for d in registry['donor_blocks'] if d['block'] == block)
        for key, path in (('queue',queue_path),('root_release',release_path),('block_freeze',block_path)):
            c.require(saved[key+'_relative'] == str(path.relative_to(c.PHASE)) and saved[key+'_sha256'] == c.sha(path), 'Original registry donor custody changed')
        for entry, receipt in zip(q['entries'],complete['completed']):
            row = rows_by_id[entry['cell_id']];cell = next(s for s in spec['cells'] if s['cell_id'] == row['cell_id'])
            c.terminal(receipt)
            c.require(row['donor_execution_receipt'] == receipt and row['original_donor_directory_relative'] == prefix
                      and row['cell_id'] not in seen and cell['schedule'] == SCHEDULE, 'Original receipt/cell custody differs')
            job_path = c.replica(donor,entry['job_relative']);job = c.read_metadata(job_path)
            c.require(row['job_relative'] == str(job_path.relative_to(c.PHASE)) and row['job_sha256'] == c.sha(job_path)
                      == entry['job_sha256'] == receipt['job_sha256'] and all(job[k] == v for k,v in cell.items())
                      and job['cohort_plan_sha256'] == registry['cohort_plan_sha256'] and job['fits_authorized'] is True
                      and job['purpose'] == 'TRAIN_VALID_prospective_private_transfer_fit' and job['VALID_values_access'] is True
                      and job['TEST_access'] is False and job['retry'] is False
                      and job['soft_seconds'] == spec['fit_bounds_by_cell'][cell['cell']]['soft_seconds']
                      and entry['hard_seconds'] == spec['fit_bounds_by_cell'][cell['cell']]['hard_seconds'], 'Exact original per-fit scientific/source scope differs')
            admission = original_admission(donor,row,job,science);fixed_inputs(job,inputs)
            c.require(q['training_source_manifest_sha256'] == row['source_manifest_sha256']
                      and saved['actual_source_manifest_sha256'] == row['source_manifest_sha256']
                      and q['python_executable'] == admission['python_executable']
                      and all(q[k] == root_release[k] for k in ('python_executable','environment_overrides','queue_hard_seconds','resource_limits','execution_blocks')),
                      'Original actual queue source/runtime/bounds differ')
            original_phase = admission['authorized_repository']+'/experiments_iclr/postsubmission_20260930/'
            argv = [admission['python_executable'],original_phase+admission['program']['path'],'--job',original_phase+entry['job_relative'],
                    '--output',original_phase+entry['output_relative']]
            c.require(receipt['child_identity']['argv'] in (argv,[argv[0],'-B',*argv[1:]]), 'Original owned child argv differs')
            c.identity(receipt['child_identity'],receipt['child_identity']['argv'],receipt['observed_cwd'],admission['authorized_repository'])
            exit_path = c.replica(donor,prefix+'/logs/'+row['cell_id']+'.EXIT.json')
            c.require(c.read_metadata(exit_path) == {k:v for k,v in receipt.items() if k not in ('freeze_relative','freeze_sha256')},
                      'Original registry completion is not the actual saved owned EXIT receipt')
            freeze = artifacts(row)
            c.require(freeze == c.replica(donor,entry['output_relative']+'/FREEZE.json')
                      and receipt['freeze_relative'] == entry['output_relative']+'/FREEZE.json'
                      and receipt['freeze_sha256'] == row['freeze_sha256'], 'Original source FREEZE identity changed')
            seen.add(row['cell_id'])
    c.require(seen == set(spec['execution_order']), 'All original30 source/job/terminal/artifact custody required')


def companion_custody(group, inputs):
    """Reauthenticate all9 through the ordinary local collector metadata helpers."""
    actual, declared_inputs, plan_path = c.authenticate_nine(group['collection'])
    c.require(declared_inputs == inputs and c.sha(plan_path) == group['registry']['cohort_plan_sha256'], 'Actual exact9 common custody differs')
    for row, record in zip(group['rows'],actual):
        donor, job, receipt = record['donor'],record['job'],record['receipt']
        expected = {'cell_id':record['cell']['cell_id'],'block':donor['block'],'provider':donor['provider'],
                    'hostname':donor['hostname'],'GPU_UUID':donor['GPU_UUID'],'source_manifest_sha256':donor['provider_source_manifest']['sha256'],
                    'program_sha256':c.PROGRAM_SHA,'original_donor_directory_relative':donor['donor_directory_relative'],
                    'job_relative':str(record['job_path'].relative_to(c.PHASE)),'job_sha256':receipt['job_sha256'],
                    'freeze_relative':str((record['output']/'FREEZE.json').relative_to(c.PHASE)),'freeze_sha256':receipt['freeze_sha256'],
                    'selected_checkpoint_relative':record['checkpoint']['path'],'checkpoint_sha256':record['checkpoint']['sha256'],
                    'selected_VALID_logits_relative':record['logits']['path'],'VALID_logits_sha256':record['logits']['sha256'],
                    'donor_execution_receipt':receipt}
        c.require(all(row[k] == v for k,v in expected.items()), 'Actual complete exact9 row identity differs')
        descriptor = c.read_metadata(c.binding(row['provider_custody_binding']))
        c.require(descriptor['schema'] == 'authenticated_row0_companion_provider_custody_descriptor_v1'
                  and descriptor['descriptor_is_pre_fit_admission_object'] is False
                  and descriptor['collector_authenticated_pre_fit_evidence'] is True
                  and all(descriptor[k] == donor[k] for k in ('provider','hostname','GPU_UUID'))
                  and descriptor['provider_source_manifest'] == donor['provider_source_manifest'] and descriptor['program'] == donor['program']
                  and descriptor['pre_fit_evidence'] == c.prefit_proof(record), 'Actual derived descriptor pre-fit evidence differs')
        runtime = descriptor['runtime_provenance']
        c.require(runtime['declared_runtime_versions'] == job['runtime_versions']
                  and runtime['qualified_runtime_versions'] == record['qualified_runtime_versions'], 'Actual declared/qualified runtime differs')
        if runtime['training_runtime_observation_available'] is True:
            observed = c.binding(runtime['training_runtime_observation'])
            c.require(observed == record['output']/'CONFIG.json' and runtime['observed_training_runtime_versions'] == job['runtime_versions'],
                      'Actual CONFIG observation custody differs')
            # No CONFIG parse here. The all39 D2 semantic pass cross-checks it.
        else:
            c.require(runtime['training_runtime_observation_available'] is False and runtime['training_runtime_observation'] is None
                      and runtime['observed_training_runtime_versions'] is None and runtime['unavailable_reason'], 'Missing runtime observation must remain explicit')
        artifacts(row);fixed_inputs(job,inputs)


def authenticate_all39(release):
    joined, inputs = populations(release)
    original_custody(joined[0],inputs)
    companion_custody(joined[1],inputs)
    rows = [row for group in joined for row in group['rows']]
    c.require(len(rows) == len({r['cell_id'] for r in rows}) == 39, 'Actual all39 metadata/source/job/terminal/artifact pass required')
    return rows


def observe_histories(rows):
    # Called only after authenticate_all39 returns successfully. No expected
    # history SHA is obtained by parsing any source FREEZE or history.
    records = []
    for row in rows:
        path = c.phase_file(str(c.relative(row['freeze_relative']).parent/'VALID_HISTORY.jsonl'))
        records.append({'cell_id':row['cell_id'],
                        'freeze_binding':{'path':row['freeze_relative'],'sha256':row['freeze_sha256']},
                        'history_binding':c.reference(path)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    c.require(INVENTORY_SOURCE_REVIEWED and c.COLLECTOR_SOURCE_REVIEWED,
              'Preparation is hard disabled pending separate source/custody/root inventory review')
    release_path = c.phase_file(str(args.release.absolute().relative_to(c.PHASE)))
    release = c.read_metadata(release_path)
    c.require(release['root_history_inventory_approved'] is True and release['all39_required'] is True
              and release['retrospective_custody_only'] is True and release['source_review_evidence']
              and all(release[k] is False for k in ('fits_authorized','TEST_access','scores_read','pre_fit_authority','original_score_recalculation')),
              'Separate root release for actual complete39 retrospective byte custody required')
    evidence(release['source_review_evidence']);c.verify_packet(release['inventory_source_manifest_sha256'])
    rows = authenticate_all39(release)
    records = observe_histories(rows)
    output = c.fresh_output(args.output);output.mkdir()
    (output/'HISTORY_INVENTORY_RELEASE.json').write_bytes(release_path.read_bytes())
    c.write(output/'VALID_HISTORY_INVENTORY.json',{
        'schema':'retrospective_full39_VALID_history_inventory_v1','complete':True,'physical_fits':39,
        'created_UTC':datetime.now(timezone.utc).isoformat(),'created_after_all39_terminal_and_completeness_checks':True,
        'history_or_FREEZE_JSON_parsed':False,'scores_read':False,'pre_fit_authority':False,
        'registries':release['registries'],'records':records,'release_binding':c.reference(release_path),
        'inventory_source_manifest_sha256':release['inventory_source_manifest_sha256'],
        'authority':'Retrospective byte custody only; not pre-fit freeze or independent historical authority.',
        'source_FREEZE_history_reference_crosscheck_deferred_to_full39_D2':True})
    print('Observed all39 source-fixed history byte custody without parsing: '+str(output))


if __name__ == '__main__':
    main()
