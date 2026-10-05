#!/usr/bin/env python3
"""Root-only allocation b0 terminal/artifact preflight; no history or quality observation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import socket
import subprocess
import custody_helpers as c

HERE = Path(__file__).resolve().parent


def read_metadata(path):
    c.require(Path(path).name not in ('FREEZE.json','CONFIG.json','EXTERNAL_ANCHORS.json'),
              'Fit outcome/runtime payload is deferred until full39')
    return c.read_metadata(path)


def live_known_handle(handle):
    """Only two known /proc stat handles. Never inspect a reused PID's argv."""
    path = Path('/proc')/str(handle['PID'])/'stat'
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return {**handle,'known_queue_handle_absent':True,'observation':'known PID absent'}
    fields = raw[raw.rfind(')')+2:].split()  # Same stat/start_ticks extraction as reviewed identity().
    actual_start = int(fields[19])
    c.require(actual_start != handle['start_ticks'], 'Known original allocation b0 queue is still present')
    return {**handle,'known_queue_handle_absent':True,'observation':'PID reused; original start handle absent',
            'foreign_argv_cwd_or_children_observed':False}


def fresh_physical(bindings):
    c.require(platform.system() == 'Linux' and socket.gethostname() == bindings['hostname'],
              'Only the authorized allocation host may perform this preflight')
    result = subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],
                            capture_output=True,text=True,check=True,timeout=bindings['query_timeout_seconds'])
    rows = result.stdout.strip().splitlines();c.require(len(rows) == 1,'Authorized singleton GPU required')
    uuid,free = [x.strip() for x in rows[0].split(',')]
    c.require(uuid == bindings['GPU_UUID'] and free.isdigit()
              and int(free)*1024**2 >= bindings['minimum_fresh_GPU_free_bytes'], 'Fresh authorized GPU/free-memory minimum differs')
    return {'hostname':bindings['hostname'],'GPU_UUID':uuid,'GPU_free_bytes':int(free)*1024**2,
            'minimum_fresh_GPU_free_bytes':bindings['minimum_fresh_GPU_free_bytes']}


def prefit(block,queue,launch,evidence):
    if block['family'] == 'original30':
        registration = read_metadata(c.binding(block['genuine_prefit_registration']))
        admission = read_metadata(c.binding(block['genuine_prefit_admission']))
        c.require(registration['registered_before_provider_first_fit'] is True and registration['block'] == 'b0'
                  and registration['donor_directory_relative'] == block['directory']
                  and registration['fallback_activated'] is False
                  and registration['queue_sha256'] == block['queue']['sha256']
                  and registration['provider_admission'] == block['genuine_prefit_admission']
                  and launch['donor_registration_sha256'] == block['genuine_prefit_registration']['sha256']
                  and launch['provider_admission_sha256'] == block['genuine_prefit_admission']['sha256']
                  and admission['approved'] is True and admission['admitted_before_provider_first_fit'] is True,
                  'Genuine original b0 registration/admission/launch authority required')
        c.require(admission['provider_source_manifest'] == block['source_manifest']
                  and admission['program'] == block['program'] and admission['runtime_versions'] == block['runtime_versions'],
                  'Actual original admitted source/runtime differs')
        evidence.extend([block['genuine_prefit_registration'],block['genuine_prefit_admission']])
    else:
        review = read_metadata(c.binding(block['root_job_review']))
        auth = read_metadata(c.binding(block['prefit_freeze_authentication']))
        c.require(review['root_provider_jobs_reviewed'] is True and review['full_nine_cell_spec_reviewed'] is True
                  and review['all_generated_job_hashes_checked'] is True
                  and review['scores_read'] is False and review['TEST_access'] is False
                  and review['freeze_authentication_sha256'] == block['prefit_freeze_authentication']['sha256']
                  and review['queue_sha256'] == block['queue']['sha256']
                  and review['generated_job_hashes'] == {k:v['sha256'] for k,v in block['job_bindings'].items()}
                  and auth['scientific_children_started'] == 0 and auth['fits'] == 0
                  and auth['scores_read'] is False and launch['provider_job_review'] == block['root_job_review'],
                  'Actual root reviewed F1 b0 job/freeze/launch authority required')
        evidence.extend([block['root_job_review'],block['prefit_freeze_authentication']])


def block_custody(bindings,block,evidence):
    plan = read_metadata(c.binding(block['plan']));queue = read_metadata(c.binding(block['queue']))
    release = read_metadata(c.binding(block['root_release']));launch = read_metadata(c.binding(block['launch']))
    evidence.extend([block[k] for k in ('plan','queue','root_release','launch','source_manifest','program','gate','operational_manifest')])
    source_manifest = c.verify_manifest(block['source_manifest']);c.binding(block['program']);c.binding(block['gate'])
    c.verify_manifest(block['operational_manifest'])
    for path,expected in read_metadata(source_manifest).get('dependencies',{}).items():
        ref = {'path':path,'sha256':expected}
        if Path(path).name == 'SOURCE_MANIFEST.json':c.verify_manifest(ref)
        else:c.binding(ref)
        evidence.append(ref)
    c.require([r for r in plan['cells'] if r['cell_id'].startswith('b0_')] == block['cells']
              and plan['complete_cycle_cost_evidence'] == block['complete_cycle_cost_evidence']
              and plan['selection'] == 'first_maximum_complete_VALID_MRR_rounded4'
              and plan['root_adopted_after_TRAIN_cost'] is True and plan['TEST_closed'] is True,
              'Exact original b0 science/cost adoption differs')
    for ref in plan['complete_cycle_cost_evidence']:c.binding(ref);evidence.append(ref)
    c.require(queue['execution_blocks'] == ['b0'] and [e['cell_id'] for e in queue['entries']] == block['ordered_cell_ids']
              and queue['training_source_manifest_sha256'] == block['source_manifest']['sha256']
              and queue['pilot_source_manifest_sha256'] == block['operational_manifest']['sha256']
              and queue['cohort_plan_sha256'] == block['plan']['sha256']
              and queue['root_release_sha256'] == block['root_release']['sha256']
              and queue['retry'] is False and queue['read_scores_or_change_family'] is False and queue['TEST_access'] is False,
              'Exact original allocation b0 queue source/plan/scope required')
    c.require(all(queue[k] == release[k] for k in
                  ('python_executable','environment_overrides','queue_hard_seconds','resource_limits','execution_blocks')),
              'Actual frozen b0 runtime/resources differ from original release')
    c.require(launch['queue_sha256'] == block['queue']['sha256'] and launch['cohort_plan_sha256'] == block['plan']['sha256']
              and launch['hostname'] == bindings['hostname'] and launch['detached_launches'] == 1
              and launch['scores_read'] is False and launch['TEST_access'] is False
              and {k:launch['queue_identity'][k] for k in ('PID','start_ticks')} == block['known_queue_handle'],
              'Genuine known b0 launch handle differs')
    prefit(block,queue,launch,evidence)
    prefix = block['directory'];start_path = c.phase_file(prefix+'/QUEUE_START.json');start = read_metadata(start_path)
    completed_path = c.phase_file(prefix+'/BLOCK_FREEZE.json');completed = read_metadata(completed_path)
    c.require(start['queue_sha256'] == completed['queue_sha256'] == block['queue']['sha256']
              and start['execution_blocks'] == completed['execution_blocks'] == ['b0']
              and start['physical_fits'] == completed['physical_fits'] == len(block['ordered_cell_ids'])
              and start['full_scientific_cohort_fits'] == block['full_family_fits']
              and completed['selected_blocks_complete'] is True
              and [r['cell_id'] for r in completed['completed']] == block['ordered_cell_ids']
              and completed['cohort_plan_sha256'] == block['plan']['sha256']
              and all(completed[k] is False for k in ('TEST_access','comparative_scoring_performed','retry'))
              and not (completed_path.parent/'QUEUE_FAILURE.json').exists(), 'Actual successful complete original b0 whole block required')
    evidence.extend([c.reference(start_path),c.reference(completed_path)])
    repository = bindings['repository'];phase = repository+'/experiments_iclr/postsubmission_20260930/'
    queue_argv = [queue['python_executable'],'-B',phase+str(Path(block['operational_manifest']['path']).parent/'run_queue.py'),
                  '--queue',phase+prefix+'/QUEUE.json']
    c.identity(start['supervisor_identity'],queue_argv,repository,repository)
    c.require(all(start['supervisor_identity'][k] == launch['queue_identity'][k] for k in ('PID','start_ticks','pgid','sid','argv')),
              'Actual saved queue launch/start identity differs')
    records = []
    for entry,receipt in zip(queue['entries'],completed['completed']):
        c.terminal(receipt);cell = next(r for r in block['cells'] if r['cell_id'] == entry['cell_id'])
        job_path = c.binding(block['job_bindings'][cell['cell_id']]);job = read_metadata(job_path)
        c.require(entry['job_relative'] == prefix+'/jobs/'+cell['cell_id']+'.json'
                  and entry['output_relative'] == prefix+'/runs/'+cell['cell_id']
                  and receipt['job_sha256'] == entry['job_sha256'] == c.sha(job_path)
                  and all(job[k] == v for k,v in cell.items())
                  and job['source_manifest_sha256'] == block['source_manifest']['sha256']
                  and job['program_sha256'] == block['program']['sha256']
                  and job['cohort_plan_sha256'] == block['plan']['sha256']
                  and job['training_step_gate'] == {'approved':True,**block['gate']}
                  and job['fits_authorized'] is True and job['VALID_values_access'] is True
                  and job['TEST_access'] is False and job['retry'] is False
                  and job['soft_seconds'] == block['fit_bounds_by_cell'][cell['cell']]['soft_seconds']
                  and entry['hard_seconds'] == block['fit_bounds_by_cell'][cell['cell']]['hard_seconds'],
                  'Exact original qualified b0 scientific job/source required')
        for role in ('source_review','runtime_qualification','feature_authority','negative_pool_authority'):
            c.require(job[role]['approved'] is True and job[role]['evidence'],'Actual frozen authority required')
            for ref in job[role]['evidence']:c.binding(ref);evidence.append(ref)
        data_ref = {'path':job['available_manifest_relative'],'sha256':job['available_manifest_sha256']}
        c.binding(data_ref);evidence.append(data_ref)
        versions = c.qualified_runtime(bindings['provider'],job)
        c.require(versions == block['runtime_versions'],'Qualified/job runtime descriptor differs')
        child_path = c.phase_file(prefix+'/logs/'+cell['cell_id']+'.CHILD_STARTED.json')
        exit_path = c.phase_file(prefix+'/logs/'+cell['cell_id']+'.EXIT.json')
        child = read_metadata(child_path);exit_record = read_metadata(exit_path)
        c.require(exit_record == {k:v for k,v in receipt.items() if k not in ('freeze_relative','freeze_sha256')}, 'Actual per-child EXIT receipt differs')
        argv = [queue['python_executable'],'-B',phase+block['program']['path'],'--job',phase+entry['job_relative'],'--output',phase+entry['output_relative']]
        c.identity(receipt['child_identity'],argv,receipt['observed_cwd'],repository)
        c.require(child['identity'] == child['raw_identity_observation'] == receipt['child_identity'] == receipt['raw_identity_observation']
                  and child['identity_admitted_for_signals'] is True and child['argv'] == argv
                  and child['limits'] == dict(queue['resource_limits'],external_hard_seconds_per_cell=entry['hard_seconds'])
                  and child['observed_cwd'] == repository and receipt['child_identity']['ppid'] == start['supervisor_identity']['PID']
                  and receipt['child_identity']['start_ticks'] >= start['supervisor_identity']['start_ticks']
                  and c.timestamp(start['UTC']) <= c.timestamp(child['UTC']) <= c.timestamp(receipt['UTC']),
                  'Actual saved b0 queue->owned child->EXIT chronology differs')
        output = c.PHASE/entry['output_relative']
        c.require(receipt['freeze_relative'] == entry['output_relative']+'/FREEZE.json'
                  and not (output/'FAILURE.json').exists(),'Successful source-fixed b0 output required')
        freeze = c.phase_file(receipt['freeze_relative']);c.require(c.sha(freeze) == receipt['freeze_sha256'],'Source FREEZE bytes changed')
        checkpoint = c.phase_file(entry['output_relative']+'/selected_checkpoint.pt')
        prediction = c.phase_file(entry['output_relative']+'/selected_VALID_logits.pt')
        artifacts = {'FREEZE':c.reference(freeze),'checkpoint':c.reference(checkpoint),'VALID_prediction':c.reference(prediction)}
        evidence.extend([c.reference(job_path),c.reference(child_path),c.reference(exit_path),*artifacts.values()])
        records.append({'cell_id':cell['cell_id'],'family':block['family'],'actual_child_identity':receipt['child_identity'],
                        'terminal_EXIT':c.reference(exit_path),'artifacts':artifacts,'artifact_hashes_observed_retrospectively':True,
                        'FREEZE_selected_artifact_semantics_checked':False,'CONFIG_runtime_observed':False})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args();release_path = c.phase_file(str(args.release.resolve(strict=True).relative_to(c.PHASE)))
    release = read_metadata(release_path)
    c.require(release['root_b0_current_custody_preflight_approved'] is True
              and all(release[k] is False for k in ('fits_authorized','scores_read','TEST_access','history_bytes_access','signals_authorized')),
              'Disabled until root metadata-only b0 preflight release')
    c.verify_manifest({'path':str((HERE/'MANIFEST.json').relative_to(c.PHASE)),'sha256':release['source_manifest_sha256']})
    c.require(c.sha(HERE/'BINDINGS.json') == release['bindings_sha256'] and release['source_review_evidence'],'Exact reviewed current-preflight source/bindings required')
    for ref in release['source_review_evidence']:c.binding(ref)
    bindings = read_metadata(HERE/'BINDINGS.json');fresh_physical(bindings)
    before = [live_known_handle(b['known_queue_handle']) for b in bindings['blocks']]
    evidence = [];records = []
    for block in bindings['blocks']:records.extend(block_custody(bindings,block,evidence))
    c.require(len(records) == len({r['cell_id'] for r in records}) == 13,'All fixed10+3 b0 donors required')
    after = [live_known_handle(b['known_queue_handle']) for b in bindings['blocks']]
    physical = fresh_physical(bindings)
    output = args.output.resolve();c.require(output.is_relative_to(c.PHASE.resolve()) and not output.exists() and output.parent.is_dir(),'Fresh root preflight directory required')
    output.mkdir()
    result = {'schema':'root_current_allocation_b0_terminal_artifact_custody_v1','UTC':datetime.now(timezone.utc).isoformat(),
        'root_authenticated_all13_terminal_and_artifact_bytes':True,
        'predetermined_b0_donors':{b['family']:b['directory'] for b in bindings['blocks']},
        'both_prior_queue_processes_absent':True,'both_means_original_allocation_main_b0_and_F1_b0_only':True,
        'known_b0_queue_handle_checks_before':before,'known_b0_queue_handle_checks_after':after,
        'fresh_authorized_physical_resource_observation':physical,'all13':records,
        'evidence':list({(r['path'],r['sha256']):r for r in evidence}.values()),
        'release_binding':c.reference(release_path),'source_manifest_sha256':release['source_manifest_sha256'],
        'scores_read':False,'TEST_access':False,'history_bytes_observed':False,'FREEZE_JSON_parsed':False,
        'CONFIG_JSON_parsed':False,'checkpoint_or_prediction_deserialized':False,'signals_sent':[],
        'fits_launched':0,'scientific_completion_or_selected_state_semantics_claimed':False,
        'full39_semantic_FREEZE_CONFIG_history_gate_still_required':True}
    with (output/'B0_CUSTODY.json').open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print('Authenticated current allocation b0 all13 terminal/artifact bytes and both known b0 queue handles absent; no science/outcome claim.')


if __name__ == '__main__':
    main()
