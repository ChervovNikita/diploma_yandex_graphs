#!/usr/bin/env python3
"""Freeze four complete predetermined allocation queues; starts zero scientific children."""
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import protocol as p


def family_module(family):
    path = p.HERE/family/'pilot_common.py'
    spec = importlib.util.spec_from_file_location('replication_'+family, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    args = parser.parse_args()
    phase = p.REMOTE_PHASE
    release_path = p.file_in(phase, str(args.release.resolve(strict=True).relative_to(phase)))
    release = p.read(release_path)
    a = p.verify_frozen_protocol(phase, release)
    control = phase/p.relative(release['execution_control_relative'])
    p.require(not control.exists() and control.parent.is_dir(), 'Fresh replication control directory required')
    roots = {q['queue_id']:phase/p.relative(q['execution_directory_relative']) for q in a['queues']}
    p.require(all(not root.exists() and root.parent.is_dir() for root in roots.values()), 'Every new attempt directory must be fresh')
    modules = {family:family_module(family) for family in ('original30','companion9')}
    plans = {}
    for family,c in modules.items():
        c.physical_host(); c.verify_training()
        p.require(Path(a['python_executable']).is_file(), 'Qualified interpreter missing')
        c.check_environment(a['environment_overrides'])
        plans[family] = p.canonical_plan(phase, a, family)
    # Every byte/authority is checked before any output directory is made.
    jobs = {}
    for row in a['new_selected_attempts']:
        job = p.job_from_preview(phase, a, row)
        c = modules[row['family']]
        p.require(job['source_manifest_sha256'] == c.SOURCE_SHA
                  and job['program_sha256'] == p.sha(c.SOURCE/'run.py')
                  and job['training_step_gate'] == {'approved':True,'path':c.GATE_RELATIVE,'sha256':c.GATE_SHA}
                  and job['TEST_access'] is False and job['retry'] is False, 'Exact qualified source/gate/scope required')
        for key in ('source_review','runtime_qualification','feature_authority','negative_pool_authority'):
            p.require(job[key]['approved'] is True and job[key]['evidence'], 'Qualified job authority required')
            for ref in job[key]['evidence']:
                p.binding(phase, ref)
        p.binding(phase, {'path':job['available_manifest_relative'],'sha256':job['available_manifest_sha256']})
        p.require(job['schedule'] == {'max_cycles':60,'eval_every_cycles':5,'validation_miss_limit':11}
                  and job['runtime_versions'] == p.read(p.HERE/('BASE_JOB_'+row['family']+'.json'))['runtime_versions'],
                  'Sixty-cycle scientific recipe/runtime changed')
        jobs[row['cell_id']] = job
    p.require(len(jobs) == 26 and list(jobs) == [i for q in a['queues'] for i in q['ordered_cell_ids']], 'Exactly ordered26 required')
    control.mkdir()
    p.write(control/'ROOT_RELEASE.json', release)
    frozen = {'schema':'prospective_allocation_replication_freeze_authentication_v1',
        'UTC':datetime.now(timezone.utc).isoformat(),'scientific_children_started':0,'scores_read':False,
        'TEST_access':False,'physical_fits':26,'logical_scientific_fits':39,
        'operational_manifest_sha256':release['operational_manifest_sha256'],
        'amendment_sha256':release['amendment_sha256'],'attempt_history_sha256':release['attempt_history_sha256'],
        'root_release':p.reference(phase, control/'ROOT_RELEASE.json'),'queues':{},'job_hashes':{}}
    for index,descriptor in enumerate(a['queues']):
        c = modules[descriptor['family']]
        root = roots[descriptor['queue_id']]
        root.mkdir(); (root/'jobs').mkdir(); (root/'runs').mkdir(); (root/'logs').mkdir()
        (root/'ROOT_RELEASE.json').write_bytes(release_path.read_bytes())
        (root/'COHORT_PLAN.json').write_bytes(plans[descriptor['family']][0].read_bytes())
        if descriptor['family'] == 'original30':
            (root/'EXTERNAL_ANCHORS.json').write_bytes(p.binding(phase,a['external_anchors']).read_bytes())
        entries = []
        for cell in descriptor['ordered_cell_ids']:
            job = jobs[cell]; path = root/'jobs'/(cell+'.json'); p.write(path, job)
            row = next(r for r in a['new_selected_attempts'] if r['cell_id'] == cell)
            entries.append({'cell_id':cell,'job_relative':str(path.relative_to(phase)),
                'job_sha256':p.sha(path),'output_relative':str((root/'runs'/cell).relative_to(phase)),
                'hard_seconds':row['fit_bounds_seconds']['hard_seconds'], 'physical_attempt_id':row['physical_attempt_id']})
            frozen['job_hashes'][cell] = p.sha(path)
        queue = {**descriptor,'schema':'explicit_prospective_allocation_replication_queue_v1',
            'serial_index':index,'pilot_source_manifest_sha256':release['operational_manifest_sha256'],
            'training_source_manifest_sha256':c.SOURCE_SHA,'cohort_plan_sha256':descriptor['cohort_plan']['sha256'],
            'root_release_sha256':p.sha(root/'ROOT_RELEASE.json'),'execution_blocks':[descriptor['block']],
            'python_executable':a['python_executable'],'environment_overrides':a['environment_overrides'],
            'resource_limits':a['resource_limits'],'entries':entries,'retry':False,'stop_on_operational_failure':True,
            'read_scores_or_change_family':False,'TEST_access':False,'scores_read':False,
            'replication_not_setup_fallback_or_resumption':True,'attempt_history_sha256':release['attempt_history_sha256'],
            'execution_control_relative':release['execution_control_relative']}
        p.require(queue['queue_hard_seconds'] == sum(e['hard_seconds'] for e in entries)
                  +len(entries)*a['resource_limits']['resource_wait_seconds']+queue['queue_overhead_seconds'],
                  'Conservative complete-queue allowance differs')
        p.write(root/'QUEUE.json',queue)
        registration = {'schema':'genuine_prefit_prospective_replication_registration_v1',
            'UTC':frozen['UTC'],'queue_id':descriptor['queue_id'],'attempt_id':descriptor['attempt_id'],
            'provider':p.PROVIDER,'hostname':a['hostname'],'GPU_UUID':p.GPU_UUID,
            'registered_before_first_new_fit':True,'scientific_children_started':0,'scores_read':False,
            'chosen_cells':descriptor['ordered_cell_ids'],'job_hashes':{i:frozen['job_hashes'][i] for i in descriptor['ordered_cell_ids']},
            'queue':p.reference(phase,root/'QUEUE.json'),'root_release':p.reference(phase,root/'ROOT_RELEASE.json'),
            'amendment_sha256':release['amendment_sha256'],'attempt_history_sha256':release['attempt_history_sha256'],
            'replication_exception_explicitly_approved':True,'no_automatic_retry_within_attempt':True,
            'old_attempts_preserved_current_status_unknown':True}
        p.write(root/'ATTEMPT_REGISTRATION.json',registration)
        frozen['queues'][descriptor['queue_id']] = {'queue':p.reference(phase,root/'QUEUE.json'),
            'root_release':p.reference(phase,root/'ROOT_RELEASE.json'),
            'registration':p.reference(phase,root/'ATTEMPT_REGISTRATION.json'),
            'ordered_cell_ids':descriptor['ordered_cell_ids'],'attempt_id':descriptor['attempt_id']}
    p.write(control/'FROZEN_AUTHENTICATION.json', frozen)
    print('Frozen all26 exact jobs and four serial queues; zero scientific children started.')


if __name__ == '__main__':
    main()
