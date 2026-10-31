#!/usr/bin/env python3
"""Allocation-host one-shot launch of one fixed queue after reviewed all26 freeze."""
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import subprocess
import protocol as p


def common(family):
    path = p.HERE/family/'pilot_common.py'
    spec = importlib.util.spec_from_file_location('launch_'+family,path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def reviewed_freeze(phase, review_ref, release, amendment):
    review = p.read(p.binding(phase,review_ref))
    auth = p.read(p.binding(phase,review['frozen_authentication']))
    p.require(review['all26_generated_jobs_reviewed'] is True
              and review['full39_collection_contract_reviewed'] is True
              and review['explicit_replication_exception_reviewed'] is True
              and review['scores_read'] is False and review['TEST_access'] is False
              and review['generated_job_hashes'] == auth['job_hashes']
              and list(auth['queues']) == sorted(q['queue_id'] for q in amendment['queues'])
              and set(auth['job_hashes']) == {r['cell_id'] for r in amendment['new_selected_attempts']},
              'Actual all26 generated job/complete39/replication review required')
    p.require(auth['scientific_children_started'] == 0 and auth['scores_read'] is False
              and auth['physical_fits'] == 26 and auth['logical_scientific_fits'] == 39
              and auth['operational_manifest_sha256'] == release['operational_manifest_sha256']
              and auth['amendment_sha256'] == release['amendment_sha256']
              and auth['attempt_history_sha256'] == release['attempt_history_sha256'],
              'Actual fresh all26 prefit freeze authority differs')
    for descriptor in amendment['queues']:
        frozen = auth['queues'][descriptor['queue_id']]
        queue = p.read(p.binding(phase,frozen['queue']))
        root_release = p.read(p.binding(phase,frozen['root_release']))
        registration = p.read(p.binding(phase,frozen['registration']))
        p.require(root_release == release and frozen['ordered_cell_ids'] == descriptor['ordered_cell_ids']
                  and queue['serial_index'] == amendment['serial_queue_order'].index(descriptor['queue_id'])
                  and [e['cell_id'] for e in queue['entries']] == descriptor['ordered_cell_ids']
                  and registration['registered_before_first_new_fit'] is True
                  and registration['queue'] == frozen['queue'] and registration['root_release'] == frozen['root_release'],
                  'Genuine prefit whole-block registration/queue differs')
        p.require(review['queues'][descriptor['queue_id']] == frozen['queue'], 'Root-reviewed exact queue required')
        for entry in queue['entries']:
            row = next(r for r in amendment['new_selected_attempts'] if r['cell_id'] == entry['cell_id'])
            path = p.file_in(phase,entry['job_relative'])
            p.require(p.read(path) == p.job_from_preview(phase,amendment,row)
                      and p.sha(path) == entry['job_sha256'] == auth['job_hashes'][entry['cell_id']],
                      'All generated jobs must match disabled previews exactly except fit enablement')
    return review,auth


def previous_terminal(phase,a,index,supervisor):
    """Only saved owned terminal metadata; no scores, history or foreign signals."""
    evidence = []
    for descriptor in a['queues'][:index]:
        root = phase/p.relative(descriptor['execution_directory_relative'])
        freeze = p.read(p.file_in(phase,str((root/'BLOCK_FREEZE.json').relative_to(phase))))
        start = p.read(p.file_in(phase,str((root/'QUEUE_START.json').relative_to(phase))))
        launch = p.read(p.file_in(phase,str((root/'LAUNCH_RECEIPT.json').relative_to(phase))))
        p.require(not (root/'QUEUE_FAILURE.json').exists()
                  and freeze['selected_blocks_complete'] is True
                  and freeze['physical_fits'] == len(descriptor['ordered_cell_ids'])
                  and [r['cell_id'] for r in freeze['completed']] == descriptor['ordered_cell_ids']
                  and freeze['queue_sha256'] == start['queue_sha256'] == launch['queue_sha256'] == p.sha(root/'QUEUE.json')
                  and all(r['terminal_wait_observed'] is True and r['exit_code'] == 0 and r['reason'] is None
                          and r['signals_sent'] == [] and r['retry'] is False for r in freeze['completed']),
                  'Every preceding full queue must be owned-terminal complete')
        owner = launch['queue_identity']
        observed = supervisor.identity(owner['PID'])
        p.require(observed is None or observed['start_ticks'] != owner['start_ticks'],
                  'Previous scientific queue is still present; serial release only')
        evidence.append({'queue_id':descriptor['queue_id'],'block_freeze':p.reference(phase,root/'BLOCK_FREEZE.json'),
                         'queue_identity':owner,'saved_start':p.reference(phase,root/'QUEUE_START.json'),
                         'prior_queue_identity_no_longer_present':True})
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--job-review',type=Path,required=True)
    parser.add_argument('--job-review-sha256',required=True)
    parser.add_argument('--queue-id',required=True)
    args = parser.parse_args();phase = p.REMOTE_PHASE
    release = p.read(p.file_in(phase,str(args.release.resolve(strict=True).relative_to(phase))))
    a = p.verify_frozen_protocol(phase,release)
    review_ref = {'path':str(args.job_review.resolve(strict=True).relative_to(phase)), 'sha256':args.job_review_sha256}
    review,auth = reviewed_freeze(phase,review_ref,release,a)
    p.require(args.queue_id in a['serial_queue_order'],'Only the fixed four allocation queues are admitted')
    index = a['serial_queue_order'].index(args.queue_id);descriptor = a['queues'][index]
    c = common(descriptor['family']);c.physical_host();c.verify_training();s = c.supervisor()
    predecessors = previous_terminal(phase,a,index,s)
    root = phase/p.relative(descriptor['execution_directory_relative']);queue_path = root/'QUEUE.json'
    q = p.read(p.binding(phase,auth['queues'][args.queue_id]['queue']))
    p.require(not (root/'QUEUE_START.json').exists() and not (root/'LAUNCH_INTENT.json').exists()
              and not (root/'LAUNCH_RECEIPT.json').exists() and not (root/'QUEUE_FAILURE.json').exists(),
              'One launch only; no automatic retry or resumption')
    p.require(all(not (phase/p.relative(e['output_relative'])).exists() for e in q['entries']), 'Fresh fit outputs required')
    rows = s.query(['--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],q['resource_limits']['telemetry_timeout_seconds'])
    p.require(len(rows) == 1,'Singleton physical GPU required')
    uuid,free = [v.strip() for v in rows[0].split(',')]
    p.require(uuid == p.GPU_UUID and free.isdigit()
              and int(free)*1024**2 >= q['resource_limits']['minimum_fresh_GPU_free_bytes'], 'Fresh reviewed GPU resource minimum required')
    intent = {'schema':'genuine_allocation_replication_launch_intent_v1','UTC':datetime.now(timezone.utc).isoformat(),
        'queue_id':args.queue_id,'attempt_id':descriptor['attempt_id'],'serial_index':index,
        'root_job_review':review_ref,'frozen_authentication':review['frozen_authentication'],
        'queue':p.reference(phase,queue_path),'predecessors':predecessors,'scores_read':False,'TEST_access':False,
        'automatic_retry':False,'all26_job_hashes_reviewed_before_first_fit':True}
    # Exclusive file creation is a permanent claim, including failed launches.
    p.write(root/'LAUNCH_INTENT.json',intent)
    argv = [q['python_executable'],'-B',str(p.HERE/descriptor['family']/'run_queue.py'),'--queue',str(queue_path)]
    environment = c.check_environment(q['environment_overrides'])
    with (root/'logs'/'queue.stdout.log').open('xb') as stdout,(root/'logs'/'queue.stderr.log').open('xb') as stderr:
        process = subprocess.Popen(argv,stdin=subprocess.DEVNULL,cwd=p.REMOTE_REPO,env=environment,
                                   stdout=stdout,stderr=stderr,start_new_session=True)
        owner = s.identity(process.pid)
        cwd = (Path('/proc')/str(process.pid)/'cwd').resolve(strict=True)
        admitted = owner is not None and owner['argv'] == argv and owner['pgid'] == owner['sid'] == process.pid and cwd == p.REMOTE_REPO
        receipt = {'schema':'actual_allocation_replication_launch_receipt_v1','UTC':s.now(),
            'queue_id':args.queue_id,'attempt_id':descriptor['attempt_id'],'hostname':a['hostname'],'GPU_UUID':p.GPU_UUID,
            'queue_identity':owner,'argv':argv,'observed_cwd':str(cwd),'identity_admitted_for_signals':admitted,
            'queue_sha256':p.sha(queue_path),'root_release_sha256':p.sha(root/'ROOT_RELEASE.json'),
            'operational_manifest_sha256':release['operational_manifest_sha256'],'source_manifest_sha256':c.SOURCE_SHA,
            'qualifier_sha256':c.GATE_SHA,'physical_fits':len(q['entries']),'full_scientific_cohort_fits':39,
            'root_job_review':review_ref,'launch_intent':p.reference(phase,root/'LAUNCH_INTENT.json'),
            'detached_launches':1,'attempts':1,'retry':False,'scores_read':False,'TEST_access':False}
        p.write(root/'LAUNCH_RECEIPT.json',receipt)
        p.require(admitted,'Fresh launched queue identity was not admitted; preserve receipt, no signal/retry')
    print('Launched one fixed complete allocation queue: '+args.queue_id)


if __name__ == '__main__':
    main()
