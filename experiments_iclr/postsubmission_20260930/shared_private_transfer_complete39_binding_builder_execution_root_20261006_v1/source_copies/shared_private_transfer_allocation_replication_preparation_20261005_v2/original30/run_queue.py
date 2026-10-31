#!/usr/bin/env python3
"""Bounded one-GPU paired queue; observes owned processes, never quality scores."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import pilot_common as c
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import protocol as p


def output_bytes(path):
    """Only this freshly created fit output, without following any symlink."""
    root=Path(path)
    if not root.exists():return 0
    total=0;pending=[root]
    while pending:
        with os.scandir(pending.pop()) as items:
            for item in items:
                if item.is_symlink():raise ValueError('Own fit output contains a symlink; refuse traversal')
                if item.is_dir(follow_symlinks=False):pending.append(Path(item.path))
                elif item.is_file(follow_symlinks=False):total+=item.stat(follow_symlinks=False).st_size
                else:raise ValueError('Unexpected nonregular entry in own fit output')
    return total


def await_resources(s,limits,receipt_path):
    """Fixed finite pre-child resource window; no fit starts or method is skipped."""
    started=time.monotonic();attempts=0
    while True:
        c.physical_host();attempts+=1
        rows=s.query(['--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],limits['telemetry_timeout_seconds'])
        if len(rows)!=1:raise ValueError('Physical singleton GPU changed')
        uuid,free=[part.strip() for part in rows[0].split(',')]
        if uuid!=c.GPU_UUID or not free.isdigit():raise ValueError('Authorized GPU/free-memory telemetry differs')
        status={'UTC':s.now(),'GPU_UUID':uuid,'GPU_free_bytes':int(free)*1024**2,'attempts':attempts,
                'elapsed_seconds':time.monotonic()-started,'scientific_child_started':False}
        c.write(receipt_path,status)
        if status['GPU_free_bytes']>=limits['minimum_fresh_GPU_free_bytes']:return status
        remaining=limits['resource_wait_seconds']-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('Fixed fresh-GPU resource window exhausted before child launch')
        time.sleep(min(limits['poll_interval_seconds'],remaining))


def run_fit(s,root,entry,queue,environment,output):
    cell=entry['cell_id'];job=c.phase_file(entry['job_relative'])
    limits=dict(queue['resource_limits'],external_hard_seconds_per_cell=entry['hard_seconds'])
    await_resources(s,limits,root/'logs'/(cell+'.PREFLIGHT.json'))
    argv=[queue['python_executable'],'-B',str(c.SOURCE/'run.py'),'--job',str(job),'--output',str(output)]
    stdout_path=root/'logs'/(cell+'.stdout.log');stderr_path=root/'logs'/(cell+'.stderr.log')
    started=time.monotonic();reason=None;owner=None;raw_owner=None;cwd=None;signals=[];refusal=None;observed=False
    max_rss=max_gpu=max_logs=max_output=0;incomplete_count=0;last_incomplete=None
    with stdout_path.open('xb') as stdout,stderr_path.open('xb') as stderr:
        process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,cwd=c.REPO,env=environment,
                                 stdout=stdout,stderr=stderr,start_new_session=True)
        try:
            raw_owner=s.identity(process.pid)
            cwd=(Path('/proc')/str(process.pid)/'cwd').resolve(strict=True)
            if raw_owner is None or raw_owner['argv']!=argv or raw_owner['pgid']!=process.pid or raw_owner['sid']!=process.pid or cwd!=c.REPO.resolve():
                reason='Initial fresh owned child/session/argv/cwd could not be verified; no signal authorized'
            else:owner=raw_owner
        except Exception as error:
            reason='Initial identity observation: '+type(error).__name__+': '+str(error)
        c.write(root/'CURRENT_PROCESS.json',{'UTC':s.now(),'cell_id':cell,'identity':owner,'argv':argv,
                'raw_identity_observation':raw_owner,'identity_admitted_for_signals':owner is not None,
                'cwd':str(c.REPO),'observed_cwd':str(cwd) if cwd is not None else None,'limits':limits,'source_manifest_sha256':c.SOURCE_SHA})
        c.write(root/'logs'/(cell+'.CHILD_STARTED.json'),{'UTC':s.now(),'identity':owner,'argv':argv,
                'raw_identity_observation':raw_owner,'identity_admitted_for_signals':owner is not None,
                'cwd':str(c.REPO),'observed_cwd':str(cwd) if cwd is not None else None,'limits':limits})
        try:
            while process.poll() is None:
                if reason is not None:break
                remaining=entry['hard_seconds']-(time.monotonic()-started)
                if remaining<=0:reason='external_hard_wall_bound'
                else:
                    rows=s.owned_tree(owner)
                    incomplete=[row for row in rows if not row.get('observation_complete',True)]
                    if incomplete:
                        incomplete_count+=1;last_incomplete={'UTC':s.now(),'rows':incomplete}
                        if process.poll() is not None:break
                    rss,gpu=s.resources(rows,limits,remaining)
                    max_rss=max(max_rss,rss);max_gpu=max(max_gpu,gpu)
                    max_logs=max(max_logs,stdout_path.stat().st_size+stderr_path.stat().st_size)
                    max_output=max(max_output,output_bytes(output))
                    if max_rss>limits['owned_tree_RSS_cap_bytes']:reason='owned_tree_RSS_cap'
                    elif max_gpu>limits['owned_tree_GPU_memory_cap_bytes']:reason='owned_tree_GPU_memory_cap'
                    elif max_logs>limits['combined_child_log_cap_bytes']:reason='combined_child_log_cap'
                    elif max_output>limits['own_fit_output_cap_bytes']:reason='own_fit_output_cap'
                    elif time.monotonic()-started>=entry['hard_seconds']:reason='external_hard_wall_bound'
                    c.write(root/'CURRENT_RESOURCES.json',{'UTC':s.now(),'cell_id':cell,'identity':owner,
                            'elapsed_seconds':time.monotonic()-started,'owned_RSS_bytes':rss,'owned_GPU_bytes':gpu,
                            'own_output_bytes':max_output,'own_log_bytes':max_logs,'scores_read':False})
                if reason:
                    try:
                        sent=s.kill_owned(process,owner,reason)
                        if sent:signals.append(sent)
                    except s.IncompleteExitObservation as error:refusal=type(error).__name__+': '+str(error)
                    break
                time.sleep(min(limits['poll_interval_seconds'],max(.01,remaining)))
        except (Exception,KeyboardInterrupt) as error:
            reason='required_bounds_telemetry_failure: '+type(error).__name__+': '+str(error)
            try:
                sent=s.kill_owned(process,owner,reason) if owner is not None else None
                if sent:signals.append(sent)
            except Exception as error:refusal=type(error).__name__+': '+str(error)
        finally:
            reason,refusal,observed=s.observe_terminal(process,owner,started,limits,signals,reason,refusal)
    max_logs=max(max_logs,stdout_path.stat().st_size+stderr_path.stat().st_size)
    max_output=max(max_output,output_bytes(output))
    if reason is None and max_logs>limits['combined_child_log_cap_bytes']:reason='combined_child_log_cap_at_exit'
    if reason is None and max_output>limits['own_fit_output_cap_bytes']:reason='own_fit_output_cap_at_exit'
    receipt={'UTC':s.now(),'cell_id':cell,'job_sha256':entry['job_sha256'],
        'exit_code':process.returncode if observed else None,'exit_code_authority':'subprocess.Popen.wait/poll' if observed else None,
        'terminal_wait_observed':observed,'child_identity':owner,'cwd':str(c.REPO),'observed_cwd':str(cwd) if cwd is not None else None,'reason':reason,'signals_sent':signals,
        'raw_identity_observation':raw_owner,'identity_admitted_for_signals':owner is not None,
        'signal_refusal':refusal,'incomplete_exit_observation_samples':incomplete_count,'last_incomplete_exit_observation':last_incomplete,
        'elapsed_seconds':time.monotonic()-started,'max_sampled_owned_RSS_bytes':max_rss,'max_sampled_owned_GPU_bytes':max_gpu,
        'max_sampled_own_output_bytes':max_output,'max_sampled_own_log_bytes':max_logs,
        'stdout_sha256':c.sha(stdout_path),'stderr_sha256':c.sha(stderr_path),'scores_read':False,'attempts':1,'retry':False}
    c.write(root/'logs'/(cell+'.EXIT.json'),receipt)
    return receipt


FAMILY='original30'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    args=parser.parse_args();queue=c.read(args.queue)
    c.physical_host();c.verify_packet(queue['pilot_source_manifest_sha256']);c.verify_training();s=c.supervisor()
    root=(c.PHASE/c.relative_path(queue['execution_directory_relative'])).resolve(strict=True)
    p.require(args.queue.resolve()==root/'QUEUE.json' and not (root/'QUEUE_START.json').exists(),
              'Exact fresh queue only; no implicit retry/resumption')
    release=c.read(root/'ROOT_RELEASE.json');a=p.verify_frozen_protocol(c.PHASE,release)
    descriptor=next(q for q in a['queues'] if q['queue_id']==queue['queue_id'])
    p.require(descriptor['family']==FAMILY and all(queue[k]==v for k,v in descriptor.items()),
              'Exact fixed family/block/attempt/order/bounds required')
    p.require([e['cell_id'] for e in queue['entries']]==descriptor['ordered_cell_ids'], 'Whole block order required')
    p.require(c.sha(root/'ROOT_RELEASE.json')==queue['root_release_sha256']
              and c.sha(root/'COHORT_PLAN.json')==descriptor['cohort_plan']['sha256'], 'Frozen release/plan changed')
    p.canonical_plan(c.PHASE,a,FAMILY)
    if FAMILY=='original30':
        p.require(c.sha(root/'EXTERNAL_ANCHORS.json')==a['external_anchors']['sha256'],'Immutable original anchors changed')
    p.require(queue['retry'] is False and queue['read_scores_or_change_family'] is False
              and queue['TEST_access'] is False and queue['replication_not_setup_fallback_or_resumption'] is True,
              'Explicit counted replication scope required')
    for k in ('python_executable','environment_overrides','resource_limits'):
        p.require(queue[k]==a[k], 'Qualified allocation runtime/resources changed')
    control=c.PHASE/p.relative(queue['execution_control_relative'])
    intent=p.read(root/'LAUNCH_INTENT.json')
    review=p.read(p.binding(c.PHASE,intent['root_job_review']))
    auth=p.read(p.binding(c.PHASE,review['frozen_authentication']))
    p.require(review['all26_generated_jobs_reviewed'] is True and review['full39_collection_contract_reviewed'] is True
              and review['explicit_replication_exception_reviewed'] is True and review['scores_read'] is False
              and review['TEST_access'] is False and review['generated_job_hashes']==auth['job_hashes'],
              'All26 actual jobs must be root reviewed before first fit')
    p.require(auth['scientific_children_started']==0 and auth['queues'][queue['queue_id']]['queue']['sha256']==c.sha(args.queue),
              'Genuine prefit freeze differs')
    registration=p.read(p.binding(c.PHASE,auth['queues'][queue['queue_id']]['registration']))
    p.require(registration['registered_before_first_new_fit'] is True and registration['chosen_cells']==descriptor['ordered_cell_ids'],
              'Actual new physical attempt must be registered before fit')
    for entry in queue['entries']:
        row=next(r for r in a['new_selected_attempts'] if r['cell_id']==entry['cell_id'])
        expected=p.job_from_preview(c.PHASE,a,row)
        p.require(entry['physical_attempt_id']==row['physical_attempt_id']
                  and entry['job_relative']==descriptor['execution_directory_relative']+'/jobs/'+entry['cell_id']+'.json'
                  and entry['output_relative']==descriptor['execution_directory_relative']+'/runs/'+entry['cell_id']
                  and entry['hard_seconds']==row['fit_bounds_seconds']['hard_seconds'], 'Exact attempt/path/bound differs')
        job=c.phase_file(entry['job_relative'])
        p.require(c.read(job)==expected and c.sha(job)==entry['job_sha256']==auth['job_hashes'][entry['cell_id']],
                  'Reviewed generated job differs from disabled exact preview')
    environment=c.check_environment(queue['environment_overrides']);started=time.monotonic();completed=[]
    c.write(root/'QUEUE_START.json',{'UTC':s.now(),'supervisor_identity':s.identity(os.getpid()),'queue_sha256':c.sha(args.queue),
        'physical_fits':len(queue['entries']),'full_scientific_cohort_fits':39,'execution_blocks':[descriptor['block']],
        'queue_id':queue['queue_id'],'attempt_id':descriptor['attempt_id'],'attempt_history_sha256':release['attempt_history_sha256'],
        'TEST_access':False,'comparative_scoring_performed':False,'retry':False,'new_counted_replication':True})
    try:
        for entry in queue['entries']:
            c.verify_packet(queue['pilot_source_manifest_sha256']);c.verify_training();c.physical_host()
            job_path=c.phase_file(entry['job_relative'])
            if c.sha(job_path)!=entry['job_sha256']:raise ValueError('Frozen job changed')
            output=(c.PHASE/c.relative_path(entry['output_relative'])).resolve()
            if output.exists() or not output.parent.is_dir():raise ValueError('No overwrite or implicit retry')
            remaining=queue['queue_hard_seconds']-(time.monotonic()-started)
            if remaining<entry['hard_seconds']+queue['resource_limits']['resource_wait_seconds']:
                raise TimeoutError('Whole frozen queue budget cannot cover next fit; no horizon shortening')
            receipt=run_fit(s,root,entry,queue,environment,output)
            if receipt['exit_code']!=0 or receipt['reason'] is not None or receipt['signals_sent']:
                raise RuntimeError(entry['cell_id']+' operational failure; preserve every artifact and stop, no retry')
            freeze=output/'FREEZE.json'
            if not freeze.is_file() or (output/'FAILURE.json').exists():raise ValueError('Authoritative zero exit lacks complete training freeze')
            # No FREEZE/history/CONFIG parsing. Only ordinary custody hash of source FREEZE.
            completed.append({**receipt,'freeze_relative':str(freeze.relative_to(c.PHASE)),'freeze_sha256':c.sha(freeze)})
            c.write(root/'QUEUE_PROGRESS.json',{'UTC':s.now(),'completed':len(completed),'total':len(queue['entries']),
                'queue_id':queue['queue_id'],'attempt_id':descriptor['attempt_id'],'full_scientific_cohort_fits':39,'scores_read':False})
            print('Completed operationally: '+entry['cell_id'],flush=True)
        c.write(root/'BLOCK_FREEZE.json',{'complete':False,'selected_blocks_complete':True,'execution_blocks':[descriptor['block']],
            'queue_id':queue['queue_id'],'attempt_id':descriptor['attempt_id'],'physical_fits':len(completed),'completed':completed,
            'queue_sha256':c.sha(args.queue),'cohort_plan_sha256':queue['cohort_plan_sha256'],
            'attempt_history_sha256':release['attempt_history_sha256'],'inclusive_seconds':time.monotonic()-started,
            'TEST_access':False,'comparative_scoring_performed':False,'retry':False,'scores_read':False,
            'full39_collection_still_required_before_history_or_scoring':True})
    except (Exception,KeyboardInterrupt) as error:
        c.write(root/'QUEUE_FAILURE.json',{'UTC':s.now(),'error':type(error).__name__+': '+str(error),'completed':completed,
            'inclusive_seconds':time.monotonic()-started,'partial_outputs_preserved':True,'retry':False,'scores_read':False})
        raise


if __name__=='__main__':main()
