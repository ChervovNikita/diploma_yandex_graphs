"""One-command physical-GPU queue dry-run and execution-interface skeleton.

This source generation has no Popen/fit/aggregation dispatch. It proves the
finite queue, restart decisions and missing admissions before a production
successor is reviewed. No model/data/tensor imports or numerical tests.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import sys
from launch_contracts import (HERE,WORKERS,guard,read,receipt,verify,write,require,digest,
                             qualification_blocks,durable_action,legacy,fixed_argv)


def go_blocks(pin,manifest_pin,worker,source_manifest):
    if not pin:return ['human_full_campaign_GO_missing']
    value=read(pin['path'],pin)
    checks=(value.get('schema')=='expanded-finite-human-GO-v1',value.get('user_explicit_GO')is True,
        value.get('manifest_sha256')==manifest_pin['sha256'],value.get('source_manifest_sha256')==source_manifest['sha256'],
        worker in value.get('allowed_workers',[]),bool(value.get('human_instruction_evidence')))
    if not all(checks):return ['human_GO_does_not_bind_this_manifest_source_worker']
    verify(value['human_instruction_evidence'])
    # Root/human custody must authenticate intent; this package creates no GO.
    return []


def inspect(manifest_path,worker,state_path=None,go_path=None):
    guard()  # Current task executes only on allocation; no remote host is contacted.
    manifest_pin=receipt(manifest_path);manifest=read(manifest_path)
    require(manifest['schema']=='expanded-finite-launch-skeleton-v1','Expanded source skeleton manifest required')
    require(worker in WORKERS,'One named physical GPU worker required')
    hosts=read(manifest['host_policy']['path'],manifest['host_policy'])
    admissions=read(manifest['admissions']['path'],manifest['admissions'])
    own=[t for t in manifest['tasks']if t['worker']==worker]
    require(len({t['id']for t in manifest['tasks']})==6730,'Finite unique queue task IDs required')
    if not hosts['workers'][worker]['assignment_enabled']:
        require(not own,'Disabled host must remain unassigned')
        return {'schema':'expanded-physical-GPU-dry-run-v1','status':'DISABLED_UNPROBED_UNASSIGNED','worker':worker,
                'assigned_tasks':0,'host_contacted':False,'children_started':0}
    source_manifest=receipt(HERE/'MANIFEST.json')
    global_blocks=['source_execution_bridge_not_enabled_in_this_generation']
    go_pin=receipt(go_path)if go_path else None
    global_blocks+=go_blocks(go_pin,manifest_pin,worker,source_manifest)
    state={}
    if state_path:
        document=read(state_path)
        require(document['schema']=='expanded-worker-durable-state-v1'and document['manifest_sha256']==manifest_pin['sha256']and document['worker']==worker,
                'State must bind exact immutable queue/worker')
        state=document['tasks'];require(set(state).issubset({t['id']for t in own}),'State contains foreign worker IDs')
    decisions={};counts=Counter();blocks=Counter();queued_commands=[];dependencies=Counter()
    for task in own:
        action=durable_action(task,state);decisions[task['id']]=action;counts[action['action']]+=1
    for task in own:
        action=decisions[task['id']]
        if action['action'].startswith('skip_'):continue
        reasons=list(global_blocks)+qualification_blocks(task,admissions['tasks'].get(task['id'],{}),hosts['workers'][worker])
        if action['action'].startswith('blocked_'):reasons.append(action['action'])
        for dep in task['dependencies']:
            require(dep in decisions,'Dependency assigned to another physical GPU')
            dep_action=decisions[dep]['action']
            if dep_action=='skip_no_selected_state':reasons.append('I4_dependency_has_no_selected_state')
            elif dep_action!='skip_complete':reasons.append('I4_waits_for_own_selected_VALID_dependency')
            elif not state[dep].get('own_selected_VALID_export_receipt'):
                reasons.append('I4_dependency_selected_export_authority_missing')
            else:
                metadata=state[dep]['own_selected_VALID_export_receipt']
                require(Path(metadata['path']).name=='selected_VALID.pt'and len(metadata['sha256'])==64,'Exact own selected VALID receipt required')
        if task['dependencies']:dependencies['export_tasks_waiting_on_four_own_fits']+=1
        reasons=sorted(set(reasons));blocks.update(reasons)
        if not reasons:
            # Typed proposal only; production successor must retain claim/lease/
            # STOP/resume/admission rules described in ADAPTATION_PLAN.md.
            queued_commands.append({'id':task['id'],'argv':fixed_argv(task,task['execution_binding'],manifest['output_root'],action.get('resume_checkpoint'))})
    reviewed=legacy(manifest['legacy_supervisor_sources'])
    require(callable(reviewed.admissible)and callable(reviewed.probe_system)and callable(reviewed.terminal_record),'Reviewed reservation/probe/terminal mechanisms must remain available')
    require(not queued_commands,'This source generation cannot admit or dispatch children')
    require(not any(k in sys.modules for k in ('torch','numpy','ogb','torch_geometric','scipy')),'No numerical source imports')
    return {'schema':'expanded-physical-GPU-dry-run-v1','status':'BLOCKED_SOURCE_PREPARATION_ONLY','worker':worker,'manifest':manifest_pin,
        'maximum_owned_children_per_physical_GPU':2,'currently_compatible_admissions':0,
        'assigned_tasks':len(own),'assigned_physical_fit_intentions':sum(t['kind']=='fit'for t in own),
        'assigned_I4_export_only_tasks':sum(t['kind']=='aggregate_I4_exports'for t in own),
        'protocol_fit_counts':{track:sum(t['kind']=='fit'and t['protocol_track']==track for t in own)for track in('native_replica','long_convergence')},
        'durable_queue_actions':dict(counts),'blocking_reason_task_counts':dict(blocks),'dependency_summary':dict(dependencies),
        'case_prior_weight_coverage_fit_intentions':sum(t.get('cost_case_prior')is not None for t in own),
        'case_priors_are_target_fit_measurements':False,'queue_order':'source-reference-first followed by matched cohort groups;all protocols/controller pins remain explicit',
        'completed_physical_fits_would_be_skipped':True,'interrupted_claims_require_reviewed_same_fit_resume_disposition':True,
        'claim_stealing_or_automatic_retries':False,'live_or_orphan_workers_are_not_relaunched':True,
        'main_M1_and_shared_mean_extra_scoring_tasks':0,'I4_member_prediction_forwards':0,
        'legacy_mechanisms_pinned_and_reusable':['admissible','probe_system','terminal_record','worker lease/permanent claims/shared STOP'],
        'server77_contacted':False,'host_resource_probe_performed':False,'resource_profiles_invented':False,'ETA':None,
        'commands_dispatched':0,'children_started':0,'models_constructed':0,'data_or_prediction_tensors_opened':0,'TEST_accessed':False,
        'required_production_successor':'Implement reviewed dispatch/resume journal adapter after exact profiles/source/provider/lifecycle/GO gates;v1 has no child dispatch path'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['dry-run','run']);p.add_argument('--manifest',required=True);p.add_argument('--worker',choices=WORKERS,required=True);p.add_argument('--state');p.add_argument('--go');p.add_argument('--output')
    a=p.parse_args()
    if a.mode=='run':
        # This is a source skeleton, not an accidentally usable campaign launcher.
        print('{"status":"BLOCKED","reason":"This reviewed source generation is dry-run only;production dispatch successor,qualification and human GO are required","children_started":0}')
        raise SystemExit(2)
    value=inspect(a.manifest,a.worker,a.state,a.go)
    if a.output:write(Path(a.output),value)
    import json;print(json.dumps(value,sort_keys=True))
