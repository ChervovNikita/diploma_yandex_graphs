"""Actual normalization metadata -> one immutable, unadmitted allocation projection."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
import bootstrap
from binding_rules import (E, PHASE_LITERAL, ROOT_LITERAL, digest, pin, json_pin, require,
                           normalized_execution_binding, verify_composed_job)
from reporting_binding import bind_reporting

HERE=Path(__file__).resolve().parent
ROOT=Path(ROOT_LITERAL)
PREPARATION_SHA='9d4a9c494b8b6e83d2aeaf4480c30e9628e5ad23c2adb7695bf9bc0c7134b6c7'


def load(path,name):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


def write(path,value):
    with Path(path).open('x')as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    return pin(path)


def project_task(task,worker,host,original):
    if original is not None and original.get('normalization_composition')is not None:
        require(worker=='allocation/gpu0'and host['host']=='allocation'and host['project_root']==ROOT_LITERAL
                and host['phase_relative']==PHASE_LITERAL,'No composed77 projection is authorized')
        require(digest(original)==task['runtime_job_hash'],'Canonical composed job changed')
        value=copy.deepcopy(task);value.update(worker=worker,canonical_runtime_job_hash=digest(original),
            runtime_binding=normalized_execution_binding(original,ROOT,PHASE_LITERAL))
        return value
    parent=load(E/'portable_dispatcher_v3/prepare_projection.py','_normalization_dispatch_parent_projector')
    return parent.project_task(task,worker,host,original)


def canonical_tasks(parent_release,delta,candidate_pin):
    require(len(parent_release['tasks'])==6730 and len(delta['changed_tasks'])==2890,'Exact finite parent/migration required')
    changed={row['parent_task']['id']:row for row in delta['changed_tasks']}
    require(len(changed)==2890,'Duplicate normalization replacement')
    old={task['id']:task for task in parent_release['tasks']}
    require(set(changed).issubset(old),'Unknown parent task in replacement')
    result=[]
    for task in parent_release['tasks']:
        row=changed.get(task['id'])
        if row is not None:
            require(row['parent_task']==task and task['kind']=='fit','Changed task does not preserve its exact parent')
            value=copy.deepcopy(row['successor_task'])
        else:value=copy.deepcopy(task)
        value.pop('binding',None);value.pop('execution_status',None)
        value['execution_candidates']=copy.deepcopy(candidate_pin)
        result.append(value)
    require(len({t['id']for t in result})==6730,'Task IDs collide')
    require(len({t['output_relative']for t in result})==6730,'Output paths collide')
    require(sum(t['kind']=='fit'for t in result)==6460,'Fit count changed')
    return result


def verify_canonical_plan(release,skeleton):
    """Future production authority binds this exact derived canonical inventory."""
    inputs=json.loads((HERE/'INPUTS.json').read_text())
    require(skeleton['schema']=='normalization-portable-canonical-plan-v1'
            and skeleton['normalization_preparation']==inputs['normalization_preparation'],
            'Wrong normalized canonical authority')
    prep=json_pin(inputs['normalization_preparation'],ROOT)
    require(prep['source_manifest']==inputs['migration_source_manifest'] and prep['counts']['changed']==2890,
            'Exact migration producer required')
    parent=json_pin(prep['parent_inputs']['portable_release'],ROOT)
    delta=json_pin(prep['artifacts']['dispatcher_delta'],ROOT)
    require(skeleton['parent_release']==prep['parent_inputs']['portable_release']
            and skeleton['tasks']==canonical_tasks(parent,delta,release['normalization_successor']['candidates']),
            'Canonical inventory differs from exact normalization replacements')
    return True


def source_closure(parent_release,source_pin,prep,prep_pin):
    rows={row['path']:copy.deepcopy(row)for row in parent_release['source_files']}
    def add(path):
        value=pin(path)
        if value['path']in rows:
            require(all(rows[value['path']][k]==value[k]for k in('path','bytes','sha256')if k in rows[value['path']]),
                    'Conflicting source identity')
        else:rows[value['path']]=value
    for folder in (HERE,E/'composable_normalization_runtime_v1',E/'composable_normalization_v2',
                   E/'benchmark_reporting_normalization_rebind_v1',E/'composable_normalization_final_scoring_v2'):
        add(folder/'MANIFEST.json')
        for row in json.loads((folder/'MANIFEST.json').read_text())['files']:
            path=(folder/row['path']).resolve();require(path.is_relative_to(folder),'Source member escapes package')
            now=pin(path);require(all(now[k]==row[k]for k in('bytes','sha256')),'Frozen source member changed')
            add(path)
    # Retain exact dependencies without another native evidence grid.
    for folder in ('composable_normalization_runtime_v1','benchmark_reporting_normalization_rebind_v1',
                   'composable_normalization_final_scoring_v2'):
        for row in json.loads((E/folder/'SOURCE_BINDINGS.json').read_text())['files']:
            path=E.parent/row['path_relative_to_phase']
            require(path.resolve().is_relative_to(ROOT),'Frozen source dependency escapes project')
            actual=pin(path)
            require(all(actual[k]==row[k]for k in('bytes','sha256')),'Frozen producer/reporting/scorer dependency changed')
            add(path)
    add(E/'private_bn_state_v3/banks.py');add(E/'private_bn_state_v3/MANIFEST.json')
    add(E/'expanded_study_design_v1/PROPOSED_ROSTER.json')
    add(Path(prep_pin['path']));add(Path(prep['source_manifest']['path']))
    return list(rows.values())


def prepare(output,source_pin,reporting_index_pin=None):
    inputs=json.loads((HERE/'INPUTS.json').read_text());prep_pin=inputs['normalization_preparation']
    require(prep_pin['sha256']==PREPARATION_SHA,'Use the actual root-observed migration generation')
    prep=json_pin(prep_pin,ROOT)
    require(prep['status']=='metadata_generated_not_admitted'and prep['source_manifest']==inputs['migration_source_manifest']
            and prep['counts']['fits']==6460 and prep['counts']['changed']==2890 and prep['counts']['unchanged']==3570
            and prep['counts']['exports']==270 and prep['full_campaign_GO']is False,'Wrong actual normalization generation')
    parent=json_pin(prep['parent_inputs']['portable_release'],ROOT)
    old_candidates=json_pin(prep['parent_inputs']['portable_candidates'],ROOT)
    delta=json_pin(prep['artifacts']['dispatcher_delta'],ROOT)
    index=json_pin(prep['artifacts']['index'],ROOT)
    require(parent['human_GO']is False and parent['all_tasks_qualified']is False and parent['host']=='allocation',
            'Do not transfer an execution admission')
    parent_owner=json_pin(parent['ownership'],ROOT)
    require(all(t['worker']=='allocation/gpu0'for t in parent['tasks']),'Keep current allocation-only group placement')
    require(delta['parent_release']==prep['parent_inputs']['portable_release']
            and delta['parent_candidates']==prep['parent_inputs']['portable_candidates'],'Dispatch delta lineage differs')
    changed={row['parent_task']['id']:row for row in delta['changed_tasks']}
    require(len(changed)==2890 and len(old_candidates)==6730,'Replacement/candidate count differs')
    candidates=[]
    for old in old_candidates:
        row=changed.get(old['id'])
        if row is not None:
            require(row['parent_candidate']==old,'Changed candidate parent differs')
            candidates.append(copy.deepcopy(row['successor_candidate']))
        else:candidates.append(copy.deepcopy(old))
    require(all(c['admitted']is False and c['worker']=='allocation/gpu0'for c in candidates),'Candidates cannot carry admission or77 ownership')
    expected={r['id']:r for r in index['fits']}
    closure=source_closure(parent,source_pin,prep,prep_pin)
    files={row['path']:row['sha256']for row in closure}
    for row in changed.values():
        candidate=row['successor_candidate'];job=json_pin(candidate['job_descriptor'],ROOT)
        require(job['id']==candidate['id']and digest(job)==candidate['runtime_job_hash']
                and candidate['job_descriptor']==expected[job['id']]['descriptor'],'Exact generated composed descriptor required')
        require(candidate['runtime_entrypoint']==pin(E/'composable_normalization_runtime_v1/runtime/run_fit.py'),
                'Changed candidates must use the genuine composed producer')
        verify_composed_job(job,ROOT,files)
    exports=[task for task in parent['tasks']if task['kind']!='fit']
    require(len(exports)==270 and all(all(dep not in changed for dep in task['dependencies'])for task in exports),
            'I4 dependencies must remain native and unchanged')
    require(delta['unchanged_I4_exports']==exports
            and set(delta['unchanged_task_ids'])=={t['id']for t in parent['tasks']if t['id']not in changed},
            'Exact unchanged task/export custody differs')
    output.mkdir();folder=output/'allocation';folder.mkdir()
    try:
        candidates_pin=write(folder/'CANDIDATES.json',candidates)
        canonical=canonical_tasks(parent,delta,candidates_pin)
        reporting=reporting_checks=None
        if reporting_index_pin is not None:
            require(reporting_index_pin==inputs['actual_reporting_index'],'Only the exact root-observed reporting INDEX is supported')
            reporting,reporting_checks,reporting_files=bind_reporting(reporting_index_pin,candidates,canonical,prep_pin,prep)
            have={p['path']:p for p in closure}
            for value in reporting_files:
                require(value['path']not in have or have[value['path']]==value,'Reporting closure conflict')
                have[value['path']]=value
            closure=list(have.values());files={p['path']:p['sha256']for p in closure}
        old_assignment={row['id']:row for row in parent_owner['assignments']}
        new_parent_ids={row['successor_task']['id']:fid for fid,row in changed.items()}
        assignments=[]
        for task in canonical:
            parent_id=new_parent_ids.get(task['id'],task['id']);prior=old_assignment[parent_id]
            require(prior['worker']==task['worker']=='allocation/gpu0'and prior['cohort']==task['group_id'],
                    'Cohort/worker placement changed')
            assignments.append({'id':task['id'],'worker':task['worker'],'canonical_runtime_job_hash':task.get('runtime_job_hash'),
                'science_contract_hash':task['science_contract_hash'],'dependencies':copy.deepcopy(task['dependencies']),
                'output_relative':task['output_relative'],'cohort':task['group_id']})
        owner=copy.deepcopy(parent_owner);owner['assignments']=assignments
        ownership=load(E/'portable_dispatcher_v3/ownership.py','_normalization_dispatch_original_ownership')
        ownership.validate(owner,canonical)
        owner_pin=write(output/'OWNERSHIP.json',owner)
        canonical_pin=write(folder/'CANONICAL_PLAN.json',{'schema':'normalization-portable-canonical-plan-v1',
            'normalization_preparation':prep_pin,'parent_release':prep['parent_inputs']['portable_release'],
            'legacy_supervisor_sources':copy.deepcopy(parent['legacy_supervisor_sources']),'tasks':canonical})
        projected=[]
        for task in canonical:
            original=json_pin(task['source_descriptor'],ROOT)if task['kind']=='fit'and task['id']in new_parent_ids else None
            if task['kind']=='fit'and original is None:
                # This unchanged source/projected descriptor has already passed
                # the pinned complete portable3 audit; no new relocation occurs.
                value=copy.deepcopy(task)
            else:value=project_task(task,task['worker'],owner['hosts']['allocation'],original)
            value.update(binding=None,execution_status='blocked_normalization_release_not_admitted')
            projected.append(value)
        release=copy.deepcopy(parent)
        release.update(canonical_skeleton=canonical_pin,ownership=owner_pin,tasks=projected,source_files=closure,
            source_bundle_sha256=digest(files),all_tasks_qualified=False,human_GO=False,TEST_access=False)
        release['normalization_successor']={'preparation':prep_pin,'candidates':candidates_pin,
            'control_links':prep['artifacts']['controls'],'reporting_delta':prep['artifacts']['reporting_delta'],
            'source_manifest':source_pin,'actual_admissions':0,'claims_created':0,
            'reporting_and_scoring_rebind_pending':reporting is None}
        release.pop('reporting',None)
        if reporting is not None:release['reporting']=reporting
        verify_canonical_plan(release,json_pin(canonical_pin,ROOT))
        release_pin=write(folder/'RELEASE_UNADMITTED.json',release)
        result={'schema':'normalization-portable-dispatch-preparation-v1','status':'metadata_prepared_not_admitted',
            'source_manifest':source_pin,'normalization_preparation':prep_pin,'parent_release':prep['parent_inputs']['portable_release'],
            'ownership':owner_pin,'candidates':candidates_pin,'canonical_plan':canonical_pin,'release':release_pin,
            'counts':{'tasks':6730,'fits':6460,'composed_candidates':2890,'unchanged_fit_candidates':3570,'I4_exports':270},
            'planned_maximum_children_per_physical_GPU':2,'eventual_three_GPU_target':6,
            'current_assignment':'allocation/gpu0_only','admitted_concurrency':0,'measured_pair_qualification':False,
            'server77_enabled':False,'claims_created':0,'native_jobs_started':0,'TEST_payload_accessed':False,'human_GO':False,
            'remaining_blocks':prep['remaining_dependencies'],
            'reporting':reporting,'reporting_candidate_crosscheck':reporting_checks,
            'reporting_hook_pending':reporting is None}
        require('torch'not in sys.modules,'Metadata preparation must not import Torch')
        write(output/'PREPARATION.json',result);return result
    except BaseException as error:
        write(output/'FAILED.json',{'status':'failed','error':type(error).__name__,'message':str(error),
            'traceback':traceback.format_exc(limit=12),'complete_generation':False,'native_work':0,'claims_created':0})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-manifest-sha256',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--reporting-index');parser.add_argument('--reporting-index-sha256');args=parser.parse_args()
    # Reuse the already reviewed allocation route guard and source verifier.
    migration=load(E/'normalization_campaign_migration_v1/generate.py','_normalization_dispatch_migration_guard')
    migration.guard()
    output=Path(args.output).resolve();require(output.parent==HERE and not output.exists(),'Fresh direct-child output required')
    source_pin=pin(HERE/'MANIFEST.json');require(source_pin['sha256']==args.source_manifest_sha256,'Exact reviewed source required')
    manifest=json_pin(source_pin,ROOT)
    require(manifest['schema']=='normalization-portable-dispatch-source-v1','Wrong dispatcher source schema')
    for row in manifest['files']:
        actual=pin(HERE/row['path']);require(all(actual[k]==row[k]for k in('bytes','sha256')),'Source member changed')
    for row in json.loads((HERE/'SOURCE_BINDINGS.json').read_text())['files']:
        actual=pin(E.parent/row['path_relative_to_phase']);require(all(actual[k]==row[k]for k in('bytes','sha256')),'Frozen parent source changed')
    require(bool(args.reporting_index)==bool(args.reporting_index_sha256),'Supply both explicit reporting INDEX path and SHA, or neither')
    registry_pin=None
    if args.reporting_index:
        registry_path=Path(args.reporting_index).resolve()
        require(registry_path.is_relative_to(ROOT)and registry_path.suffix=='.json','Project-confined reporting metadata required')
        registry_pin=pin(registry_path)
        require(registry_pin['sha256']==args.reporting_index_sha256,'Explicit reporting INDEX SHA differs')
    started=time.perf_counter();result=prepare(output,source_pin,registry_pin)
    print(json.dumps({'preparation':pin(output/'PREPARATION.json'),'counts':result['counts'],
        'elapsed_seconds':time.perf_counter()-started,'native_work':0,'claims_created':0,'human_GO':False}))


if __name__=='__main__':main()
