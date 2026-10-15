"""Exact optional reporting metadata binding; never qualification or admission."""
from pathlib import Path
from binding_rules import E, ROOT_LITERAL, pin, json_pin, require

REPORTER_SHA='e7056159aa1e4f88c0b71595a3822edf687bed9b0e1c472f84af88dee7161420'
REPORTER_BYTES=3686
SCORER_SHA='af692038bec60bbb54e9ed45d659289af98a0cb844ab0637c8923ee6ce70b67b'
REPORTER=E/'benchmark_reporting_normalization_rebind_v1'
SCORER=E/'composable_normalization_final_scoring_v2'


def validate_registry_identity(registry,manifest,preparation,scorer):
    marker=registry.get('normalization_rebind',{})
    require(marker.get('schema')=='normalization-reporting-registry-rebind-v1'
            and marker.get('source_manifest')==manifest and marker.get('migration_preparation')==preparation,
            'Exact normalized reporting generation required; old callback-only authority refused')
    require(registry.get('selected_serving_source')==scorer and marker.get('selected_serving_source_bound')is True
            and marker.get('changed_specs')==2890 and marker.get('unchanged_specs')==3570
            and marker.get('native_qualification_granted')is False and marker.get('human_GO')is False,
            'Repaired scorer/source-only reporting declarations differ')


def crosscheck_profiles(registry,candidates,tasks,read):
    """Inspect complete scalar JSON identities, without opening model/data files."""
    by_id={c['id']:c for c in candidates};task_by_id={t['id']:t for t in tasks}
    require(len(candidates)==len(by_id)==len(tasks)==len(task_by_id)==6730
            and set(by_id)==set(task_by_id),'Complete unique candidate/task inventory required')
    require(len(registry['preset_files'])==27
            and len({p['path']for p in registry['preset_files']})==27,'Exactly27 unique reporting profiles required')
    seen=set();fits=exports=predictors=0
    for profile_pin in registry['preset_files']:
        profile=read(profile_pin)
        marker=profile.get('normalization_rebind',{})
        require(marker.get('schema')=='normalization-reporting-profile-rebind-v1'
                and marker.get('source_manifest')==registry['normalization_rebind']['source_manifest']
                and marker.get('reporting_delta')==registry['normalization_rebind']['reporting_delta']
                and marker.get('native_qualification_granted')is False and marker.get('human_GO')is False
                and profile.get('selected_serving_source')==registry['selected_serving_source'],
                'Profile normalized producer/scorer lineage differs')
        preset=profile['registry']['presets']
        require(len(preset)==1 and Path(profile_pin['path']).stem==preset[0]['id'],'Profile preset identity differs')
        for fid,spec in profile['fit_specs'].items():
            require(fid not in seen and fid in by_id,'Duplicate or unknown reporting fit');seen.add(fid)
            candidate,task=by_id[fid],task_by_id[fid]
            require(task['kind']=='fit' and task['cell_id']==preset[0]['id'] and candidate['admitted']is False
                    and candidate['worker']==task['worker']=='allocation/gpu0'
                    and spec['identity']['fit_id']==fid and spec['descriptor']==candidate['job_descriptor']
                    and spec['job_hash']==candidate['runtime_job_hash']==task['runtime_job_hash']
                    and spec['runtime_entrypoint']==candidate['runtime_entrypoint']
                    and spec['output_relative']==task['output_relative'],
                    'Reporting fit descriptor/runtime/output differs from exact candidate')
            fits+=1
        for fid,spec in profile['I4_specs'].items():
            require(fid not in seen and fid in by_id,'Duplicate or unknown reporting I4');seen.add(fid)
            candidate,task=by_id[fid],task_by_id[fid]
            require(task['kind']!='fit' and task['cell_id']==preset[0]['id'] and candidate['admitted']is False
                    and candidate['job_descriptor']is None and candidate['worker']==task['worker']=='allocation/gpu0'
                    and spec['identity']['fit_id']==fid and spec['aggregator_source']==candidate['runtime_entrypoint']
                    and spec['output_relative']==task['output_relative']
                    and spec['component_fit_ids']==task['dependencies'] and len(task['dependencies'])==4
                    and all(dep in profile['fit_specs']for dep in task['dependencies']),
                    'Reporting I4 source or native component dependencies changed')
            exports+=1
        predictors+=len(profile['registry']['expected_predictors'])
    require(seen==set(by_id) and (fits,exports,predictors)==(6460,270,6190),
            'Reporting/candidate complete coverage differs')
    return {'profiles':27,'fit_specs':fits,'I4_specs':exports,'predictor_rows':predictors,
            'descriptor_runtime_output_and_native_dependencies_match':True}


def bind_reporting(registry_pin,candidates,tasks,preparation,prep):
    """The explicit registry pin is supplied by the parent; no discovery/merge."""
    root=Path(ROOT_LITERAL)
    manifest=pin(REPORTER/'MANIFEST.json')
    require(manifest['sha256']==REPORTER_SHA and manifest['bytes']==REPORTER_BYTES,'Frozen reporter manifest differs')
    source=json_pin(manifest,root)
    require(source['schema']=='normalization-reporting-rebind-source-v1','Wrong reporting source schema')
    scorer=pin(SCORER/'MANIFEST.json');require(scorer['sha256']==SCORER_SHA,'Repaired selected scorer required')
    registry=json_pin(registry_pin,root)
    validate_registry_identity(registry,manifest,preparation,scorer)
    require(registry['normalization_rebind']['reporting_delta']==prep['artifacts']['reporting_delta']
            and registry['normalization_rebind']['control_links']==prep['artifacts']['controls'],
            'Reporting delta/controls do not bind actual normalization generation')
    checked=crosscheck_profiles(registry,candidates,tasks,lambda value:json_pin(value,root))
    hook={'bridge':pin(REPORTER/'backend_bridge.py'),'entrypoint':pin(REPORTER/'runtime_entry.py'),
          'registry':registry_pin,'manifest':manifest}
    return hook,checked,[registry_pin,*registry['preset_files']]
