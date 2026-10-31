"""Metadata supplement for source geometry, retained failures and startup scope."""
import hashlib
import json
from pathlib import Path
from source_geometry import extract

HERE=Path(__file__).resolve().parent;PHASE=HERE.parent


def pin(path):
    raw=path.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def main():
    metadata=PHASE/'training_pipeline_v2_20261012/preparation'
    names=('nsd_resource_results_root_20261012_v1','corrected_nsd_resource_results_root_20261012_v1',
           'bayesian_bsnn_resource_results_root_20261012_v1','bayesian_bsnn_resource_results_root_20261012_v2')
    rows=[];failures=[];bindings=[]
    for name in names:
        path=metadata/name/'CLOSURE.json';value=json.loads(path.read_text());assert value['all_children_closed'] is True
        assert value.get('scientific_fits',value.get('fits',0))==0
        bindings.append({'remote_path':name+'/CLOSURE.json','local_source':str(path.relative_to(PHASE)),**pin(path)})
        for row in value['attempts']:
            entry={'case_id':row['case_id'],'source_closure':name,'result':row.get('result'),
                   'exit_code':row['exit_code'],'child_closed':row['child_absent']}
            if row['exit_code']:
                entry['failure_retained']=True
                entry['failure_scope']='native M1 repeated sparse CUDA logit bit equality overstrict; source prefix successor qualified' if name.startswith('bayesian') else 'original callback source gradient/ownership issue; corrected source successor independently qualified'
                failures.append(entry);continue
            resource=row['resource']
            entry.update(VRAM_reserved_bytes=resource.get('CUDA_peak_reserved_bytes',resource.get('cuda_peak_reserved_bytes')),
                         complete_state_serialized_proxy_bytes=resource.get('complete_checkpoint_bytes',resource.get('random_complete_engine_restore',{}).get('bytes')),
                         resource_scope='zero-update actual full-input source callback, source RNG/state and serializer; no optimizer-kernel/convergence/ETA qualification',
                         scientific_fits=0,optimizer_updates=0,scores=0)
            if name=='nsd_resource_results_root_20261012_v1':
                entry.update(current_factory_equality_proved=False,
                             source_scope='earlier NSD source/adjoint and enginev12 receipt; disclose as prior source proxy, never automatic current equality')
                if 'first-action' in row['case_id']:
                    entry['resource_scope']='first-action retained common-V capacity only; no complete TRAIN callback or full serialized state'
                if 'active_untrained_capacity' in row['case_id']:
                    entry['resource_scope']='untrained random capacity prototype; not a scientifically paid warm/active ancestor'
            else:
                entry.update(current_source_callback_qualified=True,
                             all_other_masks_seeds_future_learned_states_and_77_devices_automatically_qualified=False)
            rows.append(entry)
    catalog_path=PHASE/'checkpoint_retention_proposal_20261012_v1/current_v21_resource_catalog_20261012_v1/CATALOG.json'
    catalog=json.loads(catalog_path.read_text());nested=[]
    for row in catalog['root_supervised_engineering_queue']:
        c=row['source_run']['config']
        if isinstance(c.get('native'),dict):
            nested.append({'study_id':row['study_id'],'method':row['method'],'run_id':row['source_run_id'],
                           'old_generic_shape':row['shape'],'actual_representative_declared_geometry':extract(c),
                           'representative_is_not_proved_maximum_across_final_master':True})
    result={'schema':'finite-source-resource-startup-supplement-v1','closure_bindings':bindings,
            'source_scoped_passed_receipts':rows,'all_prior_failures':failures,
            'earlier_NSD_passing_receipts_count':sum(r['source_closure']=='nsd_resource_results_root_20261012_v1' for r in rows),
            'corrected_current_NSD_passing_receipts_count':sum(r['source_closure']=='corrected_nsd_resource_results_root_20261012_v1' for r in rows),
            'current_Bayesian_fixed_family_passing_receipts_count':sum(r['source_closure'].startswith('bayesian') for r in rows),
            'nested_native_representative_geometry_corrections':nested,
            'catalogue_basis':{'path':str(catalog_path.relative_to(PHASE)),**pin(catalog_path),
                               'plan_hash':catalog['summary']['plan_hash'],'not_final_master':True},
            'source_families_unknown_remain_unknown':True,'no_per_fit_independent_resource_case_required_if_source_family_maximum_full_state_envelope_proved':True,
            'physical_77_runtime_disk_quota_RAM_VRAM_admission_pending':True,
            'whole_master_storage_and_all_source_geometry_coverage_proved':False,'scientific_fits':0,'GO':False}
    (HERE/'RESOURCE_SCOPE_SUPPLEMENT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'older_NSD_proxies':result['earlier_NSD_passing_receipts_count'],
                     'current_NSD_cases':result['corrected_current_NSD_passing_receipts_count'],
                     'current_Bayesian_cases':result['current_Bayesian_fixed_family_passing_receipts_count'],
                     'preserved_failures':len(failures),'nested_native_representatives':len(nested),'GO':False}))

if __name__=='__main__':main()
