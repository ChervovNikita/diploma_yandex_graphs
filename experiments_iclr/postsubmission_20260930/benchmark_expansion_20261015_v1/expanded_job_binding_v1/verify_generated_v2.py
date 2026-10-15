"""Independent source/JSON verification of one generated descriptor inventory."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = ROOT/'experiments_iclr/postsubmission_20260930'
HERE = PHASE/'benchmark_expansion_20261015_v1/expanded_job_binding_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def load(path, expected=None):
    path=Path(path).resolve()
    assert path.is_relative_to(PHASE) and path.suffix in {'.json','.py'}
    assert not ({'working_data','datasets','processed','raw','results'} & set(path.relative_to(PHASE).parts))
    raw=path.read_bytes()
    if expected:
        assert hashlib.sha256(raw).hexdigest()==expected['sha256']
        if 'bytes' in expected:assert len(raw)==expected['bytes']
    return json.loads(raw)


def main(output_name):
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert Path(__file__).resolve().parent==HERE
    assert output_name.replace('_','').isalnum() and '/' not in output_name
    output=HERE/output_name
    ready=load(output/'READINESS.json')
    for pin in ready['artifacts'].values():load(pin['path'],pin)
    registry=load(ready['registry']['path'],ready['registry'])
    bindings={(r['cell_id'],r['method_id']):r for r in registry['entries']}
    index=load(output/'DESCRIPTOR_INDEX.json')['fits']
    indexed={r['id']:r for r in index}
    assert len(indexed)==len(index)==6460
    old=load(PHASE/'benchmark_expansion_20261015_v1/job_binding_preparation_v1/generated_v2/DESCRIPTOR_INDEX.json')['fits']
    old={r['id']:r for r in old}
    rng_audit=load(PHASE/'benchmark_expansion_20261015_v1/qualification/runtime_v2_source_rng_v1.json',
                   {'sha256':'d8be44fce4ac8d531689f67547e2d4069fd354af8dc571fbf5cffcfe12bbad4c'})
    assert rng_audit['TEST_accessed'] is False and rng_audit['model_calls']==0
    statuses=Counter();blocks=Counter();source_rng=0;aliases=0
    for row in index:
        statuses[row['status']]+=1
        if row['method_id']=='aggregation_mean_globalJ_v1' and row['ancestor']:
            assert row['id'].endswith('/wrapper-boundary_global_half_pool');aliases+=1
        assert row['launch_admitted'] is False
        if row['ancestor']:
            assert row['id'] in old and row['ancestor_contract_hash']==old[row['id']]['contract_hash']
        else:assert row['id'] not in old
        if row['status']=='blocked':
            value=load(row['blocked_record']['path'],row['blocked_record'])
            assert value['source_or_context_required']==row['blocked_reason']
            assert row['descriptor'] is None and row['runtime_job_hash'] is None
            assert value['fallback_or_seed_cache_substitution'] is False
            blocks[row['blocked_reason']]+=1
            continue
        job=load(row['descriptor']['path'],row['descriptor'])
        assert job['id']==row['id'] and digest(job)==row['runtime_job_hash']
        assert digest(job['expanded_contract'])==row['expanded_contract_hash']==job['expanded_contract_hash']
        assert job['scientific_method_id']==row['method_id']
        native=job['expanded_contract']['native_settings']
        assert job['recipe']['native_fixed_config']==native['native_config']
        assert job['seed']==row['initialization_seed']==job['recipe']['_seed']==job['recipe']['seed']
        prep=job['descriptor_preparation']
        assert not prep['scientific_launch_enabled'] and not prep['resource_admission'] and not prep['human_GO_received']
        assert prep['resource_binding'] is None and not prep['data_receipts_refreshed'] and not prep['data_or_tensor_files_opened_by_compiler']
        if row['ancestor']:
            assert job['source_plan_entry']['contract_hash']==old[row['id']]['contract_hash']
            assert job['source_plan_contract_hash']==old[row['id']]['contract_hash']
            assert digest(job['source_plan_entry']['contract'])==old[row['id']]['contract_hash']
        else:
            assert 'source_plan_entry' not in job and 'source_plan_contract_hash' not in job
        if row['method_id']=='native_single':
            assert job['procedure']==job['base_procedure']=='native_single' and not job.get('session_transform')
            assert row['initialization_seed']==row['cohort_seed']+[0,100003,200003,300007][row['body']]
        else:
            binding=bindings[(row['cell_id'],row['method_id'])]
            assert job['procedure']==binding['runtime_procedure'] and job['base_procedure']==binding['base_procedure']
            assert job['callback_sha256']==binding['callback']['sha256']
            assert job['method_settings']==binding['method_settings']
            assert row['body']==(0 if row['ancestor'] else None)
            if row['method_id']=='aggregation_mean_globalJ_v1' and row['ancestor']:
                assert row['id'].endswith('/wrapper-boundary_global_half_pool')
        if row['family']=='link':
            assert not job.get('session_transform') and prep['runtime_link_successor_required']
            assert prep['runtime_selected_serving_qualified'] is False
            if row['cell_id']=='collab-buddy' and row['method_id'].startswith('context_affinity_'):
                assert set(job['method_settings'])=={'manifest_sha256'}
            if 'context_manifest' in job.get('method_settings',{}):assert row['cohort_seed']==0
        if job.get('source_rng_emulation'):
            source_rng+=1
            assert row['family']=='ogb' and row['track']=='native_replica' and row['method_id']=='native_single'
            assert job['source_rng_emulation']['audit_sha256']=='d8be44fce4ac8d531689f67547e2d4069fd354af8dc571fbf5cffcfe12bbad4c'
    assert source_rng==160 and aliases==270 and sum(r['ancestor'] for r in index)==2480
    banks=load(output/'LOGICAL_BANKS.json')['logical_banks'];bank_ids={b['id']:b for b in banks}
    assert len(banks)==len(bank_ids)==5920
    i4_count=0
    for bank in banks:
        assert all(f in indexed for f in bank['fit_ids'])
        if bank['id'].endswith('/native-I4'):
            i4_count+=1;m1=bank_ids[bank['id'].removesuffix('native-I4')+'native-M1']
            assert len(set(bank['fit_ids']))==4 and m1['fit_ids']==bank['fit_ids'][:1]
            own=[indexed[f] for f in bank['fit_ids']]
            assert [r['body'] for r in own]==[0,1,2,3]
            assert [r['initialization_seed']-r['cohort_seed'] for r in own]==[0,100003,200003,300007]
    assert i4_count==270
    controls=load(output/'CONTROL_LINKS.json')['entries'];diagnostics=0
    assert len(controls)==5110
    for row in controls:
        fit=indexed[row['fit_id']]
        for control in row['controls']:
            for target in control['fit_ids']:
                other=indexed[target]
                assert (other['cell_id'],other['split'],other['cohort_seed'],other['track'])==(fit['cell_id'],fit['split'],fit['cohort_seed'],fit['track'])
            if control['kind']=='diagnostic_same_selected_fit':
                diagnostics+=1
                assert control['fit_ids']==[row['fit_id']] and control['extra_fits']==0 and control['selection_changed'] is False
    assert diagnostics==270
    assert not any(k in sys.modules for k in ('torch','torch_geometric','numpy','scipy','ogb'))
    result={'schema':'expanded-generated-independent-metadata-verification-v1','status':'pass',
        'output':str(output),'readiness_sha256':hashlib.sha256((output/'READINESS.json').read_bytes()).hexdigest(),
        'fit_intentions':6460,'statuses':dict(statuses),'blocked_by_reason':dict(blocks),'ancestor_ids_and_contracts_verified':2480,
        'half_pool_aliases_one_physical_fit_each':aliases,'source_OGB_rng_emulation_descriptors_verified':source_rng,
        'native_I4_ownership_and_main_M1_body0_pairs':i4_count,'logical_banks':len(banks),'controls':len(controls),
        'same_selected_fit_mean_diagnostics_zero_extra_fits':diagnostics,'data_or_tensor_files_opened':0,
        'model_callbacks_or_transforms_imported':False,'numerical_libraries_imported':False,'models_constructed':0,
        'scientific_fits_started':0,'TEST_accessed':False,'resource_admission':False,'runtime_lifecycle_qualification':False}
    target=HERE/('VERIFICATION_v2_'+output_name+'.json')
    with target.open('x') as out:json.dump(result,out,indent=2,sort_keys=True);out.write('\n')
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'generated_v1')
