"""Finite exact hash comparison; provider/receipt bytes are declared, not equality gates."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    results={};bindings={}
    expected_rows=[(seed,cycle) for seed in [20261005,0,1,2] for cycle in range(60)]
    components={'TRAIN_pairs','native_negative_bank','outer_permutation','endpoint_inner',
                'matched_random_inner','outer_ids_and_endpoints','support_indices'}
    for host in ('singleton','gpu77'):
        root=HERE/host;receipt=json.loads((root/'EXECUTION_RECEIPT.json').read_text())
        result=json.loads((root/'result/RESULT.json').read_text())
        assert receipt['status']=='PASS' and receipt['exit_code']==0 and receipt['terminal_wait_observed'] is True
        assert receipt['reason'] is None and receipt['signals_sent']==[] and receipt['attempts']==1 and receipt['retry'] is False
        assert receipt['owned_PID_absent'] is True and receipt['fits']==0
        assert result['status']=='PASS' and result['fits']==0 and result['VALID_TEST_values_access'] is False
        assert result['models_imported'] is False and result['optimizers_created'] is False
        assert result['program_sha256']=='be04d25383f086d15ade82678eeb15096cf932de1296ee0748f979be8e202871'
        assert [(row['seed'],row['cycle']) for row in result['transcripts']]==expected_rows
        assert all(set(row['component_hashes'])==components and row['episodes_including_tail']==61
                   and row['last_outer_rows']==30 and row['outer_order_rows']==3870
                   and row['global_rng_unchanged'] is True for row in result['transcripts'])
        assert receipt['result_sha256']==sha(root/'result/RESULT.json')
        results[host]=result
        bindings[host]={'result_sha256':sha(root/'result/RESULT.json'),'execution_receipt_sha256':sha(root/'EXECUTION_RECEIPT.json'),
                        'release_sha256':sha(root/'ROOT_RELEASE.json'),'wall_seconds':receipt['wall_seconds'],
                        'source_manifest_sha256':result['source_manifest_sha256'],'hostname':result['hostname'],
                        'python':result['python'],'runtime_versions':result['runtime_versions'],'providers':result['providers'],
                        'declared_PYTHONPATH':result['declared_PYTHONPATH']}
    mismatches=[]
    for a,b in zip(results['singleton']['transcripts'],results['gpu77']['transcripts']):
        for component in sorted(components):
            if a['component_hashes'][component]!=b['component_hashes'][component]:
                mismatches.append({'seed':a['seed'],'cycle':a['cycle'],'component':component,
                    'singleton':a['component_hashes'][component],'gpu77':b['component_hashes'][component]})
        if a['canonical_sha256']!=b['canonical_sha256']:
            mismatches.append({'seed':a['seed'],'cycle':a['cycle'],'component':'full_canonical_raw_engineering_transcript',
                    'singleton':a['canonical_sha256'],'gpu77':b['canonical_sha256']})
    for component in ('TRAIN_pair_hash','inputs','runtime_versions'):
        if results['singleton'][component]!=results['gpu77'][component]:
            mismatches.append({'component':component,'singleton':results['singleton'][component],'gpu77':results['gpu77'][component]})
    comparison={'UTC':datetime.now(timezone.utc).isoformat(),'status':'PASS' if not mismatches else 'FAIL',
                'all_component_hashes_equal':not mismatches,'seeds':[20261005,0,1,2],'cycles':list(range(60)),
                'paired_seed_cycle_draws':240,'compared_component_hashes':1680,'complete_episode_pairs_per_host':14640,
                'canonical_raw_transcripts_equal':not mismatches,'components':sorted(components),'mismatches':mismatches,
                'provider_bindings':bindings,'provider_receipt_binary_equality_required':False,'fit_admission':False,
                'models_executed':False,'optimizers_created':False,'fits':0,'VALID_TEST_values_access':False,
                'engineering_raw_artifacts':'Retained in each host result directory with path/hash/bytes in that host RESULT.json; not duplicated locally.'}
    with (HERE/'COMPARISON.json').open('x') as stream:json.dump(comparison,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps({'status':comparison['status'],'comparison_sha256':sha(HERE/'COMPARISON.json'),
                      'draws':240,'component_hashes':1680,'mismatches':mismatches}))

if __name__=='__main__':main()
