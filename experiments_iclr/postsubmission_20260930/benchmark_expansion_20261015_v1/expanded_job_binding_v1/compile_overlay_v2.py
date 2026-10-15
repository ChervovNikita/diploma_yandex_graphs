"""Compile a compact source successor; reuse immutable parent fit descriptors."""
from __future__ import annotations
import copy
from collections import Counter
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=ROOT/'experiments_iclr/postsubmission_20260930'
HERE=PHASE/'benchmark_expansion_20261015_v1/expanded_job_binding_v1'
PARENT_READY_SHA='807022215b2dd05bf4cdef73b19a4cae851f877beeea121b2091501dfc851e8f'
OVERLAY_SHA='3c2739cbd9e060b2eb979de717acea9d4aaf9f1ee7aedaac969e90fc2735b05a'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def source_pin(path,expected=None):
    p=Path(path).resolve();assert p.is_relative_to(PHASE) and p.suffix in {'.py','.json','.md','.txt','.xml','.html'}
    assert not ({'working_data','datasets','processed','raw','results'} & set(p.relative_to(PHASE).parts))
    raw=p.read_bytes();pin={'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    if expected:
        for k in ('bytes','sha256'):
            if k in expected:assert pin[k]==expected[k],str(p)
    return pin


def load(path,expected=None):
    source_pin(path,expected)
    return json.loads(Path(path).read_text())


def write(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode()
    with path.open('xb') as out:out.write(raw)
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def main():
    # Guard is literal and precedes all project reads/imports/writes.
    import socket,subprocess
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert Path(__file__).resolve().parent==HERE
    parent=HERE/'generated_v1';output=HERE/'generated_v2';assert not output.exists()
    parent_ready_pin=source_pin(parent/'READINESS.json',{'sha256':PARENT_READY_SHA})
    ready=load(parent_ready_pin['path'])
    verified=load(HERE/'VERIFICATION_v2_generated_v1.json')
    assert verified['status']=='pass' and verified['readiness_sha256']==PARENT_READY_SHA
    overlay_pin=source_pin(HERE/'OVERLAY_v2.json',{'sha256':OVERLAY_SHA})
    overlay=load(overlay_pin['path']);assert overlay['schema']=='expanded-fit-source-overlay-v2' and overlay['runnable'] is False
    assert load(PHASE/overlay['handoff']['path_relative_to_phase'],overlay['handoff'])['current_source_supported_seeds']==[0]
    source_receipts=[source_pin(PHASE/p['path_relative_to_phase'],p) for p in overlay['source_files']]
    original_registry=load(PHASE/overlay['parent_registry']['path_relative_to_phase'],overlay['parent_registry'])
    original_sources=load(ready['artifacts']['sources']['path'],ready['artifacts']['sources'])
    for pin in original_sources['source_and_context_metadata']:source_pin(pin['path'],pin)
    bindings={(r['cell_id'],r['method_id']):r for r in overlay['entries']}
    assert len(bindings)==4 and all(r['supported_seeds']==[0] for r in bindings.values())
    import ast
    for b in bindings.values():
        p=PHASE/b['callback']['path_relative_to_phase'];source_pin(p,b['callback'])
        assert b['callback']['function'] in {n.name for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
        assert not b.get('session_transform') and set(b['method_settings'])=={'manifest_sha256','context_manifest','context_manifest_sha256'}
        source_receipts.append(source_pin(b['context_manifest_receipt']['path'],b['context_manifest_receipt']))
    index=load(ready['artifacts']['index']['path'],ready['artifacts']['index'])['fits']
    assert len(index)==len({r['id'] for r in index})==6460
    indexed={r['id']:r for r in index};new_rows=[];changed=[];new_descriptors=[];new_blocked=[]
    runtime_folder=PHASE/original_registry['runtime_package']
    sys.path.insert(0,str(runtime_folder))
    spec=importlib.util.spec_from_file_location('_overlay_metadata_validator',runtime_folder/'run_fit.py')
    runtime=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runtime;spec.loader.exec_module(runtime)
    assert not any(k in sys.modules for k in ('torch','numpy','scipy','torch_geometric','ogb'))
    output.mkdir();(output/'jobs').mkdir();(output/'blocked').mkdir()
    for old in index:
        key=(old['cell_id'],old['method_id']);binding=bindings.get(key)
        if binding is None:
            # Authenticate inherited descriptor/blocked bytes, without writing copies.
            pin=old['descriptor'] or old['blocked_record'];source_pin(pin['path'],pin)
            new_rows.append(copy.deepcopy(old));continue
        assert old['status']=='blocked' and old['cell_id']=='collab-buddy' and not old['ancestor']
        prior=load(old['blocked_record']['path'],old['blocked_record'])
        contract=copy.deepcopy(prior['expanded_contract'])
        assert digest(contract)==old['expanded_contract_hash']
        contract.update(source_registry=overlay_pin,binding_entry_sha256=digest(binding))
        row=copy.deepcopy(old);row['expanded_contract_hash']=digest(contract)
        row['previous_generation_record']=old['blocked_record']
        filename=hashlib.sha256(old['id'].encode()).hexdigest()+'.json'
        if old['cohort_seed']!=0:
            reason=binding['unsupported_seed_reason']
            row['blocked_reason']=reason
            block={'schema':'expanded-blocked-fit-intention-v2','index':row,'expanded_contract':contract,'binding':binding,
                   'source_or_context_required':reason,'source_now_frozen':True,'native_hyperparameters_changed':False,
                   'fallback_or_seed_cache_substitution':False,'scientific_launch_enabled':False,
                   'parent_intention_record':old['blocked_record']}
            row['blocked_record']=write(output/'blocked'/filename,block);new_blocked.append(row)
        else:
            template_id=old['id'].rsplit('/wrapper-',1)[0]+'/wrapper-gnnm_boundary_native'
            template_row=indexed[template_id];template=load(template_row['descriptor']['path'],template_row['descriptor'])
            assert digest(template)==template_row['runtime_job_hash']
            job=copy.deepcopy(template)
            job.pop('source_plan_entry',None);job.pop('source_plan_contract_hash',None);job.pop('session_transform',None)
            job.update(id=old['id'],body=None,scientific_method_id=old['method_id'],
                       procedure=binding['runtime_procedure'],base_procedure=binding['base_procedure'],
                       callback_module=str(PHASE/binding['callback']['path_relative_to_phase']),callback_sha256=binding['callback']['sha256'],
                       method_settings=copy.deepcopy(binding['method_settings']),expanded_contract=contract,expanded_contract_hash=digest(contract))
            job['descriptor_preparation'].update(registry=overlay_pin,ancestor_descriptor=None,ancestor_contract_preserved=False,
                previous_generation_intention=old['blocked_record'],predecessor_template_descriptor=template_row['descriptor'],
                runtime_binding_migration={'from_previous_generation_status':'blocked_missing_source','to_runtime_procedure':job['procedure'],
                                           'scientific_method_id':old['method_id'],'source_only':True},
                native_CPU_gate='pending_at_handoff;not_qualification_by_compiler',runtime_link_successor_required=True,
                pending=copy.deepcopy(binding['pending']))
            assert job['seed']==job['cohort_seed']==0 and job['split']==0
            assert job['recipe']['native_fixed_config']==contract['native_settings']['native_config']
            assert job['recipe']['architecture'].upper()=='BUDDY'
            assert job['recipe']['official_split_bindings'][str(job['split'])]==contract['role_bindings']
            assert not job['descriptor_preparation']['resource_admission'] and not job['descriptor_preparation']['scientific_launch_enabled']
            runtime.validate_job(job,ROOT)
            row.pop('blocked_reason',None);row.pop('blocked_record',None)
            row.update(status='descriptor_prepared_not_admitted',descriptor=write(output/'jobs'/filename,job),runtime_job_hash=digest(job),
                       runtime_metadata_validation='pass_metadata_only')
            new_descriptors.append(row)
        new_rows.append(row);changed.append(row)
    assert len(new_rows)==6460 and len(changed)==40 and len(new_descriptors)==4 and len(new_blocked)==36
    new_index={r['id']:r for r in new_rows};assert new_index.keys()==indexed.keys()
    assert sum(r['ancestor'] for r in new_rows)==2480
    statuses=Counter(r['status'] for r in new_rows)
    blocks=Counter(r['blocked_reason'] for r in new_rows if r['status']=='blocked')
    assert statuses=={'descriptor_prepared_not_admitted':6370,'blocked':90}
    banks=load(ready['artifacts']['banks']['path'],ready['artifacts']['banks'])
    for b in banks['logical_banks']:
        if 'fit_status' in b and len(b['fit_ids'])==1:b['fit_status']=new_index[b['fit_ids'][0]]['status']
    control=load(ready['artifacts']['controls']['path'],ready['artifacts']['controls'])
    for r in control['entries']:
        for c in r['controls']:
            if 'fit_status' in c and len(c['fit_ids'])==1:c['fit_status']=new_index[c['fit_ids'][0]]['status']
    assert len(banks['logical_banks'])==5920 and len(control['entries'])==5110
    artifacts=copy.deepcopy(ready['artifacts'])
    artifacts['index']=write(output/'DESCRIPTOR_INDEX.json',{'schema':'expanded-fit-overlay-index-v2','parent_index':ready['artifacts']['index'],'fits':new_rows})
    artifacts['blocked']=write(output/'BLOCKED_FITS.json',{'schema':'expanded-blocked-fit-index-v2','fits':[r for r in new_rows if r['status']=='blocked']})
    artifacts['banks']=write(output/'LOGICAL_BANKS.json',banks)
    artifacts['controls']=write(output/'CONTROL_LINKS.json',control)
    artifacts['sources']=write(output/'SOURCE_RECEIPTS.json',{'overlay':overlay_pin,'parent_sources':ready['artifacts']['sources'],
        'parent_verification':source_pin(HERE/'VERIFICATION_v2_generated_v1.json'),'source_and_context_metadata':source_receipts,
        'data_receipts_refreshed':False,'model_source_imports':False})
    artifacts['delta']=write(output/'DELTA.json',{'schema':'expanded-fit-source-overlay-delta-v2','parent_readiness':parent_ready_pin,
        'overlay':overlay_pin,'same_fit_intention_ids':6460,'new_concrete_descriptors':new_descriptors,'reclassified_context_blocks':new_blocked,
        'inherited_unchanged_index_rows':6420,'inherited_physical_fit_intentions':6456,'inherited_concrete_descriptor_files':6366,
        'new_concrete_descriptor_files':4,'new_blocked_metadata_records':36,'copied_ancestor_descriptor_files':0,
        'scientific_or_native_controller_changes':False,'resource_admission':False,'scientific_fits_started':0})
    result=copy.deepcopy(ready)
    result.update(schema='expanded-6460-source-overlay-readiness-v2',UTC=datetime.now(timezone.utc).isoformat(),
        remote_output=str(output),registry={'base':ready['registry'],'overlay':overlay_pin},parent_readiness=parent_ready_pin,
        concrete_descriptors=6370,blocked_fit_intentions=90,blocked_by_reason=dict(blocks),runtime_metadata_validation_passes=6370,
        additional_metadata_validation_passes=4,inherited_validation_passes=6366,source_receipt_files=len(source_receipts),
        descriptor_storage={'reused_parent_descriptors':6366,'new_descriptors':4,'new_blocked_metadata_records':36,'unchanged_ancestor_files_copied':0},
        all_native_method_preset_source_bindings_exist=True,source_binding_is_runtime_qualification=False,
        native_BUDDY_interaction_CPU_gate='pending_at_handoff;not_run_by_compiler',artifacts=artifacts)
    readiness_pin=write(output/'READINESS.json',result)
    # Verify every newly written artifact/descriptor and reused count, no model/data access.
    for r in new_descriptors:
        job=load(r['descriptor']['path'],r['descriptor']);assert digest(job)==r['runtime_job_hash']
    assert not any(k in sys.modules for k in ('torch','numpy','scipy','torch_geometric','ogb'))
    verification={'schema':'expanded-source-overlay-metadata-verification-v2','status':'pass','readiness':readiness_pin,
        'fit_intentions':6460,'unchanged_intention_ID_set':True,'descriptor_prepared':6370,'blocked':90,'blocked_by_reason':dict(blocks),
        'new_descriptor_hashes_and_metadata_validator_passes':4,'reused_descriptor_file_receipts_authenticated':6366,
        'parent_independent_verification':source_pin(HERE/'VERIFICATION_v2_generated_v1.json'),
        'all_2480_ancestor_ID_contract_records_preserved':True,'all_paired_control_and_logical_bank_IDs_preserved':True,
        'parent_output_rewritten':False,'models_constructed':0,'data_or_tensor_files_opened':0,'scientific_fits_started':0,
        'GPU_compute_started':False,'TEST_accessed':False,'runtime_qualification':False,'resource_admission':False}
    verification_pin=write(output/'VERIFICATION.json',verification)
    print(json.dumps({'readiness':readiness_pin,'verification':verification_pin,'descriptor_prepared':6370,'blocked':90,
        'blocked_by_reason':dict(blocks),'new_descriptors':4,'reused_descriptors':6366,'scientific_fits_started':0},sort_keys=True))


if __name__=='__main__':main()
