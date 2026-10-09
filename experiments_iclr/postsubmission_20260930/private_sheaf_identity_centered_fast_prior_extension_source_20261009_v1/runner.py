"""Inactive three centered fits; original18 origins required and all21 closure."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import resource
import time
from support import HERE,admission,borrowed,key,load_centered_fit,read,require,sha,source_checks

VERSIONS=('torch','numpy','scipy','torch-geometric','torch-sparse','torch-scatter','torch-householder','scikit-learn')


def run(args):
    pins,protocol=source_checks(),read(HERE/'PROTOCOL.json')
    release,receipt,config,output,origins=admission(args,pins,protocol)
    old,common,helpers,placement=borrowed(pins)
    fit=load_centered_fit(pins)
    output.mkdir(parents=True,exist_ok=False)
    started,usage0=time.perf_counter(),resource.getrusage(resource.RUSAGE_SELF)
    records=[]; failure=None
    schedule=[dict(kind='centered_shared_fit',base_seed=base) for base in protocol['seeds']]
    try:
        common.write_json(output/'SCHEDULE.json',dict(protocol=protocol,root_release=release,root_admission=receipt,
            original18_origin_receipt=origins,components=schedule,automatic_retry=False,comparative_opening_authorized=False))
        import numpy as np
        import torch
        from sklearn.metrics import roc_auc_score
        versions={name:importlib.metadata.version(name) for name in VERSIONS}
        require(versions==release['expected_runtime_versions'] and str(torch.__version__)=='2.1.2+cu118' and np.__version__=='1.26.4','Exact existing runtime')
        device=torch.device(release['device'])
        require(device==torch.device('cuda:0') and torch.cuda.device_count()==1,'One root-owned visible cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(release['deterministic_algorithms'])
        arrays,role_meta,role_sha=common.read_roles(np,args.roles)
        require(role_meta.get('exposure_classification')=='original_paper_benchmark_exploratory','Same already used exploratory benchmark')
        data={name:torch.from_numpy(value).to(device) for name,value in arrays.items()}
        data['cpu_edge_index']=torch.from_numpy(arrays['edge_index'])
        common.write_json(output/'ROLE_METADATA.json',role_meta)
        adapter=common.load_adapter()
        identity=dict(source_seal_sha256=sha(HERE/'SEAL.json'),manifest_sha256=read(HERE/'SEAL.json')['manifest_sha256'],
            execution_source_commit=release['execution_source_commit'],admission_receipt_sha256=sha(args.admission),
            role_archive_sha256=release['roles_sha256'],role_metadata_sha256=role_sha,native_commit=pins['native_commit'],
            runtime_versions=versions,configuration=config,optimizer=receipt['optimizer'],
            centered_optimizer_groups=dict(slow_incidence_decay=0.0005,slow_other_decay=0.0005,fast_decay=0.0),
            centered_prior=dict(coefficient=0.0005,center=1.0,reduction='lambda/(2M) sum all private factor squared distances'),
            vanilla_control_source_seal_sha256=pins['scientific_core_seal_sha256'],original18_origins_sha256=release['original18_origins']['sha256'],
            revised_scientific_scope_sha256=release['revised_scientific_scope']['sha256'],root_release_sha256=sha(args.release),
            context_regularizer=0.0,TEST_truth_present=False,fresh_dataset_or_unused_split_claim=False,
            candidate_predeclared_co_primary=True,original18_already_complete=True,required_logical_records_before_comparison=21)
        common.write_json(output/'RUN_IDENTITY.json',identity)
        for base in protocol['seeds']:
            record=fit.fit(np,torch,roc_auc_score,old,common,helpers,placement,adapter,receipt['optimizer'],config,data,role_meta,identity,output,base)
            require(key(record) not in {key(row) for row in records},'No duplicate/replacement centered attempts')
            records.append(record); common.append_jsonl(output/'CENTERED_RECORDS.jsonl',record)
    except BaseException as error:
        failure=common.failure_record(error,'prospective_centered_three_fit_extension')
        raise
    finally:
        for item in schedule:
            if key(item) in {key(row) for row in records}:continue
            path=output/'shared'/('seed'+str(item['base_seed']))/'RESULT.json'
            row=read(path) if path.exists() else dict(item,status='failed_before_component',actual_fit_started=False,failure=failure,automatic_retry=False)
            if row['status']=='started':row.update(status='interrupted',failure=failure)
            records.append(row)
        complete=failure is None and len(records)==3 and all(row['status']=='complete' for row in records)
        common.write_json(output/'CENTERED_RECORDS.json',records)
        usage=resource.getrusage(resource.RUSAGE_SELF)
        common.write_json(output/'ALL21_CLOSURE.json',dict(complete=complete,original18_complete=True,centered3_complete=complete,
            logical_records=[{name:row[name] for name in ('kind','base_seed','status','member') if name in row} for row in origins['records']+records],
            required_logical_records=21,original18_origins_sha256=release['original18_origins']['sha256'],
            original_costs_fully_charged=True,new_centered_execution_seconds=time.perf_counter()-started,
            CPU_user_seconds=usage.ru_utime-usage0.ru_utime,CPU_system_seconds=usage.ru_stime-usage0.ru_stime,
            process_RSS_high_water_cumulative_bytes=common.process_peak_rss_bytes(),
            original_or_reused_predictions_replayed=False,new_independent_fits=0,
            all21_required_before_comparative_outcome_access=True,comparative_opening_authorized=False,TEST_truth_present=False,
            failure=failure,automatic_retry=False,configuration_search=False))
        gc.collect()
    require(complete,'Both full recipes and all21 records required; no survivor or partial comparative opening')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    for name in ('release','admission','roles','output'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(inactive=True,numeric_model_or_role_import=False,protocol=read(HERE/'PROTOCOL.json'))));return
    require(args.release and args.admission and args.roles and args.output,'Explicit root release/admission/roles/fresh output required')
    run(args)


if __name__=='__main__':main()
