"""Wrap an unchanged disabled sealed link study in master-ingestable metadata."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import socket
import subprocess

from partition import assign,DEFAULT_WORKERS
from spec import canonical,fingerprint,validate


def make_plan(study,study_path,runtime,capacities,output_relative):
    validate(study)
    if set(capacities)!=set(DEFAULT_WORKERS) or any(type(v)!=int or v<=0 for v in capacities.values()):
        raise ValueError('Explicit positive capacities for all three logical workers required')
    if study['release'].get('scientific_launch_enabled') is not False or study['release'].get('human_GO_required') is not True:
        raise ValueError('Wrap only a disabled study requiring human GO')
    records={study['study_id']+'/'+r['run_id']:r for r in study['runs']}
    for key,row in records.items():
        if any(study['study_id']+'/'+p not in records for p in row['parents']):raise ValueError('Missing exact own-study parent')
    keys=sorted(records);cost=sum(r['config']['estimated_seconds'] for r in records.values())
    maximum=max(r['config']['reservation_mib'] for r in records.values())*1024**2
    placement=assign([{'id':study['study_id'],'study_ids':[study['study_id']],'run_keys':keys,
        'estimated_seconds':cost,'maximum_cuda_bytes':maximum,'allowed_workers':['allocation/gpu0']}],capacities)
    gates={}
    for name,stages,required in (('whole_search',['search','search_selection'],['comparative']),
                                 ('whole_comparative',['comparative'],['bank_export','evaluation'])):
        cohort=sorted(k for k,r in records.items() if r['config']['stage'] in stages)
        gates[name]={'path':output_relative+'/gate_receipts/'+name+'.json','source_stages':stages,
            'required_for_stages':required,'required_hosts':['allocation'],'all_packet_source_run_keys':cohort,
            'input_packet_gate_receipts_accepted':False}
    plan={'schema_version':1,'slots_per_gpu':2,'torch_threads_per_fit':2,'runtime_package':runtime,
        'runtime_launcher':runtime+'/launcher.py','output_root':output_relative+'/execution',
        'studies':[{'path':study_path,'spec_hash':fingerprint(study)}],'groups':placement['groups'],
        'worker_capacities_cuda_bytes':capacities,'estimated_gpu_seconds':placement['estimated_gpu_seconds'],
        'estimates_are_measurements':False,'barriers':{},'global_gates':gates,
        'host_bindings':{'allocation':{'status':'pending_live_verification_before_launch'},
            'server77':{'status':'planning_only_access_not_authorized'}},
        'scientific_launch_enabled':False,'human_GO_required':True,'approval_receipt_created':False}
    counts=Counter(r['kind'] for r in records.values());fit_rows=[r for r in records.values() if r['kind']=='model_fit']
    reuse=[r for r in records.values() if r['kind']=='ensemble_assembly' and 'reuse_export_parent' in r['config']]
    epochs=sum(r['budget']['maximum'] for r in fit_rows)
    summary={'schema':'disabled-link-plan-summary-v1','status':'MATERIALIZED_DISABLED_REQUIRED_COMPARISONS_AND_RESOURCES_PENDING',
        'plan_hash':fingerprint(plan),'study_hash':fingerprint(study),'run_records':len(keys),
        'model_fits':counts['model_fit'],'search_model_fits':sum(r['config']['stage']=='search' for r in fit_rows),
        'comparative_model_fits':sum(r['config']['stage']=='comparative' for r in fit_rows),
        'search_selection_jobs':counts['selection'],'bank_exports':counts['ensemble_assembly'],
        'four_member_prediction_exports':counts['ensemble_assembly']-len(reuse),'member0_copy_exports':len(reuse),
        'evaluation_jobs':counts['evaluation'],'CAL_fit_jobs':counts['calibration'],'registered_epochs':epochs,
        'native_declared_updates_per_epoch':17,'source_bound_declared_optimizer_updates':epochs*17,
        'update_count_basis':'authenticated TRAIN1179052; native floor batch65536; full100epochs; no executed optimizer updates',
        'reservation_mib':max(r['config']['reservation_mib'] for r in records.values()),'resource_scope':'provisional_unmeasured',
        'resources':study['resources'],'per_worker':{},'pending_required_work':[
            {'id':'original_GNNM_link_task_adaptation','status':'blocked_unmaterialized','fit_count':None,'required_comparison':True,
             'exact_gap':'Separate original boundary-encoder/query-pair-decoder, TRAIN masking/link objective/sampler, own SELECT/restore/export adapter are absent.'},
            {'id':'link_CUDA_sparse_resource_qualification','status':'pending_zero_update_qualifier','optimizer_updates':0,'measurement_available':False},
            {'id':'NCN_NCNC_author_final_TRAIN_VALID_protocol','status':'blocked_unmaterialized','fit_count':None,
             'exact_gap':'Separate author-final TRAIN+VALID support and fit-time diagnostics/export/ancestry protocol must be independently verified.'},
            {'id':'root_global_plan_gates','status':'pending_parent_combined_plan'},
            {'id':'human_GO','status':'not_received'}],
        'original_GNNM_link_fits_counted':0,'author_protocol_replication_fits_counted':0,
        'confidence_claims_available':False,'all_249_predictive_candidates_covered':False,
        'scientific_launch_enabled':False,'human_GO_required':True,'GO_receipt_created':False,
        'training_started':False,'benchmark_training_started':False,'benchmark_optimizer_fits_executed':0,
        'scientific_training_runs':0,'predictive_scores_computed':0,'source_data_arrays_opened':0,
        'old_worker_source_bindings_preserved':True,'root_common_controller_and_gates_are_separate':True,
        'three_worker_capacities_are_planning_bounds_not_live_proofs':True}
    assigned='allocation/gpu0'
    for worker in DEFAULT_WORKERS:
        own=worker==assigned;pair=(2*maximum if own else 0)
        summary['per_worker'][worker]={'run_records':len(keys) if own else 0,'model_fits':len(fit_rows) if own else 0,
            'estimated_gpu_seconds':placement['estimated_gpu_seconds'][worker],'capacity_bytes':capacities[worker],
            'maximum_run_reservation_bytes':maximum if own else 0,'two_largest_registered_reservations_bytes':pair,
            'two_slot_pair_fits_planning_bound':pair<=capacities[worker],'concurrent_throughput_measured':False,
            'live_hardware_and_external_headroom_qualification_pending':True}
    return plan,summary


def receipt(root,path):
    path=Path(path).resolve();root=Path(root).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():raise ValueError('Confine metadata receipt')
    raw=path.read_bytes();return {'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project-root','study','capacities','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--runtime',required=True)
    args=parser.parse_args()
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise RuntimeError('Metadata preparation requires verified allocation')
    root=args.project_root.resolve();output=args.output.resolve()
    if not output.is_relative_to(root) or output.exists():raise ValueError('Use fresh confined own metadata output')
    inputs=[receipt(root,args.study),receipt(root,args.capacities),receipt(root,Path(__file__))]
    study=json.loads(args.study.read_text());capacities=json.loads(args.capacities.read_text())['worker_capacities_cuda_bytes']
    plan,summary=make_plan(study,inputs[0]['path'],args.runtime,capacities,str(output.relative_to(root)))
    summary['metadata_inputs']=inputs
    output.mkdir(parents=True)
    (output/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    (output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'status':summary['status'],'plan_hash':summary['plan_hash'],'run_records':summary['run_records'],
        'model_fits':summary['model_fits'],'scientific_launch_enabled':False,'human_GO_required':True}))


if __name__=='__main__':main()
