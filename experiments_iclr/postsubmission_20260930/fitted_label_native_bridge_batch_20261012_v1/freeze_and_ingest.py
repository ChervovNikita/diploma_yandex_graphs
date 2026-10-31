"""Freeze owned source and ingest an actual disabled packet; no fitting."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

CPU_RESULT='fitted_label_native_bridge_cpu_source_check_20261012_v1/RESULT.json'
OUTPUT='massive_paid_native_label_packet_20261012_v1'
CUSTODIES={d:'training_pipeline_v2_20261012/prepared_data/'+d+'/CUSTODY.json' for d in ('wikics','amazon-photo')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);args=p.parse_args()
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Exact authorized allocation route required before project access')
    root=Path(args.project_root).resolve();stage=Path(__file__).resolve().parent
    if not stage.is_relative_to(root):raise ValueError('Project-only owned stage required')
    def receipt(path):
        raw=path.read_bytes();return {'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    if any((stage/n).exists() for n in ('METADATA_QA.json','FROZEN_RUNTIME_MANIFEST.json','ACTUAL_CONTROLLER_INGEST.json','FREEZE_FAILURE.json')) or (root/OUTPUT).exists():
        raise ValueError('Fresh immutable freeze and packet required')
    try:
        inventory=json.loads((stage/'SOURCE_INVENTORY_V2.json').read_text())
        for row in inventory:
            raw=(stage/row['path']).read_bytes()
            if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Owned freeze source changed: '+row['path'])
        cpu_path=root/CPU_RESULT;cpu=json.loads(cpu_path.read_text())
        if cpu.get('status')!='passed' or cpu.get('actual_optimizer_updates')!=0 or cpu.get('scientific_fits')!=0 or cpu.get('CUDA') is not False or cpu.get('GO') is not False:
            raise ValueError('Prior synthetic CPU source/state qualification must be successful and non-scientific')
        arms=[c for c in cpu['cases'] if 'arm' in c]
        if len(arms)!=18 or len(cpu['cases'])!=20:raise ValueError('All18 arm×class cases and two gate cases required')
        old={row['path']:row for row in json.loads((stage/'STAGING_MANIFEST_V1.json').read_text())['files']}
        numeric_files=('runtime/adapters/label_stage.py','runtime/adapters/native_polynormer_qk.py','runtime/adapters/classifier.py','runtime/engine.py','runtime/metrics.py','runtime/spec.py','runtime/run_study.py')
        for name in numeric_files:
            bound=receipt(stage/name);expected=old[name]
            if any(bound[k]!=expected[k] for k in ('bytes','sha256')):raise ValueError('Previously numeric-qualified module changed: '+name)
        python=root.parent.parent/'.venv/bin/python';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(stage/'runtime')+os.pathsep+str(stage/'tests'))
        command=[str(python),'-B','-m','unittest','test_paid_label_packet','-v'];start=time.monotonic()
        completed=subprocess.run(command,cwd=stage,env=env,capture_output=True,text=True,timeout=60)
        qa={'scope':'pure standard-library synthetic metadata tests; no real graph/target decoding or model execution',
            'command':command,'exit_code':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr,'seconds':time.monotonic()-start,
            'tests':6,'scientific_fits':0,'CUDA':False,'GO':False}
        (stage/'METADATA_QA.json').write_text(json.dumps(qa,indent=2)+'\n')
        if completed.returncode:raise ValueError('Owned metadata QA failed; retained')
        sys.path.insert(0,str(stage/'runtime'))
        from compile_master_plan import controller_bindings,load_packet,bank_roster
        runtime=str((stage/'runtime').relative_to(root))
        manifest={'schema':'paid-native-label-owned-controller-extension-v1','runtime_package':runtime,
            'controller_files':controller_bindings(root,runtime),'source_inventory':receipt(stage/'SOURCE_INVENTORY_V2.json'),
            'all_runtime_files':[receipt(f) for f in sorted((stage/'runtime').rglob('*')) if f.is_file()],
            'metadata_QA':receipt(stage/'METADATA_QA.json'),'synthetic_numeric_QA':receipt(cpu_path),
            'numeric_qualified_module_receipts':[receipt(stage/f) for f in numeric_files],
            'previous_provisional_manifest':receipt(stage/'STAGING_MANIFEST_V1.json'),
            'inheritance':'Exact v6 source copy except compile_master_plan.py and run_study.py; own label adapters/compiler added. launcher.py and start_massive.sh unchanged.',
            'semantics_changes':['native_capture selection-kind stage kept outside whole_search closure',
                'explicit paid-native-capture-label-stage-v1 lifecycle and exact new paid parent/capture graph checks',
                'nine label arms typed by all source owners and complete old-state cycles; zero NEW slow bodies; acquisition charged separately',
                'source head forward/backward counts and replaced nonlinear-readout Adam constructor overhead'],
            'numeric_engine_behavior':'Inherited; only paid-parent identifier parsing expanded in dispatcher.',
            'scientific_launch_enabled':False,'human_GO_required':True,'actual_fits':0,'GO':False}
        (stage/'FROZEN_RUNTIME_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        command=[str(python),'-B',str(stage/'runtime/compile_paid_label_packet.py'),'--project-root',str(root),'--runtime-package',runtime,'--output',OUTPUT,'--custody',*CUSTODIES.values()]
        completed=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=120)
        compile_result={'command':command,'exit_code':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr,
            'scope':'Actual custody byte hashes, source hashes and finite metadata compile only; targets not decoded; no model instantiated',
            'scientific_fits':0,'CUDA':False,'GO':False}
        (stage/'ACTUAL_PACKET_COMPILE.json').write_text(json.dumps(compile_result,indent=2)+'\n')
        if completed.returncode:raise ValueError('Disabled packet compilation failed; retained')
        packet=root/OUTPUT
        entry={'id':'paid_native_label_stage_v1','plan':receipt(packet/'PLAN.json'),'summary':receipt(packet/'SUMMARY.json'),
            'coverage':[receipt(stage/'FROZEN_RUNTIME_MANIFEST.json'),receipt(stage/'PROTOCOL.md')]}
        loaded=load_packet(root,entry);t=loaded['counts']['totals']
        expected={'run_records':4800,'atomic_fit_run_groups':1444,'paid_optimizer_starts':2056,'physical_slow_body_instances':526,
            'known_registered_optimizer_step_calls':10280000,'partial_registered_known_source_TRAIN_forward_calls':24340000,
            'partial_registered_known_source_TRAIN_backward_calls':21280000,'unstepped_pre_install_Adam_constructor_objects':204,
            'paid_native_capture_jobs':94,'search_selection_jobs':22,'bank_exports':1080,'scalar_CAL_fit_jobs':1080,
            'registered_scalar_CAL_updates':540000,'evaluation_jobs':1080,'genuine_separately_selected_I4_banks':90}
        if any(t.get(k)!=v for k,v in expected.items()):raise ValueError('Exact finite packet accounting changed')
        original=sum(r['kind']=='model_fit' and r['method']=='original_gnnm' for s in loaded['studies'] for r in s['runs'])
        if original!=126:raise ValueError('Original GNNM new-data coverage changed')
        role_meta={d:json.loads((root/path).read_text()) for d,path in CUSTODIES.items()};captures=[];endpoints=[]
        for study in loaded['studies']:
            meta=role_meta[study['dataset']];N=meta['model_graph']['nodes'];C=meta['model_graph']['classes']
            cap_count=sum(r['config']['stage']=='native_capture' for r in study['runs'])
            captures.append({'dataset':study['dataset'],'capture_files':cap_count,'nodes':N,'hidden_width':512,'classes':C,
                'float32_H_plus_native_logits_payload_bytes_per_capture':N*(512+C)*4,'float32_payload_bytes_all_captures':cap_count*N*(512+C)*4,
                'serialized_bytes_measured':False,'native_weights_not_duplicated_in_capture':True})
            sizes={r['split_id']:r['role_counts'] for r in meta['splits']};banks=bank_roster(study)
            payload=sum(b['routes']*C*(sizes[b['split']]['cal']+sizes[b['split']]['evaluate'])*4 for b in banks)
            endpoints.append({'dataset':study['dataset'],'bank_count':len(banks),'logit_tensor_files':2*len(banks),
                'float32_logit_payload_bytes':payload,'serialized_ids_and_container_overhead_included':False})
        storage={'schema':'paid-native-label-storage-accounting-v1','scope':'exact finite object counts and raw tensor payload math; serialized disk and peak-resource admission remain unmeasured',
            'fit_groups':1444,'successful_fit_full_state_files_per_group':['selected.pt','final_training.pt','latest_trajectory.pt'],
            'successful_fit_full_state_file_slots':4332,'native_capture_files':94,'endpoint_logit_files':2160,
            'total_named_tensor_file_slots_excluding_retries':6586,'captures':captures,'endpoint_logits':endpoints,
            'capture_H_plus_native_logits_float32_payload_total_bytes':sum(c['float32_payload_bytes_all_captures'] for c in captures),
            'endpoint_logits_float32_payload_total_bytes':sum(c['float32_logit_payload_bytes'] for c in endpoints),
            'label_map_parameter_counts_from_synthetic_same_width_class_CPU_qualification':[{k:c[k] for k in ('arm','classes','parameters','source_optimizer_owners')} for c in arms],
            'full_state_bytes_not_inferred_from_parameters':'Contains buffers, optimizer state, RNG and native local snapshots; exact bytes vary and must be measured in admitted engineering/runtime.',
            'shared_native_capture_is_94_files_not_918_files':True,'actual_fits':0,'GO':False}
        (packet/'STORAGE_ACCOUNTING.json').write_text(json.dumps(storage,indent=2,sort_keys=True)+'\n')
        report={'schema':'paid-native-label-actual-controller-ingestion-v1','entry':entry,'loaded_studies':len(loaded['studies']),
            'counts':loaded['counts'],'references':loaded['references'],'original_gnnm_new_data_fit_records':original,
            'declared_source_data_bindings_rehashed_by_loader':False,'compiler_rehashed_actual_custody_bytes':True,
            'storage_accounting':receipt(packet/'STORAGE_ACCOUNTING.json'),'synthetic_CPU_qualification':receipt(cpu_path),
            'actual_input_resources_qualified':False,'actual_fits':0,'CUDA':False,'GO':False}
        (stage/'ACTUAL_CONTROLLER_INGEST.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        (packet/'MASTER_INPUT_ENTRY.json').write_text(json.dumps(entry,indent=2,sort_keys=True)+'\n')
        seal={'schema':'paid-native-label-disabled-packet-freeze-v1','files':[receipt(f) for f in sorted(packet.rglob('*')) if f.is_file()],
            'owned_runtime_manifest':receipt(stage/'FROZEN_RUNTIME_MANIFEST.json'),'controller_ingestion':receipt(stage/'ACTUAL_CONTROLLER_INGEST.json'),
            'actual_input_resources_qualified':False,'scientific_launch_enabled':False,'GO':False}
        (packet/'MANIFEST.json').write_text(json.dumps(seal,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'packet_manifest':receipt(packet/'MANIFEST.json'),'controller_manifest':receipt(stage/'FROZEN_RUNTIME_MANIFEST.json'),
            'ingestion':receipt(stage/'ACTUAL_CONTROLLER_INGEST.json'),'entry':receipt(packet/'MASTER_INPUT_ENTRY.json'),
            'fits_planned':1444,'native_acquisitions':526,'label_fits':918,'Adam_calls_planned':10280000,'original_gnnm_new_data_fits':126,'fits_executed':0,'GO':False}))
    except BaseException as e:
        (stage/'FREEZE_FAILURE.json').write_text(json.dumps({'status':'failed','error':str(e),'traceback':traceback.format_exc(),'scientific_fits':0,'GO':False},indent=2)+'\n')
        raise


if __name__=='__main__':main()
