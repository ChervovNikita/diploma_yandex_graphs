"""Pure CPU metadata QA/freeze; actual-input and scientific launch stay disabled."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);args=p.parse_args()
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Exact authorized allocation route required')
    root=Path(args.project_root).resolve();stage=Path(__file__).resolve().parent
    if not stage.is_relative_to(root): raise ValueError('Project-only stage required')
    def receipt(path):
        raw=path.read_bytes();return {'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    for row in json.loads((stage/'SOURCE_INVENTORY.json').read_text()):
        raw=(stage/row['path']).read_bytes()
        if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']: raise ValueError('Frozen controller source changed')
    qa_path=stage/'ENGINEERING_QA.json'
    if qa_path.exists() or (stage/'MANIFEST.json').exists(): raise ValueError('Fresh successor qualification required')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(stage/'runtime')+os.pathsep+str(stage/'tests'))
    command=[str(root.parent.parent/'.venv/bin/python'),'-B','-m','unittest','test_native_polynormer_qk_accounting','test_gradient_owner_accounting','-v']
    started=time.monotonic();completed=subprocess.run(command,cwd=stage,env=env,capture_output=True,text=True,timeout=60)
    result={'command':command,'exit_code':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr,
            'wall_seconds':time.monotonic()-started,'scope':'standard-library metadata accounting tests only','scientific_fits':0,'GO':False}
    qa_path.write_text(json.dumps(result,indent=2)+'\n')
    if completed.returncode: raise ValueError('Accounting QA failed; retained')
    sys.path.insert(0,str(stage/'runtime'))
    from compile_master_plan import controller_bindings,load_packet,typed_counts
    previous=root/'master_controller_source_20261012_v5/MANIFEST.json';old=json.loads(previous.read_text())
    manifest=copy.deepcopy(old)
    manifest.update(runtime_package=str((stage/'runtime').relative_to(root)),files=controller_bindings(root,str((stage/'runtime').relative_to(root))),
                    engineering_QA=receipt(qa_path),passed_engineering_tests=7,
                    prior_controller_manifest=receipt(previous),source_snapshot=receipt(stage/'SOURCE_INVENTORY.json'),
                    new_scope='Native Polynormer Q/K exact two-view TRAIN costs and unstepped pre-install Adam constructor overhead; prior ownership laws retained',
                    scientific_fits=0,benchmark_scores=0,start_commands_are_disabled_until_GO=True)
    for name,value in (('MANIFEST.json',manifest),('HOST_BINDINGS.json',manifest['host_bindings']),('RENDEZVOUS_POLICY.json',manifest['phase_rendezvous'])):
        (stage/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    packet=root/'massive_native_polynormer_qk_packet_20261012_v2'
    command=[str(root.parent.parent/'.venv/bin/python'),'-B',str(root/'native_polynormer_qk_cpu_stage_20261012_v3/runtime/compile_native_polynormer_qk_matrix.py'),
             '--project-root',str(root),'--source-bindings','native_polynormer_qk_cpu_stage_20261012_v3/SOURCE_BINDINGS.json',
             '--runtime-package','native_polynormer_qk_cpu_stage_20261012_v3/runtime','--output',str(packet),
             '--custody','training_pipeline_v2_20261012/prepared_data/wikics/CUSTODY.json','training_pipeline_v2_20261012/prepared_data/amazon-photo/CUSTODY.json']
    completed=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=60)
    compile_result={'command':command,'exit_code':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr,'fits':0,'GO':False}
    (stage/'COMPILE_RESULT.json').write_text(json.dumps(compile_result,indent=2)+'\n')
    if completed.returncode: raise ValueError('Disabled packet compile failed; retained')
    entry={'id':'native_polynormer_qk_v2','plan':receipt(packet/'PLAN.json'),'summary':receipt(packet/'SUMMARY.json')}
    loaded=load_packet(root,entry);costs=loaded['counts'];total=costs['totals']
    if (total['atomic_fit_run_groups']!=2610 or total['paid_optimizer_starts']!=2610 or total['physical_slow_body_instances']!=2610
            or costs['model_optimizer_step_call_total']!=13050000
            or total['partial_registered_known_source_TRAIN_forward_calls']!=45000000
            or total['partial_registered_known_source_TRAIN_backward_calls']!=45000000
            or total['unstepped_pre_install_Adam_constructor_objects']!=2088
            or total['genuine_separately_selected_I4_banks']!=450 or total['run_records']!=6680):
        raise ValueError('Native source whole-roster counts/ancestry changed')
    report={'entry':entry,'typed_counts':costs,'validated_disabled_packet':True,'fits':0,'GO':False,
            'qualification_receipt':receipt(root/'native_polynormer_qk_cpu_check_20261012_v2/RESULT.json'),
            'scope':'CPU synthetic source engineering and exact metadata counts only; resource admission and real geometry remain pending'}
    (stage/'NATIVE_QK_INGEST.json').write_text(json.dumps(report,indent=2)+'\n')
    seal_files=[receipt(path) for path in sorted(packet.rglob('*')) if path.is_file()]
    (packet/'MANIFEST.json').write_text(json.dumps({'schema':'native-polynormer-qk-disabled-packet-v2','files':seal_files,
                                                  'source_runtime_manifest':receipt(root/'native_polynormer_qk_cpu_stage_20261012_v3/MANIFEST.json'),
                                                  'synthetic_CPU_qualification':report['qualification_receipt'],'controller_ingestion':receipt(stage/'NATIVE_QK_INGEST.json'),
                                                  'actual_input_geometry_qualified':False,'scientific_launch_enabled':False,'GO':False},indent=2)+'\n')
    print(json.dumps({'controller_manifest':receipt(stage/'MANIFEST.json'),'packet_manifest':receipt(packet/'MANIFEST.json'),
                      'ingest':receipt(stage/'NATIVE_QK_INGEST.json'),'model_fits_planned':2610,'fits_executed':0,'GO':False}))


if __name__=='__main__':main()
