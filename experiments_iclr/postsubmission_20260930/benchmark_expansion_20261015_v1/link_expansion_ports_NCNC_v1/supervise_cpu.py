"""Finite sequential CPU engineering checks; no fit scheduler or retry policy."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

METHODS = ('aggregation_conditional_globalJ_v1','link_endpoint_responsibility_private_v2','portable_input_residual_be_v1',
    'aggregation_mean_globalJ_v1','aggregation_global_globalJ_v1','link_query_iid_responsibility_private_v2',
    'link_unit_responsibility_private_v2','portable_input_residual_only_v1')


def main():
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.root).resolve();output=Path(args.output).resolve();source=Path(__file__).resolve().parent
    assert output.is_relative_to(root) and source.is_relative_to(root) and not output.exists()
    output.mkdir(parents=True)
    started=time.perf_counter();report={'schema':'finite-NCNC-CPU-port-check-supervision-v1','status':'running','methods':list(METHODS),'cases':[],
        'maximum_total_optimizer_transitions':16,'maximum_FULL_VALID_calls':24,'maximum_children':1,'threads_per_child':2,
        'timeout_seconds_per_child':1800,'address_space_limit_bytes_per_child':64*1024**3,'scientific_fits_started':0,'GPU_updates':0,'TEST_accessed':False}
    def write():
        report['elapsed_seconds']=time.perf_counter()-started
        temporary=output/'SUMMARY.json.part';temporary.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');temporary.replace(output/'SUMMARY.json')
    write()
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for method in METHODS:
        path=output/method;row={'method':method,'status':'running','output':str(path)};report['cases'].append(row);write()
        command=[sys.executable,'-B','-u',str(source/'qualify_NCNC_cpu.py'),'--root',str(root),'--method',method,'--output',str(path)]
        with (output/(method+'.log')).open('xb') as log:
            try:
                child=subprocess.run(command,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                row['returncode']=child.returncode
            except subprocess.TimeoutExpired:
                row.update(status='timeout',costs_after_last_receipt='unknown; retain last stage and actual transition counters')
        result=json.loads((path/'RESULT.json').read_text()) if (path/'RESULT.json').is_file() else None
        if result is not None:
            row.update(actual_optimizer_transitions=result['actual_optimizer_transitions'],VALID_calls=result['VALID_calls'],
                native_constructions=result['native_constructions'],last_stage=result['stage'],failure=result.get('failure'),
                observed_elapsed_seconds=result['elapsed_seconds'],process_peak_RSS_KiB=result['process_peak_RSS_KiB'])
        if row['status']=='running':row['status']=result['status'] if result and row.get('returncode')==0 else 'failed'
        write()
        if row['status']!='passed':
            report['stopped_after_failed_case']=method;break
    report.update(status='passed' if all(row['status']=='passed' for row in report['cases']) else 'failed',
        actual_optimizer_transitions=sum(row.get('actual_optimizer_transitions',0) for row in report['cases']),
        actual_FULL_VALID_calls=sum(row.get('VALID_calls',0) for row in report['cases']))
    write();print(json.dumps(report));return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
