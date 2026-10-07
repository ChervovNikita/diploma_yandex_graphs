"""Data-only reviewed exporter execution and scalar/metadata CPU custody."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import time

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
HERE=Path(__file__).resolve().parent
SOURCE=PHASE/'learnable_internal_be_contrastive_multitask_suite_20261007_v4'
PYTHON=PHASE/'native_ncn_runtime_20261005_v1/.venv/bin/python'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def guard():
    if socket.gethostname()!='anogena-2-0':raise ValueError('Wrong authorized singleton host')
    if subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()!=[GPU]:
        raise ValueError('Wrong authorized singleton UUID')
    if not HERE.is_relative_to(PHASE) or Path.cwd().resolve()!=REPO:
        raise ValueError('Own phase/repository only')


def main():
    started=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['wikics','collab','molhiv'],required=True)
    args=parser.parse_args();guard();task=args.task
    job_path=HERE/'jobs'/str(task+'.json');job=json.loads(job_path.read_text())
    if job['data_export_authorized'] is not True or job['scientific_fit_authorized'] is not False or job['TEST_access'] is not False:
        raise ValueError('Only separately authorized safe data export')
    if sha(SOURCE/'MANIFEST.json')!=job['source_manifest_sha256'] or sha(SOURCE/'export_roles.py')!=job['source_program_sha256']:
        raise ValueError('Reviewed immutable exporter differs')
    receipt_path=HERE/'receipts'/str(task+'_EXECUTION.json')
    if receipt_path.exists():raise ValueError('One owned attempt; no overwrite/retry')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(PHASE/'native_ncn_dependency_overlay_20261005_v1')+':'+str(REPO/'.venv/lib/python3.11/site-packages'))
    peak_rss=0;status='startup_failed';exit_code=None
    log=HERE/'logs'/str(task+'.log')
    with log.open('xb') as output:
        child=subprocess.Popen([str(PYTHON),'-B',str(SOURCE/'export_roles.py'),'--job',str(job_path)],
            cwd=str(REPO),env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            try:
                for line in Path('/proc/'+str(child.pid)+'/status').read_text().splitlines():
                    if line.startswith(('VmRSS:','VmHWM:')):peak_rss=max(peak_rss,int(line.split()[1])*1024)
            except FileNotFoundError:pass
            if time.monotonic()-started>1800:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
                status='data_export_timeout';break
            time.sleep(.1)
        exit_code=child.wait()
        if status!='data_export_timeout':status='complete' if exit_code==0 else 'export_failed'
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    folder=PHASE/job['output_directory'];files=[]
    if folder.exists():
        for path in sorted(folder.iterdir()):
            if path.is_file():files.append(dict(path=str(path.relative_to(PHASE)),bytes=path.stat().st_size,sha256=sha(path)))
    receipt=dict(schema='internal-be-safe-role-export-owned-execution-v1',task=task,status=status,exit_code=exit_code,
        hostname=socket.gethostname(),physical_gpu_uuid=GPU,CUDA_VISIBLE_DEVICES='',
        source_manifest_sha256=job['source_manifest_sha256'],source_program_sha256=job['source_program_sha256'],
        job_sha256=sha(job_path),source_review_sha256=job['source_review']['sha256'],
        inclusive_seconds=time.monotonic()-started,user_CPU_seconds=after.ru_utime-before.ru_utime,
        system_CPU_seconds=after.ru_stime-before.ru_stime,peak_RSS_bytes=max(peak_rss,int(after.ru_maxrss)*1024),
        input_blocks=after.ru_inblock-before.ru_inblock,output_blocks=after.ru_oublock-before.ru_oublock,
        output_files=files,total_output_bytes=sum(row['bytes'] for row in files),log_sha256=sha(log),
        log_bytes=log.stat().st_size,scientific_fits=0,predictive_scores_computed=False,TEST_target_values_parsed=False,
        all_graph_public_count_metadata_parsed=task=='molhiv',binaries_server_only=True,automatic_retry=False)
    receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(receipt,sort_keys=True))
    if exit_code!=0:raise SystemExit(1)


if __name__=='__main__':main()
