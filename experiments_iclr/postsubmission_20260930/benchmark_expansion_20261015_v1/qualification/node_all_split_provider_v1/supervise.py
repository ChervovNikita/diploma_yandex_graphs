import os,subprocess,json,time,hashlib
from pathlib import Path
root=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');e=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1');q=Path(__file__).resolve().parent;d=e/'node_all_split_provider_audit_v1';started=time.perf_counter()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
record={'source_manifest_sha256':hashlib.sha256((d/'MANIFEST.json').read_bytes()).hexdigest(),'models_constructed_by_supervisor':0,'scientific_jobs_started':0}
try:
 with (q/'child.log').open('xb') as log:
  result=subprocess.run([str(root/'.venv/bin/python'),'-B',str(d/'audit.py')],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=480)
 record.update(exit_code=result.returncode,status='passed' if result.returncode==0 else 'failed')
except subprocess.TimeoutExpired:
 record.update(exit_code=None,status='timed_out',timeout_seconds=480)
finally:
 record['elapsed_seconds']=time.perf_counter()-started
 (q/'TERMINAL.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
