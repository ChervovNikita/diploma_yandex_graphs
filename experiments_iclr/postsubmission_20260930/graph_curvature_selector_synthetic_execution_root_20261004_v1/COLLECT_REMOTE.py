import json,pathlib,subprocess,time
observed=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()
assert [line.strip() for line in observed if line.strip()]==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
root=pathlib.Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/graph_curvature_selector_synthetic_execution_root_20261004_v1')
names=['STAGING_RECEIPT.json','EXECUTION_INPUT.json','START_GUARD.json','PID_RECEIPT.json','EXECUTION_START.json','SUITE_PID_RECEIPT.json','SUITE_STDOUT.txt','WRAPPER_STDOUT.txt','SYNTHETIC_RESULT.json','EXECUTION_FINISH.json','EXECUTION_FAILURE.json']
files={}
for name in names:
 path=root/name
 if path.exists():
  assert path.stat().st_size<=131072, 'Refuse large receipt: '+name
  files[name]=path.read_text()
assert sum(len(text.encode()) for text in files.values())<=262144
print(json.dumps(dict(collected_unix=time.time(),GPU_UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac',execution_root=str(root),files=files)))
