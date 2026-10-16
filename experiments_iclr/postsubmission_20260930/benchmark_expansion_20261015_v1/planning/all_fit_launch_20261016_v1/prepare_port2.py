"""Run the repaired metadata projection on the guarded allocation, without fits."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent
E=HERE.parents[1]
sys.path.insert(0,str(E))
import remote_control
inputs=json.loads((HERE/'PORT2_PROJECTION_ORIGINAL_INPUTS.json').read_text())
code='''from pathlib import Path
import json,subprocess,os,hashlib,time
root=Path(ROOT_VALUE);e=root/'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1';inputs=INPUTS_VALUE
port=e/'server51_campaign_port_v2';wrapper=e/'server51_campaign_projection_cache_v2';host=port/'hosts/allocation_gpu0.json'
env=dict(os.environ,BENCHMARK_HOST_CONFIG=str(host),BENCHMARK_HOST_CONFIG_SHA256=hashlib.sha256(host.read_bytes()).hexdigest(),CUDA_VISIBLE_DEVICES='GPU-44039938-fd82-41d2-fefd-de71514e2fac',PYTHONDONTWRITEBYTECODE='1')
argv=[str(root/'.venv/bin/python'),'-B',str(wrapper/'prepare_cached.py'),'--wrapper-sha256','5452d9bebd6ce780e86aed7f5439823e15daebfda77e7fff7c4088c85dcc0f64','--cache-receipt',str(wrapper/'CACHE_RECEIPT_01.json'),'--canonical-plan',inputs['canonical_plan']['path'],'--queue-dir',inputs['queue_dir'],'--reporting-index',inputs['reporting_index']['path'],'--parent-release',inputs['parent_release']['path'],'--output',str(port/'generated_v1'),'--source-manifest-sha256','0be1ec6926b717206f403b5ed2524b7fae8a3594947d06e75b8f075f4ef7fc15']
started=time.monotonic();r=subprocess.run(argv,capture_output=True,text=True,env=env,cwd=root,timeout=900)
out={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'argv':argv,'wall_seconds':time.monotonic()-started,'native_work':0}
with (e/'planning/all_fit_launch_20261016_v1/PORT2_PROJECTION_RUN_01.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))'''.replace('ROOT_VALUE',repr(remote_control.REMOTE_ROOT)).replace('INPUTS_VALUE',repr(inputs))
out=json.loads(remote_control.call(code,timeout=950))
(HERE/'PORT2_PROJECTION_RUN_01.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out));raise SystemExit(out['returncode'])
