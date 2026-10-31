"""Run the repaired metadata projection on the guarded allocation, without fits."""
from pathlib import Path
import sys,json,base64,zlib
HERE=Path(__file__).resolve().parent
E=HERE.parents[1]
sys.path.insert(0,str(E))
import remote_control
inputs=json.loads((HERE/'PORT2_PROJECTION_ORIGINAL_INPUTS.json').read_text())
code='''from pathlib import Path
import json,subprocess,os,hashlib,time
root=Path(ROOT_VALUE);e=root/'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1';inputs=INPUTS_VALUE
port=e/'server51_campaign_port_v5';wrapper=e/'server51_campaign_projection_cache_v5';host=port/'hosts/allocation_gpu0.json'
env=dict(os.environ,BENCHMARK_HOST_CONFIG=str(host),BENCHMARK_HOST_CONFIG_SHA256=hashlib.sha256(host.read_bytes()).hexdigest(),CUDA_VISIBLE_DEVICES='GPU-44039938-fd82-41d2-fefd-de71514e2fac',PYTHONDONTWRITEBYTECODE='1')
env.pop('BENCHMARK_HOST_CONFIG_SHA256',None)
argv=[str(root/'.venv/bin/python'),'-B',str(wrapper/'prepare_cached.py'),'--wrapper-sha256','37e103190a4142f9a03545ce86d41f01b419d574f43f71203c554612f258603b','--cache-receipt',str(wrapper/'CACHE_RECEIPT_01.json'),'--canonical-plan',inputs['canonical_plan']['path'],'--queue-dir',inputs['queue_dir'],'--reporting-index',inputs['reporting_index']['path'],'--parent-release',inputs['parent_release']['path'],'--output',str(port/'generated_v1'),'--source-manifest-sha256','3fa47ca1ec206990fd25e37aa2f273e97aae8ec9e37c47800e9cfeda75654d13']
started=time.monotonic();r=subprocess.run(argv,capture_output=True,text=True,env=env,cwd=root,timeout=900)
out={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'argv':argv,'wall_seconds':time.monotonic()-started,'native_work':0}
with (e/'planning/all_fit_launch_20261016_v1/PORT5_PROJECTION_RUN_01.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))'''.replace('ROOT_VALUE',repr(remote_control.REMOTE_ROOT)).replace('INPUTS_VALUE',repr(inputs))
transport_code='import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(base64.b64encode(zlib.compress(code.encode())).decode())+')))'
assert len(transport_code.encode())<5000
out=json.loads(remote_control.call(transport_code,timeout=950))
(HERE/'PORT5_PROJECTION_RUN_01.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out));raise SystemExit(out['returncode'])
