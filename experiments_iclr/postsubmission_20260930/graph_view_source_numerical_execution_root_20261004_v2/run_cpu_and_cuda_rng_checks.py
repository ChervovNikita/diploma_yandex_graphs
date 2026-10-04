"""Exact v2 CPU checks plus a tiny CUDA device-alias RNG correctness check."""
from pathlib import Path
import importlib.util

HERE=Path(__file__).resolve().parent
P=HERE.parent
previous=P/'graph_view_source_numerical_execution_root_20261004_v1/run_cpu_checks.py'
spec=importlib.util.spec_from_file_location('retained_cpu_source_transport_v1',previous)
client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
client.HERE=HERE
client.SOURCE=P/'accuracy_first_graph_view_source_preparation_20261004_v2'
client.MANIFEST='eb34fcb8bc406219007ee39ee9789f1de9bc56acf3cb16164aeab95b4493752f'

CUDA_CODE=r'''
from pathlib import Path
from types import SimpleNamespace
import json,sys,torch,numpy as np
source=Path.cwd()/'experiments_iclr/postsubmission_20260930/accuracy_first_graph_view_source_preparation_20261004_v2'
sys.path.insert(0,str(source))
from runtime import verify_package
verify_package('eb34fcb8bc406219007ee39ee9789f1de9bc56acf3cb16164aeab95b4493752f',
               '55f4146b450f2e04c518e07df9a118bb21f7dc65e18dc659eb34e6dbe8edfb46')
from bank import rng_state,restore_rng,seed_all
assert torch.cuda.device_count()==1
rt=SimpleNamespace(torch=torch,numpy=np)
seed_all(rt,17,'cuda')
results=[]
for device in ('cuda',torch.device('cuda'),'cuda:0'):
 state=rng_state(rt,device)
 assert state['selected_device']==dict(type='cuda',index=0,canonical='cuda:0')
 assert state['cuda'] is not None
 expected=torch.rand(16,device='cuda')
 restore_rng(rt,'cuda:0',state)
 actual=torch.rand(16,device='cuda')
 assert torch.equal(actual,expected)
 restore_rng(rt,torch.device('cuda'),state)
 assert torch.equal(torch.cuda.get_rng_state(0),state['cuda'])
 results.append(dict(device_spelling=str(device),normalized='cuda:0',next_draw_equal=True,raw_state_restored=True))
print(json.dumps(dict(status='PASSED',cases=results,torch_version=torch.__version__,
 peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
 model_forward_or_training_updates=0,real_data_read=False)))
'''

client.REMOTE=client.REMOTE.replace('graph_view_source_numerical_execution_root_20261004_v1',HERE.name).replace(
    'accuracy_first_graph_view_source_preparation_20261004_v1','accuracy_first_graph_view_source_preparation_20261004_v2')
client.REMOTE='CUDA_CHECK_CODE='+repr(CUDA_CODE)+'\n'+client.REMOTE
client.REMOTE=client.REMOTE.replace("value=dict(UTC=datetime.now(timezone.utc).isoformat()", """cuda_check=None
if r.returncode==0:
 gpu_env=dict(env,CUDA_VISIBLE_DEVICES=UUID)
 g=subprocess.run([str(repo/'.venv/bin/python'),'-B','-c',CUDA_CHECK_CODE],cwd=repo,env=gpu_env,capture_output=True,text=True,timeout=60)
 cuda_check=dict(exit_code=g.returncode,stdout=g.stdout,stderr=g.stderr,co_resident_with_original_Amazon_queue=True)
 if g.returncode==0:cuda_check['result']=json.loads(g.stdout)
value=dict(UTC=datetime.now(timezone.utc).isoformat()""")
client.REMOTE=client.REMOTE.replace("with (root/'NUMERICAL_CHECKS.json').open('x')", "value['CUDA_device_alias_RNG_check']=cuda_check\nwith (root/'NUMERICAL_CHECKS.json').open('x')")

if __name__=='__main__':client.main()
