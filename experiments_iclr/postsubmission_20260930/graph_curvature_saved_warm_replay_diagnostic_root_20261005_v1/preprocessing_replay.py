"""One bounded native preprocessing replay; no model fit or evaluation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
ROUTE = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
REMOTE_PHASE = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930'
PYTHON = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/bin/python'
RUNNER = 'graph_curvature_selector_exact_native_qualification_source_20261004_v2/runner.py'

def main():
    destination = HERE/'PREPROCESSING_REPLAY_RESULT.json'
    assert not destination.exists()
    runner_sha = hashlib.sha256((PHASE/RUNNER).read_bytes()).hexdigest()
    code = 'RUNNER='+repr(RUNNER)+'\nEXPECTED_SHA='+repr(runner_sha)+'\n'+r'''
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys,time
phase=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
assert Path.cwd()==phase and os.environ['GNNM_SSH_DESTINATION']=='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10).stdout.strip().split(',')
assert len(gpu)==2 and gpu[0].strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac' and int(gpu[1])>=32768
started=time.perf_counter()
import numpy as np
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
path=phase/RUNNER;data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==EXPECTED_SHA
spec=importlib.util.spec_from_file_location('saved_warm_preprocessing_probe',path)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;exec(compile(data,str(path),'exec'),m.__dict__)
x,e,labels,binding=m.load_declared_inputs('Squirrel',np,torch,'cuda:0')
del labels
driver,selector=m.load_driver();sources=driver.load_sources();api=sources['native_adapter']
prepared=[];costs=[]
for i in range(2):
 torch.cuda.synchronize();wall=time.perf_counter()
 g=api.prepare_graph(x,e,driver.CELLS['Squirrel'],dict(probe='two_exact_repetitions'),dict(graph_input=binding['graph_input']))
 torch.cuda.synchronize();costs.append(time.perf_counter()-wall)
 prepared.append((g.teacher_input.detach().cpu(),g.teacher_edge_index.detach().cpu()))
 del g
def digest(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
a,b=prepared[0][0],prepared[1][0]
result=dict(schema='native-preprocessing-replay-v1',equal_teacher_input=torch.equal(a,b),equal_teacher_edges=torch.equal(prepared[0][1],prepared[1][1]),different_elements=int((a!=b).sum()),max_abs_difference=float((a-b).abs().max()),teacher_input_shape=list(a.shape),teacher_input_hashes=[digest(pair[0]) for pair in prepared],raw_feature_hash=digest(x.cpu()),raw_edge_hash=digest(e.cpu()),preprocessing_seconds=costs,wall_seconds=time.perf_counter()-started,torch=torch.__version__,runner_sha256=EXPECTED_SHA,source_binding=binding,new_training=False,new_model_forward=False,heldout_labels_read=False,predictive_values_read=False,signals_sent=False)
print(json.dumps(result))
'''
    command = 'cd '+shlex.quote(REMOTE_PHASE)+' && '+shlex.join(['env','GNNM_SSH_DESTINATION='+ROUTE,'PYTHONDONTWRITEBYTECODE=1','CUBLAS_WORKSPACE_CONFIG=:4096:8','NVIDIA_TF32_OVERRIDE=0','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','/usr/bin/timeout','--signal=TERM','--kill-after=5s','120s',PYTHON,'-B','-'])
    started = datetime.now(timezone.utc).isoformat()
    r = subprocess.run(['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',ROUTE,command],input=code,capture_output=True,text=True,timeout=150)
    receipt = dict(start_UTC=started,terminal_UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stderr=r.stderr,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),executed_code_sha256=hashlib.sha256(code.encode()).hexdigest(),route=ROUTE,bounded_server_timeout_seconds=120)
    if r.returncode == 0:receipt['result']=json.loads(r.stdout)
    else:receipt['stdout']=r.stdout
    with destination.open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps({k:receipt.get(k) for k in ('start_UTC','terminal_UTC','exit_code')}))
    if 'result' in receipt:print(json.dumps({k:receipt['result'][k] for k in ('equal_teacher_input','equal_teacher_edges','different_elements','max_abs_difference','wall_seconds')}))
    raise SystemExit(r.returncode)

if __name__ == '__main__':
    main()
