from pathlib import Path
import json,socket,subprocess,os,hashlib,time
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'query_value_gate_complete_readout_activation_root_20261009_v1'
assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='a675ffbb5be59de55018bd4eb694207c7717f1e4'
F=P/'query_value_gate_full15_execution_root_20261009_v1'
assert json.loads((F/'FAMILY_CLOSURE.json').read_text())['complete'] and json.loads((F/'PARENT_TERMINAL.json').read_text())['family_complete']
A.mkdir(exist_ok=False)
value=dict(reserved=True,execution_commit='a675ffbb5be59de55018bd4eb694207c7717f1e4',handles=[],automatic_retry=False)
(A/'LAUNCH.json').write_text(json.dumps(value,indent=2)+'\n')
def identity(pid):
 s=Path('/proc',str(pid),'stat').read_text();v=s[s.rfind(')')+2:].split();return dict(pid=pid,start_ticks=int(v[19]),state=v[0],group=int(v[2]),session=int(v[3]))
runtime={'GPU_name': 'NVIDIA A100-SXM4-80GB', 'GPU_uuid': 'GPU-44039938-fd82-41d2-fefd-de71514e2fac', 'PYTHONPATH': ['/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_ncn_dependency_overlay_20261005_v1', '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/lib/python3.11/site-packages'], 'hostname': 'anogena-2-0', 'numpy': '1.26.4', 'phase': '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930', 'python': '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python', 'repository': '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs', 'torch': '2.1.2+cu118'}
for job in [{'name': 'reconstruct15', 'relative': 'query_value_gate_complete_readout_source_20261009_v1/readout_once.py', 'sha256': 'ef717cc982a049a358cb7d29524414440892334a22feb3f9121fe51c169f2de9'}]:
 f=P/job['relative'];assert hashlib.sha256(f.read_bytes()).hexdigest()==job['sha256']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=runtime['GPU_uuid'],PYTHONPATH=os.pathsep.join(runtime['PYTHONPATH']),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1');env.pop('PYTHONHOME',None)
 with (A/(job['name']+'.log')).open('xb') as log:
  child=subprocess.Popen([runtime['python'],'-B',str(f),'--authorized'],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 handle=identity(child.pid);assert handle['group']==handle['session']==child.pid
 value['handles'].append(dict(**job,identity=handle));(A/'LAUNCH.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
