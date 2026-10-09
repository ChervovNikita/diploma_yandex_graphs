from pathlib import Path
import socket,subprocess,json,hashlib,os,importlib.util,traceback
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'bsnn_richer_full_input_qualification_activation_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,stdin=subprocess.DEVNULL).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
d=dict(names=sorted(p.name for p in A.iterdir()),numerical_work=False)
for n in ['LAUNCH.json','WORKER_OWNER.json','TERMINAL.json']:
 if (A/n).exists():d[n]=json.loads((A/n).read_text())
if (A/'OWNER.log').exists():d['owner_log']=(A/'OWNER.log').read_text()[-5000:]
try:
 os.chdir(R)
 p=P/'bsnn_richer_cayley_d2_f32_L4_normal_host_launch_source_20261009_v1/SUPERVISOR.py'
 spec=importlib.util.spec_from_file_location('_read_only_bsnn_gate',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 release=A/'ROOT_RELEASE.json';cfg,*rest=m.gate(release,hashlib.sha256(release.read_bytes()).hexdigest(),True)
 d['gate_passed']=True
except BaseException as e:d.update(gate_passed=False,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
print(json.dumps(d),flush=True)
