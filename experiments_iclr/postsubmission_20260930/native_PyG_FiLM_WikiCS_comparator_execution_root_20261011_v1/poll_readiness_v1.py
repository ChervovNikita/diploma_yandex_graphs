"""Fetch a complete owner result; no model payload or scientific score."""
import base64,importlib.util,json,shlex,zlib
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent
REMOTE=r"""
import base64,json,socket,subprocess,sys,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
H=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_PyG_FiLM_WikiCS_comparator_execution_root_20261011_v1')
def read(name):
 p=H/name;return json.loads(p.read_text()) if p.exists() else None
end=read('readiness_owner_v1/OWNER_END.json');value=dict(owner_start=read('readiness_owner_v1/OWNER_START.json'),owner_end=end,scientific_fit_or_selection=False,TEST_access=False)
if end:
 if end['successful_readiness']:
  value['complete_result']=read('readiness_v1/READINESS_RESULT.json')
  value['config']=read('READINESS_CONFIG_V1.json');value['admission']=read('READINESS_ADMISSION_V1.json');value['runtime']=read('RUNTIME_FINGERPRINT_V1.json')
 else:
  value['failure']=read('readiness_v1/FAILURE.json');value['stderr']=(H/'readiness_owner_v1/stderr.log').read_text()[-4000:]
print(json.dumps(value))
"""
def main():
 spec=importlib.util.spec_from_file_location('payload_transport',P/'publication/publish_exact_inventory_v19.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,'FiLM-readiness','literal-route','closed-result-only'])]
 r=mod.terminal_transport(argv,input=base64.b64encode(zlib.compress(b'{}')).decode(),capture_output=True,text=True,timeout=60);assert r.returncode==0,r.stderr[-1500:]
 value=json.loads(r.stdout);path=H/'READINESS_OBSERVATION_V1.json';assert not path.exists();path.write_text(json.dumps(value,indent=2)+'\n')
 print(json.dumps({k:v for k,v in value.items() if k not in ('config','admission','runtime')}))
if __name__=='__main__':main()
