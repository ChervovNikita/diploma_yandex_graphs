from pathlib import Path
import socket,subprocess,json,hashlib,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';D=P/'label_only_four_bank_first_screen_activation_root_20261008_v1';S=P/'label_only_four_bank_WikiCS_scientific_owner_source_20261008_v1'
assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='3ce6fae0c381093a2de54804e924e7237902ba75'
assert json.loads((D/'PREFLIGHT.json').read_text())['passed'] and not(D/'LAUNCH.json').exists()
assert hashlib.sha256((D/'ROOT_RELEASE.json').read_bytes()).hexdigest()=='2595992d33567549573ff605da3cb5119958699d4d8097082ce9af54a8fbe7d6'
argv=['/usr/bin/python3','-B',str(S/'launch.py'),'--release',str(D/'ROOT_RELEASE.json'),'--release-sha256','2595992d33567549573ff605da3cb5119958699d4d8097082ce9af54a8fbe7d6','--authorized']
r=subprocess.run(argv,cwd=R,capture_output=True,text=True,timeout=35)
x={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'argv':argv,'route_guard_verified':True,'automatic_retry':False}
(D/'LAUNCH_EXECUTION_RECEIPT.json').write_text(json.dumps(x,indent=2)+'\n')
assert r.returncode==0,r.stderr
x['launch']=json.loads((D/'LAUNCH.json').read_text());print(json.dumps(x))
