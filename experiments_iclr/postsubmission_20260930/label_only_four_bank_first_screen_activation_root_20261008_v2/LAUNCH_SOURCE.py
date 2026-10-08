from pathlib import Path
import socket,subprocess,json,hashlib,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';D=P/'label_only_four_bank_first_screen_activation_root_20261008_v2';S=P/'label_only_four_bank_WikiCS_scientific_owner_source_20261008_v2'
assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='7ea03c95c7791251b2461fd98bbe125582314ae0'
assert json.loads((D/'PREFLIGHT.json').read_text())['passed'] and not(D/'LAUNCH.json').exists()
assert hashlib.sha256((D/'ROOT_RELEASE.json').read_bytes()).hexdigest()=='412f0b7ec0fcf2ca184ced408f76e97826ba62291c40a4a0ef768b25e7f5a4e4'
argv=['/usr/bin/python3','-B',str(S/'launch.py'),'--release',str(D/'ROOT_RELEASE.json'),'--release-sha256','412f0b7ec0fcf2ca184ced408f76e97826ba62291c40a4a0ef768b25e7f5a4e4','--authorized']
r=subprocess.run(argv,cwd=R,capture_output=True,text=True,timeout=35)
x={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'argv':argv,'route_guard_verified':True,'automatic_retry':False}
(D/'LAUNCH_EXECUTION_RECEIPT.json').write_text(json.dumps(x,indent=2)+'\n')
assert r.returncode==0,r.stderr
x['launch']=json.loads((D/'LAUNCH.json').read_text());print(json.dumps(x))
