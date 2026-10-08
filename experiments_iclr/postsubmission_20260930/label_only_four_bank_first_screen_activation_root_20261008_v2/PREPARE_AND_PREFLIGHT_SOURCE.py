from pathlib import Path
import socket,subprocess,json,base64,hashlib,datetime,importlib.util,sys
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';D=P/'label_only_four_bank_first_screen_activation_root_20261008_v2';S=P/'label_only_four_bank_WikiCS_scientific_owner_source_20261008_v2'
assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(sys.stdin.read());assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==payload['commit'];assert not D.exists();D.mkdir()
for name,value in payload['files'].items():
 assert name in ('ROOT_RELEASE.json','ROOT_REVIEW.json')
 data=base64.b64decode(value['base64']);assert hashlib.sha256(data).hexdigest()==value['sha256'];(D/name).write_bytes(data)
spec=importlib.util.spec_from_file_location('_root_four_bank_preflight',S/'owned.py');owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)
cfg,pins,phase=owned.validate(D/'ROOT_RELEASE.json',payload['files']['ROOT_RELEASE.json']['sha256'],True)
x={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'source_commit':payload['commit'],'owner_manifest_sha256':cfg['owner_manifest_sha256'],'release_sha256':payload['files']['ROOT_RELEASE.json']['sha256'],'route_verified':True,'source_data_engineering_bindings_verified':True,'numerical_modules_imported':False,'training_or_scoring_performed':False}
(D/'PREFLIGHT.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
