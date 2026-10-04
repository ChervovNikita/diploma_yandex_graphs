"""Install only the sealed repo overlay; no numerical/data execution."""
from pathlib import Path
import base64
import hashlib
import importlib.util
import json
import zlib

PHASE = Path(__file__).resolve().parents[1]
HERE = PHASE / 'pencil_repo_owned_dependency_install_execution_root_20261004_v1'
HERE.mkdir(exist_ok=False)
SOURCE = PHASE / 'pencil_repo_owned_dependency_install_plan_20261004_v1'
EXPECTED = 'ff60385484e771ec33a8749ecd7c87ce95888ab27a1fcc002107b0c0c3545b3d'
sha = lambda raw: hashlib.sha256(raw).hexdigest()
manifest = SOURCE / 'MANIFEST.json'
assert sha(manifest.read_bytes()) == EXPECTED
rows = json.loads(manifest.read_text())['files']
payload = []
for row in rows:
    path = SOURCE / row['path']
    raw = path.read_bytes()
    assert sha(raw) == row['sha256'] and len(raw) == row['bytes']
    payload.append(dict(path=str(path.relative_to(PHASE)), sha256=sha(raw),
                        data=base64.b64encode(raw).decode()))
payload.append(dict(path=str(manifest.relative_to(PHASE)), sha256=EXPECTED,
                    data=base64.b64encode(manifest.read_bytes()).decode()))

driver = r'''from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys
root=Path(__file__).resolve().parent
repo=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
phase=repo/'experiments_iclr/postsubmission_20260930'
source=phase/'pencil_repo_owned_dependency_install_plan_20261004_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):
 with (root/name).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
def utc():return datetime.now(timezone.utc).isoformat()
result={'UTC_started':utc(),'numerical_execution':False,'dataset_access':False,'global_environment_changed':False}
try:
 assert Path.cwd()==repo and os.uname().nodename=='peptide'
 assert sha(source/'MANIFEST.json')=='ff60385484e771ec33a8749ecd7c87ce95888ab27a1fcc002107b0c0c3545b3d'
 for row in json.loads((source/'MANIFEST.json').read_text())['files']:
  assert sha(source/row['path'])==row['sha256']
 plan=json.loads((source/'DEPENDENCY_INSTALL_PLAN.json').read_text())
 python=plan['interpreter']['path'];assert sha(python)==plan['interpreter']['sha256']
 assert sha(source/plan['requirements_file'])==plan['requirements_sha256']
 target=Path(plan['target']);assert target.is_relative_to(repo) and not target.exists()
 lock=target.parent/'INSTALL_SPENT_ATTEMPT.json';target.parent.mkdir(parents=True,exist_ok=True)
 with lock.open('x') as stream:json.dump({'UTC':utc(),'driver':str(root/'install_driver.py')},stream)
 env=os.environ.copy();env.update(plan['installation_environment']);env['TMPDIR']=str(root/'pip_tmp')
 (root/'pip_tmp').mkdir()
 new={r['name'] for r in plan['new_packages']}
 expected={k:v for k,v in plan['complete_expected_distribution_versions'].items() if k not in new}
 probe='import importlib.metadata as m,json; names='+repr(list(expected))+';print(json.dumps({n:m.version(n) for n in names}))'
 before=subprocess.run([python,'-B','-c',probe],env=env,capture_output=True,text=True,check=True)
 observed=json.loads(before.stdout);assert observed==expected,(observed,expected)
 save('BASE_BEFORE.json',observed)
 with (root/'PIP_STDOUT.txt').open('xb') as out,(root/'PIP_STDERR.txt').open('xb') as err:
  pip=subprocess.run(plan['proposed_install_argv'],cwd=repo,env=env,stdout=out,stderr=err,timeout=900)
 result['pip_exit_code']=pip.returncode
 assert pip.returncode==0,'Pinned pip installation failed; preserve partial target, no automatic retry'
 after=subprocess.run([python,'-B','-c',probe],env=env,capture_output=True,text=True,check=True)
 assert json.loads(after.stdout)==observed
 save('BASE_AFTER.json',json.loads(after.stdout))
 env['PYTHONPATH']=plan['proposed_work_environment_PYTHONPATH']
 names=plan['complete_expected_distribution_versions'];modulemap=plan['module_map_for_future_admission']
 probe='import importlib.metadata as m,importlib.util as u,json; names='+repr(list(names))+'; mods='+repr(modulemap)+'; print(json.dumps({"versions":{n:m.version(n) for n in names},"origins":{n:{"origin":u.find_spec(mod).origin,"search_locations":list(u.find_spec(mod).submodule_search_locations or [])} for n,mod in mods.items()}}))'
 admission=subprocess.run([python,'-B','-c',probe],cwd=repo,env=env,capture_output=True,text=True,check=True)
 admitted=json.loads(admission.stdout);assert admitted['versions']==names
 for item in admitted['origins'].values():
  paths=([item['origin']] if item['origin'] else [])+item['search_locations']
  assert paths and all(Path(path).resolve().is_relative_to(target) for path in paths)
 save('DISTRIBUTION_ADMISSION.json',admitted)
 inventory=[]
 for path in sorted(target.rglob('*')):
  assert not path.is_symlink(),'Unexpected overlay symlink'
  if path.is_file():inventory.append({'path':str(path.relative_to(target)),'bytes':path.stat().st_size,'sha256':sha(path)})
 save('INSTALLED_FILE_INVENTORY.json',inventory)
 result.update(status='COMPLETE_REPO_OVERLAY_INSTALL_ONLY',overlay=str(target),installed_files=len(inventory),installed_bytes=sum(r['bytes'] for r in inventory),inventory_sha256=sha(root/'INSTALLED_FILE_INVENTORY.json'),admission_sha256=sha(root/'DISTRIBUTION_ADMISSION.json'),new_packages=16,core_versions_unchanged=True)
except BaseException as exc:
 result.update(status='FAILED_NO_AUTOMATIC_RETRY',error_type=type(exc).__name__,error=str(exc))
finally:
 result['UTC_completed']=utc();save('INSTALL_RESULT.json',result)
sys.exit(0 if result['status']=='COMPLETE_REPO_OVERLAY_INSTALL_ONLY' else 1)
'''
local_driver = HERE / 'install_driver.py'
local_driver.write_text(driver)
payload.append(dict(path=str(local_driver.relative_to(PHASE)), sha256=sha(driver.encode()),
                    data=base64.b64encode(driver.encode()).decode()))
spec = importlib.util.spec_from_file_location('owned_relay', PHASE / 'ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py')
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)
relay.HERE = HERE
encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode(),9)).decode()
code = '''from pathlib import Path
import base64,hashlib,json,os,zlib
repo=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
assert Path.cwd()==repo and os.uname().nodename=='peptide'
phase=repo/'experiments_iclr/postsubmission_20260930'
rows=json.loads(zlib.decompress(base64.b64decode(ENCODED)))
for row in rows:
 path=phase/row['path'];assert path.resolve().is_relative_to(phase)
 raw=base64.b64decode(row['data']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
 if path.exists():assert not path.is_symlink() and path.read_bytes()==raw
 else:
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as stream:stream.write(raw)
print(json.dumps({'status':'STAGED_EXACT_INSTALLER_ONLY','files':len(rows)}))
'''.replace('ENCODED',repr(encoded))
staged = relay.run('pencil_overlay_v1_stage_20261004',code)
(HERE/'STAGE_RECEIPT.json').write_text(json.dumps(staged,indent=2)+'\n')
admission = dict(user_task_authorizes_repo_environment=True,source_manifest_sha256=EXPECTED,
                 installation_only=True,scientific_execution_authorized=False,
                 automatic_retry=False,driver_sha256=sha(driver.encode()))
(HERE/'ROOT_INSTALL_ADMISSION.json').write_text(json.dumps(admission,indent=2)+'\n')
code = '''from pathlib import Path
import hashlib,json,os,subprocess
repo=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
assert Path.cwd()==repo and os.uname().nodename=='peptide'
root=repo/'experiments_iclr/postsubmission_20260930/pencil_repo_owned_dependency_install_execution_root_20261004_v1'
assert not (root/'DETACHED_LAUNCH.json').exists() and not (root/'INSTALL_RESULT.json').exists()
assert hashlib.sha256((root/'install_driver.py').read_bytes()).hexdigest()==@@DRIVER_SHA@@
with (root/'ROOT_INSTALL_ADMISSION.json').open('x') as stream:json.dump(@@ADMISSION@@,stream,indent=2)
with (root/'DRIVER_STDOUT.txt').open('xb') as out,(root/'DRIVER_STDERR.txt').open('xb') as err:
 child=subprocess.Popen(['/usr/bin/python3','-I','-S','-B',str(root/'install_driver.py')],cwd=repo,stdout=out,stderr=err,start_new_session=True)
 stat=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(') ',1)[1].split()
 receipt={'pid':child.pid,'start_ticks':int(stat[19]),'root':str(root),'installation_only':True,'new_packages':16}
 with (root/'DETACHED_LAUNCH.json').open('x') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps(receipt))
'''.replace('@@DRIVER_SHA@@',repr(sha(driver.encode()))).replace('@@ADMISSION@@',repr(admission))
compile(driver, 'install_driver.py', 'exec')
compile(code, 'remote_launch.py', 'exec')
launched = relay.run('pencil_overlay_v1_launch_20261004',code)
(HERE/'DETACHED_LAUNCH.json').write_text(json.dumps(launched,indent=2)+'\n')
print(json.dumps(launched))
