from pathlib import Path
import hashlib,json,socket,subprocess,platform,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'private_sheaf_baseline_screen_activation_root_20261009_v1';D=P/'private_sheaf_baseline_screen_execution_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert not Path('/proc/545332').exists() and not Path('/proc/545335').exists()
def bind(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return {'path':str(f),'sha256':h.hexdigest(),'bytes':f.stat().st_size}
out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoints':[],'receipt_bindings':{},'runtime':json.loads((D/'RUNTIME_AND_ROLES.json').read_text()),'python':platform.python_version(),'roles':bind(P/'private_sheaf_train_valid_roles_allocation_root_20261009_v1/roles.npz')}
for config,epoch in [('d4_f16_L2',405),('d4_f16_L4',372)]:
 d=D/('native_single__'+config+'__seed2207');x=json.loads((d/'RESULT.json').read_text());assert x['status']=='failed' and x['epochs_completed']==500 and x['selected_epoch']==epoch
 out['checkpoints'].append(dict(config_id=config,seed=2207,selected_epoch=epoch,**bind(d/'SELECTED_STATE.pt'),original_result_path=str(d/'RESULT.json'),original_result_sha256=bind(d/'RESULT.json')['sha256']))
for name,rel in [('closed_native15','private_sheaf_baseline_screen_activation_root_20261009_v1/TERMINAL.json'),('original_screen_release','private_sheaf_baseline_screen_activation_root_20261009_v1/RELEASE.json'),('runtime_qualification','private_sheaf_qualified_dependency_probe_root_20261009_v1/RUNTIME_AND_ROLE_METADATA.json'),('role_custody','private_sheaf_train_valid_roles_allocation_root_20261009_v1/ROLE.json')]:out['receipt_bindings'][name]=bind(P/rel)
print(json.dumps(out))
