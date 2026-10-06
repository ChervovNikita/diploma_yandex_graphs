
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,socket,subprocess
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split();assert uuids==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
root=phase/'shared_private_transfer_complete39_history_inventory_execution_root_20261006_v1';names=['HISTORY_INVENTORY_RELEASE.json','VALID_HISTORY_INVENTORY.json']
assert {p.name for p in root.iterdir()}==set(names) and root.resolve(strict=True).is_relative_to(phase) and not root.is_symlink()
value=json.loads((root/'VALID_HISTORY_INVENTORY.json').read_text())
assert value['schema']=='retrospective_allocation_replication_full39_history_inventory_v1' and value['complete'] is True and value['selected_logical_scientific_fits']==39 and value['new_physical_fits']==26
assert value['created_after_all39_terminal_source_artifact_checks'] is True and value['history_or_FREEZE_JSON_parsed'] is False and value['scores_read'] is False and value['pre_fit_authority'] is False
assert value['complete39_registry']=={'path': 'shared_private_transfer_complete39_collection_execution_root_20261006_v1/COLLECTION_FREEZE.json', 'sha256': 'b3b318ed1a64bf73a5e453c4665183e3540fc7f8ef114fa41345922cf4860d25'} and len(value['records'])==39
assert value['inventory_manifest_sha256']=='c29ccad053cfcaa4c11499854fdaca962ade48a910efa82435241d7d2f3e2ec6'
assert hashlib.sha256((root/'HISTORY_INVENTORY_RELEASE.json').read_bytes()).hexdigest()=='d84424c9a4746c24d26a1c96f9bc686fe9ed6038b3d575e2339b244bd6905c78'
files=[]
for name in names:
 p=root/name;assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root) and p.stat().st_size<1_000_000
 b=p.read_bytes();before=hashlib.sha256(b).hexdigest();oldmode=oct(p.stat().st_mode&0o777);p.chmod(0o444)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==before and p.stat().st_mode&0o777==0o444
 files.append({'path':str(p.relative_to(phase)),'bytes':len(b),'sha256':before,'mode_before':oldmode,'mode':'0444','utf8':b.decode()})
assert hashlib.sha256((phase/'shared_private_transfer_allocation_replication_preparation_20261005_v2/ATTEMPT_HISTORY.json').read_bytes()).hexdigest()=='e3c70e8b22f71075078f376a82ec2534cb89d07cd508c4e2a2361dc2c0484970'
print(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'hostname':socket.gethostname(),'GPU_UUIDs':uuids,'files':files,'complete39_history_inventory':{'path':str((root/'VALID_HISTORY_INVENTORY.json').relative_to(phase)),'sha256':files[1]['sha256']},'history_JSON_or_scores_parsed':False,'no77_access_or_GPU_actions':True,'original_UNKNOWN_preserved':True}))
