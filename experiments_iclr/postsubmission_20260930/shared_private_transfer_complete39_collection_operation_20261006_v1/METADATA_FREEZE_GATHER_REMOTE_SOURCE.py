
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,socket,subprocess
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split();assert uuids==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
root=phase/'shared_private_transfer_complete39_collection_execution_root_20261006_v1';names=json.loads('["COLLECTION_RELEASE.json", "ATTEMPT_HISTORY.json", "COLLECTION_FREEZE.json", "provider_custody_b0_S_end_joint.json", "provider_custody_b0_J4_end_joint.json", "provider_custody_b0_E_end_joint.json", "provider_custody_b0_E_end_live.json", "provider_custody_b0_E_end_detached.json", "provider_custody_b0_S_end_live.json", "provider_custody_b0_S_end_detached.json", "provider_custody_b0_U_end_live.json", "provider_custody_b0_E_random_live.json", "provider_custody_b0_E_random_detached.json", "provider_custody_b1_E_random_detached.json", "provider_custody_b1_E_random_live.json", "provider_custody_b1_U_end_live.json", "provider_custody_b1_S_end_detached.json", "provider_custody_b1_S_end_live.json", "provider_custody_b1_E_end_detached.json", "provider_custody_b1_E_end_live.json", "provider_custody_b1_E_end_joint.json", "provider_custody_b1_J4_end_joint.json", "provider_custody_b1_S_end_joint.json", "provider_custody_b2_S_end_live.json", "provider_custody_b2_S_end_detached.json", "provider_custody_b2_U_end_live.json", "provider_custody_b2_E_random_live.json", "provider_custody_b2_E_random_detached.json", "provider_custody_b2_S_end_joint.json", "provider_custody_b2_J4_end_joint.json", "provider_custody_b2_E_end_joint.json", "provider_custody_b2_E_end_live.json", "provider_custody_b2_E_end_detached.json", "provider_custody_b0_F1_end_joint.json", "provider_custody_b0_F1_end_live.json", "provider_custody_b0_F1_end_detached.json", "provider_custody_b1_F1_end_detached.json", "provider_custody_b1_F1_end_live.json", "provider_custody_b1_F1_end_joint.json", "provider_custody_b2_F1_end_joint.json", "provider_custody_b2_F1_end_live.json", "provider_custody_b2_F1_end_detached.json"]');order=json.loads('["b0_S_end_joint", "b0_J4_end_joint", "b0_E_end_joint", "b0_E_end_live", "b0_E_end_detached", "b0_S_end_live", "b0_S_end_detached", "b0_U_end_live", "b0_E_random_live", "b0_E_random_detached", "b1_E_random_detached", "b1_E_random_live", "b1_U_end_live", "b1_S_end_detached", "b1_S_end_live", "b1_E_end_detached", "b1_E_end_live", "b1_E_end_joint", "b1_J4_end_joint", "b1_S_end_joint", "b2_S_end_live", "b2_S_end_detached", "b2_U_end_live", "b2_E_random_live", "b2_E_random_detached", "b2_S_end_joint", "b2_J4_end_joint", "b2_E_end_joint", "b2_E_end_live", "b2_E_end_detached", "b0_F1_end_joint", "b0_F1_end_live", "b0_F1_end_detached", "b1_F1_end_detached", "b1_F1_end_live", "b1_F1_end_joint", "b2_F1_end_joint", "b2_F1_end_live", "b2_F1_end_detached"]')
assert root.resolve(strict=True).is_relative_to(phase) and not root.is_symlink() and len(names)==42
assert {p.name for p in root.iterdir()}==set(names)
freeze=root/'COLLECTION_FREEZE.json';registry=json.loads(freeze.read_text())
assert registry['schema']=='authenticated_complete_prospective_allocation_replication39_v1' and registry['complete'] is True and registry['full39_terminal_source_artifact_custody_passed'] is True
assert registry['selected_logical_scientific_fits']==39 and registry['selected_physical_fits']==39 and registry['new_physical_fits']==26
assert [r['cell_id'] for r in registry['completed']]==order
for key in ('comparative_scoring_performed','TEST_access','fits_authorized','history_bytes_observed','history_JSON_parsed','FREEZE_JSON_parsed','CONFIG_JSON_parsed','selected_prediction_payloads_deserialized','quality_fields_accessed_or_emitted','automatic_retry'):assert registry[key] is False
assert registry['collection_release_sha256']=='b9375929fcc458d21a3309757b4ff515d26eaf37d6edc3ff32a70159a03b065e'
assert registry['collector_manifest_sha256']=='c29ccad053cfcaa4c11499854fdaca962ade48a910efa82435241d7d2f3e2ec6'
assert hashlib.sha256((root/'COLLECTION_RELEASE.json').read_bytes()).hexdigest()==registry['collection_release_sha256']
original_history=phase/'shared_private_transfer_allocation_replication_preparation_20261005_v2/ATTEMPT_HISTORY.json'
assert hashlib.sha256(original_history.read_bytes()).hexdigest()==hashlib.sha256((root/'ATTEMPT_HISTORY.json').read_bytes()).hexdigest()=='e3c70e8b22f71075078f376a82ec2534cb89d07cd508c4e2a2361dc2c0484970'
for name in names:
 p=root/name;assert p.suffix=='.json' and p.name not in ('FREEZE.json','CONFIG.json','VALID_HISTORY.jsonl') and p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root) and p.stat().st_size<1_000_000
files=[]
for name in names:
 p=root/name;before=hashlib.sha256(p.read_bytes()).hexdigest();mode_before=oct(p.stat().st_mode&0o777)
 p.chmod(0o444)
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==before and p.stat().st_mode&0o777==0o444
 files.append({'path':str(p.relative_to(phase)),'bytes':len(b),'sha256':before,'mode_before':mode_before,'mode':'0444','utf8':b.decode()})
print(json.dumps({'schema':'actual_complete39_collection_metadata_freeze_and_gather_v1','UTC':datetime.now(timezone.utc).isoformat(),'hostname':socket.gethostname(),'GPU_UUIDs':uuids,
 'repository':str(repo),'files':files,'JSON_metadata_file_count':42,'complete39_registry':{'path':str(freeze.relative_to(phase)),'sha256':hashlib.sha256(freeze.read_bytes()).hexdigest()},
 'original_UNKNOWN_history_preserved':True,'scientific_artifact_or_directory_chmod':False,'history_or_payload_semantics_tensor_score_held_D2_model_or_fit_execution':False,'GPU_jobs_launched_or_stopped':0}))
