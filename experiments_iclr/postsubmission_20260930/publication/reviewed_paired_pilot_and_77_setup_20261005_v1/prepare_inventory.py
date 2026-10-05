"""Prepare metadata-only publication inventory; never launch or publish anything."""
from pathlib import Path
import ast
import hashlib
import json
from datetime import datetime,timezone

PHASE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
HEAD='0788c2ba88f4d68823e0767dc7c3f21c6851ae33'
PREFIX='experiments_iclr/postsubmission_20260930/'
BRANCH='codex/postsubmission-research-20260930'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,obj):
 path=OUT/name
 assert not path.exists(),name
 path.write_text(json.dumps(obj,indent=2)+'\n')

def check_packet(folder,manifest_name,pinned):
 root=PHASE/folder;manifest=root/manifest_name
 assert sha(manifest)==pinned,(folder,'manifest changed')
 payload=json.loads(manifest.read_text())
 for row in payload['files']:
  path=root/row['path']
  assert path.is_file() and not path.is_symlink()
  assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],str(path)
 return {'path':str(manifest.relative_to(PHASE)),'sha256':pinned,'payloads_verified':len(payload['files'])}

bindings=[
 check_packet('shared_private_transfer_paired_pilot_preparation_20261005_v1','SOURCE_MANIFEST.json','e609add68a00d45f5c580d3da50e0fd99d3294406abc32a2e355a7ea3f5879c4'),
 check_packet('shared_private_transfer_paired_pilot_preparation_20261005_v2','SOURCE_MANIFEST.json','1b52c20c4228cfb34a7667de056333a60c3e30e766a96a2949b3659382a63579'),
 check_packet('shared_private_transfer_gpu77_portability_preparation_20261005_v2','MANIFEST.json','fd9fb425604b0905887071c905c2a27935a44bd8739e1ff2d7548539efb5f36d'),
 check_packet('shared_private_transfer_gpu77_environment_execution_20261005_v1','MANIFEST.json','14b6a9106a018d578f7a815fc5ab77c0d0b7eef415cee86baf2d7a2f3d1928c6'),
]
for folder,expected in [('private_transfer_pilot_independent_execution_audit_20261005_v1','7a441485ffdd033953379425830de3c116dc64409614c4098b76089b8c9a2c9e'),('private_transfer_pilot_independent_execution_audit_20261005_v2','0050a8c4875e24fb999ac8b86f348d67ae56e970c19f8d06ab9cad3f68e84517')]:
 bindings.append(check_packet(folder,'REVIEW_MANIFEST.json',expected))
 seal=json.loads((PHASE/folder/'REVIEW_SEAL.json').read_text())
 assert seal['review_manifest_sha256']==expected
assert sha(PHASE/'shared_private_transfer_paired_pilot_launch_receipts_root_20261005_v2/stage_and_launch.py')=='ad9b71832e3da8891bec7720547a9bf50dc87b55bc3d207f4fb8b914522c892a'

published={}
for receipt in (PHASE/'publication').glob('*/COMMIT_RECEIPT.json'):
 meta=json.loads(receipt.read_text())
 if meta.get('branch')!=BRANCH or not meta.get('commit'): continue
 inv=receipt.parent/'INVENTORY.json'
 if not inv.exists():continue
 for row in json.loads(inv.read_text()).get('files',[]):
  published.setdefault(row['target'],set()).add(row['sha256'])
assert json.loads((PHASE/'publication/private_transfer_cost_prior_update_20261005_v1/COMMIT_RECEIPT.json').read_text())['commit']==HEAD

selected=set()
def add(rel):
 path=PHASE/rel
 assert path.is_file() and not path.is_symlink(),rel
 assert path.resolve().is_relative_to(PHASE)
 selected.add(rel)
def allfiles(folder):
 for path in sorted((PHASE/folder).rglob('*')):
  if path.is_file():add(str(path.relative_to(PHASE)))
for folder in ['shared_private_transfer_paired_pilot_preparation_20261005_v1','shared_private_transfer_paired_pilot_preparation_20261005_v2','private_transfer_pilot_independent_execution_audit_20261005_v1','private_transfer_pilot_independent_execution_audit_20261005_v2','shared_private_transfer_gpu77_portability_preparation_20261005_v2','private_transfer_complete_cost_root_adoption_20261005_v1','shared_private_transfer_paired_pilot_execution_root_20261005_v2']:
 allfiles(folder)
env='shared_private_transfer_gpu77_environment_execution_20261005_v1/'
for name in ['REPORT.md','MANIFEST.json','SEAL.json','CORE_ORIGINS.json','CORE_REQUIREMENTS.txt','ENVIRONMENT_VERIFICATION.json','SETUP_PHYSICAL_TERMINAL.json','INSTALLER_START.json','INSTALLER_TERMINAL.json','ROOT_SETUP_AUTHORIZATION.json','CONDA_EXPLICIT.txt','PIP_FREEZE.txt','PROVIDERS.json','SOURCE_STAGE_INVENTORY.json','SOURCE_STAGE_RECEIPT.json','INPUT_STAGE_INVENTORY.json','INPUT_STAGE_RECEIPT.json','SOURCE_TRANSFER_ARCHIVE.json','bounded_install.py','collect_providers.py','SETUP_COMMANDS.sh.txt']:
 add(env+name)
launch='shared_private_transfer_paired_pilot_launch_receipts_root_20261005_v2/'
for name in ['LAUNCH_RECEIPT.json','ROOT_RELEASE.json','PROVIDER_ADMISSION.json','DONOR_REGISTRATION.json','TRANSFER_INVENTORY.json','ROOT_REVIEW.md','stage_and_launch.py','monitor_metadata.py','observation_20261005T055257Z/OBSERVATION.json','observation_20261005T055509Z/OBSERVATION.json']:
 add(launch+name)
obs='shared_backbone_private_transfer_complete_cost_execution_root_20261005_v3/observation_005/'
for folder in ['02_capable_single_live_transfer','03_untied4_live_transfer','04_ordinary_native4_ordinary_joint','05_shared_f4_ordinary_joint']:
 for name in ['CHILD_STARTED.json','EXECUTION_RECEIPT.json','result/COST_RESULT.json']:
  add(obs+folder+'/'+name)
for name in ['PUSH_ACK.json','PUSH_RECEIPT.json']:
 add('publication/private_transfer_cost_prior_update_20261005_v1/'+name)
for rel in ['amazon_polynormer_paired_family_execution_root_20261003_v3/v6_queue_launch_execution_receipts_20261003_v2/MONITOR_0066_RESULT.json', 'pencil_citeseer_native300_launch_receipts_20261005_v1/metadata_20261005T053242Z/COMPACT_METADATA.json', 'pencil_citeseer_native300_launch_receipts_20261005_v1/physical_20261005T052008Z/OBSERVATION.json']:
 add(rel)
for rel in ['PUBLIC_STATUS.md','RESEARCH_STATE.md','research_ledger.json']:
 add(rel)
for name in ['fetch_frozen_b0_metadata.py','METADATA_FETCH_RECEIPT.json','STATUS_EDIT_RECEIPT.json','prepare_inventory.py']:
 add(str((OUT/name).relative_to(PHASE)))

excluded=['original datasets/data archives/tensors/checkpoints','raw singleton FP32 RESULT.json (2,051,914 bytes; retained in server evidence and excluded by publisher <2,000,000-byte limit)','already published unchanged qualified training source v1/v2 and prior-work decision packets','duplicate transport/POLL/installation-report payloads from completed 77 setup','mutable runtime logs, OBSERVATION_LATEST_COMPACT.txt and ongoing sampling/qualification artifacts','credential values or key content','original manuscript score changes and new scientific execution']
notes=OUT/'INVENTORY_NOTES.md'
assert not notes.exists()
notes.write_text('''# Publication scope: reviewed private-transfer pilot and completed 18.77 setup\n\nThis packet records completed source review, prospective scientific decisions, a verified control-first singleton launch and completed repository-local runtime installation. It establishes no predictive success, methodological novelty or acceptance. Root owns publication approval; this preparation performs no commit or push.\n\n## Included\n\n- Both immutable pilot preparations and both independent execution audits, including the preserved v1 material findings and reviewed v2 repairs.\n- The exact full 30-cell prospective plan, all jobs, external anchors, b0 queue/release/admission and small physical launch observations. No scientific result payload from an in-progress fit is included.\n- The completed full-TRAIN control costs and frozen numeric horizon. The original candidate supervisor failure remains disclosed, with its completed cycle used for budgeting only.\n- Completed 18.77 portability and environment origins, pinned versions, staging hashes and physical terminal records. This is a selected compact subset of the sealed environment packet. Omitted setup transports and large pip installation reports remain bound by the original unchanged MANIFEST and retained in server/project evidence; the subset is not represented as a complete raw setup archive.\n- A narrow current-status update, preserved preceding state, one additive ledger event and main README update. No original score or scientific source is changed.\n\n## Excluded\n\n'''+''.join('- '+x+'\n' for x in excluded)+'\n## Scientific limits\n\nThe first live cell is a capable-single ordinary-joint control. No live-transfer candidate fit or new candidate accuracy result existed at the 05:55:09 UTC physical observation. All 30 fits require complete authenticated provider-specific records before comparative analysis. The 18.77 runtime setup does not itself admit sampling, the finite FP32 qualification or quality fits. Separate CPU transcripts have launch receipts, but their mutable qualification/sampling artifacts are excluded from this publication. Existing ordinary independent-four and native-training unframed-F4 scores are reused verbatim. J4 is jointly trained; it is not the independently trained reference.\n\nSELAR already supplies virtual/meta/recomputed-commit scheduling. This study tests the endpoint-conditioned ensemble rule; persistence and recomputation are not claimed as methodological novelty. A development lead would still need fresh confirmation.\n')
add(str(notes.relative_to(PHASE)))

rows=[];deduplicated=[]
for rel in sorted(selected):
 path=PHASE/rel
 assert path.stat().st_size<2_000_000,rel
 if path.suffix=='.py':ast.parse(path.read_text())
 if path.suffix=='.json':json.loads(path.read_text())
 target=PREFIX+rel
 record={'source':rel,'target':target,'bytes':path.stat().st_size,'sha256':sha(path)}
 if target in published and record['sha256'] in published[target] and rel not in ['PUBLIC_STATUS.md','RESEARCH_STATE.md','research_ledger.json']:
  deduplicated.append(record);continue
 rows.append(record)
readme=OUT/'README_MAIN.md'
rows.append({'source':str(readme.relative_to(PHASE)),'target':'README.md','bytes':readme.stat().st_size,'sha256':sha(readme)})
assert len({r['target'] for r in rows})==len(rows)
write('PACKET_BINDINGS.json',{'schema':'compact_publication_packet_bindings_v1','bindings':bindings,'source_science_or_scores_changed':False})
binding_path=OUT/'PACKET_BINDINGS.json'
rows.append({'source':str(binding_path.relative_to(PHASE)),'target':PREFIX+str(binding_path.relative_to(PHASE)),'bytes':binding_path.stat().st_size,'sha256':sha(binding_path)})
write('INVENTORY.json',{'branch':BRANCH,'expected_head':HEAD,'message':'Preserve reviewed endpoint-transfer pilot launch and 18.77 runtime setup','files':rows,'remove':[]})
write('PREPARATION_RECEIPT.json',{'UTC':datetime.now(timezone.utc).isoformat(),'inventory_sha256':sha(OUT/'INVENTORY.json'),'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'deduplicated_unchanged_already_published':deduplicated,'excluded_categories':excluded,'source_packet_bindings_verified':True,'JSON_and_Python_AST_static_checks_passed':True,'models_or_statistics_executed':False,'commit_or_push':False})
print(json.dumps({'inventory':str((OUT/'INVENTORY.json').relative_to(PHASE)),'sha256':sha(OUT/'INVENTORY.json'),'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'deduplicated':len(deduplicated)}))
