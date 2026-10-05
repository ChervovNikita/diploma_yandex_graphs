from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent;P=HERE.parent.parent
PREFIX='experiments_iclr/postsubmission_20260930/'
roots=[
 'coordination_snapshots/20261005_allocation_only_continuation_v1',
 'shared_private_transfer_fixed39_d2_analysis_preparation_20261005_v3',
 'shared_private_transfer_fixed39_d2_analysis_independent_review_20261005_v3',
 'shared_private_transfer_fixed39_d2_v3_root_adoption_20261005_v1',
 'persistent_optimizer_history_shared_graph_literature_20261005_v1',
 'literature_memory/index_v68',
 'literature_memory/index_v68_root_adoption_20261005_v1',
 'private_transfer_pubmed_fp32_qualification_launch_preparation_20261005_v1',
 'private_transfer_pubmed_fp32_qualification_launch_preparation_20261005_v2',
 'private_transfer_pubmed_fp32_qualification_launcher_independent_review_20261005_v1',
 'private_transfer_pubmed_fp32_qualification_launcher_independent_review_20261005_v2',
 'private_transfer_pubmed_fp32_qualification_resource_observation_root_20261005_v1',
 'private_transfer_pubmed_fp32_qualification_execution_root_20261005_v1',
 'private_transfer_pubmed_fp32_qualification_terminal_diagnostic_root_20261005_v1',
 'shared_private_transfer_paired_pilot_launch_receipts_root_20261005_v2/observation_20261005T124947Z',
 'shared_private_transfer_row0_companion_root_release_20261005_v1/observation_20261005T124947Z',
 'pencil_citeseer_native300_launch_receipts_20261005_v1/metadata_20261005T124349Z',
 'pencil_citeseer_native300_launch_receipts_20261005_v1/metadata_20261005T125037Z',
 'pencil_citeseer_native300_launch_receipts_20261005_v1/physical_20261005T125036Z',
]
selected={'CONTEXT.md','PUBLIC_STATUS.md','RESEARCH_STATE.md','research_ledger.json'}
for rel in roots:
 root=P/rel
 assert root.is_dir(),rel
 for f in root.rglob('*'):
  assert not f.is_symlink()
  if f.is_file():
   assert f.suffix in ('.py','.json','.jsonl','.md','.txt','.diff','.patch')
   if f.name not in ('CONSOLE.jsonl',):selected.add(str(f.relative_to(P)))
monitor=P/'amazon_polynormer_paired_family_execution_root_20261003_v3/v6_queue_launch_execution_receipts_20261003_v2'
for seq in (77,78,79):
 for suffix in ('REMOTE_CODE.py.txt','TRANSPORT.json','RESULT.json'):
  selected.add(str((monitor/('MONITOR_%04d_%s'%(seq,suffix))).relative_to(P)))
for name in ('PUSH_RECEIPT.json','ROOT_PUSH_ACKNOWLEDGEMENT.json','COMMIT_RECEIPT.json','PUSH_RECEIPT_FETCH_SOURCE.py.txt','PUSH_RECEIPT_FETCH_TRANSPORT.json'):
 selected.add('publication/allocation_pubmed_inputs_and_literature67_20261005_v1/'+name)
for name in ('BUILD_INVENTORY.py','PUBMED_ROOT_RELEASE_PLAN.md','PENDING_PUBLICATION.md','INITIAL_INSPECTION_SUPERSEDED.json','INVENTORY.json','INVENTORY_PLAN.json','INSPECT_RECEIPT.json'):
 selected.add(str((HERE/name).relative_to(P)))
rows=[]
def row(f,target):
 raw=f.read_bytes();assert len(raw)<2_000_000 and f.resolve().is_relative_to(P.resolve()) and not f.is_symlink()
 return {'source':str(f.relative_to(P)),'target':target,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
rows.append(row(HERE/'README_MAIN.md','README.md'))
for rel in sorted(selected):rows.append(row(P/rel,PREFIX+rel))
assert len({r['target'] for r in rows})==len(rows)
obj={'branch':'codex/postsubmission-research-20260930','expected_head':'2b44c654fa8398594c373d3b2393198235fc9801','message':'Preserve allocation-only owned Pubmed qualification, reviewed D2 closure and scoped literature history','remove':[],'files':rows}
with (HERE/'INVENTORY_v2.json').open('x') as s:json.dump(obj,s,indent=2);s.write('\n')
print(json.dumps({'paths':len(rows),'bytes':sum(r['bytes'] for r in rows),'raw_models_or_predictions':False,'outside_allocation_contact':False}))
