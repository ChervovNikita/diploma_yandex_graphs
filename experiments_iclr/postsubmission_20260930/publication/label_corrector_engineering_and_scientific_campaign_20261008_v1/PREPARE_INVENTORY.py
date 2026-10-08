from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parents[2]
D=Path(__file__).resolve().parent
prefix='experiments_iclr/postsubmission_20260930/'
roots=['restored_gpu77_and_allocation_observation_20261008T175103Z','label_only_four_bank_full_input_cuda_engineering_adoption_root_20261008_v1','label_only_common_mask_correction_covariance_20261008_root_v1','label_only_four_bank_WikiCS_scientific_owner_source_20261008_v1','label_only_four_bank_WikiCS_scientific_owner_root_review_20261008_v1','label_only_post_family_error_analysis_plan_20261008_v1']
files=[P/n for n in ('PUBLIC_STATUS.md','RESEARCH_STATE.md','research_ledger.json')]
for root in roots:
 r=P/root
 assert r.is_dir(),r
 files.extend(x for x in r.rglob('*') if x.is_file())
for name in ('PUSH_RECEIPT.json','PUSH_RECEIPT_RETRY01.json','GPU77_SYNC_RESULT.json'):
 files.append(P/'publication/label_corrector_integration_and_controls_20261008_v1'/name)
for ident in ('restore_observation_20261008T175103Z','label_corrector_53e_exact_sync_20261008T1753Z'):
 files.append(P/'gpu77_connection_recovery_v1/commands'/(ident+'.txt'))
 files.extend((P/'gpu77_connection_recovery_v1/commands'/ident).glob('*'))
files.append(Path(__file__).resolve())
rows=[]
for f in sorted(set(files)):
 rel=str(f.relative_to(P))
 assert not f.is_symlink() and f.stat().st_size<2000000 and f.suffix in ('.json','.md','.py','.txt','.log','.csv')
 rows.append(dict(source=rel,target=prefix+rel,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
plan=dict(branch='codex/postsubmission-research-20260930',expected_head='53e73cf737b15385ece81d3cb6b12212fb1e5db1',message='Adopt full-input correction engineering and owned three-seed screen',files=rows,remove=[])
with (D/'INVENTORY.json').open('x') as h:json.dump(plan,h,indent=2);h.write('\n')
print(json.dumps({'files':len(rows),'bytes':sum(x['bytes'] for x in rows)}))
