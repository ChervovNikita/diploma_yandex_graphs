"""Add completed independent reviews and preserve the superseded audit verdict."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

P=Path(__file__).resolve().parents[1]
O=P/'publication/actual_count_census_update_20261004_v1'
UTC=datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
 with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')

reviews={
 'full_TRAIN_census':'graph_count_conditioned_train_support_census_independent_actual_audit_20261004_v1',
 'grouped_exact_core_v2':'graph_count_conditioned_pattern_loss_grouped_independent_source_review_20261004_v2',
 'native_control_superseded_PASS':'pubmed_native_only_continuation_control_fresh_source_review_20261004_v1',
 'native_control_corrective_BLOCKED':'pubmed_native_only_continuation_control_fresh_source_review_20261004_v2',
}
evidence={}
for name,relative in reviews.items():
 folder=P/relative;f=folder/'REVIEW.json';d=json.loads(f.read_text())
 for row in json.loads((folder/'MANIFEST.json').read_text())['files']:
  p=folder/row['path'];assert p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
 evidence[name]=dict(path=relative+'/REVIEW.json',status=d['status'],sha256=sha(f),manifest_sha256=sha(folder/'MANIFEST.json'))
assert evidence['full_TRAIN_census']['status']=='PASS_SCOPED_ACTUAL_METADATA_AUDIT'
assert evidence['grouped_exact_core_v2']['status']=='PASS'
assert evidence['native_control_superseded_PASS']['status']=='PASS'
assert evidence['native_control_corrective_BLOCKED']['status']=='BLOCKED'
folder=P/'actual_count_and_native_reviews_root_followup_20261004_v1';folder.mkdir()
value=dict(UTC=UTC,status='ADOPTED_SCOPED_CENSUS_AND_CORE_AUDITS_AND_CORRECTIVE_NATIVE_BLOCK',reviews=evidence,
 census_checks=963,census_independent_raw_histogram_reread=False,
 census_positive_shared_association_support_fraction=0.058241900275735295,
 census_positive_shared_association_with_genuine_subset_fraction=115762/3342336,
 native_control_v1_execution_authorized=False,native_control_v1_PASS_superseded=True,
 native_control_v2_repair_under_preparation=True,
 vectorized_single_core_v3_under_preparation=True,source_reviews_are_predictive_evidence=False,
 original_scores_changed=False,confirmed_new_methodological_advantage=False,fresh_manuscript_acceptance=False)
save(folder/'ROOT_FOLLOWUP.json',value)
(folder/'SUMMARY.md').write_text('''# Actual audit follow-up

The independent scoped census audit passed 963 checks of local receipts, compact population/marginal accounting and custody links. It did not independently reread server-retained histogram bytes. Shared association involving a genuine subset occurs in 3.4635% of positive query occurrences; the larger 9.1725% either-genuine fraction includes cases with a forced counterpart.

The grouped core-v2 independent source review passed within its declared source-only scope. It grants no numerical execution or native feasibility evidence. Its remaining QA and dispatch limits are preserved; a complete exact vectorized single-control successor is under preparation.

The native-only control reviewer corrected an initial mistaken PASS after a focused check exposed a late-interruption handling defect. Both immutable reviews are preserved; the corrective BLOCKED verdict governs. No release or run used the mistaken PASS. A minimal separately sealed supervisor repair is under preparation.

No manuscript acceptance or new methodological predictive advantage is established by these audits.
''')
rows=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.iterdir()) if p.is_file()]
save(folder/'MANIFEST.json',dict(UTC=UTC,files=rows,original_scores_changed=False))
save(folder/'SEAL.json',dict(manifest_sha256=sha(folder/'MANIFEST.json'),sealed=True))
ledger=json.loads((P/'research_ledger.json').read_text());key='actual_count_and_native_review_correction_20261004_v1';assert key not in ledger
ledger[key]=value;ledger['updated_UTC']=ledger['updated_utc']=UTC
(P/'research_ledger.json').write_text(json.dumps(ledger,indent=2,sort_keys=True,allow_nan=False)+'\n')
for name in ['PUBLIC_STATUS.md','RESEARCH_STATE.md']:
 text=(P/name).read_text()
 text=text.replace('A fresh native-only repeat control is under review; it has a different warm-state history and cannot alone establish cause.',
   'A fresh native-only repeat control is being repaired after its independent reviewer corrected an initial mistaken PASS to BLOCKED for a supervisor interruption defect. Both reviews are preserved; no run used that PASS. Its warm-state history differs and cannot alone establish cause.')
 text=text.replace('A successor closes endpoint/float32/ragged gaps; it is sealed, disabled and under independent review.',
   'The grouped successor received a scoped independent source PASS and remains unexecuted. A complete vectorized single-control successor is under preparation before further numerical QA.')
 text=text.replace('Group dispatch and the sequential single-control implementation need practical optimization before full-batch feasibility is assumed.',
   'The independent scoped metadata audit passed963 checks. Group dispatch and the sequential single-control implementation need practical optimization before full-batch feasibility is assumed.')
 text=text.replace('Review and execute the fresh252-update native-only control;',
   'Repair and independently review the fresh252-update native-only control after the corrective BLOCKED source audit; then execute once admitted;')
 text=text.replace('Independently review grouped core-v2 endpoint/float32/ragged assurance and run actual successor QA after separate admission.',
   'Preserve the scoped grouped-core-v2 source PASS. Finish/review the exact vectorized single core-v3 and execute one complete successor QA after separate admission, including both-side genuine subsets.')
 text=text.replace('Audit full completed TRAIN census.', 'Preserve the independent scoped audit of the full completed TRAIN census.')
 (P/name).write_text(text)

inventory=json.loads((O/'INVENTORY.json').read_text());extra=[Path(__file__).resolve()]
for relative in list(reviews.values())+['actual_count_and_native_reviews_root_followup_20261004_v1']:
 extra.extend(f for f in sorted((P/relative).rglob('*')) if f.is_file())
existing={row['source'] for row in inventory['files']}
for f in extra:
 relative=str(f.relative_to(P))
 if relative not in existing:
  inventory['files'].append(dict(source=relative,target='experiments_iclr/postsubmission_20260930/'+relative,bytes=f.stat().st_size,sha256=sha(f)))
  existing.add(relative)
for row in inventory['files']:
 f=P/row['source'];assert f.is_file() and f.stat().st_size<2_000_000
 row.update(bytes=f.stat().st_size,sha256=sha(f))
(O/'INVENTORY.json').write_text(json.dumps(inventory,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(status='COMPLETED_AUDITS_ADOPTED_WITH_CORRECTIVE_BLOCK_PRESERVED',files=len(inventory['files']),bytes=sum(r['bytes'] for r in inventory['files']))))
