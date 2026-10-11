"""Publish a fixed explicit inventory in small sequential commits."""
import hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parents[2]
B=Path(__file__).resolve().parent
prefix='experiments_iclr/postsubmission_20260930/'
branch='codex/postsubmission-research-20260930'
helper=P/'publication/publish_exact_inventory_v27.py'
dirs=['complete_anonymous_quality_draft_20261011_v5','Physics_original_boundary_GNNM_anchor_source_20261011_v1','Physics_original_boundary_GNNM_anchor_readiness_execution_20261011_v1','Physics_original_boundary_GNNM_anchor_acquisition_execution_20261011_v1','native_TABM_joint_route_dropout_source_20261011_v1','native_TABM_joint_route_dropout_source_20261011_v2','shared_route_information_exchange_architecture_scout_20261011_v1','shared_route_information_exchange_architecture_scout_20261011_v2','activation_dependent_route_dropout_hypothesis_root_20261011_v1','route_dropout_pilot_runtime_binding_root_20261011_v1','research_continuation_after_Physics_complete_20261011_v1']
suffixes={'.py','.json','.md','.txt','.diff','.bib','.tex','.sty','.bst','.csv','.png','.log','.raw','.xml','.svg'}
files=[P/'RESEARCH_STATE.md',P/'research_ledger.json',helper,Path(__file__).resolve()]
for d in dirs:
 files.extend(x for x in (P/d).rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.suffix in suffixes)
files=sorted(set(files),key=lambda x:str(x.relative_to(P)))
rows=[]
for x in files:
 assert x.resolve().is_relative_to(P) and not x.is_symlink()
 data=x.read_bytes();assert len(data)<2000000
 source=str(x.relative_to(P))
 rows.append(dict(source=source,target=prefix+source,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
head=json.loads((P/'publication/LATEST_VERIFIED_20261011_V17.json').read_text())['commit']
batches=[];batch=[];size=0
for row in rows:
 if size+row['bytes']>2100000 and batch:batches.append(batch);batch=[];size=0
 batch.append(row);size+=row['bytes']
if batch:batches.append(batch)
with (B/'COMPLETE_INVENTORY.json').open('x') as f:json.dump(dict(files=rows,batches=len(batches),total_bytes=sum(x['bytes'] for x in rows)),f,indent=2)
print(json.dumps(dict(files=len(rows),parts=len(batches),bytes=sum(x['bytes'] for x in rows))),flush=True)
for n,part in enumerate(batches,1):
 d=B/('part'+str(n));d.mkdir()
 inv=d/'INVENTORY.json'
 inv.write_text(json.dumps(dict(branch=branch,expected_head=head,message='Preserve complete V5 and frozen route acquisition source: part '+str(n)+' of '+str(len(batches)),files=part,remove=[]),indent=2)+'\n')
 subprocess.run([sys.executable,str(helper),'inspect','--inventory',str(inv.relative_to(P)),'--receipt',str((d/'INSPECT_RECEIPT.json').relative_to(P))],check=True)
 subprocess.run([sys.executable,str(helper),'commit','--inventory',str((d/'INVENTORY_PLAN.json').relative_to(P)),'--receipt',str((d/'COMMIT_TRANSPORT.json').relative_to(P))],check=True)
 receipt=json.loads((d/'COMMIT_RECEIPT.json').read_text());head=receipt['commit']
 print(json.dumps(dict(part=n,parts=len(batches),head=head)),flush=True)
summary=dict(UTC=datetime.now(timezone.utc).isoformat(),complete=True,commit=head,parts=len(batches),files=len(rows),bytes=sum(x['bytes'] for x in rows),push_verified=False,final_commit_receipt=str((B/('part'+str(len(batches)))/'COMMIT_RECEIPT.json').relative_to(P)))
(B/'COMPLETE_BATCH_COMMITS.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
