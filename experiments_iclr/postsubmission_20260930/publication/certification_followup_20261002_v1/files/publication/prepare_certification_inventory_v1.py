"""Snapshot sealed repairs, metadata certificates and scoped research notes."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
OUT = PHASE/'publication/certification_followup_20261002_v1'
HEAD = '2ac000e04740df22c7e601cf84d9c7639de0c1d7'
ROOTS = (
    'continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v3',
    'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v1',
    'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v2',
    'modern_teacher_execution_root_v1/certification_v3',
    'graph_init_source_audit_v1/modern_v3_recheck',
    'graph_init_mechanism_analysis_root_v1',
    'literature_root_followup_20261002_v1',
    'literature_memory/index_v6','literature_memory/index_v7','literature_memory/index_v8',
)
EXTENSIONS = {'.json','.py','.md','.txt','.xml','.html','.csv','.diff','.patch','.log','.jsonl','.sh'}

def main():
    OUT.mkdir(exist_ok=False)
    paths=[]
    for rel in ROOTS:
        paths.extend(p for p in sorted((PHASE/rel).rglob('*'))
                     if p.is_file() and not p.is_symlink() and p.suffix in EXTENSIONS)
    paths.extend(PHASE/rel for rel in ('PUBLIC_STATUS.md','RESEARCH_STATE.md','research_ledger.json',
        'literature_memory/build_index_v4.py','publication/REVIEWED_SOURCE_COMMIT_20261002_v3.json',
        'publication/PUSH_VERIFIED_20261002_v3.json','publication/prepare_certification_inventory_v1.py',
        'protocols/FETCH_AUTHORIZED_EVIDENCE_v138.json','protocols/FETCH_AUTHORIZED_EVIDENCE_v139.json',
        'protocols/UPLOAD_QUALIFICATION_PUBLICATION_v1.json','protocols/UPLOAD_MODERN_CERTIFICATION_v3.json'))
    rows=[]
    for path in sorted(set(paths)):
        data=path.read_bytes();rel=path.relative_to(PHASE).as_posix()
        if len(data)>=10*1024**2: raise ValueError('Large artifact excluded')
        if path.suffix=='.py': ast.parse(data,filename=rel)
        if path.suffix=='.json': json.loads(data)
        target=OUT/'files'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(data)
        rows.append(dict(target='experiments_iclr/postsubmission_20260930/'+rel,
            source_snapshot=target.relative_to(PHASE).as_posix(),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
    readme=(PHASE/'publication/README_RESEARCH_UPDATE_20261002_v3.md').read_text()
    readme=readme.replace('Fresh engineering audits found qualification/replay custody gaps and initializer registry/release issues. Separately versioned repairs are in progress before full scientific fits.',
        'The modern qualification/replay repairs passed an independent sealed-source recheck. Eighteen exact metadata certificates were issued, with numerical seed17/config0 evidence kept distinct from authorized target roles. The separately sealed five-arm initialization implementation awaits its repaired-source recheck and actual full-graph AD/Adam qualification before scientific fits.')
    readme=readme.replace('literature_memory/index_v5/README.md','literature_memory/index_v8/README.md')
    readme=readme.replace('All current work uses the explicitly authorized single-GPU allocation.',
        'A new primary reading on conventional GCN/GATv2 ensemble uncertainty is saved with its scope and limits. The conditional mechanism analysis records that centered tangent effects cancel in mean logits to first order at equal step length; actual safeguarded arms may accept different steps. Neither disagreement nor local training descent establishes heldout benefit.\n\nAll current work uses the explicitly authorized single-GPU allocation.')
    data=readme.encode();target=OUT/'files/README.md';target.write_bytes(data)
    rows.append(dict(target='README.md',source_snapshot=target.relative_to(PHASE).as_posix(),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
    inventory=dict(created_UTC=datetime.now(timezone.utc).isoformat(),branch='codex/postsubmission-research-20260930',
        expected_head=HEAD,files=rows,scientific_utility_claim=False,
        original_score_files_included=False,credentials_or_binary_artifacts_included=False)
    (OUT/'INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(inventory=str(OUT/'INVENTORY.json'),files=len(rows),bytes=sum(r['bytes'] for r in rows))))

if __name__=='__main__':main()
