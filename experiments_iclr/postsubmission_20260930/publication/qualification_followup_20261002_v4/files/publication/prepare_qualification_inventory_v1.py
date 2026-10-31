"""Snapshot reviewed source/receipts, excluding arrays, checkpoints and keys."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PHASE=Path(__file__).resolve().parents[1]
ROOTS=(
    'continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v1',
    'continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v2',
    'modern_teacher_execution_root_v1','modern_teacher_source_audit_v2',
    'graph_init_source_audit_v1','source_audit_snapshots/graph_init_draft01',
    'coordinate_source_independent_review_v1/graph_tangent_initialization_review_v1',
    'coordinate_source_independent_review_v1/morgan_primary_resolution_v1',
)
SUFFIXES={'.py','.json','.md','.csv','.jsonl','.txt','.html','.xml','.sha256','.log','.patch','.diff','.bib','.sh'}
QUOTED_PREFIXES=(
    'coordinate_source_independent_review_v1/graph_tangent_initialization_review_v1/sources/',
    'coordinate_source_independent_review_v1/morgan_primary_resolution_v1/sources/',
)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',required=True)
    args=parser.parse_args()
    out=PHASE/'publication'/args.version
    if out.parent!=PHASE/'publication' or out.exists():raise ValueError('New publication snapshot version required')
    out.mkdir()
    paths=[]
    for rel in ROOTS:
        paths+=sorted(p for p in (PHASE/rel).rglob('*') if p.is_file() and not p.is_symlink() and p.suffix in SUFFIXES)
    paths += [PHASE/p for p in (
        'PUBLIC_STATUS.md','research_ledger.json','protocols/launch_modern_root_v1.py',
        'publication/PUSH_VERIFIED_20261002_v2.json','publication/REVIEWED_SOURCE_COMMIT_20261002_v2.json',
        'publication/prepare_qualification_inventory_v1.py','publication/SOURCE_INVENTORY_PREPARE_FAILED_v3.json',
        'literature_memory/build_index_v2.py','literature_memory/build_index_v3.py','literature_memory/INDEX_BUILD_FAILED_v2.json')]
    for version in ('index_v3','index_v4','index_v5'):
        paths+=sorted(p for p in (PHASE/'literature_memory'/version).iterdir() if p.is_file())
    fetches={'FETCH_AUTHORIZED_EVIDENCE_v'+str(i)+'.json' for i in range(124,138)}
    for path in sorted((PHASE/'protocols').glob('*.json')):
        if path.name.startswith(('UPLOAD_MODERN','MODERN_QUALIFICATION_LAUNCH','PHOTO_REPEATABILITY','UPLOAD_PHOTO_REPEATABILITY','MODERN_SOURCE_PACK_AUDIT_LAUNCH')) or path.name in fetches:
            paths.append(path)
    rows=[];quoted_syntax=[]
    for path in sorted(set(paths)):
        data=path.read_bytes();rel=path.relative_to(PHASE).as_posix()
        if len(data)>=10*1024**2:raise ValueError('Large payload excluded: '+rel)
        if path.suffix=='.py':
            try:ast.parse(data.decode(),filename=rel)
            except SyntaxError as error:
                if not rel.startswith(QUOTED_PREFIXES):raise
                quoted_syntax.append(dict(path=rel,error_type=type(error).__name__,message=str(error),scope='Pinned author text as literature evidence, not executable study source.'))
        if path.suffix=='.json':json.loads(data)
        snapshot=out/'files'/rel;snapshot.parent.mkdir(parents=True,exist_ok=True)
        with snapshot.open('xb') as handle:handle.write(data)
        rows.append(dict(target='experiments_iclr/postsubmission_20260930/'+rel,
            source_snapshot=snapshot.relative_to(PHASE).as_posix(),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
    readme=(PHASE/'publication/README_RESEARCH_UPDATE_20261002_v3.md').read_bytes()
    snapshot=out/'files/README.md';snapshot.write_bytes(readme)
    rows.append(dict(target='README.md',source_snapshot=snapshot.relative_to(PHASE).as_posix(),sha256=hashlib.sha256(readme).hexdigest(),bytes=len(readme)))
    result=dict(created_UTC=datetime.now(timezone.utc).isoformat(),branch='codex/postsubmission-research-20260930',
        expected_head='e35dcdc28986ea7258b572251bacba1f45ad4a12',files=rows,source_only_or_qualified_receipts=True,
        scientific_utility_claim=False,credentials_or_binary_artifacts_included=False,original_score_files_included=False,
        quoted_author_syntax_errors_preserved=quoted_syntax)
    (out/'INVENTORY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(files=len(rows),bytes=sum(row['bytes'] for row in rows),quoted_nonrunnable_author_files=len(quoted_syntax),inventory=str(out/'INVENTORY.json'))))

if __name__=='__main__':main()
