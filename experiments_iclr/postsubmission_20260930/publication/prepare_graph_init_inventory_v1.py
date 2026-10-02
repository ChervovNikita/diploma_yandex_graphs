"""Snapshot reviewed closed initialization evidence and new literature, excluding binaries."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
OUT = PHASE/'publication/graph_init_followup_20261002_v1'
HEAD = '946384acbac8663dc51636e568b33f9c2e83e0bf'
ROOTS = (
    'graph_init_execution_root_v1', 'graph_init_execution_continuation_v1',
    'graph_init_root_decisions_v1', 'graph_init_resource_review_root_v1',
    'graph_init_source_audit_v1/round17_v2_recheck',
    'graph_init_source_audit_v1/execution_root_v1_recheck',
    'graph_init_source_audit_v1/continuation_v1_recheck',
    'industrial_graph_followup_v1', 'industrial_dataset_acquisition_root_v1',
    'heterogeneous_factor_gap_v1', 'literature_memory/index_v9', 'literature_memory/index_v10',
)
EXTENSIONS = {'.json', '.py', '.md', '.txt', '.html', '.csv', '.diff', '.patch', '.log', '.jsonl', '.sh', '.yaml', '.yml', '.toml'}

def main():
    OUT.mkdir(exist_ok=False)
    paths = []
    for relative in ROOTS:
        paths.extend(path for path in sorted((PHASE/relative).rglob('*'))
                     if path.is_file() and not path.is_symlink() and path.suffix in EXTENSIONS)
    for relative in ('PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json',
        'literature_memory/build_index_v5.py', 'literature_memory/build_index_v6.py',
        'publication/REVIEWED_SOURCE_COMMIT_20261002_v4.json', 'publication/PUSH_VERIFIED_20261002_v4.json',
        'publication/prepare_graph_init_inventory_v1.py',
        'continuous_method_gap_search_v1/round15_graph_route_initialization/prototype/graph_band_route_initializer.py'):
        paths.append(PHASE/relative)
    paths.extend(PHASE/'protocols'/('FETCH_AUTHORIZED_EVIDENCE_'+str(i)+'.json') for i in range(141,148))
    paths.extend(path for path in (PHASE/'protocols').glob('UPLOAD_GRAPH_INIT_*.json'))
    paths.append(PHASE/'protocols/UPLOAD_INDUSTRIAL_ACQUISITION_v1.json')
    rows = []
    for path in sorted(set(paths)):
        data = path.read_bytes(); relative = path.relative_to(PHASE).as_posix()
        if len(data) >= 10*1024**2:
            raise ValueError('Large payload excluded')
        if path.suffix == '.py':
            ast.parse(data, filename=relative)
        if path.suffix == '.json':
            json.loads(data)
        snapshot = OUT/'files'/relative; snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open('xb') as stream:
            stream.write(data)
        rows.append(dict(target='experiments_iclr/postsubmission_20260930/'+relative,
            source_snapshot=snapshot.relative_to(PHASE).as_posix(),
            sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    readme = (PHASE/'publication/certification_followup_20261002_v1/files/README.md').read_text()
    old = 'The separately sealed five-arm initialization implementation awaits its repaired-source recheck and actual full-graph AD/Adam qualification before scientific fits.'
    new = ('The five-arm initializer and finite execution wrapper passed independent source reviews. '
        'All three full-graph Squirrel cold qualifications passed; the initial check took15.197 seconds and '
        'allocated at most4.92GB in recorded operations. Photo17 then failed one small scalar-loss finite-difference '
        'check, stopping the queue before useful warm/continuation fits. The failure remains retained; a separate '
        'precision diagnostic is in preparation. Compatibility checks establish no accuracy improvement.')
    if old not in readme:
        raise ValueError('Reviewed README predecessor differs')
    readme = readme.replace(old, new).replace('literature_memory/index_v8/README.md', 'literature_memory/index_v10/README.md')
    anchor = 'All current work uses the explicitly authorized single-GPU allocation.'
    addition = ('Two new industrial primaries, GraphLandv5 and GraphPFNv4, identify Tolokers2 classification and '
        'Artnetviews regression with competitive contemporary references. The exact Tolokers2 archive was '
        'acquired and checksum-verified; label arrays remain undecoded. Two further scoped primaries found '
        'substantial heterogeneous-adapter overlap, so type-specific factors alone are not promoted as novelty. '
        'Saved source recipes and access limits accompany every conclusion.\n\n')
    readme = readme.replace(anchor, addition+anchor)
    data = readme.encode(); snapshot = OUT/'files/README.md'; snapshot.write_bytes(data)
    rows.append(dict(target='README.md', source_snapshot=snapshot.relative_to(PHASE).as_posix(),
        sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    inventory = dict(created_UTC=datetime.now(timezone.utc).isoformat(),
        branch='codex/postsubmission-research-20260930', expected_head=HEAD, files=rows,
        scientific_utility_claim=False, original_score_files_included=False,
        credentials_or_binary_artifacts_included=False)
    (OUT/'INVENTORY.json').write_text(json.dumps(inventory, indent=2)+'\n')
    print(json.dumps(dict(files=len(rows), bytes=sum(row['bytes'] for row in rows),
                         inventory=str(OUT/'INVENTORY.json'))))

if __name__ == '__main__':
    main()
