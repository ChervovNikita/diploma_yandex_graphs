"""Snapshot reviewed closed initialization evidence and new literature, excluding binaries."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
OUT = PHASE/'publication/precision_followup_20261002_v1'
HEAD = '28d95833fb86f56e96ce07e64a79d49f2954994e'
ROOTS = (
    'photo_fd_arithmetic_diagnostic_v1', 'photo_fd_arithmetic_diagnostic_root_v1',
    'graph_init_source_audit_v1/photo_fd_failure_analysis_v1',
    'graph_init_source_audit_v1/photo_fd_arithmetic_diagnostic_v1_recheck',
    'graph_factor_distinct_gap_scout_v1', 'closest_graph_init_primary_closure_v1',
    'literature_memory/index_v11', 'industrial_native_pilot_preparation_v1',
    'industrial_runtime_capability_root_v1',
)

EXTENSIONS = {'.json', '.py', '.md', '.txt', '.html', '.csv', '.diff', '.patch', '.log', '.jsonl', '.sh', '.yaml', '.yml', '.toml'}

def main():
    OUT.mkdir(exist_ok=False)
    paths = []
    for relative in ROOTS:
        paths.extend(path for path in sorted((PHASE/relative).rglob('*'))
                     if path.is_file() and not path.is_symlink() and path.suffix in EXTENSIONS)
    for relative in ('PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json',
        'literature_memory/build_index_v7.py', 'REVIEW_RESPONSE_TRACKER.md',
        'publication/REVIEWED_SOURCE_COMMIT_20261002_v5.json', 'publication/PUSH_VERIFIED_20261002_v5.json',
        'publication/prepare_precision_inventory_v1.py',
        'continuous_method_gap_search_v1/round15_graph_route_initialization/prototype/graph_band_route_initializer.py'):
        paths.append(PHASE/relative)
    paths.extend(PHASE/'protocols'/('FETCH_AUTHORIZED_EVIDENCE_'+str(i)+'.json') for i in range(148,150))
    paths.extend(path for path in (PHASE/'protocols').glob('UPLOAD_PHOTO_FD_*.json'))
    paths.append(PHASE/'protocols/UPLOAD_GRAPH_INIT_PUBLICATION_v1.json')
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
    readme = (PHASE/'publication/graph_init_followup_20261002_v1/files/README.md').read_text()
    old = 'The failure remains retained; a separate precision diagnostic is in preparation. Compatibility checks establish no accuracy improvement.'
    new = ('The separate exact-state diagnostic reproduced the original failure exactly. '
        'Promoting FP32 per-example CE before mean reduction reduced the failed errors from11.27%/20.14% '
        'to0.0692%/0.6029%; all fixed directions/steps passed unchanged thresholds. '
        'The49.899-second numerical diagnostic supports a prospective measurement amendment; '
        'the original study stays failed and no useful performance training has started. '
        'Compatibility and arithmetic checks establish no accuracy improvement.')
    if old not in readme:
        raise ValueError('Reviewed README predecessor differs')
    readme = readme.replace(old, new).replace('literature_memory/index_v10/README.md', 'literature_memory/index_v11/README.md')
    anchor = 'All current work uses the explicitly authorized single-GPU allocation.'
    addition = ('Two further scoped papers, GMNN and B³F-GNN, plus one official FAGEL diagram are retained. '
        'An edge-pair learning proposal was not promoted because of exact factorization/prior overlap. '
        'MORGAN/FAGEL full papers remain inaccessible in bounded retrieval, with that uncertainty preserved. '
        'The Tolokers2 native pilot has a sealed preparation protocol and source drafts; operational isolation, '
        'preprocessing and native/COO numerical parity remain pending. The allocation provides native DGL2.4 '
        'and user namespaces; a runnable isolated worker is still in preparation. No Tolokers2 label arrays '
        'or models were accessed in this preparation.\n\n')
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
