"""Local source receipt/scope/custody checks only; no numerical/model operations."""
from pathlib import Path
import hashlib
import json

P = Path(__file__).resolve().parent
R = P.parent
failures = []


def read(name):
    return json.loads((P/name).read_text())


def digest(path):
    b = path.read_bytes()
    return {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def check(ok, message):
    if not ok:
        failures.append(message)


m = read('MANIFEST.json')
for f in m['files']:
    check(digest(P/f['path']) == {k:f[k] for k in ['bytes','sha256']}, 'packet hash mismatch: '+f['path'])
check({f['path'] for f in m['files']} == {str(f.relative_to(P)) for f in P.rglob('*') if f.is_file() and f.name not in m['excludes']}, 'manifest inventory mismatch')
for f in read('INPUT_BINDINGS.json')['inputs']:
    check(digest(R/f['path']) == {k:f[k] for k in ['bytes','sha256']}, 'input custody mismatch: '+f['path'])
discovery = read('DISCOVERY_RECEIPTS.json')
check([v['call'] for v in discovery] == list(range(1,7)), 'discovery count/order mismatch')
for f in discovery:
    if f['status'] == 200:
        check(digest(P/f['file']) == {k:f[k] for k in ['bytes','sha256']}, 'discovery receipt digest mismatch')
primary = []
for file in sorted(P.glob('PRIMARY_RETRIEVAL*.json')):
    primary.extend(json.loads(file.read_text()))
check(len(primary) == 5, 'selected-source retrieval accounting mismatch')
for f in primary:
    if f['status'] == 200:
        check(digest(P/f['file']) == {k:f[k] for k in ['bytes','sha256']}, 'primary receipt digest mismatch')
check(len({f['url'] for f in primary}) == len(primary), 'blocked/selected route repeated')
for e in read('PRIMARY_EXCERPTS.json')['excerpts']:
    base = R if e.get('base') == 'research_root' else P
    file = base/e['file']
    check(digest(file) == e['file_binding'], 'excerpt file binding mismatch')
    if e['coordinate_kind'].startswith('PDF'):
        t = '\n'.join(file.read_text().splitlines()[e['line_start']-1:e['line_end']])
    else:
        b = json.loads(file.read_text())[e['block_index']]
        check(b['id'] == e['html_id'], 'HTML ID coordinate mismatch')
        t = b['text']
    check(t == e['text'] and hashlib.sha256(t.encode()).hexdigest() == e['text_sha256'], 'excerpt text/hash mismatch')
scopes = read('READ_SCOPES.json')
check(scopes['public_discovery_calls'] == 6 <= scopes['discovery_call_limit'], 'discovery limit/accounting mismatch')
check(scopes['first_scoped_method_identities'] == 2 <= scopes['new_method_identity_limit'], 'new method scope limit/accounting mismatch')
check(scopes['retained_pencil_scope_extensions'] == 1 and scopes['total_scoped_method_read_events'] == 3, 'retained scope accounting mismatch')
check(not set(scopes['pencil']['prior_ids']) & set(scopes['pencil']['new_ids']), 'selected PENCIL scope repeats prior IDs')
check(scopes['full_primary_papers_read'] == scopes['author_code_read_events'] == 0, 'bounded reading accounting mismatch')
idx = (R/'literature_memory/index_v32/LITERATURE_INDEX.json').read_text().casefold()
for s in read('PAPER_CONCLUSIONS.json')['new_records']:
    check(s['canonical_id'] not in idx and s['verified_title'].casefold() not in idx, 'new method identity already indexed')
    check(digest(P/s['primary_file']) == s['source_binding'], 'source conclusion binding mismatch')
check('2602.01553' in idx, 'retained PENCIL identity missing from index')
custody = read('PRIOR_REPORT_CUSTODY.json')
check(custody['targets'] == 270 and custody['all_unchanged'], 'prior report preservation mismatch')
for f in custody['files']:
    check(digest(R/f['path']) == f['expected'], 'old report custody mismatch: '+f['path'])
result = {'status': 'PASS' if not failures else 'FAIL', 'failures': failures,
          'packet_files': len(m['files']), 'bound_inputs': len(read('INPUT_BINDINGS.json')['inputs']),
          'discovery_calls': len(discovery), 'selected_source_retrievals': len(primary),
          'new_method_identities': 2, 'retained_PENCIL_scope_extensions': 1,
          'primary_excerpts': len(read('PRIMARY_EXCERPTS.json')['excerpts']),
          'prior_reports_preserved': 270, 'full_paper_certifications': 0,
          'verification_scope': 'File hashes, exact passage coordinates, discovery/reading counts and input/report custody only; no modeling, dataset operation or quality verdict.'}
(P/'VERIFICATION.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
raise SystemExit(bool(failures))
