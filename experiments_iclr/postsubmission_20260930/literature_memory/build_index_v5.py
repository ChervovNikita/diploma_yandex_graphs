"""Index existing conclusions without silently inventing read scope or novelty."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
ROOTS = (
    'continuous_method_gap_search_v1',
    'continuous_graph_efficiency_gap_v1',
    'coordinate_source_independent_review_v1',
    'literature_gap_search_root_v8',
    'literature_root_followup_20261002_v1',
    'industrial_graph_followup_v1',
)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    out = PHASE / 'literature_memory' / args.version
    if out.parent != PHASE / 'literature_memory' or out.exists():
        raise ValueError('Require a new simple index version')
    records, catalogs = [], []
    for name in ROOTS:
        root = PHASE / name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob('*CONCLUSIONS.json')):
            if not path.is_file() or path.is_symlink():
                continue
            data = json.loads(path.read_text())
            entries = data.get('records', data.get('papers', data.get('entries', [])))
            if not isinstance(entries, list):
                raise ValueError('Unsupported conclusion schema: '+str(path))
            catalogs.append(dict(path=str(path.relative_to(PHASE)), sha256=sha(path),
                                 kind='structured_paper_conclusions', records=len(entries)))
            for entry in entries:
                canonical_id = entry.get('canonical_id')
                inherited = entry.get('original_record', {})
                if not canonical_id and inherited.get('canonical_doi'):
                    canonical_id = 'doi:'+inherited['canonical_doi']
                if not canonical_id:
                    raise ValueError('Conclusion needs verified canonical_id or inherited DOI: '+str(path))
                records.append(dict(canonical_id=canonical_id,
                                    conclusion_file=str(path.relative_to(PHASE)),
                                    conclusion_file_sha256=sha(path),
                                    conclusion=entry))
        for path in sorted(root.rglob('REPORT.md')):
            if not path.is_file() or path.is_symlink():
                continue
            text = path.read_text()
            title = next((line.lstrip('# ').strip() for line in text.splitlines() if line.strip()), '')
            catalogs.append(dict(path=str(path.relative_to(PHASE)), sha256=sha(path),
                                 kind='existing_packet_report', title=title,
                                 scope='Consult the report and its source map; listing alone does not certify a full-paper read.'))
    out.mkdir()
    index = dict(schema='gnnm-literature-memory-index-v1',
                 created_UTC=datetime.now(timezone.utc).isoformat(),
                 policy='Consult by canonical ID/title before fetching or rereading. Revisit only a named unresolved passage, new version or changed scientific question. Keep prior conclusions and failures. Attributed ingredients can support an extension; novelty requires a precise supported delta.',
                 paper_records=records, existing_packets=catalogs)
    (out/'LITERATURE_INDEX.json').write_text(json.dumps(index, indent=2)+'\n')
    lines = ['# Literature memory', '', index['policy'], '',
             '## Paper-specific conclusions', '']
    for entry in records:
        conclusion = entry['conclusion']
        lines.append('- **'+entry['canonical_id']+'**: '+conclusion.get('title', conclusion.get('verified_title', ''))+
                     ' — `'+entry['conclusion_file']+'`')
    lines.extend(['', '## Existing research packets', '',
                  'These reports retain earlier conclusions and source-reading limits. The index does not turn retrieval into a full read.', ''])
    for row in catalogs:
        if row['kind'] == 'existing_packet_report':
            lines.append('- '+row['title']+' — `'+row['path']+'`')
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(out), paper_records=len(records), catalog_entries=len(catalogs))))

if __name__ == '__main__':
    main()
