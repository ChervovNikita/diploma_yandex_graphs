"""Bounded source-only retrieval; never fetch graph archives or splits."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / 'sources'
HEADERS = {'User-Agent': 'GNNM-confirmation-protocol-source-audit/1.0',
           'Accept': 'application/vnd.github+json'}


def fetch(name, url, purpose):
    dest = SOURCES / name
    row = {'name': name, 'requested_url': url, 'purpose': purpose,
           'started_utc': datetime.now(timezone.utc).isoformat(),
           'dataset_content_request': False}
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=35) as response:
            content = response.read(8 * 1024 * 1024 + 1)
            if len(content) > 8 * 1024 * 1024:
                raise RuntimeError('Source retrieval size cap exceeded')
            row.update(status=response.status, final_url=response.url,
                       content_type=response.headers.get('Content-Type'))
        with dest.open('xb') as handle:
            handle.write(content)
        row.update(success=True, bytes=len(content),
                   sha256=hashlib.sha256(content).hexdigest(),
                   path=str(dest.relative_to(ROOT)))
    except Exception as exc:
        row.update(success=False, error_type=type(exc).__name__, error=str(exc))
    row['ended_utc'] = datetime.now(timezone.utc).isoformat()
    return row


def main():
    if (ROOT / 'RETRIEVAL_MANIFEST.json').exists():
        raise RuntimeError('Immutable retrieval packet already exists')
    rows = []
    commit_specs = [
        ('lrgb_commit.json', 'https://api.github.com/repos/vijaydwivedi75/lrgb/commits/main', 'Pin official LRGB dataset/config/evaluator sources'),
        ('pyg_270_tag.json', 'https://api.github.com/repos/pyg-team/pytorch_geometric/git/ref/tags/2.7.0', 'Pin official PyG GNNBenchmarkDataset provider'),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows.extend(pool.map(lambda spec: fetch(*spec), commit_specs))
    lrgb = json.loads((SOURCES / 'lrgb_commit.json').read_text())['sha']
    pyg_obj = json.loads((SOURCES / 'pyg_270_tag.json').read_text())['object']
    if pyg_obj['type'] != 'commit':
        raise RuntimeError('Annotated tag requires explicit dereference; stop')
    pyg = pyg_obj['sha']
    specs = [
        ('lrgb_peptides_functional.py', f'https://raw.githubusercontent.com/vijaydwivedi75/lrgb/{lrgb}/graphgps/loader/dataset/peptides_functional.py', 'Official peptide features, MD5 identities, split API; no data acquisition'),
        ('lrgb_metric_wrapper.py', f'https://raw.githubusercontent.com/vijaydwivedi75/lrgb/{lrgb}/graphgps/metrics/metric_wrapper.py', 'Official masked AP evaluation conventions'),
        ('lrgb_README.md', f'https://raw.githubusercontent.com/vijaydwivedi75/lrgb/{lrgb}/README.md', 'Official task counts, terms and primary bibliography'),
        ('lrgb_tree.json', f'https://api.github.com/repos/vijaydwivedi75/lrgb/git/trees/{lrgb}?recursive=1', 'Find frozen official baseline configuration filenames'),
        ('pyg_gnn_benchmark_dataset.py', f'https://raw.githubusercontent.com/pyg-team/pytorch_geometric/{pyg}/torch_geometric/datasets/gnn_benchmark_dataset.py', 'Independent CIFAR10 superpixel task compatibility'),
        ('lrgb_arxiv_abs.html', 'https://arxiv.org/abs/2206.08164', 'Original benchmark primary citation'),
        ('reassess_arxiv_abs.html', 'https://arxiv.org/abs/2309.00367', 'Primary warning against weak message-passing controls'),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows.extend(pool.map(lambda spec: fetch(*spec), specs))
    tree = json.loads((SOURCES / 'lrgb_tree.json').read_text())['tree']
    paths = [r['path'] for r in tree if r['type'] == 'blob' and
             ('peptides-func' in r['path'].lower() or 'peptides_func' in r['path'].lower())
             and r['path'].endswith(('.yaml', '.yml'))]
    selected = [p for p in paths if any(t in p.lower() for t in ('gine', 'gated', 'gcn', 'gps'))][:6]
    config_specs = [(f"lrgb_config_{i}_{p.split('/')[-1]}",
                     f'https://raw.githubusercontent.com/vijaydwivedi75/lrgb/{lrgb}/{p}',
                     'Published full-data backbone recipe context; no score use')
                    for i, p in enumerate(selected)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows.extend(pool.map(lambda spec: fetch(*spec), config_specs))
    manifest = {'schema': 'gnnm-confirmation-source-retrieval-v1',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'pinned_lrgb_commit': lrgb, 'pinned_pyg_270_commit': pyg,
                'requests': rows, 'config_candidates': paths,
                'downloaded_dataset_archives': False,
                'executed_retrieved_sources': False,
                'all_outputs_inside': str(ROOT)}
    with (ROOT / 'RETRIEVAL_MANIFEST.json').open('x') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'retrievals': len(rows), 'successful': sum(r['success'] for r in rows),
                      'lrgb_commit': lrgb, 'pyg_commit': pyg, 'selected_configs': selected}))


if __name__ == '__main__':
    main()
