"""Retain failed paths; inspect author-provided stronger baseline source."""
import json
from pathlib import Path
from retrieve_primary import fetch

ROOT = Path(__file__).resolve().parent


def main():
    if (ROOT / 'RETRIEVAL_ADDENDUM.json').exists():
        raise RuntimeError('Existing immutable addendum')
    base = json.loads((ROOT / 'RETRIEVAL_MANIFEST.json').read_text())
    commit = base['pinned_lrgb_commit']
    rows = [fetch('lrgb_metric_wrapper_corrected.py',
                  f'https://raw.githubusercontent.com/vijaydwivedi75/lrgb/{commit}/graphgps/metric_wrapper.py',
                  'Corrected observed official metric path; prior 404 retained'),
            fetch('lrgb_metrics_ogb.py',
                  f'https://raw.githubusercontent.com/vijaydwivedi75/lrgb/{commit}/graphgps/metrics_ogb.py',
                  'Exact AP masking and averaging evaluator'),
            fetch('reassess_author_commit.json',
                  'https://api.github.com/repos/Toenshoff/LRGB/commits/main',
                  'Pin reassessment author repository; no baseline scores accessed')]
    ref = None
    if rows[-1]['success']:
        ref = json.loads((ROOT/'sources/reassess_author_commit.json').read_text())['sha']
        rows.append(fetch('reassess_author_tree.json',
                          f'https://api.github.com/repos/Toenshoff/LRGB/git/trees/{ref}?recursive=1',
                          'Identify reassessed Peptides GINE recipe filenames'))
        if rows[-1]['success']:
            tree = json.loads((ROOT/'sources/reassess_author_tree.json').read_text())['tree']
            paths = [r['path'] for r in tree if r['type'] == 'blob' and
                     'peptides' in r['path'].lower() and 'func' in r['path'].lower() and
                     r['path'].endswith(('.yaml', '.yml', '.json')) and
                     'gine' in r['path'].lower()]
            for i, path in enumerate(paths[:4]):
                rows.append(fetch(f'reassess_config_{i}_{Path(path).name}',
                                  f'https://raw.githubusercontent.com/Toenshoff/LRGB/{ref}/{path}',
                                  'Author reassessed GINE configuration; source only'))
    output = {'schema': 'gnnm-confirmation-source-retrieval-addendum-v1',
              'requests': rows, 'reassess_author_commit': ref,
              'downloaded_dataset_archives': False,
              'reassess_config_paths': paths if ref and rows[3]['success'] else []}
    with (ROOT/'RETRIEVAL_ADDENDUM.json').open('x') as f:
        json.dump(output, f, indent=2)
        f.write('\n')
    print(json.dumps({'requests': len(rows), 'successful': sum(r['success'] for r in rows),
                      'reassess_commit': ref,
                      'config_paths': output['reassess_config_paths']}))


if __name__ == '__main__':
    main()
