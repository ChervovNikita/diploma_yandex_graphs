"""Freeze historical GNNM scores at the previously VALID-selected depths."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = {
    'roman-empire': 'results2/roman_empire',
    'amazon-ratings': 'results2/amazon_ratings',
    'minesweeper': 'results/minesweeper',
    'tolokers': 'results3/tolokers',
    'questions': 'results/questions',
}


def main():
    selected_path = ROOT / 'research_audits/historical_lrwd_20261111/reference_validation_selected_depths.csv'
    with selected_path.open(newline='') as stream:
        selections = [row for row in csv.DictReader(stream) if row['method'] == 'GNNM']
    sources, output = {}, []
    for dataset, directory in SOURCE_DIRS.items():
        path = ROOT / directory / 'tabm.csv'
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream, skipinitialspace=True))
        for chosen in (row for row in selections if row['dataset'] == dataset):
            depth = int(chosen['num_layers'])
            cell = [row for row in rows
                    if row.get('model') == chosen['model'] and row.get('num_layers') == str(depth)
                    and row.get('hidden_dim') == '512' and float(row['lr']) == 3e-5]
            if sorted(int(row['split']) for row in cell) != list(range(10)):
                raise ValueError('Incomplete historical cell: ' + str(chosen))
            for row in cell:
                output.append(dict(dataset=dataset, model=chosen['model'], split=int(row['split']),
                                   num_layers=depth, val_metric=float(row['val_metric']),
                                   test_metric=float(row['test_metric']),
                                   lr=3e-5, weight_decay=0.0, hidden_dim=512,
                                   source=str(path.relative_to(ROOT))))
    if len(output) != 350:
        raise ValueError('Expected 350 frozen GNNM reference scores')
    target = Path(__file__).with_name('historical_baseline.csv')
    with target.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(sorted(output, key=lambda row: (row['dataset'], row['model'], row['split'])))
    manifest = dict(selection='Previously frozen mean VALID over ten official splits; no TEST ranking',
                    selected_depths_sha256=hashlib.sha256(selected_path.read_bytes()).hexdigest(),
                    source_files=sources, rows=len(output),
                    baseline_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    target.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
