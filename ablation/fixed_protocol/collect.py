"""Collect the final campaign into new tables; earlier results are only read.

    python -m ablation.fixed_protocol.collect --root CAMPAIGN_ROOT --out results_fixed_protocol_20261117

The archived main table is read from archive_main.csv (frozen from results_merged).

Writes into a new folder (the name must contain ``fixed_protocol``; an existing
non-empty folder is refused):

    runs/<folder>/{base,ensemble,tabm}.csv  main table in the archive layout: archived
                                            runs plus the retrained ones, which replace
                                            the archived run of the same cell
    main_runs.csv            every main-table run with its source (archive or rerun)
    main_table.csv           test mean per dataset, backbone and method, depth chosen
                             per split by validation as in the paper
    parity.csv               the ten parity runs next to their archived scores
    rerun_changes.csv        old against new score of every retrained run
    roman_ablation.csv       every Roman variant run with paired BASE/ENS/GNNM
    roman_ablation_summary.csv  test mean per backbone and variant, depth chosen per
                             split by validation (paper rule) and by mean validation
    node_contrast.csv        GNNM + contrastive term against GNNM at the same cell
    status.json              missing and unfinished score tasks
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from ablation.p08_training_steps.estimate import DATASET_FOLDERS, METHODS, PAPER_MODELS, read_results_csv
from .protocol import PROTOCOL_ID, check_spec, fresh_output_dir

REPO = Path(__file__).resolve().parents[2]
KEYS = ['dataset', 'model', 'num_layers', 'split']
ARCHIVE_CSV = Path(__file__).with_name('archive_main.csv')
SETTING_METHOD = {'BASE': 'base', 'ENS': 'ens', 'GNNM': 'gnnm'}


def score_results(root, prefix):
    """Finished score tasks whose identifier starts with ``prefix``."""
    rows, unfinished = [], []
    base = Path(root) / 'tasks' / prefix
    for task_dir in sorted(path for path in base.glob('**/output') if path.is_dir()):
        done = task_dir / 'done.json'
        task_id = str(task_dir.parent.relative_to(Path(root) / 'tasks'))
        if not done.exists():
            unfinished.append(task_id)
            continue
        result = json.loads(done.read_text())
        config = result.get('config', {})
        # Engineering smoke runs override the protocol on purpose; they never enter a table.
        if config.get('role') != 'scientific':
            raise ValueError(f'{task_id}: not a scientific fit (role {config.get("role")!r})')
        check_spec(config)
        rows.append(dict(task_id=task_id, dataset=result['dataset'], model=result['model'],
                         num_layers=int(result['num_layers']), split=int(result['split']),
                         setting=result.get('setting'), val_metric=result.get('val_metric'),
                         test_metric=result.get('test_metric'),
                         best_step=config.get('best_step'), stopped_step=config.get('stopped_step'),
                         stop_reason=config.get('stop_reason'),
                         unresolved=len(result.get('unresolved_convergence', [])),
                         protocol_id=config.get('protocol_id')))
    frame = pd.DataFrame(rows)
    if len(frame) and (frame.protocol_id != PROTOCOL_ID).any():
        raise ValueError(f'{prefix}: a result was trained under another protocol')
    return frame, unfinished


def archived_main(results):
    frames = []
    for folder in DATASET_FOLDERS:
        for method, (file_name, _) in METHODS.items():
            frame = read_results_csv(Path(results) / folder / file_name)
            frame = frame[frame.model.isin(PAPER_MODELS)][KEYS + ['val_metric', 'test_metric']].copy()
            frame['method'], frame['folder'], frame['source'] = method, folder, 'archive'
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def merged_main(archive, rerun):
    if not len(rerun):
        return archive.copy()
    new = rerun.assign(method=rerun.setting.map(SETTING_METHOD), source='rerun')
    folders = archive.drop_duplicates('dataset').set_index('dataset').folder
    new['folder'] = new.dataset.map(folders)
    kept = archive.merge(new[KEYS + ['method']].assign(replaced=True), on=KEYS + ['method'], how='left')
    kept = kept[kept.replaced.isna()].drop(columns='replaced')
    columns = KEYS + ['method', 'folder', 'source', 'val_metric', 'test_metric']
    return pd.concat([kept[columns], new[columns + ['best_step', 'stopped_step', 'stop_reason']]],
                     ignore_index=True)


def per_split_depth(frame, group):
    """The paper's rule: on every split, the depth with the best validation score."""
    ordered = frame.sort_values('num_layers')
    best = ordered.loc[ordered.groupby(group + ['split']).val_metric.idxmax()]
    return best.groupby(group).agg(test=('test_metric', 'mean'), val=('val_metric', 'mean'),
                                   splits=('split', 'nunique')).reset_index()


def mean_val_depth(frame, group):
    """One depth per cell: the best mean validation score over the splits."""
    means = frame.groupby(group + ['num_layers']).agg(val=('val_metric', 'mean'), test=('test_metric', 'mean'),
                                                       splits=('split', 'nunique')).reset_index()
    return means.sort_values('num_layers').loc[means.groupby(group).val.idxmax()].reset_index(drop=True)


def write_archive_layout(main, out):
    names = {method: file_name for method, (file_name, _) in METHODS.items()}
    for (folder, method), frame in main.groupby(['folder', 'method']):
        target = out / 'runs' / folder
        target.mkdir(parents=True, exist_ok=True)
        frame.drop(columns=['folder', 'method']).sort_values(KEYS).to_csv(target / names[method], index=False)


def load_archive(path=None):
    """The archived main table, frozen in archive_main.csv; or rebuilt from the
    archive folders when ``path`` is a results folder such as results_merged."""
    if path is None:
        return pd.read_csv(ARCHIVE_CSV)
    return archived_main(path)


def collect(root, archive, out):
    status = {}
    rerun, status['main_rerun_unfinished'] = score_results(root, 'main_rerun')
    parity, status['parity_unfinished'] = score_results(root, 'parity')
    roman, status['roman_unfinished'] = score_results(root, 'score/roman')
    contrast, status['contrast_unfinished'] = score_results(root, 'node_contrast_score')
    main = merged_main(archive, rerun)
    main.to_csv(out / 'main_runs.csv', index=False)
    write_archive_layout(main, out)
    per_split_depth(main, ['dataset', 'model', 'method']).to_csv(out / 'main_table.csv', index=False)
    if len(rerun):
        changes = rerun.assign(method=rerun.setting.map(SETTING_METHOD)).merge(
            archive, on=KEYS + ['method'], how='left', suffixes=('', '_archive'))
        changes.to_csv(out / 'rerun_changes.csv', index=False)
    if len(parity):
        parity.assign(method=parity.setting.map(SETTING_METHOD)).merge(
            archive, on=KEYS + ['method'], how='left', suffixes=('', '_archive')).to_csv(out / 'parity.csv', index=False)
    if len(roman):
        controls = main[(main.dataset == 'roman-empire')].copy()
        controls['setting'] = controls.method.map({v: k for k, v in SETTING_METHOD.items()})
        table = pd.concat([roman[KEYS + ['setting', 'val_metric', 'test_metric']],
                           controls[KEYS + ['setting', 'val_metric', 'test_metric']]], ignore_index=True)
        table.to_csv(out / 'roman_ablation.csv', index=False)
        paper = per_split_depth(table, ['model', 'setting']).rename(columns={'test': 'test_per_split_depth'})
        single = mean_val_depth(table, ['model', 'setting']).rename(
            columns={'test': 'test_mean_val_depth', 'num_layers': 'mean_val_depth'})
        paper.merge(single[['model', 'setting', 'mean_val_depth', 'test_mean_val_depth']],
                    on=['model', 'setting']).to_csv(out / 'roman_ablation_summary.csv', index=False)
    if len(contrast):
        gnnm = main[main.method == 'gnnm'][KEYS + ['val_metric', 'test_metric']]
        contrast[KEYS + ['val_metric', 'test_metric']].merge(
            gnnm, on=KEYS, suffixes=('_contrast', '_gnnm')).to_csv(out / 'node_contrast.csv', index=False)
    status.update(protocol_id=PROTOCOL_ID, root=str(Path(root).resolve()),
                  rerun_scores=len(rerun), parity_scores=len(parity), roman_scores=len(roman),
                  contrast_scores=len(contrast),
                  unresolved_reruns=int(rerun.unresolved.sum()) if len(rerun) else 0)
    (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--results', type=Path, default=None,
                        help='archived results folder to read instead of archive_main.csv')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = fresh_output_dir(args.out, REPO)
    status = collect(args.root, load_archive(args.results), out)
    print(json.dumps({key: value if not isinstance(value, list) else len(value)
                      for key, value in status.items()}, indent=2))


if __name__ == '__main__':
    main()
