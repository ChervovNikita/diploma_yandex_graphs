"""Fixed reference recipes and paired cohort definitions."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

from ablation.common.resumable import source_fingerprint
from ablation.common.worker import Job


REPOSITORY = Path(__file__).resolve().parents[2]
RECIPES = {'ogbg-molbace': ['gcn', 'gin'], 'ogbg-molhiv': ['gin']}
METHODS = [f'ens_m{member}' for member in range(4)] + ['gnnm']
EXTENDED_METHODS = METHODS + ['gnnm_contrast', 'gnnm_private_bn', 'gnnm_private_bn_contrast']
REFERENCE = {
    'protocol': 'ogb-molecular-native-gnnm-v1',
    'num_layers': 5, 'hidden_dim': 300, 'dropout': 0.5, 'batch_size': 32,
    'lr': 0.001, 'weight_decay': 0.0, 'optimizer': 'Adam', 'scheduler': None,
    'epochs': 100, 'early_stopping': None, 'virtual_node': False,
    'residual': False, 'graph_pooling': 'mean', 'jumping_knowledge': 'last',
    'features': 'full', 'split': 'official_scaffold', 'metric': 'rocauc',
    'selection': 'first_max_validation_rocauc', 'pooling': 'mean_logits',
    'loss': 'mean_member_masked_bce_with_logits',
}


def provenance():
    vendored = REPOSITORY / 'external_sources/ogb_molecular'
    manifest = json.loads((vendored / 'manifest.json').read_text())
    for name, expected in manifest['local_sha256'].items():
        actual = hashlib.sha256((vendored / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Pinned molecular source changed: {name}')
    sources = list(Path(__file__).parent.glob('*.py'))
    sources += [REPOSITORY / name for name in (
        'models.py', 'ablation/common/worker.py', 'ablation/common/resumable.py',
        'ablation/common/contrastive.py')]
    sources += [vendored / name for name in manifest['local_sha256']]
    sources += [vendored / 'manifest.json']
    return {
        'source_sha256': source_fingerprint(sources),
        'reference_commit': manifest['commit'],
        'versions': {name: importlib.metadata.version(name)
                     for name in ('torch', 'torch-geometric', 'ogb', 'numpy', 'scikit-learn')},
    }


def make_jobs(args):
    binding = provenance()
    jobs = []
    for dataset in args.datasets:
        for model in RECIPES[dataset]:
            if args.models and model not in args.models:
                continue
            for cohort in args.cohorts:
                for method in args.methods:
                    member = int(method[-1]) if method.startswith('ens_m') else 0
                    config = dict(REFERENCE, **binding, dataset=dataset, model=model,
                                  cohort=cohort, method=method, seed=1000 * cohort + member,
                                  data_dir=str(confined_path(args.data_dir)),
                                  members=4 if method.startswith('gnnm') else 1,
                                  smoke_epochs=args.smoke_epochs,
                                  smoke_updates=args.smoke_updates,
                                  smoke=bool(args.smoke_epochs or args.smoke_updates))
                    config['defer_test'] = getattr(args, 'defer_test', False)
                    if getattr(args, 'convergence', False):
                        config.update(epochs=1000, cap_schedule=[1000, 2000], minimum_epochs=0, patience=50,
                                      early_stopping='first_max_valid_after_minimum',
                                      training_controller='min0_caps1000_2000_patience50')
                    config['normalization'] = ('route_private_running_statistics_shared_affine'
                                               if 'private_bn' in method else 'native')
                    key = f'{dataset}/{model}/cohort{cohort}/{method}'
                    jobs.append(Job(key=key, config=config))
    return jobs


def confined_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(REPOSITORY):
        raise ValueError(f'Path must be inside {REPOSITORY}: {path}')
    return path
