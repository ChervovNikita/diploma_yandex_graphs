"""The fixed hyperparameter grid and method definitions of the GAT pilot."""

from itertools import product


ENSEMBLE_MEMBERS = 4
METHODS = ['base', 'gnnm'] + [f'ens_m{k}' for k in range(ENSEMBLE_MEMBERS)]
DATASETS = ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']
LAYERS = [1, 2, 3, 4, 5]
PILOT_SPLITS = [0, 1, 2]
HIDDEN_DIM = 512

GRID_LR = 1e-4
REFERENCE_LR = 3e-5
# The main-table setting: the same dropout for features and attention, eight heads.
REFERENCE = {'dropout': 0.2, 'attention_dropout': 0.2, 'num_heads': 8, 'lr': REFERENCE_LR}


def config_name(config):
    return (f"do{config['dropout']}_att{config['attention_dropout']}"
            f"_h{config['num_heads']}_lr{config['lr']:g}"
            + (f"_wd{config['weight_decay']:g}" if config.get('weight_decay', 0) else ''))


def build_configs():
    configs = [
        {'dropout': dropout, 'attention_dropout': attention, 'num_heads': heads, 'lr': GRID_LR}
        for dropout, attention, heads in product([0.0, 0.1, 0.2, 0.5], [0.0, 0.2], [2, 8])
    ]
    configs.append(dict(REFERENCE))
    return {config_name(config): config for config in configs}


CONFIGS = build_configs()
REFERENCE_NAME = config_name(REFERENCE)


def steps_for(dataset):
    """Step limits of the archived main table: Tolokers was trained for longer."""
    return 20000 if dataset == 'tolokers' else 5000


def seed_for(method, split):
    if method.startswith('ens_m'):
        return split * 1000 + int(method.removeprefix('ens_m'))
    return split


def members_for(method):
    return ENSEMBLE_MEMBERS if method == 'gnnm' else 1


def configs_for(family):
    """Named prospective grids retain the colleague's original grid unchanged."""
    if family == 'legacy':
        return dict(CONFIGS)
    lrwd = [dict(REFERENCE, lr=lr, weight_decay=wd)
            for lr, wd in product([3e-5, 1e-4, 3e-4], [0.0, 1e-4, 1e-2])]
    extra = {config_name(config): config for config in lrwd}
    if family == 'lrwd':
        return extra
    if family == 'combined':
        return {**CONFIGS, **extra}
    raise ValueError(f'Unknown grid family {family}')
