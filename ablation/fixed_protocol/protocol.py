"""The frozen node training protocol of the final rebuttal runs.

This is the submitted protocol of the paper with exactly one change: the step
limit is 40000 instead of 5000 (20000 on Tolokers), so that no fit stops at the
limit. Read PROTOCOL.md before changing anything in this file. Every node fit of
the final campaign carries PROTOCOL_ID, and ``check_spec`` refuses to train a
fit whose protocol fields differ from the frozen values.
"""

import json
from pathlib import Path

PROTOCOL_ID = 'paper-patience300-cap40000-v1'
CAMPAIGN_ID = 'iclr-rebuttal-fixed-protocol-20261116'

# Fields that define how long a fit trains and which checkpoint it keeps.
FROZEN = {
    'min_steps': 0,             # no minimum training horizon
    'patience': 300,            # steps without improvement before stopping
    'selection_interval': 10,   # best checkpoint only on steps divisible by ten
    'num_steps': 40000,         # the only change against the paper
    'continuation_steps': 40000,
}
# Shared model settings of the main table, unchanged.
MAIN_TABLE = {
    'hidden_dim': 512,
    'lr': 3e-05,
    'weight_decay': 0.0,
    'dropout': 0.2,
    'attention_dropout': 0.2,
    'num_heads': 8,
}
# How fits store their outputs; not part of the scientific protocol.
STORAGE = {'defer_test': True, 'save_selected': True}

# Earlier result locations that a new run must never write into.
PROTECTED_ROOTS = (
    'results_merged',
    'results',
    'results2',
    'results3',
    'ablation/campaign/runs/production_owner_20261112_giga',
)
OUTPUT_PREFIX = 'fixed_protocol'


def node_protocol():
    return dict(FROZEN, **STORAGE, protocol_id=PROTOCOL_ID)


def check_spec(spec):
    """Raise unless a fit specification uses the frozen protocol."""
    if spec.get('protocol_id') != PROTOCOL_ID:
        raise ValueError(f"Fit protocol {spec.get('protocol_id')!r} is not {PROTOCOL_ID!r}")
    changed = {key: (spec.get(key), value) for key, value in FROZEN.items()
               if spec.get(key) != value}
    if changed:
        raise ValueError(f'Fit changes frozen protocol fields (found, frozen): {changed}')
    if spec.get('hidden_dim') != MAIN_TABLE['hidden_dim']:
        raise ValueError('Hidden width differs from the main table')


def check_output_root(root, repo):
    """Refuse output roots that could overwrite or mix with earlier results."""
    root, repo = Path(root).resolve(), Path(repo).resolve()
    relative = root.relative_to(repo)
    for protected in PROTECTED_ROOTS:
        protected_path = (repo / protected).resolve()
        if root == protected_path or protected_path in root.parents or root in protected_path.parents:
            raise ValueError(f'{relative} overlaps the earlier result location {protected}')
    if OUTPUT_PREFIX not in root.name:
        raise ValueError(f'Output folder name must contain {OUTPUT_PREFIX!r}: {relative}')
    marker = root / 'PROTOCOL_ID'
    if marker.exists() and marker.read_text().strip() != PROTOCOL_ID:
        raise ValueError(f'{relative} belongs to protocol {marker.read_text().strip()}')
    plan = root / 'plan.json'
    if plan.exists() and json.loads(plan.read_text()).get('campaign_id') != CAMPAIGN_ID:
        raise ValueError(f'{relative} belongs to another campaign')


def mark_output_root(root, repo):
    check_output_root(root, repo)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'PROTOCOL_ID').write_text(PROTOCOL_ID + '\n')


def fresh_output_dir(path, repo):
    """A collector output folder: guarded like a root and never reused."""
    path = Path(path)
    check_output_root(path, repo)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'{path} already contains results; choose a new folder')
    mark_output_root(path, repo)
    return path
