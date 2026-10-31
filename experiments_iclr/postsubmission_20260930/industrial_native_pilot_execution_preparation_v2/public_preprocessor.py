"""Complete public-only native preprocessing. No raw labels or provider imports."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import dgl
import numpy as np
import pandas as pd
import torch
import yaml
from native_transforms import Setting, transform_num_features, transform_cat_features


@dataclass
class PreparedPublic:
    graph: object
    features: dict
    official_masks: dict
    source_masks: dict
    column_lineage: dict
    node_count: int

    def tensors(self, device):
        graph = self.graph.to(device)
        arrays = self.features
        numerical = None if arrays['num_features'] is None else torch.as_tensor(arrays['num_features']).to(device)
        other = [torch.as_tensor(arrays[key]).to(device) for key in ('cat_features', 'frac_features')
                 if arrays[key] is not None]
        return graph, numerical, torch.cat(other, dim=1) if other else None

    @property
    def width(self):
        return sum(array.shape[1] for array in self.features.values() if array is not None)

    @property
    def numerical_width(self):
        array = self.features['num_features']
        return 0 if array is None else array.shape[1]


def load_public(root, frozen_roles, compact):
    root = Path(root)
    with (root / 'info.yaml').open() as file:
        info = yaml.safe_load(file)
    frame = pd.read_csv(root / 'features.csv', index_col=0).astype(np.float32)
    if len(frame) != len(frozen_roles['node_ids']):
        raise ValueError('Public row count differs from frozen axis')
    kept = frame.nunique() != 1
    original_columns = frame.columns.tolist()
    frame = frame.loc[:, kept]
    columns = frame.columns.tolist()
    fraction_names = [name for name in info['fraction_features_names'] if name in columns]
    category_names = [name for name in info['categorical_features_names'] if name in columns]
    numerical_names = [name for name in info['numerical_features_names']
                       if name not in fraction_names and name in columns]
    fractions = frame.loc[:, fraction_names].values.astype(np.float32) if fraction_names else None
    categories = frame.loc[:, category_names].values.astype(np.int32) if category_names else None
    numerical = frame.loc[:, numerical_names].values.astype(np.float32) if numerical_names else None
    if fractions is not None and np.isnan(fractions).any():
        raise ValueError('Native fraction features cannot contain NaNs')
    if categories is not None:
        mask = np.array([len(np.unique(column)) > 1 for column in categories.T])
        if not mask.any():
            raise ValueError('Native categorical cleanup assertion failed')
        categories = categories[:, mask]
        category_names = [name for name, keep in zip(category_names, mask) if keep]
    moved = []
    if fractions is not None:
        binary = np.all((fractions == 0.) | (fractions == 1.), axis=0)
        if binary.any():
            moved = [name for name, keep in zip(fraction_names, binary) if keep]
            binary_values = fractions[:, binary].astype(np.float32)
            categories = np.concatenate([categories, binary_values], axis=1) if categories is not None else binary_values
            fractions = None if binary.all() else fractions[:, ~binary]
            fraction_names = [name for name, keep in zip(fraction_names, binary) if not keep]
            category_names += moved
    masks_frame = pd.read_csv(root / 'split_masks_RL.csv', index_col=0).astype(bool)
    official = {role: masks_frame[role].values for role in ('train', 'val', 'test')}
    source = {}
    for role in ('train', 'val'):
        mask = np.zeros(len(frame), dtype=bool)
        mask[compact[role]['rows']] = True
        if not np.all(~mask | official[role]):
            raise ValueError('Source mask exceeds official role')
        source[role] = mask
    # Test missingness is never acquired: official test mask is public only.
    edges = pd.read_csv(root / 'edgelist.csv').values[:, :2]
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError('Native edge endpoints must already be integer CSV columns')
    if edges.ndim != 2 or edges.shape[1] != 2 or (edges < 0).any() or (edges >= len(frame)).any():
        raise ValueError('Public edge endpoints exceed native feature-row node axis')
    endpoint = torch.from_numpy(edges).T
    graph = dgl.graph((endpoint[0], endpoint[1]), num_nodes=len(frame), idtype=torch.int32)
    graph = dgl.remove_self_loop(graph)
    graph = dgl.to_simple(graph)
    graph = dgl.to_bidirected(graph)
    return PreparedPublic(graph, {'num_features': numerical, 'cat_features': categories,
                                 'frac_features': fractions}, official, source,
                          {'original_columns': original_columns, 'kept_columns': columns,
                           'numerical': numerical_names, 'categorical': category_names,
                           'fraction': fraction_names, 'binary_fractions_moved': moved}, len(frame))


def transform_public(public, policies):
    # Match upstream GRAPH_FEATURE_KEYS transform order, preserving dtype.
    masks = {**public.source_masks, 'test': public.official_masks['test']}
    for key, policy_key in (('num_features', 'num_policy'), ('frac_features', 'frac_policy'),
                            ('cat_features', 'cat_policy')):
        array = public.features[key]
        if array is None:
            continue
        fn = transform_cat_features if key == 'cat_features' else transform_num_features
        kwargs = {'setting': Setting.TRANSDUCTIVE, 'policy': policies[policy_key]}
        if key != 'cat_features':
            kwargs['seed'] = policies['seed']
        public.features[key] = fn(array, masks, **kwargs)
        if not np.isfinite(public.features[key]).all():
            raise ValueError('Nonfinite transformed public features')
    if public.width == 0:
        raise ValueError('No encoded public features')
    return public
