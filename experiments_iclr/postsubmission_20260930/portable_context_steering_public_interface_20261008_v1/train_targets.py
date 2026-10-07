"""Fixed context relations constructed only from caller TRAIN labels and graph.

No model, development label, outcome, author archive or host is an input.
Numerical modules are imported only when preparation is called.
"""
from dataclasses import dataclass
import hashlib

KEYS = ('train_ids', 'train_labels', 'panel_rows', 'masks', 'permuted_masks', 'permutations')
NEIGHBORS = 16
PERMUTATION_SEED = 991327
MINIMUM_TARGET_TV = .05


def require(condition, message):
    if not condition:
        raise ValueError(message)


def content_sha(array):
    import numpy as np
    array = np.ascontiguousarray(array)
    value = hashlib.sha256(str((array.dtype.str, array.shape)).encode())
    value.update(array.tobytes())
    return value.hexdigest()


def validate_logical_arrays(arrays, train_ids, train_labels, cpu_panel, device_panel):
    import numpy as np
    require(set(arrays) == set(KEYS), 'Exact prepared target fields required')
    for name in ('train_ids','train_labels','panel_rows','permutations'):
        require(arrays[name].dtype == np.int64, 'Exact integer target identities required')
    ids, labels, rows = (arrays[name] for name in ('train_ids','train_labels','panel_rows'))
    require(ids.shape == labels.shape == (580,) and rows.shape == (512,), 'Complete580/512 prepared identities')
    require(np.array_equal(ids, train_ids) and np.array_equal(labels, train_labels),
            'Original ordered TRAIN IDs/labels differ from frozen targets')
    require(np.array_equal(rows, cpu_panel) and np.array_equal(rows, device_panel),
            'Frozen panel differs from current original CPU or actual device panel')
    require(len(np.unique(ids)) == 580 and len(np.unique(rows)) == 512
        and np.all((rows >= 0) & (rows < 580)), 'Unique complete TRAIN/panel identities')
    base, permuted, permutations = (arrays[name] for name in ('masks','permuted_masks','permutations'))
    require(base.shape == permuted.shape == (4,580,580)
        and base.dtype == permuted.dtype == bool and permutations.shape == (4,580),
        'Complete fixed base/permuted relations required')
    incompatible = labels[:,None] != labels[None,:]
    for mask in (base, permuted):
        require(np.all(np.diagonal(mask, axis1=1, axis2=2))
            and not np.any(mask & incompatible[None]), 'Fixed labels/self positives changed')
    member = np.zeros(580, dtype=bool); member[rows] = True
    for route, permutation in enumerate(permutations):
        require(np.array_equal(np.sort(permutation), np.arange(580))
            and np.array_equal(labels[permutation], labels)
            and np.array_equal(member[permutation], member),
            'Frozen permutation is not the complete class/panel-preserving bijection')
        require(np.array_equal(permuted[route], base[route][permutation[:,None], permutation[None,:]]),
                'Frozen permuted relation does not match its declared permutation')
    require(np.array_equal(base[:,rows[:,None],rows[None,:]].sum(-1),
                          permuted[:,rows[:,None],rows[None,:]].sum(-1)),
            'A scored anchor positive count changed')


@dataclass(frozen=True)
class PreparedTargets:
    arrays: dict
    metadata: dict
    array_hashes: dict

    def validate_current(self, train, torch, device):
        require(self.array_hashes == {key: content_sha(value) for key, value in self.arrays.items()},
                'Prepared target content changed')
        cpu = torch.linspace(0, 579, steps=512, device='cpu').long().numpy()
        actual = torch.linspace(0, 579, steps=512, device=device).long().cpu().numpy()
        validate_logical_arrays(self.arrays, train['ids'].cpu().numpy(),
                               train['y'].cpu().numpy(), cpu, actual)


def prepare_targets(train, torch, device):
    """Construct one immutable target set before any model or optimizer exists."""
    import numpy as np
    from context_positive_masks import (graph_contexts, positive_masks,
                                       sampled_weights, within_class_permuted)
    ids, labels = train['ids'].cpu().numpy(), train['y'].cpu().numpy()
    require(ids.shape == labels.shape == (580,) and ids.dtype == labels.dtype == np.int64,
            'Complete original-order580 caller TRAIN IDs/labels required')
    rows = torch.linspace(0, 579, steps=512, device='cpu').long().numpy()
    actual = torch.linspace(0, 579, steps=512, device=device).long().cpu().numpy()
    require(np.array_equal(rows, actual), 'Original CPU/device512 panel identities differ')
    contexts = graph_contexts(train['x'].cpu().numpy(), train['edge_index'].cpu().numpy(), ids)
    masks = positive_masks(contexts, labels, neighbors=NEIGHBORS)
    permuted, permutations = within_class_permuted(masks, labels, PERMUTATION_SEED, panel_rows=rows)
    arrays = dict(train_ids=ids.copy(), train_labels=labels.copy(), panel_rows=rows.copy(),
                  masks=masks, permuted_masks=permuted, permutations=permutations)
    validate_logical_arrays(arrays, ids, labels, rows, actual)
    route = sampled_weights(masks, rows, 'route')
    common = sampled_weights(masks, rows, 'common')
    randomized = sampled_weights(permuted, rows, 'route')
    common_tv = float(.5 * np.abs(route-common).sum(-1).mean())
    permuted_tv = float(.5 * np.abs(route-randomized).sum(-1).mean())
    require(common_tv >= MINIMUM_TARGET_TV and permuted_tv >= MINIMUM_TARGET_TV,
            'Fixed TRAIN target differentiation gate failed; no permutation-seed retry')
    hashes = {key: content_sha(value) for key, value in arrays.items()}
    for value in arrays.values():
        value.setflags(write=False)
    metadata = dict(source='caller complete TRAIN graph/features/labels only', neighbors=NEIGHBORS,
        fixed_permutation_seed=PERMUTATION_SEED, panel_objects=512, complete_TRAIN_objects=580,
        CPU_device_panel_identity=True, context_operator='directed destination row-normalized A+I; deduplicated edges; one self-loop',
        contexts=['X', 'X-PX', 'PX', 'P^2X'], post_panel_restriction_row_normalization=True,
        route_common_mean_target_TV=common_tv, route_permuted_mean_target_TV=permuted_tv,
        exact_scored_positive_counts_preserved=True, class_and_self_positives_preserved=True,
        zero_signature_rows=np.count_nonzero(np.linalg.norm(contexts, axis=-1)==0, axis=1).tolist(),
        original_panel_raw_int64le_sha256=hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest(),
        frozen_array_content_sha256=hashes, VALID_labels_or_model_predictions_used=False,
        official_source_certified_from_arbitrary_NPZ=False, no_target_or_seed_retry=True)
    bundle = PreparedTargets(arrays, metadata, hashes)
    bundle.validate_current(train, torch, device)
    return bundle
