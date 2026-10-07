"""Consume the prospectively frozen TRAIN targets; never regenerate a relation."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from context_positive_masks import sampled_weights

ARCHIVE_SHA = '80d30ed946a1408f861a773952b174713c487765c75e95b8e25456b30478eae6'
PANEL_SHA = '9b742aa62fdd93496c2604edc8198b988a20f7f2ba83922a0f1a16597b9860bd'
TRAIN_SHA = 'dce6c4a982027604dad198533352d397c5d1b7a0a6d6fa4868468c872457f9e2'
MASK_SHA = '7615d8441a699f297314e7c838b9eb897b9e3dbfe22d1be350742e83f21405a9'
ROLE_SHA = 'b4769a68123433fc498b6abfafbcc3190f104749eafd72b34384e70dd41ff6aa'
KEYS = ('train_ids','train_labels','panel_rows','masks','permuted_masks','permutations')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def content_sha(array):
    array = np.ascontiguousarray(array)
    value = hashlib.sha256(str((array.dtype.str, array.shape)).encode())
    value.update(array.tobytes())
    return value.hexdigest()


def validate_logical_arrays(arrays, train_ids, train_labels, cpu_panel, device_panel):
    require(set(arrays) == set(KEYS), 'Exact frozen archive fields required')
    for name in ('train_ids','train_labels','panel_rows','permutations'):
        require(arrays[name].dtype == np.int64, 'Exact integer frozen identities required')
    ids, labels, rows = (arrays[name] for name in ('train_ids','train_labels','panel_rows'))
    require(ids.shape == labels.shape == (580,) and rows.shape == (512,), 'Complete580/512 frozen identities')
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
class FrozenTargets:
    arrays: dict
    metadata: dict
    array_hashes: dict

    def validate_current(self, train, torch, device):
        require(self.array_hashes == {key: content_sha(value) for key, value in self.arrays.items()},
                'Frozen in-memory target content changed')
        cpu = torch.linspace(0,579,steps=512,device='cpu').long().numpy()
        actual = torch.linspace(0,579,steps=512,device=device).long().cpu().numpy()
        validate_logical_arrays(self.arrays, train['ids'].cpu().numpy(),
            train['y'].cpu().numpy(), cpu, actual)


def load_frozen_targets(archive, receipt, role_manifest, train_sha, train, torch, device):
    """Validate hashes and logical contents before any model/cell is constructed."""
    archive, receipt, role_manifest = map(Path, (archive, receipt, role_manifest))
    require(train_sha == TRAIN_SHA and file_sha(archive) == ARCHIVE_SHA
        and archive.stat().st_size == 2724674 and file_sha(role_manifest) == ROLE_SHA,
        'Exact prospective TRAIN/role/target archive bindings required')
    result = json.loads(receipt.read_text())
    require(result.get('status') == 'ELIGIBLE' and result.get('TRAIN_only') is True
        and result.get('VALID_or_TEST_labels_accessed') is False
        and result.get('model_predictions_or_checkpoint_access') is False
        and result.get('scientific_fits') == 0
        and result.get('role_manifest_sha256') == ROLE_SHA
        and result.get('train_npz_sha256') == TRAIN_SHA
        and result.get('mask_source_sha256') == MASK_SHA
        and result.get('target_archive_sha256') == ARCHIVE_SHA
        and result.get('original_panel_sha256') == PANEL_SHA
        and result.get('permutation_seed') == 991327 and result.get('neighbor_count') == 16
        and result.get('panel_count') == 512 and result.get('complete_TRAIN_count') == 580,
        'Exact eligible source/preprocessing/panel/permutation receipt required')
    with np.load(archive, allow_pickle=False) as source:
        require(set(source.files) == set(KEYS), 'No extra frozen-target fields')
        arrays = {key: source[key].copy() for key in KEYS}
    require(hashlib.sha256(arrays['panel_rows'].astype('<i8').tobytes()).hexdigest() == PANEL_SHA,
            'Canonical original ordered panel bytes differ')
    route = sampled_weights(arrays['masks'], arrays['panel_rows'], 'route')
    common = sampled_weights(arrays['masks'], arrays['panel_rows'], 'common')
    permuted = sampled_weights(arrays['permuted_masks'], arrays['panel_rows'], 'route')
    tv_common = float(.5*np.abs(route-common).sum(-1).mean())
    tv_permuted = float(.5*np.abs(route-permuted).sum(-1).mean())
    require(tv_common >= .05 and tv_permuted >= .05
        and np.isclose(tv_common, result['target_vs_common_mean_TV'], atol=1e-12, rtol=1e-12)
        and np.isclose(tv_permuted, result['target_vs_permuted_mean_TV'], atol=1e-12, rtol=1e-12),
        'Frozen target eligibility differs from prospective receipt; no retry')
    hashes = {key: content_sha(value) for key, value in arrays.items()}
    for value in arrays.values():
        value.setflags(write=False)
    bundle = FrozenTargets(arrays, dict(target_archive_sha256=ARCHIVE_SHA,
        target_preflight_receipt_sha256=file_sha(receipt), role_manifest_sha256=ROLE_SHA,
        train_npz_sha256=TRAIN_SHA, original_panel_sha256=PANEL_SHA,
        mask_source_sha256=MASK_SHA, permutation_seed=991327,
        route_common_mean_target_TV=tv_common, route_permuted_mean_target_TV=tv_permuted,
        exact_scored_positive_counts_preserved=True, class_and_self_positives_preserved=True,
        target_relations_regenerated=False, frozen_array_content_sha256=hashes), hashes)
    bundle.validate_current(train, torch, device)
    return bundle
