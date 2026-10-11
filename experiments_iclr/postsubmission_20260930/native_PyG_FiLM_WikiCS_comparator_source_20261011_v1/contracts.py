"""Derived finite source/data/runtime admission; stdlib only."""
import hashlib
import json
from pathlib import Path
import socket

STUDY = 'native_PyG_FiLM_WikiCS'
SEEDS = [7301, 7403, 7507]
ARMS = ['FiLM_M1', 'FiLM_shared4', 'FiLM_I4']
FIXED = {'depth': 2, 'hidden': 128, 'dropout': .2, 'learning_rate': .001,
    'max_updates': 1000, 'patience_updates': 300, 'rank': 16,
    'member_seed_stride': 1000003, 'factor_seed_offset': 2000003,
    'dropout_seed_offset': 3000007, 'block_seed_offset': 4000037,
    'baseline_policy': 'fresh_required', 'stage': 'F', 'K': None,
    'k_selection_receipt': None, 'study_kind': STUDY}
SOURCE_KEYS = {'training_family', 'film_adapter', 'contracts', 'analysis',
    'math_utils', 'readiness', 'common_routes', 'factors', 'models', 'run_common',
    'film_conv', 'message_passing'}
PINNED = {'common_routes': '84ff13b28ef471ac0a4199e10d4284f24647d396e021646e2a21a06f96fb0773',
    'factors': '9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3',
    'models': '07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303',
    'run_common': 'e5256b39aef9fdc688832320a9e7d8aa24f244f48f6939d0b0daa342639a1ebc',
    'film_conv': '060f41df9462a8ad1c14abfaa364da7e6fe8e66dcd4ad4f28301e68f0e036635',
    'message_passing': '66e4ef4afa1d1b2d46c805b8987b6ea0c52ec5c95a9beee8a62758358e481e6f'}
ROLE_HASHES = {'train': '88c36e1983f17a9e53d3f13a473baaf9eff86f52bc4be56aba012321983fdc6d',
              'valid': '591fe3a05d4bb80b9bfaf047292c04f91163abf60c43106d13f98046258268ad'}
FILM_COUNT_KEYS = ('film_conv_calls', 'film_node_dense_maps', 'film_node_dense_rows',
    'film_node_dense_MACs', 'film_relation_builder_rows', 'film_skip_builder_rows',
    'film_message_rows', 'film_edge_modulation_multiplications',
    'film_edge_beta_additions', 'film_edge_ReLU_coordinates',
    'film_skip_modulation_multiplications', 'film_skip_beta_additions',
    'film_skip_ReLU_coordinates', 'film_mean_output_rows',
    'film_factual_self_message_rows')
PROTOCOL = {'schema_version': 1, 'study_kind': STUDY, 'arms': ARMS, 'seeds': SEEDS,
    'banks': 9, 'optimizer_bundles': 18, 'native_body_records': 18,
    'native_routes_per_roster_forward': 27, 'operator': 'installed_PyG2.7.0_FiLMConv',
    'aggregation': 'mean', 'activation': 'native_edgewise_ReLU_and_skip_ReLU',
    'relations': 1, 'width': 128, 'depth': 2,
    'self_policy': 'original_stored_self_records_in_ordinary_mean_plus_native_FiLM_skip; no_loop_changes',
    'serving': 'probability_mean', 'objective': 'mean own CE for shared; unscaled own CE for singles/I4',
    'selector': 'earliest_strict_maximum_full_VALID_probability_mean_accuracy; own_body_for_independent_fits',
    'calibration': {'folds': 5, 'fold_seed': 11709, 'scalar_fits': 45,
        'updates': 500, 'optimizer': 'CPU float64 Adam', 'learning_rate': .01,
        'betas': [.9, .999], 'epsilon': 1e-8, 'weight_decay': 0,
        'initial_log_T': 0, 'endpoint': 'fixed final update',
        'operator': 'temperature of member log probabilities before probability mean',
        'retries': 0, 'grid': False, 'all_VALID_refit': False},
    'all_banks_before_comparison': True, 'all45_scalar_attempts_before_interpretation': True,
    'TEST_access': False, 'automatic_confirmation_launch': False,
    'baseline_policy': 'fresh_required', 'historical_logit_anchors': False,
    'new_accuracy_or_novelty_gate': False,
    'scope': 'Encountered WikiCS development; three paired optimizer repeats on one full graph; larger native FiLM capacity disclosed; no reported-paper reproduction or novelty claim'}


def require(value, message):
    if not value:
        raise ValueError(message)


def receipt(item):
    require(isinstance(item, dict) and set(item) == {'path', 'bytes', 'sha256'}, 'Exact portable file receipt')
    p = Path(item['path']).resolve()
    require(p.is_file() and type(item['bytes']) is int and p.stat().st_size == item['bytes'], 'File size differs')
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    require(h.hexdigest() == item['sha256'], 'File hash differs: '+str(p))
    return p


def validate(cfg, *, action, executing_source):
    require(action in ('fixed_family_fit', 'runtime_readiness_check'), 'Fixed roster or one root runtime check')
    require(cfg.get('seeds') == SEEDS and all(cfg.get(k) == v for k, v in FIXED.items()),
            'Exact original WikiCS recipe and FiLM comparator identity required')
    limits = cfg.get('family_limits')
    require(isinstance(limits, dict) and set(limits) == {'wall_seconds', 'cuda_bytes', 'rss_bytes'}
            and all(type(v) is int and v > 0 for v in limits.values()), 'Finite root wall/CUDA/RSS caps')
    sources, roles = cfg.get('source_receipts'), cfg.get('safe_roles')
    require(isinstance(sources, dict) and set(sources) == SOURCE_KEYS and isinstance(roles, dict)
            and set(roles) == {'train', 'valid'}, 'All active code and original safe roles must be bound')
    paths = {k: receipt(v) for k, v in sources.items()}
    role_paths = {k: receipt(v) for k, v in roles.items()}
    here = Path(__file__).resolve().parent
    for k, name in (('training_family', 'run_family.py'), ('film_adapter', 'film_adapter.py'),
                    ('contracts', 'contracts.py'), ('analysis', 'analysis.py'),
                    ('math_utils', 'math_utils.py'), ('readiness', 'readiness.py')):
        require(paths[k] == here/name, 'Active local source path differs: '+k)
    require(Path(executing_source).resolve() == paths['training_family'], 'Executing family source differs')
    for k, digest in PINNED.items():
        require(sources[k]['sha256'] == digest, 'Canonical native dependency differs: '+k)
    require(Path(cfg['native_repo']).resolve() == paths['models'].parent
            and paths['run_common'] == paths['models'].with_name('run_common.py')
            and Path(cfg['common_routes']).resolve() == paths['common_routes']
            and Path(cfg['factors']).resolve() == paths['factors'], 'Exact native/route/factor mapping')
    require(isinstance(cfg.get('device'), str) and cfg['device'].startswith('cuda'), 'Original explicit CUDA runtime')
    require(all(roles[k]['sha256'] == ROLE_HASHES[k] and role_paths[k] == Path(cfg[k+'_npz']).resolve() for k in roles),
            'Original full WikiCS TRAIN/VALID bytes and role paths; no substituted masks')
    admission = json.loads(receipt(cfg['root_admission']).read_text())
    source_hashes = {str(paths[k]): sources[k]['sha256'] for k in sources}
    science = {k: v for k, v in cfg.items() if k != 'root_admission'}
    canonical = hashlib.sha256(json.dumps(science, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(admission.get('status') == 'HOST_GPU_SOURCE_DATA_ADMITTED'
            and admission.get('action') == action and admission.get('study_kind') == STUDY
            and admission.get('hostname') == socket.gethostname()
            and isinstance(admission.get('gpu_uuid'), str) and admission['gpu_uuid'].startswith('GPU-')
            and isinstance(admission.get('runtime_fingerprint_sha256'), str)
            and len(admission['runtime_fingerprint_sha256']) == 64
            and admission.get('source_sha256') == source_hashes
            and admission.get('safe_role_sha256') == {k: v['sha256'] for k, v in roles.items()}
            and admission.get('science_config_sha256') == canonical,
            'Actual root host/GPU/runtime/code/data/config admission for this action required')
    return {'stage': 'F', 'K': None, 'pilot_provenance_read': False, 'study_kind': STUDY,
        'action': action, 'source_sha256': source_hashes,
        'safe_role_sha256': {k: v['sha256'] for k, v in roles.items()},
        'root_runtime_admission': admission, 'baseline_policy': 'fresh_required'}
