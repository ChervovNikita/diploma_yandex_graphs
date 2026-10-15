"""One missing MolBACE arm, composed from existing versioned mechanisms.

No fit loop, native operator, scientific setting, or model is run at import.
Root must review/admit the three frozen jobs before existing run_fit execution.
"""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

METHOD = 'gnnm_residual_only_private_bn_v1'
NATIVE_CONFIG = {'JK': 'last', 'batch_size': 32, 'dataset': 'ogbg-molbace',
                 'drop_ratio': 0.5, 'emb_dim': 300, 'feature': 'full', 'gnn': 'gin',
                 'graph_pooling': 'mean', 'lr': 0.001, 'num_layer': 5, 'residual': False}
POLICY = {'direction': 'max', 'evaluate_TRAIN_each_epoch': False, 'maximum_epochs': 1000,
          'metric': 'rocauc', 'minimum_epochs': 100, 'patience': 50, 'ties': 'first',
          'track': 'matched_pilot'}
PINS = {
    'portable_private_input_v2/MANIFEST.json': '21ad8f2c644a1833f0015dfc4c0c0193bcfc9598b6f84d98b9d686330bed8115',
    'molecular/session_extension.py': '8e2cc88b08d8a704c198af3d84a7a65cc5e63262fa3426776aaf4b83ecd1fe9d',
    'molecular/objectives.py': '75e7d59f8360aa2688b9cd974e9e28863e475c683f3dd073e2eacd6665aff71a',
    'adapters/input_residual.py': 'c0def85281f68e91461474812202a6cffb1df2e4dabcf2c78b35d84c76f1776d',
    'private_bn_state_v3/session_extension.py': '3d9d9c47d2dd421d7bb54560f7d4466fff839c6d360e8571328dcf3f7874e14c',
    'private_bn_state_v3/banks.py': 'd90a74bf98be32408cad306f35ab5ebd1ec80153981a0b7f92cba07e7f447fea',
}


def settings_for():
    return {'portable': {'rank_cap': 8,
                         'frame_policy': 'disjoint_if_fit_else_separate_orthonormal_frames',
                         'manifest_sha256': PINS['portable_private_input_v2/MANIFEST.json']},
            'ingredient': {'auxiliary': False, 'input_residual': False, 'adapter_rank': 8,
                           'dependency_sha256': {k: PINS[k] for k in
                                                 ('molecular/objectives.py', 'adapters/input_residual.py')}},
            'banks_sha256': PINS['private_bn_state_v3/banks.py']}


def _load(root, relative):
    path = root / relative
    name = '_residual_only_bn_' + hashlib.sha256(path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def transform(session, job):
    if job.get('procedure') != METHOD or job.get('method_settings') != settings_for() \
            or job.get('family') != 'ogb' or job.get('base_procedure') != 'gnnm_boundary_native' \
            or job.get('seed') not in (101, 202, 303) or job.get('split') != 0 \
            or job.get('TEST_access') is not False or job.get('policy') != POLICY \
            or session.config != NATIVE_CONFIG or session.procedure != 'gnnm_boundary_native' \
            or session.updates != 0:
        raise ValueError('Only the frozen fresh three-seed MolBACE missing arm is admitted')
    if hasattr(session, 'extension_metadata') or hasattr(session.model, 'route_bn_state'):
        raise ValueError('A fresh unextended boundary session is required')
    root = Path(__file__).resolve().parents[1]
    for relative, expected in PINS.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Retained mechanism source changed: ' + relative)
    native, optimizer = session.model.native, session.optimizer
    native_parameters = tuple(native.parameters())
    native_count = sum(p.numel() for p in native_parameters)
    removed_count = sum(p.numel() for p in session.model.private_parameters())
    portable_job = copy.deepcopy(job)
    portable_job['procedure'] = 'portable_input_residual_only_v1'
    portable_job['method_settings'] = copy.deepcopy(job['method_settings']['portable'])
    _load(root, 'portable_private_input_v2/extension.py').transform(session, portable_job)
    portable_metadata = copy.deepcopy(session.portable_private_input_metadata)
    ingredient_job = copy.deepcopy(job)
    ingredient_job['method_settings'] = copy.deepcopy(job['method_settings']['ingredient'])
    _load(root, 'molecular/session_extension.py').transform(session, ingredient_job)
    own_update = session.train_update
    bn_job = copy.deepcopy(job)
    bn_job['method_settings'] = {'banks_sha256': PINS['private_bn_state_v3/banks.py']}
    _load(root, 'private_bn_state_v3/session_extension.py').transform(session, bn_job)
    model = session.model
    bn_metadata = copy.deepcopy(model.route_bn_state.metadata())
    if model.native is not native or session.optimizer is not optimizer or session.train_update != own_update \
            or hasattr(model, 'input_bank') or hasattr(model, 'output_bank') \
            or len(tuple(native.parameters())) != len(native_parameters) \
            or any(current is not old for current, old in zip(native.parameters(), native_parameters)):
        raise ValueError('Only private route parameterization/state may change')
    private = tuple(model.private_parameters())
    if sum(p.numel() for p in private) != 19200 or len(private) != 2 \
            or any(p.numel() and p.count_nonzero().item() for p in (model.input_residual.up,)):
        raise ValueError('Exactly rank8 four-route zero-up private residual required')
    model.assert_source_identity()
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    private_bytes = sum(p.numel() * p.element_size() for p in private)
    accounting = {'native_trainable_parameters': native_count,
                  'private_trainable_parameters': 19200,
                  'total_trainable_parameters': sum(p.numel() for p in model.parameters()),
                  'removed_BE_parameters': removed_count,
                  'extra_fraction_of_native': 19200 / native_count,
                  'native_backbone_copies': 1, 'additional_BN_learned_parameters': 0,
                  'native_buffer_bytes': sum(b.numel() * b.element_size() for b in native.buffers()),
                  'private_BN_buffer_bytes': bn_metadata['bank_bytes'],
                  'parameter_bytes': parameter_bytes, 'private_parameter_bytes': private_bytes,
                  'Adam_two_moment_bytes_if_all_parameters_receive_gradients': 2 * parameter_bytes,
                  'extra_Adam_two_moment_bytes': 2 * private_bytes,
                  'optimizer_step_scalar_and_serialization_overhead': 'actual checkpoint, separate from dense moment estimate'}
    binding = {'schema': 'molbace-residual-only-private-bn-v1', 'method': METHOD,
               'seed': job['seed'], 'settings': copy.deepcopy(job['method_settings']), 'source_sha256': PINS}
    combined = {**binding, 'portable': portable_metadata, 'normalization': bn_metadata,
                'parameter_and_state_accounting': accounting,
                'training_update': 'unchanged molecular/session_extension.py own mean BCE, auxiliary=False',
                'native_initial_eval_function': 'zero residual and cloned initial per-route BN buffers; BE fully removed',
                'TEST_accessed': False, 'quality_or_novelty_claim': False}
    original_metadata, original_state, original_load = session.metadata, session.state, session.load_state

    def metadata(self):
        value = original_metadata()
        value['candidate_extension'] = copy.deepcopy(combined)
        return value

    def state(self):
        value = original_state()
        value['residual_only_private_bn_binding'] = copy.deepcopy(binding)
        private_ids = {id(p) for p in private}
        optimizer_bytes = private_optimizer_bytes = 0
        for parameter, fields in getattr(self.optimizer, 'state', {}).items():
            for tensor in fields.values():
                if hasattr(tensor, 'numel') and hasattr(tensor, 'element_size'):
                    size = tensor.numel() * tensor.element_size()
                    optimizer_bytes += size
                    if id(parameter) in private_ids:
                        private_optimizer_bytes += size
        value['residual_only_private_bn_resource_accounting'] = {
            **copy.deepcopy(accounting),
            'actual_optimizer_state_tensor_bytes': optimizer_bytes,
            'actual_private_optimizer_state_tensor_bytes': private_optimizer_bytes,
            'scope': 'All live optimizer tensors including scalar steps at this checkpoint; serialization overhead separate'}
        return value

    def load_state(self, value):
        if value.get('residual_only_private_bn_binding') != binding:
            raise ValueError('Checkpoint method, seed, source or BN composition differs')
        source = dict(value)
        source.pop('residual_only_private_bn_binding')
        source.pop('residual_only_private_bn_resource_accounting')
        original_load(source)

    session.extension_metadata = copy.deepcopy(combined)
    session.metadata = types.MethodType(metadata, session)
    session.state = types.MethodType(state, session)
    session.load_state = types.MethodType(load_state, session)
    return session
