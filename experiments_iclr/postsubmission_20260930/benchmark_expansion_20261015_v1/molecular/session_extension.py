"""Optional real-data molecular candidate updates for the bounded pilot."""
import importlib.util
import hashlib
import math
from pathlib import Path
import sys
import types

import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def transform(session, job):
    settings = job.get('method_settings', job.get('extension_settings'))
    if not isinstance(settings, dict):
        raise ValueError('Frozen method settings are required')
    root = Path(__file__).resolve().parents[1]
    for relative, expected in settings.get('dependency_sha256', {}).items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen extension dependency changed: ' + relative)
    metadata = {'schema': 'molecular-candidate-session-v1',
                'method_id': job['procedure'], 'TEST_accessed': False}
    if settings.get('input_residual', False):
        residual = load(root / 'adapters/input_residual.py', 'gnnm_pilot_input_residual')
        metadata['input_residual'] = residual.attach(
            session, seed=job['seed'], rank=int(settings.get('adapter_rank', 8)))
    auxiliary = None
    if settings.get('auxiliary', False):
        module = load(root / 'molecular/objectives.py', 'gnnm_pilot_graph_objective')
        auxiliary = module.MolecularObjective(module.ObjectiveConfig(
            mode='alignment_variance', temperature=0.2, coefficient=0.05,
            std_fraction=0.5, gradient_scope='all'))
        metadata['auxiliary'] = auxiliary.state_dict()
    metadata['source_mean_own_loss_preserved'] = True
    session.extension_metadata = metadata
    session.procedure = 'gnnm_boundary_native'

    def train_update(self):
        self.model.train()
        batch, positions, skipped = self._next_train_batch()
        self.last_train_positions = positions.clone()
        self.optimizer.zero_grad()
        routes = [self.model.run_member(member, batch) for member in range(4)]
        own_losses = torch.stack([self._native_loss(route[0], batch) for route in routes])
        own = own_losses.mean()
        loss, diagnostics = own, None
        if auxiliary is not None:
            extra = auxiliary(prehead=torch.stack([route[2] for route in routes]),
                              labels=batch.y, graph_ids=batch.smoke_graph_id.reshape(-1),
                              role='TRAIN')
            loss = extra.scalar_total(own)
            diagnostics = extra.diagnostics
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite frozen pilot objective')
        loss.backward()
        self.optimizer.step()
        self.model.assert_source_identity()
        self.updates += 1
        return {'loss': float(loss.detach()), 'native_own_loss': float(own.detach()),
                'native_member_losses': own_losses.detach().cpu().tolist(),
                'optimizer_updates': 1, 'update': self.updates,
                'TRAIN_graph_ids': self.train_ids[positions].tolist(),
                'native_skipped_batches': skipped, 'native_batch_graphs': len(positions),
                'native_route_forwards': 4,
                'work': {'optimizer_transitions': 1, 'reverse_collections': 1,
                         'extra_backbone_forwards': 0},
                'auxiliary_diagnostics': diagnostics}

    session.train_update = types.MethodType(train_update, session)
    original_metadata = session.metadata
    def combined_metadata(self):
        value = original_metadata()
        value['candidate_extension'] = metadata
        return value
    session.metadata = types.MethodType(combined_metadata, session)
    return session
