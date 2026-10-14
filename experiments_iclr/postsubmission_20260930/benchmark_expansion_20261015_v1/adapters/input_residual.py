"""Private nonlinear input residuals around an unchanged shared native encoder.

This is an adapter/LoRA-inspired candidate, not a newly established principle.
For molecular models the insertion point is the complete native AtomEncoder
output, before the original message-passing body. No categorical IDs, source
parameters, bond encoders, graph topology or native optimizer options change.
"""
import types

import torch


def attach(session, *, seed, rank=8):
    model = session.model
    if not hasattr(model, 'native') or not hasattr(model, '_member'):
        raise TypeError('The declared molecular four-route boundary wrapper is required')
    if hasattr(model, 'private_input_residual'):
        raise ValueError('Private input residual already attached')
    width = model.native.gnn_node.atom_encoder.atom_embedding_list[0].embedding_dim
    if rank <= 0 or 4 * rank > width:
        raise ValueError('Four orthogonal private subspaces must fit the native input width')
    generator = torch.Generator(device='cpu').manual_seed(int(seed) + 901003)
    # One orthogonal frame gives distinct initial down-projection subspaces.
    basis, _ = torch.linalg.qr(torch.randn(width, 4 * rank, generator=generator),
                               mode='reduced')

    class PrivateInputResidual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            reference = next(model.native.parameters())
            self.down = torch.nn.Parameter(basis.T.reshape(4, rank, width).to(reference))
            self.up = torch.nn.Parameter(reference.new_zeros(4, width, rank))

        def forward(self, encoded, member):
            hidden = torch.nn.functional.gelu(
                torch.nn.functional.linear(encoded, self.down[member]))
            return encoded + torch.nn.functional.linear(hidden, self.up[member])

    adapter = PrivateInputResidual()
    model.add_module('private_input_residual', adapter)
    source_private = model.private_parameters
    def private_parameters(self):
        return tuple(source_private()) + tuple(self.private_input_residual.parameters())
    model.private_parameters = types.MethodType(private_parameters, model)
    encoder = model.native.gnn_node.atom_encoder
    def post(module, inputs, output):
        member = model._member
        return output if member is None else adapter(output, member)
    model._handles.append(encoder.register_forward_hook(post))
    source_weight = encoder.atom_embedding_list[0].weight
    owners = [group for group in session.optimizer.param_groups
              if any(parameter is source_weight for parameter in group['params'])]
    if len(owners) != 1:
        raise ValueError('Native input parameter requires exactly one optimizer group')
    owners[0]['params'].extend(adapter.parameters())
    model.assert_source_identity()
    metadata = {'mechanism': 'encoded + U_member * GELU(V_member * encoded)',
                'location': 'native full AtomEncoder output before shared message passing',
                'rank': rank, 'members': 4,
                'initializer': 'one isolated orthogonal down-frame; zero up matrices',
                'initial_function_change_relative_to_base_GNNM': 'exact zero added residual',
                'extra_parameters': sum(p.numel() for p in adapter.parameters()),
                'native_parameters_replaced': 0, 'extra_backbone_copies': 0,
                'optimizer_group': 'existing native atom embedding group',
                'novelty_claim': False}
    return metadata
