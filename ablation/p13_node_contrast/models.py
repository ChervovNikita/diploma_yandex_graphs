"""Keep the original node backbone and capture its live final prehead."""
import torch
from torch import nn

from models import TABMModel
from ablation.common.contrastive import unit_contrastive_loss, private_gradient_step
from ablation.common.tabm_trainer import member_loss


class NodeContrastiveModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.register_buffer('auxiliary_step', torch.zeros((), dtype=torch.long))
        self.register_buffer('auxiliary_seed', torch.tensor(torch.initial_seed() % (2**63 - 1)))
        self.last_auxiliary = None

    def forward(self, graph, x, tabm_seed):
        return self.base(graph, x, tabm_seed=tabm_seed)

    def prehead_forward(self, data, member):
        captured = []
        handle = self.base.output_be_block.register_forward_pre_hook(
            lambda module, arguments: captured.append(arguments[0]))
        try:
            output = self.base(data, data.x, tabm_seed=member)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError('Expected one original final prehead')
        return output, captured[0]

    def contrastive_train_step(self, data, optimizer, is_binary, members):
        if members != 4:
            raise ValueError('The fixed contrastive extension requires four original paths')
        outputs = [self.prehead_forward(data, member) for member in range(members)]
        task = torch.stack([member_loss(out, data, is_binary) for out, _ in outputs]).mean()
        features = torch.stack([feature[data.train_mask] for _, feature in outputs])
        auxiliary, diagnostic = unit_contrastive_loss(
            features, data.y[data.train_mask],
            int(self.auxiliary_seed) + 104729 * int(self.auxiliary_step))
        block = self.base.input_be_block
        private_gradient_step(self, optimizer, task, auxiliary, (block.R, block.S, block.B))
        self.auxiliary_step.add_(1)
        self.last_auxiliary = dict(diagnostic, loss=float(auxiliary.detach()))
        return task.detach()


def build_model(variant, model_name, num_layers, input_dim, hidden_dim, output_dim, m, device,
                hidden_dim_multiplier=1, num_heads=8, normalization='LayerNorm', dropout=0.2):
    if variant != 'gnnm_contrast' or m != 4:
        raise ValueError('Only the four-path fixed-config node contrastive arm is defined')
    base = TABMModel(model_name=model_name, num_layers=num_layers, input_dim=input_dim,
                     hidden_dim=hidden_dim, output_dim=output_dim,
                     hidden_dim_multiplier=hidden_dim_multiplier, num_heads=num_heads,
                     normalization=normalization, dropout=dropout, tabm_inits=m, device=device)
    return NodeContrastiveModel(base).to(device)
