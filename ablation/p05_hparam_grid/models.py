"""Main-table models with separately configurable attention dropout and head count."""

from torch import nn

from models import Model, TABMModel


class PlainModel(nn.Module):
    """A single ordinary GNN behind the interface used by the trainer."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, graph, x, tabm_seed):
        return self.model(graph, x)


def set_attention_dropout(model, probability):
    """Override only the attention dropout of the graph layers that have one."""
    for block in model.residual_modules:
        conv = getattr(block.module, 'conv', None)
        if conv is not None and hasattr(conv, 'dropout'):
            conv.dropout = probability


def build_model(method, model_name, num_layers, input_dim, hidden_dim, output_dim, members,
                device, dropout, attention_dropout, num_heads):
    arguments = dict(
        model_name=model_name, num_layers=num_layers, input_dim=input_dim,
        hidden_dim=hidden_dim, output_dim=output_dim, hidden_dim_multiplier=1,
        num_heads=num_heads, normalization='LayerNorm', dropout=dropout,
    )
    if method == 'gnnm':
        model = TABMModel(tabm_inits=members, device=device, **arguments)
        set_attention_dropout(model, attention_dropout)
    else:
        inner = Model(**arguments)
        set_attention_dropout(inner, attention_dropout)
        model = PlainModel(inner)
    return model.to(device)
