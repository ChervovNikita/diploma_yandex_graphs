"""The exact main-table models used by the member diversity study."""

from models import Model, TABMModel

from ablation.p05_hparam_grid.models import PlainModel


DROPOUT = 0.2
NUM_HEADS = 8


def build_model(method, model_name, num_layers, input_dim, hidden_dim, output_dim, members, device):
    arguments = dict(
        model_name=model_name, num_layers=num_layers, input_dim=input_dim,
        hidden_dim=hidden_dim, output_dim=output_dim, hidden_dim_multiplier=1,
        num_heads=NUM_HEADS, normalization='LayerNorm', dropout=DROPOUT,
    )
    if method == 'gnnm':
        return TABMModel(tabm_inits=members, device=device, **arguments).to(device)
    return PlainModel(Model(**arguments)).to(device)
