"""Full ablations of ensemble size and initialization, on the archived ablation model."""

from ablation.models_ablation import TABMAblationModel


VARIANTS = {'m2': 2, 'm4': 4, 'm8': 8, 'xavier_all': 4, 'ones_all': 4}
# m4 only verifies this code path against the main table; M=4 itself comes from p01.
DEFAULT_VARIANTS = ['m2', 'm8', 'xavier_all']


def build_ablation_model(model_name, num_layers, input_dim, hidden_dim, output_dim, m,
                         init_scheme, device):
    return TABMAblationModel(
        model_name=model_name, num_layers=num_layers, input_dim=input_dim,
        hidden_dim=hidden_dim, output_dim=output_dim, hidden_dim_multiplier=1,
        num_heads=8, normalization='LayerNorm', dropout=0.2, tabm_inits=m,
        init_scheme=init_scheme, device=device,
    ).to(device)


def build_model(variant, model_name, num_layers, input_dim, hidden_dim, output_dim, m, device):
    scheme = variant if variant in ('xavier_all', 'ones_all') else 'default'
    return build_ablation_model(model_name, num_layers, input_dim, hidden_dim, output_dim, m,
                                scheme, device)
