"""One fixed learning rule with required controls; no hypothesis/parameter grid."""
CONDITIONS = ('own_floor', 'private_cmcl', 'all_block_cmcl', 'vanilla_cmcl', 'private_uniform', 'private_constant_credit')
SEEDS = (9101, 9203, 9307)
NATIVE = dict(dataset='pubmed', base='mono', K=2, hidden=256, nlayer=2,
              n_head=8, d_ffn=32, q=1.6, multi=2., dropout=.5, dprate=.8,
              lr=.005, weight_decay=.001, attn_lr=.0005, attn_wd=1e-8,
              num_features=500, num_classes=3)
MEMBERS, CLASSES, OWNER_K, BETA, LAMBDA = 4, 3, 3, .75, 1.
MAXIMUM_EPOCHS, PATIENCE = 2000, 250
SPLIT_SEED = 190111


def condition(name):
    if name not in CONDITIONS:
        raise ValueError('Only declared own/CMCL/private-uniform controls')
    return dict(name=name, own_floor=name != 'vanilla_cmcl',
                assignment=name in ('private_cmcl', 'all_block_cmcl', 'vanilla_cmcl'),
                auxiliary='CMCL' if name in ('private_cmcl', 'all_block_cmcl', 'vanilla_cmcl')
                          else 'uniform' if name == 'private_uniform' else 'constant' if name == 'private_constant_credit' else None,
                private_auxiliary_only=name in ('private_cmcl', 'private_uniform', 'private_constant_credit'))
