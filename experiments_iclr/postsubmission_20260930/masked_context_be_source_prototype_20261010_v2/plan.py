"""Fixed source proposal and arithmetic forecast; no numerical imports."""
import json

SEEDS = (9101, 9203, 9307)
CONDITIONS = (
    'shared4_own', 'shared4_masked_ce', 'shared4_core',
    'shared4_core_shuffled', 'shared4_core_rewired', 'single_native',
    'independent4_native', 'single_four_view_core', 'untied4_shared_decoder_core')
STAGE1 = ('shared4_own', 'shared4_core', 'independent4_native')
NATIVE_MONO = dict(dataset='pubmed', base='mono', K=2, hidden=256, nlayer=2,
                   n_head=8, d_ffn=32, q=1.6, multi=2., dropout=.5, dprate=.8,
                   lr=.005, weight_decay=.001, attn_lr=.0005, attn_wd=1e-8,
                   max_epochs=2000, native_early_stopping=250,
                   num_features=500, num_classes=3)
RULE = dict(members=4, factual_ce=1., masked_ce=.5, core=.1,
            temperature=.2, negative_count=32, norm_floor=1e-8,
            factor_initialization='unit', partition_seed=190101,
            negative_seed_offset=700001, ownership_seed_offset=800001,
            decoder='one common Linear(hidden, input_features, bias=True)',
            decoder_optimizer='base Adam; shared decoder always uses mean route reconstruction data gradient',
            serving='factual mean member class probabilities')


def condition(name):
    if name not in CONDITIONS:
        raise ValueError('Exact fixed condition required')
    shared = name.startswith('shared4')
    untied = name in ('independent4_native', 'untied4_shared_decoder_core')
    masked = name not in ('shared4_own', 'single_native', 'independent4_native')
    core = masked and name != 'shared4_masked_ce'
    return dict(name=name, members=4 if shared or untied else 1,
                bodies=4 if untied else 1, shared_BE=shared, untied=untied,
                masked=masked, core=core,
                masked_views=4 if masked else 0,
                factual_views=4 if shared or untied else 1,
                ownership_shuffle=name == 'shared4_core_shuffled',
                auxiliary_rewire=name == 'shared4_core_rewired',
                all_view_single=name == 'single_four_view_core',
                genuine_independent=name == 'independent4_native',
                shared_decoder=core)


def plan():
    return dict(status='inactive source proposal; no scientific execution',
        original_population='PubMed complete graph / official Planetoid split, inactive',
        concrete_amendment='PubMed complete graph / exact native PolyFormer class-balanced 60/20/20 split generation; pending root freeze',
        published_recipe=NATIVE_MONO, rule=RULE, seeds=list(SEEDS),
        stage1=[dict(condition=c, seed=s) for s in SEEDS for c in STAGE1],
        stage2=[dict(condition=c, seed=s) for s in SEEDS for c in CONDITIONS if c not in STAGE1],
        stage1_records=9, stage2_records=18, positive_complete_family_records=27,
        scientific_runner_enabled=False, measured_resource_evidence=False,
        original_paper_scores_recomputed=False, confirmation_or_novelty_claimed=False)


def resource_forecast(n=19717, f=500, c=3):
    """Shape ledger, not a measured GPU peak or ETA."""
    h, tokens, layers, ff, heads, wide = 256, 3, 2, 32, 8, 512
    dense_body = f*h+h + h*h+h + h*c+c
    per_layer = tokens*(h*wide+wide+wide*h+h) + 2*h*h + heads*tokens + 2*h + (h*ff+ff+ff*h+h) + 2*h
    native = dense_body + layers*per_layer
    per_route_factors = (f+h)+(h+h)+(h+c)+layers*(tokens*2*(h+wide)+2*(h+h)+2*(h+ff))
    shared = native + 4*per_route_factors
    decoder = h*f+f
    bytes_per_bank = (tokens*n*f)*4
    byte_ledger = dict(factual_and_four_masked_feature_banks=5*bytes_per_bank,
        one_poly_token_hidden_tensor=n*tokens*h*4,
        one_attention_score_tensor=n*heads*tokens*tokens*4,
        one_route_masked_reconstructions_approximately=(n//4+1)*f*4,
        one_512_anchor_negative_gather=512*32*f*4,
        full_route_negative_gathers_retained_by_autograd_approximately=(n//4+1)*32*f*4,
        full_negative_id_bank_approximately=(n//4+1)*32*8,
        shared_core_parameter_gradient_Adam_lower_bound=16*(shared+decoder),
        untied_core_parameter_gradient_Adam_lower_bound=16*(4*native+decoder))
    counts = {c: dict(training_fullgraph_forwards=condition(c)['factual_views']+condition(c)['masked_views'],
                     serving_fullgraph_forwards=condition(c)['members'],
                     decoder_parameters=decoder if condition(c)['core'] else 0,
                     predictor_parameters=shared if condition(c)['shared_BE'] else condition(c)['bodies']*native)
              for c in CONDITIONS}
    stage1_forward_per_seed = sum(counts[c]['training_fullgraph_forwards'] for c in STAGE1)
    all_forward_per_seed = sum(row['training_fullgraph_forwards'] for row in counts.values())
    return dict(kind='symbolic dimensions, no empirical timing or memory qualification',
        native_parameters=native, shared_BE4_parameters=shared, private_BE4_factors=4*per_route_factors,
        common_decoder_parameters=decoder, bytes=byte_ledger, counts_per_update=counts,
        stage1_maximum_training_forwards=3*2000*stage1_forward_per_seed,
        complete27_maximum_training_forwards=3*2000*all_forward_per_seed,
        max_epoch_is_envelope='Native early-stop250 is a pending selector/protocol binding, not a shortened toy horizon',
        wall_seconds=None, GPU_peak_bytes=None,
        missing_memory='Full native activations/autograd tape, temporary stacks, attention kernels, gradients, allocator and preprocessing transient peaks',
        timing_recipe='Measure complete fullgraph TRAIN-only steps under identical runtime and outer supervision; multiply per-condition measured update/serving rates by frozen horizon, then add preparation, evaluation, checkpoint and transfer costs.')


if __name__ == '__main__':
    print(json.dumps(dict(proposal=plan(), forecast=resource_forecast()), indent=2, sort_keys=True))
