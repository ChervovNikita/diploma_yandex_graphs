"""New frozen exploratory family; no numerical imports or active releases."""
SEEDS = (9101, 9203, 9307)
CONDITIONS = ('single_native', 'single_mean4_dropout', 'independent4_own')
SPLIT = dict(dataset='PubMed', split_protocol='class_stratified_floor60_20_20',
             split_identity='PubMed-class-stratified-floor60-20-20', split_seed=190111,
             full_graph_shape=[19717, 500], edge_shape=[2, 88648],
             TRAIN_count=11829, VALID_count=3942,
             TRAIN_class_counts=[2461, 4643, 4725], VALID_class_counts=[820, 1547, 1575])
SELECTOR = 'first strict maximum own raw-logit max(1)[1] VALID accuracy'
MAX_EPOCHS, PATIENCE = 2000, 250
SERVER_HOST = 'anogena-2-0'
SERVER_GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SERVER_PHASE = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930'
LIMITS = dict(GPU_bytes=32*1024**3, RSS_bytes=16*1024**3,
              output_bytes=512*1024**2, log_bytes=8*1024**2,
              automatic_retry=False, external_active_seconds=9000, external_cleanup_seconds=10)
ENGINEERING_LIMITS = dict(LIMITS, external_active_seconds=600)
SERVING = dict(warmup_calls=3, measured_calls=10, device='cuda:0',
               include_softmax_probability_mean=True, optimizer_or_VALID_selection=False,
               timing='synchronized CPU wall and CUDA events; construction/restore/preprocessing separate')


def body_roster(seed, condition):
    if condition not in CONDITIONS: raise ValueError('Frozen condition required')
    return [dict(body_id=f'body{m}', initialization_seed=seed+1009*m,
                 factual_dropout_seeds=[seed+1009*m+300001+1009*v
                                        for v in range(4 if condition == 'single_mean4_dropout' else 1)],
                 max_epochs=MAX_EPOCHS, patience=PATIENCE, selector=SELECTOR,
                 fresh_full_native_fit=True, reuse_other_record=False)
            for m in range(4 if condition == 'independent4_own' else 1)]


def roster():
    return [dict(record_id=f'seed{s}__{c}', condition=c, seed=s, enabled=False,
                 body_roster=body_roster(s,c), max_epochs=MAX_EPOCHS, patience=PATIENCE,
                 selector=SELECTOR, limits=dict(LIMITS), serving=dict(SERVING),
                 TEST_access=False, CORE=False, masking=False, HPO=False, **SPLIT)
            for s in SEEDS for c in CONDITIONS]


def forecast():
    return dict(schema='PubMed-strong-reference-symbolic-cost-v1', measured=False,
                new_groups=9, retained_shared_own_anchors=3, comparison_groups=12,
                fresh_full_body_trajectories=18, maximum_private_epochs=36000,
                maximum_training_fullgraph_forwards=54000, maximum_backwards=54000,
                maximum_Adam_steps=36000, maximum_regular_VALID_forwards=36000,
                selected_private_restore_forwards=18, selected_assembled_I4_forwards=12,
                serving_warmup_forwards=54, serving_measured_forwards=180,
                body_constructions=30, fit_body_constructions=18, I4_assembly_body_constructions=12,
                preprocessing_banks=21, maximum_selected_checkpoint_writes=36000,
                saved_new_float32_logit_bytes=18*19717*3*4,
                native_predictor_parameters=2069875, mean4_predictor_parameters=2069875,
                I4_predictor_parameters=8279500, anchor_predictor_parameters=2125647,
                anchor_extra_factor_coordinates=55772, decoder_parameters=0,
                complete9_additive_active_plus_cleanup_cap_seconds=9*9010,
                TRAIN_only_qualification_owners=3, qualification_active_plus_cleanup_cap_seconds=3*610,
                qualification_training_body_trajectories=6, qualification_training_forwards=9,
                qualification_backwards=9, qualification_Adam_steps=6,
                qualification_serving_forwards=10, qualification_body_constructions=10,
                wall_seconds=None, GPU_peak_bytes=None, RSS_peak_bytes=None,
                historical_costs='PRIOR_COSTS.json is measured history only; it cannot certify fresh fits or own selectors',
                actual_costs_required='Count every new fit, qualification, evaluation, checkpoint write, I4 assembly, serving call, comparison and owner cleanup, including failures; no cost subtraction for deterministic coincidences',
                peak_limit='Native activations, attention kernels, Adam states, fullgraph preprocessing, checkpoint copies and allocator peaks must be measured under the exact source/runtime',
                stop_rule='A cap hit is a retained failure. Never shorten a private horizon, substitute a seed, reuse a fit, or compare a partial family.')


def description():
    return dict(schema='PubMed-strong-reference-disabled-plan-v1', enabled=False,
                question='Four-factual-dropout gradient averaging versus a competent ordinary-reference gap',
                new_records=roster(), fresh_full_body_trajectories=18, comparison_groups=12,
                post_screen_exploration=True, original_paper_scores_recomputed=False,
                CORE_continuation_eligible=False, further18_activated=False,
                numerical_calls_performed=0, costs=forecast())
