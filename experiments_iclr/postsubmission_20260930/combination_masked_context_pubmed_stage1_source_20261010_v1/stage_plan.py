"""Frozen nine-record source plan. Stdlib only; all releases are disabled."""
SEEDS = (9101, 9203, 9307)
CONDITIONS = ('shared4_own', 'shared4_core', 'independent4_native')
SPLIT = dict(dataset='PubMed', split_protocol='class_stratified_floor60_20_20',
             split_identity='PubMed-class-stratified-floor60-20-20', split_seed=190111,
             full_graph_shape=[19717, 500], TRAIN_count=11829, VALID_count=3942,
             TRAIN_class_counts=[2461, 4643, 4725], VALID_class_counts=[820, 1547, 1575])
SELECTOR = 'first strict maximum factual mean-member-probability pooled VALID accuracy'
MAX_EPOCHS, PATIENCE = 2000, 250
SERVER_HOST = 'anogena-2-0'
SERVER_GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SERVER_PHASE = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930'
LIMITS = dict(GPU_bytes=32*1024**3, RSS_bytes=16*1024**3,
              output_bytes=512*1024**2, log_bytes=8*1024**2,
              automatic_retry=False, external_active_seconds=9000,
              external_cleanup_seconds=10)
QUALIFICATION_UPDATES = dict(shared4_core=3, shared4_own=1, independent4_native=1)


def roster():
    return [dict(record_id=f'seed{seed}__{name}', condition=name, seed=seed,
                 members=4, bodies=4 if name == 'independent4_native' else 1,
                 enabled=False, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                 selector=SELECTOR, TEST_access=False, limits=dict(LIMITS), **SPLIT)
            for seed in SEEDS for name in CONDITIONS]


def description():
    return dict(schema='masked-context-PubMed-disabled-stage1-plan-v1',
                enabled=False, records=roster(), complete_family_count=9,
                further18_activated=False, numerical_calls_performed=0,
                independent_selector_limit='Primary common bank selector; individually selected ordinary independent members remain required for an eventual superiority claim.',
                competence_limit='Three TRAIN updates establish bounded engineering behavior only. Complete nine records and selected member/class readouts precede continuation; capable singles, own-selected independent ensembles and unused confirmation remain required for stronger claims.')


def forecast(native_forecast):
    return dict(schema='masked-context-stage1-symbolic-cost-v1', measured=False,
                native_shape_ledger=native_forecast, records=9, native_body_constructions=18,
                maximum_bank_updates=18000, maximum_training_fullgraph_forwards=96000,
                maximum_regular_VALID_member_forwards=72000,
                maximum_backward_calls=72000, maximum_Adam_steps=42000,
                selected_factual_member_forwards=36, selected_masked_member_forwards=12,
                preprocessing_banks=21, saved_factual_float32_logit_bytes=8517744,
                saved_masked_float32_logit_bytes=2839248,
                maximum_selected_checkpoint_writes=18000,
                wall_seconds=None, GPU_peak_bytes=None, RSS_peak_bytes=None,
                limits=LIMITS, scientific_wall_cap_pending=False,
                complete9_additive_active_plus_cleanup_cap_seconds=81090,
                caps_prospective_and_root_admission_pending=True,
                timing_rule='After root qualification, use same-runtime per-condition complete-update and factual-serving times times2000, plus preprocessing, checkpoint I/O, selected diagnostics and owner overhead. Do not infer a complete-run ETA from three updates.',
                omitted_peaks='Native autograd tapes, negative reconstruction gathers, stacked serving logits, CUDA allocator, full graph preprocessing, checkpoint copies and optimizer state.',
                stop_rule='Any frozen wall/resource cap is a preserved failure, never a completed shortened scientific record or an automatic retry.')
