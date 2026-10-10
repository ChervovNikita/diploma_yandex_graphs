"""Frozen positive-continuation controls. Stdlib only; source is disabled."""
SEEDS=(9101,9203,9307)
STAGE1=('shared4_own','shared4_core','independent4_native')
CONDITIONS=('shared4_masked_ce','shared4_core_shuffled','shared4_core_rewired',
            'single_native','single_four_view_core','untied4_shared_decoder_core')
ALL_CONDITIONS=('shared4_own','shared4_masked_ce','shared4_core','shared4_core_shuffled',
    'shared4_core_rewired','single_native','independent4_native','single_four_view_core','untied4_shared_decoder_core')
SPLIT=dict(dataset='PubMed',split_protocol='class_stratified_floor60_20_20',
    split_identity='PubMed-class-stratified-floor60-20-20',split_seed=190111,
    full_graph_shape=[19717,500],TRAIN_count=11829,VALID_count=3942,
    TRAIN_class_counts=[2461,4643,4725],VALID_class_counts=[820,1547,1575])
SELECTOR='first strict maximum factual mean-member-probability pooled VALID accuracy'
MAX_EPOCHS,PATIENCE=2000,250
SERVER_HOST='anogena-2-0'
SERVER_GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SERVER_PHASE='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930'
LIMITS=dict(GPU_bytes=32*1024**3,RSS_bytes=16*1024**3,output_bytes=512*1024**2,
    log_bytes=8*1024**2,automatic_retry=False,external_active_seconds=9000,external_cleanup_seconds=10)
QUALIFICATION_UPDATES={name:1 for name in CONDITIONS}
REWIRE=dict(double_edge_swap_seed=190121,accepted_swaps_per_nonself_undirected_edge=10,
    maximum_proposals_per_requested_swap=100,one_common_graph_for_all_optimizer_seeds=True,
    preserves='complete node universe, every nonself degree, isolated nodes, undirected edge count and original self edges',
    connected_components_preserved=False,labels_used=False,quality_or_overlap_selected=False)
PURPOSES={
    'shared4_masked_ce':'Core minus masked CE isolates added contrastive reconstruction under the same owned masking and factual/masked CE exposure; masked CE minus own measures masking/extra CE opportunity.',
    'shared4_core_shuffled':'Core minus shuffled tests persistent ownership against per-update permutation of the same four fixed mask banks. Mask populations, all four exposures and scalar weights remain fixed.',
    'shared4_core_rewired':'Core minus rewired tests auxiliary original-graph alignment after complete degree-preserving rewiring. Factual graph/serving and missing-feature ownership remain unchanged. Component/homophily/neighborhood structure can change, limiting causal interpretation.',
    'single_native':'Complete capable native single reference at the same seed, architecture, optimizer, full horizon and representative roles; source matching alone does not prove competence.',
    'single_four_view_core':'Single all-view minus native tests auxiliary learning without a bank; core minus all-view single compares shared-bank utility against one full body receiving every auxiliary view with the same mean auxiliary coefficients.',
    'untied4_shared_decoder_core':'Untied core minus ordinary independent4 tests auxiliary learning with free private bodies coupled through one decoder. Core minus untied core compares practical sharing/coordinates, with body gradients unscaled and decoder mean data gradient fixed; this is not an ordinary independent acquisition control.'}
CONTROL_PRIORITY=('single_native','single_four_view_core','shared4_masked_ce',
    'untied4_shared_decoder_core','shared4_core_shuffled','shared4_core_rewired')


def spec(name):
    if name not in ALL_CONDITIONS:raise ValueError('Exact fixed condition required')
    untied=name in ('independent4_native','untied4_shared_decoder_core')
    shared=name.startswith('shared4')
    masked=name not in ('shared4_own','single_native','independent4_native')
    core=masked and name!='shared4_masked_ce'
    members=4 if shared or untied else 1
    return dict(members=members,bodies=4 if untied else 1,shared_BE=shared,untied=untied,masked=masked,core=core,
        all_view_single=name=='single_four_view_core',masked_views=4 if masked else 0,
        factual_views=members,optimizer_steps=4+(1 if core else 0) if untied else 1+(1 if core else 0),
        backwards=5 if name=='single_four_view_core' else members)


def roster(conditions=CONDITIONS):
    return [dict(record_id=f'seed{seed}__{name}',condition=name,seed=seed,enabled=False,
        max_epochs=2000,patience=250,selector=SELECTOR,TEST_access=False,limits=dict(LIMITS),
        **SPLIT,**spec(name)) for seed in SEEDS for name in conditions]


def description():
    return dict(schema='masked-context-disabled-stage2-plan-v1',enabled=False,
        positive_complete_nine_required=True,records=roster(),retained_stage1_records=9,
        new_records=18,complete_family_records=27,rewire=REWIRE,purposes=PURPOSES,
        control_priority=list(CONTROL_PRIORITY),
        primary_reference_priority='After whole Stage1 closure and passing frozen screen: native single3 first; separately own-selected ordinary-independent reference banks remain a primary prerequisite for a stronger claim, before lower-priority perturbation interpretation. They are not silently added to this18/27 source roster.',
        partial_family_interpretation=False,automatic_retry=False,TEST_access=False,
        stronger_claim_limit='One encountered graph/split, optimizer replicates, common independent-bank selector. Preserve all27 and all failures. Capable singles and own-selected independent ensembles plus unused confirmation remain required.')


def forecast(native_forecast):
    counts={name:spec(name) for name in CONDITIONS}
    return dict(schema='masked-context-stage2-symbolic-cost-v1',measured_complete_run=False,
        exact_native_shape_ledger=native_forecast,condition_work=counts,new_records=18,
        retained_records=9,complete_family_records=27,
        maximum_training_fullgraph_forwards=6000*sum(r['factual_views']+r['masked_views'] for r in counts.values()),
        maximum_regular_VALID_member_forwards=6000*sum(r['members'] for r in counts.values()),
        maximum_backward_calls=6000*sum(r['backwards'] for r in counts.values()),
        maximum_Adam_steps=6000*sum(r['optimizer_steps'] for r in counts.values()),
        selected_factual_member_forwards=3*sum(r['members'] for r in counts.values()),
        selected_masked_member_forwards=3*sum(r['masked_views'] for r in counts.values()),
        preprocessing_banks=3*sum(5 if r['masked'] else 1 for r in counts.values()),
        native_body_constructions=3*sum(r['bodies'] for r in counts.values()),
        saved_factual_float32_logit_bytes=3*19717*3*4*sum(r['members'] for r in counts.values()),
        saved_masked_float32_logit_bytes=3*19717*3*4*sum(r['masked_views'] for r in counts.values()),
        maximum_selected_checkpoint_writes=36000,limits=LIMITS,
        new18_additive_active_plus_cleanup_cap_seconds=162180,complete27_additive_cap_seconds=243270,
        rewire_generation_envelope_seconds=600,rewire_generation_GPU_bytes=0,rewire_generation_RSS_bytes=16*1024**3,
        wall_estimate_seconds=None,new_control_GPU_peak_bytes=None,
        limitation='Stage1 rates do not qualify shuffled/rewired/single/all-view/untied-decoder controls. Complete TRAIN-only timing, graph generation and selected-output peaks require future root evidence; no proxy may become a shortened horizon or favorable retry.')
