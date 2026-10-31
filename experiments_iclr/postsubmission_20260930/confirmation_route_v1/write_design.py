"""Source-only prospective confirmation design. No ML libraries/data access."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(name, obj):
    with (ROOT / name).open('x') as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write('\n')


def main():
    inputs = [
        'RESEARCH_STATE.md',
        'sharing_policy/PROPOSAL.md',
        'sharing_policy/replay_recurrence.py',
        'sharing_policy/recurrence_collector.py',
        'new_tasks/RESEARCH_NOTE.md',
        'new_tasks/GINE_MODEL_SPEC.md',
        'new_tasks/gine_adapters.py',
        'new_tasks/molhiv_pilot.py',
        'new_tasks/mnist_pilot.py',
        'new_tasks/mnist_adapters.py',
        'new_tasks/development_freezes_v1/frozen_plan/molhiv_source_freeze.json',
        'new_tasks/development_freezes_v1/frozen_plan/mnist_superpixels_source_freeze.json',
        'protocols/ppi_development_v3/JOINT_DESIGN.json',
        'spectral_privacy/PROPOSAL.md',
        'spectral_privacy/ppi_f2_audit_v1/WARM_START_OUTLINE.md',
        'spectral_privacy/ppi_f2_audit_v1/F2_SUMMARY_v1.md',
        'closest_graph_ensemble_sources_v1/REPORT.md',
    ]
    emit('INPUT_MANIFEST.json', {'schema': 'gnnm-confirmation-inputs-v1',
         'scope': 'read-only development designs/method sources/literature; no historical predictive scores',
         'sha256': {path: digest(PHASE/path) for path in inputs}})
    protocol = {
        'schema': 'gnnm-independent-confirmation-route-v1',
        'registered_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'prospective_design_registered_not_data_or_runtime_frozen_not_launch_ready',
        'primary_route': 'Peptides-func full-data sharing-policy confirmation',
        'method_gate': 'Only enter after complete existing PPI/MolHIV development comparisons pass their already frozen quality/mechanism rules, with locked policy version and applicable molecular adapter. Do not weaken those rules here. A lower-label-only or newly amended method requires a distinct prospective protocol before any confirmation outcome; v1 does not silently choose the better label budget.',
        'candidate_lock': {'currently_complete': False,
            'required': ['development gate decisions with all required arms and adverse evidence',
                         'candidate policy/source/checkpoint/probe hashes and immutable recipe',
                         'development-fixed private-block control index',
                         'one strongest compatible size-matched efficient control chosen exclusively on development outcomes, plus its source hash',
                         'data/provider/runtime/idle-GPU hashes and authorization evidence',
                         'prospective test-evaluator and profiler source']},
        'data': {
            'name': 'Peptides-func', 'provider': 'LRGB official PeptidesFunctionalDataset',
            'pinned_provider_commit': '2410e9ee8d9309dd2120809d119a46092788eb89',
            'graphs_published': 15535, 'targets': 10,
            'features': 'OGB categorical atoms9/bonds3 from official SMILES2graph; mean graph readout, no positional encoding in primary matched mechanism arms',
            'split': 'official provided random stratified graph split; preserve exact IDs; no split seed search',
            'raw_md5': '701eb743e899f4d793f0e13c8fa5a1b4',
            'split_md5': '5a0114bdadc80b94fc7ae974f13ef061',
            'raw_sha256_processed_tensor_split_hashes': 'required at later data freeze; not fabricated here',
            'expected_split_counts_for_planning_only': {'train': 10873, 'val': 2331, 'test': 2331},
            'split_counts_verified_from_dataset': False,
            'exact_split_count_gate': 'Verify actual official split lists before outcome access. Counts are a planning assumption, not recovered archive evidence. If inconsistent, document and amend workload/source design prospectively without examining model performance.',
            'graph_independence_limit': 'Disjoint official graph IDs do not establish independent peptide sequences, scaffolds or biology. The random split is not a scaffold/OOD benchmark.',
            'corpus_overlap_gate': 'Before performance scoring, compare canonical molecular identities and, where released, normalized peptide-sequence identities across official splits and existing development corpora. Use input-only metadata and archive results. If an independent confirmation test overlaps development training/selection examples, suspend v1 and define a new no-overlap protocol before any test labels or predictions are opened. Do not silently delete examples or relabel the official benchmark.',
            'test_targets_read': False, 'dataset_downloaded': False,
        },
        'recipe': {
            'primary_label_fraction': 1.0,
            'secondary_label_fractions': [],
            'use_complete_official_training': True,
            'width': 128, 'layers': 3, 'members': 4,
            'hidden_mlp_multiplier': 2, 'dropout': 0.1,
            'normalization': 'per-node LayerNorm',
            'atom_bond_encoder': 'same bond-aware MolecularProjectorEnsemble/GINE model in all sharing layouts',
            'pooling': 'mean member raw logits; sigmoid of that mean is deployed probability',
            'loss': 'equal observed-graph average of mean-member, finite-target BCE; no target/class weighting',
            'optimizer': 'AdamW', 'learning_rate': 0.001, 'weight_decay': 0.0, 'epsilon': 1e-8,
            'batch_size': 128, 'precision': 'float32', 'AMP': False,
            'num_workers': 0, 'threads': 1,
            'initial_epochs': 100, 'prospective_maximum_epochs': 250,
            'checkpoint': 'maximum complete-validation official macroAP; then minimum pooled graph-mean BCE; then earliest epoch',
            'selection_opportunities': 'Every epoch for all contrasted arms;100 initially or250 if the prespecified continuation triggers, never arm-specific',
            'seeds': list(range(10)),
            'initial_resource_pilot_seeds': [0,1,2],
            'pilot_is_not_confirmation': True,
            'graph_order_seed_rule': '19709 +1000*optimizer_seed +epoch; identical graph orders across contrasted arms',
        },
        'sharing_selector_mapping': {
            'policy': 'recurrence-budget-v2 from bound replay_recurrence.py; no confirmation tuning',
            'blocks': 'All parameters of each of three complete GINE blocks, including per-block BondEncoder/epsilon/normalization/MLP; atom lookup and boundary projectors excluded',
            'units': 'Whole observed training graphs; exact canonical molecule or peptide-sequence duplicates must stay in the same P/Q/A/B group',
            'grouping': 'Union-find over shared canonical SMILES and normalized released peptide_seq; do not invent similarity thresholds or use labels to choose groups',
            'folds': 'Deterministic training-only group allocation: seed700 P/Q, approximately equal observed-graph counts; seed701/702/703 Q-to-A/B allocations, group-disjoint. Greedy count balancing with seeded group ordering, exact tie P/A first. Freeze all group/graph IDs and counts before probe fitting.',
            'loss_weights': 'Each graph has equal weight within a fold; report unequal fold sizes. Recurrence uses the equal-weight average of two fold means, not an unreported graph-count weighted replacement.',
            'probe': 'TIED fits P only for25 epochs; archive epochs0/10/25; identical probe window to existing PPI design; validation/test never enter folds or gradients',
            'statistic': 'S=max(0,median nine C); N=median nine N; eligible iff S>N and per-partition median C>0; rank by(S-N)/(3*block_parameter_count), earlier block breaks exact ties',
            'budget': 'At most one whole private block; equal GINE-block cost required by CPU gate; permit no eligible block=>TIED. Never force a nonzero layout.',
            'final_training': 'Restart canonical paired initialization; selected layout locks before any validation comparison; fit every arm on complete official training for same budget',
            'control_index': 'Development-fixed best private block, chosen using only full PPI/MolHIV development results and frozen before any Peptides outcome; no confirmation validation oracle substituted',
            'readiness': 'Current GINE probe_blocks API exists, but no Peptides data/collector/final-training runner or exact-null gate has been executed',
        },
        'arms': {
            'primary_10_seed': ['training_only_policy', 'TIED', 'development_fixed_private_block', 'development_frozen_size_matched_efficient_control'],
            'maximum_initial_fits': 40,
            'reuse': 'If policy selects TIED or the fixed control, reuse that exact canonical run; never train duplicate cells just to inflate count',
            'efficient_control_choice': 'Choose once on development, before confirmation acquisition, among source-compatible exact-count matched BASE, matched independent ENS4, shared-trunk multihead, and a prospectively gated declared all-layer GINE BatchEnsemble port. If needed control is missing or failed, route is not ready. Frozen pool and selection rule must be logged; no confirmation selection.',
            'parameter_matching': 'Target the fixed one-private-block width128 budget with output10; enumerate actual counts at integer widths8..512, choose closest, tie smaller, <=1% mismatch. A policy choosing TIED underspends that cap; report actual counts. Never reuse MolHIV output1 widths without recounting.',
            'development_prerequisites': 'All fixed layouts, fully private, BASE, source GNNM, multihead, independent ENS, matched BASE/ENS and closest all-layer comparator needed by the existing development claim must be completed before reducing to this confirmation family.',
            'comparators_not_claimed_exact': 'All-layer GINE is Kim-inspired declared adaptation; thesis verified GCN/GIN/GAT and probability averaging. Primary GNNM uses raw-logit mean; report pooling distinction. No improvised MIMO graph packing is an exact MIMO reproduction.',
        },
        'primary_comparison_family': [
            {'id':'sharing_signal','contrast':'policy minus TIED','endpoint':'test graph-mean mean-member BCE','direction':'negative'},
            {'id':'adaptive_location','contrast':'policy minus development-fixed private block','endpoint':'test graph-mean mean-member BCE','direction':'negative'},
            {'id':'practical_quality','contrast':'policy minus development-frozen size-matched efficient control','endpoint':'official test macroAP','direction':'positive','practical_mean_margin':0.005},
        ],
        'uncertainty': {
            'seed_summary': 'All10 paired deltas, mean, sampleSD, min/max and two-sided95% paired-t interval(df9). Conditional optimizer/probe variation on one fixed split; not10 independent datasets.',
            'paired_randomization': 'If reported, exact two-sided mean-delta sign-flip enumerating all1024 assignments; symmetry/exchangeability null caveat stated; Holm over the three prespecified primary contrasts at alpha0.05. It supplements effect sizes and intervals, never proves generality.',
            'minimum_two_sided_p': {'3_nonzero_seeds':0.25,'5_nonzero_seeds':0.0625,'10_nonzero_seeds':0.001953125},
            'graph_intervals': 'Retain member logits; paired2000-resample bootstrap of test graphs and separately exact-molecule/peptide-sequence union-find clusters, using same sampled units for both arms and all targets. Also crossed seed+cluster resampling with fixed RNG300930. Report percentile95% intervals and all assumptions. No node/member/target resampling as independent tasks.',
            'bootstrap_AP_guard': 'Original official eligible target set stays fixed. Mark replicas unable to score every originally eligible target invalid; report valid fraction. If>5% invalid or fewer than30 independent clusters, AP graph/cluster interval is unstable and cannot support the confirmation gate; do not silently change the target set or redraw until favorable.',
            'ten_targets': 'Biological functions are correlated; macroAP ten labels is one task endpoint, not ten independent confirmations.',
        },
        'confirmation_gate': {
            'complete': 'All10 seeds and all primary arms, retained unsuccessful/zero-private cells, exact source/data/protocol replay, test scoring exactly once after global checkpoint locks',
            'method': 'Both policy-minus-TIED and policy-minus-fixed block BCE mean deltas<0, with upper95% paired-seed CI<0 and Holm-adjusted evidence consistent with benefit; graph-cluster sensitivity must not contradict the direction. If adaptive-location fails, do not claim the learned location is better than a fixed layer.',
            'quality': 'MacroAP versus frozen efficient control mean gain>=0.005, lower95% paired-seed CI>0, Holm-adjusted evidence consistent with benefit; graph/cluster uncertainty must be shown. Otherwise use a narrower description of measured tradeoff or failed confirmation, never claim universal outperformance.',
            'cost': 'No latency/memory claim without selected-checkpoint matched-device profiling. Store actual parameters/buffers/optimizer memory; charge probe, fold collection, final fits, all retries and profiling. Size reduction alone cannot pass a speed/peak-memory gate.',
            'claim_limit': 'Prospective transfer on one held-out molecular corpus under one official random split and a fixed small-backbone budget. No SOTA, scaffold-OOD, recommendation-system, universal-noise or generalization-theorem claim.',
        },
        'budget_gate': {
            'late_condition_per_fit': 'Selected epoch>=91 and least-squares slope of validation macroAP on epochs81..100 strictly positive',
            'repeat_threshold': 'Any primary arm has late_condition in>=3 of10 seeds',
            'continuation': 'The extension to250 epochs is registered before first outcome and applies to every primary arm in every seed, preserving common optimizer/RNG/source/graph-order stream. Keep100-epoch and250-epoch reports. Source/checkpoint support must be locked before100-epoch fitting; no selective source repair/extension after seeing promising cells.',
            'terminal_gate': 'At250 use analogous final10% criterion(selected epoch>=226 and slope on last50 epochs positive); repeated>=3/10 marks budget-limited evidence, not confirmed superiority. Without resource to complete common continuation, confirmation is incomplete.',
            'no_early_stop': 'No arm-specific early stopping or post-score epoch/rank/width/split search; validation selects checkpoint only',
        },
        'test_access': {
            'policy': 'No confirmation predictive metric before candidate/source/data/fold/resource/analysis locks and all checkpoint identities. Dedicated single evaluator after global locks; retain evaluator manifest and every output. Public labels are not cryptographically hidden.',
            'loader': 'Split train/validation processing from test targets; do not reuse the earlier freezer that hashed test y. Input-only test structure/split audits allowed. No per-test y fingerprint or class summary before scoring.',
            'post_unseal': 'Do not revise method/recipe or replace this task on outcomes. Any failure stays in ledger. Later method versions cannot count this corpus as fresh confirmation.',
        },
        'stronger_reference': {
            'reassessed_GINE_commit': 'c3500cb702dcc5bdbe02783b94f24eebb7fb52c9',
            'file':'configs/LRGB-tuned/peptides-func-GINE.yaml',
            'recipe':'8layers,width160,Atom+RWSE20-step kernel,3-layer readout,250epochs,batch200,AdamW0.001,cosine5warmup,gradient clipping; actual author loss/logging code must also be bound',
            'role':'A meaningful external-backbone reference, with separate features and resource budget; not part of the cheap matched mechanism comparison',
            'not_ready':'No author model/provider runtime/replay or actual fits run. Cannot copy its recipe only and claim exact reproduction.',
            'claim_gate':'A claim against strong existing methods requires this fixed reference or an equivalently justified stronger benchmark first. Current3layer100epoch internal study alone cannot establish it; do not widen claims on a favorable pilot.',
        },
        'workload': {
            'planning_train_graphs_unverified':10873,
            'batch128_expected_steps_per_epoch':85,
            '40fit_100epoch_steps_excluding_probe_validation':340000,
            '40fit_250epoch_steps_excluding_probe_validation':850000,
            '10probe_25epoch_halftrain_batch128_steps_approx':10750,
            'full_data_resource_pilot_3seeds_4arms_100epoch_steps':102000,
            'published_mean_graph_nodes':150.94,'published_mean_directed_edges':307.30,
            'mean_batch_nodes_approx':19320.32,'mean_batch_edges_approx':39334.4,
            'four_member_width128_float32_one_node_state_MiB_approx':37.735,
            'uncertainty':'Published means are no worst-case batch bound. Measure real full-pipeline peak and synchronized cost; no GPU-hour ETA invented from CPU checks.',
            'initial_resource_cap_gpu_hours':48,
            'cap_behavior':'If measured pilot projection or common continuation exceeds cap, preserve partial evidence as incomplete and ask for GPU resources before larger jobs; do not shrink to3 seeds and call it confirmed.',
        },
        'resource_request': {
            'minimum':'One exclusively allocated idle CUDA GPU, >=24GB VRAM, explicit UUID/index and authorized repository path; A10040/80GB preferred for grouped GINE batches, not a verified minimum memory bound',
            'execution':'Serial fits, repository-local environment/cache/temp/data/results only; no sudo; no dataset acquisition or launch by this source-plan task',
            'duration':'Up to48GPUhours for bounded confirmation; measured full-data pilot is required before an honest ETA',
        },
        'spectral_applicability': {
            'BCE_multilabel':'Objective and centered-gradient matrix risk statistic can apply to10 graph labels;121 node targets is not required. Independent scoring units/loss weighting must be redefined as whole training peptide groups, with missing labels handled identically.',
            'source_port':'Existing wrapper corrects SAGE lin_l neighbor maps. GINE lacks that exact map: choose and freeze a specific square bond-aware transform, preserve all bond messages, and prove zero-A forward/gradient identity plus sum-of-private gradients and native optimizer scale. No such Peptides port has been executed.',
            'no_premature_confirmation':'No spectral rank/accuracy claim from local PPI F2 proxy or unchanged rank16/n01 port. Mixed full-label F2 results remain. Matched selected/random/raw basis warm-start arms need a separate development freeze and predictive gate first.',
            'CE_regression_link':'Gradient-statistic construction is objective-general, but predictive alignment, loss normalization, target transform, graph/scaffold/pair units and warm-start controls change. Current PPI runner is not directly applicable to multiclassCE, molecular regression, or link negatives. Those adaptations must not be called the same validated mechanism without new gates.',
        },
        'ranked_alternatives': [
            {'rank':1,'task':'Peptides-func','decision':'Primary sharing-policy route above; strongest existing molecular compatibility and published standard full-data splits'},
            {'rank':2,'task':'CIFAR10 superpixels','decision':'If centered/narrow SAGE passes the existing MNIST/PPI development gates, freeze a separate10paired-seed protocol before CIFAR outcomes. InputRGB3+pos2=>5; official image-graph train/val/test, CE/accuracy; same mean-SAGE node-logit readout and four centering/feedback cells; matched packed independent ensemble and matched single/multihead controls. Current MNIST runner hardcodes input3/nameMNIST/counts55k5k10k, so new provider/source gate is necessary. Distinct image collection, no molecule bond-blind port. Published60kgraphs/mean117.6nodes/941.2edges implies about3x the MNIST edge workload; exact splits and actual runtime still unverified.'},
            {'rank':3,'task':'Molecular regression or recommendation/link prediction','decision':'Defer for confirmation now: no compatible completed runner/split/negative-sampling/evaluator gate identified. Peptides-struct shares exactly the Peptides-func molecule corpus and cannot be another independent confirmation. A regression/link port introduces new loss and leakage/negative-sampling choices, so it is development rather than independent confirmation until fixed elsewhere.'},
        ],
        'operations_in_this_task': {'dataset_downloads':0,'server_access':False,'training_fits':0,'predictive_scores_read':False,'historical_scores_changed':False,'review_verdicts':0},
    }
    emit('CONFIRMATION_DESIGN.json', protocol)
    checks = {
        'status':'PASS', 'scope':'source-only internal design consistency; no runtime/data verification',
        'ten_seeds': protocol['recipe']['seeds']==list(range(10)),
        'workload_exact_arithmetic': 40*100*85==protocol['workload']['40fit_100epoch_steps_excluding_probe_validation'],
        'all_family_extension': protocol['recipe']['prospective_maximum_epochs']==250,
        'minimum_p_values': {str(n):2/(2**n) for n in (3,5,10)},
        'test_unscored': protocol['test_access']['policy'].startswith('No confirmation'),
        'not_launch_ready': 'not_launch_ready' in protocol['status'],
        'input_files_bound':len(inputs),
    }
    assert all(checks[k] for k in ('ten_seeds','workload_exact_arithmetic','all_family_extension','test_unscored','not_launch_ready'))
    emit('SOURCE_DESIGN_CHECK.json', checks)
    print(json.dumps({'status':'PASS','inputs':len(inputs),'fits_maximum':40,'paired_seeds':10,'data_loaded':False,'launch_ready':False}))


if __name__ == '__main__':
    main()
