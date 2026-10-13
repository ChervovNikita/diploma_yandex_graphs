"""Prepare one disabled master from exact immutable packet metadata.

No numerical library, model, target-array reader, worker, remote transport,
approval writer or phase-receipt writer is called. CLI metadata access is
allocation-guarded; pure helpers support local standard-library fixtures.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

from partition import assign, DEFAULT_WORKERS
from prepare_node_data import verify_allocation
from spec import canonical, fingerprint, validate

MIB = 1024 ** 2
CONTROLLER_FILES = ('engine.py', 'metrics.py', 'spec.py', 'registry.py', 'partition.py', 'run_study.py', 'run_endpoint.py', 'launcher.py', 'phase_closure.py', 'author_cli_spec.py', 'run_author_cli.py', 'phase_rendezvous.py', 'start_massive.sh', 'compile_master_plan.py', 'prepare_node_data.py', 'prepare_sync_environment.py', 'prepare_sync_bundle.py', 'ogb_author_cli_spec.py', 'run_ogb_author_cli.py', 'ogb_declared_seed_entry.py', 'tabm_stopping.py', 'oipg_typing.py', 'bsnn_stopping.py', 'checkpoint_retention.py', 'hgt_stopping.py', 'label_corrector_semantics.py', 'native_stopping.py', 'response_semantics.py', 'retention_lifecycle.py', 'typed_context_semantics.py', 'relation_semantics.py', 'hop_complete_semantics.py')
CONTROLLER_FILES = CONTROLLER_FILES + ('structural_cardinality_semantics.py',)
CONTROLLER_FILES = CONTROLLER_FILES + ('gpu_command.py',)
STAGE_KIND = {'search': 'model_fit', 'comparative': 'model_fit', 'search_selection': 'selection',
              'bank_export': 'ensemble_assembly', 'calibration': 'calibration', 'evaluation': 'evaluation',
              'author_cli': 'author_cli_group','native_capture':'selection',
              'geometry_screen_search':'selection','geometry_screen_comparative':'selection'}
SCOPES = {'measured', 'measured_proxy', 'provisional', 'mixed', 'unknown'}
POLYFORMER_MODES = {'polyformer_single', 'polyformer_factorized_single', 'polyformer_internal_be',
                    'polyformer_original_boundary_own', 'polyformer_boundary_equipped_M1_own'}
PT_ARMS = {'ordinary_p1', 'factor_f1', 'factor_s4', 'original_gnnm', 'head_only',
           'mixed_single', 'bias_coordinate_single', 'private4', 'middle_state_mean',
           'separable4', 'continued_p1', 'continued_f1'}
PT_SOURCE_ARMS = {'ordinary_p1', 'factor_f1', 'factor_s4', 'original_gnnm'}
GRU_ARMS = PT_SOURCE_ARMS | {'B4', 'G1', 'GF1', 'GM4', 'GN4', 'continued_p1', 'continued_f1'}
LANDMARK_SOURCE_ARMS = {'ordinary_m1', 'factor_m1', 'shared_factor4', 'original_boundary'}
LANDMARK_ARMS = LANDMARK_SOURCE_ARMS | {
    'B1', 'B4', 'T16', 'S64', 'FS64', 'LM4', 'continued_m1', 'continued_f1'}
LANDMARK_FOUR_VIEW_ARMS = {'shared_factor4', 'original_boundary', 'B4', 'T16', 'LM4'}
# Counts are source TRAIN calls per complete update/cycle. Assignment CMCL
# includes the four no-tape responsibility forwards and their exact replay.
CMCL_WORK = {'own_floor': (4, 4), 'private_cmcl': (8, 8),
             'all_block_cmcl': (8, 4), 'vanilla_cmcl': (8, 4),
             'private_uniform': (4, 8), 'private_constant_credit': (4, 8)}
# optimizer owners/calls, slow bodies, served members, forwards, backwards
CORE_WORK = {'shared4_own': (1, 1, 4, 4, 4),
             'shared4_masked_ce': (1, 1, 4, 8, 4),
             'shared4_core': (2, 1, 4, 8, 4),
             'shared4_core_shuffled': (2, 1, 4, 8, 4),
             'shared4_core_rewired': (2, 1, 4, 8, 4),
             'single_native': (1, 1, 1, 1, 1),
             'independent4_native': (4, 4, 4, 4, 4),
             'single_four_view_core': (2, 1, 1, 5, 5),
             'untied4_shared_decoder_core': (5, 4, 4, 8, 4)}
POLYNORMER_QK_ARMS = {
    operator + '_' + kind: (operator, kind, 4 if kind == 'be_init' else 1)
    for operator in ('native_tied', 'active_reversible_exp', 'pre_sigmoid_split', 'full_qk')
    for kind in ('single', 'be_init')}
POLYNORMER_QK_ARMS.update(bare_native_single=('bare', 'single', 1),
                          original_gnnm=('original_boundary', 'boundary4', 4))
# served routes, source head forwards, backwards, optimizer owners per label cycle.
PAID_LABEL_WORK = {
    'staged_posterior_c4':(4,4,4,1),'staged_posterior_s_joint4head':(1,4,1,1),
    'staged_posterior_u4_sharedb':(4,4,4,4),'staged_posterior_s_one_path':(1,1,1,1),
    'query_value_gate_c4_gate':(4,4,4,1),'query_value_gate_s_joint4head_gate':(1,4,1,1),
    'query_value_gate_u4_sameb_full_untied_gate':(4,4,4,4),'query_value_gate_s_one_path_gate':(1,1,1,1),
    'query_value_gate_c4_gate_identity_erased':(4,4,4,1)}


def source_parent_arm(family, arm):
    if family == 'landmark':
        return 'ordinary_m1' if arm == 'continued_m1' else 'factor_m1'
    return 'factor_f1' if arm == 'continued_f1' else 'ordinary_p1'


def staged_family(config):
    fields = [name for name in ('pt', 'gru', 'landmark') if name + '_stage' in config]
    if len(fields) > 1:
        raise ValueError('A model fit cannot belong to two staged source families')
    return fields[0] if fields else None


def safe_path(root, value):
    root = Path(root).resolve()
    raw = Path(value)
    if raw.is_absolute() or '..' in raw.parts or not raw.parts:
        raise ValueError('Master inputs/output use project-relative paths')
    candidate = root / raw
    if any(p.is_symlink() for p in (candidate, *candidate.parents)):
        raise ValueError('Immutable packet/controller paths may not traverse symlinks')
    target = candidate.resolve()
    if not target.is_relative_to(root):
        raise ValueError('Master path escapes the project')
    return target


def receipt(root, path):
    path = safe_path(root, path)
    payload = path.read_bytes()
    return {'path': str(path.relative_to(Path(root).resolve())), 'bytes': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest()}


def validate_receipt(value):
    if (not isinstance(value, dict) or set(value) != {'path', 'bytes', 'sha256'}
            or type(value['bytes']) is not int or value['bytes'] < 0
            or not isinstance(value['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', value['sha256'])):
        raise ValueError('An exact path/bytes/SHA256 receipt is required')
    path = Path(value['path'])
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Receipt paths must be project-relative')


def read_bound(root, value):
    validate_receipt(value)
    observed = receipt(root, value['path'])
    if observed != value:
        raise ValueError('Immutable metadata changed: ' + value['path'])
    return safe_path(root, value['path']).read_bytes()


def number(value, name, *, integer=False, zero=False):
    if (type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value)
            or value < 0 or (value == 0 and not zero)):
        raise ValueError('Invalid explicit ' + name)
    return value


def merge_bindings(*rosters):
    result = {}
    for roster in rosters:
        for row in roster:
            validate_receipt(row)
            prior = result.get(row['path'])
            if prior is not None and prior != row:
                raise ValueError('Conflicting immutable file binding: ' + row['path'])
            result[row['path']] = copy.deepcopy(row)
    return [result[key] for key in sorted(result)]


SEHGNN_SUPPLY = {'shared_own_only':0,'native_pool_credit':15,'source_view_supervision':15,'uncoupled_source_contrast':18,'COMMON_cycle':18,'assigned_source_supply':18}
SEHGNN_JOINT = {'BE_P':(4,1,32,16),'BE_PS':(4,1,32,19),'ADD_P':(4,1,32,16),'ADD_PS':(4,1,32,19),'ADD_P_copied_dictionary':(4,1,32,16),'ADD_PS_neutral_all_source_average':(4,1,32,19),'independent_ADD_P':(4,4,32,16),'independent_ADD_PS':(4,4,32,19),'native_single_P':(1,1,8,4),'rank4_single_P':(1,1,8,4),'rank1_single_P':(1,1,8,4)}
SEHGNN_PREFIXES = [{'prefix_id':'native200','cap_step':200,'close_on_selector_stop':False},{'prefix_id':'native500','cap_step':500,'close_on_selector_stop':False}]


def sehgnn_semantics(run):
    c,b=run['config'],run['budget'];arm=c.get('sehgnn_arm')
    if arm in SEHGNN_SUPPLY:members,owners,fwd,back=4,1,4,4
    elif arm in SEHGNN_JOINT:members,owners,fwd,back=SEHGNN_JOINT[arm]
    elif arm in ('native_single','factorized_native_single','original_gnnm'):members,owners= (4,1) if arm=='original_gnnm' else (1,1);fwd=back=members
    else:raise ValueError('Unknown complete native SeHGNN source arm')
    if (b.get('unit')!='epochs' or b.get('policy')!='full_budget' or b.get('maximum')!=1000 or b.get('prefixes')!=SEHGNN_PREFIXES
            or c.get('sehgnn_epochs')!=1000 or c.get('backbone')!='native_SeHGNN' or c.get('width')!=512 or c.get('classes')!=5
            or c.get('feature_channels')!=25 or c.get('label_channels')!=12 or c.get('members')!=members
            or c.get('source_optimizer_count')!=owners or c.get('physical_slow_bodies')!=owners
            or c.get('historical_state_inherited') is not False or c.get('is_original_gnnm')!=(arm=='original_gnnm')
            or c.get('native_AMP_overflow_skip_allowed')!=(arm not in SEHGNN_JOINT)
            or c.get('source_optimizer_accounting')!='attempts_and_actual_committed_maps_are_distinct'
            or run['selection']!={'metric':'pooled_BCE','direction':'min','ties':'first','evaluation_interval':1,'checkpoint0':False,'patience_steps':1001}):
        raise ValueError('Complete512native five-label SeHGNN1000epoch source contract with diagnostic200/500 prefixes required')
    work={'known_source_TRAIN_forward_calls':1000*fwd,'known_source_TRAIN_backward_calls':1000*back,
          'known_sehgnn_SELECT_forward_calls':1000*members,'known_sehgnn_diagnostic_prefixes':2,
          'known_sehgnn_scalar_leaf_reverse_calls':0,'known_sehgnn_private_model_VJP_calls':1000*(back-16) if arm in SEHGNN_JOINT and members==4 else 0,
          'known_sehgnn_private_trial_attempt_upper_bound':0,'known_sehgnn_private_trial_forward_upper_bound':0}
    if arm in SEHGNN_SUPPLY and arm!='shared_own_only':
        vjp=SEHGNN_SUPPLY[arm];work['known_source_TRAIN_forward_calls']+=198*(20+11);work['known_source_TRAIN_backward_calls']+=198*vjp
        work.update(known_sehgnn_source_opportunities=198,known_sehgnn_scalar_leaf_reverse_calls=198*12,
                    known_sehgnn_private_model_VJP_calls=198*vjp,known_sehgnn_private_trial_attempt_upper_bound=198*4,
                    known_sehgnn_private_trial_forward_upper_bound=198*44)
    amp=arm not in SEHGNN_JOINT
    return {'optimizer_objects':owners,'physical_slow_bodies':owners,'known_optimizer_step_calls':None if amp else 1000*owners,
            'known_sehgnn_optimizer_attempts':1000*owners,'known_sehgnn_optimizer_commit_lower_bound':0 if amp else 1000*owners,
            'known_sehgnn_optimizer_commit_upper_bound':1000*owners,'known_sehgnn_AMP_bounded_commit_fit_groups':int(amp),
            'budget_unit':'epochs','registered_budget_units':1000,'law':'native_SeHGNN_complete_TRAIN_cycles_explicit_old_state_replay_VJPs_trials',**work}


def fit_semantics(run):
    """Known exact bridge families; unknown ownership is never silently one."""
    (config, budget) = (run['config'], run['budget'])
    (unit, maximum) = (budget['unit'], budget['maximum'])
    if 'configured_relation_source_contract' in config:
        from relation_semantics import validate_relation_run
        return validate_relation_run(run)
    if budget['policy']=='hop_complete_native_selector_stop':
        from hop_complete_semantics import validate_hop_run
        return validate_hop_run(run)
    if budget['policy'] == 'tabm_native_selector_stop':
        from tabm_stopping import validate_tabm_run
        return validate_tabm_run(run)
    if budget['policy'] == 'bsnn_native_selector_stop':
        from bsnn_stopping import validate_bsnn_run
        return validate_bsnn_run(run)
    if 'typed_context_source_contract' in config:
        from typed_context_semantics import validate_typed_context_run
        return validate_typed_context_run(run)
    if budget['policy'] == 'hgt_native_selector_stop':
        from hgt_stopping import validate_hgt_run
        return validate_hgt_run(run)
    if budget['policy'] == 'native_selector_stop':
        from native_stopping import validate_native_run
        return validate_native_run(run)
    if budget['policy'] != 'full_budget':
        raise ValueError('Master only supports registered typed completion policies')
    if 'oipg_policy' in config:
        from oipg_typing import validate_oipg_run
        return validate_oipg_run(run)
    if 'sehgnn_arm' in config:
        return sehgnn_semantics(run)
    if any(k in config for k in ('structural_source_contract','cardinality_source_contract','cardinality_control_contract','conditional_source_contract')):
        from structural_cardinality_semantics import validate_run
        return validate_run(run)
    if 'private_response_source_contract' in config:
        from response_semantics import validate_response_run
        return validate_response_run(run)
    if 'label_corrector_source_contract' in config:
        from label_corrector_semantics import validate_label_corrector_run
        return validate_label_corrector_run(run)
    family = staged_family(config)
    train_calls = {}
    constructor_overhead = {}
    if 'bsnn_source_contract' in config:
        arm = config.get('bsnn_arm')
        members = 4 if arm == 'original_GNNM' else 1
        if arm not in ('native_M1', 'original_GNNM') or unit != 'optimizer_updates' or maximum != 500 or (config.get('backbone') != 'original_DiscreteBundleSheafDiffusion_d2') or (config.get('classes') != 2) or (config.get('members') != members) or (config.get('source_optimizer_count') != 1) or (config.get('physical_slow_bodies') != 1) or (config.get('learning_rate') != 0.01) or (config.get('dropout') != 0.3) or (config.get('weight_decay') != (0.0 if arm == 'original_GNNM' else 0.0005)) or (config.get('source_complete_native_paths') != members) or (config.get('source_TRAIN_forwards_per_update') != members) or (config.get('source_TRAIN_backwards_per_update') != members) or (config.get('is_original_gnnm') != (arm == 'original_GNNM')) or (config.get('initialization_seed') != run['seed'] + 1009 * run['body']) or (run['selection'].get('checkpoint0') is not False) or (run['selection'].get('metric') != 'auroc') or (run['selection'].get('secondary') != 'nll') or (run['selection'].get('tertiary') != 'earlier_epoch'):
            raise ValueError('Exact original deterministic d2 bundle/sourceHouseholder/fixed500epochs/nativegroupedAdam or boundaryAdamW required')
        (objects, bodies, calls, law) = (1, 1, maximum, 'original_deterministic_bundle_d2_native_Adam_or_boundary_AdamW_source500')
        train_calls = {'known_source_TRAIN_forward_calls': maximum * members, 'known_source_TRAIN_backward_calls': maximum * members}
    elif 'source_GINE_contract' in config:
        arm = config.get('source_GINE_arm')
        members = 1 if arm == 'native_M1' else 4
        if arm not in ('native_M1', 'original_GNNM', 'middle_private', 'lowrank_private') or unit != 'epochs' or maximum != 100 or (config.get('backbone') != 'literal_source_GINE128_L3') or (config.get('members') != members) or (config.get('physical_slow_bodies') != 1) or (config.get('source_optimizer_count') != 1) or (config.get('model') != {'hidden': 128, 'layers': 3, 'dropout': 0.1, 'outputs': 1}) or (config.get('optimizer') != {'name': 'AdamW', 'learning_rate': 0.001, 'weight_decay': 0.0, 'eps': 1e-08, 'betas': [0.9, 0.999]}) or (config.get('training') != {'batch_size': 128, 'eval_batch_size': 128, 'views_per_update': 1}) or (config.get('source_TRAIN_forwards_per_minibatch') != members) or (config.get('source_TRAIN_backwards_per_minibatch') != 1) or (config.get('train_minibatches_per_epoch') != 10) or (config.get('is_original_gnnm') != (arm == 'original_GNNM')) or (config.get('source_rank_candidate_wrapper_constructions') != (129 if arm == 'lowrank_private' else 0)) or (config.get('initialization_seed') != run['seed'] + 1009 * run['body']) or (run['selection'].get('checkpoint0') is not False) or (run['selection'].get('metric') != 'rocauc') or (run['selection'].get('secondary') != 'pooled_bce') or (run['selection'].get('tertiary') != 'earlier_epoch'):
            raise ValueError('Literal source GINE128L3 fixedAdamW100epochs fullMolBACE10tailminibatches required')
        (objects, bodies, calls, law) = (1, 1, maximum * 10, 'literal_source_GINE_equal_graph_finite_target_BCE_one_AdamW_per_minibatch')
        train_calls = {'known_source_TRAIN_forward_calls': calls * members, 'known_source_TRAIN_backward_calls': calls}
        constructor_overhead = {'source_data_free_rank_candidate_wrapper_constructions': 129 if arm == 'lowrank_private' else 0}
    elif 'unimp_source_contract' in config:
        if unit != 'optimizer_updates' or maximum != 1500 or config.get('backbone') != 'native_Torch_UniMPv2' or (config.get('architecture') != 'single') or (config.get('members') != 1) or (config.get('physical_slow_bodies') != 1) or (config.get('source_optimizer_count') != 1) or (config.get('is_original_gnnm') is not False) or (config.get('learning_rate') != 0.001) or (config.get('weight_decay') != 0.0005) or (config.get('initialization_seed') != run['seed'] + 1009 * run['body']) or (config.get('source_TRAIN_forwards_per_update') != 1) or (config.get('source_TRAIN_backwards_per_update') != 1) or (run['body'] not in (0, 1, 2, 3)) or (run['selection'].get('checkpoint0') is not False) or (run['selection'].get('metric') != 'accuracy') or (run['selection'].get('direction') != 'max') or (run['selection'].get('ties') != 'first') or (run['selection'].get('evaluation_interval') != 1):
            raise ValueError('UniMP fixed1500epochs/source65visible35query/one CE Adam source law required')
        (objects, bodies, calls, law) = (1, 1, maximum, 'literal_Torch_UniMP_source65visible35query_one_Adam')
        train_calls = {'known_source_TRAIN_forward_calls': maximum, 'known_source_TRAIN_backward_calls': maximum}
    elif 'single8_control' in config:
        if config['single8_control'] not in ('single8_own', 'single8_alignment', 'single8_virtual_contrast') or unit != 'optimizer_updates' or maximum != 5000 or (config.get('backbone') != 'native_WikiPolynormer') or (config.get('architecture') != 'single') or (config.get('members') != 1) or (config.get('physical_slow_bodies') != 1) or (config.get('source_optimizer_count') != 1) or (config.get('is_original_gnnm') is not False) or (config.get('local_transition_updates') != 100) or (config.get('source_TRAIN_forwards_per_update') != 8) or (config.get('source_TRAIN_backwards_per_update') != 1) or (run['body'] != 0) or (run['selection'].get('checkpoint0') is not False) or (run['selection'].get('metric') != 'accuracy') or (run['selection'].get('direction') != 'max') or (run['selection'].get('ties') != 'first'):
            raise ValueError('Single8 retains one ordinary source body/Adam,eight full own TRAIN views,and deterministic single serving')
        (objects, bodies, calls, law) = (1, 1, maximum, 'literal_Single8_eight_TRAIN_views_one_native_Adam')
        train_calls = {'known_source_TRAIN_forward_calls': maximum * 8, 'known_source_TRAIN_backward_calls': maximum}
    elif 'curvature_stage' in config:
        (stage, arm) = (config['curvature_stage'], config['curvature_arm'])
        arms = ('common_only', 'fixed_first_graph_pair', 'selected_graph_pair', 'selected_permuted_span', 'selected_random_span')
        if stage not in ('warm', 'trials', 'continuation', 'cold') or maximum != {'warm': 250, 'trials': 1, 'continuation': 4750, 'cold': 5000}[stage] or unit != ('epochs' if stage == 'trials' else 'optimizer_updates') or (config.get('backbone') != 'native_Photo_Polynormer_r') or (config.get('physical_slow_bodies') != 1) or (config.get('source_optimizer_count') != 1) or (config.get('local_transition_updates') != 200) or (run['seed'] not in (88101, 88103, 88201, 88203, 88207)) or (run['body'] not in ((0, 1, 2, 3) if stage == 'cold' and arm == 'native_M1' else (0,))) or (config.get('learning_rate') not in (0.0005, 0.001, 0.002)) or (config.get('dropout') not in (0.35, 0.7, 0.85)) or (run['selection'].get('metric') != 'pooled_nll') or (run['selection'].get('direction') != 'min') or (run['selection'].get('ties') != 'first') or (run['selection'].get('evaluation_interval') != 1) or (run['selection'].get('checkpoint0') != (stage == 'continuation')) or (stage == 'continuation' and arm not in arms) or (stage == 'cold' and arm not in ('native_M1', 'original_GNNM')) or (stage in ('warm', 'trials') and arm != 'joint_five_arm_source') or (config.get('members') != (4 if stage in ('trials', 'continuation') or arm == 'original_GNNM' else 1)) or (stage == 'warm' and config.get('curvature_fixed_last') is not True) or (stage == 'trials' and config.get('curvature_joint_selector') is not True):
            raise ValueError('Exact Photo250warm/finite10trial/4750continuation/5000cold lifecycle required')
        (objects, bodies) = (1, 1)
        calls = None if stage == 'trials' else maximum
        law = 'curvature_' + stage + '_source_named_coupled_Adam'
        if stage == 'trials':
            constructor_overhead = {'bounded_optimizer_step_calls_minimum': 1, 'bounded_optimizer_step_calls_maximum': 10, 'bounded_paid_stepped_Adam_objects_minimum': 1, 'bounded_paid_stepped_Adam_objects_maximum': 10, 'unstepped_pre_install_Adam_constructor_objects': 7}
        else:
            constructor_overhead = {'serial_stepped_Adam_constructor_objects': 2 if stage in ('warm', 'cold') else 1}
        if config.get('is_original_gnnm') != (arm == 'original_GNNM'):
            raise ValueError('Actual source original boundary GNNM flag required')
    elif 'nsd_arm' in config:
        arm = config['nsd_arm']
        valid = {'assigned_fixed', 'assigned_resampled', 'layer_shared_incidence', 'node_capacity_right', 'joint_multi_transport_single', 'ordinary_M1', 'factorized_M1', 'original_gnnm', 'geometry_warm', 'geometry_branch', 'geometry_only_source'}
        members = 1 if arm in {'ordinary_M1', 'factorized_M1', 'joint_multi_transport_single'} else 4
        paths = 1 if arm in {'ordinary_M1', 'factorized_M1', 'original_gnnm'} else 4
        forwards = 4 if arm == 'original_gnnm' else paths
        backwards = 4 if arm in {'original_gnnm', 'geometry_warm', 'geometry_branch', 'geometry_only_source'} else 1
        maximum_expected = 100 if arm == 'geometry_warm' else 400 if arm == 'geometry_branch' else 500
        if arm not in valid or unit != 'optimizer_updates' or maximum != maximum_expected or (config.get('source_optimizer_count') != 1) or (config.get('physical_slow_bodies') != 1) or (config.get('members') != members) or (config.get('source_complete_native_paths') != paths) or (config.get('source_TRAIN_forwards_per_update') != forwards) or (config.get('source_TRAIN_backwards_per_update') != backwards) or (config.get('backbone') != 'native_DiscreteGeneralSheafDiffusion') or (config.get('classes') != 2) or (config.get('dropout') != 0.2) or (config.get('is_original_gnnm') != (arm == 'original_gnnm')):
            raise ValueError('Exact fullGeneral binary NSD source owners/callback/500 or paid100+400 protocol required')
        (objects, bodies, calls, law) = (1, 1, maximum, 'native_fullGeneral_NSD_own_NLL_source_Adam')
        train_calls = {'known_source_TRAIN_forward_calls': maximum * forwards, 'known_source_TRAIN_backward_calls': maximum * backwards}
    elif 'growth_stage' in config:
        (stage, arm) = (config['growth_stage'], config['growth_arm'])
        objects = 4 if arm == 'independent_graph_growth4' else 1
        if unit != 'epochs' or stage not in ('warm', 'post', 'cold_control') or maximum != (20 if stage == 'warm' else 60 if stage == 'post' else 80) or (config.get('source_optimizer_count') != objects) or (config.get('physical_slow_bodies') != objects) or (config.get('source_complete_episodes_per_cycle') != 61) or (config.get('source_Adam_passes_per_episode') != 3) or (config.get('train_minibatches_per_epoch') != 183):
            raise ValueError('Exact NCN20warm/60post/80cold complete61x3 source cycles required')
        bodies = objects
        calls = maximum * 61 * 3 * objects
        law = 'NCN_complete_source_outer_cycles61episodes_3Adam_passes_per_owner'
        train_calls = {'known_source_TRAIN_backward_calls': maximum * 61 * (12 if arm == 'original_gnnm' else 3) * objects}
    elif config.get('architecture') == 'paid_native_label_stage':
        arm = PAID_LABEL_WORK.get(config.get('paid_label_arm'))
        if arm is None or unit != 'epochs' or maximum != 5000 or (config.get('label_cycles') != 5000) or (config.get('source_optimizer_count') != arm[3]) or (config.get('physical_slow_bodies') != 0) or (config.get('members') != arm[0]) or (config.get('lifecycle', {}).get('schema') != 'paid-native-capture-label-stage-v1'):
            raise ValueError('Paid native label stage must retain paid parent, exact source owners and complete old-state label cycles')
        (objects, bodies, calls, law) = (arm[3], 0, maximum * arm[3], 'paid_native_frozen_capture_fresh_label_owners')
        train_calls = {'known_source_TRAIN_forward_calls': maximum * arm[1], 'known_source_TRAIN_backward_calls': maximum * arm[2]}
        constructor_overhead = {'unstepped_pre_install_Adam_constructor_objects': int('s_joint4head' in config['paid_label_arm'])}
    elif 'polynormer_qk_arm' in config:
        source = POLYNORMER_QK_ARMS.get(config['polynormer_qk_arm'])
        if source is None:
            raise ValueError('Unknown native Polynormer Q/K source arm')
        (operator, kind, members) = source
        expected_architecture = 'original_gnnm' if kind == 'boundary4' else 'internal_be' if members == 4 else 'single'
        if unit != 'optimizer_updates' or maximum != 5000 or config.get('polynormer_qk_operator') != operator or (config.get('source_ownership_kind') != kind) or (config.get('architecture') != expected_architecture) or (config.get('backbone') != 'native_Polynormer') or (config.get('source_optimizer_count') != 1) or (config.get('source_optimizer_calls_per_cycle') != 1) or (config.get('physical_slow_bodies') != 1) or (config.get('members') != members) or (config.get('local_transition_updates') != 100) or (config.get('source_TRAIN_forwards_per_update') != 2 * members) or (config.get('source_TRAIN_backwards_per_update') != 2 * members) or (config.get('is_original_gnnm') != (kind == 'boundary4')):
            raise ValueError('Native Polynormer Q/K requires one Adam owner, two complete own-CE views per member, and explicit actual original boundary control')
        (objects, bodies, calls, law) = (1, 1, maximum, 'native_Polynormer_QK_two_view_source_Adam')
        train_calls = {'known_source_TRAIN_forward_calls': maximum * 2 * members, 'known_source_TRAIN_backward_calls': maximum * 2 * members}
        constructor_overhead = {'unstepped_pre_install_Adam_constructor_objects': int(operator not in {'bare', 'original_boundary'})}
    elif family:
        arm = config.get(family + '_arm')
        bodies = 4 if family == 'pt' and arm == 'separable4' else 1
        valid_arms = {'pt': PT_ARMS, 'gru': GRU_ARMS, 'landmark': LANDMARK_ARMS}[family]
        source_arms = LANDMARK_SOURCE_ARMS if family == 'landmark' else PT_SOURCE_ARMS
        architecture = {'pt': 'retained_pt_routes', 'gru': 'private_gru_memory', 'landmark': 'landmark_memory'}[family]
        original = arm == ('original_boundary' if family == 'landmark' else 'original_gnnm')
        if arm not in valid_arms or config[family + '_stage'] not in {'source', 'continuation'} or (arm in source_arms) != (config[family + '_stage'] == 'source') or (config.get('source_optimizer_count') != 1) or (config.get('physical_slow_bodies') != bodies) or (unit != 'optimizer_updates') or (config.get('architecture') != ('original_gnnm' if original else architecture)):
            raise ValueError('PT owns explicitly typed source/continuation bodies and one fresh optimizer; GRU and landmark are separately typed')
        if family == 'landmark' and (config.get('members') != (4 if arm in LANDMARK_FOUR_VIEW_ARMS else 1) or (original and config.get('is_original_gnnm') is not True)):
            raise ValueError('Landmark route counts and actual original-GNNM boundary must be explicit')
        (objects, calls) = (1, maximum)
        law = family + '_' + config[family + '_stage'] + '_one_fresh_optimizer_full_updates'
    elif config.get('architecture') == 'private_cmcl':
        work = CMCL_WORK.get(config.get('cmcl_condition'))
        if work is None or unit != 'optimizer_updates' or config.get('source_optimizer_count') != 1 or (config.get('physical_slow_bodies') != 1) or (config.get('members') != 4):
            raise ValueError('Private CMCL requires its exact condition, one coupled optimizer and four shared routes')
        (objects, bodies, calls, law) = (1, 1, maximum, 'private_cmcl_one_coupled_source_Adam_per_update')
        train_calls = {'known_source_TRAIN_forward_calls': maximum * work[0], 'known_source_TRAIN_backward_calls': maximum * work[1]}
    elif config.get('architecture') == 'private_adam':
        if unit != 'optimizer_updates' or config.get('source_optimizer_count') != 1 or config.get('physical_slow_bodies') != 1 or (config.get('members') != 4) or (config.get('core_condition') != 'shared4_own') or (config.get('private_adam_normalization') != 'private_R_S_epsilon_and_weight_decay_divided_by_M'):
            raise ValueError('Private Adam changes private R/S epsilon and weight decay; gradient scaling is unchanged')
        (objects, bodies, calls, law) = (1, 1, maximum, 'private_Adam_one_source_owner_per_update')
        train_calls = {'known_source_TRAIN_forward_calls': maximum * 4, 'known_source_TRAIN_backward_calls': maximum * 4}
    elif config.get('architecture') == 'masked_core':
        work = CORE_WORK.get(config.get('core_condition'))
        if work is None or unit != 'epochs' or config.get('macro_cycle_law') != 'one_complete_old_state_all_owner_cycle' or (config.get('source_optimizer_count') != work[0]) or (config.get('optimizer_calls_per_cycle') != work[0]) or (config.get('physical_slow_bodies') != work[1]) or (config.get('members') != work[2]):
            raise ValueError('Masked CORE must charge every distinct source optimizer in a complete old-state macro cycle')
        (objects, bodies, calls, law) = (work[0], work[1], maximum * work[0], 'masked_CORE_all_source_owner_macro_cycles')
        train_calls = {'known_source_TRAIN_forward_calls': maximum * work[3], 'known_source_TRAIN_backward_calls': maximum * work[4]}
    elif 'source_arm' in config and 'source_optimizer_count' in config and ('link_inputs' not in config) and ('molecular_inputs' not in config):
        objects = number(config['source_optimizer_count'], 'source optimizer objects', integer=True)
        bodies = 4 if config['source_arm'] == 'untied' else 1
        if objects != (4 if config['source_arm'] == 'untied' else 1) or unit != 'epochs':
            raise ValueError('Coordinate four-object source cycles require epochs and the exact ownership count')
        calls = objects * maximum
        law = 'coordinate_full_graph_source_cycles'
    elif 'molecular_inputs' in config:
        (objects, bodies, calls, law) = (1, 1, None, 'molecular_minibatch_updates_per_epoch_unresolved')
        if 'train_minibatches_per_epoch' in config:
            if unit != 'epochs':
                raise ValueError('Molecular minibatch accounting requires an epoch budget')
            batches = number(config['train_minibatches_per_epoch'], 'bound TRAIN minibatches per epoch', integer=True)
            (calls, law) = (batches * maximum, 'molecular_bound_minibatch_calls_per_full_epoch')
    elif 'buddy_inputs' in config:
        if unit != 'epochs' or config.get('train_minibatches_per_epoch') != 2303 or config.get('source_optimizer_count') != 1 or (config.get('physical_slow_bodies') != 1):
            raise ValueError('BUDDY charges every complete native keep-tail minibatch map')
        (objects, bodies, calls, law) = (1, 1, maximum * 2303, 'BUDDY_native_keep_tail_minibatch_calls_per_full_epoch')
        routes = 4 if config.get('buddy_arm') == 'original_gnnm' else 1
        train_calls = {'known_source_TRAIN_forward_calls': maximum * 2303 * routes, 'known_source_TRAIN_backward_calls': maximum * 2303 * routes}
    elif 'link_inputs' in config:
        (objects, bodies, calls, law) = (1, 1, None, 'link_native_minibatch_updates_per_epoch_unresolved')
        if 'train_minibatches_per_epoch' in config:
            if unit != 'epochs':
                raise ValueError('Link native minibatch accounting requires full epochs')
            batches = number(config['train_minibatches_per_epoch'], 'bound link TRAIN minibatches per epoch', integer=True)
            (calls, law) = (batches * maximum, 'link_bound_native_drop_tail_minibatch_calls_per_full_epoch')
    elif config.get('polyformer_mode') in POLYFORMER_MODES:
        if unit != 'optimizer_updates':
            raise ValueError('Pinned PolyFormer controller owns one grouped optimizer per declared update')
        (objects, bodies, calls, law) = (1, 1, maximum, 'polyformer_one_grouped_optimizer_one_shared_slow_body')
    elif config.get('architecture') in {'single', 'factorized_single', 'original_gnnm', 'internal_be', 'joint_untied'}:
        (objects, bodies) = (1, 4 if config['architecture'] == 'joint_untied' else 1)
        calls = maximum if unit == 'optimizer_updates' else None
        law = 'native_one_optimizer_per_declared_update'
    else:
        raise ValueError('Unknown fit optimizer/body semantics; prepare an exact source-specific packet first')
    if unit == 'optimizer_updates' and objects != 1:
        raise ValueError('Optimizer-update budget cannot hide multiple paid optimizer calls')
    return {'optimizer_objects': objects, 'physical_slow_bodies': bodies, 'known_optimizer_step_calls': calls, 'budget_unit': unit, 'registered_budget_units': maximum, 'law': law, **train_calls, **constructor_overhead}


def routes(run):
    config = run['config']
    if 'members' in config:
        return number(config['members'], 'actual route count', integer=True)
    if 'molecular_inputs' in config:
        return 4 if config['arm'].startswith('be_') else 1
    if 'link_inputs' in config:
        return 1
    raise ValueError('Unknown selected-bank route contract')


def actual_seed(run):
    config = run['config']
    if 'initialization_seed' in config:
        return config['initialization_seed']
    if 'architecture' in config or 'source_arm' in config:
        return run['seed'] + 1000003 * run['body']
    raise ValueError('Independent member initialization ancestry is unresolved')


def bank_roster(study):
    runs = {r['run_id']: r for r in study['runs']}
    banks = []
    singles = {r['config']['fit_parents'][0] for r in study['runs']
               if r['kind'] == 'ensemble_assembly' and len(r['config'].get('fit_parents', [])) == 1}
    excluded = {'initialization_seed', 'dropout_seed', 'shuffle_seed', 'resolved_selection',
                'reservation_mib', 'maximum_cuda_bytes', 'estimated_seconds'}
    for row in study['runs']:
        if row['kind'] != 'ensemble_assembly':
            continue
        fits = row['config'].get('fit_parents')
        if not isinstance(fits, list) or len(fits) not in (1, 4) or len(set(fits)) != len(fits):
            raise ValueError('Each bank must declare one selected model group or four separate member fits')
        parents = [runs.get(identifier) for identifier in fits]
        if any(r is None or r['kind'] != 'model_fit' or r['config']['stage'] != 'comparative' for r in parents):
            raise ValueError('A prediction bank omits a required comparative fit')
        if not set(fits) <= set(row['parents']):
            raise ValueError('Bank fit custody must also be a dependency')
        independent = len(fits) == 4
        grouped = row['config'].get('independent_bank_contract')
        views_per_body = 1
        if grouped is not None:
            if (not independent or grouped != {'schema': 'four-own-four-view-v1', 'bodies': 4,
                    'views_per_body': 4, 'served_views': 16}
                    or any(r['config'].get('gru_stage') != 'continuation'
                           or r['config'].get('gru_arm') not in {'G1', 'GF1'} for r in parents)):
                raise ValueError('Only four complete four-view GRU learners may declare the sixteen-view bank')
            views_per_body = 4
        if independent:
            if (parents[0]['body'] != 0 or sorted(r['body'] for r in parents) != [0, 1, 2, 3]
                    or any(routes(r) != views_per_body or fit_semantics(r)['optimizer_objects'] != 1 for r in parents)
                    or len({actual_seed(r) for r in parents}) != 4):
                raise ValueError('Genuine I4 requires four separately initialized one-route optimizer trajectories')
            if any(any(r.get(key) != parents[0].get(key)
                       for key in ('method', 'adapter', 'split', 'seed', 'budget', 'selection'))
                   for r in parents[1:]):
                raise ValueError('Equipped I4 members differ in their method/source/budget/selector contract')
            rng_excluded = set(excluded)
            if 'typed_context_source_contract' in parents[0]['config']:
                rng_excluded.add('master_scaler_parent_fit_id')
            if parents[0]['config'].get('polyformer_mode') in POLYFORMER_MODES:
                streams = [r['config'].get('dropout_seeds') for r in parents]
                if (any(not isinstance(seeds, list) or not seeds
                        or any(type(seed) is not int for seed in seeds)
                        or len(set(seeds)) != len(seeds) for seeds in streams)
                        or any(set(streams[i]) & set(streams[j]) for i in range(4) for j in range(i))):
                    raise ValueError('Pinned PolyFormer I4 requires disjoint explicit private dropout seeds')
                rng_excluded.add('dropout_seeds')
                if parents[0]['config'].get('factor_initialization') == 'first_rademacher':
                    seeds = [r['config'].get('factor_seed') for r in parents]
                    if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != 4:
                        raise ValueError('Pinned PolyFormer I4 requires distinct explicit factor seeds')
                    rng_excluded.add('factor_seed')
            configs = [{k: v for k, v in r['config'].items() if k not in rng_excluded} for r in parents]
            family = staged_family(parents[0]['config'])
            if family and parents[0]['config'][family + '_stage'] == 'continuation':
                from run_study import paid_parent_ids
                ancestors = [runs.get(paid_parent_ids(r)[0]) for r in parents]
                if (any(a is None or a['kind'] != 'model_fit'
                        or a['config'].get(family + '_stage') != 'source'
                        or any(a.get(k) != r.get(k) for k in ('body', 'split', 'seed'))
                        for a, r in zip(ancestors, parents))
                        or len({a['run_id'] for a in ancestors}) != 4
                        or len({actual_seed(a) for a in ancestors}) != 4):
                    raise ValueError('Continued I4 requires four distinct matching paid source acquisitions')
                for config, ancestor in zip(configs, ancestors):
                    config['lifecycle'] = {**config['lifecycle'], 'parent_fit_ids': ['member_paid_source'],
                        'matching_paid_source_contract': {
                            **{k: ancestor[k] for k in ('method', 'adapter', 'split', 'seed', 'budget', 'selection')},
                            'config': {k: v for k, v in ancestor['config'].items() if k not in rng_excluded}}}
            if any(c != configs[0] for c in configs[1:]):
                raise ValueError('Equipped I4 members have different learning/data/selection contracts')
        banks.append({'run_id': row['run_id'], 'method': row['method'], 'split': row['split'], 'seed': row['seed'],
            'fit_parents': fits, 'routes': 4 * views_per_body if independent else routes(parents[0]),
            'genuine_I4': independent, 'independent_bodies': 4 if independent else 1,
            'views_per_body': views_per_body if independent else routes(parents[0]),
            'genuine_I16': independent and views_per_body == 4,
            'member0_reused': independent and fits[0] in singles})
    return banks


def reference_matches(study):
    """Every declared candidate cohort must retain its named reference cohort."""
    roster = [r for r in study['runs'] if r['config']['stage'] in {'comparative', 'bank_export'}]
    names = {}
    for row in roster:
        names.setdefault(row['method'], set()).add((canonical(row['split']), row['seed']))
    # Exact exported labels win. Source aliases are accepted only when no
    # method with that label exists; no cross-study lookup or fitted reuse.
    for row in roster:
        for name in (row['config'].get('arm'), row['config'].get('source_arm')):
            if name and name not in names:
                names[name] = {(canonical(r['split']), r['seed']) for r in roster
                    if r['config'].get('arm') == name or r['config'].get('source_arm') == name}
    result = []
    for comparison in study['comparisons']:
        references = comparison.get('references', []) + comparison.get('equipped_I4', [])
        candidates = comparison.get('candidates', [])
        if candidates and not references:
            raise ValueError('Candidate comparisons require explicit matched references')
        required_cohorts = set().union(*(names.get(name, set()) for name in candidates)) if candidates else set().union(*names.values())
        missing_names = [name for name in references + candidates if name not in names]
        missing_cohorts = {name: sorted(required_cohorts - names.get(name, set())) for name in references
                           if required_cohorts - names.get(name, set())}
        result.append({'references': references, 'candidates': candidates, 'missing_names': missing_names,
                       'missing_reference_cohorts': missing_cohorts,
                       'matched': not missing_names and not missing_cohorts})
    return result


def graph_check(studies, barriers):
    records = {}
    for study in studies:
        for run in study['runs']:
            key = study['study_id'] + '/' + run['run_id']
            if key in records:
                raise ValueError('Duplicate study/run identity')
            stage = run['config'].get('stage')
            if stage not in STAGE_KIND or STAGE_KIND[stage] != run['kind']:
                raise ValueError('Unsupported or mismatched phase/kind; never bypass a gate by omitting stage')
            records[key] = {'study': study['study_id'], 'run': run}
    deps = {}
    for (key, row) in records.items():
        deps[key] = [row['study'] + '/' + identifier for identifier in row['run']['parents']] + barriers.get(key, [])
        if len(deps[key]) != len(set(deps[key])) or any((p not in records for p in deps[key])):
            raise ValueError('Missing/repeated master dependency for ' + key)
        if row['run']['kind'] == 'model_fit':
            from run_study import paid_parent_ids
            run = row['run']
            model_parents = [p for p in run['parents'] if records[row['study'] + '/' + p]['run']['kind'] == 'model_fit']
            paid = paid_parent_ids(run)
            if 'sehgnn_arm' in run['config']:
                sehgnn_semantics(run)
            if 'private_response_source_contract' in run['config']:
                from response_semantics import validate_response_run
                validate_response_run(run)
            if (run['budget'].get('prefixes') and 'sehgnn_arm' not in run['config']) and 'private_response_source_contract' not in run['config'] or set(model_parents) != set(paid):
                raise ValueError('Source/warm-start/continuation lifecycle requires its explicit typed adapter contract')
            if 'typed_context_source_contract' in run['config']:
                from typed_context_semantics import validate_typed_context_run
                validate_typed_context_run(run)
                previous=run['config'].get('master_scaler_parent_fit_id')
                if previous is not None:
                    parent=records[row['study']+'/'+previous]['run']
                    if (row['study']+'/'+previous not in barriers.get(key,[]) or parent['kind']!='model_fit' or parent['config'].get('typed_context_arm')!=run['config']['typed_context_arm'] or parent['seed']!=run['seed']-1 or parent['body']!=run['body']):
                        raise ValueError('Exact preceding samecondition/body terminalmaster scaler barrier required')
            family = staged_family(run['config'])
            if family and run['config'][family + '_stage'] == 'continuation':
                if not paid:
                    raise ValueError('PT continuation must pay for its own declared source acquisition')
                parent = records[row['study'] + '/' + paid[0]]['run']
                expected_arm = source_parent_arm(family, run['config'][family + '_arm'])
                if parent['config'].get(family + '_stage') != 'source' or parent['config'].get(family + '_arm') != expected_arm or any((parent.get(k) != run.get(k) for k in ('split', 'seed', 'body'))) or (parent['config'].get('stage') != run['config']['stage']) or (parent['selection'].get('checkpoint0') is not False) or (run['selection'].get('checkpoint0') is not False) or (parent['budget']['policy'] != 'full_budget') or (parent['budget']['unit'] != 'optimizer_updates'):
                    raise ValueError('Staged continuation has mismatched or ineligible paid source ancestry')
            elif run['config'].get('curvature_stage') in ('trials', 'continuation'):
                from run_study import require_curvature_paid_parent
                if len(paid) != 1:
                    raise ValueError('Curvature typed stage requires one paid source ancestor')
                require_curvature_paid_parent(records[row['study'] + '/' + paid[0]]['run'], run)
            elif run['config'].get('nsd_arm') == 'geometry_branch':
                from run_study import require_nsd_geometry_parent
                if len(paid) != 1:
                    raise ValueError('NSD geometry requires one newly paid100 warm')
                require_nsd_geometry_parent(records[row['study'] + '/' + paid[0]]['run'], run)
                screen_id = run['config'].get('screen_parent')
                screen = records.get(row['study'] + '/' + str(screen_id), {}).get('run')
                if screen is None or screen_id not in run['parents'] or screen['kind'] != 'selection' or (screen['config'].get('stage') != 'geometry_screen_' + run['config']['stage']) or (screen['parents'] != paid) or (screen['config'].get('warm_parent') != paid[0]) or any((screen.get(k) != run.get(k) for k in ('split', 'seed', 'body'))) or any((screen['config'].get(k) != run['config'].get(k) for k in ('initialization_seed', 'learning_rate', 'dropout', 'node_inputs', 'nsd_sources'))) or (screen['config'].get('common_native_forwards') != 17) or (screen['config'].get('extra_offnode_spmm') != 17):
                    raise ValueError('Same-source common17 TRAIN-only native action screen required')
            elif run['config'].get('growth_stage') == 'post':
                from run_study import require_growth_terminal20_parent
                if len(paid) != 1:
                    raise ValueError('Growth post requires one newly paid terminal20 warm parent')
                require_growth_terminal20_parent(records[row['study'] + '/' + paid[0]]['run'], run)
            elif paid and run['config'].get('architecture') == 'paid_native_label_stage':
                parent = records[row['study'] + '/' + paid[0]]['run']
                capture = records.get(row['study'] + '/' + run['config'].get('capture_parent', ''), {}).get('run')
                if parent['config'].get('polynormer_qk_arm') != 'bare_native_single' or any((parent.get(k) != run.get(k) for k in ('split', 'seed', 'body'))) or parent['config']['node_inputs'] != run['config']['node_inputs'] or (parent['config']['stage'] != run['config']['stage']) or (parent['selection'].get('checkpoint0') is not False) or (run['selection'].get('checkpoint0') is not False) or (parent['budget'] != {'unit': 'optimizer_updates', 'policy': 'full_budget', 'maximum': 5000}) or (capture is None) or (capture['config'].get('native_capture') is not True) or (capture['kind'] != 'selection') or (capture['config'].get('stage') != 'native_capture') or (run['config']['capture_parent'] not in run['parents']) or any((capture.get(k) != run.get(k) for k in ('split', 'seed', 'body'))) or (capture['config'].get('capture_source') != run['config']['label_sources']['capture']) or (capture['config'].get('classes') != run['config']['classes']) or (capture['config'].get('native_fit_parent') != paid[0]) or (capture['parents'] != paid):
                    raise ValueError('Label stage must bind its matching new paid own-native/capture cohort')
            elif paid:
                raise ValueError('Only an explicitly typed staged continuation lifecycle is admitted')
    if set(barriers) - set(records):
        raise ValueError('Barrier addresses an unknown run')
    (done, visiting) = (set(), set())

    def visit(key):
        if key in visiting:
            raise ValueError('Cyclic master dependencies')
        if key in done:
            return
        visiting.add(key)
        for parent in deps[key]:
            visit(parent)
        visiting.remove(key)
        done.add(key)
    for key in records:
        visit(key)
    return (records, deps)


def typed_counts(studies):
    stages = {}
    totals = Counter()
    unknown_updates = []
    bounded_updates = []
    native_stopping_updates = []
    amp_bounded_updates = []
    trial_bounded_updates = []
    for study in studies:
        runs = study['runs']
        if all((run['kind'] == 'author_cli_group' for run in runs)):
            from author_cli_spec import author_counts
            quantities = author_counts(study)
            counts = {key: value for (key, value) in quantities.items() if type(value) is int}
            counts['run_records'] = len(runs)
            stages.setdefault('author_cli', Counter()).update(counts)
            totals.update(counts)
            continue
        for run in runs:
            stage = run['config']['stage']
            count = stages.setdefault(stage, Counter())
            count['run_records'] += 1
            totals['run_records'] += 1
            if run['kind'] == 'model_fit':
                semantics = fit_semantics(run)
                family = staged_family(run['config'])
                continuation = family is not None and run['config'][family + '_stage'] == 'continuation' or run['config'].get('growth_stage') == 'post' or run['config'].get('nsd_arm') == 'geometry_branch' or (run['config'].get('curvature_stage') == 'continuation')
                quantities = {'atomic_fit_run_groups': 1, 'paid_optimizer_starts': 0 if 'bounded_optimizer_step_calls_maximum' in semantics else semantics.get('serial_stepped_Adam_constructor_objects', semantics['optimizer_objects']), 'physical_slow_body_instances': semantics['physical_slow_bodies'], 'initial_training_segments': int(not continuation), 'paid_continuation_segments': int(continuation), 'multi_optimizer_cohort_groups': int(semantics['optimizer_objects'] > 1), 'multi_route_one_optimizer_groups': int(routes(run) > 1 and semantics['optimizer_objects'] == 1)}
                units = run['budget']['maximum']
                if 'growth_stage' in run['config']:
                    quantities.update(growth_registered_group_outer_cycles=units, growth_registered_owner_outer_cycles=units * semantics['optimizer_objects'], growth_new_paid_warm_fit_groups=int(run['config']['growth_stage'] == 'warm'), growth_paid_post_fit_groups=int(run['config']['growth_stage'] == 'post'))
                if 'native_stopping_updates_maximum' in semantics:
                    from native_stopping import native_count_quantities
                    quantities.update(native_count_quantities(semantics))
                    if 'source_detached_prefix_forward_calls_upper_bound' in semantics:
                        quantities['source_detached_prefix_forward_calls_upper_bound'] = semantics['source_detached_prefix_forward_calls_upper_bound']
                    native_stopping_updates.append(study['study_id'] + '/' + run['run_id'])
                else:
                    quantities['registered_' + run['budget']['unit']] = units
                if 'native_stopping_updates_maximum' in semantics:
                    pass
                elif semantics.get('known_sehgnn_AMP_bounded_commit_fit_groups'):
                    quantities['AMP_bounded_commit_fit_groups'] = 1
                    quantities['bounded_optimizer_step_calls_minimum'] = semantics['known_sehgnn_optimizer_commit_lower_bound']
                    quantities['bounded_optimizer_step_calls_maximum'] = semantics['known_sehgnn_optimizer_commit_upper_bound']
                    bounded_updates.append(study['study_id'] + '/' + run['run_id'])
                    amp_bounded_updates.append(study['study_id'] + '/' + run['run_id'])
                elif 'bounded_optimizer_step_calls_maximum' in semantics:
                    quantities['bounded_trial_fit_groups'] = 1
                    quantities['bounded_optimizer_step_calls_minimum'] = semantics['bounded_optimizer_step_calls_minimum']
                    quantities['bounded_optimizer_step_calls_maximum'] = semantics['bounded_optimizer_step_calls_maximum']
                    quantities['bounded_paid_stepped_Adam_objects_minimum'] = semantics['bounded_paid_stepped_Adam_objects_minimum']
                    quantities['bounded_paid_stepped_Adam_objects_maximum'] = semantics['bounded_paid_stepped_Adam_objects_maximum']
                    bounded_updates.append(study['study_id'] + '/' + run['run_id'])
                    trial_bounded_updates.append(study['study_id'] + '/' + run['run_id'])
                elif 'typed_context_epoch_maximum' in semantics:
                    from typed_context_semantics import count_quantities
                    quantities.update(count_quantities(semantics))
                    quantities['bounded_optimizer_step_calls_minimum']=semantics['typed_context_AMP_commit_minimum']
                    quantities['bounded_optimizer_step_calls_maximum']=semantics['typed_context_AMP_commit_maximum']
                    bounded_updates.append(study['study_id']+'/'+run['run_id'])
                    amp_bounded_updates.append(study['study_id']+'/'+run['run_id'])
                elif semantics['known_optimizer_step_calls'] is None:
                    quantities['unresolved_epoch_fit_groups'] = 1
                    unknown_updates.append(study['study_id'] + '/' + run['run_id'])
                else:
                    quantities['known_registered_optimizer_step_calls'] = semantics['known_optimizer_step_calls']
                for name in ('known_source_TRAIN_forward_calls', 'known_source_TRAIN_backward_calls'):
                    if name in semantics:
                        quantities['partial_registered_' + name] = semantics[name]
                for (name, value) in semantics.items():
                    if name.startswith(('known_sehgnn_', 'known_response_', 'known_label_corrector_', 'known_relation_')):
                        quantities['registered_' + name] = value
                if 'unstepped_pre_install_Adam_constructor_objects' in semantics:
                    quantities['unstepped_pre_install_Adam_constructor_objects'] = semantics['unstepped_pre_install_Adam_constructor_objects']
                if 'serial_stepped_Adam_constructor_objects' in semantics:
                    quantities['serial_stepped_Adam_constructor_objects'] = semantics['serial_stepped_Adam_constructor_objects']
                if 'curvature_stage' in run['config']:
                    quantities['curvature_' + run['config']['curvature_stage'] + '_paid_groups'] = 1
                if 'oipg_policy' in run['config']:
                    quantities.update({k: semantics[k] for k in ('source_complete_body_constructors', 'source_discarded_body_constructors', 'source_optimizer_constructors')})
                    quantities['registered_OIPG_source_old_state_gradient_collections'] = semantics['known_source_old_state_gradient_collections']
                count.update(quantities)
                totals.update(quantities)
            else:
                name = 'paid_NSD_geometry_common_TRAIN_screen_jobs' if stage.startswith('geometry_screen_') else 'paid_native_capture_jobs' if stage == 'native_capture' else {'selection': 'search_selection_jobs', 'ensemble_assembly': 'bank_exports', 'calibration': 'scalar_CAL_fit_jobs', 'evaluation': 'evaluation_jobs'}[run['kind']]
                count[name] += 1
                totals[name] += 1
                if run['kind'] == 'calibration':
                    steps = number(run['config']['steps'], 'scalar calibration steps', integer=True)
                    count['registered_scalar_CAL_updates'] += steps
                    totals['registered_scalar_CAL_updates'] += steps
        banks = bank_roster(study)
        totals.update({'served_banks': len(banks), 'served_route_views': sum((b['routes'] for b in banks)), 'genuine_separately_selected_I4_banks': sum((b['genuine_I4'] for b in banks)), 'genuine_four_body_one_view_I4_banks': sum((b['genuine_I4'] and (not b['genuine_I16']) for b in banks)), 'genuine_four_body_four_view_I16_banks': sum((b['genuine_I16'] for b in banks)), 'reused_exact_member0_fit_references': sum((b['member0_reused'] for b in banks))})
    return {'scope': 'exact_registered_contract_roster_not_executed', 'totals': dict(totals), 'by_stage': {key: dict(value) for (key, value) in sorted(stages.items())}, 'model_optimizer_step_call_total_fully_determined': not unknown_updates and (not bounded_updates) and (not native_stopping_updates), 'model_optimizer_step_call_total': totals['known_registered_optimizer_step_calls'] if not unknown_updates and (not bounded_updates) and (not native_stopping_updates) else None, 'native_SeHGNN_optimizer_attempt_total': totals['registered_known_sehgnn_optimizer_attempts'], 'native_SeHGNN_optimizer_commit_bounds': [totals['registered_known_sehgnn_optimizer_commit_lower_bound'], totals['registered_known_sehgnn_optimizer_commit_upper_bound']], 'model_optimizer_step_call_minimum': totals['known_registered_optimizer_step_calls'] + totals['bounded_optimizer_step_calls_minimum'] + totals['native_optimizer_updates_minimum_sum'] if not unknown_updates else None, 'model_optimizer_step_call_maximum': totals['known_registered_optimizer_step_calls'] + totals['bounded_optimizer_step_calls_maximum'] + totals['native_optimizer_update_cap_sum'] if not unknown_updates else None, 'paid_stepped_Adam_starts_minimum': totals['paid_optimizer_starts'] + totals['bounded_paid_stepped_Adam_objects_minimum'], 'paid_stepped_Adam_starts_maximum': totals['paid_optimizer_starts'] + totals['bounded_paid_stepped_Adam_objects_maximum'], 'native_stopping_fit_groups': len(native_stopping_updates), 'native_stopping_run_keys': native_stopping_updates, 'finite_bounded_optimizer_groups': len(bounded_updates), 'bounded_optimizer_run_keys': bounded_updates, 'finite_bounded_trial_groups': len(trial_bounded_updates), 'bounded_trial_run_keys': trial_bounded_updates, 'native_AMP_bounded_commit_fit_groups': len(amp_bounded_updates), 'native_AMP_bounded_commit_run_keys': amp_bounded_updates, 'curvature_source_acquisition_fit_groups': totals['curvature_warm_paid_groups'], 'curvature_paid_finite_trial_groups': totals['curvature_trials_paid_groups'], 'curvature_continuation_segments': totals['curvature_continuation_paid_groups'], 'epoch_fit_groups_with_unresolved_minibatch_update_count': len(unknown_updates), 'unresolved_update_run_keys': unknown_updates, 'actual_model_optimizer_updates': 0, 'actual_scalar_CAL_updates': 0, 'actual_predictive_scores': 0, 'cross_study_fit_reuse': [], 'source_acquisition_fit_groups': 0, 'continuation_segments': 0, 'extension_segments': 0}


def load_packet(root, entry):
    packet_id = entry['id']
    plan = json.loads(read_bound(root, entry['plan']))
    summary = json.loads(read_bound(root, entry['summary']))
    if plan.get('schema_version') != 1 or plan.get('slots_per_gpu') != 2:
        raise ValueError('Unsupported packet plan schema')
    if plan.get('scientific_launch_enabled') is not False or plan.get('human_GO_required') is not True:
        raise ValueError('Inputs must be disabled packets requiring human GO')
    plan_hash = fingerprint(plan)
    for declared in (entry.get('plan_hash'), summary.get('plan_hash')):
        if declared is not None and declared != plan_hash:
            raise ValueError('Packet PLAN/SUMMARY identity differs')
    for flag in ('training_started', 'benchmark_training_started', 'GO_receipt_created'):
        if summary.get(flag, False) is not False:
            raise ValueError('Master preparation cannot import an executed/approved packet as disabled work')
    for field in ('benchmark_optimizer_fits_executed', 'predictive_scores_computed', 'actual_model_fits',
                  'actual_predictive_scores', 'scientific_training_runs'):
        if summary.get(field, 0) != 0:
            raise ValueError('Input packet contains executed scientific work: ' + field)
    studies, bindings = [], [entry['plan'], entry['summary']]
    for row in plan['studies']:
        observed = receipt(root, row['path'])
        raw = read_bound(root, observed)
        spec = validate(json.loads(gzip.decompress(raw).decode() if row['path'].endswith('.gz') else raw))
        if fingerprint(spec) != row['spec_hash']:
            raise ValueError('Study contract differs from packet PLAN')
        if any(s['study_id'] == spec['study_id'] for s in studies):
            raise ValueError('Repeated study in packet')
        for value in spec['source']['files']:
            validate_receipt(value)
        # Metadata-only preparation preserves declared source/data bindings.
        # Target/graph arrays are never opened or rehashed here. The execution
        # controller's bind_source rechecks actual files at its admitted stage.
        bindings.append(observed)
        studies.append(spec)
    coverage = []
    for value in entry.get('coverage', []):
        raw = read_bound(root, value)
        bindings.append(value)
        coverage.append({'receipt': value, 'text': raw.decode('utf-8')})
    barriers = plan.get('barriers', {})
    records, deps = graph_check(studies, barriers)
    assigned = set()
    for group in plan['groups']:
        for key in group['run_keys']:
            if key in assigned or key not in records:
                raise ValueError('Repeated/unknown packet assignment')
            assigned.add(key)
    if assigned != set(records):
        raise ValueError('Packet has unassigned registered jobs')
    references = {s['study_id']: reference_matches(s) for s in studies}
    if any(not r['matched'] for rows in references.values() for r in rows):
        raise ValueError('A packet omits a matched required reference cohort')
    counts = typed_counts(studies)
    return {'id': packet_id, 'plan': plan, 'plan_hash': plan_hash, 'summary': summary,
            'studies': studies, 'bindings': merge_bindings(bindings), 'coverage': coverage,
            'references': references, 'counts': counts,
            'declared_source_data_bindings_rehashed': False}


def readiness(root, index):
    if index.get('schema') != 'disabled-master-input-index-v1':
        raise ValueError('Use disabled-master-input-index-v1')
    packets, report, seen = [], [], set()
    for entry in index.get('packets', []):
        if not isinstance(entry.get('id'), str) or not entry['id'] or entry['id'] in seen:
            raise ValueError('Unique explicit packet IDs required')
        seen.add(entry['id'])
        if entry.get('status') == 'pending':
            item = {'id': entry['id'], 'status': 'pending_unmaterialized', 'declaration': copy.deepcopy(entry)}
            if 'manifest' in entry:
                item['manifest'] = json.loads(read_bound(root, entry['manifest']))
            report.append(item)
            continue
        try:
            packet = load_packet(root, entry)
            packets.append(packet)
            report.append({'id': entry['id'], 'status': 'validated_disabled_metadata',
                'plan_hash': packet['plan_hash'], 'study_count': len(packet['studies']),
                'counts': packet['counts'], 'input_summary': packet['summary'],
                'references': packet['references'], 'coverage': packet['coverage']})
        except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
            report.append({'id': entry['id'], 'status': 'blocked_input', 'error_type': type(error).__name__, 'error': str(error)})
    ids = [s['study_id'] for p in packets for s in p['studies']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate study ID across immutable packets; no implicit dedup/reuse')
    merge_bindings(*(s['source']['files'] for p in packets for s in p['studies']))
    return packets, {'schema': 'disabled-master-readiness-v1', 'packets': report,
        'invalid_inputs': [r['id'] for r in report if r['status'] == 'blocked_input'],
        'pending_unmaterialized_packets': [r['id'] for r in report if r['status'] == 'pending_unmaterialized'],
        'ready_disabled_packet_count': len(packets), 'scientific_launch_enabled': False,
        'human_GO_required': True, 'all_249_predictive_candidates_covered': False}


def controller_bindings(root, relative):
    runtime = safe_path(root, relative)
    if runtime.is_relative_to(Path(root).resolve() / 'training_pipeline_v2_20261012'):
        raise ValueError('Common controller must be frozen outside the live package')
    values = [receipt(root, str((runtime / name).relative_to(Path(root).resolve()))) for name in CONTROLLER_FILES]
    tree = ast.parse((runtime / 'run_study.py').read_text())
    factories = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_factory']
    if len(factories) != 1 or not any(isinstance(n, ast.Constant) and isinstance(n.value, str)
            and 'gnnm_packet_' in n.value for n in ast.walk(factories[0])):
        raise ValueError('Common dispatcher must include the root namespace-isolating load_factory')
    return values


def scope(value, name):
    if not isinstance(value, dict) or value.get('kind') not in SCOPES or not value.get('scope'):
        raise ValueError('Explicit measured/provisional ' + name + ' scope required')
    if value['kind'] in {'measured', 'measured_proxy', 'mixed'} and not value.get('evidence'):
        raise ValueError('Measured resource scope requires saved evidence')
    if value['kind'] in {'measured', 'measured_proxy', 'mixed'}:
        for evidence in value['evidence']:
            validate_receipt(evidence)
    return copy.deepcopy(value)


def apply_resources(study, profiles):
    study = copy.deepcopy(study)
    override = profiles.get(study['study_id'])
    source_profile = copy.deepcopy(study['resources'].get('profile', {}))
    if override is None:
        scopes = {'memory': {'kind': 'unknown', 'scope': 'original packet declarations; explicit measured/provisional classification absent'},
                  'time': {'kind': 'provisional', 'scope': 'original estimated_seconds; actual training/serializer throughput unmeasured'}}
        run_overrides = {}
    else:
        scopes = {'memory': scope(override['memory'], 'memory'), 'time': scope(override['time'], 'time')}
        run_overrides = override.get('runs', {})
        if set(run_overrides) - {r['run_id'] for r in study['runs']}:
            raise ValueError('Resource override addresses an unknown registered run')
    for run in study['runs']:
        change = run_overrides.get(run['run_id'], {})
        if set(change) - {'reservation_mib', 'estimated_seconds'}:
            raise ValueError('Master resource overrides cannot alter learning/source/selector decisions')
        run['config'].update(change)
        number(run['config'].get('reservation_mib'), 'run reservation MiB', integer=True)
        number(run['config'].get('estimated_seconds'), 'run estimated seconds', zero=True)
        if 'maximum_cuda_bytes' in run['config']:
            run['config']['maximum_cuda_bytes'] = run['config']['reservation_mib'] * MIB
    study['resources']['master_resource_scope'] = {'scopes': scopes, 'original_packet_profile': source_profile,
                                                 'override': copy.deepcopy(override)}
    return study


def assemble(root, packets, ready, controller, controller_files, resources, output_relative, input_bindings, host_bindings=None):
    root = Path(root).resolve()
    capacities = resources['worker_capacities_cuda_bytes']
    if set(capacities) != set(DEFAULT_WORKERS):
        raise ValueError('Declare all three logical worker planning capacities')
    for name, value in capacities.items():
        number(value, name + ' CUDA capacity bytes', integer=True)
    if ready['invalid_inputs'] or not packets:
        raise ValueError('Cannot prepare master from invalid or empty compiled packet roster')
    studies, origins, barriers, original_groups = [], {}, {}, []
    files = merge_bindings(controller_files, input_bindings)
    for packet in packets:
        original_groups.extend(packet['plan']['groups'])
        for key, values in packet['plan'].get('barriers', {}).items():
            if key in barriers:
                raise ValueError('Repeated cross-study barrier identity')
            barriers[key] = copy.deepcopy(values)
        for original in packet['studies']:
            study = apply_resources(original, resources.get('study_profiles', {}))
            if study['study_id'] in origins:
                raise ValueError('Duplicate master study ID')
            origins[study['study_id']] = {'packet_id': packet['id'], 'input_plan_hash': packet['plan_hash'],
                'input_spec_hash': fingerprint(original), 'input_resource_scope': copy.deepcopy(original['resources'])}
            study['source']['files'] = merge_bindings(study['source']['files'], files, packet['bindings'])
            study['source']['master_controller'] = {'runtime_package': controller, 'files': controller_files,
                'adapter_dependencies': 'original frozen packet factory paths and own namespace imports preserved'}
            if all(run['kind'] == 'author_cli_group' for run in study['runs']):
                study['source']['worker_entrypoint'] = controller + '/run_author_cli.py'
            study['source']['master_input_identity'] = origins[study['study_id']]
            study['release']['global_plan_gates_required'] = True
            study['release']['master_gate_scope'] = 'all materialized packets, matched reference cohorts and assigned hosts'
            from retention_lifecycle import declare_policy
            declare_policy(root,study)
            validate(study)
            studies.append(study)
    if set(resources.get('study_profiles', {})) - set(origins):
        raise ValueError('Resource profile names an absent study')
    merge_bindings(*(s['source']['files'] for s in studies))
    records, deps = graph_check(studies, barriers)
    granularity = resources.get('group_granularity', 'study')
    if granularity not in {'study', 'dependency'}:
        raise ValueError('Use explicit study or actual dependency-component grouping')
    parents = ({key: key for key in records} if granularity == 'dependency' else
               {s['study_id']: s['study_id'] for s in studies})
    def find(identifier):
        while parents[identifier] != identifier:
            parents[identifier] = parents[parents[identifier]]
            identifier = parents[identifier]
        return identifier
    def union(left, right):
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)
    components = {}
    if granularity == 'dependency':
        for key, values in deps.items():
            for value in values:
                union(key, value)
        for group in original_groups:
            if group.get('must_remain_together'):
                for key in group['run_keys']:
                    union(min(group['run_keys']), key)
        for key in records:
            components.setdefault(find(key), []).append(key)
    else:
        for group in original_groups:
            names = {records[key]['study'] for key in group['run_keys']}
            for name in names:
                union(min(names), name)
        for key, values in deps.items():
            for value in values:
                union(records[key]['study'], records[value]['study'])
        for key, value in records.items():
            components.setdefault(find(value['study']), []).append(key)
    groups = []
    for identifier, component_keys in sorted(components.items()):
        keys = sorted(component_keys)
        names = {records[key]['study'] for key in keys}
        cost = sum(records[key]['run']['config']['estimated_seconds'] for key in keys)
        memory = max(records[key]['run']['config']['reservation_mib'] for key in keys) * MIB
        group = {'id': identifier, 'study_ids': sorted(names), 'run_keys': keys,
                 'estimated_seconds': cost, 'maximum_cuda_bytes': memory}
        if any(records[key]['run']['kind'] == 'author_cli_group' for key in keys):
            # The current source/private-library/runtime bindings are allocation-specific.
            # A separate 18.77 environment must be qualified before that protocol can move.
            group['allowed_workers'] = ['allocation/gpu0']
        groups.append(group)
    placement = assign(groups, capacities)
    assigned = {key: group['gpu_assignment'] for group in placement['groups'] for key in group['run_keys']}
    gates = {}
    gate_rows = [('whole_search', ['search', 'search_selection'], ['comparative', 'author_cli']),
        ('whole_comparative', ['comparative'], ['bank_export', 'calibration', 'evaluation']),
        ('whole_bank_exports', ['bank_export'], ['calibration']),
        ('whole_calibration', ['calibration'], ['evaluation'])]
    if any(record['run']['kind'] == 'author_cli_group' for record in records.values()):
        gate_rows.append(('whole_author', ['author_cli'], []))
    for name, phases, required in gate_rows:
        cohort = sorted(key for key, record in records.items() if record['run']['config']['stage'] in phases)
        blocked_jobs = [key for key, record in records.items() if record['run']['config']['stage'] in required]
        if blocked_jobs and not cohort:
            raise ValueError('A required global phase has no actual registered closure cohort')
        gates[name] = {'path': output_relative + '/gate_receipts/' + name + '.json',
            'source_stages': phases, 'required_for_stages': required,
            'required_hosts': sorted({assigned[key].split('/')[0] for key in cohort}),
            'all_packet_source_run_keys': cohort, 'input_packet_gate_receipts_accepted': False,
            'receipt_contract': 'status=complete;plan_hash=current_plan_hash;closed_phase=gate_name;host_receipts_keys=required_hosts'}
    plan = {'schema_version': 1, 'slots_per_gpu': 2, 'torch_threads_per_fit': 2,
        'runtime_package': controller, 'runtime_launcher': controller + '/launcher.py',
        'label_free_export_run_keys':sorted(s['study_id']+'/'+key for s in studies for key in s['release'].get('checkpoint_retention_policy',{}).get('audited_label_free_exports',{})),
        'checkpoint_retention_schema':'gnnm-prospective-checkpoint-lifetimes-v1',
        'immutable_source_bundle': {'package': controller, 'files': controller_files,
            'model_worker_entrypoint': controller + '/run_study.py', 'endpoint_worker_entrypoint': controller + '/run_endpoint.py'},
        'output_root': output_relative + '/execution', 'studies': [], 'groups': placement['groups'],
        'estimated_gpu_seconds': placement['estimated_gpu_seconds'], 'worker_capacities_cuda_bytes': capacities,
        'estimates_are_measurements': False, 'barriers': barriers, 'global_gates': gates,
        'runtime_compatibility_required': True,
        'runtime_requirements': copy.deepcopy(resources.get('runtime_requirements')),
        'host_bindings': copy.deepcopy(host_bindings or {'allocation': {'status': 'pending_live_verification_before_launch'},
            'server77': {'status': 'pending_live_verification_before_launch'}}),
        'scientific_launch_enabled': False, 'human_GO_required': True, 'approval_receipt_created': False}
    plan['group_granularity'] = granularity
    if 'phase_rendezvous' in resources:
        from phase_rendezvous import transport_policy, capacity_report
        plan['phase_rendezvous'] = copy.deepcopy(resources['phase_rendezvous'])
        transport_policy(plan, plan['phase_rendezvous'].get('remote_name'))
        plan['phase_rendezvous_capacity'] = capacity_report(plan, records,
            {key: {'worker': worker} for key, worker in assigned.items()})
    summary = {'schema': 'disabled-master-summary-v1', 'status': 'PREPARED_DISABLED_REQUIRES_COVERAGE_RESOURCE_REFERENCE_QA_AND_HUMAN_GO',
        'counts': typed_counts(studies), 'input_readiness': ready, 'input_packet_summaries': {p['id']: p['summary'] for p in packets},
        'input_packet_coverages': {p['id']: p['coverage'] for p in packets}, 'per_study': [], 'per_worker': {},
        'pending_unmaterialized_packets': ready['pending_unmaterialized_packets'],
        'all_249_predictive_candidates_covered': False, 'author_replication_ready': False,
        'scientific_launch_enabled': False, 'human_GO_required': True, 'GO_receipt_created': False,
        'actual_model_fits': 0, 'actual_predictive_scores': 0, 'source_data_arrays_opened': 0,
        'source_data_fingerprint_policy': 'preserved immutable packet declarations; controller rechecks files at admitted execution',
        'unknown_costs_are_not_zero': True, 'remote_capacities_are_planning_bounds_not_live_host_proofs': True}
    if 'phase_rendezvous_capacity' in plan:
        summary['phase_rendezvous_capacity'] = plan['phase_rendezvous_capacity']
    for study in studies:
        keys = [key for key, value in records.items() if value['study'] == study['study_id']]
        references = reference_matches(study)
        if any(not row['matched'] for row in references):
            raise ValueError('Rewritten master lost a matched reference cohort')
        workers = sorted({assigned[key] for key in keys})
        summary['per_study'].append({'study_id': study['study_id'], 'dataset': study['dataset'], 'origin': origins[study['study_id']],
            'counts': typed_counts([study]), 'references': references,
            'worker': workers[0] if len(workers) == 1 else None, 'workers': workers,
            'estimated_gpu_seconds': sum(records[k]['run']['config']['estimated_seconds'] for k in keys),
            'reservation_mib': max(records[k]['run']['config']['reservation_mib'] for k in keys),
            'resource_scope': study['resources']['master_resource_scope'],
            'preserved_comparison_contracts': study['comparisons'], 'preserved_data_evidence': study['data']['evidence']})
    for worker in DEFAULT_WORKERS:
        own = []
        for study in studies:
            own_runs = [run for run in study['runs'] if assigned[study['study_id'] + '/' + run['run_id']] == worker]
            if own_runs:
                own.append({**study, 'runs': own_runs})
        reservations = sorted((r['config']['reservation_mib'] for s in own for r in s['runs']), reverse=True)
        pair = sum(reservations[:2]) * MIB
        summary['per_worker'][worker] = {'study_ids': [s['study_id'] for s in own], 'counts': typed_counts(own),
            'estimated_gpu_seconds': placement['estimated_gpu_seconds'][worker],
            'estimated_serial_gpu_days': placement['estimated_gpu_seconds'][worker] / 86400,
            'two_slot_wall_time_is_not_half_serial_time': True, 'capacity_bytes': capacities[worker],
            'maximum_run_reservation_bytes': max(reservations, default=0) * MIB,
            'two_largest_registered_run_reservations_bytes': pair, 'worst_pair_fits_declared_capacity': pair <= capacities[worker],
            'concurrent_throughput_measured': False, 'live_hardware_and_external_headroom_qualification_pending': True,
            'resource_scopes_by_study': {s['study_id']: s['resources']['master_resource_scope'] for s in own}}
    return plan, studies, summary


def write_master(root, output_relative, plan, studies, summary):
    output = safe_path(root, output_relative)
    if output.is_relative_to(Path(root).resolve() / 'training_pipeline_v2_20261012') or output.exists():
        raise ValueError('Fresh output outside the live package required')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'studies').mkdir()
    for study in studies:
        path = output / 'studies' / (study['study_id'] + '.json.gz')
        path.write_bytes(gzip.compress((canonical(study) + '\n').encode(), mtime=0))
        plan['studies'].append({'path': str(path.relative_to(Path(root).resolve())), 'spec_hash': fingerprint(study)})
    summary['plan_hash'] = fingerprint(plan)
    for name, value in [('PLAN.json', plan), ('SUMMARY.json', summary), ('READINESS.json', summary['input_readiness'])]:
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    with (output / 'WORKER_COSTS.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['worker', 'atomic_fit_groups', 'paid_optimizer_starts', 'physical_bodies',
            'estimated_gpu_seconds', 'epoch_update_groups_unknown', 'capacity_bytes', 'worst_pair_bytes', 'worst_pair_fits'])
        writer.writeheader()
        for worker, row in summary['per_worker'].items():
            counts = row['counts']['totals']
            writer.writerow({'worker': worker, 'atomic_fit_groups': counts.get('atomic_fit_run_groups', 0),
                'paid_optimizer_starts': counts.get('paid_optimizer_starts', 0), 'physical_bodies': counts.get('physical_slow_body_instances', 0),
                'estimated_gpu_seconds': row['estimated_gpu_seconds'],
                'epoch_update_groups_unknown': row['counts']['epoch_fit_groups_with_unresolved_minibatch_update_count'],
                'capacity_bytes': row['capacity_bytes'], 'worst_pair_bytes': row['two_largest_registered_run_reservations_bytes'],
                'worst_pair_fits': row['worst_pair_fits_declared_capacity']})
    return summary['plan_hash']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--packet-index', required=True)
    parser.add_argument('--mode', choices=('dry-run', 'prepare'), default='dry-run')
    parser.add_argument('--controller-runtime')
    parser.add_argument('--resources')
    parser.add_argument('--host-bindings')
    parser.add_argument('--output')
    args = parser.parse_args()
    verify_allocation()
    root = Path(args.project_root).resolve()
    index_binding = receipt(root, args.packet_index)
    packets, ready = readiness(root, json.loads(read_bound(root, index_binding)))
    if args.mode == 'dry-run':
        print(json.dumps(ready, indent=2, allow_nan=False))
        return 2 if ready['invalid_inputs'] else 0
    if not all((args.controller_runtime, args.resources, args.output)):
        parser.error('prepare requires controller-runtime, resources and output')
    controller = str(safe_path(root, args.controller_runtime).relative_to(root))
    control_files = controller_bindings(root, controller)
    resource_binding = receipt(root, args.resources)
    resources = json.loads(read_bound(root, resource_binding))
    inputs = [index_binding, resource_binding]
    host_bindings = None
    if args.host_bindings:
        host_binding = receipt(root, args.host_bindings)
        inputs.append(host_binding)
        host_bindings = json.loads(read_bound(root, host_binding))
    for profile in resources.get('study_profiles', {}).values():
        for label in ('memory', 'time'):
            value = scope(profile[label], label)
            for evidence in value.get('evidence', []):
                if isinstance(evidence, dict):
                    read_bound(root, evidence)
                    inputs.append(evidence)
    plan, studies, summary = assemble(root, packets, ready, controller, control_files, resources, args.output, merge_bindings(inputs), host_bindings)
    digest = write_master(root, args.output, plan, studies, summary)
    print(canonical({'status': 'prepared_disabled', 'plan_hash': digest, 'output': args.output,
        'studies': len(studies), 'counts': summary['counts']['totals'], 'GO_receipt_created': False,
        'pending_packets': ready['pending_unmaterialized_packets']}))


if __name__ == '__main__':
    main()
