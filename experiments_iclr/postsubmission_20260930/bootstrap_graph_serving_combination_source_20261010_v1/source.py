"""Root-called closed-bank extension of the immutable graph serving study.

Import is stdlib only. This module owns no process, transport, retry or model
implementation. Root admits and calls run(support_path, output_path).
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
BASE = PHASE / 'common_wrapper_graph_reliability_source_20261010_v2'
BACKBONES = ('GAT', 'SAGE')
SEEDS = (7301, 7403, 7507)
UNWEIGHTED = ('shared4_coherent', 'shared4_paired_graph', 'shared4_rank1_lora_graph')
BOOTSTRAP = tuple(arm + '_bootstrap' for arm in UNWEIGHTED)
RULES = ('temperature_global', 'temperature_member', 'reliability_self',
         'linear_stacking', 'reliability_graph')
BASE_SUPPORT_SHA256 = '8dfa9fe7b73594906719c09edc0df7f60477196ee40da8c7b9fb7c62d271d727'
MAX_ABS_LOGIT_ERROR = 1e-4


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def metrics(torch, log_q, labels, native_wrong):
    """Same proper-loss/decision conventions as the preceding graph study."""
    probability = log_q.exp()
    correct = log_q.argmax(-1).eq(labels)
    truth = torch.nn.functional.one_hot(labels, probability.shape[-1]).to(probability)
    return {'nodes': len(labels), 'correct': correct.sum().item(),
            'accuracy': correct.to(torch.float64).mean().item(),
            'nll': -log_q.gather(1, labels[:, None]).mean().item(),
            'brier': (probability - truth).square().sum(-1).mean().item(),
            'repairs_native': (native_wrong & correct).sum().item(),
            'harms_native': (~native_wrong & ~correct).sum().item()}


def run(support_path, output_path):
    # The same literal route gate used by the existing root/native interface.
    if socket.gethostname() != 'anogena-2-0' or Path.cwd() != REPO:
        raise ValueError('Root must enter the literal authorized allocation/repository')
    if subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                               text=True).splitlines() != ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Wrong allocation GPU')
    if hashlib.sha256((BASE / 'support.py').read_bytes()).hexdigest() != BASE_SUPPORT_SHA256:
        raise ValueError('Immutable support helper changed')
    helper = load('_bootstrap_serving_support', BASE / 'support.py')
    require, sha, write_json, progress = helper.require, helper.sha, helper.write_json, helper.write_progress
    support_path = Path(support_path).resolve()
    require(support_path.is_relative_to(PHASE) and support_path.name == 'FROZEN_SUPPORT.json',
            'Exact root-owned support path required')
    contract = json.loads(support_path.read_text())
    require(contract['schema'] == 'bootstrap-graph-serving-combination-support-v1'
            and tuple(contract['backbones']) == BACKBONES and tuple(contract['seeds']) == SEEDS
            and tuple(contract['unweighted_arms']) == UNWEIGHTED
            and tuple(contract['bootstrap_arms']) == BOOTSTRAP
            and contract['new_banks'] == 36 and contract['native_export_calls'] == 36
            and contract['native_member_trajectories'] == 144
            and contract['nonduplicate_fusion_fit_calls'] == 900
            and contract['TEST_access'] is False and contract['new_GNN_training'] is False
            and contract['original_model_predictions_changed'] is False
            and contract['member_protection_cannot_be_repaired_by_static_serving'] is True,
            'Exact finite36-bank support required')
    fusion = contract['fusion']
    require(fusion['folds'] == 5 and fusion['fold_seed'] == 11709 and fusion['updates'] == 500
            and tuple(fusion['rules']) == RULES
            and Path(fusion['operators_path']) == BASE / 'operators.py'
            and sha(fusion['operators_path']) == fusion['operators_sha256']
            == 'f1c9a859aebc61591608e4d342b15219b1d6b01fb0dc35757b32ffa79628de8f'
            and Path(fusion['neighbor_operator_source']) == BASE / 'export_neighbors.py'
            and sha(fusion['neighbor_operator_source']) == fusion['neighbor_operator_source_sha256']
            == '68f238b2567cc50bebd80c071bb74614d5fbbf8e9ebb5fe5982dea391de48c54',
            'Reuse exact original operators and fitting opportunity')
    operators = load('_bootstrap_serving_operators', fusion['operators_path'])
    # The original export module imports the original stdlib support helper.
    previous_support = sys.modules.get('support')
    sys.modules['support'] = helper
    try:
        exporter = load('_bootstrap_serving_neighbor_operator', fusion['neighbor_operator_source'])
    finally:
        if previous_support is None:
            sys.modules.pop('support', None)
        else:
            sys.modules['support'] = previous_support

    contexts, expected = [], {}
    require([(c['backbone'], c['kind']) for c in contract['contexts']]
            == [(b, k) for b in BACKBONES for k in ('unweighted', 'bootstrap')],
            'All four native contexts in fixed order')
    for context in contract['contexts']:
        for bound in context['frozen_inputs']:
            require(sha(REPO / bound['path']) == bound['sha256'], 'Original frozen input changed')
        cfg = json.loads(Path(context['config']).read_text())
        require(cfg == context['configuration'] and sha(context['config']) == context['config_sha256']
                and cfg['backbone'] == context['backbone'] and tuple(cfg['seeds']) == SEEDS
                and sha(context['family_source']) == context['family_source_sha256'], 'Exact native config/source')
        complete_path = Path(context['family_root']) / 'COMPLETE_FAMILY.json'
        require(sha(complete_path) == context['complete_sha256']
                and sha(context['owner_end_path']) == context['owner_end_sha256'], 'Closed parent identity')
        complete = json.loads(complete_path.read_text())
        end = json.loads(Path(context['owner_end_path']).read_text())
        count = 12 if context['kind'] == 'unweighted' else 9
        require(complete['complete'] and complete['groups'] == complete['fit_units'] == count
                and complete['TEST_access'] is False and end['scientific_success']
                and end['direct_child_wait'] and end['child_pid_absent'] and end['owned_cuda_pid_absent'],
                'Full unweighted12/bootstrap9 closure required')
        index = {(r['arm'], r['seed']): r for r in complete['results']}
        arms = UNWEIGHTED if context['kind'] == 'unweighted' else BOOTSTRAP
        require([(b['arm'], b['seed']) for b in context['banks']]
                == [(arm, seed) for arm in arms for seed in SEEDS], 'All nine context banks')
        for bank in context['banks']:
            original = index[bank['arm'], bank['seed']]
            require(original == bank['original_group_record'] and len(original['fits']) == 1
                    and bank['members'] == 4 and bank['backbone'] == context['backbone']
                    and bank['key'] == f"{context['backbone']}_{bank['arm']}_seed{bank['seed']}"
                    and original['fits'][0]['start'] == bank['start_kind']
                    and original['fits'][0]['selected_state'] == bank['selected_state']
                    and original['fits'][0]['selected_step'] == bank['selected_step']
                    and sha(bank['archive']) == bank['archive_sha256'], 'Exact selected bank custody')
            require(bank['key'] not in expected, 'Duplicate bank')
            expected[bank['key']] = bank
        contexts.append((context, cfg))
    require(len(expected) == 36, 'All36 selected banks before any restore')
    checkpoint_path = support_path.parent / 'CHECKPOINT_BINDINGS.json'
    checkpoint_binding = json.loads(checkpoint_path.read_text())
    require(checkpoint_binding['schema'] == 'bootstrap-graph-selected-state-custody-v1'
            and checkpoint_binding['support_sha256'] == sha(support_path), 'Root checkpoint binding support identity')
    checkpoint_rows = checkpoint_binding['checkpoints']
    checkpoint_index = {r['path']: r for r in checkpoint_rows}
    require(len(checkpoint_rows) == len(checkpoint_index) == 36
            and set(checkpoint_index) == {b['selected_state'] for b in expected.values()},
            'Complete36 checkpoint hash roster required')
    for row in checkpoint_rows:
        require(sha(row['path']) == row['sha256'] and Path(row['path']).stat().st_size == row['bytes'],
                'Selected checkpoint binding changed')
    for bank in expected.values():
        bound = checkpoint_index[bank['selected_state']]
        require(bound['key'] == bank['key'] and bound['selected_step'] == bank['selected_step'],
                'Checkpoint key/selection binding changed')

    reuse = contract['reuse_original_graph_study']
    require(sha(reuse['path']) == reuse['sha256'] and sha(reuse['export_path']) == reuse['export_sha256']
            and tuple(reuse['backbones']) == BACKBONES and reuse['new_refits'] == 0,
            'Exact complete original graph study/export reuse')
    prior = json.loads(Path(reuse['path']).read_text())
    require(prior['complete_roster'] and prior['all_fixed_endpoints_finite']
            and len(prior['banks']) == 45 and prior['fit_calls'] == 855 and not prior['failures']
            and prior['TEST_access'] is False
            and prior['ordered_VALID_hashes'] == contract['ordered_VALID_hashes']
            and prior['fold_assignment_sha256'] == reuse['fold_assignment_sha256'], 'Complete original fusion closure')
    references = [r for r in prior['banks'] if r['backbone'] in BACKBONES]
    require({(r['backbone'], r['seed'], r['arm']) for r in references}
            == {(b, s, a) for b in BACKBONES for s in SEEDS for a in reuse['reference_arms']}
            and len(references) == 30, 'Exact30 unchanged reference banks')
    for row in references:
        require(sha(row['VALID_OOF_path']) == row['VALID_OOF_sha256'], 'Original reference OOF changed')

    output = helper.new_output(output_path)
    write_json(output / 'SUPPORT.json', contract)
    write_json(output / 'CHECKPOINT_BINDINGS.json', checkpoint_binding)
    import numpy as np
    import torch
    require(torch.__version__ == '2.1.2+cu118', 'Use original qualified native runtime')
    torch.set_num_threads(2)
    started, arrays, exports, identity = time.perf_counter(), {}, [], None
    progress(output / 'PROGRESS.json', {'banks_exported': 0, 'banks_completed': 0, 'fit_calls': 0})
    # Every selected identity was verified before the first new export.
    for context, cfg in contexts:
        native = load('_bootstrap_serving_native_' + context['backbone'] + '_' + context['kind'],
                      context['family_source'])
        scratch = output / ('native_restore_' + context['backbone'] + '_' + context['kind'])
        scratch.mkdir()
        family = native.Family(cfg, scratch, context['config_sha256'])
        require(len(family.data.x) == 11701, 'Exact full-node native support')
        all_ids = torch.arange(11701, device=family.device)
        P, isolates = exporter.neighbor_operator(torch, family.data.edge_index, 11701)
        for bank in context['banks']:
            bank_start = time.perf_counter()
            checkpoint = checkpoint_index[bank['selected_state']]
            state = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)
            state_keys = {'model', 'optimizer', 'streams', 'step', 'selection'}
            if context['kind'] == 'bootstrap':
                state_keys.add('bootstrap_weights_sha256')
                require(state['bootstrap_weights_sha256']
                        == bank['original_group_record']['fits'][0]['bootstrap']['weights_sha256'],
                        'Exact selected bootstrap weight identity')
            require(set(state) == state_keys and state['step'] == bank['selected_step'], 'Native selected state layout')
            model, unused_optimizer, unused_streams = family.make(bank['seed'], 0, bank['start_kind'], True, 4)
            del unused_optimizer, unused_streams
            model.load_state_dict(state['model'], strict=True)
            model.requires_grad_(False)
            model.eval()
            family.synchronize()
            forward_start = time.perf_counter()
            with torch.no_grad():
                full = family.logits(model, state['streams'], True, all_ids).detach().cpu()
            family.synchronize()
            forward_seconds = time.perf_counter() - forward_start
            require(full.dtype == torch.float32 and full.shape == (4, 11701, 10)
                    and torch.isfinite(full).all() and sha(checkpoint['path']) == checkpoint['sha256'],
                    'Finite exact immutable native four-route export')
            with np.load(bank['archive'], allow_pickle=False) as stored:
                require(set(stored.files) == helper.ARCHIVE_KEYS, 'Original VALID archive schema')
                a = {k: stored[k].copy() for k in helper.ARCHIVE_KEYS}
            require(sha(bank['archive']) == bank['archive_sha256'], 'Authoritative VALID archive changed during export')
            require(a['ids'].dtype == a['y'].dtype == np.int64
                    and a['ids'].shape == a['y'].shape == (5274,)
                    and np.array_equal(a['ids'], family.valid_ids.cpu().numpy())
                    and np.array_equal(a['y'], family.valid_y.cpu().numpy())
                    and len(np.unique(a['ids'])) == 5274 and set(a['y'].tolist()) == set(range(10))
                    and a['raw_logits'].dtype == a['probability_mean'].dtype == np.float32
                    and a['raw_logits'].shape == (4, 5274, 10) and a['probability_mean'].shape == (5274, 10)
                    and np.isfinite(a['raw_logits']).all() and np.isfinite(a['probability_mean']).all(),
                    'Exact finite original ordered VALID role')
            role_hashes = {k: hashlib.sha256(a[v].tobytes()).hexdigest()
                           for k, v in (('ids', 'ids'), ('labels', 'y'))}
            require(role_hashes == contract['ordered_VALID_hashes'], 'Fixed common VALID IDs/labels')
            if identity is None:
                identity = (a['ids'], a['y'])
            else:
                require(np.array_equal(a['ids'], identity[0]) and np.array_equal(a['y'], identity[1]), 'Identical roles')
            raw = torch.from_numpy(a['raw_logits']).to(torch.float64)
            member_wrong = raw.argmax(-1).ne(torch.from_numpy(a['y']))
            native_probability = torch.from_numpy(a['probability_mean']).to(torch.float64)
            native_wrong = native_probability.argmax(-1).ne(torch.from_numpy(a['y']))
            require(a['member_errors'].dtype == a['pooled_errors'].dtype == np.bool_
                    and a['member_errors'].shape == (4, 5274) and a['pooled_errors'].shape == (5274,)
                    and np.array_equal(member_wrong.numpy(), a['member_errors'])
                    and np.array_equal(native_wrong.numpy(), a['pooled_errors']), 'Unchanged archived decisions')
            exported_valid = full[:, torch.from_numpy(a['ids'])].to(torch.float64)
            difference = (exported_valid - raw).abs()
            within = bool(difference.max().item() <= MAX_ABS_LOGIT_ERROR)
            full_probability = full.to(torch.float64).log_softmax(-1).exp()
            flat = full_probability.permute(1, 0, 2).reshape(11701, -1)
            neighbor = torch.sparse.mm(P, flat).reshape(11701, 4, 10).permute(1, 0, 2)
            neighbor = neighbor[:, torch.from_numpy(a['ids'])].contiguous()
            require(torch.isfinite(neighbor).all() and (neighbor >= 0).all()
                    and torch.allclose(neighbor.sum(-1), torch.ones_like(neighbor[..., 0]), atol=1e-12, rtol=1e-12),
                    'Finite normalized label-free neighbor context')
            cache_path = output / (bank['key'] + '_NEIGHBOR_VALID.npz')
            np.savez_compressed(cache_path, ids=a['ids'], neighbor_probability_mean=neighbor.numpy())
            record = {'key': bank['key'], 'backbone': bank['backbone'], 'arm': bank['arm'], 'seed': bank['seed'],
                      'members': 4, 'archive': bank['archive'], 'archive_sha256': bank['archive_sha256'],
                      'cache': str(cache_path), 'cache_sha256': sha(cache_path), 'cache_bytes': cache_path.stat().st_size,
                      'selected_states': [{'path': checkpoint['path'], 'sha256': checkpoint['sha256'],
                                           'selected_step': state['step'], 'forward_seconds': forward_seconds}],
                      'support_train_container_sha256': sha(cfg['train_npz']), 'isolates': isolates,
                      'discrepancies': {'within_fixed_tolerance': within, 'max_abs_logit_tolerance': MAX_ABS_LOGIT_ERROR,
                                        'max_abs_logit_difference': difference.max().item(),
                                        'member_argmax_disagreements': exported_valid.argmax(-1).ne(raw.argmax(-1)).sum().item(),
                                        'acceptance_rule': 'descriptor_max_absolute_logit_error_only',
                                        'argmax_differences_used_for_acceptance': False, 'quality_labels_used': False,
                                        'bitwise_equivalence_required': False, 'replay_or_mismatch_investigation_loop': False},
                      'own_VALID_authority': 'existing_selected_VALID_raw_logits',
                      'persisted_keys': ['ids', 'neighbor_probability_mean'],
                      'full_node_labels_or_TEST_quality_exported': False, 'seconds': time.perf_counter() - bank_start}
            write_json(output / (bank['key'] + '_EXPORT.json'), record)
            exports.append(record)
            require(within, 'Single export exceeds fixed absolute1e-4; retain artifacts and stop without replay')
            arrays[bank['key']] = {'raw': raw, 'neighbor': neighbor, 'native_probability': native_probability,
                                   'native_wrong': native_wrong, 'member_wrong': member_wrong}
            del model, state, full, full_probability, neighbor, flat
            if family.device.type == 'cuda':
                torch.cuda.empty_cache()
            progress(output / 'PROGRESS.json', {'banks_exported': len(exports), 'banks_completed': 0,
                                                'fit_calls': 0, 'current_key': bank['key'],
                                                'elapsed_seconds': time.perf_counter() - started})
        del family, P
    require(len(exports) == len(arrays) == 36, 'All36 exports before any head fit')
    export_seconds = time.perf_counter() - started
    write_json(output / 'COMPLETE_EXPORT.json', {'schema': 'bootstrap-graph-serving-neighbors-v1',
               'complete': True, 'banks': exports, 'support_sha256': sha(support_path),
               'checkpoint_bindings_sha256': sha(checkpoint_path), 'selected_model_forward_calls': 36,
               'native_fullgraph_member_trajectories': 144, 'new_base_fits': 0,
               'seconds': export_seconds, 'TEST_access': False, 'labels_exported': False,
               'fullnode_probabilities_persisted': False})
    ids, y = identity
    labels = torch.from_numpy(y)
    generator = torch.Generator(device='cpu').manual_seed(11709)
    permutation = torch.randperm(len(labels), generator=generator)
    folds = torch.empty(len(labels), dtype=torch.long)
    folds[permutation] = torch.arange(len(labels)) % 5
    fold_sha = hashlib.sha256(folds.numpy().tobytes()).hexdigest()
    require(fold_sha == reuse['fold_assignment_sha256'], 'Exact original label-free folds')
    write_json(output / 'FOLDS.json', {'seed': 11709, 'count': 5, 'ids_sha256': role_hashes['ids'],
               'fold_assignment_sha256': fold_sha, 'counts': [(folds == k).sum().item() for k in range(5)],
               'rule': 'Fixed torch CPU randperm; permutation position modulo5; labels not used.',
               'assessment_scope': 'Encountered VALID; all-label base selection; no whole-pipeline crossfit.'})
    records, failures, total_fits = [], [], 0
    fitting_started = time.perf_counter()
    for bank in expected.values():
        a = arrays[bank['key']]
        log_p = a['raw'].log_softmax(-1)
        probability = log_p.exp()
        uniform_log = torch.logsumexp(log_p, dim=0) - math.log(4)
        graph_f = operators.features(torch, probability, a['neighbor'])
        self_f = operators.features(torch, probability, probability)
        native_wrong = a['native_wrong']
        truth = torch.nn.functional.one_hot(labels, 10).to(torch.float64)
        native_metrics = {'nodes': len(labels), 'correct': (~native_wrong).sum().item(),
                          'accuracy': (~native_wrong).to(torch.float64).mean().item(),
                          'nll': -uniform_log.gather(1, labels[:, None]).mean().item(),
                          'brier': (a['native_probability'] - truth).square().sum(-1).mean().item(),
                          'convention': 'Archived float32 served decisions/probabilities; stable raw-logit float64 NLL.'}
        rule_records, oof = {}, {}
        for rule in RULES:
            rule_start, fit_records = time.perf_counter(), []
            result = torch.full((5274, 10), float('nan'), dtype=torch.float64)
            context = graph_f if rule == 'reliability_graph' else self_f
            for fold in range(5):
                fit_ids, held_ids = torch.where(folds != fold)[0], torch.where(folds == fold)[0]
                total_fits += 1
                fit_start = time.perf_counter()
                try:
                    held_log, fit_record = operators.fit_fold(torch, rule, log_p, probability, context,
                                                              labels, fit_ids, held_ids)
                    result[held_ids] = held_log
                    fit_record.update(fold=fold, status='finite_fixed_endpoint', seconds=time.perf_counter() - fit_start)
                except (FloatingPointError, RuntimeError) as error:
                    fit_record = {'fold': fold, 'status': 'failed_retained', 'error': str(error),
                                  'seconds': time.perf_counter() - fit_start, 'fixed_budget_updates': 500}
                    failures.append({'key': bank['key'], 'rule': rule, **fit_record})
                fit_records.append(fit_record)
            finite = bool(torch.isfinite(result).all())
            oof[rule] = result
            rule_records[rule] = {'status': 'finite_fixed_endpoint' if finite else 'failed_retained',
                                  'fits': fit_records, 'seconds': time.perf_counter() - rule_start}
            if finite:
                value = metrics(torch, result, labels, native_wrong)
                covered, correct = ~a['member_wrong'].all(0), result.argmax(-1).eq(labels)
                value.update(repaired_native_pooling_losses=(covered & native_wrong & correct).sum().item(),
                             remaining_native_pooling_losses=(covered & native_wrong & ~correct).sum().item())
                rule_records[rule]['metrics'] = value
        prediction_path = output / (bank['key'] + '_OOF_VALID.npz')
        np.savez_compressed(prediction_path, ids=ids, folds=folds.numpy(), rule_names=np.array(RULES),
                            log_probability=np.stack([oof[r].numpy() for r in RULES]))
        record = {'key': bank['key'], 'backbone': bank['backbone'], 'arm': bank['arm'], 'seed': bank['seed'],
                  'members': 4, 'native_archived': native_metrics,
                  'uniform_FP64': metrics(torch, uniform_log, labels, native_wrong),
                  'native_member_accuracy': (~a['member_wrong']).to(torch.float64).mean(1).tolist(),
                  'native_covered_nodes': (~a['member_wrong'].all(0)).sum().item(),
                  'native_pooling_loss_nodes': ((~a['member_wrong'].all(0)) & native_wrong).sum().item(),
                  'rules': rule_records, 'VALID_OOF_path': str(prediction_path), 'VALID_OOF_sha256': sha(prediction_path),
                  'archive_sha256': bank['archive_sha256'], 'neighbor_cache_sha256': sha(output / (bank['key'] + '_NEIGHBOR_VALID.npz')),
                  'changed_base_member_predictions': False}
        write_json(output / (bank['key'] + '.json'), record)
        records.append(record)
        progress(output / 'PROGRESS.json', {'banks_exported': 36, 'banks_completed': len(records),
                                            'fit_calls': total_fits, 'current_key': bank['key'],
                                            'elapsed_seconds': time.perf_counter() - started})
    require(len(records) == 36 and total_fits == 900, 'Full declared36-bank900-endpoint roster')
    result = {'schema': 'bootstrap-graph-serving-development-v1', 'complete_roster': True,
              'all_fixed_endpoints_finite': not failures, 'banks': records, 'reused_reference_banks': references,
              'reused_reference_study': reuse, 'reused_original_study_failures_and_costs': {
                  'failures': prior['failures'], 'fit_calls': prior['fit_calls'], 'seconds': prior['seconds']},
              'failures': failures, 'fit_calls': total_fits, 'maximum_small_head_updates': 450000,
              'support_sha256': sha(support_path), 'checkpoint_bindings_sha256': sha(checkpoint_path),
              'export_sha256': sha(output / 'COMPLETE_EXPORT.json'), 'fold_assignment_sha256': fold_sha,
              'ordered_VALID_hashes': role_hashes, 'torch': torch.__version__, 'numpy': np.__version__,
              'device': 'CPU_float64_small_heads', 'seconds': time.perf_counter() - started,
              'head_fitting_seconds': time.perf_counter() - fitting_started, 'export_seconds': export_seconds,
              'new_base_fits': 0, 'new_selected_model_forward_calls': 36, 'new_native_member_trajectories': 144,
              'new_reference_refits': 0, 'TEST_access': False, 'changed_base_member_predictions': False,
              'no_final_refit_or_held_fold_checkpoint_selection': True,
              'member_protection_cannot_be_repaired_by_static_serving': True,
              'interpretation': 'Complete fixed serving interaction on encountered development. Failed member protection remains. No novelty, unused confirmation or automatic scientific promotion.'}
    write_json(output / 'COMPLETE_STUDY.json', result)
    return {'complete_roster': True, 'all_fixed_endpoints_finite': not failures, 'banks': 36,
            'reused_reference_banks': 30, 'fit_calls': 900, 'selected_model_forward_calls': 36,
            'native_member_trajectories': 144, 'new_base_fits': 0, 'TEST_access': False}
