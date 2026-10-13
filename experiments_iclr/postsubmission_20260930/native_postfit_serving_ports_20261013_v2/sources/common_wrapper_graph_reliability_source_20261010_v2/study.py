"""Root-called encountered-VALID fusion study; no backbone, launcher or owner."""
import hashlib
import math
from pathlib import Path
import time

from operators import features, fit_fold
from support import (ARCHIVE_KEYS, ARMS, BACKBONES, PHASE, RULES, SEEDS, load_contract,
                     new_output, require, root_runtime, sha, write_json, write_progress)


def metrics(torch, log_q, labels, native_wrong):
    probability = log_q.exp()
    correct = log_q.argmax(-1).eq(labels)
    truth = torch.nn.functional.one_hot(labels, probability.shape[-1]).to(probability)
    return {'nodes': len(labels), 'correct': correct.sum().item(),
            'accuracy': correct.to(torch.float64).mean().item(),
            'nll': -log_q.gather(1, labels[:, None]).mean().item(),
            'brier': (probability - truth).square().sum(-1).mean().item(),
            'repairs_native': (native_wrong & correct).sum().item(),
            'harms_native': (~native_wrong & ~correct).sum().item()}


def run(support_path, export_path, output_path):
    """This function call is not admitted by source preparation alone."""
    root_runtime()
    contract = load_contract(support_path)
    export_path = Path(export_path).resolve()
    require(export_path.is_relative_to(PHASE), 'In-scope root-owned export required')
    import json
    export = json.loads((export_path / 'COMPLETE_EXPORT.json').read_text())
    require(export['schema'] == 'common-wrapper-reliability-neighbors-v1'
            and export['complete'] is True and len(export['banks']) == 45
            and export['TEST_access'] is False and export['labels_exported'] is False
            and export['support_sha256'] == sha(support_path), 'Complete bound45-bank neighbor export')
    expected = {b['key']: b for row in contract['families'] for b in row['banks']}
    require(set(expected) == {b['key'] for b in export['banks']} and len(expected) == 45,
            'Exact complete bank identity required')
    for row in export['banks']:
        bank = expected[row['key']]
        require(row['archive'] == bank['archive'] and row['archive_sha256'] == bank['archive_sha256']
                and sha(row['archive']) == row['archive_sha256']
                and sha(row['cache']) == row['cache_sha256']
                and row['discrepancies']['within_fixed_tolerance'] is True,
                'Authoritative archive/cache changed or single export unqualified')
    output = new_output(output_path)
    write_json(output / 'SUPPORT.json', contract)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    started, arrays, identity = time.perf_counter(), {}, None
    # Validate the whole45-bank input roster before fitting the first head.
    for row in export['banks']:
        bank = expected[row['key']]
        with np.load(row['archive'], allow_pickle=False) as stored:
            require(set(stored.files) == ARCHIVE_KEYS, 'Original VALID schema required')
            a = {k: stored[k].copy() for k in ARCHIVE_KEYS}
        with np.load(row['cache'], allow_pickle=False) as cache:
            require(set(cache.files) == {'ids', 'neighbor_probability_mean'}, 'Label-free VALID-only cache schema')
            cache_ids, neighbor = cache['ids'].copy(), cache['neighbor_probability_mean'].copy()
        require(a['ids'].dtype == a['y'].dtype == np.int64 and a['ids'].shape == a['y'].shape == (5274,)
                and np.array_equal(cache_ids, a['ids'])
                and a['raw_logits'].shape == (bank['members'], 5274, 10)
                and a['raw_logits'].dtype == np.float32
                and a['probability_mean'].shape == (5274, 10) and a['probability_mean'].dtype == np.float32
                and neighbor.shape == a['raw_logits'].shape and neighbor.dtype == np.float64,
                'Exact VALID role/member/cache layout')
        require(len(np.unique(a['ids'])) == 5274 and a['ids'].min() >= 0 and a['ids'].max() < 11701
                and set(a['y'].tolist()) == set(range(10))
                and np.isfinite(a['raw_logits']).all() and np.isfinite(a['probability_mean']).all()
                and np.isfinite(neighbor).all() and (neighbor >= 0).all()
                and np.allclose(neighbor.sum(-1), 1, atol=1e-12, rtol=1e-12), 'Finite exact ten-class VALID operands')
        if identity is None:
            identity = (a['ids'], a['y'])
        else:
            require(np.array_equal(identity[0], a['ids']) and np.array_equal(identity[1], a['y']),
                    'Every bank receives identical ordered VALID labels')
        raw = torch.from_numpy(a['raw_logits']).to(torch.float64)
        labels = torch.from_numpy(a['y'])
        member_wrong = raw.argmax(-1).ne(labels)
        native_probability = torch.from_numpy(a['probability_mean']).to(torch.float64)
        native_wrong = native_probability.argmax(-1).ne(labels)
        require(a['member_errors'].dtype == a['pooled_errors'].dtype == np.bool_
                and a['member_errors'].shape == (bank['members'], 5274)
                and a['pooled_errors'].shape == (5274,)
                and np.array_equal(member_wrong.numpy(), a['member_errors'])
                and np.array_equal(native_wrong.numpy(), a['pooled_errors']), 'Unchanged authoritative native decisions')
        arrays[row['key']] = {'raw': raw, 'neighbor': torch.from_numpy(neighbor),
                             'native_probability': native_probability, 'native_wrong': native_wrong,
                             'member_wrong': member_wrong}
    ids, y = identity
    role_hashes = {name: hashlib.sha256(value.tobytes()).hexdigest()
                   for name, value in (('ids', ids), ('labels', y))}
    require(role_hashes == contract['ordered_VALID_hashes'], 'Exact admitted VALID role fingerprints')
    labels = torch.from_numpy(y)
    generator = torch.Generator(device='cpu').manual_seed(11709)
    permutation = torch.randperm(len(labels), generator=generator)
    folds = torch.empty(len(labels), dtype=torch.long)
    folds[permutation] = torch.arange(len(labels)) % 5
    fold_sha = hashlib.sha256(folds.numpy().tobytes()).hexdigest()
    write_json(output / 'FOLDS.json', {'seed': 11709, 'count': 5, 'ids_sha256': role_hashes['ids'],
                                     'fold_assignment_sha256': fold_sha,
                                     'rule': 'Fixed torch CPU randperm; permutation position modulo5; labels not used.',
                                     'counts': [(folds == k).sum().item() for k in range(5)],
                                     'assessment_scope': 'Encountered VALID; base selectors used all labels; no whole-pipeline crossfit.'})
    records, failures, total_fits = [], [], 0
    write_progress(output / 'PROGRESS.json', {'banks_completed': 0, 'fit_calls': 0,
                                             'current_key': None, 'elapsed_seconds': time.perf_counter() - started})
    export_index = {r['key']: r for r in export['banks']}
    for backbone in BACKBONES:
        for seed in SEEDS:
            for arm in ARMS:
                key = f'{backbone}_{arm}_seed{seed}'
                bank, a = expected[key], arrays[key]
                log_p = a['raw'].log_softmax(-1)
                probability = log_p.exp()
                uniform_log = torch.logsumexp(log_p, dim=0) - math.log(bank['members'])
                graph_f = features(torch, probability, a['neighbor'])
                self_f = features(torch, probability, probability)
                native_wrong = a['native_wrong']
                truth = torch.nn.functional.one_hot(labels, 10).to(torch.float64)
                native = {'nodes': len(labels), 'correct': (~native_wrong).sum().item(),
                          'accuracy': (~native_wrong).to(torch.float64).mean().item(),
                          'nll': -uniform_log.gather(1, labels[:, None]).mean().item(),
                          'brier': (a['native_probability'] - truth).square().sum(-1).mean().item(),
                          'convention': 'Archived float32 served decisions/probabilities; stable float64 raw-logit NLL follows native Family.metrics.'}
                rule_records, oof = {}, {}
                for rule in RULES:
                    rule_start, fit_records = time.perf_counter(), []
                    if bank['members'] == 1 and rule in ('reliability_self', 'reliability_graph'):
                        oof[rule] = uniform_log.clone()
                        rule_records[rule] = {'status': 'structural_M1_no_weighting_effect_no_fit',
                                              'metrics': metrics(torch, oof[rule], labels, native_wrong),
                                              'fits': [], 'seconds': 0.0}
                        continue
                    if bank['members'] == 1 and rule == 'temperature_member':
                        oof[rule] = oof['temperature_global'].clone()
                        rule_records[rule] = {'status': 'prospective_duplicate_M1_temperature_global',
                                              'metrics': rule_records['temperature_global'].get('metrics'),
                                              'duplicate_of': 'temperature_global', 'fits': [], 'seconds': 0.0}
                        continue
                    result = torch.full((5274, 10), float('nan'), dtype=torch.float64)
                    context = graph_f if rule == 'reliability_graph' else self_f
                    for fold in range(5):
                        fit_ids, held_ids = torch.where(folds != fold)[0], torch.where(folds == fold)[0]
                        total_fits += 1
                        fit_start = time.perf_counter()
                        try:
                            held_log, fit_record = fit_fold(torch, rule, log_p, probability, context,
                                                           labels, fit_ids, held_ids)
                            result[held_ids] = held_log
                            fit_record.update(fold=fold, status='finite_fixed_endpoint', seconds=time.perf_counter() - fit_start)
                        except (FloatingPointError, RuntimeError) as error:
                            fit_record = {'fold': fold, 'status': 'failed_retained', 'error': str(error),
                                          'seconds': time.perf_counter() - fit_start, 'fixed_budget_updates': 500}
                            failures.append({'key': key, 'rule': rule, **fit_record})
                        fit_records.append(fit_record)
                    finite = bool(torch.isfinite(result).all())
                    oof[rule] = result
                    rule_records[rule] = {'status': 'finite_fixed_endpoint' if finite else 'failed_retained',
                                          'fits': fit_records, 'seconds': time.perf_counter() - rule_start}
                    if finite:
                        value = metrics(torch, result, labels, native_wrong)
                        covered = ~a['member_wrong'].all(0)
                        correct = result.argmax(-1).eq(labels)
                        value.update(repaired_native_pooling_losses=(covered & native_wrong & correct).sum().item(),
                                     remaining_native_pooling_losses=(covered & native_wrong & ~correct).sum().item())
                        rule_records[rule]['metrics'] = value
                prediction_path = output / (key + '_OOF_VALID.npz')
                np.savez_compressed(prediction_path, ids=ids, folds=folds.numpy(),
                                    rule_names=np.array(RULES), log_probability=np.stack([oof[r].numpy() for r in RULES]))
                record = {'key': key, 'backbone': backbone, 'arm': arm, 'seed': seed, 'members': bank['members'],
                          'native_archived': native, 'uniform_FP64': metrics(torch, uniform_log, labels, native_wrong),
                          'native_member_accuracy': (~a['member_wrong']).to(torch.float64).mean(1).tolist(),
                          'native_covered_nodes': (~a['member_wrong'].all(0)).sum().item(),
                          'native_pooling_loss_nodes': ((~a['member_wrong'].all(0)) & native_wrong).sum().item(),
                          'rules': rule_records, 'VALID_OOF_path': str(prediction_path), 'VALID_OOF_sha256': sha(prediction_path),
                          'archive_sha256': bank['archive_sha256'], 'neighbor_cache_sha256': export_index[key]['cache_sha256'],
                          'changed_base_member_predictions': False}
                write_json(output / (key + '.json'), record)
                records.append(record)
                write_progress(output / 'PROGRESS.json', {'banks_completed': len(records),
                                                         'fit_calls': total_fits, 'current_key': key,
                                                         'elapsed_seconds': time.perf_counter() - started})
    require(len(records) == 45 and total_fits == 855, 'Complete equal-control45-bank roster')
    result = {'schema': 'common-wrapper-reliability-development-v1', 'complete_roster': True,
              'all_fixed_endpoints_finite': not failures, 'banks': records, 'failures': failures,
              'fit_calls': total_fits, 'maximum_small_head_updates': total_fits * 500,
              'support_sha256': sha(support_path), 'export_sha256': sha(export_path / 'COMPLETE_EXPORT.json'),
              'torch': torch.__version__, 'numpy': np.__version__, 'device': 'CPU_float64_small_heads',
              'fold_assignment_sha256': fold_sha, 'ordered_VALID_hashes': role_hashes,
              'seconds': time.perf_counter() - started, 'new_base_fits_or_forwards': 0, 'TEST_access': False,
              'no_final_refit_or_held_fold_checkpoint_selection': True,
              'interpretation': 'All selected banks and fixed controls retained. Aggregator-only OOF is encountered development; no novelty, generalization or confirmation claim and no automatic scientific GO.'}
    write_json(output / 'COMPLETE_STUDY.json', result)
    return {'complete_roster': True, 'all_fixed_endpoints_finite': not failures, 'banks': 45,
            'fit_calls': total_fits, 'TEST_access': False}
