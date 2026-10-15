"""One fixed-readout CPU pass over all completed33 saved VALID banks.

Source only until root review. No fits, checkpoint loads, forwards or selection.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import socket
import statistics
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
E = HERE.parents[1]
SEEDS = [101, 202, 303]
RESIDUAL = 'residual_only_private_BN'
READOUTS = ['saved_mean_logit', 'arithmetic_probability']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def log_sigmoid(z):
    return -max(0., -z) - math.log1p(math.exp(-abs(z)))


def log_mean_exp(values):
    largest = max(values)
    return largest + math.log(math.fsum(math.exp(v-largest) for v in values)) - math.log(len(values))


def metrics(labels, scores, log_p, log_q):
    """Uniform per-graph losses; binary AUC gives exactly equal scores half credit."""
    require(len(labels) == len(scores) == len(log_p) == len(log_q) == 151 and
            all(math.isfinite(v) for seq in (scores, log_p, log_q) for v in seq),
            'Complete finite stable VALID scores required')
    losses = [-lp if y else -lq for y, lp, lq in zip(labels, log_p, log_q)]
    # exp(log wrong-class probability) is stable even near a saturated endpoint.
    brier = [math.exp(2*(lq if y else lp)) for y, lp, lq in zip(labels, log_p, log_q)]
    positive = [i for i, y in enumerate(labels) if y == 1]
    negative = [i for i, y in enumerate(labels) if y == 0]
    require((len(positive), len(negative)) == (130, 21), 'Frozen VALID class counts differ')
    wins = sum(scores[p] > scores[n] for p in positive for n in negative)
    ties = sum(scores[p] == scores[n] for p in positive for n in negative)

    def subset(indices):
        return {'graphs': len(indices), 'BCE': math.fsum(losses[i] for i in indices)/len(indices),
                'Brier': math.fsum(brier[i] for i in indices)/len(indices)}

    return {**subset(list(range(151))), 'AUC': (wins+.5*ties)/2730,
            'positive_negative_score_ties': ties,
            'by_class': {'0': subset(negative), '1': subset(positive)},
            'false_negatives': sum(log_p[i] < log_q[i] for i in positive),
            'false_positives': sum(log_p[i] >= log_q[i] for i in negative)}


def metric_delta(candidate, reference):
    return {'AUC_pp': 100*(candidate['AUC']-reference['AUC']),
            'BCE': candidate['BCE']-reference['BCE'],
            'Brier': candidate['Brier']-reference['Brier'],
            'class0_BCE': candidate['by_class']['0']['BCE']-reference['by_class']['0']['BCE'],
            'class1_BCE': candidate['by_class']['1']['BCE']-reference['by_class']['1']['BCE']}


def mean_metrics(rows):
    return {key: statistics.mean(values) for key, values in {
        'AUC': [r['AUC'] for r in rows], 'BCE': [r['BCE'] for r in rows],
        'Brier': [r['Brier'] for r in rows],
        'class0_BCE': [r['by_class']['0']['BCE'] for r in rows],
        'class1_BCE': [r['by_class']['1']['BCE'] for r in rows]}.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    require(socket.gethostname() == 'anogena-2-0' and subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines()
        == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac'], 'Dedicated allocation required before project metadata reads')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU diagnostic requires CUDA hidden')
    started = time.perf_counter()
    plan = json.loads((HERE/'PLAN.json').read_text())
    root = Path(plan['project_root'])
    require(HERE == root/'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/research/residual_positive_bce_pooling_v1'
            and Path(sys.prefix).resolve() == (root/'.venv').resolve(), 'Exact reviewed source and original interpreter required')
    require(pin(HERE/'MANIFEST.json')['sha256'] == args.manifest_sha256, 'Reviewed manifest differs')
    for item in json.loads((HERE/'MANIFEST.json').read_text())['files']:
        path = (HERE/item['path']).resolve()
        require(path.is_relative_to(HERE) and pin(path)['sha256'] == item['sha256'] and
                pin(path)['bytes'] == item['bytes'], 'Own source changed')
    output = Path(args.output_dir).resolve()
    require(output.parent == (E/'analysis_outputs').resolve() and not output.exists(),
            'Fresh direct analysis_outputs child required; preserve every attempt')
    loader_path = Path(plan['verified_loader']['path'])
    require(pin(loader_path) == plan['verified_loader'], 'Completed diagnostic verification helper differs')
    definition = importlib.util.spec_from_file_location('_completed_saved_bank_custody', loader_path)
    helper = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(helper)
    verified = helper.verified
    output.mkdir(parents=True, exist_ok=False)
    try:
        original = json.loads(verified(plan['completed33_authority']).read_text())
        diagnostic = json.loads(verified(plan['completed_probability_diagnostic']).read_text())
        require(original['schema'] == 'residual-only-selected-VALID-33-bank-addition-v1' and original['status'] == 'complete'
                and original['TEST_accessed'] is False and len(original['rows']) == 33
                and len(original['training_fit_outcomes']) == 39, 'Completed33 authority differs')
        require(diagnostic['schema'] == 'completed-probability-failure-descriptive-v1' and diagnostic['status'] == 'complete'
                and diagnostic['TEST_accessed'] is False and diagnostic['completed33_authority'] == plan['completed33_authority']
                and diagnostic['verified_loader'] == plan['verified_loader'], 'Completed verified diagnostic differs')
        require({(r['seed'], r['bank']) for r in original['rows']} ==
                {(seed, bank) for seed in SEEDS for bank in plan['banks']}, 'All11 banks x3 seeds required')
        require(diagnostic['source_fit_receipts'] == [r['receipt'] for r in original['training_fit_outcomes']],
                'Own fit receipt ledger differs')
        fits = {}
        for outcome in original['training_fit_outcomes']:
            fit = json.loads(verified(outcome['receipt']).read_text())
            require(fit['status'] == 'complete' and fit['job_id'] == outcome['job_id'] and fit['TEST_accessed'] is False
                    and not fit.get('failure') and not fit.get('close_error'), 'Completed fit metadata differs')
            fits[outcome['job_id']] = fit
        records = diagnostic['saved_collection_receipts']
        require(len(fits) == 39 and len(records) == 27 and
                [r['collection'] for r in records] == [r['receipt'] for r in original['route_collection_outcomes']],
                'Exact39 exports/27 collection ledger required')
        collections = {}
        for record in records:
            collection = json.loads(verified(record['collection']).read_text())
            fit = fits[record['job_id']]
            require(collection['status'] in ('complete', 'complete_with_sanity_warning') and collection['role'] == 'VALID'
                    and collection['TEST_accessed'] is False and collection['job_id'] == record['job_id']
                    and collection['job_hash'] == fit['job_hash'] and collection['selected_epoch'] == fit['selected_epoch']
                    and collection['selected_checkpoint'] == fit['selected_checkpoint']
                    and collection['saved_VALID_export'] == fit['selected_VALID_export']
                    and collection['raw_tensor_receipt'] == record['raw']
                    and collection['actual_work']['TRAIN_callbacks'] == collection['actual_work']['optimizer_steps'] == 0,
                    'Completed collection custody differs')
            collections[record['job_id']] = collection
        import torch
        torch.set_num_threads(2)
        exports = {}
        reference_ids = reference_labels = None
        for ident, fit in fits.items():
            saved = torch.load(verified(fit['selected_VALID_export']), map_location='cpu', weights_only=False)
            require(saved['schema'] == 'own-selected-VALID-logits-v1' and saved['role'] == 'VALID'
                    and saved['TEST_accessed'] is False and saved['job_id'] == ident and saved['job_hash'] == fit['job_hash']
                    and saved['selected_epoch'] == fit['selected_epoch'] and saved['selected_checkpoint'] == fit['selected_checkpoint']
                    and saved['validation'] == fit['selected_VALID'], 'Selected export custody differs')
            ids, y, z = saved['ids'], saved['labels'], saved['raw_logits']
            require(tuple(z.shape) == tuple(y.shape) == (151, 1) and ids.shape == (151,)
                    and torch.unique(ids).numel() == 151 and z.dtype == torch.float32
                    and torch.isfinite(z).all().item() and torch.isfinite(y).all().item()
                    and int((y == 1).sum()) == 130 and int((y == 0).sum()) == 21, 'Frozen finite VALID151 required')
            if reference_ids is None:
                reference_ids, reference_labels = ids, y
            require(torch.equal(ids, reference_ids) and torch.equal(y, reference_labels), 'Exact VALID order/labels differ')
            exports[ident] = saved
        routes = {}
        for ident, collection in collections.items():
            raw = torch.load(verified(collection['raw_tensor_receipt']), map_location='cpu', weights_only=False)
            saved = exports[ident]
            require(raw['role'] == 'VALID' and raw['TEST_accessed'] is False and raw['job_id'] == ident
                    and raw['job_hash'] == saved['job_hash'] and raw['selected_epoch'] == saved['selected_epoch']
                    and raw['selected_checkpoint'] == saved['selected_checkpoint']
                    and torch.equal(raw['ids'], reference_ids) and torch.equal(raw['labels'], reference_labels)
                    and tuple(raw['member_raw_logits'].shape) == (4, 151, 1)
                    and torch.isfinite(raw['member_raw_logits']).all().item(), 'Saved native routes differ')
            routes[ident] = raw['member_raw_logits']
        labels = [int(v) for v in reference_labels[:, 0].tolist()]
        previous = {(r['seed'], r['bank']): r for r in diagnostic['rows']}
        require(len(previous) == 33, 'Complete prior diagnostic rows required')
        rows = []
        for authority in original['rows']:
            seed, bank, owners = authority['seed'], authority['bank'], authority['source_job_ids']
            if bank == 'I4':
                require(len(owners) == 4 and [exports[i]['body'] for i in owners] == [0, 1, 2, 3]
                        and len({exports[i]['seed'] for i in owners}) == 4, 'Four independent own-selected bodies required')
                members = torch.stack([exports[i]['raw_logits'] for i in owners])
                pool = members.mean(0)  # exact original stored-FP32 I4 authority
            else:
                require(len(owners) == 1, 'One fitted bank authority required')
                pool = exports[owners[0]]['raw_logits']
                members = pool.unsqueeze(0) if bank == 'M1' else routes[owners[0]]
            native = pool[:, 0].double().tolist()
            member_lists = [m[:, 0].double().tolist() for m in members]
            lp = [log_sigmoid(z) for z in native]
            lq = [log_sigmoid(-z) for z in native]
            baseline = metrics(labels, native, lp, lq)
            require(abs(baseline['AUC']-authority['pool_AUC']) < 1e-12 and
                    abs(baseline['BCE']-authority['uniform_graph_BCE']) < 1e-12 and
                    abs(baseline['Brier']-authority['pooled_metrics']['Brier']) < 1e-12,
                    'Saved original AUC/BCE/Brier authority differs')
            for cls in ('0', '1'):
                require(abs(baseline['by_class'][cls]['BCE']-previous[(seed, bank)]['pool']['by_class'][cls]['mean_BCE']) < 1e-12,
                        'Prior completed class BCE differs')
            mix_lp = [log_mean_exp([log_sigmoid(m[i]) for m in member_lists]) for i in range(151)]
            mix_lq = [log_mean_exp([log_sigmoid(-m[i]) for m in member_lists]) for i in range(151)]
            alternative = metrics(labels, [a-b for a, b in zip(mix_lp, mix_lq)], mix_lp, mix_lq)
            require(max(abs(math.exp(a)+math.exp(b)-1) for a, b in zip(mix_lp, mix_lq)) < 1e-12,
                    'Stable mixture probabilities must sum to one')
            if bank == 'M1':
                require(all(abs(v) < 1e-12 for v in metric_delta(alternative, baseline).values()), 'Singleton fixed readouts differ')
            rows.append({'seed': seed, 'bank': bank, 'source_job_ids': owners, 'selected_epochs': authority['selected_epochs'],
                         'readouts': dict(zip(READOUTS, (baseline, alternative))),
                         'probability_minus_logit': metric_delta(alternative, baseline),
                         'reconstructed_member_mean_minus_saved_pool_max_abs': float((members.mean(0)-pool).abs().max()),
                         'retained_collection_sanity': None if bank in ('M1', 'I4') else collections[owners[0]]['sanity_against_saved_export']})
        lookup = {(r['seed'], r['bank']): r for r in rows}
        pairs = []
        for seed in SEEDS:
            item = {'seed': seed, 'residual_minus_I4': {name: metric_delta(
                lookup[(seed, RESIDUAL)]['readouts'][name], lookup[(seed, 'I4')]['readouts'][name]) for name in READOUTS}}
            a = item['residual_minus_I4']['saved_mean_logit']['class1_BCE']
            b = item['residual_minus_I4']['arithmetic_probability']['class1_BCE']
            item['fraction_positive_class_excess_remaining'] = b/a if a > 0 else None
            pairs.append(item)
        summaries = [{'bank': bank, 'seeds': SEEDS, 'mean_readouts': {name: mean_metrics(
            [lookup[(seed, bank)]['readouts'][name] for seed in SEEDS]) for name in READOUTS}}
            for bank in plan['banks']]
        pair_means = {name: {key: statistics.mean(p['residual_minus_I4'][name][key] for p in pairs)
                            for key in pairs[0]['residual_minus_I4'][name]} for name in READOUTS}
        a, b = [pair_means[name]['class1_BCE'] for name in READOUTS]
        result = {'schema': 'completed33-fixed-pool-class-comparison-v1', 'status': 'complete', 'rows': rows,
                  'three_seed_means': summaries, 'paired_residual_I4': pairs, 'mean_residual_minus_I4': pair_means,
                  'ratio_of_mean_positive_class_excess_remaining': b/a if a > 0 else None,
                  'source': pin(__file__), 'manifest': pin(HERE/'MANIFEST.json'), 'PLAN': pin(HERE/'PLAN.json'),
                  'completed33_authority': plan['completed33_authority'],
                  'completed_probability_diagnostic': plan['completed_probability_diagnostic'],
                  'readout_rule': 'Saved selected pool is authoritative; I4 is exact FP32 mean of four own-selected exports. Alternative is uniform arithmetic mean sigmoid across the same native members. All33 get both rules; none is selected.',
                  'BCE_rule': 'FP64 scalar log-sigmoid then logsumexp for mean p and mean (1-p); no sigmoid clipping or fitted correction.',
                  'AUC_rule': 'Baseline raw logits; mixture stable log odds log(mean p)-log(mean (1-p)). Exact numeric ties get half credit over all130x21 pairs. Class-only AUC is undefined and omitted.',
                  'threshold_rule': 'Positive iff log p >= log(1-p); equality is positive; no threshold tuning.',
                  'custody': 'Scalar summaries only. No graph IDs, per-example arrays or predictions are written.',
                  'scope': 'Descriptive exposed checkpoint-selection VALID comparison; no inference of causal training mechanism, generalization, or serving preference.',
                  'scientific_fits': 0, 'native_forwards': 0, 'optimizer_steps': 0, 'checkpoint_payloads_loaded': 0,
                  'selector_changes': 0, 'TEST_accessed': False, 'best_readout_selected': False,
                  'threads': 2, 'saved_exports_loaded': 39, 'saved_route_collections_loaded': 27,
                  'wall_seconds': time.perf_counter()-started,
                  'own_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        lines = ['# Current33 fixed pooling and class BCE', '',
                 'Both fixed readouts are shown for all11 banks x3 seeds; original scores and selection stay authoritative.', '',
                 '## Residual minus I4', '',
                 '| Seed | Positive BCE excess: logit | Positive BCE excess: probability | Fraction remaining | Residual AUC change (pp) | I4 AUC change (pp) |',
                 '| --- | ---: | ---: | ---: | ---: | ---: |']
        for pair in pairs:
            seed = pair['seed']; d = pair['residual_minus_I4']; fraction = pair['fraction_positive_class_excess_remaining']
            lines.append(f"| {seed} | {d['saved_mean_logit']['class1_BCE']:.6f} | {d['arithmetic_probability']['class1_BCE']:.6f} | {fraction:.2%} | {lookup[(seed, RESIDUAL)]['probability_minus_logit']['AUC_pp']:.4f} | {lookup[(seed, 'I4')]['probability_minus_logit']['AUC_pp']:.4f} |")
        lines += ['', f"Mean positive-class excess changes from {a:.6f} to {b:.6f}; {b/a:.2%} remains (ratio of seed-mean signed excesses).", '',
                  '## All banks, three-seed arithmetic means', '',
                  '| Bank | AUC %: logit / probability | BCE: logit / probability | Positive BCE: logit / probability |',
                  '| --- | ---: | ---: | ---: |']
        for item in summaries:
            x, y = [item['mean_readouts'][name] for name in READOUTS]
            lines.append(f"| {item['bank']} | {100*x['AUC']:.4f} / {100*y['AUC']:.4f} | {x['BCE']:.6f} / {y['BCE']:.6f} | {x['class1_BCE']:.6f} / {y['class1_BCE']:.6f} |")
        lines += ['', 'BCE uses stable log probabilities without clipping. Mixture AUC uses stable log odds, with exact numerical ties worth half credit; tie counts are in RESULT.json. Class-only AUC is undefined. RESULT.json also gives both class losses, Brier, fixed-threshold errors, every seed and signed paired differences.', '',
                  'Native saved pools are used even when collected member means have small reconstruction differences; those differences and retained collection sanity are disclosed. No alternative is promoted from VALID. No fit, calibration, prediction edit, model/checkpoint load, forward, update or TEST access occurred.', '']
        (output/'REPORT.md').write_text('\n'.join(lines))
        (output/'RESULT.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
        print(json.dumps({'status': 'complete', 'banks': 33, 'positive_excess_saved_logit': a,
                          'positive_excess_probability': b, 'fraction_remaining': b/a,
                          'wall_seconds': result['wall_seconds']}))
        return 0
    except BaseException as error:
        (output/'GATE.json').write_text(json.dumps({'status': 'failed_no_partial_diagnostics',
            'error': {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(limit=8)},
            'source': pin(__file__), 'TEST_accessed': False, 'scientific_fits': 0, 'native_forwards': 0,
            'optimizer_steps': 0, 'wall_seconds': time.perf_counter()-started}, indent=2, sort_keys=True)+'\n')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
