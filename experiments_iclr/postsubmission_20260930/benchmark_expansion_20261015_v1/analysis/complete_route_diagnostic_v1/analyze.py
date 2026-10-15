"""Complete selected-VALID route diagnostics from existing tensors; no models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import statistics
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, inputs_path, output):
    started = time.monotonic()
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output([
        'nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [
            'GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    root, inputs_path, output = [Path(p).resolve() for p in (root, inputs_path, output)]
    assert inputs_path.is_relative_to(root) and output.is_relative_to(root)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    output.mkdir(parents=True, exist_ok=False)
    inputs = json.loads(inputs_path.read_text())

    def verified(receipt):
        p = Path(receipt['path']).resolve()
        assert p.is_relative_to(root) and sha(p) == receipt['sha256'], str(p)
        return p

    verified(inputs['plan'])
    verified(inputs['completed_analysis'])
    import torch
    from ogb.graphproppred import Evaluator
    torch.set_num_threads(2)
    evaluator = Evaluator(name='ogbg-molbace')
    assert len(inputs['fits']) == 24 and len(inputs['cohorts']) == 3
    loaded = {}
    for row in inputs['fits']:
        fit = json.loads(verified(row['fit']).read_text())
        analysis = json.loads(verified(row['analysis']).read_text())
        routes = torch.load(verified(row['routes']), map_location='cpu', weights_only=False)
        export = torch.load(verified(row['export']), map_location='cpu', weights_only=False)
        assert fit['status'] == 'complete' and fit['job_id'] == row['job_id']
        assert fit['selected_checkpoint']['sha256'] == row['checkpoint']['sha256']
        assert fit['selected_VALID_export']['sha256'] == row['export']['sha256']
        assert analysis['job_id'] == row['job_id'] and analysis['raw_tensor_receipt'] == row['routes']
        assert routes['role'] == export['role'] == 'VALID'
        assert routes['TEST_accessed'] is False and export['TEST_accessed'] is False
        assert routes['job_hash'] == export['job_hash'] == fit['job_hash']
        assert torch.equal(routes['ids'], export['ids']) and torch.equal(routes['labels'], export['labels'])
        assert torch.isfinite(routes['member_raw_logits']).all() and torch.isfinite(export['raw_logits']).all()
        loaded[row['job_id']] = (row, fit, analysis, routes, export)

    def metric(logits, labels):
        return float(evaluator.eval({'y_true': labels.numpy(), 'y_pred': logits.numpy()})['rocauc'])

    def describe(z, pool, truth):
        y = truth[:, 0]
        pos, neg = y == 1, y == 0
        assert int(pos.sum()) == 130 and int(neg.sum()) == 21
        margin = z[:, pos, 0].double()[:, :, None] - z[:, neg, 0].double()[:, None, :]
        pm = pool[pos, 0].double()[:, None] - pool[neg, 0].double()[None, :]
        correct, wrong, tied = margin > 0, margin < 0, margin == 0
        pc, pw, pt = pm > 0, pm < 0, pm == 0
        auc = metric(pool, truth)
        assert abs(float((pc.double() + .5 * pt.double()).mean()) - auc) < 1e-12
        member_auc = [metric(v, truth) for v in z]
        probabilities = z.double().sigmoid().mean(0)
        probability_auc = metric(probabilities, truth)
        # Exact fixed-probability mixture BCE with stable logs, no clipping or fit.
        lp = torch.logsumexp(torch.nn.functional.logsigmoid(z.double()), 0) - torch.log(torch.tensor(len(z), dtype=torch.double))
        ln = torch.logsumexp(torch.nn.functional.logsigmoid(-z.double()), 0) - torch.log(torch.tensor(len(z), dtype=torch.double))
        bce_probability = float(-(truth.double() * lp + (1 - truth.double()) * ln).mean())
        bce_logit = float(torch.nn.functional.binary_cross_entropy_with_logits(pool.double(), truth.double()))
        r = {
            'member_AUC': member_auc, 'mean_member_AUC': statistics.mean(member_auc),
            'pool_AUC': auc, 'pool_minus_mean_member_AUC_pp': 100 * (auc - statistics.mean(member_auc)),
            'pool_correct': int(pc.sum()), 'pool_tied': int(pt.sum()), 'pool_reversed': int(pw.sum()),
            'all_members_reversed': int(wrong.all(0).sum()),
            'all_members_correct': int(correct.all(0).sum()),
            'member_sign_disagreement_pairs': int((correct.any(0) & ~correct.all(0)).sum()),
            'pool_reversed_despite_some_member_correct': int((pw & correct.any(0)).sum()),
            'pool_reversed_despite_strict_majority_correct': int((pw & (correct.sum(0) > len(z) / 2)).sum()),
            'pool_strict_repairs_of_all_member_reversals': int((pc & wrong.all(0)).sum()),
            'member_tied_pairs': tied.sum((1, 2)).tolist(),
            'member_mean_absolute_margin': margin.abs().mean((1, 2)).tolist(),
            'fixed_probability_AUC': probability_auc,
            'fixed_probability_minus_logit_AUC_pp': 100 * (probability_auc - auc),
            'mean_logit_BCE': bce_logit, 'mean_probability_BCE': bce_probability,
        }
        return r, pm

    rows, comparisons, warnings = [], [], []
    for cohort in inputs['cohorts']:
        native = [loaded[k] for k in cohort['native_fit_ids']]
        methods = [loaded[k] for k in cohort['method_fit_ids']]
        ids, truth = native[0][4]['ids'], native[0][4]['labels']
        for value in native + methods:
            assert torch.equal(value[4]['ids'], ids) and torch.equal(value[4]['labels'], truth)
        seed = native[0][0]['seed']
        native_logits = torch.stack([v[4]['raw_logits'] for v in native])
        native_pool = native_logits.mean(0)
        banks = [('I4', native_logits, native_pool, native), *[
            (v[0]['procedure'], v[3]['member_raw_logits'], v[3]['pooled_raw_logits'], [v]) for v in methods]]
        margins = {}
        for name, z, pool, owners in banks:
            entry, margin = describe(z, pool, truth)
            entry.update(seed=seed, method=name, selected_epochs=[v[0]['selected_epoch'] for v in owners],
                         source_job_ids=[v[0]['job_id'] for v in owners])
            if name != 'I4':
                a = owners[0][2]
                entry['original_saved_AUC'] = owners[0][1]['best_validation_metric']
                entry['reconstruction_AUC_delta_pp'] = 100 * (entry['pool_AUC'] - entry['original_saved_AUC'])
                entry['reconstruction_sanity'] = a['sanity_against_saved_export']
                entry['mean_route_CKA'] = statistics.mean(v['centered_linear_CKA'] for v in a['hidden_geometry']['pairwise'] if v['centered_linear_CKA'] is not None)
                entry['mean_same_graph_cosine'] = statistics.mean(v['same_graph_cosine_mean'] for v in a['hidden_geometry']['pairwise'])
                if not a['sanity_against_saved_export']['passed']:
                    warnings.append({'seed': seed, 'method': name, 'sanity': a['sanity_against_saved_export']})
            rows.append(entry)
            margins[name] = margin
        base = margins['gnnm_boundary_native']
        base_credit = (base > 0).double() + .5 * (base == 0).double()
        for name, margin in margins.items():
            if name == 'gnnm_boundary_native':
                continue
            credit = (margin > 0).double() + .5 * (margin == 0).double()
            comparisons.append({'seed': seed, 'method': name, 'reference': 'gnnm_boundary_native',
                                'strict_repairs': int(((base < 0) & (margin > 0)).sum()),
                                'strict_harms': int(((base > 0) & (margin < 0)).sum()),
                                'changed_credit_pairs': int((credit != base_credit).sum()),
                                'delta_AUC_pp': 100 * float((credit - base_credit).mean())})
    summaries = []
    keys = ['mean_member_AUC', 'pool_AUC', 'pool_minus_mean_member_AUC_pp',
            'all_members_reversed', 'pool_reversed_despite_some_member_correct',
            'pool_reversed_despite_strict_majority_correct', 'fixed_probability_minus_logit_AUC_pp',
            'mean_logit_BCE', 'mean_probability_BCE']
    for name in dict.fromkeys(r['method'] for r in rows):
        selected = [r for r in rows if r['method'] == name]
        assert len(selected) == 3
        summary = {k: {'values': [r[k] for r in selected],
                       'mean': statistics.mean(r[k] for r in selected),
                       'sample_SD': statistics.stdev(r[k] for r in selected)} for k in keys}
        summaries.append({'method': name, 'paired_seeds': 3, 'summary': summary})
    result = {'schema': 'complete-molbace-route-diagnostic-v1', 'status': 'complete',
              'scope': 'posthoc selected VALID; not independent confirmation or causal attribution',
              'graphs': 151, 'positive_graphs': 130, 'negative_graphs': 21,
              'pairs_per_seed': 2730, 'pairs_are_independent_repetitions': False,
              'rows': rows, 'comparisons': comparisons, 'summaries': summaries, 'warnings': warnings,
              'new_models': 0, 'new_forwards': 0, 'optimizer_steps': 0, 'TEST_accessed': False,
              'source_sha256': sha(Path(__file__)), 'input_sha256': sha(inputs_path),
              'wall_seconds': time.monotonic() - started,
              'peak_rss_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (output / 'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.project_root, args.inputs, args.output)
    print(json.dumps({k: result[k] for k in ['status', 'wall_seconds', 'peak_rss_KiB', 'warnings']}))
