"""Disabled prospective DDI TRAIN support census; no model or scoring path.

Numerical imports occur only after an exact external root release. This file
must be inspected as text/AST during source preparation, never imported here.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import resource
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            digest.update(part)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def tensor_digest(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(str((array.shape, array.dtype)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def gate(args):
    plan = json.loads((HERE / 'PLAN.json').read_text())
    release_path = args.root_release.resolve()
    expected_parent = PHASE / plan['execution_directory']
    require(release_path == expected_parent / 'ROOT_RELEASE.json'
            and not args.root_release.is_symlink()
            and sha(release_path) == args.root_release_sha256,
            'Exact external root release required')
    release = json.loads(release_path.read_text())
    require(release['status'] == 'APPROVED' and release['root_authorization_reference']
            and release['stage'] == 'DDI_native_TRAIN_support_census'
            and release['source_manifest_sha256'] == sha(HERE / 'MANIFEST.json')
            and release['plan_sha256'] == sha(HERE / 'PLAN.json'), 'Source/plan release differs')
    for key in ('seed', 'native_batch_size', 'full_batches', 'dropped_tail', 'query_chunk_size',
                'optimizer_updates', 'models_initialized', 'VALID_TEST_reads', 'automatic_retry'):
        require(release[key] == plan[key], 'Released contract differs: ' + key)
    require(plan['status'] == 'DISABLED_SOURCE_PREPARATION' and plan['optimizer_updates'] == 0
            and plan['models_initialized'] is False and plan['VALID_TEST_reads'] is False,
            'This preparation admits census only')
    for row in json.loads((HERE / 'MANIFEST.json').read_text())['files']:
        path = HERE / row['path']
        require(path.resolve().is_relative_to(HERE) and not path.is_symlink()
                and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Source changed')
    train_path = PHASE / plan['train_relative_path']
    require(train_path.resolve().is_relative_to(PHASE) and not train_path.is_symlink()
            and sha(train_path) == plan['TRAIN_ONLY_sha256'], 'Acquired TRAIN file changed')
    output = expected_parent / 'run01'
    require(not output.exists(), 'Fresh output required; no restart or automatic retry')
    output.mkdir(parents=True)
    return plan, release_path, train_path, output


def quantile(histogram, percentile):
    count = sum(histogram)
    rank = max(1, (count * percentile + 99) // 100)
    cumulative = 0
    for value, frequency in enumerate(histogram):
        cumulative += frequency
        if cumulative >= rank:
            return value
    raise RuntimeError('Histogram accounting failed')


class Accumulator:
    """Fixed-size histograms and integer reductions; no retained candidate slots."""
    def __init__(self, torch, nodes, device):
        self.torch, self.nodes, self.device = torch, nodes, device
        self.queries = 0
        self.n_hist = torch.zeros((2, nodes), dtype=torch.int64, device=device)
        self.k_hist = torch.zeros_like(self.n_hist)
        self.r_hist = torch.zeros((2, (nodes - 1) // 2 + 1), dtype=torch.int64, device=device)
        self.cn_hist = torch.zeros(nodes, dtype=torch.int64, device=device)
        self.joint = torch.zeros(9, dtype=torch.int64, device=device)
        self.totals = torch.zeros((2, 4), dtype=torch.int64, device=device)
        self.maxima = torch.zeros_like(self.totals)
        self.side_classes = torch.zeros((2, 5), dtype=torch.int64, device=device)
        self.upper = torch.tensor([0] + [(1 << b) - 1 for b in range(1, 26)], device=device)
        self.work_hist = torch.zeros((2, 26), dtype=torch.int64, device=device)
        self.combined_work_hist = torch.zeros(26, dtype=torch.int64, device=device)

    def update(self, n, k, cn):
        t = self.torch
        r = t.minimum(k, n - k)
        work = n * r
        require(bool(((0 <= k) & (k <= n)).all()), 'Count outside support')
        group = t.where(r == 0, 0, t.where(r == 1, 1, 2))
        self.joint += t.bincount(group[:, 0] * 3 + group[:, 1], minlength=9)
        self.cn_hist += t.bincount(cn, minlength=self.nodes)
        self.totals += t.stack((n.sum(0), k.sum(0), r.sum(0), work.sum(0)), dim=1)
        maxima = t.stack((n.amax(0), k.amax(0), r.amax(0), work.amax(0)), dim=1)
        self.maxima = t.maximum(self.maxima, maxima)
        classes = t.stack(((n == 0).sum(0), (k == 0).sum(0),
                           ((k == n) & (n > 0)).sum(0), (r == 1).sum(0), (r > 1).sum(0)), dim=1)
        self.side_classes += classes
        for side in range(2):
            self.n_hist[side] += t.bincount(n[:, side], minlength=self.nodes)
            self.k_hist[side] += t.bincount(k[:, side], minlength=self.nodes)
            self.r_hist[side] += t.bincount(r[:, side], minlength=self.r_hist.shape[1])
            self.work_hist[side] += t.bincount(t.bucketize(work[:, side], self.upper), minlength=26)
        self.combined_work_hist += t.bincount(t.bucketize(work.sum(1), self.upper), minlength=26)
        self.queries += len(n)
        return r

    def merge(self, other):
        for name in ('n_hist', 'k_hist', 'r_hist', 'cn_hist', 'joint', 'totals',
                     'side_classes', 'work_hist', 'combined_work_hist'):
            getattr(self, name).add_(getattr(other, name))
        self.maxima = self.torch.maximum(self.maxima, other.maxima)
        self.queries += other.queries

    def report(self, include_histograms):
        joint = self.joint.cpu().reshape(3, 3).tolist()
        totals, maxima, classes = (getattr(self, n).cpu().tolist()
                                  for n in ('totals', 'maxima', 'side_classes'))
        require(sum(map(sum, joint)) == self.queries, 'Joint count accounting failed')
        variable = sum(joint[a][b] for a in (1, 2) for b in (1, 2))
        genuine = joint[2][2]
        sides = []
        for side in range(2):
            item = dict(zip(('total_slots', 'observed_TRAIN_bits', 'r_sum', 'n_times_r_sum'), totals[side]))
            item.update(dict(zip(('max_n', 'max_k', 'max_r', 'max_n_times_r'), maxima[side])))
            item.update(dict(zip(('empty', 'zero_count', 'full_nonempty_count',
                                  'categorical_or_complement', 'genuine_subset'), classes[side])))
            item['unique_support'] = int(sum(joint[0]) if side == 0 else sum(row[0] for row in joint))
            item['fractions'] = {name: item[name] / self.queries for name in
                                 ('empty', 'zero_count', 'full_nonempty_count', 'unique_support',
                                  'categorical_or_complement', 'genuine_subset')}
            sides.append(item)
        result = dict(queries=self.queries, sides=sides, joint_group_counts=joint,
                      both_variable=variable, both_genuine=genuine,
                      either_genuine=sum(joint[2]) + sum(row[2] for row in joint) - genuine,
                      not_both_variable=self.queries - variable,
                      both_variable_fraction=variable / self.queries,
                      both_genuine_fraction=genuine / self.queries)
        if include_histograms:
            hists = {name: getattr(self, name).cpu().tolist()
                     for name in ('n_hist', 'k_hist', 'r_hist', 'cn_hist', 'work_hist', 'combined_work_hist')}
            require(all(sum(hists[name][side]) == self.queries
                        for name in ('n_hist', 'k_hist', 'r_hist', 'work_hist') for side in range(2))
                    and sum(hists['cn_hist']) == self.queries
                    and sum(hists['combined_work_hist']) == self.queries, 'Histogram accounting failed')
            result['histograms'] = hists
            result['exact_quantiles'] = {name: [{str(q): quantile(hist, q) for q in (50, 90, 95, 99)}
                                                for hist in hists[name]] for name in ('n_hist', 'k_hist', 'r_hist')}
            result['work_histogram_intervals'] = [[0, 0]] + [[1 << (b - 1), (1 << b) - 1] for b in range(1, 26)]
        return result


def population(torch, visible, teacher, queries, nodes, chunk_size, started, plan):
    """Same full native mask; chunking changes workspace only, not support."""
    stats = Accumulator(torch, nodes, queries.device)
    counts_digest = hashlib.sha256()
    for start in range(0, len(queries), chunk_size):
        require(time.monotonic() - started <= plan['wall_seconds_cap'], 'Wall cap exceeded; incomplete census')
        query = queries[start:start + chunk_size]
        left_neighbors, right_neighbors = visible[query[:, 0]], visible[query[:, 1]]
        left, right = left_neighbors & ~right_neighbors, right_neighbors & ~left_neighbors
        common = (left_neighbors & right_neighbors).sum(1, dtype=torch.int64)
        n = torch.stack((left.sum(1, dtype=torch.int64), right.sum(1, dtype=torch.int64)), dim=1)
        k = torch.stack(((left & teacher[query[:, 1]]).sum(1, dtype=torch.int64),
                         (right & teacher[query[:, 0]]).sum(1, dtype=torch.int64)), dim=1)
        stats.update(n, k, common)
        counts_digest.update(torch.cat((n, k), dim=1).cpu().contiguous().numpy().tobytes())
        del query, left_neighbors, right_neighbors, left, right, common, n, k
    return stats, counts_digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-release', type=Path, required=True)
    parser.add_argument('--root-release-sha256', required=True)
    args = parser.parse_args()
    plan, release_path, train_path, output = gate(args)
    started = time.monotonic()
    completed = 0
    try:
        import numpy as np
        import torch
        from torch_geometric.utils import negative_sampling
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Declared single CUDA route required')
        for name, version in plan['used_distribution_versions'].items():
            require(importlib.metadata.version(name) == version, 'Existing used distribution differs')
        sampler_path = Path(inspect.getsourcefile(negative_sampling)).resolve()
        sampler_pin = plan['native_negative_sampler']
        require(str(sampler_path) == sampler_pin['path'] and sha(sampler_path) == sampler_pin['sha256']
                and hashlib.sha256(inspect.getsource(negative_sampling).encode()).hexdigest()
                == sampler_pin['function_sha256'], 'Native negative sampler differs')
        stored = torch.load(train_path, map_location='cpu', weights_only=True)
        require(type(stored) is dict and set(stored) == {'edge', 'num_nodes'}, 'Acquired TRAIN-only schema differs')
        pairs_cpu = stored['edge']
        nodes, records = plan['nodes'], plan['TRAIN_records']
        require(type(stored['num_nodes']) is int and stored['num_nodes'] == nodes
                and type(pairs_cpu) is torch.Tensor and pairs_cpu.dtype == torch.int64
                and pairs_cpu.device.type == 'cpu' and list(pairs_cpu.shape) == [records, 2]
                and hashlib.sha256(pairs_cpu.contiguous().numpy().tobytes()).hexdigest()
                == plan['TRAIN_tensor_bytes_sha256'], 'TRAIN tensor binding differs')
        device = torch.device(plan['device'])
        pairs = pairs_cpu.to(device)
        require(bool((pairs >= 0).all() & (pairs < nodes).all())
                and not bool((pairs[:, 0] == pairs[:, 1]).any()), 'Invalid TRAIN endpoints')
        teacher = torch.zeros((nodes, nodes), dtype=torch.bool, device=device)
        teacher[pairs[:, 0], pairs[:, 1]] = True
        teacher[pairs[:, 1], pairs[:, 0]] = True
        require(int(teacher.sum()) == 2 * records, 'TRAIN uniqueness/undirected graph differs')
        # Both directions, no selfloops, no edge weights. Ordering is explicit.
        # The native sampler's membership/count semantics do not depend on this
        # ordering; no equality to an unacquired raw-graph byte order is claimed.
        raw_edges = torch.cat((pairs, pairs.flip(1)), dim=0).T.contiguous()
        native_tree = ast.parse((HERE / 'native_utils.py').read_text())
        iterator_class = [node for node in native_tree.body if isinstance(node, ast.ClassDef) and node.name == 'PermIterator']
        require(len(iterator_class) == 1, 'Pinned native iterator absent')
        scope = dict(torch=torch)
        exec(compile(ast.Module(body=iterator_class, type_ignores=[]), str(HERE / 'native_utils.py'), 'exec'), scope)
        random.seed(plan['seed']); np.random.seed(plan['seed'])
        torch.manual_seed(plan['seed']); torch.cuda.manual_seed_all(plan['seed'])
        torch.cuda.synchronize()
        sampler_started = time.monotonic()
        negative = negative_sampling(raw_edges, nodes).T.contiguous()
        torch.cuda.synchronize()
        sampler_seconds = time.monotonic() - sampler_started
        require(negative.dtype == torch.int64 and negative.ndim == 2 and negative.shape[1] == 2
                and len(negative) >= records and not bool((negative[:, 0] == negative[:, 1]).any())
                and not bool(teacher[negative[:, 0], negative[:, 1]].any()), 'Native TRAIN-negative draw differs')
        iterator = scope['PermIterator'](device, records, plan['native_batch_size'])
        require(len(iterator) == plan['full_batches']
                and records - len(iterator) * plan['native_batch_size'] == plan['dropped_tail'], 'Native tail policy differs')
        queried = len(iterator) * plan['native_batch_size']
        write(output / 'DRAW.json', dict(seed=plan['seed'], negative_sampler_seconds=sampler_seconds,
              requested_negative_count_default=2 * records, returned_negative_count=len(negative),
              negative_draw_digest=tensor_digest(negative), permutation_digest=tensor_digest(iterator.idx),
              tail_record_ids_digest=tensor_digest(iterator.idx[queried:]), tail_positive_queries_digest=tensor_digest(pairs[iterator.idx[queried:]]),
              tail_negative_queries_digest=tensor_digest(negative[iterator.idx[queried:]]),
              queried_per_population=queried, dropped_tail=plan['dropped_tail'],
              unqueried_negative_draw_entries=len(negative) - queried,
              negative_draw_entries_beyond_TRAIN_index_range=len(negative) - records,
              models_initialized=False, preceding_encoder_dropout_draws=False))
        overall = {name: Accumulator(torch, nodes, device) for name in ('positive', 'negative')}
        with (output / 'BATCHES.jsonl').open('x') as journal:
            for batch, record_ids in enumerate(iterator, 1):
                batch_started = time.monotonic()
                visible = teacher.clone()
                targets = pairs[record_ids]
                visible[targets[:, 0], targets[:, 1]] = False
                visible[targets[:, 1], targets[:, 0]] = False
                row = dict(batch=batch, record_ids_digest=tensor_digest(record_ids), populations={})
                for name, queries in (('positive', targets), ('negative', negative[record_ids])):
                    phase_started = time.monotonic()
                    stats, digest = population(torch, visible, teacher, queries, nodes,
                                               plan['query_chunk_size'], started, plan)
                    overall[name].merge(stats)
                    torch.cuda.synchronize()
                    row['populations'][name] = dict(stats.report(False), counts_digest=digest,
                          query_digest=tensor_digest(queries), elapsed_seconds=time.monotonic() - phase_started)
                    del stats, queries
                torch.cuda.synchronize()
                require(torch.cuda.max_memory_allocated() <= plan['cuda_allocated_bytes_cap']
                        and torch.cuda.max_memory_reserved() <= plan['cuda_reserved_bytes_cap'], 'Observed CUDA cap exceeded')
                require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= plan['host_RSS_bytes_cap'], 'Observed host peak RSS cap exceeded')
                row['elapsed_seconds'] = time.monotonic() - batch_started
                journal.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                journal.flush(); os.fsync(journal.fileno())
                completed = batch
                write(output / 'PROGRESS.json', dict(completed_full_batches=completed,
                      elapsed_seconds=time.monotonic() - started, scope='DDI_TRAIN_geometry_only'))
                del visible, targets, record_ids
        require(completed == plan['full_batches'] and all(x.queries == queried for x in overall.values()), 'Incomplete native census')
        require(sha(train_path) == plan['TRAIN_ONLY_sha256']
                and sha(release_path) == args.root_release_sha256, 'Final TRAIN/release changed')
        release = json.loads(release_path.read_text())
        require(sha(HERE / 'MANIFEST.json') == release['source_manifest_sha256']
                and sha(HERE / 'PLAN.json') == release['plan_sha256'], 'Final source controls changed')
        for pin in json.loads((HERE / 'MANIFEST.json').read_text())['files']:
            path = HERE / pin['path']
            require(path.resolve().is_relative_to(HERE) and not path.is_symlink()
                    and path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'], 'Final source payload changed')
        require(sha(sampler_path) == sampler_pin['sha256'], 'Final sampler source changed')
        require(time.monotonic() - started <= plan['wall_seconds_cap'], 'Wall cap exceeded; incomplete census')
        write(output / 'CENSUS.json', dict(status='COMPLETE_DDI_NATIVE_TRAIN_FULL_BATCH_CENSUS',
              UTC=datetime.now(timezone.utc).isoformat(), source_manifest_sha256=sha(HERE / 'MANIFEST.json'),
              plan_sha256=sha(HERE / 'PLAN.json'), root_release_sha256=args.root_release_sha256,
              TRAIN_ONLY_sha256=plan['TRAIN_ONLY_sha256'], seed=plan['seed'], full_batches=completed,
              native_batch_size=plan['native_batch_size'], dropped_tail=plan['dropped_tail'],
              query_chunk_size=plan['query_chunk_size'], populations={name: value.report(True) for name, value in overall.items()},
              wall_seconds=time.monotonic() - started, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(), host_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              input_files_opened=['TRAIN_ONLY.pt'], models_initialized=False, optimizer_updates=0,
              features_read=False, VALID_TEST_reads=False, logits_or_scores=False,
              interpretation='TRAIN support opportunity and prospective DP cost only; no learned association, transfer or predictive verdict.'))
    except BaseException as error:
        write(output / 'FAILURE.json', dict(status='INCOMPLETE_NONQUALIFYING', error_type=type(error).__name__,
              error=str(error), completed_full_batches=completed, automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
