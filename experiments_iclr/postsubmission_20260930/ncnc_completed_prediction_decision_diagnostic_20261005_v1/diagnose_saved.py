"""Frozen all25 served-prediction diagnostic; no dataset, model or scoring API."""
from pathlib import Path
import hashlib
import itertools
import json
import os
import resource
import sys
import time
import traceback

REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
HERE = Path(__file__).resolve().parent
PROTOCOL_SHA = '86f283b916cbae51c4ac97e4f098dabdde85e61bc38f34d20df83f7e8c8620d5'


def require(value, message):
    if not value:
        raise ValueError(message)


def verified(record):
    path = Path(record['path'])
    if not path.is_absolute():
        path = PHASE / path
    require(path.resolve() == path.absolute() and path.is_relative_to(PHASE), 'Bound project input required')
    raw = path.read_bytes()
    require(len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256'], 'Bound bytes differ: ' + str(path))
    return path


def tensor_sha(value):
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(str((array.shape, array.dtype)).encode())
    digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def pearson(left, right):
    # Descriptive Pearson in FP64; no fitting, calibration or query-independent inference.
    a = left.double() - left.double().mean()
    b = right.double() - right.double().mean()
    denominator = (a.square().sum() * b.square().sum()).sqrt()
    return None if float(denominator) == 0.0 else float((a * b).sum() / denominator)


def main():
    require(Path.cwd() == REPO and os.uname().nodename == 'peptide', 'Authorized18.77 project process required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        and os.environ.get('OMP_NUM_THREADS') == os.environ.get('MKL_NUM_THREADS') == '2', 'Frozen CPU/thread policy required')
    raw_protocol = (HERE / 'PROTOCOL.json').read_bytes()
    require(hashlib.sha256(raw_protocol).hexdigest() == PROTOCOL_SHA, 'Frozen protocol differs')
    protocol = json.loads(raw_protocol)
    wall, cpu = time.perf_counter(), time.process_time()
    out = HERE / 'run01'
    out.mkdir(exist_ok=False)
    result = dict(schema='ncnc-completed-TEST-decision-diagnostic-result-v1', status='STARTED',
        protocol_sha256=PROTOCOL_SHA, exploratory_posthoc_consumed_TEST=True,
        model_fits=0, model_forwards=0, optimizer_updates=0, raw_dataset_files_opened=0,
        checkpoints_or_labels_opened=False, new_predictor_created=False,
        raw_arrays_exported=False, internal_member_diversity_established=False,
        all25_cells_required=True, all5_seed_blocks_required=True,
        all10_pairs_per_seed_required=True, cells=[], seed_blocks=[])
    try:
        import torch
        import numpy as np
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        result['runtime'] = dict(python=sys.version, executable=sys.executable,
            torch=torch.__version__, numpy=np.__version__, CPU_only=True,
            torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads())
        # The local fetched master is copied byte-for-byte for provenance; it is
        # only the already-consumed completed family's published compact receipt.
        master_raw = (HERE / 'COMPLETED_MASTER.json').read_bytes()
        pin = protocol['input_master']
        require(len(master_raw) == pin['bytes'] and hashlib.sha256(master_raw).hexdigest() == pin['sha256'], 'Completed master differs')
        master = json.loads(master_raw)
        require(master['status'] == 'ALL25_FROZEN_HELDOUT_CONFIRMATION_COMPLETE'
            and master['TEST_opened'] is True and master['official_TEST_query_receipt']['complete_original_order'] is True,
            'Completed consumed official query family required')
        n = master['official_TEST_query_receipt']['positive_rows']
        require(n == 46329 and master['official_TEST_query_receipt']['negative_rows'] == 100000,
                'Original full pool counts differ')
        arms = tuple(dict.fromkeys(row['arm'] for row in protocol['cells']))
        require(len(arms) == 5 and len(protocol['cells']) == protocol['all_cells_required'] == 25,
                'Complete five-family25-cell protocol required')
        cells_by_key = {(row['arm'], row['base_seed']): row for row in master['cells']}
        bank = {}
        for row in protocol['cells']:
            arm, seed = row['arm'], row['base_seed']
            require((arm, seed) not in bank and seed in protocol['seed_blocks'], 'Duplicate/unbound cell')
            original = cells_by_key[(arm, seed)]
            require(original['status'] == 'PASS'
                and original['raw_served_score_evidence'] == row['raw_served_score_evidence']
                and original['served_score_digests'] == row['served_score_digests']
                and original['TEST_hits50'] == row['recorded_TEST_hits50'], 'Original completed cell binding differs')
            path = verified(row['raw_served_score_evidence'])
            saved = torch.load(path, map_location='cpu', weights_only=True)
            require(saved['schema'] == 'ncnc-heldout-private-returned-numerical-evidence-v2'
                and set(saved['values']) == {'positive', 'negative'}, 'Saved served-payload schema differs')
            binding = saved['binding']
            require(binding['arm'] == arm and binding['base_seed'] == seed and binding['member'] is None
                and binding['identity'] == master['identity']
                and binding['root_release_sha256'] == master['root_release_sha256']
                and binding['heldout_source_manifest_sha256'] == master['heldout_source_manifest_sha256'],
                'Saved served-cell identity differs')
            query = binding['canonical_graph_and_query_digests']
            require(query['valid_positive'] == master['official_TEST_query_receipt']['typed_query_digests']['positive_sha256']
                and query['valid_negative'] == master['official_TEST_query_receipt']['typed_query_digests']['negative_sha256']
                and query['pairs'] == master['actual_graph_receipt']['canonical_pairs_sha256'],
                'Inherited original query ordering/topology reference differs')
            positive, negative = saved['values']['positive'], saved['values']['negative']
            require(torch.is_tensor(positive) and torch.is_tensor(negative)
                and positive.device.type == negative.device.type == 'cpu'
                and positive.dtype == negative.dtype == torch.float32
                and positive.shape == (n,) and negative.shape == (100000,)
                and bool(torch.isfinite(positive).all()) and bool(torch.isfinite(negative).all()), 'Full finiteFP32 served pools differ')
            require(tensor_sha(positive) == row['served_score_digests']['positive']
                and tensor_sha(negative) == row['served_score_digests']['negative'], 'Typed served tensor digests differ')
            hits = positive > torch.topk(negative, protocol['negative_top_k'])[0][-1]
            hit_count = int(hits.sum())
            actual = hit_count / n
            require(abs(actual - row['recorded_TEST_hits50']) <= 1e-12, 'Recorded Hits50 mismatch; do not update original score')
            bank[(arm, seed)] = dict(positive=positive, negative=negative, hits=hits, hit_count=hit_count)
            result['cells'].append(dict(arm=arm, base_seed=seed, positive_count=n, negative_count=100000,
                positive_hits50=hit_count, original_recorded_hits50=row['recorded_TEST_hits50'],
                all_bytes_digests_shape_dtype_finite_and_original_hit_checks_passed=True))
        require(set(bank) == set(itertools.product(arms, protocol['seed_blocks'])), 'All25 served cells required before pair metrics')
        for seed in protocol['seed_blocks']:
            block = dict(base_seed=seed, positive_query_count=n, pairs=[])
            result['seed_blocks'].append(block)
            for a, b in itertools.combinations(arms, 2):
                x, y = bank[(a, seed)], bank[(b, seed)]
                both = int((x['hits'] & y['hits']).sum())
                a_only = int((x['hits'] & ~y['hits']).sum())
                b_only = int((~x['hits'] & y['hits']).sum())
                neither = int((~x['hits'] & ~y['hits']).sum())
                require(both + a_only + b_only + neither == n
                    and a_only - b_only == x['hit_count'] - y['hit_count'], 'Pair accounting differs')
                block['pairs'].append(dict(candidate=a, control=b, both_hit=both,
                    candidate_only_hit_rescue=a_only, control_only_hit_introduced_miss=b_only,
                    neither_hit=neither, rescue_proportion_of_all_positive_queries=a_only / n,
                    introduced_miss_proportion_of_all_positive_queries=b_only / n,
                    positive_score_Pearson=pearson(x['positive'], y['positive']),
                    negative_score_Pearson=pearson(x['negative'], y['negative'])))
            require(len(block['pairs']) == 10, 'Full10-pair seed table required')
        result.update(status='ALL25_FROZEN_SERVED_DECISION_DIAGNOSTIC_COMPLETE',
            cells_completed=25, seed_blocks_completed=5, pair_tables_completed=50,
            uncertainty_or_significance_claim=False, methodological_novelty_or_quality_advantage_established=False)
    except Exception as error:
        result.update(status='FAILED_OR_UNRESOLVED_FROZEN_SERVED_DECISION_DIAGNOSTIC',
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            retry_allowed=False)
    finally:
        result.update(total_wall_seconds=time.perf_counter() - wall,
            total_process_cpu_seconds=time.process_time() - cpu,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        with (out / 'DECISION_RESULT.json').open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
