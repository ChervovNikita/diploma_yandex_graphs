"""Disabled, fixed 24-config/three-seed native-own-best C&S evaluator.

Caller supplies root-audited numeric roles. No role loader, refit, candidate
score access, TEST input, retry, search expansion or source acquisition.
"""
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SERVER_PHASE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/'
                    'diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
PROTOCOL_PATH = 'native_own_best_train_only_cs_reference_protocol_20261008_v1/PROTOCOL.json'
PROTOCOL_SHA256 = '927a8dd637a73e1d1615aecfc1a4719383da037521777a09cbd3337b6ba17a49'
SEEDS = (6101, 6203, 6307)
CONFIG_IDS = tuple('CS%02d' % index for index in range(1, 25))
ROLE_KEYS = ('TRAIN_x', 'prepared_edge_index', 'TRAIN_ids', 'TRAIN_y',
             'development_ids', 'development_y')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tensor_digest(value):
    # Exact existing common.tensor_digest convention, without importing models.
    value = value.detach().cpu().contiguous()
    result = hashlib.sha256()
    result.update(json.dumps(dict(shape=list(value.shape), dtype=str(value.dtype)),
                             sort_keys=True).encode())
    result.update(value.numpy().tobytes(order='C'))
    return result.hexdigest()


@contextmanager
def _cost(costs, operation, torch=None, device='cpu'):
    wall, cpu = time.perf_counter(), time.process_time()
    row = dict(operation=operation, completed=False)
    costs.append(row)
    try:
        if torch is not None and torch.device(device).type == 'cuda' and torch.cuda.is_initialized():
            torch.cuda.synchronize(device)
        yield
        if torch is not None and torch.device(device).type == 'cuda' and torch.cuda.is_initialized():
            torch.cuda.synchronize(device)
        row['completed'] = True
    finally:
        row.update(wall_seconds=time.perf_counter()-wall,
                   process_cpu_seconds=time.process_time()-cpu,
                   process_cumulative_peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _error(error):
    return dict(type=type(error).__name__, message=str(error))


def _readout(torch, scores, ids, truth):
    """Raw-score accuracy; declared row normalization; exact zero-target NLL."""
    _require(tuple(scores.shape) == (11701, 10) and bool(torch.isfinite(scores).all())
             and bool((scores >= 0).all()), 'Finite nonnegative full author scores required')
    selected = scores.detach().cpu()[ids]
    correct = int((selected.argmax(-1) == truth).sum())
    values = selected.to(dtype=torch.float64)
    sums = values.sum(-1, keepdim=True)
    zero_rows = sums[:, 0] == 0
    # Zero rows become uniform, with no floor applied to any nonzero row.
    probabilities = torch.empty_like(values)
    probabilities[~zero_rows] = values[~zero_rows] / sums[~zero_rows]
    probabilities[zero_rows] = 0.1
    targets = probabilities.gather(-1, truth[:, None]).squeeze(-1)
    zero_targets = int((targets == 0).sum())
    nll = (dict(tag='positive_infinity', value=None, zero_target_count=zero_targets)
           if zero_targets else dict(tag='finite', value=float(-targets.log().mean()),
                                     zero_target_count=0))
    one_hot = torch.nn.functional.one_hot(truth, num_classes=10).to(torch.float64)
    brier = float((probabilities-one_hot).square().sum(-1).mean())
    _require(nll['tag'] != 'finite' or nll['value'] >= 0, 'Nonnegative finite NLL')
    return dict(nodes=5274, correctcount=correct, accuracy=correct/5274,
                NLL=nll, Brier=brier, uniform_zero_rows=int(zero_rows.sum()),
                score_probability_map='nonnegative row normalization; zero row uniform C10',
                readout_arithmetic_dtype='torch.float64; raw-score argmax unchanged')


def evaluate_fixed_reference(*, native_state_paths, train_data, development_ids,
                             development_truths, polynormer, release_path, output,
                             native_device='cpu', later_execution_authorized=False):
    """Run only after root release on the authorized allocation, never locally.

    All three authentic own-best file hashes and exact role tensor digests are
    root bindings, not discovered or selected here. The root release attests
    the complete correction family and prior comparative opening. Native
    states contain their original selected epochs; these are never replaced.
    """
    if later_execution_authorized is not True:
        raise PermissionError('Disabled C&S evaluator; separate root release required')
    entered_wall, entered_cpu = time.perf_counter(), time.process_time()
    _require(PHASE == SERVER_PHASE, 'Server-only raw probabilities and execution')
    output = Path(output).resolve()
    release_path = Path(release_path).resolve(strict=True)
    _require(output.is_relative_to(PHASE) and release_path.is_relative_to(PHASE)
             and not output.exists(), 'Fresh output and root release inside project phase')
    release = json.loads(release_path.read_text())
    _require(release['execution_enabled'] is True
             and release['protocol_sha256'] == PROTOCOL_SHA256
             and release['all3_native_and_all12_correction_records_complete'] is True
             and release['root_comparative_opening_authorized'] is True
             and release['native_serving_runtime_qualified'] is True
             and release['CS_sparse_runtime_qualified'] is True,
             'Root complete-family opening and bounded runtime release required')
    _require(_sha(HERE/'MANIFEST.json') == release['evaluator_manifest_sha256'],
             'Exact root-adopted evaluator packet')
    for item in json.loads((HERE/'MANIFEST.json').read_text())['files']:
        path = (PHASE/item['path']).resolve(strict=True)
        _require(path.is_relative_to(HERE) and _sha(path) == item['sha256']
                 and path.stat().st_size == item['bytes'], 'Changed evaluator payload')
    _require(_sha(PHASE/PROTOCOL_PATH) == PROTOCOL_SHA256, 'Unchanged fixed protocol')
    protocol = json.loads((PHASE/PROTOCOL_PATH).read_text())
    _require(protocol['native_seeds'] == list(SEEDS)
             and [row['id'] for row in protocol['configurations']] == list(CONFIG_IDS)
             and protocol['configuration_count'] == 24
             and protocol['complete_reference_records'] == 72, 'Exact prospective family')
    paths = {seed: Path(native_state_paths[seed]).resolve(strict=True) for seed in SEEDS}
    _require(set(native_state_paths) == set(SEEDS)
             and all(path.is_relative_to(PHASE) for path in paths.values()),
             'Exactly three server-owned native states')
    opening = (PHASE/release['whole_family_opening']['path']).resolve(strict=True)
    _require(opening.is_relative_to(PHASE)
             and _sha(opening) == release['whole_family_opening']['sha256'],
             'Root-bound whole-family opening receipt; body not interpreted')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    receipt = dict(schema='fixed-native-own-best-TRAIN-only-CS-evaluation-v1',
                   complete=False, protocol_sha256=PROTOCOL_SHA256,
                   evaluator_manifest_sha256=release['evaluator_manifest_sha256'],
                   evaluator_program_sha256=_sha(__file__),
                   root_release_sha256=_sha(release_path), native_device=native_device,
                   CS_device='cpu', TEST_access=False, refit=False, automatic_retry=False,
                   preflight_wall_seconds=time.perf_counter()-entered_wall,
                   preflight_process_cpu_seconds=time.process_time()-entered_cpu,
                   receipt_write_calls=0, completed_prior_receipt_IO_wall_seconds=0.0,
                   native_forward_calls_attempted=0, native_forward_calls_completed=0,
                   scheduled_author_propagation_passes=3*sum(
                       cfg['configuration']['num_correction_layers']+cfg['configuration']['num_smoothing_layers']
                       for cfg in protocol['configurations']),
                   selection=None, costs=[], native_records=[],
                   records=[dict(seed=seed, configuration_id=cfg['id'], method=cfg['method'],
                                 configuration=cfg['configuration'], status='pending', costs=[],
                                 scheduled_propagation_passes=cfg['configuration']['num_correction_layers']+
                                                             cfg['configuration']['num_smoothing_layers'],
                                 completed_propagation_passes=0)
                            for seed in SEEDS for cfg in protocol['configurations']])
    def flush():
        receipt['elapsed_wall_seconds'] = time.perf_counter()-started
        receipt['receipt_write_calls'] += 1
        io_started = time.perf_counter()
        _write(output/'RECEIPT.json', receipt)
        # Next receipt retains this write's cost; a final cost footer accounts
        # through the last receipt without a self-referential size/hash field.
        receipt['completed_prior_receipt_IO_wall_seconds'] += time.perf_counter()-io_started
    def cost_footer():
        _write(output/'COST_AND_FAILURE_RECEIPT.json', dict(
            complete=receipt['complete'], records=72,
            statuses={status: sum(row['status'] == status for row in receipt['records'])
                      for status in ('complete', 'failed', 'blocked')},
            reference_receipt_sha256=_sha(output/'RECEIPT.json'),
            reference_receipt_bytes=(output/'RECEIPT.json').stat().st_size,
            probability_files=sum(row.get('probability_storage_complete', False) for row in receipt['native_records']),
            probability_file_bytes=sum(path.stat().st_size for path in output.glob('NATIVE_PROBABILITIES_seed*.pt')),
            partial_probability_files=[row['probability_file'] for row in receipt['native_records']
                if row['probability_file'] is not None and not row.get('probability_storage_complete', False)],
            receipt_write_calls=receipt['receipt_write_calls'],
            receipt_IO_wall_seconds=receipt['completed_prior_receipt_IO_wall_seconds'],
            elapsed_wall_seconds_through_last_reference_receipt=time.perf_counter()-entered_wall,
            cost_scope='All stages including repeated author graph preparation; footer serialization excluded',
            raw_probability_storage='server-only', selected_reference_claim=receipt['complete']))
    flush()
    try:
        with _cost(receipt['costs'], 'numerical_dependencies_and_pinned_reference_imports'):
            import torch
            bindings = json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
            for row in bindings['files']:
                _require(_sha(PHASE/row['path']) == row['sha256'], 'Pinned evaluator dependency changed')
            source = PHASE/protocol['source_directory']
            native_adapter = _module(source/'native_adapter.py', '_fixed_CS_native_adapter')
            cs = _module(source/'cs_reference.py', '_fixed_CS_train_only_adapter')
        with _cost(receipt['costs'], 'numeric_role_joins_and_nonself_graph_preparation'):
            ids = development_ids.detach().cpu().clone()
            truth = development_truths.detach().cpu().clone()
            values = dict(TRAIN_x=train_data['x'], prepared_edge_index=train_data['edge_index'],
                          TRAIN_ids=train_data['ids'], TRAIN_y=train_data['y'],
                          development_ids=ids, development_y=truth)
            _require(set(release['numeric_role_sha256']) == set(ROLE_KEYS), 'Explicit complete numeric roles')
            for key, tensor in values.items():
                _require(_tensor_digest(tensor) == release['numeric_role_sha256'][key], 'Changed numeric role: '+key)
            _require(ids.dtype == truth.dtype == torch.long and tuple(ids.shape) == tuple(truth.shape) == (5274,)
                     and ids.unique().numel() == 5274 and int(ids.min()) >= 0 and int(ids.max()) < 11701
                     and int(truth.min()) >= 0 and int(truth.max()) < 10,
                     'Exact full5274 development evaluator role')
            train_ids, train_y = train_data['ids'].detach().cpu(), train_data['y'].detach().cpu()
            _require(train_ids.dtype == train_y.dtype == torch.long
                     and tuple(train_ids.shape) == tuple(train_y.shape) == (580,)
                     and not bool(torch.isin(ids, train_ids).any()), 'Disjoint explicit TRAIN anchors')
            edges = train_data['edge_index'].detach().cpu()
            _require(edges.dtype == torch.long and tuple(edges.shape) == (2, 442907), 'Frozen prepared graph')
            self_mask = edges[0] == edges[1]
            _require(int(self_mask.sum()) == 11701
                     and torch.equal(edges[0, self_mask].sort().values, torch.arange(11701)),
                     'Exactly one known self record per node')
            nonself = edges[:, ~self_mask].clone()
            _require(tuple(nonself.shape) == (2, 431206), 'Exact author-supplied nonself graph')
            receipt['graph'] = dict(prepared_edge_records=442907, removed_self_records=11701,
                                    supplied_nonself_records=431206,
                                    author_preparation='unchanged; repeated and charged per seed-config')
        for seed in SEEDS:
            native = dict(seed=seed, status='pending', costs=[], probability_file=None)
            receipt['native_records'].append(native)
            seed_rows = [row for row in receipt['records'] if row['seed'] == seed]
            reference = state = full = served = None
            try:
                with _cost(native['costs'], 'native_file_hash_and_trusted_checkpoint_load'):
                    _require(_sha(paths[seed]) == release['native_state_sha256'][str(seed)], 'Authentic own-best file hash')
                    state = torch.load(paths[seed], map_location='cpu', weights_only=False)
                    _require(state['run']['seed'] == seed, 'Native seed matches root binding')
                with _cost(native['costs'], 'native_constructor_and_authentic_own_best_restore', torch, native_device):
                    reference = native_adapter.reconstruct_native_own_best(
                        state=state, train_data=train_data, polynormer=polynormer,
                        development_ids=ids,
                        development_ids_sha256=release['numeric_role_sha256']['development_ids'],
                        device=native_device, later_execution_authorized=True)
                with _cost(native['costs'], 'single_full_node_native_forward', torch, native_device):
                    receipt['native_forward_calls_attempted'] += 1
                    served = reference.probabilities()
                    receipt['native_forward_calls_completed'] += 1
                    _require(served['native_forward_calls'] == 1, 'One native forward only')
                    full = served['full_probabilities'].detach().cpu()
                with _cost(native['costs'], 'server_only_once_probability_storage_and_hash'):
                    probability_path = output/('NATIVE_PROBABILITIES_seed%d.pt' % seed)
                    native.update(probability_file=probability_path.name, probability_storage_complete=False)
                    torch.save(full, probability_path)
                    native.update(probability_storage_complete=True,
                                  probability_file_bytes=probability_path.stat().st_size,
                                  probability_file_sha256=_sha(probability_path),
                                  native_state_sha256=release['native_state_sha256'][str(seed)],
                                  native_epoch=served['native_epoch'], global_mode=served['global_mode'])
                with _cost(native['costs'], 'full5274_native_probability_readout'):
                    native['readout'] = _readout(torch, full, ids, truth)
                native['status'] = 'complete'
            except Exception as error:
                native.update(status='failed', error=_error(error))
                for row in seed_rows:
                    row.update(status='blocked', error=dict(type='NativeReferenceUnavailable',
                               message='This seed native stage failed; no replacement, refit or retry'),
                               scheduled_propagation_passes=row['configuration']['num_correction_layers']+
                                                            row['configuration']['num_smoothing_layers'],
                               completed_propagation_passes=0)
                flush()
                continue
            finally:
                reference = state = served = None
                gc.collect()
            flush()
            for row in seed_rows:
                cfg = row['configuration']
                row.update(scheduled_propagation_passes=cfg['num_correction_layers']+cfg['num_smoothing_layers'],
                           completed_propagation_passes=None)
                result = None
                try:
                    with _cost(row['costs'], 'author_graph_preparation_and_all_CPU_propagation'):
                        result = cs.correct_and_smooth_train_only(
                            probabilities=full, edge_index=nonself, train_ids=train_ids,
                            train_labels=train_y, classes=10, configuration=cfg,
                            device='cpu', later_execution_authorized=True)
                    row['completed_propagation_passes'] = result['propagation_passes']
                    with _cost(row['costs'], 'full5274_development_readout'):
                        row['readout'] = _readout(torch, result['smoothed_scores'], ids, truth)
                    row.update(status='complete', author_source_sha256=result['source_sha256'],
                               input_edge_records=result['input_edge_records'],
                               heldout_truths_supplied_to_CS=False)
                except Exception as error:
                    row.update(status='failed', error=_error(error))
                finally:
                    result = None
                    flush()
            full = None
            gc.collect()
        receipt['complete'] = (len(receipt['native_records']) == 3
            and all(row['status'] == 'complete' for row in receipt['native_records'])
            and all(row['status'] == 'complete' for row in receipt['records'])
            and receipt['native_forward_calls_attempted'] == receipt['native_forward_calls_completed'] == 3)
        if receipt['complete']:
            # Sum comparison gives exact first maximum mean over equal seed counts.
            best_id, best_sum = None, -1
            means = []
            for config_id in CONFIG_IDS:
                total = sum(row['readout']['correctcount'] for row in receipt['records']
                            if row['configuration_id'] == config_id)
                means.append(dict(configuration_id=config_id, correctcount_sum=total,
                                  mean_correctcount=total/3, mean_accuracy=total/(3*5274)))
                if total > best_sum:
                    best_id, best_sum = config_id, total
            receipt['selection'] = dict(configuration_id=best_id, correctcount_sum=best_sum,
                mean_correctcount=best_sum/3, mean_accuracy=best_sum/(3*5274),
                rule='First maximum mean full5274 correctcount over3seeds; strict-greater CS01..CS24',
                all_configuration_means=means, selection_uses_NLL_or_Brier=False)
        flush()
        cost_footer()
        return dict(output=str(output), complete=receipt['complete'],
                    selected_configuration_id=receipt['selection']['configuration_id'] if receipt['selection'] else None)
    except BaseException as error:
        receipt['family_error'] = _error(error)
        for row in receipt['native_records']:
            if row['status'] == 'pending':
                row.update(status='failed', error=_error(error))
        for row in receipt['records']:
            if row['status'] == 'pending':
                row.update(status='failed' if row['costs'] else 'blocked', error=dict(type='FamilyExecutionInterrupted',
                           message='Family failed or interrupted before this record; no automatic retry'))
        receipt['complete'], receipt['selection'] = False, None
        flush()
        cost_footer()
        raise
