"""Disabled bounded CPU precision diagnostic; no qualification or model calls."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback
import zipfile

STARTED = time.monotonic()
SOURCE_RELEASED = False
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
PYTHON = PHASE / 'native_ncn_runtime_20261005_v1/.venv/bin/python'
SITE = REPO / '.venv/lib/python3.11/site-packages'
OUTPUT = 'public_path_sparse_action_derivative_cpu_precision_diagnostic_execution_root_20261006_v1'
MAX_SECONDS, MAX_RSS_BYTES, GRACE, REAP = 110, 4294967296, 2, 3
GRAD_ATOL, GRAD_RTOL = 1e-8, 1e-6


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): value.update(b)
    return value.hexdigest()


def bound(row):
    relative = Path(row['path'])
    require(relative.parts and not relative.is_absolute() and '..' not in relative.parts, 'Exact in-phase path required')
    p = PHASE / relative
    require(p.resolve().is_relative_to(PHASE), 'Path escape')
    for x in (p, *p.parents):
        if x == PHASE.parent: break
        require(not x.is_symlink(), 'Symlink refused')
    require(p.is_file() and (row.get('readonly', True) is False or p.stat().st_mode & 0o222 == 0) and sha(p) == row['sha256']
            and ('bytes' not in row or p.stat().st_size == row['bytes']), 'Exact readonly binding differs')
    return p


def load(row, name):
    p = bound(row); require(name not in sys.modules, 'Fresh helper import required')
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write(base, name, row, final=False):
    require(base.is_dir() and not base.is_symlink(), 'Owned output disappeared')
    p, temp = base / name, base / (name + '.tmp')
    require(not p.is_symlink() and not temp.exists(), 'Unsafe receipt path')
    with temp.open('x') as f:
        json.dump(row, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, p)
    if final: p.chmod(0o444)


def gamma(m, bits=24):
    value = m * 2.0 ** -bits
    require(value < 1, 'Error model operation count invalid')
    return value / (1 - value)


def selected_s_labels(np, accessor, projection):
    manifest_path, manifest, roles = accessor._projection(projection)
    label_path = accessor._verify(projection, manifest['B_labels'], frozen=True)
    with np.load(label_path, allow_pickle=False) as archive:
        require(set(archive.files) == {'ids', 'labels'}, 'B compact schema differs')
        ids = archive['ids']  # Public TRAIN IDs only; labels member is not decoded.
    require(ids.dtype == np.int64 and ids.tolist() == roles['B'], 'B IDs differ')
    positions = np.searchsorted(ids, roles['innerS']).tolist()
    values = []
    with zipfile.ZipFile(label_path) as archive, archive.open('labels.npy') as f:
        version = np.lib.format.read_magic(f)
        require(version in ((1, 0), (2, 0)), 'NPY format differs')
        header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = header(f)
        require(shape == (9797,) and not fortran and dtype == np.dtype('<i8'), 'B label header differs')
        offset = 0
        for position in positions:
            require(len(f.read((position - offset) * 8)) == (position - offset) * 8, 'Short TRAIN container')
            b = f.read(8); require(len(b) == 8, 'Short S label')
            values.append(int.from_bytes(b, 'little', signed=True)); offset = position + 1
    require(all(0 <= x < 5 for x in values), 'S class codes differ')
    return roles['innerS'], values, {'manifest_sha256': sha(manifest_path), 'B_container_sha256': sha(label_path),
        'decoded_label_values': 'exact 2449 innerS labels only; skipped other TRAIN bytes are uninterpreted',
        'A_VALID_TEST_labels_or_masks_decoded': False}


def fixed_permutation(torch, ids, labels):
    result = list(range(len(ids)))
    for label in range(5):
        positions = [i for i, x in enumerate(labels) if x == label]
        ordered = sorted(positions, key=lambda i: (hashlib.sha256(
            ('amazon-response-G0|split=0|seed=17|perm|' + str(label) + '|' + str(ids[i])).encode()).digest(), ids[i]))
        for a, b in zip(positions, ordered): result[a] = b
    return torch.tensor(result, dtype=torch.long)


def independent(torch, edges, terminals, degrees):
    """From public edges alone: rational-weight star cliques, math.fsum, FP64."""
    n = terminals.numel(); location = torch.full((24492,), -1, dtype=torch.long)
    location[terminals] = torch.arange(n)
    a, b = location[edges[0]], location[edges[1]]
    keep = (a >= 0) & (b >= 0)
    direct = torch.zeros((n, n), dtype=torch.float64); direct[a[keep], b[keep]] = 1
    keep = (a >= 0) & (b < 0)
    rows, public_u = a[keep], edges[1, keep]
    interiors, columns = torch.unique(public_u, sorted=True, return_inverse=True)
    groups = {}
    for row, interior in zip(rows.tolist(), public_u.tolist()): groups.setdefault(interior, []).append(row)
    require(all(len(v) == len(set(v)) for v in groups.values()), 'Public duplicate incidence')
    def bridge(weights):
        terms = {}
        for interior, neighbours in groups.items():
            weight = weights[interior]
            for i in neighbours:
                for j in neighbours:
                    if i != j: terms.setdefault((i, j), []).append(weight)
        matrix = torch.zeros((n, n), dtype=torch.float64); row_terms = [[] for _ in range(n)]
        for (i, j), values in terms.items():
            value = math.fsum(values); matrix[i, j] = value; row_terms[i].append(value)
        degree = torch.tensor([math.fsum(v) for v in row_terms], dtype=torch.float64)
        return torch.diag(degree) - matrix
    weights = {u: 1.0 / int(degrees[u]) for u in groups}
    B = bridge(weights); D = torch.diag(direct.sum(1)) - direct
    denominator = math.fsum((1., float(direct.sum(1).max()), float(B.diagonal().max())))
    # Exact original FP32 dense construction is a separate diagnostic path.
    incidence = torch.zeros((n, interiors.numel()), dtype=torch.float32); incidence[rows, columns] = 1
    C32 = (incidence / degrees[interiors].float()[None, :]) @ incidence.T
    C32.diagonal().zero_(); A32 = direct.float()
    dD32, dB32 = A32.sum(1), C32.sum(1)
    N32 = 1.0 + dD32.max() + dB32.max()
    return D, B, denominator, torch.diag(dD32) - A32, torch.diag(dB32) - C32, N32, bridge, groups


def cast_frozen(torch, adapter, port, lap):
    d, b = lap.direct, lap.bridge
    direct = port.SparseLaplacian(d.row, d.column, d.weight.double(), d.degree.double())
    bridge = adapter.StarBridgeLaplacian(b.row, b.column, b.inverse_full_degree.double(), b.terminal_neighbours, b.degree.double())
    return adapter.DirectAndBridgeLaplacian(direct, bridge, lap.permutation, lap.inverse_permutation, lap.normalization.double())


def sparse_mass(torch, lap, x):
    """Absolute expanded DAG terms, including algebraically cancelled self terms."""
    d, b = lap.direct, lap.bridge; z = x.abs()[lap.inverse_permutation]
    interior = torch.zeros((b.terminal_neighbours.numel(), 4), dtype=torch.float64).index_add(0, b.column, z[b.row])
    messages = b.inverse_full_degree.double()[b.column, None] * (
        b.terminal_neighbours[b.column, None].double() * z[b.row] + interior[b.column])
    bridge = torch.zeros_like(x).index_add(0, b.row, messages)[lap.permutation]
    direct = d.degree.double()[:, None] * x.abs() + torch.zeros_like(x).index_add(
        0, d.row, d.weight.double()[:, None].abs() * x.abs()[d.column])
    return (direct + bridge) / float(lap.normalization)


def stats(torch, actual, expected):
    error = (actual.double() - expected).abs()
    return {'max_abs_error': float(error.max()), 'rms_error': float(error.square().mean().sqrt())}


def report(torch, lap, lap64, frozen, reference, frozen_matrix, dense, q, v, label, node_ids):
    action = lap @ q
    gradient, = torch.autograd.grad((action * v).sum(), q)
    dense_action = dense @ q.detach()
    dense_q = q.detach().clone().requires_grad_(True)
    dense_autograd, = torch.autograd.grad((dense @ dense_q * v).sum(), dense_q)
    dense_gradient = dense.T @ v  # Exact expected path from the failed oracle.
    d, b = lap.direct, lap.bridge; n = q.shape[0]
    dmax = int(torch.bincount(d.row, minlength=n).max())
    tmax = int(torch.bincount(b.row, minlength=n).max()); kmax = int(b.terminal_neighbours.max())
    m = dmax + tmax + kmax + 8  # Conservative sum of both DAG branches, final addition and division.
    rows = []
    for kind, x, actual, dense_actual in (('action', q.detach(), action.detach(), dense_action), ('VJP', v, gradient, dense_gradient)):
        x64 = x.double(); R = reference if kind == 'action' else reference.T
        F = frozen_matrix if kind == 'action' else frozen_matrix.T
        A = dense.double() if kind == 'action' else dense.double().T
        ref = R @ x64
        if kind == 'action':
            frozen_value = frozen @ x64; source64_value = lap64 @ x64
        else:
            leaf = q.detach().double().requires_grad_(True)
            frozen_value, = torch.autograd.grad((frozen @ leaf * x64).sum(), leaf)
            leaf = q.detach().double().requires_grad_(True)
            source64_value, = torch.autograd.grad((lap64 @ leaf * x64).sum(), leaf)
        mass = sparse_mass(torch, lap, x64)
        # gamma bound is fixed analytically from graph reduction lengths, not fitted to output differences.
        sparse_round = gamma(m) * mass / (1 - gamma(m, 53))
        dense_mass = A.abs() @ x64.abs()
        dense_round = gamma(2 * n + 2) * dense_mass / (1 - gamma(2 * n + 2, 53))
        reference_mass = R.abs() @ x64.abs()
        ref_round = gamma(4 * n + 16, 53) * reference_mass / (1 - gamma(4 * n + 16, 53)) ** 2
        sparse_coeff = (F - R).abs() @ x64.abs(); dense_coeff = (A - R).abs() @ x64.abs()
        sparse_bound = sparse_round + sparse_coeff + ref_round
        dense_bound = dense_round + dense_coeff + ref_round
        pair_bound = sparse_round + dense_round + (F - A).abs() @ x64.abs() + 2 * ref_round
        s64, d64 = actual.double(), dense_actual.detach().double()
        difference = (s64 - d64).abs(); tolerance = GRAD_ATOL + GRAD_RTOL * d64.abs()
        bad = difference > tolerance
        ratio = difference / tolerance
        largest = ratio.flatten().argsort(descending=True)[:10].tolist()
        offending = torch.nonzero(bad.flatten()).flatten().tolist()
        order = sorted(set(largest + offending), key=lambda index: float(ratio.flatten()[index]), reverse=True)
        coordinates = []
        for index in order:
            i, c = divmod(index, 4)
            coordinates.append({'row': i, 'public_node_id': int(node_ids[i]), 'column': c,
                'sparse_FP32': float(s64[i, c]), 'dense_FP32': float(d64[i, c]), 'reference_FP64': float(ref[i, c]),
                'frozen_sparse_FP64': float(frozen_value[i, c]), 'production_sparse_FP64': float(source64_value[i, c]),
                'abs_difference': float(difference[i, c]), 'original_atol_plus_rtol': float(tolerance[i, c]),
                'original_tolerance_ratio': float(ratio[i, c]), 'original_coordinate_gate_fails': bool(bad[i, c]),
                'sparse_analytic_abs_bound': float(sparse_bound[i, c]), 'dense_analytic_abs_bound': float(dense_bound[i, c]),
                'sparse_dense_analytic_abs_bound': float(pair_bound[i, c]),
                'reference_abs_mass': float(reference_mass[i, c]), 'sparse_internal_abs_mass': float(mass[i, c]),
                'reference_cancellation_ratio': None if float(ref[i, c]) == 0 else float(reference_mass[i, c] / ref[i, c].abs()),
                'sparse_internal_cancellation_ratio': None if float(ref[i, c]) == 0 else float(mass[i, c] / ref[i, c].abs())})
        rows.append({'kind': kind, 'probe': label, 'original_gate_failed_coordinates': int(bad.sum()),
            'max_original_gate_ratio': float(ratio.max()), 'sparse_dense': stats(torch, actual, d64),
            'sparse_reference': stats(torch, actual, ref), 'dense_reference': stats(torch, dense_actual, ref),
            'production_sparse_FP64_reference': stats(torch, source64_value, ref),
            'frozen_sparse_FP64_frozen_coefficient_matrix': stats(torch, frozen_value, F @ x64),
            'frozen_sparse_FP64_reference': stats(torch, frozen_value, ref),
            'sparse_errors_outside_analytic_bound': int(((s64 - ref).abs() > sparse_bound).sum()),
            'dense_errors_outside_analytic_bound': int(((d64 - ref).abs() > dense_bound).sum()),
            'sparse_dense_errors_outside_analytic_bound': int((difference > pair_bound).sum()),
            'max_sparse_coefficient_error_bound': float(sparse_coeff.max()), 'max_dense_coefficient_error_bound': float(dense_coeff.max()),
            'dense_autograd_vs_explicit_transpose': stats(torch, dense_autograd, dense_gradient.double()) if kind == 'VJP' else None,
            'max_reference_FP64_rounding_bound': float(ref_round.max()), 'reduction_limits': {'direct_degree': dmax, 'terminal_star_incidences': tmax, 'star_terminal_count': kmax,
            'sparse_operation_budget': m, 'dense_operation_budget': 2 * n + 2}, 'all_original_gate_offending_and_largest_ratio_coordinates': coordinates})
    return rows


def numerical(scope, base):
    def timeout(number, frame): raise TimeoutError('CPU diagnostic wall cap')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(MAX_SECONDS - 5)
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_SECONDS, MAX_SECONDS + 1))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(SITE))
    import numpy as np
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    require(str(Path(torch.__file__).resolve()).startswith(str(SITE) + '/') and torch.__version__ == scope['expected_Torch_version']
        and torch.get_default_dtype() == torch.float32 and not torch.cuda.is_initialized(), 'Exact CPU-only Torch required')
    modules = {k: load(v, '_precision_' + k) for k, v in scope['numerical_sources'].items()}
    accessor, op, port, adapter = (modules[k] for k in ('accessor', 'operator', 'port', 'adapter'))
    for row in scope['input_bindings']: bound(row)
    projection = PHASE / scope['public_b_relative']
    ids, labels, provenance = selected_s_labels(np, accessor, projection)
    with np.load(accessor._verify(PHASE, accessor.PUBLIC_GRAPH), allow_pickle=False) as archive:
        require(set(archive.files) == {'features', 'edge_index', 'train_mask', 'val_mask', 'test_mask'}, 'Public graph schema differs')
        raw_edges = archive['edge_index']  # No features or masks decoded.
    edges_np, preprocessing = accessor._native_edges(raw_edges, PHASE)
    require(preprocessing['edge_logical_sha256'] == scope['edge_logical_sha256'], 'Original public graph recipe differs')
    edge = torch.from_numpy(edges_np); S = torch.tensor(ids); yS = torch.tensor(labels)
    require(torch.bincount(yS, minlength=5).tolist() == [631, 906, 578, 231, 103], 'Exact S classes required')
    permutation = fixed_permutation(torch, ids, labels)
    kwargs = {'node_count': 24492, 'affinity_permutation': permutation,
        'protocol_sha256': scope['protocol_sha256'], 'census_summary_sha256': scope['census_summary_sha256']}
    pins = scope['numerical_sources']
    banks = adapter._prepare_bridge_pair_banks(op, port, pins['operator']['sha256'], pins['port']['sha256'], S, yS, edge, dtype=torch.float32, **kwargs)
    banks64 = adapter._prepare_bridge_pair_banks(op, port, pins['operator']['sha256'], pins['port']['sha256'], S, yS, edge, dtype=torch.float64, **kwargs)
    edges = edge[:, edge[0] != edge[1]]; degrees = torch.bincount(edges[0], minlength=24492)
    result = {'schema': 'public_sparse_action_precision_diagnostic_result_v1', 'status': 'RUNNING_DIAGNOSTIC', 'pairs': [],
        'input_provenance': provenance, 'edge_preprocessing': preprocessing, 'CPU_only': True, 'native_model_callbacks': 0,
        'held_label_access': False, 'features_or_masks_decoded': False, 'fits': 0, 'production_sources_or_tolerances_changed': False,
        'qualification_or_training_admission': False, 'original_atol': GRAD_ATOL, 'original_rtol': GRAD_RTOL,
        'probe_limit': 'CPU trigonometric FP32 probes; original GPU probe tensors were not saved, so not claimed bit-identical to that failure.'}
    write(base, 'WORKER_RESULT.json', result)
    for live, control, live64, control64 in zip(banks.live, banks.bridge_permuted, banks64.live, banks64.bridge_permuted):
        require(time.monotonic() - STARTED < MAX_SECONDS and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < MAX_RSS_BYTES, 'CPU diagnostic cap')
        terminals = S[live.nodes]
        D, B, N, D32, B32, N32, bridge, groups = independent(torch, edges, terminals, degrees)
        n = terminals.numel(); pair = {'classes': list(live.classes), 'nodes': n, 'banks': []}
        for name, lap, lap64 in (('live', live.laplacian, live64.laplacian), ('bridge_permuted', control.laplacian, control64.laplacian)):
            p = lap.permutation
            reference = (D + B[p][:, p]) / N; dense = (D32 + B32[p][:, p]) / N32
            require(torch.equal(lap.permutation, lap64.permutation), 'Same fixed control required')
            interiors = sorted(groups)
            require(len(interiors) == lap.bridge.inverse_full_degree.numel(), 'Independent public incidence differs')
            weights = {u: float(lap.bridge.inverse_full_degree[j]) for j, u in enumerate(interiors)}
            frozen_matrix = (D + bridge(weights)[p][:, p]) / float(lap.normalization)
            frozen = cast_frozen(torch, adapter, port, lap)
            bank = {'name': name, 'normalization_sparse_FP32': float(lap.normalization), 'normalization_dense_FP32': float(N32),
                'normalization_independent_FP64': N, 'normalization_production_FP64': float(lap64.normalization),
                'dense_FP32_reference_max_coefficient_error': float((dense.double() - reference).abs().max()),
                'frozen_sparse_FP32_coefficients_reference_max_error': float((frozen_matrix - reference).abs().max()), 'probes': []}
            for probe in ('oracle_sin_cos', 'uniform_cancellation'):
                raw = torch.arange(n * 4, dtype=torch.float32).reshape(n, 4)
                q = torch.sin(raw / 97.0) if probe == 'oracle_sin_cos' else torch.full((n, 4), .25)
                v = torch.cos(raw / 89.0) if probe == 'oracle_sin_cos' else torch.full((n, 4), .25)
                bank['probes'].extend(report(torch, lap, lap64, frozen, reference, frozen_matrix, dense, q.requires_grad_(True), v, probe, terminals))
            pair['banks'].append(bank)
        result['pairs'].append(pair); result['elapsed_seconds'] = time.monotonic() - STARTED
        write(base, 'WORKER_RESULT.json', result)
        del reference, dense, frozen_matrix, D, B, D32, B32
    require(len(result['pairs']) == 10 and not torch.cuda.is_initialized(), 'Complete CPU-only diagnostic required')
    result.update(status='DIAGNOSTIC_COMPLETED_NOT_QUALIFICATION', elapsed_seconds=time.monotonic() - STARTED,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    require(result['elapsed_seconds'] < MAX_SECONDS and result['peak_rss_bytes'] < MAX_RSS_BYTES, 'Final CPU cap')
    signal.alarm(0); write(base, 'WORKER_RESULT.json', result, True)


def supervised(args, scope, base):
    require(not base.exists() and not base.is_symlink(), 'Once-only diagnostic output already claimed')
    base.mkdir(mode=0o700)
    helper = load(scope['ownership_helper'], '_precision_stdlib_ownership')
    claim = {'supervisor_pid': os.getpid(), 'supervisor_start_ticks': helper.identity(os.getpid())['starttime_ticks'],
        'token': os.urandom(16).hex(), 'scope_sha256': args.scope_sha256, 'device_inode': [base.stat().st_dev, base.stat().st_ino]}
    write(base, 'CLAIM.json', claim, True)
    argv = [str(PYTHON), '-B', str(Path(__file__).resolve()), '--execute-authorized', '--worker', '--scope', args.scope,
        '--scope-sha256', args.scope_sha256, '--token', claim['token']]
    receipt = {'status': 'FAIL_CPU_DIAGNOSTIC', 'launch_attempts': 1, 'scope_sha256': args.scope_sha256, 'automatic_retry': False,
        'max_child_seconds': MAX_SECONDS, 'max_combined_RSS_bytes': MAX_RSS_BYTES, 'termination_signals': [], 'cleanup_errors': []}
    with (base / 'worker.stdout.log').open('xb') as out, (base / 'worker.stderr.log').open('xb') as err:
        child = subprocess.Popen(argv, cwd=REPO, stdout=out, stderr=err, stdin=subprocess.DEVNULL, start_new_session=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    identity = helper.identity(child.pid); require(identity is not None, 'Owned child disappeared before registration')
    start = identity['starttime_ticks']; receipt.update(child_pid=child.pid, child_start_ticks=start, child_argv=argv)
    begun = time.monotonic(); term_at = None; code = 1
    try:
        while True:
            pid, status, usage = os.wait4(child.pid, os.WNOHANG)
            if pid:
                child.returncode = os.waitstatus_to_exitcode(status)
                receipt.update(child_exit_code=child.returncode, wait4_closed=True, wait4_peak_rss_bytes=usage.ru_maxrss * 1024,
                    wait4_user_seconds=usage.ru_utime, wait4_system_seconds=usage.ru_stime, wall_seconds=time.monotonic() - begun)
                break
            try:
                fields = (Path('/proc') / str(child.pid) / 'status').read_text().splitlines()
                rss = int(next(x for x in fields if x.startswith('VmRSS:')).split()[1]) * 1024
            except (FileNotFoundError, StopIteration): rss = 0
            elapsed = time.monotonic() - begun; combined = rss + resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            if term_at is None and (elapsed >= MAX_SECONDS or combined >= MAX_RSS_BYTES):
                helper.physical(child.pid, argv, parent=os.getpid(), start=start)
                os.killpg(child.pid, signal.SIGTERM); term_at = time.monotonic()
                receipt['termination_signals'].append({'signal': 'SIGTERM', 'elapsed_seconds': elapsed, 'combined_RSS_bytes': combined})
            elif term_at is not None and time.monotonic() - term_at >= GRACE and not any(x['signal'] == 'SIGKILL' for x in receipt['termination_signals']):
                helper.physical(child.pid, argv, parent=os.getpid(), start=start); os.killpg(child.pid, signal.SIGKILL)
                receipt['termination_signals'].append({'signal': 'SIGKILL', 'elapsed_seconds': elapsed})
            require(term_at is None or time.monotonic() - term_at < GRACE + REAP, 'Bounded owned reap failed')
            time.sleep(.1)
        receipt['combined_RSS_upper_bound_bytes'] = receipt['wait4_peak_rss_bytes'] + resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if receipt['child_exit_code'] == 0 and not receipt['termination_signals']:
            result = json.loads((base / 'WORKER_RESULT.json').read_text())
            require(result['status'] == 'DIAGNOSTIC_COMPLETED_NOT_QUALIFICATION' and len(result['pairs']) == 10
                and receipt['combined_RSS_upper_bound_bytes'] < MAX_RSS_BYTES and receipt['wall_seconds'] < MAX_SECONDS, 'Diagnostic completion/caps differ')
            receipt['status'] = 'CPU_DIAGNOSTIC_COMPLETED_OWNED_REAPED'; code = 0
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        if not receipt.get('wait4_closed'):
            try:
                helper.physical(child.pid, argv, parent=os.getpid(), start=start); os.killpg(child.pid, signal.SIGKILL)
                deadline = time.monotonic() + REAP
                while True:
                    pid, status, usage = os.wait4(child.pid, os.WNOHANG)
                    if pid:
                        child.returncode = os.waitstatus_to_exitcode(status)
                        receipt.update(wait4_closed=True, child_exit_code=child.returncode, wait4_peak_rss_bytes=usage.ru_maxrss * 1024)
                        break
                    require(time.monotonic() < deadline, 'Cleanup reap deadline exceeded')
                    time.sleep(.05)
            except BaseException as error: receipt['cleanup_errors'].append(str(error))
        for name in ('WORKER_RESULT.json', 'worker.stdout.log', 'worker.stderr.log'):
            p = base / name
            if p.is_file(): p.chmod(0o444); receipt.setdefault('preserved_files', []).append({'path':str(p.relative_to(PHASE)), 'bytes':p.stat().st_size, 'sha256':sha(p)})
        write(base, 'TERMINAL.json', receipt, True)
    print(json.dumps({'status':receipt['status'], 'terminal':str(base / 'TERMINAL.json'), 'automatic_retry':False})); return code


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--execute-authorized', action='store_true')
    parser.add_argument('--scope'); parser.add_argument('--scope-sha256'); parser.add_argument('--worker', action='store_true'); parser.add_argument('--token')
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({'status':'DISABLED_CPU_PRECISION_DIAGNOSTIC', 'SOURCE_RELEASED':False})); return 0
    require(socket.gethostname() == 'anogena-2-0' and Path.cwd().resolve() == REPO and Path(sys.executable).absolute() == PYTHON
        and sys.dont_write_bytecode and not any(x == 'torch' or x.startswith('torch.') for x in sys.modules), 'Exact fresh ordinary host/runtime/-B required')
    scope_path = Path(args.scope).absolute(); require(scope_path.is_relative_to(PHASE), 'Scope outside phase')
    scope = json.loads(bound({'path':str(scope_path.relative_to(PHASE)), 'sha256':args.scope_sha256}).read_text())
    require(scope['root_CPU_diagnostic_approved'] is True and scope['max_seconds'] == MAX_SECONDS and scope['max_combined_RSS_bytes'] == MAX_RSS_BYTES
        and scope['automatic_retry'] is False and scope['native_model_callbacks'] == 0 and scope['held_labels_authorized'] is False
        and bound(scope['source']).resolve() == Path(__file__).resolve(), 'Exact diagnostic root scope required')
    base = PHASE / OUTPUT
    if not args.worker: return supervised(args, scope, base)
    claim = json.loads((base / 'CLAIM.json').read_text())
    require(claim['token'] == args.token and claim['scope_sha256'] == args.scope_sha256 and os.getppid() == claim['supervisor_pid']
        and [base.stat().st_dev, base.stat().st_ino] == claim['device_inode'], 'Owned worker claim differs')
    helper = load(scope['ownership_helper'], '_precision_worker_stdlib_ownership')
    require(helper.identity(os.getppid())['starttime_ticks'] == claim['supervisor_start_ticks'], 'Owned parent start differs')
    try: numerical(scope, base); return 0
    except BaseException as error:
        p = base / 'WORKER_RESULT.json'; row = json.loads(p.read_text()) if p.exists() else {'pairs':[]}
        row.update(status='FAIL_CPU_DIAGNOSTIC', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic() - STARTED, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        write(base, 'WORKER_RESULT.json', row, True); return 1


if __name__ == '__main__': raise SystemExit(main())
