"""Affected-only CPU projection qualification; root alone executes.

Reads authenticated source text and one explicitly bound tiny numerical witness.
No custody.prepare, scientific arrays/models, labels, fitting, metrics or scores.
Unchanged V2 engineering qualification is reused only by exact source proof.
"""
import argparse
import ast
import contextlib
from decimal import Decimal, localcontext
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

V2_MANIFEST = 'af15ac11349ac7d9dbc8608b4362d461195409ca6348fdfc67e7ac7b4a273b21'
V2_SEAL = '2ed8fef0d08e420e140ddcddfa2929e00849797715d16e1e0be626e9cb2ef900'
WITNESS_HASH = '4a5310a1549d231fc1a2dd65838d39030a986cea1cd1ca8d022034e59efe0700'


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def descriptor(path):
    path = Path(path)
    value = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def bind_source(source, manifest_hash, seal_hash):
    manifest_row, seal_row = descriptor(source/'MANIFEST.json'), descriptor(source/'SEAL.json')
    require(manifest_row['sha256'] == manifest_hash and seal_row['sha256'] == seal_hash,
            'Root-bound source manifest/seal hash mismatch')
    manifest = json.loads((source/'MANIFEST.json').read_text())
    seal = json.loads((source/'SEAL.json').read_text())
    require(seal['manifest']['sha256'] == manifest_hash and
            seal['manifest']['bytes'] == manifest_row['bytes'], 'Seal does not bind source manifest')
    rows = []
    for row in manifest['payload']:
        relative = Path(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts and
                relative.suffix in ('.py', '.md', '.json', '.patch'), 'Source-only payload expected')
        path = source/relative
        require(not path.is_symlink(), 'Source payload symlink forbidden')
        actual = descriptor(path)
        require(actual['sha256'] == row['sha256'] and actual['bytes'] == row['bytes'],
                'Changed source payload '+row['path'])
        rows.append(actual)
    return {'manifest': manifest_row, 'seal': seal_row, 'payload': rows}


def unchanged_proof(source, predecessor):
    before = (predecessor/'numerical.py').read_text()
    after = (source/'numerical.py').read_text()
    def segments(text):
        return {node.name: ast.get_source_segment(text, node) for node in ast.parse(text).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    old, new = segments(before), segments(after)
    require(set(new)-set(old) == {'probability_hull_qp'} and not set(old)-set(new),
            'Unexpected numerical definition added/removed')
    require({name for name in old if old[name] != new[name]} == {'projection'},
            'Unrelated numerical function changed')
    # Removing exactly the new insertion and restoring projection must recover all V2 bytes,
    # including imports, constants, counters and source outside function definitions.
    recovered = after.replace(new['probability_hull_qp']+'\n\n\n', '', 1)
    recovered = recovered.replace(new['projection'], old['projection'], 1)
    require(recovered == before, 'Numerical imports/constants/non-function source changed')
    files = []
    for name in ('custody.py', 'study.py', 'run_development.py'):
        require((source/name).read_bytes() == (predecessor/name).read_bytes(),
                'Unrelated Python module changed '+name)
        files.append(descriptor(source/name))
    proof = json.loads((source/'UNCHANGED_SOURCE_PROOF.json').read_text())
    require(proof['predecessor_manifest_sha256'] == V2_MANIFEST and
            proof['predecessor_seal_sha256'] == V2_SEAL, 'Wrong predecessor proof')
    for row in proof['unchanged_numerical_functions_and_classes']:
        require(row['name'] in old and row['name'] != 'projection' and
                hashlib.sha256(old[row['name']].encode()).hexdigest() == row['sha256'] and
                old[row['name']] == new[row['name']], 'Unchanged function proof mismatch')
    return {'all_V2_source_recovered_except_declared_projection_delta': True,
            'unchanged_modules': files, 'proof': descriptor(source/'UNCHANGED_SOURCE_PROOF.json'),
            'qualification_reused_by_source_identity_only': True,
            'no_head_fit_or_sparse_qualification_repeated': True}


def check_solution(n, p, posterior, weights):
    scale = max(float(abs(p @ p.T).max()), float(abs(p @ posterior).max()), 1e-12)
    residual = weights @ p-posterior
    raw_gradient = p @ residual
    interior = weights > n.LOWER+1e-11
    require(interior.any(), 'No free projection variable')
    primal = max(abs(float(weights.sum())-1), max(0, n.LOWER-float(weights.min())))
    rows = {}
    for name, gradient in (('raw', raw_gradient), ('scaled', raw_gradient/scale)):
        level = float(gradient[interior].mean())
        stat = float(abs(gradient[interior]-level).max())
        dual = float(n.np.maximum(level-gradient[~interior], 0).max()) if (~interior).any() else 0.0
        require(stat <= 1e-10 and dual <= 1e-10 and primal <= 1e-10,
                'Independent direct-gradient KKT check failed')
        rows[name] = {'free_stationarity': stat, 'inactive_dual_error': dual}
    return {'weights': weights.tolist(), 'prediction': (weights @ p).tolist(),
            'risk': float(residual @ residual), 'scale': scale, 'primal_error': primal, 'KKT': rows}


def decimal_two_member_reference(p, posterior, free=(1, 2)):
    """Independent analytic face minimizer at 80 decimal digits; no SVD/Gram/KKT solve."""
    with localcontext() as ctx:
        ctx.prec = 80
        pp = [[Decimal.from_float(float(x)) for x in member] for member in p]
        yy = [Decimal.from_float(float(x)) for x in posterior]
        lower = Decimal.from_float(0.0125)
        a, b = free
        fixed = [j for j in range(4) if j not in free]
        mass = Decimal(1)-len(fixed)*lower
        # w_a=t, w_b=mass-t; fixed members receive the unchanged floor.
        origin = [mass*pp[b][c]+sum((lower*pp[j][c] for j in fixed), Decimal(0)) for c in range(5)]
        direction = [pp[a][c]-pp[b][c] for c in range(5)]
        numerator = sum((direction[c]*(yy[c]-origin[c]) for c in range(5)), Decimal(0))
        denominator = sum((x*x for x in direction), Decimal(0))
        t = numerator/denominator
        weights = [lower]*4
        weights[a], weights[b] = t, mass-t
        residual = [sum((weights[j]*pp[j][c] for j in range(4)), Decimal(0))-yy[c] for c in range(5)]
        gradient = [sum((pp[j][c]*residual[c] for c in range(5)), Decimal(0)) for j in range(4)]
        require(all(w >= lower for w in weights), 'Decimal witness face is not feasible')
        require(abs(gradient[a]-gradient[b]) < Decimal('1e-60') and
                all(gradient[j] >= gradient[a] for j in fixed), 'Decimal witness face is not global KKT')
        return {'weights': [float(w) for w in weights],
                'risk': float(sum((x*x for x in residual), Decimal(0))),
                'decimal_digits': 80, 'reference': 'analytic two-member line, original represented FP64 inputs',
                'free': list(free), 'inactive_dual_verified_at_decimal_precision': True}


def witness_check(n, witness):
    p = n.np.array(witness['P'], dtype=n.np.float64)
    posterior = n.np.array(witness['posterior'], dtype=n.np.float64)
    weights, checks = n.probability_hull_qp(p, posterior)
    actual = check_solution(n, p, posterior, weights[0])
    reference = decimal_two_member_reference(p, posterior)
    difference = float(abs(weights[0]-n.np.array(reference['weights'])).max())
    require(difference <= 2e-9 and abs(actual['risk']-reference['risk']) <= 2e-14,
            'Actual witness disagrees with independent decimal constrained optimum')
    require(n.np.array_equal(n.np.flatnonzero(weights[0] > n.LOWER+1e-9), n.np.array([1, 2])),
            'Witness active face changed from independent reference')
    # Check projection wiring and stable pool on this permitted probability witness only.
    q, logq, wrapped, _ = n.projection({'posterior': posterior[None]}, p[None], n.np.log(p)[None])
    require(abs(wrapped-weights).max() <= 1e-12 and
            abs(q[0]-weights[0] @ p).max() <= 1e-12 and
            abs(n.np.exp(logq)-q).max() <= 1e-12, 'Projection/pool wiring differs')
    return {'native_node_id': 709, 'authoritative_old_batch_failure_bound': True,
            'old_single_row_faces_are_diagnostic_only': True, 'solution': actual,
            'independent_reference': reference, 'max_weight_difference': difference,
            'source_checks': checks, 'projection_pool_wiring_passed': True}


def known_and_rank_cases(n):
    base = n.np.array([0.25, 0.25, 0.25, 0.125, 0.125])
    duplicate = n.np.eye(5)[[0, 0, 1, 1]]
    cases = [
        ('identical_members_minimum_norm', n.np.tile(base, (4, 1)), n.np.eye(5)[4], n.np.full(4, 0.25)),
        ('duplicate_pairs_interior_minimum_norm', duplicate, n.np.array([0.7, 0.3, 0, 0, 0]),
         n.np.array([0.35, 0.35, 0.15, 0.15])),
        ('duplicate_pairs_active_boundary_minimum_norm', duplicate, n.np.array([0.98, 0.01, 0.01, 0, 0]),
         n.np.array([0.4875, 0.4875, 0.0125, 0.0125])),
        ('unique_vertex_active_boundary', n.np.eye(5)[:4], n.np.eye(5)[0],
         n.np.array([0.9625, 0.0125, 0.0125, 0.0125])),
        ('unique_interior', n.np.eye(5)[:4], n.np.array([0.1, 0.2, 0.3, 0.4, 0]),
         n.np.array([0.1, 0.2, 0.3, 0.4])),
    ]
    direction = n.np.array([1., -1., 0., 0., 0.])
    member_coordinate = n.np.array([-3., -1., 1., 3.])
    eps = n.np.finfo(n.np.float64).eps
    # Analytic full-face singular value sqrt(40)*delta straddles tau=64*eps.
    for label, delta in [('near_agreeing', 2.**-20), ('nearly_identical', 2.**-36),
                         ('below_cutoff_2eps', 2*eps), ('below_cutoff_8eps', 8*eps),
                         ('above_cutoff_16eps', 16*eps), ('above_cutoff_1024eps', 1024*eps)]:
        p = base[None]+delta*member_coordinate[:, None]*direction[None]
        target = base+delta*direction
        below = (40**0.5)*delta <= 64*eps
        expected = n.np.full(4, 0.25) if below else n.np.array([0.1, 0.2, 0.3, 0.4])
        cases.append((label, p, target, expected))
    p = n.np.stack([row[1] for row in cases])
    posterior = n.np.stack([row[2] for row in cases])
    weights, checks = n.probability_hull_qp(p, posterior)
    rows = []
    for (name, pp, yy, expected), actual in zip(cases, weights):
        error = float(abs(actual-expected).max())
        require(error <= 2e-9, 'Known minimum-norm/effective-rank solution differs: '+name)
        rows.append({'case': name, 'expected_weights': expected.tolist(), 'max_weight_error': error,
                     'solution': check_solution(n, pp, yy, actual)})
    return {'cases': rows, 'source_checks': checks,
            'effective_rank_is_numerical_not_exact': True,
            'discarded_directions_not_claimed_below_objective_tie': True}, p, posterior


def weak_direction_risk_limit(n):
    eps = n.np.finfo(n.np.float64).eps
    base = n.np.array([0.25, 0.25, 0.25, 0.125, 0.125])
    primary = n.np.array([1., -1., 0, 0, 0])
    weak = n.np.array([0., 0., 1., -1., 0.])
    target = base+0.05*weak
    rows = []
    for factor in (2, 8, 16, 1024):
        delta = factor*eps
        p = n.np.stack([base-0.0625*primary, base+0.0625*primary, base-delta*weak, base+delta*weak])
        weights, checks = n.probability_hull_qp(p, target)
        actual = check_solution(n, p, target, weights[0])
        # Independent analytic exact optimum maximizes w3-w2, while w0=w1 cancel primary motion.
        reference_weights = n.np.array([n.LOWER, n.LOWER, n.LOWER, 1-3*n.LOWER])
        reference = check_solution(n, p, target, reference_weights)
        gap = actual['risk']-reference['risk']
        require(-2e-14 <= gap <= 8.1e-14 and gap/actual['scale'] <= 4.1e-13,
                'Numerical rank policy exceeded disclosed local raw/scaled risk bound')
        rows.append({'delta_in_eps': factor, 'solution': actual, 'exact_analytic_boundary_reference': reference,
                     'raw_risk_gap': gap, 'scaled_risk_gap': gap/actual['scale'], 'source_checks': checks})
    return {'cases': rows, 'risk_bound_is_local_numerical_qualification_not_universal_tie_claim': True}


def independent_scipy(n):
    from scipy.optimize import minimize
    rng = n.np.random.default_rng(500709)
    rows = []
    for case in range(8):
        p = rng.uniform(0.02, 1., (4, 5)); p /= p.sum(1, keepdims=True)
        posterior = rng.uniform(0.02, 1., 5); posterior /= posterior.sum()
        weights, checks = n.probability_hull_qp(p, posterior)
        fun = lambda w: float((w @ p-posterior) @ (w @ p-posterior))
        jac = lambda w: 2*p @ (w @ p-posterior)
        reference = minimize(fun, n.np.full(4, 0.25), jac=jac, method='SLSQP',
                             bounds=[(n.LOWER, 1-3*n.LOWER)]*4,
                             constraints={'type': 'eq', 'fun': lambda w: w.sum()-1,
                                          'jac': lambda w: n.np.ones(4)},
                             options={'ftol': 1e-14, 'maxiter': 400})
        require(reference.success, 'Independent SLSQP failed: '+str(reference.message))
        error = float(abs(reference.x-weights[0]).max())
        gap = abs(fun(reference.x)-fun(weights[0]))
        require(error <= 5e-6 and gap <= 1e-10, 'Independent direct-risk SLSQP disagrees')
        rows.append({'case': case, 'solution': check_solution(n, p, posterior, weights[0]),
                     'independent_SLSQP_weights': reference.x.tolist(), 'max_weight_difference': error,
                     'absolute_risk_gap': gap, 'SLSQP_iterations': int(reference.nit), 'source_checks': checks})
    return {'seed': 500709, 'cases': rows, 'reference_uses_no_face_enumeration_or_Gram_system': True}


def batch_check(n, p, posterior, witness):
    p = n.np.concatenate((n.np.array(witness['P'])[None], p))
    posterior = n.np.concatenate((n.np.array(witness['posterior'])[None], posterior))
    expected, _ = n.probability_hull_qp(p, posterior)
    index = n.np.arange(2051) % len(p)
    weights, checks = n.probability_hull_qp(p[index], posterior[index])
    error = float(abs(weights-expected[index]).max())
    require(error <= 1e-10 and checks['solutions'] == 2051, 'Projection batch/chunk solution differs')
    return {'rows': 2051, 'native_chunk_size': 2048, 'max_weight_difference': error,
            'source_checks': checks, 'witness_and_rank_cases_cross_chunk_boundary': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--seal-sha256', required=True)
    parser.add_argument('--predecessor-source', required=True)
    parser.add_argument('--witness', required=True)
    parser.add_argument('--witness-sha256', default=WITNESS_HASH)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, predecessor = Path(args.source).resolve(), Path(args.predecessor_source).resolve()
    output = Path(args.output).resolve(); output.mkdir(exist_ok=False)
    start = time.perf_counter()
    report = {'status': 'started', 'fits': 0, 'optimizer_updates': 0, 'metric_calls': 0,
              'scored_fold_outcomes': 0, 'scientific_payload_access': False,
              'only_external_numerical_input': 'root-bound P[4,5]/posterior witness', 'tests': []}
    try:
        report['source'] = bind_source(source, args.manifest_sha256, args.seal_sha256)
        report['predecessor_source'] = bind_source(predecessor, V2_MANIFEST, V2_SEAL)
        report['unchanged_source_proof'] = unchanged_proof(source, predecessor)
        bound_witness = descriptor(args.witness)
        require(args.witness_sha256 == WITNESS_HASH and bound_witness['sha256'] == WITNESS_HASH,
                'Wrong actual failing numerical witness')
        witness = json.loads(Path(args.witness).read_text())
        require(witness['native_node_id'] == 709 and witness['original_error'] == 'QP active-face solution not found' and
                witness['source']['manifest']['sha256'] == V2_MANIFEST and
                witness['source']['seal']['sha256'] == V2_SEAL and
                witness['native_node_true_label_not_exported'] is True,
                'Witness source/scope identity differs')
        report['witness'] = bound_witness
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
            os.environ[name] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        sys.dont_write_bytecode = True
        require('custody' not in sys.modules, 'Fresh CPU qualification process required')
        sys.path.insert(0, str(source))
        spec = importlib.util.spec_from_file_location('qualified_projection_numerical', source/'numerical.py')
        n = importlib.util.module_from_spec(spec); sys.modules[spec.name] = n
        spec.loader.exec_module(n)
        n.setup()
        require(not n.torch.cuda.is_initialized(), 'Qualification initialized CUDA')
        import scipy
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture): n.np.show_config()
        report['environment'] = {'python': sys.version, 'platform': platform.platform(),
                                 'numpy': n.np.__version__, 'scipy': scipy.__version__, 'torch': str(n.torch.__version__),
                                 'BLAS_config': capture.getvalue(), 'CPU_threads': 1,
                                 'thread_environment_set_before_numerical_import': True,
                                 'torch_intraop_threads': n.torch.get_num_threads(),
                                 'torch_interop_threads': n.torch.get_num_interop_threads(), 'CUDA_initialized': False}
        require(n.torch.get_num_threads() == n.torch.get_num_interop_threads() == 1, 'CPU thread count differs')
        def run(name, call):
            before = time.perf_counter()
            try:
                details = call()
                row = {'test': name, 'status': 'passed', 'details': details}
            except Exception as error:
                row = {'test': name, 'status': 'failed', 'error': str(error), 'traceback': traceback.format_exc()}
            row['wall_seconds'] = time.perf_counter()-before
            report['tests'].append(row); write(output/(name+'.json'), row)
            return row.get('details')
        run('actual_witness_decimal_reference', lambda: witness_check(n, witness))
        # Retain known synthetic arrays only for the affected batch check.
        cache = {}
        def known():
            details, cache['p'], cache['posterior'] = known_and_rank_cases(n)
            return details
        run('known_minimum_norm_boundary_effective_rank', known)
        run('weak_direction_local_risk_limit', lambda: weak_direction_risk_limit(n))
        run('independent_probability_risk_SLSQP', lambda: independent_scipy(n))
        if cache:
            run('native_batch_chunk_identity', lambda: batch_check(n, cache['p'], cache['posterior'], witness))
        require(not n.torch.cuda.is_initialized() and n.COUNTERS['H_calls'] == 0 and
                n.COUNTERS['sparse_steps'] == 0, 'Unaffected/GPU work was executed')
        report['source_post_run'] = bind_source(source, args.manifest_sha256, args.seal_sha256)
        report['source_unchanged'] = report['source_post_run'] == report['source']
        report['predecessor_source_post_run'] = bind_source(predecessor, V2_MANIFEST, V2_SEAL)
        require(descriptor(args.witness) == bound_witness, 'Numerical witness changed')
        report['source_helper_cost_counters'] = n.COUNTERS
        report['status'] = 'passed' if len(report['tests']) == 5 and all(row['status'] == 'passed' for row in report['tests']) else 'failed'
    except Exception as error:
        report['status'] = 'failed'; report['error'] = str(error); report['traceback'] = traceback.format_exc()
    finally:
        raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['cost'] = {'wall_seconds': time.perf_counter()-start,
                          'peak_RSS_bytes': int(raw_rss if sys.platform == 'darwin' else raw_rss*1024),
                          'fits': 0, 'optimizer_updates': 0, 'metric_calls': 0, 'GPU_work': False}
        report['qualifier_source'] = descriptor(Path(__file__))
        write(output/'REPORT.json', report)
        write(output/'QUALIFICATION_RESULT.json',
              {'status': 'PASS_AFFECTED_PROJECTION_QUALIFICATION' if report['status'] == 'passed' else 'FAIL_AFFECTED_PROJECTION_QUALIFICATION',
               'source_manifest_sha256': args.manifest_sha256, 'source_seal_sha256': args.seal_sha256,
               'witness_sha256': args.witness_sha256, 'report': descriptor(output/'REPORT.json'),
               'all_tests_passed': report['status'] == 'passed', 'test_count': len(report['tests']),
               'cost': report['cost'], 'scientific_execution_authorized': False,
               'unaffected_V2_qualification_requires_separate_root_binding': True})
        print(json.dumps({'status': report['status'], 'report': str(output/'REPORT.json'), 'fits': 0, 'metric_calls': 0}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
