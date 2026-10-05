"""Standalone synthetic-only engineering qualification; never reads scientific inputs.

Run only after external root authorization, in a fresh CPU process. This script
binds the explicitly selected source manifest/seal and all source payload files.
It does not import original V6 models/loaders or call any scientific custody/data
loader. One discarded tiny head fit is charged separately from development.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def descriptor(path):
    p = Path(path)
    b = p.read_bytes()
    return {'path': str(p.resolve()), 'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}


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
            seal['manifest']['bytes'] == manifest_row['bytes'], 'Source seal does not bind manifest')
    rows = []
    for row in manifest['payload']:
        relative = Path(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts and
                relative.suffix in ('.py', '.md', '.json', '.patch'), 'Source-only payload expected')
        path = source/relative
        require(not path.is_symlink(), 'Source payload symlink forbidden')
        actual = descriptor(path)
        require(actual['sha256'] == row['sha256'] and actual['bytes'] == row['bytes'],
                'Changed source file: '+row['path'])
        rows.append(actual)
    require({'numerical.py','custody.py'}.issubset({r['path'] for r in manifest['payload']}),
            'Required standalone numerical source modules absent')
    return {'manifest': manifest_row, 'seal': seal_row, 'payload': rows}


def qp_against_scipy(n):
    from scipy.optimize import minimize
    rng = n.np.random.default_rng(71303)
    matrices, linear = [], []
    for case in range(5):
        x = rng.normal(size=(4, 4))
        matrices.append(x @ x.T + 0.25*n.np.eye(4))
        linear.append(n.np.zeros(4) if case == 0 else 0.15*rng.normal(size=4))
    matrices, linear = n.np.stack(matrices), n.np.stack(linear)
    actual, diagnostics = n.simplex_qp(matrices, linear)
    rows = []
    for case, (a, b, w) in enumerate(zip(matrices, linear, actual)):
        fun = lambda value: float(value @ a @ value - 2*b @ value)
        jac = lambda value: 2*(a @ value-b)
        result = minimize(fun, n.np.full(4, 0.25), jac=jac, method='SLSQP',
                          bounds=[(n.LOWER, 1-3*n.LOWER)]*4,
                          constraints={'type':'eq', 'fun':lambda value: value.sum()-1,
                                       'jac':lambda value: n.np.ones(4)},
                          options={'ftol':1e-12, 'maxiter':300})
        require(result.success, 'Independent SciPy optimizer failed: '+str(result.message))
        difference = float(n.np.max(n.np.abs(w-result.x)))
        gap = abs(fun(w)-fun(result.x))
        require(difference <= 3e-6 and gap <= 1e-8, 'QP disagrees with independent SciPy optimum')
        rows.append({'case':case, 'source_weights':w.tolist(), 'scipy_weights':result.x.tolist(),
                     'max_weight_difference':difference, 'objective_difference':gap,
                     'scipy_iterations':int(result.nit)})
    return {'seed':71303, 'cases':rows, 'source_KKT':diagnostics}


def tiny_spd(n):
    from scipy.optimize import minimize
    diagonal = n.np.array([1., 2., 3., 4.])
    expected = (1/diagonal)/(1/diagonal).sum()
    factors = (1e-8, 1e-12, 1e-14, 1e-16, 1e-20)
    matrices = n.np.stack([factor*n.np.diag(diagonal) for factor in factors])
    weights, diagnostics = n.simplex_qp(matrices)
    error = float(n.np.max(n.np.abs(weights-expected)))
    require(error <= 2e-8, 'Tiny scaled SPD inverse-diagonal solution changed')
    independent = []
    for factor, a, actual in zip(factors, matrices, weights):
        # Independent reference removes the arbitrary scale entirely before optimization.
        reference_matrix = a/n.np.max(n.np.abs(a))
        result = minimize(lambda w: float(w @ reference_matrix @ w), n.np.full(4,0.25),
                          jac=lambda w: 2*reference_matrix @ w, method='SLSQP',
                          bounds=[(n.LOWER,1-3*n.LOWER)]*4,
                          constraints={'type':'eq','fun':lambda w:w.sum()-1,
                                       'jac':lambda w:n.np.ones(4)},
                          options={'ftol':1e-12,'maxiter':300})
        require(result.success and n.np.max(n.np.abs(result.x-actual)) <= 3e-6,
                'Tiny SPD differs from independent scale-normalized SciPy reference')
        independent.append({'factor':factor,'weights':result.x.tolist(),
                            'source_max_weight_difference':float(n.np.max(n.np.abs(result.x-actual)))})
    return {'scales':list(factors), 'known_weights':expected.tolist(),
            'weights':weights.tolist(), 'max_error':error,
            'independent_scale_normalized_Scipy':independent,'source_KKT':diagnostics}


def degenerate_projection(n):
    p0 = n.np.array([0.1, 0.2, 0.25, 0.15, 0.3])
    identical = n.np.tile(p0, (4, 1))
    duplicated = n.np.eye(5)[[0, 0, 1, 1]]
    posterior = n.np.array([[1.,0,0,0,0], [0.7,0.3,0,0,0], [0.98,0.01,0.01,0,0]])
    p = n.np.stack([identical, duplicated, duplicated])
    expected = n.np.array([[0.25]*4, [0.35,0.35,0.15,0.15],
                          [0.4875,0.4875,0.0125,0.0125]])
    a = n.np.einsum('nmc,nkc->nmk', p, p)
    b = n.np.einsum('nmc,nc->nm', p, posterior)
    weights, diagnostics = n.simplex_qp(a, b, singular=True)
    error = float(n.np.max(n.np.abs(weights-expected)))
    require(error <= 2e-10, 'Identical/duplicate-member minimum-norm solution changed')
    projected = n.np.einsum('nm,nmc->nc', weights, p)
    require(n.np.max(n.np.abs(projected[0]-p0)) <= 1e-12 and
            n.np.max(n.np.abs(projected[1]-posterior[1])) <= 1e-12 and
            n.np.max(n.np.abs(projected[2]-n.np.array([0.975,0.025,0,0,0]))) <= 1e-12,
            'Known hull probability projection changed')
    return {'cases':['identical members','two duplicate pairs, interior','two duplicate pairs, dense boundary'],
            'weights':weights.tolist(), 'projected_probability':projected.tolist(),
            'max_weight_error':error, 'source_KKT':diagnostics}


def gram_reconstruction(n):
    rng = n.np.random.default_rng(8821)
    p = rng.uniform(0.1, 1., size=(9, 4, 5))
    p /= p.sum(-1, keepdims=True)
    y = rng.integers(0, 5, size=9)
    e = p-n.np.eye(5)[y, None, :]
    direct = n.np.einsum('nmc,nkc->nmk', e, e)
    reconstructed = n.reconstruct(n.np.sum(e*e, axis=-1), n.pairwise(p))
    error = float(n.np.max(n.np.abs(direct-reconstructed)))
    require(error <= 2e-15, 'Uncentered Gram reconstruction identity failed')
    return {'seed':8821, 'cases':9, 'max_absolute_error':error}


def stable_skip(n):
    from scipy.special import logsumexp
    z = n.np.full((4, 3, 5), -10000., dtype=n.np.float32)
    z[:, :, 0] = 10000.
    z[0, 1, 1] = 10000.
    z[1, 1, 1] = 9999.
    z[:, 2, :] = 0.
    a = n.arrays(n.torch.from_numpy(z))
    require(n.np.array_equal(a['native_class'],a['native_scoring_class']),
            'M4 native scoring must retain FP32 probability-mean classes')
    w = n.np.tile(n.np.array([0.0125,0.1375,0.3,0.55]), (3, 1))
    actual_log = n.log_pool(a['logp'], w)
    independent_member = z.astype(n.np.float64)-logsumexp(z.astype(n.np.float64), axis=-1, keepdims=True)
    expected_log = logsumexp(independent_member.transpose(1,0,2)+n.np.log(w)[:,:,None], axis=1)
    error = float(n.np.max(n.np.abs(actual_log-expected_log)))
    require(n.np.isfinite(actual_log).all() and error <= 1e-10, 'Stable extreme log mixture failed')
    head = n.ResidualHead(3, 4, 337)
    with n.torch.no_grad():
        q = head(n.torch.zeros(3, 3), n.torch.from_numpy(actual_log)).numpy()
    expected_q = n.np.einsum('nm,nmc->nc', w, a['p'])
    discrepancy = float(n.np.max(n.np.abs(q-expected_q)))
    require(discrepancy <= 2e-14, 'Zero residual does not recover analytic skip')
    require(float(n.np.exp(actual_log[0,4])) == 0.0, 'Extreme case does not exercise probability underflow')
    with n.torch.no_grad():
        head.c[4] = 30000.
        revived = head(n.torch.zeros(3, 3), n.torch.from_numpy(actual_log)).numpy()
    require(revived[0,4] > 0.999, 'Finite stable log skip lost an underflowed class')
    native_expected = logsumexp(independent_member, axis=0)-math.log(4)
    require(n.np.max(n.np.abs(native_expected-a['native_log'])) <= 1e-10,
            'Native uncontaminated stable mixture changed')
    return {'member_logit_extrema':[-10000,10000], 'max_log_mixture_error':error,
            'zero_residual_max_probability_error':discrepancy,
            'underflowed_class_revived_probability':float(revived[0,4]),
            'native_log_mixture_is_uncontaminated':True}


def tiny_gap_single(n):
    # A raw FP32 logit gap survives argmax but is below a FP32 probability ULP.
    z = n.torch.zeros((1,1,5), dtype=n.torch.float32, device='cpu')
    z[0,0,1] = 1e-8
    a = n.arrays(z)
    require('native_scoring_class' in a, 'Corrected source must separate native scoring and graph classes')
    require(int(a['native_class'][0]) == 0 and int(a['native_scoring_class'][0]) == 1,
            'Tiny-gap M1 graph/scoring separation failed')
    score = n.metrics(a['native'], n.np.array([1]), native_class=a['native_scoring_class'],
                      native_log=a['native_log'])
    require(score['accuracy'] == 1.0, 'Native M1 raw-logit scoring was not preserved')
    return {'FP32_logit_gap':float(z[0,0,1]), 'graph_FP32_probability_class':0,
            'native_raw_logit_scoring_class':1, 'raw_native_accuracy':score['accuracy'],
            'probability_argmax_accuracy_would_be':0.0,
            'scoring_graph_discrepancy_count':a['native_scoring_vs_graph_discrepancies']}


def head_counts(n):
    rows = []
    for d, width, expected in ((69,44,3650), (94,32,3675), (17,156,3678)):
        head = n.ResidualHead(d, width, 21)
        count = sum(p.numel() for p in head.parameters())
        require(count == expected, 'Head parameter count differs')
        rows.append({'features':d,'width':width,'parameters':count})
    return {'heads':rows,'optimizer_updates':0}


def tiny_discarded_fit(n):
    y = n.np.arange(30, dtype=n.np.int64) % 5
    x = n.np.eye(5)[y]
    skip = n.np.full((30,5), -math.log(5))
    head, mu, sd, diagnostics = n.fit_head(x, skip, y, width=4, seed=291, regularizer=1e-4)
    trajectory = diagnostics['trajectory']
    minimum = min(row['objective'] for row in trajectory)
    first_minimum = next(row['step'] for row in trajectory if row['objective'] == minimum)
    require(diagnostics['updates'] == 150 and len(trajectory) == 151 and
            diagnostics['retained_training_step'] == first_minimum and
            diagnostics['selected']['objective'] <= diagnostics['initial']['objective'],
            'Best-training-objective/no-op retention rule failed')
    initial = n.ResidualHead(5, 4, 291)
    q = n.serve_head(head, mu, sd, x, skip)
    brier = float(n.np.mean(n.np.sum((q-n.np.eye(5)[y])**2,axis=1)))
    count = sum(p.numel() for p in head.parameters())
    penalty = sum(float(((p-i)**2).sum().detach()) for p,i in zip(head.parameters(),initial.parameters()))/count
    objective = brier+1e-4*penalty
    require(abs(objective-minimum) <= 1e-12 and brier <= 0.8+1e-12,
            'Returned head is not the selected training iterate')
    return {'kind':'discarded synthetic engineering fit, no scientific outcome',
            'fits':1,'discarded_optimizer_updates':150,'parameters':count,
            'recomputed_selected_objective':objective,'source_diagnostics':diagnostics}


def sparse_and_exclusion(n):
    from scipy import sparse
    original_n = n.N
    n.N = 7  # Qualification process only; restored in finally, no source mutation.
    try:
        pairs = n.np.array([[0,1],[1,2],[2,3],[4,5]], dtype=n.np.int64)
        edge = n.np.concatenate((pairs.T,pairs[:,::-1].T,
                                n.np.vstack((n.np.arange(7),n.np.arange(7)))),axis=1)
        t, degree, graph = n.graph(edge,n.np.zeros(7,dtype=n.np.int64))
        dense = t.toarray()
        alpha = 0.8
        h = alpha**20*n.np.linalg.matrix_power(dense,20)
        for k in range(20):
            h += (1-alpha)*alpha**k*n.np.linalg.matrix_power(dense,k)
        x = n.np.arange(21,dtype=n.np.float64).reshape(7,3)/21
        direct = h @ x
        actual = n.diffuse(t,x)
        diffusion_error = float(n.np.max(n.np.abs(actual-direct)))
        require(diffusion_error <= 1e-13, 'Restart diffusion disagrees with finite-power expression')
        q = n.np.tile(n.np.array([0.1,0.2,0.3,0.15,0.25]),(7,1))
        anchors, y = n.np.array([0,3,6]),n.np.array([0,2,4])
        seed = n.np.zeros((7,5));seed[anchors] = n.np.eye(5)[y]-q[anchors]
        corrected = q+h @ seed
        require(corrected.min() >= -1e-14 and n.np.max(n.np.abs(corrected.sum(1)-1)) <= 1e-13,
                'Chosen independent C&S case requires no simplex clipping')
        smooth_seed = corrected.copy();smooth_seed[anchors] = n.np.eye(5)[y]
        expected = h @ smooth_seed
        source = n.correct_smooth_many(t,{'case':q},anchors,y)['case']
        cs_error = float(n.np.max(n.np.abs(source-expected)))
        reset_effect = float(n.np.max(n.np.abs(expected-h @ corrected)))
        require(cs_error <= 1e-13 and reset_effect >= 1e-3, 'C&S Y-q sign or label reset differs')
        rng = n.np.random.default_rng(493)
        p = rng.uniform(0.1,1.,size=(7,4,5));p /= p.sum(-1,keepdims=True)
        distances = n.pairwise(p)
        forbidden = n.np.array([1,2,4,5])
        ctx = n.context(t,p,distances,anchors,y,forbidden)
        e = p[anchors]-n.np.eye(5)[y,None,:]
        grams = n.np.einsum('nmc,nkc->nmk',e,e)
        fields = n.np.zeros((7,16));fields[anchors] = grams.reshape(3,16)
        numerator = (h @ fields).reshape(7,4,4)
        mass = h[:,anchors].sum(1)
        expected_gram = n.np.broadcast_to(grams.mean(0),(7,4,4)).copy()
        positive = mass > 0
        expected_gram[positive] = numerator[positive]/mass[positive,None,None]
        gram_error = float(n.np.max(n.np.abs(ctx['R']-expected_gram)))
        require(gram_error <= 2e-14 and ctx['zero_mass_count'] == 2,
                'Transported Gram or global zero-mass fallback failed')
        rejected = False
        try:
            n.context(t,p,distances,anchors,y,n.np.array([0]))
        except ValueError as error:
            rejected = 'Whole-fold anchor exclusion failed' in str(error)
        require(rejected, 'Forbidden anchor overlap did not fail')
        return {'tiny_nodes':7,'source_N_restored_after_test':original_n,'diffusion_max_error':diffusion_error,
                'CS_max_error':cs_error,'CS_label_reset_effect':reset_effect,'transported_Gram_max_error':gram_error,
                'zero_mass_fallback_nodes':2,'forbidden_anchor_overlap_rejected':True,'graph':graph}
    finally:
        n.N = original_n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source','--source-directory',dest='source_directory',required=True)
    parser.add_argument('--manifest-sha256','--source-manifest-sha256',dest='source_manifest_sha256',required=True)
    parser.add_argument('--seal-sha256','--source-seal-sha256',dest='source_seal_sha256',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--execute-synthetic-qualification',action='store_true',
                        help='Optional explicit marker; supplying root-bound source hashes selects qualification')
    args = parser.parse_args()
    # External root authorization is a process boundary, not a new admission schema.
    source = Path(args.source_directory).resolve()
    output = Path(args.output).resolve();output.mkdir(exist_ok=False)
    before = time.perf_counter()
    report = {'status':'started','synthetic_only':True,'scientific_payload_access':False,
              'scientific_training_updates':0,'tests':[],'discarded_engineering_optimizer_updates':0,
              'discarded_engineering_optimizer_updates_completed':0}
    try:
        bound = bind_source(source,args.source_manifest_sha256,args.source_seal_sha256)
        report['source'] = bound
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
            os.environ[name] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        sys.dont_write_bytecode = True
        require('custody' not in sys.modules,'Fresh qualification process required')
        sys.path.insert(0,str(source))
        spec = importlib.util.spec_from_file_location('qualified_saved_logits_numerical',source/'numerical.py')
        n = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = n
        spec.loader.exec_module(n)
        n.setup()
        require(not n.torch.cuda.is_initialized(),'Qualification must not initialize CUDA')
        require(n.torch.get_num_threads() == n.torch.get_num_interop_threads() == 1,
                'One PyTorch intra/inter-op CPU thread required')
        import scipy
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):n.np.show_config()
        report['environment'] = {'python':sys.version,'platform':platform.platform(),
                                 'numpy':n.np.__version__,'scipy':scipy.__version__,'torch':str(n.torch.__version__),
                                 'BLAS_config':capture.getvalue(),'CPU_threads':1,'CUDA_initialized':False,
                                 'torch_intraop_threads':n.torch.get_num_threads(),
                                 'torch_interop_threads':n.torch.get_num_interop_threads(),
                                 'thread_environment_set_before_numerical_import':True,
                                 'torch_default_dtype':str(n.torch.get_default_dtype()),
                                 'deterministic_algorithms':n.torch.are_deterministic_algorithms_enabled(),
                                 'thread_environment':{name:os.environ.get(name) for name in
                                   ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                                    'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')}}
        tests = (qp_against_scipy,tiny_spd,degenerate_projection,gram_reconstruction,
                 stable_skip,tiny_gap_single,head_counts,tiny_discarded_fit,sparse_and_exclusion)
        for test in tests:
            start = time.perf_counter()
            if test is tiny_discarded_fit:
                # Charge the fixed attempted engineering budget even if this check fails.
                report['discarded_engineering_optimizer_updates'] = 150
            try:
                details = test(n)
                if test is tiny_discarded_fit:
                    report['discarded_engineering_optimizer_updates_completed'] = 150
                row = {'test':test.__name__,'status':'passed','details':details}
            except Exception as error:
                row = {'test':test.__name__,'status':'failed','error':str(error),'traceback':traceback.format_exc()}
            row['wall_seconds'] = time.perf_counter()-start
            report['tests'].append(row)
            write(output/(test.__name__+'.json'),row)
        require(not n.torch.cuda.is_initialized(),'Qualification initialized CUDA')
        report['source_post_run'] = bind_source(source,args.source_manifest_sha256,args.source_seal_sha256)
        report['source_unchanged'] = report['source_post_run'] == bound
        report['source_helper_cost_counters'] = n.COUNTERS
        report['status'] = 'passed' if all(r['status'] == 'passed' for r in report['tests']) else 'failed'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        report['traceback'] = traceback.format_exc()
    finally:
        raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['cost'] = {'whole_wall_seconds':time.perf_counter()-before,
                          'peak_RSS_bytes':int(raw_rss if sys.platform == 'darwin' else raw_rss*1024),
                          'discarded_engineering_fits_attempted':int(report['discarded_engineering_optimizer_updates'] > 0),
                          'discarded_engineering_optimizer_update_budget':report['discarded_engineering_optimizer_updates'],
                          'discarded_engineering_optimizer_updates_completed':report['discarded_engineering_optimizer_updates_completed'],
                          'scientific_training_updates':0,'GPU_work':False}
        report['qualifier_source'] = descriptor(Path(__file__))
        write(output/'REPORT.json',report)
        write(output/'QUALIFICATION_RESULT.json',
              {'status':'PASS_SYNTHETIC_NUMERICAL_QUALIFICATION' if report['status'] == 'passed' else
                        'FAIL_SYNTHETIC_NUMERICAL_QUALIFICATION',
               'source_manifest_sha256':args.source_manifest_sha256,
               'source_seal_sha256':args.source_seal_sha256,
               'source':report.get('source'), 'report':descriptor(output/'REPORT.json'),
               'all_tests_passed':report['status'] == 'passed','test_count':len(report['tests']),
               'cost':report['cost'],'source_qualified_only':report['status'] == 'passed',
               'scientific_execution_authorized':False})
        print(json.dumps({'status':report['status'],'report':str(output/'REPORT.json'),
                          'discarded_engineering_update_budget':report['discarded_engineering_optimizer_updates']}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
