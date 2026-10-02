"""SOURCE ONLY. Never imported or executed by the authoring agent.

One-time graph-band initialization of four private-factor routes. Published
ingredients are attributed in SOURCES.md. This is a utility hypothesis, not an
originality, generalization or low-cost claim. No training driver is supplied.

The caller supplies a frozen deterministic common-model closure accepting the
precisely admitted flat factor slice and returning target-node class logits.
Only compact TRAIN labels are accepted. Validation/final labels are not inputs.
All constants below are frozen defaults, not a hidden tuning grid.
"""
from __future__ import annotations
from collections.abc import Callable

CAP = 0.5
RELATIVE_FACTOR_RADIUS = 0.01
ARMIJO_C = 1e-4
BACKTRACK_ATTEMPTS = 6
ALGEBRA_TOLERANCE = 2e-5
FUNCTION_RMS_TOLERANCE = 1e-6


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def declared_factor_names(backbone):
    """Only intermediate stem output and pre-class projector input scales.

    Raw boundary wrappers expose these names; pass TeacherFamily.boundary rather
    than the whole TeacherFamily (whose names have a boundary. prefix).
    Common K1 shapes: PolyFormer-Mono [(1,256),(1,256)] ->512;
    Polynormer-r [(1,512),(1,512)] ->1024 at its global phase. R/S identity initialization
    and warm bias copies are a separate mandatory caller operation. No reset
    after wrapping. Stem input R, head output S, and all B are excluded here.
    """
    if backbone == 'PolyFormer-Mono':
        return ('stem.S', 'head.R')
    if backbone == 'Polynormer-r':
        return ('stem.S', 'global_head.R')
    raise ValueError('Backbone/slice must be prospectively admitted')


def bind_common_model(model, backbone, model_args, select_target_logits,
                      model_kwargs=None):
    """Build a pure K1 closure of model.boundary, the raw boundary wrapper.

    select_target_logits returns [N_target,C], selecting only the actual final
    predictive logits (global-only in the admitted final Polynormer-r phase).
    No embedding, auxiliary head or unpooled tuple may substitute for logits.
    All shared parameters and buffers are detached snapshots. Buffers are cloned
    per call, preventing a callback from changing the admitted common state.
    """
    import torch
    _require(all(not module.training for module in model.modules()),
             'Common model and all submodules must be in deterministic eval mode')
    names = declared_factor_names(backbone)
    params = dict(model.named_parameters())
    for name, value in params.items():
        if name.endswith('.R') or name.endswith('.S'):
            _require(bool(torch.equal(value.detach(), torch.ones_like(value))),
                     'All common boundary multiplicative factors must be identity')
    expected_width = 256 if backbone == 'PolyFormer-Mono' else 512
    for name in names:
        _require(name in params and tuple(params[name].shape) == (1, expected_width),
                 'Expected exact K1 modern-wrapper factor name/shape')
        _require(bool(torch.equal(params[name].detach(), torch.ones_like(params[name]))),
                 'Seeding slice must start at identity after common warm fit')
    snapshots = {name: value.detach().clone() for name, value in params.items()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    sizes = [params[name].numel() for name in names]
    theta0 = torch.cat([snapshots[name].reshape(-1) for name in names])
    kwargs = {} if model_kwargs is None else dict(model_kwargs)

    def logits_fn(theta):
        _require(tuple(theta.shape) == tuple(theta0.shape), 'Factor vector shape changed')
        replacements = dict(snapshots)
        offset = 0
        for name, size in zip(names, sizes):
            replacements[name] = theta[offset:offset+size].view_as(snapshots[name])
            offset += size
        fresh_buffers = {name: value.clone() for name, value in buffers.items()}
        result = torch.func.functional_call(model, (replacements, fresh_buffers),
                                             model_args, kwargs, strict=True)
        logits = select_target_logits(result)
        _require(logits.ndim == 2 and logits.shape[1] >= 2, 'Need target class logits [N,C]')
        return logits

    return theta0, logits_fn, dict(backbone=backbone, names=list(names),
                                 shapes=[list(params[name].shape) for name in names],
                                 dimensions=sum(sizes), identity=True)


def install_factor_slices(ensemble_model, slices, backbone):
    """Set ONLY the admitted slice on K4 model.boundary AFTER its warm copy.

    The caller must first verify every route's logits equal the common predictor
    and copy every other factor/bias as well as shared warm weights. This helper
    cannot certify a separately constructed wrapper's warm cloning operation.
    """
    import torch
    params = dict(ensemble_model.named_parameters())
    names = declared_factor_names(backbone)
    width = 256 if backbone == 'PolyFormer-Mono' else 512
    _require(tuple(slices.shape) == (4, len(names)*width), 'Need four admitted slices')
    with torch.no_grad():
        for index, name in enumerate(names):
            _require(name in params and tuple(params[name].shape) == (4, width),
                     'K4 parameter name/shape mismatch')
            params[name].copy_(slices[:, index*width:(index+1)*width])


def symmetric_normalized_adjacency(num_nodes, canonical_edges, dtype, device):
    """Unweighted simple undirected released topology, no added self loops.

    canonical_edges is int64 [2,E], lexicographic u<v, no duplicates. Isolates
    have S rows/columns zero. In a typed graph, pass the admitted union topology
    over ALL node types; the predictive encoder remains typed. This filter does
    not claim relation-specific bands. Graph acquisition is a separate root task.
    """
    import torch
    _require(canonical_edges.dtype == torch.int64 and canonical_edges.ndim == 2
             and canonical_edges.shape[0] == 2, 'Need canonical int64 [2,E]')
    edges = canonical_edges.to(device)
    _require(bool(((edges[0] < edges[1]) & (edges >= 0).all(0)
                   & (edges < num_nodes).all(0)).all()), 'Require0<=u<v<N')
    pair_keys = edges[0]*num_nodes+edges[1]
    _require(bool((pair_keys[1:] > pair_keys[:-1]).all()), 'Canonical sorted unique edges')
    directed = torch.cat((edges, edges.flip(0)), dim=1)
    degree = torch.bincount(directed[0], minlength=num_nodes).to(dtype=dtype)
    inv = degree.clamp_min(1).rsqrt()
    values = inv[directed[0]]*inv[directed[1]]
    return torch.sparse_coo_tensor(directed, values, (num_nodes, num_nodes),
                                  dtype=dtype, device=device).coalesce()


def bernstein_cubic_bands(S, residual):
    """Borrowed BernNet basis: H_b=C(3,b)((I+S)/2)^(3-b)((I-S)/2)^b.

    Three sparse matrix products produce S^j residual, j1..3. Expanded cubic
    polynomials avoid eigendecomposition. In exact arithmetic sum_b H_b=I.
    """
    import torch
    y0 = residual
    y1 = torch.sparse.mm(S, y0)
    y2 = torch.sparse.mm(S, y1)
    y3 = torch.sparse.mm(S, y2)
    return torch.stack(((y0+3*y1+3*y2+y3)/8,
                        3*(y0+y1-y2-y3)/8,
                        3*(y0-y1-y2+y3)/8,
                        (y0-3*y1+3*y2-y3)/8))


def permute_topology_nodes(S, seed, node_type_groups=None):
    """Alignment-null control: Pi S Pi^T, while labels/features stay fixed.

    It preserves graph spectrum, edge inventory and degree multiset. In typed
    graphs, prospectively supplied type groups partition all graph nodes and
    permutation stays within each node type. Merely permuting four band names
    would only reorder members and is NOT this control.
    """
    import torch
    n = S.shape[0]
    groups = [torch.arange(n)] if node_type_groups is None else node_type_groups
    all_ids = torch.cat([ids.cpu() for ids in groups])
    _require(torch.equal(all_ids.sort().values, torch.arange(n)), 'Type groups partition nodes')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    permutation = torch.arange(n)
    for ids in groups:
        ids = ids.cpu()
        permutation[ids] = ids[torch.randperm(ids.numel(), generator=generator)]
    permutation = permutation.to(S.device)
    indices = permutation[S.coalesce().indices()]
    return torch.sparse_coo_tensor(indices, S.coalesce().values().clone(), S.shape,
                                  dtype=S.dtype, device=S.device).coalesce(), permutation


def initialize_four_routes(logits_fn: Callable, theta0, S, target_nodes,
                           train_rows, train_labels, tangent_mode='graph',
                           control_seed=None):
    """Return four factor vectors and source-only diagnostics; mutate no model.

    target_nodes injects target class residuals into full typed/homogeneous graph.
    train_rows indexes target logits, train_labels contains exactly those labels.
    Empty/gauge/null diversity falls back to safeguarded common descent. Failed
    common descent falls back to the identical warm factors. It never forces
    disagreement. All line-search attempts, failures and call counts are returned.
    Exceptions are retained by the root driver; never silently replace a cell.
    """
    import torch
    import torch.nn.functional as F
    _require(tangent_mode in ('graph', 'random_tangent', 'common_only'), 'Declared control only')
    _require(tangent_mode != 'random_tangent' or isinstance(control_seed, int), 'Freeze control seed')
    _require(theta0.ndim == 1 and bool(torch.isfinite(theta0).all()), 'Finite flat factors')
    _require(S.layout == torch.sparse_coo and S.shape[0] == S.shape[1], 'Need normalized sparse S')
    _require(S.device == theta0.device and S.dtype == theta0.dtype, 'S/model dtype/device agree')
    for ids in (target_nodes, train_rows):
        _require(ids.dtype == torch.int64 and ids.ndim == 1, 'Index vectors must be int64')
        _require(ids.device == theta0.device, 'Indices must already be on model device')
        _require(ids.numel() > 0 and ids.unique().numel() == ids.numel(), 'Unique nonempty indices')
    _require(bool(((target_nodes >= 0) & (target_nodes < S.shape[0])).all()), 'Target IDs in graph')
    _require(train_labels.dtype == torch.int64 and train_labels.shape == train_rows.shape,
             'Only compact training labels [T]')
    _require(train_labels.device == theta0.device, 'Labels must be on model device')
    stats = dict(operation='graph_band_common_descent_v1', source_labels='train_only',
                 tangent_mode=tangent_mode, control_seed=control_seed,
                 vjp_forwards=1, vjp_calls=0, jvp_calls=0, graph_sparse_products=3,
                 line_search_forward_calls=0, attempts=[], cap=CAP,
                 relative_factor_radius=RELATIVE_FACTOR_RADIUS,
                 armijo_c=ARMIJO_C, backtrack_attempts_per_branch=BACKTRACK_ATTEMPTS)
    base_logits, pullback = torch.func.vjp(logits_fn, theta0)
    _require(base_logits.shape[0] == target_nodes.numel(), 'Target-row mapping mismatch')
    _require(bool(torch.isfinite(base_logits).all()), 'Nonfinite common logits')
    _require(bool(((train_rows >= 0) & (train_rows < base_logits.shape[0])).all()), 'Train rows range')
    _require(bool(((train_labels >= 0) & (train_labels < base_logits.shape[1])).all()), 'Class range')
    base_loss = F.cross_entropy(base_logits[train_rows], train_labels).detach()
    stats['baseline_train_ce'] = float(base_loss)
    residual = torch.zeros_like(base_logits)
    residual[train_rows] = (base_logits[train_rows].softmax(-1).detach()
                           - F.one_hot(train_labels, base_logits.shape[1]).to(base_logits.dtype)) / train_rows.numel()
    g = pullback(residual.detach())[0].detach()
    stats['vjp_calls'] += 1
    if tangent_mode == 'common_only':
        # The cheapest descent control omits all unused graph bands and JVPs.
        h = (g/4).repeat(4, 1)
        stats.update(graph_sparse_products=0, partition_relative_error=None)
    else:
        full_residual = torch.zeros((S.shape[0], base_logits.shape[1]),
                                    device=theta0.device, dtype=theta0.dtype)
        full_residual[target_nodes] = residual
        bands = bernstein_cubic_bands(S, full_residual).detach()
        scale = full_residual.norm().clamp_min(torch.finfo(theta0.dtype).tiny)
        partition_error = float((bands.sum(0)-full_residual).norm()/scale)
        stats['partition_relative_error'] = partition_error
        _require(partition_error <= ALGEBRA_TOLERANCE, 'Graph-band partition identity failed')
        hs = []
        for band in bands:
            cotangent = torch.zeros_like(base_logits)
            cotangent[train_rows] = band[target_nodes[train_rows]]
            hs.append(pullback(cotangent)[0].detach())
            stats['vjp_calls'] += 1
        h = torch.stack(hs)
    base_logits = base_logits.detach()
    del pullback  # Release the VJP tape before fresh JVP/line-search calls.
    _require(bool(torch.isfinite(g).all() and torch.isfinite(h).all()), 'Nonfinite gradients')
    g64, h64 = g.double(), h.double()
    g2 = g64.square().sum()
    stats['common_gradient_squared_norm'] = float(g2)
    sum_error = float((h64.sum(0)-g64).norm()/g64.norm().clamp_min(1e-20))
    stats['band_gradient_sum_relative_error'] = sum_error
    _require(sum_error <= ALGEBRA_TOLERANCE or float(g2) <= 1e-20,
             'Masked band-gradient sum differs from common gradient')
    if float(g2) <= 1e-20:
        stats.update(status='unchanged_warm', reason='zero_common_gradient', accepted_alpha=0.0)
        return theta0.detach().repeat(4, 1), stats

    # Common centering and scalar normalization preserve both required identities.
    t64 = h64 - (h64 @ g64 / g2)[:, None]*g64
    t64 = t64-t64.mean(0, keepdim=True)
    t64 = t64-(t64 @ g64 / g2)[:, None]*g64
    t64 = t64-t64.mean(0, keepdim=True)
    # A single Frobenius cap across all four rows gives an exactly norm-matched
    # random control and bounds every route's tangent by CAP*||g|| as well.
    total = t64.norm()
    if tangent_mode == 'random_tangent' and float(total) > 1e-20:
        generator = torch.Generator(device='cpu').manual_seed(control_seed)
        random = torch.randn(t64.shape, dtype=torch.float64, generator=generator).to(theta0.device)
        random = random-random.mean(0, keepdim=True)
        random = random-(random @ g64/g2)[:, None]*g64
        random = random-random.mean(0, keepdim=True)
        t64 = random*(total/random.norm().clamp_min(1e-20))
    if tangent_mode == 'common_only':
        t64 = torch.zeros_like(t64)
        total = t64.norm()
    lam = CAP*g64.norm()/total.clamp_min(1e-20)
    tangent = (lam*t64).to(theta0.dtype)
    directions = -g[None, :]-tangent
    dots = directions.double() @ g64
    orth_error = float((dots+g2).abs().max()/g2)
    zero_mean_error = float(tangent.double().mean(0).norm()/g64.norm())
    stats.update(common_scalar=float(lam), descent_relative_error=orth_error,
                 tangent_mean_relative_error=zero_mean_error,
                 tangent_frobenius_norm=float(tangent.double().norm()))
    _require(max(orth_error, zero_mean_error) <= ALGEBRA_TOLERANCE,
             'Orthogonality/zero-mean certificate failed after cast')

    def centered(logits):
        return logits-logits.mean(-1, keepdim=True)

    functional = []
    if tangent_mode == 'common_only' or float(tangent.double().norm()) <= 1e-20:
        functions = torch.zeros((4, train_rows.numel(), base_logits.shape[1]),
                                 device=theta0.device, dtype=theta0.dtype)
    else:
        for row in tangent:
            _, jvp = torch.func.jvp(logits_fn, (theta0,), (row,))
            stats['jvp_calls'] += 1
            functional.append(centered(jvp[train_rows]).detach())
        functions = torch.stack(functional)
    _require(bool(torch.isfinite(functions).all()), 'Nonfinite functional tangent')
    function_scale = float(centered(base_logits[train_rows]).square().mean().sqrt().clamp_min(1.0))
    threshold = FUNCTION_RMS_TOLERANCE*function_scale
    pair_rms = [float((functions[i]-functions[j]).square().mean().sqrt())
                for i in range(4) for j in range(i)]
    stats['source_tangent_pair_rms'] = pair_rms
    functional_ok = bool(pair_rms and min(pair_rms) > threshold)
    stats['functional_tangent_admitted'] = functional_ok

    def try_branch(branch, ds):
        max_norm = float(ds.double().norm(dim=1).max())
        alpha0 = RELATIVE_FACTOR_RADIUS*(theta0.numel()**0.5)/max(max_norm, 1e-20)
        for attempt in range(BACKTRACK_ATTEMPTS):
            alpha = alpha0/(2**attempt)
            trial = theta0[None, :]+alpha*ds
            with torch.no_grad():
                zs = torch.stack([logits_fn(row) for row in trial])
                stats['line_search_forward_calls'] += 4
                losses = torch.stack([F.cross_entropy(z[train_rows], train_labels) for z in zs])
                pooled_loss = F.cross_entropy(zs.mean(0)[train_rows], train_labels)
                finite = bool(torch.isfinite(zs).all() and torch.isfinite(losses).all()
                              and torch.isfinite(pooled_loss))
                bound = float(base_loss)-ARMIJO_C*alpha*float(g2)
                quality = finite and bool((losses.double() <= bound).all()) and float(pooled_loss) <= bound
                actual_pairs = [float((centered(zs[i, train_rows])-centered(zs[j, train_rows]))
                                      .square().mean().sqrt()) for i in range(4) for j in range(i)]
                useful = branch == 'common_only' or (min(actual_pairs)/alpha > threshold)
            record = dict(branch=branch, attempt=attempt, alpha=alpha, finite=finite,
                          route_train_ce=[float(x) for x in losses] if finite else None,
                          pooled_train_ce=float(pooled_loss) if finite else None,
                          armijo_bound=bound, quality_accepted=quality,
                          actual_centered_pair_rms=actual_pairs if finite else None,
                          functional_accepted=useful if finite else False)
            stats['attempts'].append(record)
            if quality and useful:
                stats.update(status=branch, reason='accepted', accepted_alpha=alpha)
                return trial.detach()
        return None

    if functional_ok and tangent_mode != 'common_only':
        candidate = try_branch('graph_band' if tangent_mode == 'graph' else 'random_tangent', directions)
        if candidate is not None:
            return candidate, stats
    stats['graph_branch_fallback_reason'] = ('source_null_or_duplicate_function_tangents'
                                            if not functional_ok else 'finite_step_gate_failed')
    candidate = try_branch('common_only', -g[None, :].repeat(4, 1))
    if candidate is not None:
        return candidate, stats
    stats.update(status='unchanged_warm', reason='common_armijo_failed', accepted_alpha=0.0)
    return theta0.detach().repeat(4, 1), stats


def qualify_gradient_interface(logits_fn, theta0, train_rows, train_labels, seed):
    """Future preflight only; not run by author. Train labels and source inputs only.

    Three deterministic directions check JVP/VJP duality, centered-logit central
    differences and CE directional derivatives at two fixed epsilon values.
    Every diagnostic is retained. Unsupported AD is an explicit failure, never
    silently replaced by coordinate repulsion. Caller charges all passes/time.
    """
    import torch
    import torch.nn.functional as F
    generator = torch.Generator(device='cpu').manual_seed(seed)
    z, pullback = torch.func.vjp(logits_fn, theta0)
    residual = torch.zeros_like(z)
    residual[train_rows] = (z[train_rows].softmax(-1).detach()
                           - F.one_hot(train_labels, z.shape[1]).to(z.dtype))/train_rows.numel()
    g = pullback(residual)[0]
    records = []
    for index in range(3):
        d = torch.randn(theta0.shape, generator=generator, dtype=theta0.dtype).to(theta0.device)
        d = d/d.norm().clamp_min(1e-20)*(theta0.numel()**0.5)
        cotangent = torch.randn(z.shape, generator=generator, dtype=z.dtype).to(z.device)
        _, jvp = torch.func.jvp(logits_fn, (theta0,), (d,))
        vjp = pullback(cotangent)[0]
        lhs = float((jvp.double()*cotangent.double()).sum())
        rhs = float((vjp.double()*d.double()).sum())
        dual_error = abs(lhs-rhs)/max(abs(lhs), abs(rhs), 1e-8)
        derivative = float((g.double()*d.double()).sum())
        checks = []
        for epsilon in (1e-3, 3e-4):
            with torch.no_grad():
                plus, minus = logits_fn(theta0+epsilon*d), logits_fn(theta0-epsilon*d)
                finite_difference = (plus-minus)/(2*epsilon)
                centered_fd = finite_difference-finite_difference.mean(-1, keepdim=True)
                centered_jvp = jvp-jvp.mean(-1, keepdim=True)
                logits_error = float((centered_fd-centered_jvp).double().norm()
                                     /centered_jvp.double().norm().clamp_min(1e-8))
                ce_fd = float((F.cross_entropy(plus[train_rows], train_labels)
                              - F.cross_entropy(minus[train_rows], train_labels))/(2*epsilon))
                ce_error = abs(ce_fd-derivative)/max(abs(ce_fd), abs(derivative), 1e-3)
            checks.append(dict(epsilon=epsilon, logits_relative_error=logits_error,
                               ce_directional_error=ce_error, analytic_ce_derivative=derivative,
                               finite_ce_derivative=ce_fd))
        records.append(dict(direction=index, dual_relative_error=dual_error,
                            finite_differences=checks))
    # Finite differences can be noisy near ReLU kinks. One of TWO PREDECLARED
    # steps must agree for every direction; no adaptive epsilon/seed search.
    passed = all(row['dual_relative_error'] <= 2e-4 and
                 any(check['logits_relative_error'] <= 0.05 and
                     check['ce_directional_error'] <= 0.05 for check in row['finite_differences'])
                 for row in records)
    return dict(passed=passed, records=records, vjp_forwards=1, vjp_calls=4,
                jvp_calls=3, finite_difference_forward_calls=12,
                source_labels='train_only', epsilons=[1e-3, 3e-4], random_directions=3)
