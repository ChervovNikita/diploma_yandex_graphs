"""CPU FP64 postprocessing. Imported only after complete metadata/payload binding."""
import hashlib
import itertools
import math
import time
import numpy as np
import torch
from scipy import sparse
from custody import require, N, C

M = 4
LOWER = 0.0125
RHO = (0.01, 0.1)
REG = (1e-4, 1e-2)
UPPER = np.triu_indices(M)
PAIRS = tuple(itertools.combinations(range(M), 2))
COUNTERS = {'H_calls': 0, 'sparse_steps': 0, 'logical_H_applications': 0,
            'H_wall_seconds': 0.0, 'maximum_field_width': 0, 'calls': [],
            'QP_calls': 0, 'QP_solutions': 0, 'QP_wall_seconds': 0.0}


def finite(x, name):
    require(np.isfinite(x).all(), name + ' nonfinite')


def setup():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.float64)


def folds(ids, split, outer=None, count=3):
    salt = (f'amazon-moment-v1|split={split}|outer|node=' if outer is None else
            f'amazon-moment-v1|split={split}|outer={outer}|inner|node=')
    ordered = sorted(map(int, ids), key=lambda v: (hashlib.sha256((salt + str(v)).encode()).digest(), v))
    return [np.array(ordered[j::count], dtype=np.int64) for j in range(count)]


def arrays(z):
    logp = torch.log_softmax(z.to(torch.float64), dim=-1)
    native_log = torch.logsumexp(logp, dim=0) - math.log(z.shape[0])
    p = logp.exp()
    native64 = p.mean(0).numpy()
    native32 = torch.softmax(z, dim=-1).mean(0).numpy()
    return {'z': z.to(torch.float64).numpy(), 'p': p.numpy().transpose(1, 0, 2),
            'logp': logp.numpy().transpose(1, 0, 2), 'native': native64,
            'native_log': native_log.numpy(), 'native32': native32,
            'native_class': native32.argmax(-1),
            'precision_argmax_discrepancies': int(np.sum(native32.argmax(-1) != native64.argmax(-1)))}


def graph(edge, native_class):
    keep = (edge[0] == edge[1]) | (native_class[edge[0]] == native_class[edge[1]])
    u, v = edge[:, keep]
    degree = np.bincount(u, minlength=N).astype(np.float64)
    require((degree >= 1).all(), 'Missing source self loop')
    require(len(np.unique(u*N+v)) == len(u), 'Duplicate retained edge')
    t = sparse.csr_matrix((1.0 / degree[u], (u, v)), shape=(N, N))
    t.sort_indices()
    require(t.has_sorted_indices and t.nnz == len(u), 'CSR duplicate/index identity differs')
    require(np.max(np.abs(np.asarray(t.sum(1)).ravel() - 1.0)) < 1e-12, 'Graph not row stochastic')
    return t, degree, {'retained_edges': int(len(u)), 'source_edges': int(edge.shape[1]),
                      'retained_fraction': float(len(u) / edge.shape[1]),
                      'retained_edge_logical_sha256': hashlib.sha256(edge[:, keep].tobytes()).hexdigest(),
                      'orientation': 'T[u,v]: row=edge_index[0], column=edge_index[1]',
                      'degree': {'min': float(degree.min()), 'max': float(degree.max()),
                                 'mean': float(degree.mean()), 'std': float(degree.std())},
                      'CSR': {name: {'sha256': hashlib.sha256(value.tobytes()).hexdigest(),
                                    'dtype': str(value.dtype), 'shape': list(value.shape)}
                              for name, value in (('indptr', t.indptr), ('indices', t.indices), ('data', t.data))}}


def diffuse(t, x, *, logical=1, purpose='seed_context'):
    start = time.perf_counter()
    f = x.copy()
    for _ in range(20):
        f = 0.2 * x + 0.8 * (t @ f)
    finite(f, 'Diffused field')
    elapsed = time.perf_counter()-start
    width = x.shape[1]
    COUNTERS['H_calls'] += 1
    COUNTERS['sparse_steps'] += 20
    COUNTERS['logical_H_applications'] += logical
    COUNTERS['H_wall_seconds'] += elapsed
    COUNTERS['maximum_field_width'] = max(COUNTERS['maximum_field_width'], width)
    COUNTERS['calls'].append({'purpose': purpose, 'field_width': width,
                              'logical_H_applications': logical, 'sparse_steps': 20,
                              'wall_seconds': elapsed})
    return f


def pairwise(p):
    return np.stack([np.sum((p[:, m] - p[:, n]) ** 2, axis=-1) for m, n in PAIRS], axis=-1)


def reconstruct(diagonal, distances):
    r = np.zeros((len(diagonal), M, M), dtype=np.float64)
    r[:, np.arange(M), np.arange(M)] = diagonal
    for k, (m, n) in enumerate(PAIRS):
        r[:, m, n] = r[:, n, m] = (diagonal[:, m] + diagonal[:, n] - distances[:, k]) / 2
    return r


def context(t, p, distances, anchor_ids, anchor_y, forbidden):
    """Every label access is through anchor_y; caller never passes forbidden target labels."""
    require(len(anchor_ids) == len(anchor_y) and len(np.unique(anchor_ids)) == len(anchor_ids) and
            not np.intersect1d(anchor_ids, forbidden).size, 'Whole-fold anchor exclusion failed')
    onehot = np.eye(C)[anchor_y]
    e = p[anchor_ids] - onehot[:, None, :]
    diagonal = np.sum(e * e, axis=-1)
    fields = np.zeros((N, 16), dtype=np.float64)
    fields[anchor_ids, :4] = diagonal
    fields[anchor_ids, 4:10] = distances[anchor_ids]
    fields[anchor_ids, 10:15] = onehot
    fields[anchor_ids, 15] = 1.0
    h = diffuse(t, fields)
    mass = h[:, 15]
    positive = mass > 0
    normalized = np.zeros((N, 15), dtype=np.float64)
    normalized[positive] = h[positive, :15] / mass[positive, None]
    global_diag = diagonal.mean(0)
    global_dist = distances[anchor_ids].mean(0)
    prior = onehot.mean(0)
    normalized[~positive, :4] = global_diag
    normalized[~positive, 4:10] = global_dist
    normalized[~positive, 10:15] = prior
    r = reconstruct(normalized[:, :4], normalized[:, 4:10])
    r0 = reconstruct(global_diag[None], global_dist[None])[0]
    require(np.linalg.eigvalsh(r).min() >= -1e-10 and np.linalg.eigvalsh(r0).min() >= -1e-10,
            'Transported Gram PSD failed')
    return {'R': r, 'R0': r0, 'D': normalized[:, 4:10], 'posterior': normalized[:, 10:15],
            'mass': mass, 'scale': max(float(np.trace(r0) / M), 1e-12),
            'anchor_ids': anchor_ids.copy(), 'forbidden_ids': np.asarray(forbidden).copy(),
            'zero_mass_count': int((~positive).sum())}


def simplex_qp(a, b=None, *, singular=False):
    """Batch active-face enumeration, four variables; original KKT verified afterward.

    Objective w'A w-2b'w; lower bound 0.0125. Singular posterior projection
    uses Moore-Penrose KKT solutions and minimum weight norm among tied optima.
    No ridge is silently added to the projection problem.
    """
    started = time.perf_counter()
    if a.ndim == 2:
        a = a[None]
    if b is None:
        b = np.zeros((len(a), M))
    b = np.broadcast_to(b, (len(a), M))
    finite(a, 'QP matrix'); finite(b, 'QP linear term')
    require(np.max(np.abs(a - a.transpose(0, 2, 1))) < 1e-10, 'Asymmetric QP matrix')
    require(np.linalg.eigvalsh(a).min() >= -1e-10, 'QP matrix not PSD')
    result = np.empty((len(a), M))
    peak_primal = peak_scaled_stationarity = peak_raw_stationarity = 0.0
    peak_scaled_dual = peak_raw_dual = 0.0
    for offset in range(0, len(a), 2048):
        aa, bb = a[offset:offset+2048], b[offset:offset+2048]
        scale = np.maximum(np.maximum(np.max(np.abs(aa), axis=(1, 2)), np.max(np.abs(bb), axis=1)), 1e-12)
        an, bn = aa / scale[:, None, None], bb / scale[:, None]
        best = np.full(len(aa), np.inf)
        best_norm = np.full(len(aa), np.inf)
        chosen = np.full((len(aa), M), np.nan)
        for k in range(1, M+1):
            for free in itertools.combinations(range(M), k):
                free = np.array(free)
                fixed = np.array([j for j in range(M) if j not in free], dtype=int)
                aff = an[:, free[:, None], free]
                rhs = bn[:, free].copy()
                if len(fixed):
                    rhs -= LOWER * an[:, free[:, None], fixed].sum(-1)
                matrix = np.zeros((len(aa), k+1, k+1))
                matrix[:, :k, :k] = aff
                matrix[:, :k, k] = matrix[:, k, :k] = 1
                target = np.zeros((len(aa), k+1))
                target[:, :k] = rhs
                target[:, k] = 1 - len(fixed)*LOWER
                if singular:
                    solution = np.einsum('bij,bj->bi', np.linalg.pinv(matrix, rcond=1e-12), target)
                else:
                    solution = np.linalg.solve(matrix, target[..., None])[..., 0]
                w = np.full((len(aa), M), LOWER)
                w[:, free] = solution[:, :k]
                g = np.einsum('bij,bj->bi', an, w) - bn
                level = -solution[:, k]
                feasible = ((w >= LOWER-1e-12).all(1) &
                            (np.abs(w.sum(1)-1) < 1e-12) &
                            (np.max(np.abs(g[:, free] - level[:, None]), axis=1) < 1e-10))
                if len(fixed):
                    feasible &= (g[:, fixed] >= level[:, None]-1e-10).all(1)
                objective = np.einsum('bi,bij,bj->b', w, an, w) - 2*np.sum(bn*w, axis=1)
                norm = np.sum(w*w, axis=1)
                take = feasible & ((objective < best-1e-14) |
                                   ((np.abs(objective-best) <= 1e-14) & (norm < best_norm-1e-14)))
                chosen[take], best[take], best_norm[take] = w[take], objective[take], norm[take]
        require(np.isfinite(chosen).all(), 'QP active-face solution not found')
        # Remove only roundoff below the bound; recheck all KKT conditions after cleanup.
        excess = np.maximum(chosen-LOWER, 0)
        chosen = LOWER + (1-M*LOWER)*excess/excess.sum(1, keepdims=True)
        primal = max(float(np.max(np.abs(chosen.sum(1)-1))), float(max(0, LOWER-chosen.min())))
        for normalized, matrix, linear in ((True, an, bn), (False, aa, bb)):
            g = np.einsum('bij,bj->bi', matrix, chosen) - linear
            interior = chosen > LOWER+1e-11
            level = np.sum(np.where(interior, g, 0), axis=1, keepdims=True)/interior.sum(1, keepdims=True)
            stat = float(np.max(np.where(interior, np.abs(g-level), 0)))
            dual = float(np.max(np.where(~interior, np.maximum(level-g, 0), 0)))
            require(stat <= 1e-10 and dual <= 1e-10 and primal <= 1e-10,
                    'QP feasibility/free-stationarity/inactive-dual check failed')
            if normalized:
                peak_scaled_stationarity = max(peak_scaled_stationarity, stat)
                peak_scaled_dual = max(peak_scaled_dual, dual)
            else:
                peak_raw_stationarity = max(peak_raw_stationarity, stat)
                peak_raw_dual = max(peak_raw_dual, dual)
        peak_primal = max(peak_primal, primal)
        result[offset:offset+len(aa)] = chosen
    elapsed = time.perf_counter()-started
    COUNTERS['QP_calls'] += 1
    COUNTERS['QP_solutions'] += len(a)
    COUNTERS['QP_wall_seconds'] += elapsed
    return result, {'max_primal_error': peak_primal,
                    'max_scaled_stationarity_error': peak_scaled_stationarity,
                    'max_raw_stationarity_error': peak_raw_stationarity,
                    'max_scaled_inactive_dual_error': peak_scaled_dual,
                    'max_raw_inactive_dual_error': peak_raw_dual,
                    'solutions': len(a), 'wall_seconds': elapsed}


def log_pool(logp, weights):
    return np.logaddexp.reduce(logp + np.log(weights)[:, :, None], axis=1)


def pool(p, logp, weights):
    q = np.einsum('nm,nmc->nc', weights, p)
    return q, log_pool(logp, weights)


def moment(ctx, p, logp, rho, kind, rows=None):
    ix = slice(None) if rows is None else rows
    if kind == 'global':
        matrix = ctx['R0'][None] + rho*ctx['scale']*np.eye(M)[None]
        w, checks = simplex_qp(matrix)
        w = np.broadcast_to(w, (len(p[ix]), M))
    else:
        matrix = 0.5*ctx['R'][ix] + 0.5*ctx['R0'][None]
        if kind == 'diagonal':
            matrix = np.eye(M)[None] * np.diagonal(matrix, axis1=1, axis2=2)[:, None, :]
        matrix = matrix + rho*ctx['scale']*np.eye(M)[None]
        w, checks = simplex_qp(matrix)
    q, logq = pool(p[ix], logp[ix], w)
    return q, logq, w, checks


def projection(ctx, p, logp):
    a = np.einsum('nmc,nkc->nmk', p, p)
    b = np.einsum('nmc,nc->nm', p, ctx['posterior'])
    w, checks = simplex_qp(a, b, singular=True)
    q, logq = pool(p, logp, w)
    return q, logq, w, checks


def features(a, neighbour, distances, degree, ctx, rows, *, full=False, single=False):
    if single:
        x = np.concatenate((a['p'][rows, 0], a['logp'][rows, 0], neighbour[rows],
                            np.log1p(degree[rows])[:, None], ctx['mass'][rows, None]), axis=1)
        require(x.shape[1] == 17, 'Single feature width differs')
        return x
    x = np.concatenate((a['p'][rows].reshape(len(rows), -1), a['logp'][rows].reshape(len(rows), -1),
                        a['native'][rows], a['native_log'][rows], neighbour[rows], distances[rows],
                        ctx['D'][rows], np.log1p(degree[rows])[:, None], ctx['mass'][rows, None]), axis=1)
    require(x.shape[1] == 69, 'Score feature width differs')
    if full:
        x = np.concatenate((x, ctx['R'][rows][:, UPPER[0], UPPER[1]],
                            np.broadcast_to(ctx['R0'][UPPER], (len(rows), 10)), ctx['posterior'][rows]), axis=1)
        require(x.shape[1] == 94, 'Full feature width differs')
    finite(x, 'Head features')
    return x


def project_simplex(q):
    u = np.sort(q, axis=-1)[:, ::-1]
    cumulative = np.cumsum(u, axis=-1) - 1
    k = np.arange(1, C+1)
    active = u - cumulative/k > 0
    rho = active.sum(-1)-1
    theta = cumulative[np.arange(len(q)), rho] / (rho+1)
    return np.maximum(q-theta[:, None], 0)


def correct_smooth_many(t, predictions, anchor_ids, anchor_y):
    # Batch every configuration's five fields; all receive identical anchors and H.
    keys = list(predictions)
    q = np.concatenate([predictions[k] for k in keys], axis=1)
    target = np.tile(np.eye(C)[anchor_y], (1, len(keys)))
    seed = np.zeros_like(q)
    seed[anchor_ids] = target-q[anchor_ids]
    residual = diffuse(t, seed, logical=len(keys), purpose='CS_correction')
    qc = np.concatenate([project_simplex(q[:, j*C:(j+1)*C]+residual[:, j*C:(j+1)*C])
                         for j in range(len(keys))], axis=1)
    qc[anchor_ids] = target
    smoothed = diffuse(t, qc, logical=len(keys), purpose='CS_smoothing')
    require(smoothed.min() >= -1e-12, 'Negative C&S probability')
    return {key: smoothed[:, j*C:(j+1)*C] for j, key in enumerate(keys)}


def metrics(q, y, *, native_class=None, native_log=None):
    finite(q, 'Metric probability')
    require(q.shape == (len(y), C) and q.min() >= -1e-12 and
            np.max(np.abs(q.sum(1)-1)) < 1e-10, 'Metric probability schema differs')
    pred = q.argmax(-1) if native_class is None else native_class
    logq = np.log((1-1e-12)*q + 1e-12/C) if native_log is None else native_log
    return {'count': len(y), 'Brier': float(np.mean(np.sum((q-np.eye(C)[y])**2, axis=-1))),
            'NLL': float(-np.mean(logq[np.arange(len(y)), y])),
            'accuracy': float(np.mean(pred == y)),
            'class_accuracy': [{'class': c, 'count': int(np.sum(y == c)),
                                'accuracy': float(np.mean(pred[y == c] == c)) if np.any(y == c) else None}
                               for c in range(C)]}


class ResidualHead(torch.nn.Module):
    def __init__(self, d, width, seed):
        super().__init__()
        self.A = torch.nn.Parameter(torch.zeros(C, d))
        self.W = torch.nn.Parameter(torch.empty(width, d))
        self.b = torch.nn.Parameter(torch.zeros(width))
        self.V = torch.nn.Parameter(torch.zeros(C, width))
        self.c = torch.nn.Parameter(torch.zeros(C))
        generator = torch.Generator(device='cpu').manual_seed(seed)
        limit = math.sqrt(6/(d+width))
        with torch.no_grad():
            self.W.uniform_(-limit, limit, generator=generator)

    def forward(self, x, skip_log):
        residual = x @ self.A.T + torch.relu(x @ self.W.T+self.b) @ self.V.T+self.c
        # Log-space evaluation preserves finite class scores even if exp(log q) underflows.
        return torch.softmax(skip_log+residual, dim=-1)


def grad_norm(parameters):
    return math.sqrt(sum(float((p.grad*p.grad).sum()) for p in parameters if p.grad is not None))


def optimize(parameters, objective, lr):
    """150 updates; retain best permitted training objective, including initial state.

    The caller has no scored-fold labels. This does not perform outer-D epoch selection.
    """
    parameters = list(parameters)
    optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                                 amsgrad=False, maximize=False, foreach=False, fused=False)
    best_value = math.inf
    best_state = None
    best_step = None
    trajectory = []
    start = time.perf_counter()
    for step in range(151):
        optimizer.zero_grad(set_to_none=True)
        loss, brier = objective()
        require(torch.isfinite(loss).item(), 'Training objective nonfinite')
        loss.backward()
        gn = grad_norm(parameters)
        require(math.isfinite(gn), 'Training gradient nonfinite')
        value = float(loss.detach())
        trajectory.append({'step': step, 'objective': value, 'Brier': float(brier.detach()),
                           'gradient_l2': gn, 'penalty': value-float(brier.detach())})
        if value < best_value:
            best_value, best_step = value, step
            best_state = [p.detach().clone() for p in parameters]
        if step < 150:
            optimizer.step()
            # Temperature clipping is a callback supplied by calibrator, absent for heads.
            callback = getattr(objective, 'after_step', None)
            if callback:
                callback()
    with torch.no_grad():
        for p, best in zip(parameters, best_state):
            p.copy_(best)
    return {'updates': 150, 'retained_training_step': best_step, 'best_training_objective': best_value,
            'initial': trajectory[0], 'final_update': trajectory[-1], 'selected': trajectory[best_step],
            'trajectory': trajectory,
            'wall_seconds': time.perf_counter()-start}


def fit_head(train_x, train_skip_log, train_y, *, width, seed, regularizer):
    mu, sd = train_x.mean(0), np.maximum(train_x.std(0), 1e-6)
    x = torch.from_numpy((train_x-mu)/sd)
    skip = torch.from_numpy(train_skip_log)
    target = torch.from_numpy(np.eye(C)[train_y])
    head = ResidualHead(train_x.shape[1], width, seed)
    initial = [p.detach().clone() for p in head.parameters()]
    count = sum(p.numel() for p in head.parameters())
    def objective():
        q = head(x, skip)
        brier = ((q-target)**2).sum(-1).mean()
        penalty = sum(((p-i)**2).sum() for p, i in zip(head.parameters(), initial))/count
        return brier+regularizer*penalty, brier
    with torch.no_grad():
        initial_q = head(x, skip)
        discrepancy = float((initial_q-skip.exp()).abs().max())
    diagnostics = optimize(head.parameters(), objective, 0.01)
    diagnostics['zero_residual_skip_max_absolute_discrepancy'] = discrepancy
    diagnostics['regularizer'] = regularizer
    for state in ('initial', 'final_update', 'selected'):
        diagnostics[state]['unweighted_penalty'] = diagnostics[state]['penalty']/regularizer
    diagnostics['parameters'] = count
    return head, mu, sd, diagnostics


def serve_head(head, mu, sd, x, skip_log):
    with torch.no_grad():
        return head(torch.from_numpy((x-mu)/sd), torch.from_numpy(skip_log)).numpy()


def fit_calibrator(z, ids, y, regularizer):
    tz = torch.from_numpy(z[:, ids])
    target = torch.from_numpy(np.eye(C)[y])
    tau = torch.nn.Parameter(torch.zeros(M))
    logits_w = torch.nn.Parameter(torch.zeros(M))
    def prediction(raw):
        w = LOWER+0.95*torch.softmax(logits_w, dim=0)
        return (torch.softmax(raw/tau.exp()[:, None, None], dim=-1)*w[:, None, None]).sum(0)
    def objective():
        brier = ((prediction(tz)-target)**2).sum(-1).mean()
        w = LOWER+0.95*torch.softmax(logits_w, dim=0)
        penalty = (tau*tau).mean()+((w-0.25)**2).sum()
        return brier+regularizer*penalty, brier
    def clip():
        with torch.no_grad():
            tau.clamp_(-math.log(4), math.log(4))
    objective.after_step = clip
    diagnostics = optimize((tau, logits_w), objective, 0.03)
    diagnostics['regularizer'] = regularizer
    for state in ('initial', 'final_update', 'selected'):
        diagnostics[state]['unweighted_penalty'] = diagnostics[state]['penalty']/regularizer
    serving_start = time.perf_counter()
    with torch.no_grad():
        q = prediction(torch.from_numpy(z)).numpy()
    diagnostics['saved_score_map_serving_seconds'] = time.perf_counter()-serving_start
    diagnostics['saved_score_map_serving_seconds_per_node'] = diagnostics['saved_score_map_serving_seconds']/N
    return q, {'log_temperature': tau.detach(), 'weight_logits': logits_w.detach()}, diagnostics


def seed(bank, split, outer, operator, setting):
    value = f'amazon-moment-v1|bank={bank}|split={split}|outer={outer}|operator={operator}|setting={setting}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'big') % (2**31)
