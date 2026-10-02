"""Execute with stdlib only. No candidate import, Torch, data or model run."""
import ast
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path

PACKET = Path(__file__).resolve().parents[1]
PHASE = PACKET.parent
bindings = json.loads((PACKET / 'ACTIVE_SOURCE_BINDINGS.json').read_text())
active = PHASE / bindings['active_packet']
base_path = PACKET / 'base/graph_band_route_initializer.py'
candidate_path = PACKET / 'prototype/graph_band_route_initializer.py'
base_bytes, candidate_bytes = base_path.read_bytes(), candidate_path.read_bytes()
sha = lambda b: hashlib.sha256(b).hexdigest()
checks = []


def check(name, condition):
    assert condition, name
    checks.append(dict(name=name, passed=True))


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), Q(0))


def matvec(a, x):
    return [dot(row, x) for row in a]


def add(a, b):
    return [x+y for x, y in zip(a, b)]


def sub(a, b):
    return [x-y for x, y in zip(a, b)]


def scale(a, c):
    return [c*x for x in a]


def mean(rows):
    return [sum(column, Q(0))/len(rows) for column in zip(*rows)]


def mm(a, b):
    return [[dot(row, column) for column in zip(*b)] for row in a]


def matrix_linear_combination(terms):
    return [[sum((c*a[i][j] for c, a in terms), Q(0))
             for j in range(len(terms[0][1][0]))]
            for i in range(len(terms[0][1]))]


check('pinned_base_exact_bytes', sha(base_bytes) == bindings['base_sha256'])
for row in bindings['files']:
    check('active_unchanged:' + row['path'],
          sha((active / row['path']).read_bytes()) == row['sha256'])
base, candidate = ast.parse(base_bytes), ast.parse(candidate_bytes)
check('candidate_syntax', isinstance(candidate, ast.Module))
torch_fixture_bytes = (PACKET / 'fixtures/torch_synthetic_checks.py').read_bytes()
torch_fixture = ast.parse(torch_fixture_bytes)
check('authored_Torch_fixture_syntax_only', isinstance(torch_fixture, ast.Module))


def assignments(tree, name):
    return [ast.dump(n.value, include_attributes=False)
            for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]


for name in ['CAP', 'RELATIVE_FACTOR_RADIUS', 'ARMIJO_C', 'BACKTRACK_ATTEMPTS',
             'ALGEBRA_TOLERANCE', 'FUNCTION_RMS_TOLERANCE', 'g', 'base_loss',
             't64', 'lam', 'tangent', 'directions', 'dots', 'threshold',
             'pair_rms', 'functional_ok', 'quality', 'useful', 'bound', 'actual_pairs']:
    check('unchanged_core_expression:' + name,
          assignments(base, name) == assignments(candidate, name))
base_init = next(n for n in base.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'initialize_four_routes')
new_init = next(n for n in candidate.body if isinstance(n, ast.FunctionDef)
                and n.name == 'initialize_four_routes')
check('explicit_support_and_homogeneous_contract',
      [x.arg for x in new_init.args.args][-2:]
      == ['cotangent_support', 'homogeneous_full_node_outputs'])
check('no_new_label_argument', [x.arg for x in new_init.args.args if 'label' in x.arg]
      == ['train_labels'])
check('full_vjp_maps_every_output_row',
      'cotangent = band[target_nodes]' in candidate_bytes.decode())
check('TRAIN_acceptance_view_retained',
      'functions = full_functions[:, train_rows]' in candidate_bytes.decode())
check('existing_accepted_branch_condition', any(
    isinstance(n, ast.If) and ast.dump(n.test, include_attributes=False)
    == "BoolOp(op=And(), values=[Name(id='quality', ctx=Load()), Name(id='useful', ctx=Load())])"
    for n in ast.walk(new_init)))
check('diagnostic_only_extra_common_forward',
      "logits=logits_fn(theta0-alpha*g).detach()" in candidate_bytes.decode()
      and "stats['same_alpha_common_forward_calls'] += 1" in candidate_bytes.decode())
check('shared_full_q_contrast_provenance_explicit',
      "signed_graph_contrast_functional='shared_full_node_q'" in candidate_bytes.decode()
      and "initializer_cotangent_support=cotangent_support" in candidate_bytes.decode()
      and "support_matched_remask_contrast_computed=False" in candidate_bytes.decode())

# Exact four-band witness, deliberately a two-node scalar-margin abstraction.
# With normalized adjacency S swapping nodes, the middle cubic bands are zero.
# This adds full-root directions but is NOT a four-distinct-member/native pass.
I = [[Q(1), Q(0)], [Q(0), Q(1)]]
S = [[Q(0), Q(1)], [Q(1), Q(0)]]
S2, S3 = mm(S, S), mm(mm(S, S), S)
bands = [matrix_linear_combination(terms) for terms in [
    [(Q(1, 8), I), (Q(3, 8), S), (Q(3, 8), S2), (Q(1, 8), S3)],
    [(Q(3, 8), I), (Q(3, 8), S), (Q(-3, 8), S2), (Q(-3, 8), S3)],
    [(Q(3, 8), I), (Q(-3, 8), S), (Q(-3, 8), S2), (Q(3, 8), S3)],
    [(Q(1, 8), I), (Q(-3, 8), S), (Q(3, 8), S2), (Q(-1, 8), S3)],
]]
check('exact_Bernstein_partition',
      matrix_linear_combination([(Q(1), h) for h in bands]) == I)
r = [Q(1), Q(0)]
h_full = [matvec(h, r) for h in bands]
h_train = [[h[0], Q(0)] for h in h_full]
g = r
check('full_band_gradients_sum_to_TRAIN_g',
      [sum(column, Q(0)) for column in zip(*h_full)] == g)
check('remasked_band_gradients_sum_to_TRAIN_g',
      [sum(column, Q(0)) for column in zip(*h_train)] == g)
project = lambda row: sub(row, scale(g, dot(row, g)/dot(g, g)))
t_full = [project(sub(row, scale(g, Q(1, 4)))) for row in h_full]
t_train = [project(sub(row, scale(g, Q(1, 4)))) for row in h_train]
check('projected_full_support_can_add_direction',
      any(dot(row, row) > 0 for row in t_full)
      and all(dot(row, row) == 0 for row in t_train))
check('full_tangents_centered_and_TRAIN_orthogonal',
      mean(t_full) == [Q(0), Q(0)] and all(dot(row, g) == 0 for row in t_full))
q = [sub(row, scale(r, Q(1, 4))) for row in h_full]
alpha, lam = Q(1, 20), Q(3, 7)
warm = [Q(1, 3), Q(2, 3)]
common = sub(warm, scale(g, alpha))
members = [sub(common, scale(row, alpha*lam)) for row in t_full]
signed = -sum((dot(qm, sub(zm, common)) for qm, zm in zip(q, members)), Q(0))
slope = sum((dot(qm, scale(tm, lam)) for qm, tm in zip(q, t_full)), Q(0))
check('signed_finite_construction_equals_capped_linear_prediction',
      signed == alpha*slope and signed > 0)
check('signed_construction_is_not_linear_pooled_gain', mean(members) == common)
check('TRAIN_separation_gate_still_rejects_pure_unlabeled_witness',
      all(row[0] == 0 for row in t_full))
gram = [[dot(a, b)/2 for b in t_full] for a in t_full]
check('full_Gram_sees_unlabeled_output_root', sum(gram[i][i] for i in range(4)) > 0)

# A permutation in target_nodes changes row order, not the derivative.
permutation = [1, 0]
J_target = [I[v] for v in permutation]
vjp = lambda cotangent: [dot(cotangent, column) for column in zip(*J_target)]
check('full_target_row_permutation_preserves_vjp',
      [vjp([row[v] for v in permutation]) for row in h_full] == h_full)

# A nonzero remasked tangent can have a different slope under shared full-q.
# Three-node triangle; two TRAIN roots and a coupled unlabeled root.
I3 = [[Q(int(i == j)) for j in range(3)] for i in range(3)]
S3node = [[Q(0), Q(1, 2), Q(1, 2)],
          [Q(1, 2), Q(0), Q(1, 2)], [Q(1, 2), Q(1, 2), Q(0)]]
Sp2, Sp3 = mm(S3node, S3node), mm(mm(S3node, S3node), S3node)
Hp = [matrix_linear_combination(terms) for terms in [
    [(Q(1, 8), I3), (Q(3, 8), S3node), (Q(3, 8), Sp2), (Q(1, 8), Sp3)],
    [(Q(3, 8), I3), (Q(3, 8), S3node), (Q(-3, 8), Sp2), (Q(-3, 8), Sp3)],
    [(Q(3, 8), I3), (Q(-3, 8), S3node), (Q(-3, 8), Sp2), (Q(3, 8), Sp3)],
    [(Q(1, 8), I3), (Q(-3, 8), S3node), (Q(3, 8), Sp2), (Q(-1, 8), Sp3)],
]]
Jp = [[Q(1), Q(0)], [Q(0), Q(1)], [Q(1), Q(1)]]
JpT = [list(column) for column in zip(*Jp)]
rp = [Q(1), Q(2), Q(0)]
gp = matvec(JpT, rp)
qp_full = [sub(matvec(hp, rp), scale(rp, Q(1, 4))) for hp in Hp]
qp_remask = [row[:2]+[Q(0)] for row in qp_full]
hp_remask = [matvec(JpT, row) for row in qp_remask]
tp_remask = [sub(row, scale(gp, dot(row, gp)/dot(gp, gp))) for row in hp_remask]
fp_remask = [matvec(Jp, row) for row in tp_remask]
full_q_slope = sum((dot(qm, fm) for qm, fm in zip(qp_full, fp_remask)), Q(0))
matched_remask_slope = sum((dot(qm, fm) for qm, fm in zip(qp_remask, fp_remask)), Q(0))
check('shared_full_q_is_not_remasked_norm_squared', full_q_slope != matched_remask_slope)
check('support_matched_remask_uncapped_slope_is_projected_norm_squared',
      matched_remask_slope == sum((dot(tm, tm) for tm in tp_remask), Q(0))
      and matched_remask_slope > 0)

# Exact detached complementary soft-target reduction; no CE numeric evaluation.
p = [Q(1, 2), Q(1, 2)]
eta = Q(1, 4)
class_q = [[[-value/2, value/2] for value in qm] for qm in q]
targets = [[sub(p, scale(qrow, eta)) for qrow in qm] for qm in class_q]
check('proper_complementary_targets_exist_for_small_eta', all(
    sum(row, Q(0)) == 1 and min(row) >= 0 for qm in targets for row in qm))
check('soft_CE_warm_derivative_is_exact_cotangent', all(
    sub(p, target) == scale(qrow, eta)
    for qm, tm in zip(class_q, targets) for qrow, target in zip(qm, tm)))
bad_target = sub(p, scale(class_q[0][1], Q(3)))
clipped = [max(Q(0), x) for x in bad_target]
clipped = scale(clipped, 1/sum(clipped, Q(0)))
check('clipping_invalid_targets_changes_direction',
      min(bad_target) < 0 and sub(p, clipped) != scale(class_q[0][1], Q(3)))

# General frozen-linear squared-loss null with the SAME fixed preconditioner.
# Average member parameters/outputs equal common descent at every step, although
# each member may differ and its average individual squared loss can increase.
A = [[Q(1), Q(2)], [Q(3), Q(-1)]]
AT = [list(column) for column in zip(*A)]
P = [[Q(2), Q(1)], [Q(1), Q(3)]]
y = [Q(1), Q(-2)]
learning_rate = Q(1, 30)
quadratic_gradient = lambda theta: matvec(AT, sub(matvec(A, theta), y))
update = lambda theta: sub(theta, scale(matvec(P, quadratic_gradient(theta)), learning_rate))
reference = common
states = members
for step in range(15):
    reference, states = update(reference), [update(state) for state in states]
    check('frozen_linear_pooled_null:step' + str(step),
          mean(states) == reference and mean([matvec(A, state) for state in states]) == matvec(A, reference))
    risks = [dot(sub(matvec(A, state), y), sub(matvec(A, state), y))/2 for state in states]
    common_risk = dot(sub(matvec(A, reference), y), sub(matvec(A, reference), y))/2
    check('frozen_linear_mean_member_loss_not_better:step' + str(step),
          sum(risks, Q(0))/4 >= common_risk)

for row in bindings['files']:
    check('active_still_unchanged:' + row['path'],
          sha((active / row['path']).read_bytes()) == row['sha256'])
result = dict(schema='full-node-cotangent-stdlib-checks-v1',
              executed_stdlib_only=True, torch_imported=False,
              candidate_imported=False, torch_numerical_tests_run=False,
              torch_fixture_syntax_checked=True,
              torch_fixture_sha256=sha(torch_fixture_bytes),
              actual_model_or_data_run=False, base_sha256=sha(base_bytes),
              candidate_sha256=sha(candidate_bytes), checks=checks,
              exact_signed_construction_value=str(signed),
              exact_first_order_slope=str(slope),
              cross_support_witness_shared_full_q_slope=str(full_q_slope),
              cross_support_witness_matched_remask_slope=str(matched_remask_slope),
              witness_limit='Two-node scalar margin; duplicate/zero band tangents. Existing TRAIN separation would reject it.',
              frozen_linear_limit='Fixed common J, squared loss, ordinary SGD or shared fixed linear preconditioner only; no CE/nonlinear/Adam guarantee.')
(PACKET / 'STDLIB_RESULTS.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
print(json.dumps(dict(checks=len(checks), passed=True, candidate_sha256=sha(candidate_bytes),
                      torch_numerical_tests_run=False)))
