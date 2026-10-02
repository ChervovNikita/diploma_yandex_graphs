"""Executed stdlib source/symbolic checks only; no candidate import or Torch."""
import ast
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path

packet = Path(__file__).resolve().parents[1]
bindings = json.loads((packet / 'SOURCE_BINDINGS.json').read_bytes())
phase = Path(bindings['path_root'])
sha = lambda data: hashlib.sha256(data).hexdigest()
checks = []


def check(name, condition):
    assert condition, name
    checks.append(dict(name=name, passed=True))


def preserved(when):
    for row in bindings['preserved_v1_files']:
        check(when+':v1:'+row['path'], sha((phase / bindings['preserved_v1_packet']
                                         / row['path']).read_bytes()) == row['sha256'])
    for row in bindings['active_files']:
        check(when+':active:'+row['path'], sha((phase / bindings['active_packet']
                                             / row['path']).read_bytes()) == row['sha256'])


preserved('before')
base_bytes = (packet / 'base/graph_band_route_initializer.py').read_bytes()
source_bytes = (packet / 'prototype/paired_shared_alpha_initializer.py').read_bytes()
fixture_bytes = (packet / 'fixtures/torch_paired_checks.py').read_bytes()
check('reused_v1_primitive_exact_bytes', sha(base_bytes) == bindings['reused_primitive_sha256'])
base_tree, tree, fixture_tree = map(ast.parse, (base_bytes, source_bytes, fixture_bytes))
check('paired_source_syntax', isinstance(tree, ast.Module))
check('Torch_fixture_syntax_only', isinstance(fixture_tree, ast.Module))
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'initialize_paired_four_arms')
check('exact_named_paired_API', [n.arg for n in function.args.args] == [
    'logits_fn', 'theta0', 'S', 'S_permuted', 'target_nodes', 'train_rows', 'train_labels',
    'homogeneous_full_node_outputs'])
check('only_compact_TRAIN_labels', [n.arg for n in function.args.args if 'label' in n.arg]
      == ['train_labels'])
text = source_bytes.decode()
for name, snippet in [
    ('explicit_no_fallback', 'fallback_enabled=False'),
    ('common_grid', 'for attempt in range(base.BACKTRACK_ATTEMPTS)'),
    ('common_radius_bound', 'bound_norm = math.sqrt(1+base.CAP**2)*float(g64.norm())'),
    ('alpha0_policy', 'alpha0 = base.RELATIVE_FACTOR_RADIUS*math.sqrt(theta0.numel())/bound_norm'),
    ('half_grid', 'alpha = alpha0/(2**attempt)'),
    ('joint_success_only', "joint_accepted = all(row['accepted'] for row in arm_records.values())"),
    ('explicit_joint_failure', "reason='shared_alpha_grid_exhausted'"),
    ('same_alpha_common_reuse', 'common_z = None if common_zs is None else common_zs[0]'),
    ('full_output_row_mapping', 'cotangent = band[target_nodes]'),
    ('original_remask_mapping', 'cotangent[train_rows] = band[target_nodes[train_rows]]'),
    ('trial_exception_receipts', "errors.setdefault(name, []).append(dict("),
    ('geometry_error_receipt', 'self.report = report'),
]:
    check(name, snippet in text)


def assignments(parsed, name):
    return [ast.dump(n.value, include_attributes=False) for n in ast.walk(parsed)
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                 for t in n.targets)]


class BaseConstantNames(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'base' and node.attr.isupper():
            return ast.Name(id=node.attr, ctx=ast.Load())
        return self.generic_visit(node)


constant_normalized_tree = BaseConstantNames().visit(ast.parse(source_bytes))
check('pinned_projection_centering_expressions',
      assignments(tree, 't64')[:4] == assignments(base_tree, 't64')[:4])
check('pinned_TRAIN_CE_quality_expression',
      assignments(tree, 'quality') == assignments(base_tree, 'quality'))
check('pinned_Armijo_bound_expression',
      assignments(constant_normalized_tree, 'bound') == assignments(base_tree, 'bound'))
check('four_authored_CPU_tests', len([n for n in ast.walk(fixture_tree)
                                    if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')]) == 4)
check('independent_reference_and_failure_authored', all(s in fixture_bytes.decode() for s in [
    'Analytic full-output Jacobian', 'plus @ plus @ plus', 'test_forced_joint_failure',
    'test_rejected_forward_exceptions', 'test_geometry_exception']))
native = json.loads((packet / 'NATIVE_SOURCE_BINDINGS.json').read_bytes())
for row in native['source_files']+native['source_receipts']+[native['graph_input_metadata_receipt']]:
    check('native_source_metadata_binding:'+row['path'],
          sha((phase / row['path']).read_bytes()) == row['sha256'])
active_hashes = {row['path']: row['sha256'] for row in bindings['active_files']}
check('native_recipe_binds_active_v3_precision_source',
      native['prior_context_source_bindings']['sha256'] == active_hashes['SOURCE_BINDINGS.json'])
check('native_recipe_binds_active_v3_precision_protocol',
      native['prior_context_protocol']['sha256'] == active_hashes['PROTOCOL.json'])

# Exact radius proof: orthogonal zero-mean tangents under one Frobenius bound.
g = [Q(3), Q(4)]
tangents = [[Q(1), Q(-3, 4)], [Q(-1), Q(3, 4)], [Q(0), Q(0)], [Q(0), Q(0)]]
dot = lambda a, b: sum((x*y for x, y in zip(a, b)), Q(0))
g2, cap, radius, dimension = dot(g, g), Q(1, 2), Q(1, 100), 2
check('exact_orthogonality', all(dot(g, t) == 0 for t in tangents))
check('exact_zero_mean', all(sum(column, Q(0)) == 0 for column in zip(*tangents)))
check('exact_Frobenius_bound', sum((dot(t, t) for t in tangents), Q(0)) <= cap**2*g2)
bound2 = (1+cap**2)*g2
alpha0_squared = radius**2*dimension/bound2
for index, t in enumerate(tangents+[[Q(0), Q(0)]]):
    direction = [-x-y for x, y in zip(g, t)]
    check('exact_route_bound:'+str(index), dot(direction, direction) <= bound2)
    check('exact_shared_alpha_radius:'+str(index),
          alpha0_squared*dot(direction, direction) <= radius**2*dimension)
check('six_shared_grid_steps_nested', all(Q(1, 2**(k+1)) < Q(1, 2**k) for k in range(5)))
preserved('after')
result = dict(schema='paired-shared-alpha-stdlib-checks-v1', checks=checks,
              executed_stdlib_only=True, paired_source_imported=False, torch_imported=False,
              torch_fixture_syntax_checked=True, torch_fixture_executed_by_author=False,
              scientific_model_data_or_GPU_execution=False,
              paired_source_sha256=sha(source_bytes), torch_fixture_sha256=sha(fixture_bytes))
(packet / 'STDLIB_RESULTS.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
print(json.dumps(dict(passed=True, checks=len(checks), source_sha256=sha(source_bytes),
                      torch_fixture_executed=False)))
