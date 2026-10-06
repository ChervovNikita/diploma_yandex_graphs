"""Local stdlib audit of pinned saved aggregate JSON only; no raw data access."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import math
import sys

P = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
EXEC = P / 'amazon_saved_utility_target_sign_cpu_execution_root_20261006_v1'
RESULT = EXEC / 'TARGET_SIGN_DIAGNOSTIC.json'
EXPECTED = '870f80ab176d8f14593d4e622b7f6b79f29f4e4d7fc9eab913b4fb302a506bf2'
assert RESULT.stat().st_size == 1003141
assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED

def read(path):
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

result = read(RESULT)
receipt = read(EXEC / 'EXECUTE_RECEIPT.json')
release = read(EXEC / 'ROOT_RELEASE.json')
census = read(P / 'amazon_fixed_pair_S_graph_support_census_execution_root_20261006_v2/observation/amazon_fixed_pair_S_graph_support_census_execution_root_20261006_v2/census/CENSUS.json')
pairs = [(a, b) for a in range(5) for b in range(a + 1, 5)]
counts = [631, 906, 578, 231, 103]
banks = ['finite_response', 'first_order_utility']
modes = ['member_only', 'row_column_tangent']
energy_fields = {'quadratic_unsigned_without_half', 'quadratic_target_signed_without_half',
    'node_member_scale_denominator', 'edge_weight_member_scale_denominator',
    'node_normalized_unsigned', 'node_normalized_target_signed',
    'node_normalized_signed_minus_unsigned', 'edge_normalized_unsigned',
    'edge_normalized_target_signed', 'relative_orientation_contrast',
    'relative_orientation_contrast_defined', 'cross_edge_weighted_inner_product',
    'gauge_identity_residual', 'field_frobenius_squared', 'signed_field_norm_residual',
    'row_member_center_residual_max_abs', 'column_mean_max_abs_before_orientation',
    'target_weighted_signed_column_mean_max_abs'}
cell_fields = {'episode', 'classes', 'pair_nodes', 'pair_class_counts', 'graph',
    'shared_finite_reference_scale_squared', 'finite_member_centered_norm_squared',
    'source_recorded_utility_scale', 'recomputed_utility_scale',
    'source_utility_scale_minus_recomputed', 'readouts'}
graph_fields = {'undirected_nonloop_edges', 'same_target_edges', 'cross_target_edges',
    'uniform_edge_weight', 'edge_weight_mass', 'same_target_edge_weight_mass',
    'cross_target_edge_weight_mass', 'isolated_pair_nodes', 'nonisolated_pair_nodes'}
issues = []
arithmetic = {}
residuals = {}
checks = 0

def structural(condition, name):
    global checks
    checks += 1
    if not condition:
        issues.append(name)

def compare(name, actual, expected):
    global checks
    checks += 1
    if actual is None or expected is None:
        if actual is not expected:
            issues.append(name + ': null mismatch')
        return
    if not (math.isfinite(actual) and math.isfinite(expected)):
        issues.append(name + ': nonfinite')
        return
    absolute = abs(actual - expected)
    reference = max(1.0, abs(actual), abs(expected))
    row = arithmetic.setdefault(name, {'comparisons': 0, 'max_absolute_difference': 0.0,
                                       'max_difference_in_epsilon_scaled_units': 0.0})
    row['comparisons'] += 1
    row['max_absolute_difference'] = max(row['max_absolute_difference'], absolute)
    row['max_difference_in_epsilon_scaled_units'] = max(
        row['max_difference_in_epsilon_scaled_units'], absolute / (sys.float_info.epsilon * reference))
    # Fixed float64 arithmetic check only, not a scientific result-selection threshold.
    if absolute > 64 * sys.float_info.epsilon * reference:
        issues.append(name + ': exceeds fixed float64 rounding allowance')

def residual(name, value, reference=1.0):
    structural(math.isfinite(value), name + ': finite')
    row = residuals.setdefault(name, {'observations': 0, 'max_abs': 0.0, 'max_abs_over_reference': 0.0})
    row['observations'] += 1
    row['max_abs'] = max(row['max_abs'], abs(value))
    row['max_abs_over_reference'] = max(row['max_abs_over_reference'], abs(value) / reference)

structural(result['pair_episode_cells'] == len(result['cells']) == 160, 'all160 cells')
structural(result['cost_banks'] == banks and result['centering_readouts'] == modes, 'all fixed bank/mode names')
structural(result['epsilon'] == 0.001, 'fixed epsilon')
structural([(r['episode'], tuple(r['classes'])) for r in result['cells']]
           == [(e, p) for e in range(1, 17) for p in pairs], 'all ordered episode/pair identities')
structural(result['source_sha256'] == release['source_sha256'], 'release/source binding')
structural(result['inputs_sha256'] == release['inputs_sha256'], 'release/INPUTS binding')
stdout = json.loads(receipt['result']['stdout'])
structural(stdout['sha256'] == receipt['result']['aggregate']['sha256'] == EXPECTED,
           'receipt/stdout/actual aggregate SHA')
structural(stdout['bytes'] == receipt['result']['aggregate']['bytes'] == 1003141,
           'receipt/stdout/actual aggregate bytes')
structural(receipt['ssh_exit_code'] == receipt['result']['exit_code'] == 0, 'transport/child success')
structural(receipt['automatic_retry'] is False and receipt['result']['retry'] is False,
           'no retry in receipt')

graphs = {}
for cell in result['cells']:
    pair = tuple(cell['classes']); n = counts[pair[0]] + counts[pair[1]]
    structural(set(cell) == cell_fields, 'cell schema')
    structural(cell['pair_nodes'] == n, 'pair node denominator')
    structural(cell['pair_class_counts'] == {str(c): counts[c] for c in pair}, 'class denominators')
    graph = cell['graph']; saved = census['pairs'][pairs.index(pair)]['induced_exact_K']
    structural(set(graph) == graph_fields, 'graph schema')
    if pair in graphs:
        structural(graph == graphs[pair], 'episode-invariant graph metadata')
    else:
        graphs[pair] = graph
    edges, weight = graph['undirected_nonloop_edges'], graph['uniform_edge_weight']
    structural(edges == saved['undirected_distinct_nonloop_edges'], 'census edge count')
    structural(graph['same_target_edges'] == saved['same_class_edges'], 'census same-target edges')
    structural(graph['cross_target_edges'] == saved['cross_class_edges'], 'census cross-target edges')
    structural(graph['isolated_pair_nodes'] == saved['pair_isolated_nodes_in_this_graph'], 'census isolates')
    structural(graph['same_target_edges'] + graph['cross_target_edges'] == edges, 'edge coverage')
    structural(graph['isolated_pair_nodes'] + graph['nonisolated_pair_nodes'] == n, 'node coverage')
    structural(0 < weight <= 1, 'positive bounded uniform weight')
    implied_max_degree = round(1.0 / weight - 1.0)
    compare('uniform_weight_reciprocal_integer_degree', weight, 1.0 / (1.0 + implied_max_degree))
    compare('edge_weight_mass', graph['edge_weight_mass'], edges * weight)
    compare('same_target_edge_weight_mass', graph['same_target_edge_weight_mass'], graph['same_target_edges'] * weight)
    compare('cross_target_edge_weight_mass', graph['cross_target_edge_weight_mass'], graph['cross_target_edges'] * weight)
    scale2 = cell['shared_finite_reference_scale_squared']
    structural(scale2 > 0, 'positive shared finite scale')
    compare('shared_finite_scale2', scale2, 0.001 ** 2 + cell['finite_member_centered_norm_squared'] / (n * 4))
    structural(set(cell['readouts']) == set(banks), 'both banks')
    compare('utility_scale_from_member_norm', cell['recomputed_utility_scale'],
            math.sqrt(0.001 ** 2 + cell['readouts']['first_order_utility']['member_only']['field_frobenius_squared'] / (n * 4)))
    compare('source_utility_scale_discrepancy', cell['source_utility_scale_minus_recomputed'],
            cell['source_recorded_utility_scale'] - cell['recomputed_utility_scale'])
    residual('source_utility_scale_minus_recomputed', cell['source_utility_scale_minus_recomputed'])
    for bank in banks:
        records = cell['readouts'][bank]
        structural(set(records) == set(modes) | {'member_column_offset_squared', 'projection_pythagorean_residual'}, 'bank readout schema')
        un = records['member_only']; projected = records['row_column_tangent']
        structural(records['member_column_offset_squared'] >= 0, 'nonnegative offset norm')
        compare('projection_pythagorean_residual_report', records['projection_pythagorean_residual'],
                un['field_frobenius_squared'] - projected['field_frobenius_squared'] - records['member_column_offset_squared'])
        residual(bank + '.projection_pythagorean', records['projection_pythagorean_residual'], max(1.0, un['field_frobenius_squared']))
        compare('unsigned_energy_translation_invariance', un['quadratic_unsigned_without_half'], projected['quadratic_unsigned_without_half'])
        if bank == 'finite_response':
            compare('finite_member_norm', un['field_frobenius_squared'], cell['finite_member_centered_norm_squared'])
        for mode in modes:
            row = records[mode]; prefix = bank + '.' + mode + '.'
            structural(set(row) == energy_fields, 'energy readout schema')
            for field, value in row.items():
                if field != 'relative_orientation_contrast_defined' and value is not None:
                    structural(type(value) in (int, float) and math.isfinite(value), 'finite scalar ' + field)
            raw, signed = row['quadratic_unsigned_without_half'], row['quadratic_target_signed_without_half']
            structural(raw >= 0 and signed >= 0 and row['field_frobenius_squared'] >= 0, 'nonnegative energies/norm')
            node_den, edge_den = n * 4 * scale2, edges * weight * 4 * scale2
            compare('shared_node_denominator', row['node_member_scale_denominator'], node_den)
            compare('shared_edge_denominator', row['edge_weight_member_scale_denominator'], edge_den)
            compare('node_unsigned', row['node_normalized_unsigned'], raw / node_den)
            compare('node_signed', row['node_normalized_target_signed'], signed / node_den)
            compare('node_signed_minus_unsigned', row['node_normalized_signed_minus_unsigned'], (signed - raw) / node_den)
            compare('edge_unsigned', row['edge_normalized_unsigned'], raw / edge_den if edge_den > 0 else None)
            compare('edge_signed', row['edge_normalized_target_signed'], signed / edge_den if edge_den > 0 else None)
            compare('orientation_contrast', row['relative_orientation_contrast'], (signed - raw) / (signed + raw) if signed + raw > 0 else None)
            structural(row['relative_orientation_contrast_defined'] is (signed + raw > 0), 'defined contrast flag')
            compare('gauge_identity_residual_report', row['gauge_identity_residual'], signed - raw - 4 * row['cross_edge_weighted_inner_product'])
            residual(prefix + 'gauge_identity', row['gauge_identity_residual'], max(1.0, raw + signed))
            residual(prefix + 'signed_norm', row['signed_field_norm_residual'], max(1.0, row['field_frobenius_squared']))
            residual(prefix + 'row_center', row['row_member_center_residual_max_abs'])
            residual(prefix + 'column_mean', row['column_mean_max_abs_before_orientation'])
            compare('J_squared_column_mean', row['target_weighted_signed_column_mean_max_abs'], row['column_mean_max_abs_before_orientation'])

def summary(values):
    defined = [v for v in values if v is not None]
    return {'complete_cells': len(values), 'defined_cells': len(defined), 'undefined_cells': len(values)-len(defined),
            'equal_cell_mean': math.fsum(defined)/len(defined) if defined else None,
            'min': min(defined) if defined else None, 'max': max(defined) if defined else None}

def orientation(values):
    return {'signed_lower': sum(v is not None and v < 0 for v in values),
            'unsigned_lower': sum(v is not None and v > 0 for v in values),
            'equal': sum(v == 0 for v in values if v is not None),
            'undefined': sum(v is None for v in values)}

fields = ['node_normalized_unsigned', 'node_normalized_target_signed',
          'node_normalized_signed_minus_unsigned', 'relative_orientation_contrast']
series_fields = ['quadratic_unsigned_without_half', 'quadratic_target_signed_without_half',
    'node_normalized_unsigned', 'node_normalized_target_signed', 'node_normalized_signed_minus_unsigned',
    'edge_normalized_unsigned', 'edge_normalized_target_signed', 'relative_orientation_contrast']
all_summaries = {}; directions = {}; per_pair = []
for bank in banks:
    all_summaries[bank] = {}; directions[bank] = {}
    for mode in modes:
        all_summaries[bank][mode] = {}
        rows = [c['readouts'][bank][mode] for c in result['cells']]
        for field in fields:
            recomputed = summary([r[field] for r in rows])
            all_summaries[bank][mode][field] = recomputed
            original = result['equal_cell_summaries'][bank][mode][field]
            for key in ['complete_cells', 'defined_cells', 'undefined_cells']:
                structural(original[key] == recomputed[key], 'summary ' + key)
            for key in ['equal_cell_mean', 'min', 'max']:
                compare('all_cell_summary_' + key, original[key], recomputed[key])
        directions[bank][mode] = orientation([r['relative_orientation_contrast'] for r in rows])

for pair in pairs:
    cells = [c for c in result['cells'] if tuple(c['classes']) == pair]
    structural([c['episode'] for c in cells] == list(range(1,17)), 'complete per-pair episode series')
    entry = {'classes': list(pair), 'episodes': list(range(1,17)), 'pair_nodes': cells[0]['pair_nodes'],
             'graph': graphs[pair], 'implied_max_degree_from_weight': round(1 / graphs[pair]['uniform_edge_weight'] - 1),
             'shared_finite_reference_scale_squared': [c['shared_finite_reference_scale_squared'] for c in cells], 'readouts': {}}
    for bank in banks:
        entry['readouts'][bank] = {}
        for mode in modes:
            rows = [c['readouts'][bank][mode] for c in cells]
            entry['readouts'][bank][mode] = {'series': {f:[r[f] for r in rows] for f in series_fields},
                'summaries': {f:summary([r[f] for r in rows]) for f in fields},
                'orientation_counts': orientation([r['relative_orientation_contrast'] for r in rows])}
    per_pair.append(entry)

mixed = []
for cell in result['cells']:
    contrasts = {b:{m:cell['readouts'][b][m]['relative_orientation_contrast'] for m in modes} for b in banks}
    if any(v is None or v >= 0 for modes_ in contrasts.values() for v in modes_.values()):
        mixed.append({'episode':cell['episode'],'classes':cell['classes'],'contrasts':contrasts})

timing = {'internal_algorithm_seconds': result['internal_seconds'],
          'whole_child_observed_seconds': receipt['result']['whole_child_observed_seconds'],
          'transport_interval_seconds': (datetime.fromisoformat(receipt['terminal_UTC']) - datetime.fromisoformat(receipt['start_UTC'])).total_seconds(),
          'RSS_or_CPU_resource_bill_supplied': False}
recomputed = {'schema':'saved_aggregate_independent_recomputation_v1','input_sha256':EXPECTED,
    'checked_cells':160,'checked_energy_readouts':640,'structural_and_arithmetic_checks':checks,
    'issues':issues,'arithmetic_comparisons':arithmetic,'saved_residual_maxima':residuals,
    'recomputed_equal_cell_summaries':all_summaries,'orientation_counts':directions,
    'all10_complete_per_pair16_episode_series':per_pair,'mixed_opposite_or_undefined_cells':mixed,'timing':timing,
    'limits':'Raw costs/labels/topology were not redecoded. Checks independently recompute algebra and summaries from saved aggregates; field/centering residuals remain producer-reported aggregates. No independence test or empirical selection threshold.'}
target = OUT / 'RECOMPUTED_AGGREGATES.json'
with target.open('x') as stream:
    json.dump(recomputed,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')
target.chmod(0o444)
print(json.dumps({'issues':issues,'checks':checks,'orientation_counts':directions,'timing':timing,
    'residual_maxima':residuals,'max_summary_mean_difference':arithmetic['all_cell_summary_equal_cell_mean']['max_absolute_difference'],
    'recomputed_aggregate_bytes':target.stat().st_size,'recomputed_aggregate_sha256':hashlib.sha256(target.read_bytes()).hexdigest()},sort_keys=True))
