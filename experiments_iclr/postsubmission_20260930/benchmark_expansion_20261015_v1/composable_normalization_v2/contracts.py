"""Explicit new-job composition contract; no descriptor or roster is written."""
import copy
import hashlib
import json

SCHEMA = 'native-route-BN-composition-v1'
SUFFIX = '/normalization-route-private-v1'
POLICY = 'private_native_running_state_shared_affine_v1'
CONTROL = 'gnnm_private_bn_state_v3'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def eligible(parent, roster):
    method = parent.get('scientific_method_id')
    cell = next((row for row in roster['cells'] if row['id'] == parent.get('recipe_id')), None)
    if cell is None or CONTROL not in cell['shared_method_ids'] or method not in cell['shared_method_ids'] \
            or method in ('gnnm_boundary_native', CONTROL):
        raise ValueError('Only declared native-BN candidate/control applications may migrate')
    if parent.get('family') != cell['task_family'] or parent.get('TEST_access') is not False \
            or 'normalization_composition' in parent \
            or parent.get('policy', {}).get('track') == 'source_reference':
        raise ValueError('One new matched shared fit with TEST closed is required')
    return cell


def migrated_job(parent, declaration, roster):
    """Pure proposed successor; production authority remains outside this API."""
    eligible(parent, roster)
    required = {'schema', 'policy', 'source_manifest_sha256', 'banks_sha256',
                'parent_descriptor', 'parent_job_sha256', 'matched_control_method_id'}
    if set(declaration) != required or declaration['schema'] != SCHEMA \
            or declaration['policy'] != POLICY or declaration['matched_control_method_id'] != CONTROL \
            or declaration['parent_job_sha256'] != digest(parent):
        raise ValueError('Exact source, parent and matched-normalization declaration required')
    for key in ('source_manifest_sha256', 'banks_sha256', 'parent_job_sha256'):
        if not isinstance(declaration[key], str) or len(declaration[key]) != 64 \
                or any(c not in '0123456789abcdef' for c in declaration[key]):
            raise ValueError('Exact SHA256 binding required')
    result = copy.deepcopy(parent)
    result['id'] += SUFFIX
    result['normalization_composition'] = copy.deepcopy(declaration)
    # Native settings, loss, gradients, RNG, selection and source ancestry stay
    # intact; the new expanded contract states the added normalization policy.
    if 'expanded_contract' in result:
        result['expanded_contract']['id'] = result['id']
        result['expanded_contract']['normalization_composition'] = copy.deepcopy(declaration)
        result['expanded_contract_hash'] = digest(result['expanded_contract'])
    return result


def migrated_controls(row, fit_id_map, matched_control_fit):
    """Rebind ingredients; retain original G0 as a separate total comparison."""
    if row['fit_id'] not in fit_id_map:
        return copy.deepcopy(row)
    result = copy.deepcopy(row)
    result['fit_id'] = fit_id_map[row['fit_id']]
    for link in result['controls']:
        link['fit_ids'] = [fit_id_map.get(value, value) for value in link['fit_ids']]
        if link.get('resolved_control') == 'gnnm_boundary_native':
            link['comparison_role'] = 'total_extension_vs_original_GNNM'
    matched = {'symbolic_control': 'matched_normalization_baseline',
               'resolved_control': CONTROL, 'kind': 'paired_shared_fit',
               'fit_ids': [matched_control_fit], 'comparison_role': 'primary_ingredient_attribution'}
    result['controls'].append(matched)
    result['normalization_attribution_reference'] = copy.deepcopy(matched)
    return result
