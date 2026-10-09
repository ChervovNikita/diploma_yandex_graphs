"""Stdlib source/hash/policy checks only; no numerical or scientific execution."""
import argparse
import ast
import hashlib
import importlib.util
import importlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def policy_fixtures():
    spec = importlib.util.spec_from_file_location('_pilot_policy_source_only_fixture', HERE / 'control_policy.py')
    policy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = policy
    spec.loader.exec_module(policy)  # Only this standard-library policy module.
    class Scalar(float):
        def mean(self):
            return self
    def key(kind, member, source=None):
        return (kind, member, source)
    anchor = dict(J=Scalar(-.4), full_pool=Scalar(1.8))
    for m in range(4):
        anchor['full_own:' + str(m)] = Scalar(.8 + m/10)
    for m in range(3):
        anchor['probe_own:' + str(m)] = Scalar(1. + m/10)
        anchor['absent:' + str(m)] = Scalar(1.4 + m/10)
        anchor['source_j:' + str(m)] = Scalar(-.3 - m/10)
    original = lambda *args: dict(anchor)
    helper = SimpleNamespace(_terms=original, OutputKey=key,
        _observed_logp=lambda value, *args: value,
        _mean_pool_nll=lambda values, config: Scalar(-math.log(sum(math.exp(float(v)) for v in values)/len(values))))
    outputs = {key('train_full', m): Scalar(-.5 - m/10) for m in range(4)}
    expected = dict(native_pool_credit=-math.log(sum(math.exp(float(v)) for v in outputs.values())/4),
        source_view_supervision=1.1, uncoupled_source_contrast=-.5)
    for arm in policy.ARMS:
        with policy.objective_scope(helper, arm):
            result = helper._terms(outputs, {}, None, policy.ASSIGNMENTS, None)
            target = expected.get(arm, -.4)
            require(abs(float(result['J'])-target) < 1e-12, 'Declared scalar arithmetic: ' + arm)
            risks = [name for name in anchor if name.startswith(('full_own:', 'probe_own:', 'absent:'))] + ['full_pool']
            require(len(risks) == 11 and all(result[name] == anchor[name] for name in risks), 'All eleven risks preserved')
            if arm in expected:
                require(all(result['source_j:' + str(m)] == result['J'] for m in range(3)), 'Controls repeat global objective guard')
        require(helper._terms is original, 'Scope restores actual helper')
    try:
        with policy.objective_scope(helper, 'native_pool_credit'):
            raise ValueError('fixture')
    except ValueError:
        pass
    require(helper._terms is original, 'Exception restores helper')
    class Base:
        assignments = policy.ASSIGNMENTS
        def require_enabled(self):
            pass
    class Session:
        config = Base()
        counters = {'own_epochs': 0}
        def __init__(self):
            self.counters = dict(own_epochs=0)
            self.helper = helper
        def train_epoch(self):
            self.counters['own_epochs'] += 1
            return {}
        def correct(self, displacement):
            return dict(status='zero', private_L2_dose=0.)
    for arm in policy.ARMS:
        session = Session()
        scheduled = policy.ScheduledSession(session, arm, {})
        completed = []
        for epoch in range(200):
            scheduled.train_epoch()
            value = scheduled.correct({})
            if value['status'] == 'zero':
                completed.append(epoch+1)
        require(completed == ([] if arm == 'shared_own_only' else list(range(15, 201, 5))), 'Actual completed-update cadence')
        require(session.config.assignments == policy.ASSIGNMENTS, 'Transient assignment restored')
        require(scheduled.scheduled_slots == 38 and scheduled.opportunities == (0 if arm == 'shared_own_only' else 38), 'Scheduled slots separated from actual source work')
        require(scheduled.zero == (0 if arm == 'shared_own_only' else 38) and scheduled.accepted == scheduled.rejected == 0, 'All actual statuses counted')
        if arm == 'COMMON_cycle':
            require(scheduled.exposures == dict(actor=39, director=39, keyword=36), 'Honest partial-cycle exposure counts')
    return dict(control_scalar_arithmetic_and_scope_restore=True, exact_11_anchor_preservation=True,
                completed_update_schedule_and_partial_COMMON_cycle=True, numerical_gradient_or_cone_qualification=False)


def check():
    sources = read(HERE / 'SOURCE_BINDINGS.json')
    require(sources['backend_bindings_pending'] is False, 'Exact reviewed backend binding must precede final check')
    for row in sources['files']:
        path = (HERE.parent / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE.parent) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Exact immutable source dependency')
    numerical = {'torch', 'numpy', 'dgl', 'torch_sparse', 'sklearn'}
    for path in HERE.glob('*.py'):
        tree = ast.parse(path.read_text())
        if path.name == 'family_driver.py':
            config = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PilotConfig')
            defaults = {node.target.id: ast.literal_eval(node.value) for node in config.body
                        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
            require(all(defaults[name] is False for name in ('enabled', 'root_source_review_approved',
                'root_scientific_release_approved', 'native_and_independent_reference_competence_adopted')), 'Actual callable defaults remain inactive')
        elif path.name == 'independent4_qualification.py':
            config = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'QualificationConfig')
            defaults = {node.target.id: ast.literal_eval(node.value) for node in config.body
                        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
            require(defaults['enabled'] is defaults['root_source_review_approved'] is False, 'Independent4 qualification default is inactive')
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                require(all(alias.name.split('.')[0] not in numerical for alias in node.names), 'No numerical import')
            elif isinstance(node, ast.ImportFrom):
                require((node.module or '').split('.')[0] not in numerical, 'No numerical provider import')
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'load' and isinstance(node.func.value, (ast.Name, ast.Subscript)):
                if isinstance(node.func.value, ast.Name) and node.func.value.id not in ('torch',):
                    continue
                keywords = {item.arg: ast.literal_eval(item.value) for item in node.keywords}
                require(keywords == {'map_location':'cpu', 'weights_only':True}, 'Every owned numerical checkpoint read is safe')
    for path in HERE.glob('*.json'):
        read(path)
    protocol, release = read(HERE / 'PROTOCOL.json'), read(HERE / 'RELEASE.disabled.json')
    require(protocol['enabled'] is False and release['enabled'] is False and release['root_source_review_approved'] is False
        and release['root_scientific_release_approved'] is False, 'Prospective default-disabled packet')
    require(protocol['declared_candidate'] == 'assigned_source_supply' and len(protocol['arms']) == 6
        and protocol['pairs'] == [{'outer_role_seed':s,'base_seed':s,'member_and_independent_body_seeds':[1000*s+m+1 for m in range(4)]} for s in (1,2,3)], 'Fixed complete paired conditions')
    require(protocol['schedule']['warmup_actual_own_epochs'] == 10 and protocol['schedule']['first_correction_after_completed_own_update'] == 15
        and protocol['schedule']['cadence_completed_updates'] == 5 and protocol['schedule']['maximum_epochs'] == 200, 'Task-native prospective amendment')
    review = read(HERE / 'SOURCE_REVIEW.json')
    require(review['control_source_sha256'] == sha(HERE / 'control_policy.py')
        and review['diagnostics_source_sha256'] == sha(HERE / 'selected_diagnostics.py')
        and review['whole_family_and_independent_drivers_reviewed_in_this_focused_audit'] is False,
        'Exact focused audit scope, no whole-driver review inference')
    independent_release = read(HERE / 'INDEPENDENT4_QUALIFICATION_RELEASE.disabled.json')
    require(independent_release['enabled'] is False and independent_release['root_source_review_approved'] is False
        and independent_release['body_seeds'] == [1001,1002,1003,1004] and independent_release['actual_body_updates'] == 8
        and independent_release['native_epochs_per_body'] == 1 and independent_release['quality_scoring'] is False,
        'Minimal fixed native independent4 qualification release')
    driver_text = (HERE / 'family_driver.py').read_text()
    driver_tree = ast.parse(driver_text)
    shared = next(node for node in driver_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_shared_family')
    independent_text = (HERE / 'independent_controls.py').read_text()
    independent_tree = ast.parse(independent_text)
    reference = next(node for node in independent_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_independent_family')
    require('config.require_enabled()' in ast.get_source_segment(driver_text, shared)
        and 'config.require_reference_fit_enabled()' in ast.get_source_segment(independent_text, reference)
        and 'self.native_and_independent_reference_competence_adopted is True' in driver_text,
        'Reference fits can establish pending competence; shared fits retain stronger admission')
    result = policy_fixtures()
    providers_before = set(sys.modules)
    sys.path.insert(0, str(HERE))
    try:
        for name in ('family_driver', 'independent_controls', 'independent4_qualification'):
            module = importlib.import_module(name)
            require(Path(module.__file__).resolve().parent == HERE, 'Exact local default import')
        require(not {name.split('.')[0] for name in set(sys.modules)-providers_before} & numerical, 'Default imports add no numerical provider')
    finally:
        sys.path.pop(0)
    result.update(status='passed', source_AST_hash_and_stdlib_policy_only=True, default_disabled=True,
        actual_local_stdlib_default_imports=True,
        complete_paired_family_source_declarations=True, no_Torch_graph_model_data_provider_host_or_quality_execution=True,
        selected_U_D_and_repair_harm_numerical_qualification=False, runtime_or_scientific_admission=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check()
    if args.output:
        require(args.output.resolve().parent == HERE and not args.output.exists(), 'Fresh source-only receipt')
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
