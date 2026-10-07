"""Inactive shared ROUTE denominator successor; original public full driver."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from denominator_adapter import install as install_denominator

HERE = Path(__file__).resolve().parent
PUBLIC_CONTEXT_MANIFEST_SHA = '2222beb59a2279602cc72723a6cd924ec6978fbd6a02e46a73db781b31736ed5'
FIXED_SCIENTIFIC_SEEDS = (8101, 8203, 8307)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_public_driver(root):
    root = Path(root).resolve()
    manifest = root/'MANIFEST.json'
    if sha(manifest) != PUBLIC_CONTEXT_MANIFEST_SHA:
        raise ValueError('Exact immutable public context CLI required')
    for row in json.loads(manifest.read_text())['files']:
        path = root/row['path']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Immutable public context component differs')
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('_inactive_denominator_public_driver', root/'train.py')
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def specification():
    return dict(variant='shared_route_omit_unselected_same_class_denominator', inactive_source=True,
        original_public_context_manifest_sha256=PUBLIC_CONTEXT_MANIFEST_SHA,
        wrapper_sha256=sha(__file__), objective_sha256=sha(HERE/'denominator_objective.py'),
        adapter_sha256=sha(HERE/'denominator_adapter.py'),
        same_targets_panel_initializer_own_views_coefficient_temperature=True,
        same_joint_selector_1100_horizon_serving_and_call_rules=True,
        COMMON_symmetry_claim=False, scientific_admission=False,
        no_tuning_grid=True, no_competence_or_quality_guarantee=True)


def configure_driver(driver):
    """In-memory facade successor; never alter a sealed public source file."""
    original_factory, original_recipe = driver.make_session, driver.configured_recipe
    def recipe(public, method):
        if method != 'shared_route':
            raise ValueError('One inactive shared ROUTE denominator contrast only')
        value = original_recipe(public, method)
        value['denominator_ablation'] = specification()
        return value
    def factory(public, method, seed, device, polynormer, train, targets):
        if method != 'shared_route':
            raise ValueError('One inactive shared ROUTE denominator contrast only')
        session, facade, policy, preparation = original_factory(public, method, seed, device, polynormer, train, targets)
        stats = install_denominator(session, facade, policy)
        session.config['denominator_ablation'] = specification()
        preparation = dict(preparation, denominator_ablation=stats, denominator_source=specification())
        return session, facade, policy, preparation
    driver.configured_recipe, driver.make_session = recipe, factory
    driver.public_identity = lambda method: 'inactive_context_denominator__shared_route_omit_unselected_same_class'
    return driver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable-inactive-ablation', action='store_true',
        help='Explicit execution intent only; this source packet grants no scientific admission')
    parser.add_argument('--public-context', type=Path,
        default=HERE.parents[1]/'portable_context_steering_public_interface_20261008_v1')
    parser.add_argument('--public-interface', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=FIXED_SCIENTIFIC_SEEDS, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--valid', type=Path, required=True)
    parser.add_argument('--polynormer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.enable_inactive_ablation:
        parser.error('INACTIVE: a prospective decision/protocol and explicit execution intent are required')
    driver = configure_driver(load_public_driver(args.public_context))
    saved_argv = sys.argv
    try:
        sys.argv = [str(Path(driver.__file__)), '--method', 'shared_route', '--seed', str(args.seed),
            '--device', args.device, '--public-interface', str(args.public_interface),
            '--train', str(args.train), '--valid', str(args.valid), '--polynormer', str(args.polynormer),
            '--output', str(args.output)]
        driver.main()
    finally:
        sys.argv = saved_argv


if __name__ == '__main__':
    main()
