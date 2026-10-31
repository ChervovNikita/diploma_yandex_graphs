"""Full native WikiCS four-policy source; no launch or parameter grid.

run_complete is the explicit callable full-recipe interface. The author CLI
requires a reviewed enabled release, reserved anchor source custody and no
comparison/anchor numerical reuse. Whole original12 closure gates later opening.
All numerical imports remain inside the original fresh Session/driver call.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
from permissions import POLICIES, metadata as permission_metadata, partition, require
from replay_adapter import install as install_replay

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SEEDS = (6101, 6203, 6307)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def phase_path(relative):
    path = (PHASE / relative).resolve()
    require(path != PHASE and path.is_relative_to(PHASE), 'Project-phase path required')
    return path


def bound(row):
    path = phase_path(row['path'])
    require(path.is_file() and isinstance(row['sha256'], str) and len(row['sha256']) == 64
            and sha(path) == row['sha256'], 'Exact bound source/input required: ' + row['path'])
    if 'bytes' in row:
        require(path.stat().st_size == row['bytes'], 'Bound file size changed')
    return path


def sealed(row):
    manifest = bound(row); root = manifest.parent
    for item in read(manifest)['files']:
        path = (root / item['path']).resolve()
        require(path.is_relative_to(root) and path.stat().st_size == item['bytes']
                and sha(path) == item['sha256'], 'Sealed payload changed: ' + item['path'])
    return root


def module(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)] if package else None)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def dependencies():
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    public_root = sealed(pins['public_manifest'])
    sealed(pins['constructor_manifest']); sealed(pins['attention_manifest'])
    sealed(pins['pool_adapter_manifest'])
    for key in ('public_portable', 'public_driver', 'constructor_adapter', 'public_replay', 'profile', 'native'):
        bound(pins[key])
    # The registered runtime/provider plus native source fixes this rule. No
    # unsupported independent-query projection is silently reclassified.
    for name in ('portable', 'data_interface'):
        prior = sys.modules.get(name)
        require(prior is None or Path(prior.__file__).resolve().parent == public_root,
                'Use a fresh process; a different public module is cached')
    public = module('portable', bound(pins['public_portable']))
    sys.path.insert(0, str(public_root))
    driver = module('_relation_original_full_driver', bound(pins['public_driver']))
    constructor = module('_relation_original_attention_constructor', bound(pins['constructor_adapter']))
    replay = module('_relation_original_public_replay', bound(pins['public_replay']))
    pool = module('_relation_existing_pool_adapter',
                  phase_path(pins['pool_adapter_manifest']['path']).parent / '__init__.py', package=True)
    return pins, public, driver, constructor, replay, pool


def identity(policy):
    require(policy in POLICIES, 'One fixed declared policy')
    return 'graph_relation_credit__' + policy


def descriptor(session):
    return dict(method=identity(session.relation_partition['policy']), risk_beta=.5,
        F='Two-view mean of mean-member complete TRAIN CE',
        J='Per-view .5F+.5CE(mean member class probabilities), then view mean',
        auxiliary_weight=0., no_embedding_objective=True, inference='Mean member class probabilities',
        underlying_constructor='Original be_unit_contrastive unit factory; contrastive flag explicitly disabled before replay',
        effective_model_contrastive_flag=session.model.contrastive,
        private_local_attention=session.private_local_attention.metadata(),
        partition=permission_metadata(session), replay_mode=session.relation_replay_mode,
        replay_AST_sha256=session.relation_replay_AST_sha256,
        two_disjoint_VJPs_per_member_view=True, same_old_state_one_Adam=True,
        work=dict(session.execution_totals), validation_work=dict(session.relation_validation_work),
        last_TRAIN_objectives=session.last_relation_losses,
        new_method_source_manifest_sha256=sha(HERE / 'SOURCE_MANIFEST.json'),
        train_program_sha256=sha(__file__), constructor_AST_sha256=session.private_constructor_AST_sha256,
        native_qk_shared=True, independent_Q_and_K_maps_added=False,
        global_value_factor_banks_receive_J=session.relation_partition['policy'] in ('allJ', 'phiJ'),
        shared_global_value_weights_receive_J=session.relation_partition['policy'] == 'allJ',
        owned_allocator_cap_bytes=session.relation_allocator_cap_bytes,
        competence_or_novelty_or_efficiency_guarantee=False, exact_resume_supported=False)


def make_session(public, constructor, replay, pool, pins, policy, seed, device, polynormer):
    require(policy in POLICIES and type(seed) is int, 'Explicit registered policy/integer seed')
    require(sha(polynormer) == pins['native']['sha256'], 'Pinned native Polynormer')
    # Reuse the already qualified insertion before original fresh Adam. The
    # constructor's arm suffix affects only the legacy objective flag; unit
    # weights/factors, native reset, optimizer and streams are the same factory.
    facade = constructor.adapted_public(public, phase_path(pins['attention_manifest']['path']).parent)
    session = facade.Session('wikics', 'be_unit_contrastive', seed, device, polynormer)
    require(session.steps == 0 and session.model.contrastive, 'Fresh original unit constructor')
    session.model.contrastive = False
    session.config = copy.deepcopy(session.config)
    session.config['contrastive'].update(alignment_weight=0., residual_weight=0.)
    session.relation_partition = partition(session, bound(pins['profile']), policy)
    factor_type = session.core['factors'].FactorLinear
    for value in session.model.modules():
        if isinstance(value, factor_type):
            require(session.torch.equal(value.r, session.torch.ones_like(value.r))
                    and session.torch.equal(value.s, session.torch.ones_like(value.s)), 'Exact unit dense factor start')
    for name, parameter in session.relation_partition['all']:
        if session.relation_partition['roles'][name] == 'private_local_scorer':
            require(all(session.torch.equal(parameter[0], parameter[m]) for m in range(1, 4)),
                    'All private scorer rows must copy the same native initialization')
    session.relation_pool = pool.served_pool_supervision
    session.relation_allocator_cap_bytes = None
    session.last_relation_losses = None
    session.relation_validation_work = dict(evaluations=0, member_forwards=0, dispatch_seconds=0.)
    install_replay(session, replay, bound(pins['public_replay']))
    return session


def run_complete(policy, seed, train, valid, output, polynormer, device='cpu', allocator_cap_bytes=None):
    """One fresh full task with caller paths; no shortened budget or auto-loop."""
    output = Path(output); started = time.monotonic(); box = {}
    try:
        require(policy in POLICIES and type(seed) is int and not output.exists(), 'Fresh explicit full cell')
        pins, public, driver, constructor, replay, pool = dependencies()
        dependency_setup_seconds = time.monotonic() - started
        method = identity(policy)
        original_write, original_snapshot, original_evaluate = driver.json_write, driver.joint_snapshot, driver.evaluate

        def recipe(task):
            require(task == 'wikics', 'Complete original WikiCS only')
            config = copy.deepcopy(public.recipe(task)); config['arms'] = [method]
            config['contrastive'].update(alignment_weight=0., residual_weight=0.)
            config['graph_relation_credit'] = dict(policy=policy, risk_beta=.5, auxiliary=False)
            return config

        def factory(task, arm, chosen_seed, chosen_device, native, ncn_model, ncn_utils):
            require(task == 'wikics' and arm == method and chosen_seed == seed
                    and ncn_model is None and ncn_utils is None, 'One explicitly labelled native cell')
            session = make_session(public, constructor, replay, pool, pins, policy, seed, chosen_device, native)
            session.relation_allocator_cap_bytes = allocator_cap_bytes
            session.config = recipe(task); box['session'] = session
            return session

        def annotate(value):
            session = box.get('session')
            return dict(value, method_identity=method, risk_beta=.5,
                graph_relation_credit=descriptor(session) if session else dict(policy=policy, source_prepared=True),
                source_manifest_sha256=sha(HERE / 'SOURCE_MANIFEST.json'),
                successor_inclusive_seconds=time.monotonic() - started,
                dependency_setup_seconds=dependency_setup_seconds)

        def snapshot(session, epoch, metric, per, run):
            value = original_snapshot(session, epoch, metric, per, annotate(run))
            value['graph_relation_credit'] = descriptor(session)
            return value

        def evaluate(session, train, valid):
            began = time.monotonic(); result = original_evaluate(session, train, valid)
            session.relation_validation_work['evaluations'] += 1
            session.relation_validation_work['member_forwards'] += 4
            session.relation_validation_work['dispatch_seconds'] += time.monotonic() - began
            return result

        def write(path, value):
            session = box.get('session'); name = Path(path).name
            if isinstance(value, dict) and name in ('RUN.json', 'COMPLETE.json', 'PROGRESS.json', 'FAILURE.json'):
                value = annotate(value)
                if name == 'COMPLETE.json':
                    expected = dict(shadow_member_forwards=8800, replay_member_forwards=8800,
                        output_cotangent_collections=2200, member_reverse_collections=17600,
                        optimizer_bank_updates=1100, exact_member_RNG_endpoint_checks=1100)
                    require(session.steps == value['steps'] == value['epochs'] == 1100
                            and session.execution_totals == expected
                            and session.relation_validation_work['evaluations'] == 1100
                            and session.relation_validation_work['member_forwards'] == 4400,
                            'Complete horizon and actual matched two-block work required')
            elif name == 'VALID_TRACE.json' and session is not None:
                for row in value:
                    if 'graph_relation_policy' not in row:
                        row['graph_relation_policy'] = policy
                        row['TRAIN']['supervised_objectives'] = dict(session.last_relation_losses)
            original_write(path, value)

        driver.Session, driver.recipe = factory, recipe
        driver.json_write, driver.joint_snapshot, driver.evaluate = write, snapshot, evaluate
        argv = [str(HERE / 'train.py'), '--task', 'wikics', '--arm', method, '--seed', str(seed),
                '--device', str(device), '--train', str(train), '--valid', str(valid),
                '--output', str(output), '--polynormer', str(polynormer)]
        previous = sys.argv
        try:
            sys.argv = argv
            driver.main()  # Original complete loop, data roles, joint selector and local restore.
        finally:
            sys.argv = previous
        return dict(output=str(output), method_identity=method, TEST_scoring=False)
    except BaseException as error:
        if not output.exists():
            output.mkdir(parents=True, exist_ok=False)
            value = dict(complete=False, error_type=type(error).__name__, error=str(error),
                policy=policy, seed=seed, seconds=time.monotonic() - started, automatic_retry=False,
                failure_stage='Source setup before original full driver', TEST_scoring=False)
            (output / 'FAILURE.json').write_text(json.dumps(value, indent=2) + '\n')
        raise


def reconstruct_selected(state, device='cpu'):
    """Trusted selected state reconstruction only; no optimizer/RNG resume."""
    policy = state['graph_relation_credit']['partition']['policy']
    pins, public, driver, constructor, replay, pool = dependencies()
    session = make_session(public, constructor, replay, pool, pins, policy,
                           state['run']['seed'], device, bound(pins['native']))
    saved = state['graph_relation_credit']
    require(saved['train_program_sha256'] == sha(__file__)
            and saved['new_method_source_manifest_sha256'] == sha(HERE / 'SOURCE_MANIFEST.json')
            and saved['partition'] == permission_metadata(session)
            and saved['constructor_AST_sha256'] == session.private_constructor_AST_sha256,
            'Exact selected source/partition/constructor identity')
    require(type(state['global']) is bool and state['config']['model'] == session.config['model'],
            'Explicit selected native stage/model recipe')
    session.model.load_state_dict(state['model'], strict=True)
    session.config = copy.deepcopy(state['config']); session.model.set_global(state['global']); session.model.eval()
    def serving_only(*args, **kwargs):
        raise RuntimeError('Selected reconstruction supports serving, not training/resume')
    session.train_step = serving_only
    return session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args(); require(sha(args.release) == args.release_sha256, 'Exact separate cell release')
    release = read(args.release)
    for flag in ('enabled', 'root_execution_authorized', 'source_review_approved', 'native_qualification_approved'):
        require(release.get(flag) is True, 'Disabled/unqualified scientific release: ' + flag)
    require(release['policy'] in POLICIES and release['seed'] in SEEDS
            and release['risk_beta'] == .5 and release['epochs'] == 1100
            and release['auxiliary'] is False and release['TEST_access'] is False
            and release['automatic_retry'] is False, 'Fixed full twelve-fit family only')
    require(sha(HERE / 'SOURCE_MANIFEST.json') == release['source_manifest_sha256'], 'Exact reviewed successor source')
    for row in read(HERE / 'SOURCE_MANIFEST.json')['files']:
        require(sha(HERE / row['path']) == row['sha256'], 'Changed successor payload')
    require(socket.gethostname() == 'peptide' and Path.cwd() == Path(release['runtime_repository'])
            and str(Path(sys.executable).absolute()) == release['runtime_python']
            and os.environ.get('CUDA_VISIBLE_DEVICES') == release['physical_gpu_uuid'], 'Declared normal77 host/runtime/GPU')
    protocol = read(HERE / 'PROTOCOL.json'); pins = read(HERE / 'SOURCE_BINDINGS.json')
    require(release['reserved_plain_anchors'] == protocol['reserved_plain_anchors']
            and release['train'] == pins['author_train'] and release['valid'] == pins['author_valid']
            and release['polynormer'] == pins['native'], 'Exact prospectively reserved anchors and author data/source roles')
    require(release.get('comparative_opening_authorized') is False
            and release.get('anchor_reuse_authorized') is False,
            'Training release must not authorize comparison or anchor numerical reuse')
    union = release.get('original12_union_closure')
    if union is None:
        require(release.get('original12_union_pending') is True,
                'Absent union is explicitly pending; no numerical anchor reuse')
    else:
        closure = read(bound(union))
        require(closure['complete'] is True and closure['original_fixed_cells'] == 12
                and closure['original_completed_cells'] == 10 and len(closure['new_completed']) == 2
                and closure['original_complete_false_closure_immutable'] is True,
                'Exact whole original12 union when supplied')
    for row in release['reserved_plain_anchors']:
        bound(row['release']); bound(row['completion']); bound(row['exit'])
    qualification = read(bound(release['native_qualification']))
    require(qualification['complete'] is True and qualification['source_manifest_sha256'] == release['source_manifest_sha256']
            and qualification['policies'] == list(POLICIES)
            and qualification['source_static_only'] is False
            and qualification['physical_gpu_uuid'] == release['physical_gpu_uuid'],
            'Actual successor same-GPU native qualification, not inherited fixture credit')
    import torch
    torch.cuda.set_device(0)
    cap = release['owned_GPU_allocator_cap_bytes']
    total = torch.cuda.get_device_properties(0).total_memory
    require(cap == 34359738368 and cap <= total, 'Original owned 32GiB process allocator cap')
    torch.cuda.set_per_process_memory_fraction(cap / total, 0)
    run_complete(release['policy'], release['seed'], bound(release['train']), bound(release['valid']),
                 phase_path(release['output']), bound(release['polynormer']), 'cuda:0', cap)


if __name__ == '__main__':
    main()
