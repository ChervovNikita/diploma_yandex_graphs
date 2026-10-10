"""Disabled full1100/100 native M1 reference, including one private I4 constituent."""
import argparse
import copy
import resource
from pathlib import Path
import sys
import time

import common as g
import reference


def dependencies():
    pins = g.source_gate()
    root = g.bound(pins['public_manifest']).parent
    for name in ('portable', 'data_interface'):
        prior = sys.modules.get(name)
        g.require(prior is None or Path(prior.__file__).resolve().parent == root,
                  'Fresh process or exact original public module')
        if prior is None:
            g.module(root/(name+'.py'), name)
    public = sys.modules['portable']
    driver = g.module(g.bound(pins['public_driver']), '_matched_native_original_full_driver')
    row_law = g.module(g.bound(pins['candidate_initializer']), '_matched_native_exact_row_seed_law')
    return pins, public, driver, row_law


def recipe(public):
    config = copy.deepcopy(public.recipe('wikics'))
    config['arms'] = ['single']
    config['contrastive'].update(alignment_weight=0., residual_weight=0.)
    return config


def make_session(pins, public, row_law, spec, device, polynormer):
    g.spec_check(spec)
    facade = reference.adapted_public(public, spec['paired_seed'], spec['member_index'],
        row_law.row_seed, pins['public_portable']['sha256'])
    session = facade.Session('wikics', 'single', spec['body_seed'], device, polynormer)
    session.config = recipe(public)
    reference.install_accounting(session)
    return session


def describe(session, spec):
    return reference.descriptor(session, spec, g.sha(g.HERE/'SOURCE_MANIFEST.json'), g.sha(__file__))


def run_complete(spec, train, valid, output, polynormer, device, *, later_execution_authorized=False):
    g.require(later_execution_authorized is True, 'Inactive full-fit callable; separate root admission')
    g.spec_check(spec)
    started = time.monotonic(); before = resource.getrusage(resource.RUSAGE_SELF)
    pins, public, driver, row_law = dependencies()
    box, failure = {}, None
    original_snapshot, original_write, original_evaluate = driver.joint_snapshot, driver.json_write, driver.evaluate

    def factory(task, arm, body_seed, chosen_device, native, ncn_model, ncn_utils):
        g.require(task == 'wikics' and arm == 'single' and body_seed == spec['body_seed']
                  and ncn_model is None and ncn_utils is None, 'One declared native M1 body')
        session = make_session(pins, public, row_law, spec, chosen_device, native)
        box['session'] = session
        return session

    def evaluate(session, train_data, valid_data):
        pooled_metric, per = original_evaluate(session, train_data, valid_data)
        g.require(len(per) == 1, 'Exactly one native body evaluated')
        work = session.matched_reference_work
        work['VALID_evaluations'] += 1; work['VALID_member_forwards'] += 1
        # Original I4 own_best/own_local uses per[m], not the pooled metric.
        # Serial M1 trajectories use that exact source selector for I4 constituents.
        metric = per[0] if spec['family'] == 'native_independent4' else pooled_metric
        return metric, per

    def snapshot(session, epoch, metric, per, run):
        state = original_snapshot(session, epoch, metric, per, run)
        state['matched_reference'] = describe(session, spec)
        state['matched_reference']['selector_metric_origin'] = (
            'original independent4 per[member] raw-logit accuracy' if spec['family'] == 'native_independent4'
            else 'original single M1 pooled-probability accuracy')
        return state

    def write(path, value):
        session = box.get('session')
        if isinstance(value, dict) and Path(path).name in ('RUN.json', 'COMPLETE.json', 'PROGRESS.json', 'FAILURE.json'):
            value = dict(value, matched_reference=describe(session, spec) if session else dict(spec=spec),
                         successor_inclusive_seconds=time.monotonic()-started)
            if Path(path).name == 'COMPLETE.json':
                g.require(value['epochs'] == value['steps'] == session.steps == 1100,
                          'Original complete1100 horizon')
                expected = dict(TRAIN_attempts=1100, completed_TRAIN_updates=1100,
                    actual_TRAIN_member_view_forwards=2200, differentiated_own_CE_view_terms=2200,
                    original_total_backward_invocations=1100, optimizer_steps=1100,
                    VALID_evaluations=1100, VALID_member_forwards=1100)
                g.require(session.matched_reference_work == expected, 'Complete original two-view M1 work')
        original_write(path, value)

    driver.Session = factory
    driver.recipe = lambda task: recipe(public) if task == 'wikics' else (_ for _ in ()).throw(ValueError('WikiCS only'))
    driver.evaluate, driver.joint_snapshot, driver.json_write = evaluate, snapshot, write
    argv = [str(g.HERE/'train.py'), '--task', 'wikics', '--arm', 'single', '--seed', str(spec['body_seed']),
            '--device', str(device), '--train', str(train), '--valid', str(valid), '--output', str(output),
            '--polynormer', str(polynormer)]
    previous = sys.argv
    try:
        sys.argv = argv
        driver.main()  # Original full loop, strict selector and coherent M1 local model/Adam restore.
    except BaseException as error:
        failure = dict(type=type(error).__name__, error=str(error), automatic_retry=False, partial_work_retained=True)
        raise
    finally:
        sys.argv = previous
        after = resource.getrusage(resource.RUSAGE_SELF)
        session = box.get('session')
        cost = dict(schema='matched-native-M1-full-fit-cost-v1', spec=spec,
            inclusive_wall_seconds=time.monotonic()-started, CPU_user_seconds=after.ru_utime-before.ru_utime,
            CPU_system_seconds=after.ru_stime-before.ru_stime,
            peak_RSS_bytes=int(after.ru_maxrss*(1024 if sys.platform.startswith('linux') else 1)),
            work=dict(session.matched_reference_work) if session else None,
            peak_CUDA_allocated_bytes=session.torch.cuda.max_memory_allocated(session.device) if session and session.device.type == 'cuda' else None,
            peak_CUDA_reserved_bytes=session.torch.cuda.max_memory_reserved(session.device) if session and session.device.type == 'cuda' else None,
            initializer_seconds=session.matched_native_start['elapsed_seconds'] if session else None,
            failure=failure, source_manifest_sha256=g.sha(g.HERE/'SOURCE_MANIFEST.json'),
            original_recipe_and_selector_preserved=True, external_owner_terminal_and_mirroring_cost=None,
            TEST_access=False, automatic_retry=False)
        output = Path(output)
        if output.is_dir():
            g.write(output/'COST.json', cost)


def reconstruct_selected(state, device='cpu'):
    saved = state['matched_reference']; spec = saved['spec']; g.spec_check(spec)
    pins, public, _, row_law = dependencies()
    session = make_session(pins, public, row_law, spec, device, g.bound(pins['native']))
    current = describe(session, spec)
    for key in ('spec', 'source_manifest_sha256', 'train_program_sha256', 'native_scorer_start',
                'constructor_AST_sha256', 'core', 'native', 'body_count', 'optimizer_count', 'dense_parameterization'):
        g.require(saved[key] == current[key], 'Exact selected reference source/native start')
    g.require(state['run']['task'] == 'wikics' and state['run']['arm'] == 'single'
              and state['run']['seed'] == spec['body_seed'] and type(state['global']) is bool
              and state['config'] == recipe(public), 'Original own-selected configuration/mode')
    session.model.load_state_dict(state['model'], strict=True)
    session.model.set_global(state['global']); session.model.eval()

    def refusal(*args, **kwargs):
        raise RuntimeError('Selected reference is serving-only; no optimizer/RNG/history resume')

    session.train_step = refusal
    return session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True); parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args(); release, pins = g.admission(args.release, args.release_sha256)
    g.require(release['schema'] == 'matched-native-reference-M1-release-v1', 'Exact M1 body release')
    spec = release['spec']; g.spec_check(spec)
    g.require(release['epochs'] == 1100 and release['local_epochs'] == 100
              and release['train'] == pins['author_train'] and release['valid'] == pins['author_valid']
              and release['polynormer'] == pins['native']
              and release['physical_gpu_uuid'] == pins['GPU_per_seed'][str(spec['paired_seed'])]
              and release['owned_GPU_allocator_cap_bytes'] == 32*1024**3,
              'Fixed source/roles/route/full horizon/cap')
    expected = 'combination_matched_native_reference_execution_root_20261010_v1/seed%d/%s/m%d' % (
        spec['paired_seed'], spec['family'], spec['member_index'])
    g.require(release['output'] == expected, 'Fixed fresh body slot; no reassignment')
    output = g.inside(expected)
    g.require(not output.exists() and output.parent.is_dir(), 'Fresh prepared own output; no retry')
    import torch
    torch.cuda.set_device(0)
    cap = release['owned_GPU_allocator_cap_bytes']; total = torch.cuda.get_device_properties(0).total_memory
    g.require(total >= cap, 'Admitted allocator cap fits physical GPU')
    torch.cuda.set_per_process_memory_fraction(cap/total, 0)
    run_complete(spec, g.bound(release['train']), g.bound(release['valid']), output,
                 g.bound(release['polynormer']), 'cuda:0', later_execution_authorized=True)


if __name__ == '__main__':
    main()
