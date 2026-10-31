"""Enabled engineering release only: one real TRAIN update per policy/stage.

No fabricated graph, VALID metric, prediction readout, training family or
coefficient search. This is a native-path qualification entry, not a result.
Numerical imports occur only after a separate reviewed release passes.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
from train import HERE, PHASE, POLICIES, bound, dependencies, make_session, phase_path, read, sha, require, admission


def permission_metadata(session):
    return sys.modules["permissions"].metadata(session)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    require(sha(args.release) == args.release_sha256, 'Exact separate qualification release')
    release = read(args.release)
    admission(release, qualification=True, release_path=args.release)
    require(release['schema'] == 'independent-native-local-scorer-qualification-release-v1'
        and release['VALID_scores_read'] is False, 'Exact disabled engineering-only release')
    require(release.get('enabled') is True and release.get('root_qualification_authorized') is True
            and release.get('source_review_approved') is True, 'Qualification is inactive/unreviewed')
    require(release['source_manifest_sha256'] == sha(HERE / 'SOURCE_MANIFEST.json')
            and release['policies'] == list(POLICIES) and release['seed'] == 6101,
            'One native engineering pass on the frozen successor')
    require(socket.gethostname() == 'peptide' and Path.cwd() == Path(release['runtime_repository'])
            and str(Path(sys.executable).absolute()) == release['runtime_python']
            and os.environ.get('CUDA_VISIBLE_DEVICES') == release['physical_gpu_uuid'],
            'Declared normal77 runtime and owned visible GPU')
    train_path, valid_path = bound(release['train']), bound(release['valid'])
    native = bound(release['polynormer']); output = phase_path(release['output'])
    require(not output.exists(), 'Fresh engineering output; no retry')
    require(output.parent.is_dir() and not output.is_relative_to(HERE), 'Fresh separate engineering scope')
    output.mkdir(exist_ok=False); os.umask(0o077)
    began = time.monotonic(); rows = []; session = None
    counts = dict(Session_factory_attempts=0, Session_factory_completions=0, TRAIN_update_attempts=0,
        TRAIN_update_completions=0, Adam_step_attempts=0, Adam_step_completions=0)
    try:
        import torch
        torch.cuda.set_device(0)
        total = torch.cuda.get_device_properties(0).total_memory
        cap = release['owned_GPU_allocator_cap_bytes']
        require(cap == 34359738368 and cap <= total, 'Original 32GiB owned allocator cap')
        torch.cuda.set_per_process_memory_fraction(cap / total, 0)
        pins, public, driver, constructor, replay, pool = dependencies()
        # Preserve the original evaluator-before-Session import opportunity.
        from ogb.graphproppred import Evaluator as GraphEvaluator
        from ogb.linkproppred import Evaluator as LinkEvaluator
        data = sys.modules['data_interface']
        train, valid, origin = data.load_train_valid('wikics', train_path, valid_path)
        for policy in POLICIES:
            for global_stage in (False, True):
                started = time.monotonic(); torch.cuda.reset_peak_memory_stats(0)
                counts['Session_factory_attempts'] += 1
                session = make_session(public, constructor, replay, pool, pins,
                    policy, release['seed'], 'cuda:0', native)
                counts['Session_factory_completions'] += 1
                session.model.set_global(global_stage)
                require(session.steps == 0 and not session.optimizers[0].state, 'Fresh state and Adam for each stage')
                require(session.independent_native_start['before_original_Adam'] is True
                    and session.independent_native_start['generated_rows'] == 56
                    and session.independent_native_start['default_CPU_CUDA_RNG_unchanged'] is True,
                    'Exact independently drawn rows with untouched default RNG')
                for member, stream in enumerate(session.streams):
                    with torch.random.fork_rng(devices=[0]):
                        torch.manual_seed(release['seed'] + 1009 * member + 300001)
                        torch.cuda.manual_seed(release['seed'] + 1009 * member + 300001)
                        require(torch.equal(stream['cpu'], torch.get_rng_state())
                            and torch.equal(stream['cuda'], torch.cuda.get_rng_state(0)), 'Original member dropout stream bytes')
                batches = data.train_batches('wikics', train, 1, release['seed'], session.device)
                batch, labels = next(batches)
                try:
                    next(batches)
                except StopIteration:
                    pass
                else:
                    raise ValueError('Exactly one complete580-label native TRAIN batch')
                optimizer = session.optimizers[0]; original_step = optimizer.step
                observation = {}; actual_calls = [0]
                before_versions = tuple(p._version for _, p in session.relation_partition['all'])

                def observed_step(*positional, **keyword):
                    require(not observation and tuple(p._version for _, p in session.relation_partition['all'])
                            == before_versions, 'One Adam after all old-state VJPs')
                    require(session.execution_totals['member_reverse_collections'] == 16,
                            'Two actual disjoint collections for every member/view')
                    group_rows = []
                    for group, loss in zip(session.relation_partition['groups'], session.relation_partition['selectors']):
                        active = [(name, p) for name, p in group if p.grad is not None]
                        require(active and all(torch.isfinite(p.grad).all() for _, p in active),
                                'Actual finite recipient/complement gradients')
                        group_rows.append(dict(loss=loss, group_parameter_names=[name for name, _ in group],
                            actual_nonNone_gradient_names=[name for name, _ in active],
                            squared_gradient_norm=sum(float(p.grad.detach().double().square().sum()) for _, p in active)))
                    inactive = [name for name, p in session.relation_partition['all'] if p.grad is None]
                    if not global_stage:
                        global_names = [name for name, _ in session.relation_partition['all']
                                        if name.startswith('models.0.body.global_attn.')
                                        or name.startswith('models.0.body.pred_global.')]
                        require(set(global_names) <= set(inactive), 'Inactive global/QK gradients must remain None')
                    else:
                        qk = [name + '.' + leaf for name in permission_metadata(session)['tied_QK_map_names']
                              for leaf in ('r', 's')]
                        require(all(dict(session.relation_partition['all'])[name].grad is not None for name in qk),
                                'Native global tied-QK factor pullback must execute')
                    observation.update(groups=group_rows, inactive_gradient_names=inactive,
                        versions_unchanged_before_Adam=True, independent_gradient_equivalence_claimed=False)
                    actual_calls[0] += 1
                    counts['Adam_step_attempts'] += 1
                    value = original_step(*positional, **keyword)
                    counts['Adam_step_completions'] += 1
                    return value

                optimizer.step = observed_step
                counts['TRAIN_update_attempts'] += 1
                session.train_step(batch, labels)
                counts['TRAIN_update_completions'] += 1
                optimizer.step = original_step
                expected = dict(shadow_member_forwards=8, replay_member_forwards=8,
                    output_cotangent_collections=2, member_reverse_collections=16,
                    optimizer_bank_updates=1, exact_member_RNG_endpoint_checks=1)
                require(session.steps == 1 and actual_calls == [1] and session.execution_totals == expected,
                        'One actual complete update and unchanged view/RNG/native transition')
                peak = torch.cuda.max_memory_reserved(0)
                require(peak <= cap, 'Native path exceeds the original owned cap')
                rows.append(dict(policy=policy, global_stage=global_stage, fresh_seed=6101,
                    complete=True, fullgraph_nodes=11701, TRAIN_labels=580,
                    gradients=observation, work=dict(session.execution_totals),
                    peak_CUDA_reserved_bytes=peak, elapsed_seconds=time.monotonic() - started,
                    parameter_partition=permission_metadata(session), initializer=session.independent_native_start,
                    VALID_metrics_read=False))
                (output / 'PROGRESS.json').write_text(json.dumps(dict(rows=rows, quality_scores_read=False), indent=2) + '\n')
                del batch, labels, session
                session = None
        result = dict(complete=True, source_manifest_sha256=release['source_manifest_sha256'],
            policies=list(POLICIES), stages=['native_local', 'native_global'], source_static_only=False,
            rows=rows, actual_counts=counts, real_complete_TRAIN_updates=4, native_models_constructed=4,
            fabricated_graph_or_numerical_fixture_used=False,
            VALID_scores_read=False, TEST_access=False, scientific_training_fits=0,
            new_method_utility_or_gradient_equivalence_claimed=False, inclusive_seconds=time.monotonic() - began,
            physical_gpu_uuid=release['physical_gpu_uuid'], data_origin=origin)
        (output / 'QUALIFIED.json').write_text(json.dumps(result, indent=2) + '\n')
    except BaseException as error:
        (output / 'FAILURE.json').write_text(json.dumps(dict(complete=False, error_type=type(error).__name__,
            error=str(error), completed_engineering_check_rows=len(rows), actual_counts=counts, rows=rows,
            current_actual_work=dict(session.execution_totals) if session is not None else None,
            seconds=time.monotonic() - began, automatic_retry=False, scientific_fits=0), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
