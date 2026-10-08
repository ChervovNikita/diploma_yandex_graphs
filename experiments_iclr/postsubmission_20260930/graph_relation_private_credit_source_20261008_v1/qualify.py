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
from permissions import POLICIES, metadata as permission_metadata, require
from train import HERE, PHASE, bound, dependencies, make_session, phase_path, read, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    require(sha(args.release) == args.release_sha256, 'Exact separate qualification release')
    release = read(args.release)
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
    output.mkdir(parents=True, exist_ok=False); os.umask(0o077)
    began = time.monotonic(); rows = []; session = None
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
                session = make_session(public, constructor, replay, pool, pins,
                    policy, release['seed'], 'cuda:0', native)
                session.model.set_global(global_stage)
                require(session.steps == 0 and not session.optimizers[0].state, 'Fresh state and Adam for each stage')
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
                    return original_step(*positional, **keyword)

                optimizer.step = observed_step
                session.train_step(batch, labels)
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
                    parameter_partition=permission_metadata(session), VALID_metrics_read=False))
                (output / 'PROGRESS.json').write_text(json.dumps(dict(rows=rows, quality_scores_read=False), indent=2) + '\n')
                del batch, labels, session
                session = None
        result = dict(complete=True, source_manifest_sha256=release['source_manifest_sha256'],
            policies=list(POLICIES), stages=['native_local', 'native_global'], source_static_only=False,
            rows=rows, real_complete_TRAIN_updates=8, native_models_constructed=8,
            fabricated_graph_or_numerical_fixture_used=False,
            VALID_scores_read=False, TEST_access=False, scientific_training_fits=0,
            new_method_utility_or_gradient_equivalence_claimed=False, inclusive_seconds=time.monotonic() - began,
            physical_gpu_uuid=release['physical_gpu_uuid'], data_origin=origin)
        (output / 'QUALIFIED.json').write_text(json.dumps(result, indent=2) + '\n')
    except BaseException as error:
        (output / 'FAILURE.json').write_text(json.dumps(dict(complete=False, error_type=type(error).__name__,
            error=str(error), completed_engineering_updates=len(rows), rows=rows,
            seconds=time.monotonic() - began, automatic_retry=False, scientific_fits=0), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
