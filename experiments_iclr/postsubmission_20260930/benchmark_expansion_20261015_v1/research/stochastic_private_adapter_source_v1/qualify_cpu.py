"""UNEXECUTED: root-admitted full-data, two-update CPU lifecycle for one arm.

Two constructions, two optimizer transitions total, three whole VALID calls.
No fit, selector, TEST indexing, extra model forward, or automatic retry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback


def guard():
    if socket.gethostname() != 'anogena-2-0' or subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines() != [
            'GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Dedicated allocation required before project metadata reads')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This bounded qualifier is CPU only')


def pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def same_noise(torch, a, b):
    return all(a[k] == b[k] for k in ('schema', 'seeds', 'counts')) and all(
        torch.equal(x, y) for x, y in zip(a['states'], b['states']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--arm', choices=('fixed', 'learned'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--admission', required=True)
    parser.add_argument('--admission-sha256', required=True)
    args = parser.parse_args()
    guard()
    import extension as ex
    root, here, output = Path(args.project_root).resolve(), Path(__file__).resolve().parent, Path(args.output).resolve()
    ex.require(here.is_relative_to(root) and output.is_relative_to(root) and not output.exists()
               and Path(sys.prefix).resolve() == (root / '.venv').resolve(), 'Original virtualenv and fresh contained output required')
    admission = Path(args.admission).resolve()
    ex.require(admission.is_relative_to(root) and pin(admission)['sha256'] == args.admission_sha256, 'Root admission differs')
    a, manifest = json.loads(admission.read_text()), pin(here / 'MANIFEST.json')
    ex.require(a.get('schema') == 'root-reviewed-stochastic-private-CPU-v1' and a.get('approved') is True
               and a.get('source_reviewed') is True and a.get('arm') == args.arm
               and a.get('manifest_sha256') == manifest['sha256'] and a.get('new_scientific_fits') == 0
               and a.get('TEST_access') is False and a.get('maximum_native_constructions') == 2
               and a.get('maximum_optimizer_transitions') == 2 and a.get('maximum_full_VALID_calls') == 3,
               'Exact bounded root CPU admission required')
    ex.verify_sources(root, data=True)
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = {'schema': 'stochastic-private-CPU-qualification-v1', 'status': 'running', 'arm': args.arm,
              'source': pin(__file__), 'manifest': manifest, 'admission': pin(admission), 'device': 'cpu', 'threads': 2,
              'scientific_fit': False, 'scientific_selection': False, 'TEST_accessed': False, 'automatic_retries': 0,
              'construction_attempts': 0, 'constructions_completed': 0, 'native_forward_attempts': 0,
              'native_forwards_completed': 0, 'VALID_attempts': 0, 'VALID_completed': 0,
              'optimizer_calls': {'attempts': [0], 'completed': [0]}, 'gradient_receipts': []}
    adapters, hooks, observer = [], [], None

    def write():
        if observer is not None:
            report['optimizer_calls'] = observer.state()
        report.update(wall_seconds=time.perf_counter() - started,
                      own_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        (output / 'RESULT.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Bounded CPU qualification interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        import torch
        torch.set_num_threads(2)
        sys.path.insert(0, str(ex.E / 'runtime'))
        from adapters import create_adapter
        from run_fit import validate_job, OptimizerObserver
        from common import seed_all, global_rng, restore_rng, cpu_clone
        planned = ex.descriptor(101, args.arm)
        policy, job = validate_job(planned), dict(planned, device='cpu')
        report['planned_job'], report['execution_override'] = planned, {'device': 'cpu'}

        def construct():
            report['construction_attempts'] += 1
            seed_all(torch, job['seed'])
            adapter = create_adapter(torch, job, policy, root)
            adapters.append(adapter)
            report['constructions_completed'] += 1
            ex.require(len(adapter.session.train_ids) == 1210 and len(adapter.session.valid_ids) == 151,
                       'Full original data roles required; no smoke subset')

            def before(module, arguments):
                report['native_forward_attempts'] += 1

            def after(module, arguments, result):
                report['native_forwards_completed'] += 1

            hooks.extend([adapter.model.native.register_forward_pre_hook(before), adapter.model.native.register_forward_hook(after)])
            streams = adapter.session.stochastic_noise
            begin = streams.begin

            def checked_begin():
                global_before = torch.get_rng_state().clone()
                train_before = adapter.train_generator.get_state().clone()
                valid_before = adapter.session.valid_generator.get_state().clone()
                begin()
                ex.require(torch.equal(global_before, torch.get_rng_state())
                           and torch.equal(train_before, adapter.train_generator.get_state())
                           and torch.equal(valid_before, adapter.session.valid_generator.get_state()),
                           'Noise draw altered native dropout/sampler RNG')

            streams.begin = checked_begin
            for name, parameter in adapter.model.input_residual.named_parameters():
                def grad_hook(gradient, label=name, session=adapter.session):
                    record = {'before_update': session.updates, 'parameter': label,
                              'finite': bool(torch.isfinite(gradient).all()),
                              'nonzero': int(torch.count_nonzero(gradient)), 'norm': float(gradient.norm())}
                    report['gradient_receipts'].append(record)
                    ex.require(record['finite'], 'All adapter gradients must be finite')
                hooks.append(parameter.register_hook(grad_hook))
            return adapter

        def validate(adapter):
            before = adapter.session.stochastic_noise.state()
            report['VALID_attempts'] += 1
            value = adapter.validate()
            ex.require(len(value['ids']) == 151 and torch.equal(value['ids'], adapter.session.valid_ids)
                       and value['complete_source_VALID'] is True and value['TEST_accessed'] is False,
                       'Whole original VALID151 required')
            ex.require(same_noise(torch, before, adapter.session.stochastic_noise.state()), 'Mean VALID consumed noise RNG')
            report['VALID_completed'] += 1
            return value

        source = construct()
        batch = next(iter(source.train_loader))
        positions = torch.tensor([source._position[int(i)] for i in batch.smoke_graph_id.reshape(-1)], dtype=torch.long)
        ex.require(batch.x.shape[0] != 1 and int(batch.batch[-1]) != 0, 'First native minibatch must be eligible; no resampling')
        source.session._next_train_batch = lambda: (batch, positions, 0)
        observer = OptimizerObserver(torch, source.optimizers)
        first = source.session.train_update()
        report['first_update'] = first
        ex.require(first['KL_penalty'] == 0 and first['noise_draw_counts'] == [1] * 4,
                   'At q=p, initial KL is exactly zero; one draw per route')
        valid = validate(source)
        initial_grads = {r['parameter']: r for r in report['gradient_receipts'] if r['before_update'] == 0}
        ex.require(initial_grads['down']['nonzero'] == 0 and initial_grads['up']['nonzero'] > 0
                   and (args.arm == 'fixed' or initial_grads['rho']['nonzero'] == 0),
                   'U0 permits U learning while mean/rho task gradients begin at zero')
        ex.require(source.model.input_residual.up.count_nonzero().item() > 0, 'First update must move U')
        next_digest = source.session.stochastic_noise.preview_digest()
        payload = {'adapter': cpu_clone(torch, source.state()), 'global_rng': cpu_clone(torch, global_rng(torch)),
                   'planned_job': planned, 'next_noise_sha256': next_digest, 'optimizer_calls': observer.state(),
                   'engineering_only': True}
        checkpoint = output / 'ENGINEERING_after_one.pt'
        torch.save(payload, checkpoint)
        report['checkpoint'] = pin(checkpoint)
        report['optimizer_calls'] = observer.state()
        ex.require(report['optimizer_calls'] == {'attempts': [1], 'completed': [1]}, 'One transition before save')
        observer.close()
        observer = None
        fresh = construct()
        fresh.session._next_train_batch = lambda: (batch, positions, 0)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        ex.require(saved['planned_job'] == planned, 'Own engineering checkpoint differs')
        fresh.load_state(saved['adapter'])
        ex.require(fresh.session.stochastic_noise.preview_digest() == saved['next_noise_sha256'], 'Next noise replay differs')
        replay = validate(fresh)
        ex.require(torch.equal(valid['outputs'], replay['outputs']) and torch.equal(valid['ids'], replay['ids']),
                   'Fresh-restored full VALID must be bit exact')
        fresh.load_state(saved['adapter'])
        restore_rng(torch, saved['global_rng'])
        observer = OptimizerObserver(torch, fresh.optimizers, previous=saved['optimizer_calls'])
        second = fresh.session.train_update()
        report['second_update_after_fresh_restore'] = second
        ex.require(second['noise_draw_sha256'] == saved['next_noise_sha256']
                   and second['noise_draw_counts'] == [2] * 4, 'Resumed update must use exact next independent draws')
        later_grads = {r['parameter']: r for r in report['gradient_receipts'] if r['before_update'] == 1}
        ex.require(later_grads['down']['nonzero'] > 0
                   and (args.arm == 'fixed' or later_grads['rho']['nonzero'] > 0),
                   'Down/rho gradients must activate after U moves')
        validate(fresh)
        ex.require(observer.state() == {'attempts': [2], 'completed': [2]} and fresh.session.updates == 2,
                   'Exactly two total optimizer transitions required')
        ex.require(report['native_forward_attempts'] == report['native_forwards_completed'] == 68,
                   'Two four-route updates + three complete five-batch VALID calls = 68 native forwards')
        report.update(status='passed', fresh_restore_VALID_bit_exact=True, noise_replay_exact=True,
                      candidate_metadata=fresh.metadata(), engineering_completed_epochs=0, no_selector_called=True,
                      limits='Same first genuine shuffled batch reused twice; full data binding, no full epoch/fit/CUDA/cost admission.')
    except BaseException as error:
        report.update(status='failed', failure={'type': type(error).__name__, 'message': str(error),
                                               'traceback': traceback.format_exc(limit=12)})
    finally:
        write()
        if observer is not None:
            observer.close()
        for hook in hooks:
            hook.remove()
        for adapter in adapters:
            try:
                adapter.close()
            except Exception as error:
                report.setdefault('close_errors', []).append(str(error))
                report['status'] = 'failed'
        write()
    print(json.dumps({k: report[k] for k in ('status', 'arm', 'constructions_completed', 'native_forwards_completed', 'VALID_completed', 'optimizer_calls')}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
