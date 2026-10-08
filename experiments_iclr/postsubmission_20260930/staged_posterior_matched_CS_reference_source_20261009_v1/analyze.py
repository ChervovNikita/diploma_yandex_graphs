"""Fixed secondary C&S serving adaptation; complete staged family first."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
P = HERE.parent


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def bound(row):
    path = (P / row['path']).resolve(strict=True)
    require(path.is_relative_to(P) and sha(path) == row['sha256']
            and path.stat().st_size == row['bytes'], 'Changed exact input binding')
    return path


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run(authorized=False):
    require(authorized is True, 'Disabled later scientific execution')
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    protocol = read(HERE / 'PROTOCOL.json')
    for row in read(HERE / 'MANIFEST.json')['files']:
        f = HERE / row['path']
        require(f.stat().st_size == row['bytes'] and sha(f) == row['sha256'], 'Changed analysis source')
    for row in pins.values():
        bound(row)
    runtime = read(bound(pins['runtime']))
    require(P == Path(runtime['phase']) and socket.gethostname() == runtime['hostname']
            and Path(sys.executable).resolve() == Path(runtime['python']).resolve(), 'Pinned singleton runtime')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
            text=True, timeout=5).splitlines() == [runtime['GPU_uuid']], 'Authorized singleton GPU route')
    stage = module(bound(pins['stage_program']), '_bounded_CS_staged_source')
    family = P / protocol['stage_family']
    records = stage.verify_closed_family(family)
    require(len(records) == 12 and read(family / 'FAMILY_CLOSURE.json')['complete'] is True
            and read(family / 'PARENT_TERMINAL.json')['family_complete'] is True,
            'Complete staged family and scientific owner terminal before payload loading')
    old = read(bound(pins['CS_receipt']))
    require(old['complete'] is True and len(old['records']) == 72
            and all(r['status'] == 'complete' for r in old['records'])
            and old['selection']['configuration_id'] == protocol['configuration_id'], 'Complete old72 common recipe')
    native_rows = {r['seed']: r for r in read(bound(pins['native_metadata']))['native_states']}
    native = {r['seed']: r for r in old['native_records']}
    require(set(native) == set(native_rows) == set(protocol['seeds'])
            and all(native[s]['native_state_sha256'] == native_rows[s]['sha256'] for s in native),
            'Same authentic own-selected native states')
    output = P / protocol['output_relative']
    require(not output.exists(), 'Fresh output only; retain failures, no retries')
    output.mkdir()
    started, cpu = time.monotonic(), time.process_time()
    rows, error = [], None
    work = dict(native_forwards=0, training_updates=0, recipe_searches=0,
                propagation_calls_attempted=0, propagation_calls_completed=0, propagation_passes=0)
    def stopped(number, frame):
        raise TimeoutError('Fixed180second C&S adaptation budget or signal ' + str(number))
    handlers = {n: signal.signal(n, stopped) for n in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM)}
    signal.setitimer(signal.ITIMER_REAL, protocol['budget_seconds'])
    try:
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '1'
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(str(torch.__version__) == runtime['torch'] and np.__version__ == runtime['numpy'], 'Qualified providers')
        roles = read(bound(pins['stage_program']).parent / 'INPUT_FILES.json')
        with np.load(bound(roles['train']), allow_pickle=False) as f:
            edges, anchors, labels = [torch.from_numpy(f[k].copy()) for k in ('edge_index', 'ids', 'y')]
        with np.load(bound(roles['valid']), allow_pickle=False) as f:
            ids, truth = [torch.from_numpy(f[k].copy()) for k in ('ids', 'y')]
        require(edges.shape == (2, 442907) and anchors.shape == labels.shape == (580,)
                and ids.shape == truth.shape == (5274,) and not bool(torch.isin(ids, anchors).any()), 'Whole original roles')
        self_mask = edges[0] == edges[1]
        require(int(self_mask.sum()) == 11701
                and torch.equal(edges[0, self_mask].sort().values, torch.arange(11701)), 'Original prepared self records')
        edges = edges[:, ~self_mask]
        helper = module(bound(pins['CS_helper']), '_bounded_CS_original_helper')
        def normalized(scores):
            require(bool(torch.isfinite(scores).all()) and bool((scores >= 0).all()), 'Nonnegative finite score map')
            value = scores.to(torch.float64)
            total = value.sum(-1, keepdim=True)
            zero = total[:, 0] == 0
            probability = torch.empty_like(value)
            probability[~zero] = value[~zero] / total[~zero]
            probability[zero] = 0.1
            return probability
        def readout(probability):
            true = probability.gather(1, truth[:, None])[:, 0]
            zeros = int((true == 0).sum())
            return dict(correctcount=int((probability.argmax(-1) == truth).sum()),
                        accuracy=float((probability.argmax(-1) == truth).double().mean()),
                        NLL=None if zeros else float(-true.log().mean()), NLL_infinite=bool(zeros),
                        zero_target_count=zeros,
                        Brier=float(((probability-torch.nn.functional.one_hot(truth, 10))**2).sum(-1).mean()))
        for seed in protocol['seeds']:
            prior = native[seed]
            record = next(r for r in old['records'] if r['seed'] == seed
                          and r['configuration_id'] == protocol['configuration_id'])
            require(record['configuration'] == protocol['configuration'], 'Exactly selected existing recipe')
            path = bound(dict(path=str((bound(pins['CS_receipt']).parent / prior['probability_file']).relative_to(P)),
                              bytes=prior['probability_file_bytes'], sha256=prior['probability_file_sha256']))
            p0 = torch.load(path, map_location='cpu', weights_only=False)
            block = family / ('seed' + str(seed))
            closure = read(block / 'COMPLETE.json')
            capture_path = block / 'FIXED_CAPTURE.pt'
            require(sha(capture_path) == closure['fixed_capture_sha256'], 'Authenticated staged native cache')
            capture = torch.load(capture_path, map_location='cpu', weights_only=False)
            cached = torch.softmax(capture['native_logits'], -1)
            difference = float((cached-p0).abs().max())
            require(capture['native_binding'] == native_rows[seed] and difference <= 2e-5
                    and torch.equal(cached.argmax(-1), p0.argmax(-1)), 'Native identity and practical probability join')
            work['propagation_calls_attempted'] += 1
            result = helper.correct_and_smooth_train_only(probabilities=p0, edge_index=edges,
                train_ids=anchors, train_labels=labels, classes=10, configuration=protocol['configuration'],
                device='cpu', later_execution_authorized=True)
            work['propagation_calls_completed'] += 1
            work['propagation_passes'] += result['propagation_passes']
            cs = normalized(result['smoothed_scores'])[ids]
            base = normalized(p0)[ids]
            mixed = 0.2 * base + 0.8 * cs
            raw_stats, base_stats, mix_stats = readout(cs), readout(base), readout(mixed)
            require(raw_stats['correctcount'] == record['readout']['correctcount']
                    and raw_stats['zero_target_count'] == record['readout']['NLL']['zero_target_count']
                    and base_stats['correctcount'] == prior['readout']['correctcount'], 'Existing counts preserved, no replacements')
            before, after = base.argmax(-1) == truth, mixed.argmax(-1) == truth
            repairs, harms = int((~before & after).sum()), int((before & ~after).sum())
            require(repairs-harms == mix_stats['correctcount']-base_stats['correctcount'], 'Exact repair-harm accounting')
            file = output / ('seed' + str(seed) + '.pt')
            torch.save(dict(seed=seed,ids=ids,truth=truth,native=base,normalized_CS=cs,bounded_CS=mixed), file)
            rows.append(dict(seed=seed,native=base_stats,raw_CS=raw_stats,bounded_CS=mix_stats,
                             repairs=repairs,harms=harms,native_cached_max_abs_difference=difference,
                             raw_file=dict(path=file.name,bytes=file.stat().st_size,sha256=sha(file))))
        write(output / 'RESULT.json', dict(complete=True, rows=rows, work=work,
              source_manifest_sha256=sha(HERE/'MANIFEST.json'), protocol_sha256=sha(HERE/'PROTOCOL.json'),
              primary_gate_changed=False, original_scores_replaced=False, secondary_consumed_development_only=True,
              new_method_or_published_CS_result=False, raw_arrays_server_only=True))
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        write(output / 'FAILURE.json', dict(error=error, completed_rows=rows, work=work, automatic_retry=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        write(output / 'COST_TERMINAL.json', dict(complete=error is None and len(rows)==3,error=error,work=work,
              wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
              max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              unobserved_partial_propagation_cost=work['propagation_calls_attempted']>work['propagation_calls_completed'],
              automatic_retry=False,CPU_threads=1))
        for number, handler in handlers.items():
            signal.signal(number, handler)
    return dict(complete=True,output=str(output))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorized',action='store_true')
    print(json.dumps(run(parser.parse_args().authorized)))
