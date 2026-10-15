"""Own versus equal-route mixture moments at existing selected weights.

Derived from bn_bank_assignment_v1/diagnose.py; keep its authentic frozen
adapter construction and adapter.validate collector. No training or TEST.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback


def main():
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    import selected_molecular as previous
    previous.require_allocation()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or len(sys.argv) != 4:
        raise ValueError('CPU-only: diagnose.py EXACT_ROOT SEED_INDEX FRESH_OUTPUT')
    root, index, output = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3]).resolve()
    assert here.is_relative_to(root) and output.is_relative_to(root) and not output.exists()
    inputs = json.loads((here / 'INPUTS.json').read_text())
    for pin in inputs['sources']:
        previous.verify_pin(previous.safe_path(pin['path'], root), pin['sha256'])
    entry = inputs['fit_results'][index]
    result_path = previous.safe_path(entry['path'], root)
    previous.verify_pin(result_path, entry['sha256'])
    fit = result_path.parent
    result, job = json.loads(result_path.read_text()), json.loads((fit / 'JOB.json').read_text())
    prior_pin = inputs['prior_diagnostics'][index]
    prior_path = previous.safe_path(prior_pin['path'], root)
    previous.verify_pin(prior_path, prior_pin['sha256'])
    prior = json.loads(prior_path.read_text())
    assert prior['status'] == 'complete' and prior['native_forwards'] == 100
    assert prior['seed'] == job['seed']
    assert prior['checkpoint']['sha256'] == result['selected_checkpoint']['sha256']
    assert [c['condition'] for c in prior['conditions']] == ['own','tied','shift1','shift2','shift3']
    assert result['status'] == 'complete' and job['family'] == 'ogb' and job['TEST_access'] is False
    assert job['seed'] == (101, 202, 303)[index] and job['recipe']['dataset'] == 'ogbg-molbace'
    runtime = Path(result['runtime_sources']['run_fit.py']['path']).parent
    for name, pin in result['runtime_sources'].items():
        previous.verify_pin(previous.safe_path(pin['path'], root), pin['sha256'])
        assert Path(pin['path']).parent == runtime
    sys.path.insert(0, str(runtime))
    from adapters import create_adapter
    from bookkeeping import Policy
    from common import digest, seed_all
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    previous.verify_pin(fit / 'selected.pt', result['selected_checkpoint']['sha256'])
    previous.verify_pin(fit / 'selected_VALID.pt', result['selected_VALID_export']['sha256'])
    selected = torch.load(fit / 'selected.pt', map_location='cpu', weights_only=False)
    saved = torch.load(fit / 'selected_VALID.pt', map_location='cpu', weights_only=False)
    assert result['job_hash'] == selected['job_hash'] == saved['job_hash'] == digest(job)
    assert saved['role'] == 'VALID' and saved['TEST_accessed'] is False
    binding = selected['runtime_binding']
    assert binding['runtime_sha256'] == {k:v['sha256'] for k,v in result['runtime_sources'].items()}
    for path, sha in binding['dependency_sha256'].items():
        previous.verify_pin(previous.safe_path(path, root), sha)
    previous.verify_pin(previous.safe_path(job['callback_module'], root), binding['callback_sha256'])
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {'schema':'fixed-selected-BN-mixture-moments-v1', 'status':'running', 'seed':job['seed'],
              'selected_epoch':selected['selector']['selected_epoch'], 'role':'VALID',
              'fit_result':previous.pin(result_path), 'checkpoint':previous.pin(fit/'selected.pt'),
              'inputs':previous.pin(here/'INPUTS.json'), 'source':previous.pin(Path(__file__)),
              'conditions':[], 'bank_moment_summaries':[],
              'prior_assignment':previous.pin(prior_path), 'prior_conditions':prior['conditions'],
              'readout':'original fixed mean native raw logits via adapter.validate',
              'native_forwards':0, 'optimizer_updates':0, 'TRAIN_calls':0,
              'TEST_accessed':False, 'selection_changed':False, 'scientific_fits':0,
              'device':'cpu', 'threads':2}
    adapter = hook = None

    def timeout(*_):
        raise TimeoutError('Frozen 300-second diagnostic limit')

    def forbid(*_, **__):
        raise RuntimeError('No training or optimizer calls in this diagnostic')

    def parameters(model):
        h=hashlib.sha256()
        for name, value in model.named_parameters():
            h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    try:
        seed_all(torch, job['seed'])
        construction = copy.deepcopy(job); construction['device'] = 'cpu'
        adapter = create_adapter(torch, construction, Policy.from_dict(job['policy'], 'ogb'), root)
        assert adapter.callback_receipt['sha256'] == binding['callback_sha256']
        assert adapter.extension_receipt['sha256'] == binding['extension_sha256']
        adapter.load_state(copy.deepcopy(selected['adapter']))
        for optimizer in adapter.optimizers: optimizer.step = forbid
        adapter.train_epoch = forbid
        adapter.session.train_update = forbid
        model = adapter.model
        manager = model.route_bn_state
        original = copy.deepcopy(manager.state_dict())
        parameter_sha = parameters(model)
        assert len(manager.banks) > 0 and all(len(banks)==4 for banks in manager.banks)
        for banks in manager.banks:
            assert all(torch.equal(banks[0].num_batches_tracked, b.num_batches_tracked) for b in banks)

        # Stable equal-route law of total variance, accumulated in float64.
        # These are plug-in moments of saved EMA banks, not recalibrated moments.
        targets = []
        for layer, banks in enumerate(manager.banks):
            stored_means = torch.stack([original[f'banks.{layer}.{m}.running_mean'] for m in range(4)])
            means = stored_means.double()
            variances = torch.stack([original[f'banks.{layer}.{m}.running_var'].double() for m in range(4)])
            assert torch.isfinite(means).all() and torch.isfinite(variances).all()
            assert (variances >= 0).all()
            mean = means.mean(0)
            within = variances.mean(0)
            between = (means - mean).square().mean(0)
            variance = within + between
            assert torch.isfinite(variance).all() and (variance >= 0).all()
            # Match the old arithmetic tied mean exactly; change only variance.
            targets.append((stored_means.mean(0), variance))
            report['bank_moment_summaries'].append({
                'layer':manager.names[layer], 'channels':mean.numel(),
                'within_variance_mean':float(within.mean()),
                'between_mean_variance_mean':float(between.mean()),
                'between_mean_variance_max':float(between.max()),
                'mixture_variance_mean':float(variance.mean()),
                'between_fraction_of_total_sum':float(between.sum()/variance.sum()) if variance.sum()>0 else None})

        def count(_module, _args, _output):
            report['native_forwards'] += 1
            if report['native_forwards'] > 40: raise RuntimeError('Frozen forward limit exceeded')

        hook = model.native.register_forward_hook(count)
        outputs = {}
        for name in ('own', 'mixture'):
            manager.load_state_dict(copy.deepcopy(original))
            if name == 'mixture':
                with torch.no_grad():
                    for banks, (mean, variance) in zip(manager.banks, targets):
                        for bank in banks:
                            bank.running_mean.copy_(mean)
                            bank.running_var.copy_(variance)
            # Counters, eps, momentum and affine parameters remain unchanged.
            model.eval()
            before = time.perf_counter()
            with torch.no_grad(): actual = adapter.validate()
            ids, labels, logits = (actual[k].detach().cpu() for k in ('ids','labels','outputs'))
            assert len(ids)==151 and torch.equal(ids,saved['ids'].cpu())
            assert torch.allclose(labels,saved['labels'].cpu(),atol=0,rtol=0,equal_nan=True)
            assert actual['role']=='VALID' and actual['TEST_accessed'] is False and actual['complete_source_VALID'] is True
            assert logits.shape == saved['raw_logits'].shape and torch.isfinite(logits).all()
            metrics = {k:float(v) for k,v in actual['metrics'].items()}
            y, z = labels.double().reshape(-1), logits.double().reshape(-1)
            row = {'condition':name, 'rocauc':metrics['rocauc'],
                   'BCE':float(torch.nn.functional.binary_cross_entropy_with_logits(z,y)),
                   'Brier':float((z.sigmoid()-y).square().mean()),
                   'VALID_graphs':len(ids), 'seconds':time.perf_counter()-before}
            if name=='own':
                delta=(logits-saved['raw_logits'].cpu()).abs()
                report['own_reconstruction']={'max_abs_logit_difference':float(delta.max()),
                    'AUC_abs_difference':abs(row['rocauc']-float(saved['validation']['metrics']['rocauc']))}
                torch.testing.assert_close(logits,saved['raw_logits'].cpu(),atol=1e-4,rtol=1e-4)
            report['conditions'].append(row)
            outputs[name]=logits
        manager.load_state_dict(original)
        assert parameters(model)==parameter_sha and report['native_forwards']==40
        assert not torch.cuda.is_initialized()
        for key,value in original.items(): assert torch.equal(manager.state_dict()[key],value)
        torch.save({'role':'VALID','TEST_accessed':False,'ids':ids,'labels':labels,
                    'condition_logits':outputs,'selected_checkpoint':report['checkpoint']}, output/'logits.pt')
        report.update(status='complete', parameters_unchanged=True, banks_restored=True,
                      artifact=previous.pin(output/'logits.pt'))
    except BaseException as error:
        report.update(status='failed',failure={'type':type(error).__name__,'message':str(error),
                      'traceback':traceback.format_exc(limit=10)})
    finally:
        signal.alarm(0)
        if hook is not None: hook.remove()
        if adapter is not None: adapter.close()
        report.update(wall_seconds=time.perf_counter()-started,
                      peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (output/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','seed','native_forwards','wall_seconds','failure')}))
    return 0 if report['status']=='complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
