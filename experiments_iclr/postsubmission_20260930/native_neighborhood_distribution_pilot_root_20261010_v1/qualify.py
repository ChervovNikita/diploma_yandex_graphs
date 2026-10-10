"""Five actual full-graph TRAIN cases; no VALID scoring or scientific family."""
from pathlib import Path
import hashlib
import importlib.util
import json
import socket
import subprocess
import sys
import time

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
H = Path(__file__).resolve().parent

def main():
    assert socket.gethostname() == 'anogena-2-0' and Path.cwd() == R
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    frozen = json.loads((H / 'QUALIFICATION_FREEZE.json').read_text())
    for row in frozen['repo_files']:
        assert hashlib.sha256((R / row['path']).read_bytes()).hexdigest() == row['sha256'], row['path']
    for row in frozen['bound_files']:
        assert hashlib.sha256((P / row['path']).read_bytes()).hexdigest() == row['sha256'], row['path']
    output = H / 'actual_qualification_v1'
    output.mkdir(exist_ok=False)
    source = P / 'native_neighborhood_quantile_native_source_20261010_v1/native_extension.py'
    spec = importlib.util.spec_from_file_location('actual_native_quartile_source', source)
    extension = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = extension
    spec.loader.exec_module(extension)
    config = H / 'CONFIG.json'
    family = extension.family_class()(json.loads(config.read_text()), output, hashlib.sha256(config.read_bytes()).hexdigest())
    torch = family.torch
    torch.set_num_threads(2)
    def forbidden(*args, **kwargs):
        raise RuntimeError('VALID scoring is forbidden during qualification')
    family.metrics = forbidden
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    cases = []
    for index, arm in enumerate(extension.ARMS):
        bank, members = index < 3, 4 if index < 3 else 1
        family.current.update(arm=arm, seed=7301, member=0)
        model, optimizer, streams = family.make(7301, 0, 'baseline', True, members)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        native, corrected = family.predictions(model, streams, bank)
        assert torch.equal(native, corrected), 'Zero residual must give exact native scores'
        labels = family.data.y[family.train_ids].repeat(members)
        native_loss = family.F.cross_entropy(native[:, family.train_ids].flatten(0, 1), labels)
        loss = .5 * native_loss + .5 * family.F.cross_entropy(corrected[:, family.train_ids].flatten(0, 1), labels)
        first = torch.autograd.grad(native_loss, native, retain_graph=True)[0]
        total = torch.autograd.grad(loss, native, retain_graph=True)[0]
        assert torch.allclose(first, total, atol=1e-6, rtol=1e-5)
        loss.backward()
        assert torch.isfinite(loss).item()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.dist_U.grad is not None and torch.count_nonzero(model.dist_U.grad).item() == 0
        added = {n for n, p in model.named_parameters() if n.startswith('dist_')}
        assert added and added.issubset(model.state_dict())
        output_parameter = model.dist_residual[2].weight if index >= 3 else model.dist_V
        assert output_parameter.grad is not None and output_parameter.grad.abs().sum().item() > 0
        output_gradient = float(output_parameter.grad.norm())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        n2, c2 = family.predictions(model, streams, bank)
        loss2 = .5 * family.F.cross_entropy(n2[:, family.train_ids].flatten(0, 1), labels) + .5 * family.F.cross_entropy(c2[:, family.train_ids].flatten(0, 1), labels)
        loss2.backward()
        assert torch.isfinite(n2).all() and torch.isfinite(c2).all()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        direction_gradient = float(model.dist_U.grad.abs().sum())
        assert direction_gradient > 0, 'Live projected-distribution gradients after output begins learning'
        independent_storage = None
        if index == 4:
            shadow, shadow_optimizer, shadow_streams = family.make(7301, 1, 'baseline', True, 1)
            independent_storage = not ({id(p) for p in model.parameters()} & {id(p) for p in shadow.parameters()})
            assert independent_storage and model._distribution_seeds != shadow._distribution_seeds
            del shadow, shadow_optimizer, shadow_streams
        saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            model.dist_U.add_(1)
        model.load_state_dict(saved, strict=True)
        assert all(torch.equal(model.state_dict()[k], v) for k, v in saved.items())
        cases.append(dict(arm=arm, native_routes=members, zero_residual_native_identity=True,
            initial_native_gradient_identity=True, initial_direction_gradient_zero=True,
            output_gradient_norm=output_gradient, direction_gradient_sum_after_update=direction_gradient,
            finite_parameter_gradients=True, all_added_parameters_registered_and_restored=True,
            independent_storage_checked=independent_storage, distribution_seeds=model._distribution_seeds,
            parameters=sum(p.numel() for p in model.parameters()), TRAIN_initial_loss=float(loss)))
        del model, optimizer, streams, native, corrected, labels, native_loss, loss, first, total, output_parameter, n2, c2, loss2, saved
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    value = dict(qualified=True, cases=cases, seconds=time.perf_counter()-start,
        full_native_forwards=10, native_member_trajectories=28, full_parameter_backwards=10,
        discarded_optimizer_updates=5, scientific_fits=0, independent_shadow_constructions=1,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        descriptor_support=family.support_record, VALID_labels_loaded_only_by_existing_role_loader=True,
        VALID_quality_scored=False, TEST_access=False, scientific_quality_evidence=False,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    assert value['cuda_peak_allocated_bytes'] < 32 * 1024**3
    (output / 'QUALIFICATION.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(dict(qualified=True, cases=len(cases), seconds=value['seconds'],
        peak_GiB=value['cuda_peak_allocated_bytes']/1024**3, scientific_quality_evidence=False)))

if __name__ == '__main__':
    main()
