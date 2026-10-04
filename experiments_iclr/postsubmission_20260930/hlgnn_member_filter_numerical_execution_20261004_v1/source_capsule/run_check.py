"""One bounded CPU-only invocation of the unchanged planned equivalence checks."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    assert socket.gethostname() == 'peptide' and Path.cwd() == REPO
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and os.environ['PYTHONDONTWRITEBYTECODE'] == '1'
    assert os.environ['OMP_NUM_THREADS'] == os.environ['MKL_NUM_THREADS'] == os.environ['OPENBLAS_NUM_THREADS'] == '1'
    assert 'torch' not in sys.modules and sha(HERE/'MANIFEST.json') == args.manifest_sha256
    source = json.loads((HERE/'MANIFEST.json').read_text())
    for row in source['files']:
        path = HERE/row['path']
        assert path.resolve().is_relative_to(HERE) and not path.is_symlink()
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    authority = json.loads((HERE/'metadata/RUNTIME_AUTHORITY.json').read_text())
    assert Path(sys.executable).resolve() == Path(authority['interpreter_path']).resolve()
    assert sha(authority['interpreter_path']) == authority['interpreter_sha256']
    runtime_rows = authority['runtime_source_pins'] + authority['runtime_binary_files'] + [authority['negative_sampler']]
    for row in runtime_rows:
        path = Path(row['path'])
        assert path.is_file() and not path.is_symlink() and sha(path) == row['sha256']
        assert 'bytes' not in row or path.stat().st_size == row['bytes']
    for name, version in authority['distribution_versions'].items():
        assert importlib.metadata.version(name) == version
    resource.setrlimit(resource.RLIMIT_CPU, (45, 45))
    import torch
    import numpy
    import torch_geometric
    import torch_sparse
    import torch_scatter
    from torch_sparse import matmul
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert str(torch.get_default_device()) == 'cpu'
    assert torch.__version__ == authority['torch_version']
    assert Path(torch_sparse.__file__).resolve() == Path(authority['runtime_source_pins'][0]['path']).resolve()
    assert Path(torch_scatter.__file__).resolve() == Path(authority['runtime_source_pins'][1]['path']).resolve()
    known_binary_paths = {Path(row['path']).resolve() for row in authority['runtime_binary_files']}
    extension_dirs = {Path(torch_sparse.__file__).resolve().parent, Path(torch_scatter.__file__).resolve().parent}
    loaded = {Path(path).resolve() for path in torch.ops.loaded_libraries if Path(path).resolve().parent in extension_dirs}
    assert loaded and loaded <= known_binary_paths
    sys.path.insert(0, str(HERE/'native'))
    sys.path.insert(0, str(HERE))
    import member_filter
    import planned_equivalence
    assert Path(inspect.getfile(member_filter.gcn_norm)).resolve() == Path(authority['runtime_source_pins'][2]['path']).resolve()
    spec = importlib.util.spec_from_file_location('hlgnn_native_cpu_reference', HERE/'native/layer.py')
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    assert Path(sys.modules['utils'].__file__).resolve() == HERE/'native/utils.py'
    assert Path(sys.modules['negative_sample'].__file__).resolve() == HERE/'native/negative_sample.py'
    graph = planned_equivalence.irregular_graph()
    sample = member_filter.SharedPowerHLGNN(5,4).double()
    _, P = sample.context(torch.zeros(6,5,dtype=torch.float64), graph)
    propagated_one = matmul(P, torch.ones(6,1,dtype=torch.float64)).reshape(-1)
    assert not torch.allclose(propagated_one, torch.ones_like(propagated_one))
    assertions = []
    original_assert_close = torch.testing.assert_close

    def record_assert_close(actual, expected, *positional, **kwargs):
        frame = inspect.currentframe().f_back
        local = frame.f_locals
        name = 'output'
        if actual is not local.get('factored') and actual is not local.get('output_adapter'):
            name = local.get('name', 'gradient')
            if frame.f_code.co_name == 'check_native_reduction':
                index = next(i for i, value in enumerate(local['gradients_adapter']) if value is actual)
                name = 'input' if index == 0 else local['names'][index-1]
        row = dict(function=frame.f_code.co_name,kind=name,shape=list(actual.shape),dtype=str(actual.dtype),device=str(actual.device),
            rtol=kwargs.get('rtol'),atol=kwargs.get('atol'))
        for key in ('K','members','private_alpha','storage'):
            if key in local:row[key] = local[key]
        assert actual.device.type == expected.device.type == 'cpu'
        difference = (actual.detach()-expected.detach()).abs()
        row['max_absolute_error'] = difference.max().item() if difference.numel() else 0.0
        scale = kwargs['atol'] + kwargs['rtol']*expected.detach().abs()
        row['max_error_over_tolerance'] = (difference/scale).max().item() if difference.numel() else 0.0
        try:
            original_assert_close(actual, expected, *positional, **kwargs)
        except BaseException:
            row['status'] = 'FAIL';assertions.append(row);raise
        row['status'] = 'PASS';assertions.append(row)

    torch.testing.assert_close = record_assert_close
    result = dict(status='RUNNING',assertions=assertions,CPU_only=True,training=False,heldout_reads=False,GPU_computation=False)
    try:
        cases = planned_equivalence.check_factorization()
        native_result = planned_equivalence.check_native_reduction(native.HLGNN)
        result.update(status='PASS_OUTPUT_AND_FULL_GRADIENT_EQUIVALENCE',factorization_cases=cases,
            factorization_comparisons=len(cases),native_reduction=native_result,
            irregular_weighted_graph=dict(nodes=6,undirected_records=8,normalized_P_times_one=propagated_one.tolist(),
                max_deviation_from_one=(propagated_one-1).abs().max().item(),P1_nonconstant=True),
            comparison_summary=dict(assertions=len(assertions),outputs=sum(r['kind']=='output' for r in assertions),
                full_gradients=sum(r['kind']!='output' for r in assertions),
                max_absolute_error=max(r['max_absolute_error'] for r in assertions),
                max_error_over_tolerance=max(r['max_error_over_tolerance'] for r in assertions)))
    except BaseException as error:
        result.update(status='FAIL_OUTPUT_OR_GRADIENT_EQUIVALENCE',failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    finally:
        torch.testing.assert_close = original_assert_close
    assert not torch.cuda.is_initialized()
    for row in source['files']:
        assert sha(HERE/row['path']) == row['sha256']
    for row in runtime_rows:
        assert sha(row['path']) == row['sha256']
    modules = (torch,numpy,torch_geometric,torch_sparse,torch_scatter,torch._C)
    paths = {str(Path(module.__file__).resolve()) for module in modules}
    paths.update((inspect.getfile(torch_geometric.nn.MessagePassing),inspect.getfile(member_filter.gcn_norm)))
    paths.update(str(path) for path in loaded)
    observed_runtime = [dict(path=path,bytes=Path(path).stat().st_size,sha256=sha(path)) for path in sorted(paths)]
    result.update(source_manifest_sha256=args.manifest_sha256,interpreter_path=sys.executable,interpreter_sha256=sha(sys.executable),
        runtime_authority_sha256=sha(HERE/'metadata/RUNTIME_AUTHORITY.json'),runtime_authority_pins_match_before_after=True,
        observed_runtime_files=observed_runtime,versions={m.__name__:m.__version__ for m in modules if hasattr(m,'__version__')},
        CUDA_initialized=False,source_custody_match_after=True,threads=1,interop_threads=1,default_device='cpu',
        source_repair=False,complete_native_layer_module_imported=True,source_mutations=False,installs=False,
        elapsed_seconds=time.monotonic()-started,process_max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        numerical_dtype='float64',graph_operator_gradient_checked=False,
        limits='Synthetic tied-parameter/context equivalence on a fixed P; not speed, memory scaling, novelty, prediction quality or a predictive-family release.')
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0 if result['status']=='PASS_OUTPUT_AND_FULL_GRADIENT_EQUIVALENCE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
