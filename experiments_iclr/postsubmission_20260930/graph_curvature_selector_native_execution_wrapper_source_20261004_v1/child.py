"""Owned graph child: deterministic CUDA policy then exact fresh qualifier."""
import argparse
import importlib.util
from common import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('graph',choices=['Squirrel','Photo'])
    args = parser.parse_args()
    gpu = gpu_snapshot()
    require(gpu['free_MiB'] >= MINIMUM_FREE_MIB, '32GiB current GPU headroom floor not met')
    source_check = verify_sources()
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8' and torch.cuda.is_available()
        and torch.cuda.device_count() == 1, 'Exact CUDA runtime/device policy required')
    write_new(EXECUTION/(args.graph+'_CUDA_POLICY.json'),dict(gpu=gpu,source_check=source_check,
        cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],deterministic_algorithms=True,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),python=sys.executable,torch=torch.__version__))
    path = PHASE/RUNNER
    spec = importlib.util.spec_from_file_location('exact_native_qualification',path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    result = module.run_qualification(args.graph,str(EXECUTION/args.graph),device='cuda:0')
    require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        and torch.are_deterministic_algorithms_enabled(), 'CUDA policy changed during qualification')
    print(json.dumps(dict(graph=args.graph,status=result['status'],predictive_continuation=False)))
    return 0 if result['status'] == 'PASSED_ACTUAL_NATIVE_WARM_ENGINEERING_ONLY' else 1


if __name__ == '__main__':
    raise SystemExit(main())
