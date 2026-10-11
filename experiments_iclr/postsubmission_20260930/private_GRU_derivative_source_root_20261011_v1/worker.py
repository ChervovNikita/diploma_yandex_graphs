"""Execute exactly the sealed nonzero-readout fixture, with no dataset inputs."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import resource
import time
import traceback
from runtime import admit, route, write


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    route()
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    admit(config)
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    value = {'UTC': datetime.now(timezone.utc).isoformat(), 'stage': 'derivative',
             'fixtures': 1, 'scientific_fits': 0, 'model_optimizer_updates': 0,
             'core_manifest_sha256': config['core_manifest_sha256']}
    try:
        import torch
        assert torch.cuda.device_count() == 1
        torch.cuda.reset_peak_memory_stats()
        core = Path(config['core_root'])
        backend = load('admitted_private_GRU_backend', core / 'gru_backend.py')
        check = load('admitted_private_GRU_gradient', core / 'gradient_check.py')
        result = check.run(torch, backend)
        torch.cuda.synchronize()
        value.update(result)
        value.update(torch_version=torch.__version__,
                     peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved())
        write(output / 'DERIVATIVE.json', value)
    except Exception as error:
        value.update(complete=False, failure_type=type(error).__name__,
                     failure=str(error), traceback=traceback.format_exc())
        write(output / 'FAILURE.json', value)
        raise
    finally:
        write(output / 'COSTS.json', dict(
            seconds=time.monotonic() - started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            scientific_fits=0, fixtures=1))
    print(json.dumps(dict(complete=True, stage='derivative', fixtures=1)), flush=True)


if __name__ == '__main__':
    main()
