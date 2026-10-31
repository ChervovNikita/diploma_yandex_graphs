"""Repo-cwd bridge with explicit CUDA initialization before peak reset.

Preserves the sealed fixture and failed v1 invocation. Only runtime startup
changes: the selected CUDA context exists before the fixture resets its peak.
"""

import argparse
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE))


def main():
    if any(name == "propagation_cost_impl_v1" or name.startswith("propagation_cost_impl_v1.")
           for name in sys.modules):
        raise RuntimeError("Propagation package already loaded before bridge")
    if not (PHASE / "propagation_cost_impl_v1/SOURCE_MANIFEST_local_run01.json").is_file():
        raise RuntimeError("Sealed propagation implementation is missing")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), default="cpu")
    args, _ = parser.parse_known_args()
    if args.device.startswith("cuda"):
        import torch
        torch.cuda.set_device(0)
        torch.cuda.init()
    runpy.run_module("propagation_cost_impl_v1.runtime_fixture", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
