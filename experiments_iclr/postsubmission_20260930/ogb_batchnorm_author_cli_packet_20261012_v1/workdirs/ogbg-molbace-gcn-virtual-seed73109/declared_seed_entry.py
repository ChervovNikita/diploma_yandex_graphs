"""External declared replication seeds; native OGB CLI bytes remain unchanged."""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

SEEDS = (73101, 73103, 73107, 73109, 73121, 73127, 73133, 73141, 73147, 73151)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declared-seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("native_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    native = args.native_argv
    if not native or native[0] != "--" or len(native) < 2 or native[1] != "main_pyg.py":
        parser.error("Use the bound unchanged native main_pyg.py invocation")
    path = Path("main_pyg.py").resolve()
    if path.parent != Path.cwd().resolve() or path.is_symlink():
        parser.error("The native CLI must be the bound private cwd source")
    # This extension is explicit. No deterministic algorithm mode is forced;
    # DataLoader/native evaluation keep their advancing process RNG behavior.
    import random
    import numpy as np
    import torch
    random.seed(args.declared_seed)
    np.random.seed(args.declared_seed)
    torch.manual_seed(args.declared_seed)
    torch.cuda.manual_seed_all(args.declared_seed)
    print("DECLARED_SEED_PROCESS_START", args.declared_seed, flush=True)
    sys.argv = native[1:]
    runpy.run_path(str(path), run_name="__main__")
    print("DECLARED_SEED_PROCESS_RETURN 0", flush=True)


if __name__ == "__main__":
    main()
