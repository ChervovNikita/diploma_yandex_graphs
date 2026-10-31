"""Repository-cwd bridge to the sealed synthetic propagation fixture.

The authorization supervisor requires a phase script path. This bridge only
supplies package discovery; the sealed fixture retains its size/output gates.
"""

from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parents[1]
PACKAGE = PHASE / "propagation_cost_impl_v1"
sys.path.insert(0, str(PHASE))


def main():
    # Avoid accidentally reusing a similarly named, previously loaded package.
    if any(name == "propagation_cost_impl_v1" or name.startswith("propagation_cost_impl_v1.")
           for name in sys.modules):
        raise RuntimeError("Propagation package already loaded before bridge")
    if not (PACKAGE / "SOURCE_MANIFEST_local_run01.json").is_file():
        raise RuntimeError("Sealed propagation implementation is missing")
    runpy.run_module("propagation_cost_impl_v1.runtime_fixture", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
