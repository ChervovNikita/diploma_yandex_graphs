"""Create an explicit allocation dependency/RNG audit without models or data."""
import argparse
from pathlib import Path

from common import atomic_json, inside, require_allocation
from source_rng import make_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--redesign-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require_allocation()
    root = Path(args.project_root).resolve()
    output = inside(args.output, root)
    if output.exists():
        raise ValueError("Choose a fresh audit output; preserve previous version-bound audits")
    import torch
    audit = make_audit(torch, inside(args.redesign_root, root))
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, audit)
    print("Saved finite loader/source audit; zero model calls and zero TEST data access")


if __name__ == "__main__":
    main()
