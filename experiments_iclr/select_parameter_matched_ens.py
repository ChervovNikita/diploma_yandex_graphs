"""Select a four-model SAGE ensemble width by parameter count alone.

The fixed candidates are declared here before any training result is read.
This script touches no validation or test scores. It counts the exact models
used by ``projector_controls.py`` on the Roman Empire architecture.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_iclr"))
from projector_controls import make_model  # noqa: E402

OUT = ROOT / "experiments_iclr" / "parameter_matched_ens_results"
WIDTHS = (192, 224, 256, 288, 320)
INPUT_DIM = 300
OUTPUT_DIM = 18
DEPTH = 5
MEMBERS = 4
TARGET_WIDTH = 512


def count_variant(variant: str, width: int) -> int:
    args = SimpleNamespace(model="SAGE", num_layers=DEPTH,
                           hidden_dim=width, m=MEMBERS)
    model = make_model(args, variant, INPUT_DIM, OUTPUT_DIM,
                       torch.device("cpu"))
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    return count


def choose_width(target: int, candidates: dict[int, int]) -> int:
    if not candidates:
        raise ValueError("No predeclared ensemble widths")
    return min(candidates, key=lambda width: (abs(candidates[width] - target), width))


def main() -> None:
    torch.set_num_threads(1)
    target = count_variant("gnnm", TARGET_WIDTH)
    if target != 6_737_728:
        raise RuntimeError(f"GNNM parameter count changed: {target}")
    candidates = {width: count_variant("ens_pooled", width) for width in WIDTHS}
    chosen = choose_width(target, candidates)
    result = {
        "dataset": "roman-empire",
        "model": "SAGE",
        "num_layers": DEPTH,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "members": MEMBERS,
        "target_variant": "gnnm",
        "target_width": TARGET_WIDTH,
        "target_trainable_parameters": target,
        "candidate_variant": "ens_pooled",
        "candidate_widths": list(WIDTHS),
        "candidate_trainable_parameters": {str(w): n for w, n in candidates.items()},
        "selection_rule": "Minimum absolute difference in trainable parameter count; smaller width breaks a tie. No validation or test score participates.",
        "selected_width": chosen,
        "selected_trainable_parameters": candidates[chosen],
        "selected_minus_target_parameters": candidates[chosen] - target,
        "selected_to_target_parameter_ratio": candidates[chosen] / target,
        "training_protocol": "Same five Roman Empire masks 0-4, SAGE depth 5, AdamW lr 3e-5 with zero weight decay, at most 5000 steps, pooled validation checkpoint selection every 10 steps, 300-step patience; only hidden width differs from the width-512 ENS control.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "selection.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
