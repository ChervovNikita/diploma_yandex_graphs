"""Freeze UNTIED widths by parameter count only, before any narrow run."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
STUDY = HERE / "roman_mechanism_v3_prepared"
sys.path.insert(0, str(STUDY))
import roman_multimask as roman  # noqa: E402

OUTPUT = HERE / "ROMAN_NARROW_UNTIED_WIDTH_LOCK.json"
DEPTHS = (2, 5)
WIDTHS = range(32, 129)
INPUT_DIM = 300
CLASSES = 18


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameters(arm: str, depth: int, width: int) -> int:
    roman.MASK = 0
    roman.DEPTH = depth
    roman.WIDTH = width
    model, _ = roman.make_model(arm, INPUT_DIM, CLASSES, torch.device("cpu"))
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    assert not OUTPUT.exists()
    assert sha(STUDY / "ROMAN_MECHANISM_SOURCE_FREEZE.json") == (
        "489451d58772c073e7be04bd1d1f7859644f56a6626d9b4a14cd6f752acc0088")
    records = []
    for depth in DEPTHS:
        target = parameters("tied", depth, 128)
        assert target == {2: 208960, 5: 456640}[depth]
        candidates = [{"width": width,
                       "untied_parameter_count": parameters(
                           "untied_propagation", depth, width)}
                      for width in WIDTHS]
        selected = min(candidates,
                       key=lambda row: (abs(row["untied_parameter_count"] - target),
                                        row["width"]))
        records.append({"depth": depth, "tied_width": 128,
                        "tied_parameter_count": target,
                        "selected_untied_width": selected["width"],
                        "selected_untied_parameter_count": selected["untied_parameter_count"],
                        "absolute_parameter_difference": abs(
                            selected["untied_parameter_count"] - target),
                        "candidates": candidates})
    OUTPUT.write_text(json.dumps({
        "status": "PARAMETER_COUNT_ONLY_WIDTH_LOCK_BEFORE_NARROW_TRAINING",
        "width_selection": "minimize absolute parameter-count difference over integer widths 32..128; choose smaller width on ties",
        "data": "Roman Empire official mask 0; input dimension 300; 18 classes",
        "original_v3_source_freeze_sha256": sha(
            STUDY / "ROMAN_MECHANISM_SOURCE_FREEZE.json"),
        "model_source_sha256": sha(STUDY / "models.py"),
        "roman_adapter_source_sha256": sha(STUDY / "roman_multimask.py"),
        "width_lock_source_sha256": sha(Path(__file__)),
        "test_scores_consulted_for_width_selection": False,
        "records": records}, indent=2, sort_keys=True) + "\n")
    for row in records:
        print("WIDTH_LOCK", row["depth"], row["selected_untied_width"],
              row["selected_untied_parameter_count"],
              row["tied_parameter_count"], flush=True)
    print("WIDTH_LOCK_SHA256", sha(OUTPUT), flush=True)


if __name__ == "__main__":
    main()
