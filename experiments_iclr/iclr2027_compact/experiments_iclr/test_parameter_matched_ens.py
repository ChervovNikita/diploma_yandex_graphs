"""CPU smoke check for count-only selection and pooled ENS training."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import Data

from projector_controls import train_one
from select_parameter_matched_ens import OUT, choose_width, count_variant


def main() -> None:
    torch.set_num_threads(1)
    selection = json.loads((OUT / "selection.json").read_text())
    counts = {int(k): int(v) for k, v in
              selection["candidate_trainable_parameters"].items()}
    target = count_variant("gnnm", 512)
    assert target == selection["target_trainable_parameters"]
    assert choose_width(target, counts) == selection["selected_width"] == 256
    assert count_variant("ens_pooled", 256) == selection["selected_trainable_parameters"]
    assert choose_width(100, {1: 90, 2: 110}) == 1

    node_count = 12
    edge = torch.tensor([
        list(range(node_count)) + list(range(1, node_count)) + [0],
        list(range(1, node_count)) + [0] + list(range(node_count)),
    ], dtype=torch.long)
    data = Data(x=torch.arange(node_count * 5, dtype=torch.float32).view(node_count, 5) / 60,
                y=torch.tensor([0, 1, 2] * 4), edge_index=edge)
    masks = []
    for start, stop in ((0, 6), (6, 9), (9, 12)):
        mask = torch.zeros((node_count, 1), dtype=torch.bool)
        mask[start:stop, 0] = True
        masks.append(mask)
    args = SimpleNamespace(dataset="roman-empire", model="SAGE", num_layers=1,
                           hidden_dim=8, lr=3e-5, m=4, num_steps=11,
                           common_backbone_init=False)
    scratch = Path(__file__).resolve().parent / ".tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temp:
        row = train_one(args, "ens_pooled", 0, data, tuple(masks),
                        3, False, torch.device("cpu"), Path(temp))
        assert row["variant"] == "ens_pooled" and row["m"] == 4
        assert row["best_step"] in (1, 10)
        assert row["num_steps"] == 11
        assert Path(row["checkpoint"]).is_file()
        with np.load(row["prediction_file"]) as saved:
            assert saved["member_logits"].shape == (4, 3, 3)
            np.testing.assert_array_equal(saved["y_true"], data.y[9:].numpy())
    print("Count-only selection and pooled ENS CPU smoke check passed")


if __name__ == "__main__":
    main()
