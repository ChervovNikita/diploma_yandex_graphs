"""Small CPU checks for split isolation and joint pooled selection.

No OGB download or GPU is used. Temporary output stays inside this repo.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ogbn_arxiv_pilot import (  # noqa: E402
    Configuration, GraphBundle, better_validation, pooled_metrics,
    repo_path, run_one,
)
from ogbn_arxiv_diagnostics import analyze_predictions  # noqa: E402


def test_joint_pool_selection() -> None:
    labels = torch.tensor([0, 1])
    individually_good = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    individually_bad = torch.tensor([[0.0, 40.0], [40.0, 0.0]])
    moderate = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    epoch_one = [individually_good] * 3 + [individually_bad]
    epoch_two = [moderate] * 4
    acc_one, ce_one = pooled_metrics(epoch_one, labels)
    acc_two, ce_two = pooled_metrics(epoch_two, labels)
    assert acc_one == 0.0 and acc_two == 1.0
    assert better_validation(acc_one, ce_one, -float("inf"), float("inf"))
    assert better_validation(acc_two, ce_two, acc_one, ce_one)
    assert not better_validation(acc_one, ce_one, acc_two, ce_two)
    # The first three members individually prefer epoch one by CE. A joint
    # pooled checkpoint therefore cannot be replaced by member-wise picks.
    assert pooled_metrics([individually_good], labels)[1] < pooled_metrics([moderate], labels)[1]


def synthetic_bundle(test_labels: torch.Tensor) -> GraphBundle:
    n = 12
    edge = torch.tensor(
        [[i for i in range(n - 1)] + [i for i in range(1, n)],
         [i for i in range(1, n)] + [i for i in range(n - 1)]],
        dtype=torch.long,
    )
    x = torch.arange(n * 4, dtype=torch.float32).view(n, 4) / 48.0
    labels = torch.tensor([0, 1] * (n // 2), dtype=torch.long)
    train_idx = torch.arange(0, 6)
    valid_idx = torch.arange(6, 9)
    test_idx = torch.arange(9, 12)
    return GraphBundle(
        graph=SimpleNamespace(edge_index=edge), x=x,
        train_idx=train_idx, valid_idx=valid_idx,
        test_idx_cpu=test_idx,
        train_y=labels[train_idx], valid_y=labels[valid_idx],
        test_y_cpu=test_labels.clone(), num_classes=2,
    )


def rows_without_time(path: Path) -> list[tuple[str, ...]]:
    columns = ("epoch", "train_mean_member_ce", "valid_pooled_accuracy",
               "valid_pooled_ce", "is_selected_so_far")
    with path.open(newline="") as f:
        return [tuple(row[key] for key in columns) for row in csv.DictReader(f)]


def test_test_label_isolation() -> None:
    torch.set_num_threads(1)
    cfg = Configuration(layers=1, hidden_dim=8, dropout=0.0, lr=0.01,
                        max_epochs=3, min_epochs=3, patience=3)
    device = torch.device("cpu")
    temp_parent = repo_path("experiments_iclr/.tmp/ogbn_arxiv_tests")
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_parent) as temp:
        root = Path(temp)
        first = run_one(synthetic_bundle(torch.zeros(3, dtype=torch.long)),
                        cfg, "base", 17, device, root / "first")
        with np.load(root / "first" / "selected_predictions.npz") as archive:
            selected_predictions = archive["test_member_logits"].mean(axis=0).argmax(axis=1)
        second = run_one(synthetic_bundle(torch.tensor(selected_predictions)),
                         cfg, "base", 17, device, root / "matching")
        opposite = torch.tensor((selected_predictions + 1) % 2)
        third = run_one(synthetic_bundle(opposite), cfg, "base", 17,
                        device, root / "opposite")
        assert first["selected_epoch"] == second["selected_epoch"] == third["selected_epoch"]
        assert second["test_accuracy"] == 1.0 and third["test_accuracy"] == 0.0
        for name in ("matching", "opposite"):
            assert rows_without_time(root / "first" / "epochs.csv") == \
                rows_without_time(root / name / "epochs.csv")
            a = torch.load(root / "first" / "selected_checkpoint.pt",
                           map_location="cpu", weights_only=True)["state_dict"]
            b = torch.load(root / name / "selected_checkpoint.pt",
                           map_location="cpu", weights_only=True)["state_dict"]
            assert a.keys() == b.keys()
            assert all(torch.equal(a[key], b[key]) for key in a)
            with np.load(root / "first" / "selected_predictions.npz") as pa, \
                    np.load(root / name / "selected_predictions.npz") as pb:
                np.testing.assert_array_equal(pa["valid_member_logits"], pb["valid_member_logits"])
                np.testing.assert_array_equal(pa["test_member_logits"], pb["test_member_logits"])


def test_four_member_training_and_joint_checkpoint() -> None:
    torch.set_num_threads(1)
    cfg = Configuration(layers=1, hidden_dim=8, dropout=0.0, lr=0.01,
                        max_epochs=3, min_epochs=3, patience=3)
    temp_parent = repo_path("experiments_iclr/.tmp/ogbn_arxiv_tests")
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_parent) as temp:
        for variant in ("ens", "gnnm"):
            out = Path(temp) / variant
            result = run_one(synthetic_bundle(torch.zeros(3, dtype=torch.long)),
                             cfg, variant, 23, torch.device("cpu"), out)
            rows = rows_without_time(out / "epochs.csv")
            selected = [int(row[0]) for row in rows if row[-1] == "1"]
            assert selected and result["selected_epoch"] == selected[-1]
            with np.load(out / "selected_predictions.npz") as archive:
                assert archive["valid_member_logits"].shape == (4, 3, 2)
                assert archive["test_member_logits"].shape == (4, 3, 2)
                pooled = archive["valid_member_logits"].mean(axis=0)
                acc = (pooled.argmax(axis=1) == archive["valid_labels"]).mean()
                assert abs(float(acc) - result["valid_accuracy"]) < 1e-7


def test_decision_diagnostics() -> None:
    logits = np.array([
        [[2.0, 0.0], [2.0, 0.0]],
        [[0.0, 2.0], [2.0, 0.0]],
    ])
    row, groups = analyze_predictions(logits, np.array([0, 1]))
    assert row["mean_member_accuracy"] == 0.25
    assert row["pooled_accuracy"] == 0.5
    assert row["pairwise_prediction_disagreement"] == 0.5
    assert [group["node_count"] for group in groups] == [1, 1, 0]


if __name__ == "__main__":
    test_joint_pool_selection()
    test_test_label_isolation()
    test_four_member_training_and_joint_checkpoint()
    test_decision_diagnostics()
    print("CPU synthetic checks passed: split isolation, joint selection, and diagnostics")
