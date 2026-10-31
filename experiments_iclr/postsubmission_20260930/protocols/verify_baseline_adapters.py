"""CPU-only initialization/gradient checks for new-study sharing controls."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from baseline_adapters import (IndependentEnsemble, ProjectorEnsemble,
    SelectiveProjectorEnsemble, SharedTrunkMultihead, SingleModel,
    model_options, parameter_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite check record")
    torch.set_num_threads(1)
    torch.manual_seed(930)
    features = torch.randn(17, 7, dtype=torch.float64)
    graph = Data(edge_index=torch.randint(17, (2, 53)))
    reports = []
    for output_dim in (1, 3, 121):
        options = model_options(7, output_dim, hidden_dim=16, layers=3, dropout=0)
        original = ProjectorEnsemble(options).double()
        shared = SelectiveProjectorEnsemble(original).double()
        partial = SelectiveProjectorEnsemble(original, (1,)).double()
        private = SelectiveProjectorEnsemble(original, (0, 1, 2)).double()
        original_logits = original(graph, features)
        for control in (shared, partial, private):
            torch.testing.assert_close(control(graph, features), original_logits, rtol=0, atol=1e-12)
        counts = {"original": parameter_count(original), "shared": parameter_count(shared),
                  "partial": parameter_count(partial), "private": parameter_count(private)}
        assert counts["shared"] == counts["original"]
        assert counts["private"] - counts["shared"] == 3 * (counts["partial"] - counts["shared"])
        # At copied initialization, a shared block gradient equals the sum
        # over its corresponding member-private gradients under mean loss.
        original_logits.square().mean().backward()
        private(graph, features).square().mean().backward()
        gradient_error = 0.0
        for index, block in enumerate(original.model.residual_modules):
            for name, parameter in block.named_parameters():
                corresponding = [dict(member.named_parameters())[name]
                                 for member in private.private_blocks[str(index)]]
                total = torch.stack([p.grad for p in corresponding]).sum(0)
                torch.testing.assert_close(total, parameter.grad, rtol=1e-10, atol=1e-12)
                gradient_error = max(gradient_error, float((total - parameter.grad).abs().max()))
        for cls, members in ((SingleModel, 1), (IndependentEnsemble, 4), (SharedTrunkMultihead, 4)):
            control = cls(options).double()
            logits = control(graph, features)
            assert logits.shape == (members, len(features), output_dim)
            logits.square().mean().backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in control.parameters())
        reports.append({"output_dim": output_dim, "counts": counts,
                        "copied_initialization_logits": "equal",
                        "shared_gradient_sum_max_error": gradient_error})
    files = [Path(__file__), Path(__file__).with_name("baseline_adapters.py"),
             Path(__file__).resolve().parents[1] / "source_snapshot" / "models.py"]
    result = {"status": "passed", "scope": "synthetic CPU implementation checks; no benchmark score",
              "torch": torch.__version__, "checks": reports,
              "source_sha256": {str(p.relative_to(Path(__file__).resolve().parents[1])):
                                 hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
