"""Copied exact existing CPU-tree/exact-state/owned-stream utilities."""
# capture_rng, restore_rng, synchronize and parameter_counts origin:
# private_sheaf_train_valid_runner_20261009_v2/baseline_runner.py
# Original source notice: SPDX-License-Identifier: Apache-2.0
# cpu_tree and exact origin: bsnn_cayley_d2_train_valid_runner_source_20261009_v1/support.py
import random


def capture_rng(np, torch):
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(np, torch, state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])

def synchronize(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def parameter_counts(model):
    parameters = list(model.parameters())  # native members here have no sharing
    return {"active": sum(p.numel() for p in parameters if p.requires_grad),
            "total": sum(p.numel() for p in parameters),
            "unique_parameter_objects": len(parameters),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters)}

def cpu_tree(torch, value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(torch, v) for k, v in value.items()}
    if isinstance(value, tuple): return tuple(cpu_tree(torch, v) for v in value)
    if isinstance(value, list): return [cpu_tree(torch, v) for v in value]
    return value

def exact(torch, left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if type(left) is not type(right): return False
    if isinstance(left, dict): return set(left) == set(right) and all(exact(torch, v, right[k]) for k, v in left.items())
    if isinstance(left, (tuple, list)): return len(left) == len(right) and all(exact(torch, a, b) for a, b in zip(left, right))
    return left == right
