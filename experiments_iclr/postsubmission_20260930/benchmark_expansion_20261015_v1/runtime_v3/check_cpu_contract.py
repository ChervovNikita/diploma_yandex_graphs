"""Small CPU admission/RNG contract checks; no native model/data or CUDA."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

import common
import run_fit


class CPUOnlyTorch:
    def __getattr__(self, name):
        if name == "cuda":
            raise AssertionError("CPU engineering attempted a CUDA API")
        return getattr(torch, name)


class TraceCUDA:
    def __init__(self, trace):
        self.trace = trace

    def manual_seed_all(self, seed):
        self.trace.append(("CUDA_seed_all", seed))

    def get_rng_state_all(self):
        self.trace.append(("CUDA_get_all",))
        return [torch.tensor([17], dtype=torch.uint8)]

    def set_rng_state_all(self, states):
        self.trace.append(("CUDA_set_all", [s.tolist() for s in states]))


class TraceTorch:
    def __init__(self):
        self.trace = []
        self.cuda = TraceCUDA(self.trace)

    def manual_seed(self, seed):
        self.trace.append(("torch_manual_seed", seed))
        return torch.random.default_generator.manual_seed(seed)

    def __getattr__(self, name):
        return getattr(torch, name)


def main():
    before_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    saved = common.global_rng(CPUOnlyTorch(), device="cpu")
    checks = []
    try:
        cpu = CPUOnlyTorch()
        common.seed_all(cpu, 14005101, device="cpu")
        state = common.global_rng(cpu, device="cpu")
        a = (random.random(), np.random.random(), torch.rand(8))
        common.restore_rng(cpu, state, device="cpu")
        b = (random.random(), np.random.random(), torch.rand(8))
        assert a[0] == b[0] and a[1] == b[1] and torch.equal(a[2], b[2])
        assert state["torch_cuda"] == []
        assert run_fit.resource_record(cpu, device="cpu")["cuda_resource_collection"] is False
        checks.append("CPU seed/capture/restore/resources invoke no CUDA API and reproduce all three RNGs")
        bad = copy.deepcopy(state)
        bad["torch_cuda"] = [torch.tensor([1], dtype=torch.uint8)]
        try:
            common.restore_rng(cpu, bad, device="cpu")
        except ValueError:
            pass
        else:
            raise AssertionError("CPU accepted CUDA RNG state")
        checks.append("CPU refuses CUDA RNG state")
        parent_path = Path(__file__).resolve().parent.parent / "runtime_v2/common.py"
        spec = importlib.util.spec_from_file_location("_runtime_v2_rng_contract", parent_path)
        parent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parent)
        traces = []
        for module in (parent, common):
            proxy = TraceTorch()
            module.seed_all(proxy, 17)
            value = module.global_rng(proxy)
            module.restore_rng(proxy, value)
            traces.append(proxy.trace)
        assert traces[0] == traces[1]
        checks.append("default GPU RNG API order and arguments match immutable v2 through a no-GPU trace")
        job = {"schema": "native-full-epoch-fit-v1", "family": "ogb", "TEST_access": False,
            "recipe": {"dataset": "ogbg-molbace"}, "procedure": "native_single", "seed": 17,
            "policy": {"track": "matched_pilot", "maximum_epochs": 2, "minimum_epochs": 2,
                "patience": 2, "metric": "rocauc"}}
        run_fit.validate_job(job)
        cpu_job = {**job, "device": "cpu", "execution_mode": "cpu_engineering_v1", "engineering_only": True}
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        run_fit.validate_job(cpu_job)
        refused = 0
        for delta in ({"execution_mode": "production_gpu"}, {"engineering_only": False},
                      {"device": "cuda:0"}, {"execution_mode": "unknown"}):
            try:
                run_fit.validate_job({**cpu_job, **delta})
            except ValueError:
                refused += 1
        assert refused == 4
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        try:
            run_fit.validate_job(cpu_job)
        except ValueError:
            pass
        else:
            raise AssertionError("CPU admitted visible CUDA")
        checks.append("legacy GPU descriptor accepted; CPU requires explicit mode/engineering marker/hidden CUDA")
    finally:
        common.restore_rng(CPUOnlyTorch(), saved, device="cpu")
        if before_env is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = before_env
    print(json.dumps({"schema": "runtime-v3-CPU-contract-checks-v1", "status": "passed",
        "torch": str(torch.__version__), "checks": checks, "native_constructions": 0,
        "model_forwards": 0, "optimizer_steps": 0, "task_data_accessed": False, "CUDA_execution": False}, indent=2))


if __name__ == "__main__":
    main()
