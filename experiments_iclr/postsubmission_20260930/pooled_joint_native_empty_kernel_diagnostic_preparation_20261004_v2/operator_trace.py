"""Observational fences and argument metadata; no numerical replacements."""
from contextlib import contextmanager
from pathlib import Path
import json
import math
import os
import time
import torch
from torch.utils._python_dispatch import TorchDispatchMode


def metadata(value):
    if torch.is_tensor(value):
        return {"type": "Tensor", "shape": list(value.shape), "dtype": str(value.dtype),
                "device": str(value.device), "layout": str(value.layout),
                "requires_grad": value.requires_grad}
    if type(value).__name__ == "SparseTensor":
        # Sizes/nnz are cached metadata; no coo/csr construction, tensor-value
        # reads, device transfers, new numerical operations or RNG calls.
        return {"type": "SparseTensor", "sizes": list(value.sizes()), "nnz": value.nnz()}
    if isinstance(value, (list, tuple)):
        return [metadata(x) for x in value]
    if isinstance(value, dict):
        return {str(k): metadata(v) for k,v in value.items()}
    # JSON tags describe metadata only; original operation arguments are untouched.
    if isinstance(value, float) and not math.isfinite(value):
        return {"type": "python_float", "value": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__}


class Trace:
    def __init__(self, output, atomic_json, *, limits):
        self.output, self.atomic_json = Path(output), atomic_json
        self.handle = (self.output / "OPERATORS.jsonl").open("x", buffering=1)
        self.started, self.counter = time.monotonic(), 0
        self.limits = limits
        self.max_events = limits["trace_events"]
        self.stack = []
        self.last = None
        self.first_error = None
        self.peak_bytes = {"allocated":0,"reserved":0}

    def event(self, event, **fields):
        # Reserve the last slot for a terminal observer-bound record. Such a
        # stop cannot be attributed to a native kernel.
        if self.counter >= self.max_events - 1:
            exc = RuntimeError("Diagnostic trace bound exceeded; no kernel attribution")
            if self.first_error is None:
                self.first_error = self._write_event("TRACE_BOUND_STOP_NO_KERNEL_ATTRIBUTION",
                    exception_type=type(exc).__name__,exception=str(exc))
                self.atomic_json(self.output / "FIRST_ERROR.json", self.first_error)
            raise exc
        return self._write_event(event, **fields)

    def _write_event(self, event, **fields):
        self.counter += 1
        row = {"sequence": self.counter, "elapsed_seconds": time.monotonic()-self.started,
               "event": event, "stack": list(self.stack), **fields}
        self.handle.write(json.dumps(row, allow_nan=False) + "\n")
        self.handle.flush(); os.fsync(self.handle.fileno())
        self.last = row
        return row

    def call(self, label, function, args=(), kwargs=None, extra=None, stage=None):
        kwargs = {} if kwargs is None else kwargs
        self.event("BEFORE_SYNC", label=label, args=metadata(args), kwargs=metadata(kwargs), extra=extra)
        try:
            torch.cuda.synchronize(0)
        except BaseException as exc:
            self.error("BEFORE_SYNC_ERROR_PRIOR_OPERATION", label, exc)
            raise
        self.event("BEGIN", label=label, args=metadata(args), kwargs=metadata(kwargs), extra=extra)
        if stage is not None:
            self.event("NATIVE_CALL_STAGE_SYNCHRONIZED", label=label, **stage)
        self.stack.append(label)
        try:
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                self.error("CALL_THROW_ERROR", label, exc)
                raise
            try:
                # Fence before any following operation can inherit an async error.
                torch.cuda.synchronize(0)
            except BaseException as exc:
                self.error("AFTER_SYNC_ERROR", label, exc)
                raise
            allocated = torch.cuda.max_memory_allocated(0)
            reserved = torch.cuda.max_memory_reserved(0)
            self.peak_bytes = {"allocated":max(allocated,self.peak_bytes["allocated"]),
                               "reserved":max(reserved,self.peak_bytes["reserved"])}
            if allocated > self.limits["CUDA_allocated_bytes"] or reserved > self.limits["CUDA_reserved_bytes"]:
                exc = RuntimeError("Diagnostic CUDA memory bound exceeded; no kernel attribution")
                self.error("RESOURCE_BOUND_STOP_NO_KERNEL_ATTRIBUTION",label,exc)
                raise exc
            self.event("END_SYNCHRONIZED", label=label, result=metadata(result))
            return result
        finally:
            self.stack.pop()

    def error(self, event, label, exc):
        if self.first_error is None:
            self.first_error = self.event(event, label=label, exception_type=type(exc).__name__, exception=str(exc))
            self.atomic_json(self.output / "FIRST_ERROR.json", self.first_error)
        # After a failed CUDA operation, callers propagate without another
        # kernel/fence/RNG probe. Nested wrappers do not replace first_error.

    def close(self):
        self.handle.close()


class SynchronizedOperators(TorchDispatchMode):
    def __init__(self, trace):
        super().__init__()
        self.trace = trace

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # Calling the same overload from this handler disables this mode for
        # that call by PyTorch's dispatch protocol. No argument/output edits.
        return self.trace.call(str(func), func, args, kwargs)


@contextmanager
def native_boundaries(mods, trace):
    native = mods["native"]
    original_forward = native.IncompleteCN1Predictor.forward
    original_spmm = native.spmm_add
    original_overlap = native.adjoverlap
    forward_stack = []
    forward_count = 0
    def forward(self, x, adj, tar_ei, filled1=False, depth=None):
        nonlocal forward_count
        forward_count += 1
        call_number = forward_count
        effective = self.depth if depth is None else depth
        stage = {"native_call_number":call_number,"parent_native_call":None if not forward_stack else forward_stack[-1],
                 "recursion_level":len(forward_stack),"effective_depth":effective,
                 "training":self.training,"query_columns":tar_ei.shape[1]}
        forward_stack.append(call_number)
        try:
            return trace.call("native.IncompleteCN1Predictor.forward#"+str(call_number), original_forward,
                              (self,x,adj,tar_ei), {"filled1":filled1,"depth":depth},stage=stage)
        finally:
            forward_stack.pop()
    def spmm(adj, x):
        return trace.call("native.spmm_add", original_spmm, (adj,x))
    def overlap(*args, **kwargs):
        return trace.call("native.adjoverlap", original_overlap, args, kwargs)
    native.IncompleteCN1Predictor.forward = forward
    native.spmm_add = spmm
    native.adjoverlap = overlap
    try:
        yield
    finally:
        native.IncompleteCN1Predictor.forward = original_forward
        native.spmm_add = original_spmm
        native.adjoverlap = original_overlap
