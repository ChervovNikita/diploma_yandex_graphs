"""Disabled fixed synthetic branch diagnosis; no fit or qualification result.

Root may run after source review with --execute-authorized. No dataset reader,
state selection, new tolerance, backbone change or source-file mutation exists.
Temporary functional activation hooks always return the original outputs and
restore in finally. Transformed tensor values are explicitly skipped.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import signal
import socket
import sys
import time
import traceback

REPO = Path("/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs")
PHASE = REPO / "experiments_iclr/postsubmission_20260930"
QUALIFIER = "learnability_responsibility_native_numerical_worker_preparation_20261005_v1/qualify.py"
QUALIFIER_SHA = "976332545f78f1fe9642b2a4fa9d61127b427739cd85ea9a26301afc92cb61f7"
SCALES = (1e-5, 1e-6, 1e-7)
CALL_PHASES = ("before", "probe_private_gradient", "probe_adapted",
               "main_private_gradient", "query_adapted")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ActivationTrace:
    """Observe ordinary inputs; never retain a transformed tensor or frame."""

    def __init__(self, torch, functional, family):
        self.torch, self.functional = torch, functional
        self.module_names = {id(module): name for name, module in family.named_modules()}
        self.original = {"relu": functional.relu, "leaky_relu": functional.leaky_relu}
        self.context = None
        self.records, self.skipped = [], []
        self.callback_count = 0
        self.restored = False

    def _identity(self, kind, frame):
        context = self.context
        ordinal = context["activation_ordinal"]
        context["activation_ordinal"] += 1
        index = frame.f_locals.get("i")
        return {"phase": context["phase"], "member": context["member"],
                "callback_ordinal": context["callback_ordinal"], "activation_ordinal": ordinal,
                "activation": kind, "module_path": self.module_names.get(id(frame.f_locals.get("self"))),
                "loop_layer_index": index if type(index) is int else None,
                "source_path": frame.f_code.co_filename, "source_line": frame.f_lineno,
                "source_function": frame.f_code.co_name}

    def _observe(self, kind, args, kwargs, frame):
        if self.context is None:
            self.skipped.append({"reason": "outside_labeled_callback", "activation": kind})
            return
        identity = self._identity(kind, frame)
        value = args[0] if args else kwargs.get("input")
        torch = self.torch
        try:
            if not isinstance(value, torch.Tensor):
                self.skipped.append({"identity": identity, "reason": "non_tensor_input"})
                return
            backend = getattr(torch._C, "_functorch", None)
            detector = getattr(backend, "is_functorch_wrapped_tensor", None)
            if detector is None:
                self.skipped.append({"identity": identity, "reason": "transform_detector_unavailable"})
                return
            if detector(value):
                self.skipped.append({"identity": identity, "reason": "transformed_tensor_not_inspected"})
                return
            if type(value) is not torch.Tensor or value.layout != torch.strided:
                self.skipped.append({"identity": identity, "reason": "unsupported_tensor_type_or_layout"})
                return
            # Clone before the original activation, including any inplace case.
            plain = value.detach().cpu().clone()
            finite = torch.isfinite(plain)
            mask = torch.zeros_like(plain, dtype=torch.int8)
            mask[plain < 0] = -1
            mask[plain > 0] = 1
            mask[~finite] = 2
            counts = {"negative": int((mask == -1).sum()), "zero": int((mask == 0).sum()),
                      "positive": int((mask == 1).sum()), "nonfinite": int((mask == 2).sum())}
            self.records.append({"identity": identity, "shape": list(plain.shape),
                "dtype": str(plain.dtype), "counts": counts,
                "sign_mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
                "sign_mask": mask, "values": plain,
                "negative_slope": (args[1] if len(args) > 1 else kwargs.get("negative_slope", 0.01))
                                  if kind == "leaky_relu" else None})
        except TimeoutError:
            raise
        except Exception as error:
            self.skipped.append({"identity": identity, "reason": "capture_unsupported",
                                 "error_type": type(error).__name__, "error": str(error)})

    def _wrapper(self, kind):
        original = self.original[kind]

        def hooked(*args, **kwargs):
            frame = sys._getframe(1)
            try:
                self._observe(kind, args, kwargs, frame)
            finally:
                del frame
            return original(*args, **kwargs)  # Original arguments and result, unchanged.

        return hooked

    def __enter__(self):
        try:
            self.functional.relu = self._wrapper("relu")
            self.functional.leaky_relu = self._wrapper("leaky_relu")
        except BaseException:
            self.restore()
            raise
        return self

    def __exit__(self, kind, error, trace):
        self.restore()

    def restore(self):
        try:
            self.functional.relu = self.original["relu"]
        finally:
            self.functional.leaky_relu = self.original["leaky_relu"]
            self.context = None
            self.restored = (self.functional.relu is self.original["relu"] and
                             self.functional.leaky_relu is self.original["leaky_relu"])

    def reset_point(self):
        self.context = None
        self.records, self.skipped = [], []
        self.callback_count = 0

    def forward(self, original):
        def traced(core, private):
            ordinal = self.callback_count
            self.callback_count += 1
            wave = ordinal // 4
            self.context = {"phase": CALL_PHASES[wave] if wave < len(CALL_PHASES) else "unclassified",
                            "member": ordinal % 4, "callback_ordinal": ordinal,
                            "activation_ordinal": 0}
            try:
                return original(core, private)
            finally:
                self.context = None
        return traced


def compare_masks(torch, base, shifted):
    """All captured sites and all changed coordinates; no truncation or cutoff."""
    before = {json.dumps(row["identity"], sort_keys=True): row for row in base}
    after = {json.dumps(row["identity"], sort_keys=True): row for row in shifted}
    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    transitions, changes = [], 0
    for key in sorted(set(before) & set(after)):
        left, right = before[key], after[key]
        if left["shape"] != right["shape"]:
            transitions.append({"identity": left["identity"], "reason": "shape_changed",
                                "base_shape": left["shape"], "shifted_shape": right["shape"]})
            continue
        a, b = left["sign_mask"], right["sign_mask"]
        changed = a != b
        count = int(changed.sum())
        changes += count
        if not count:
            continue
        coordinates = torch.nonzero(changed, as_tuple=False)
        indexes = tuple(coordinates[:, dimension] for dimension in range(coordinates.shape[1]))
        counts = {f"{x}_to_{y}": int(((a == x) & (b == y)).sum())
                  for x in (-1, 0, 1, 2) for y in (-1, 0, 1, 2) if x != y}
        transitions.append({"identity": left["identity"], "changed_count": count,
            "transition_counts": counts, "coordinates": coordinates.tolist(),
            "base_signs": a[indexes].tolist(), "shifted_signs": b[indexes].tolist(),
            "base_values": left["values"][indexes].tolist(),
            "shifted_values": right["values"][indexes].tolist()})
    return {"changed_coordinates": changes, "changed_sites": len(transitions),
            "site_identity_missing": missing, "site_identity_extra": extra, "transitions": transitions}


def _metadata(records):
    return [{key: value for key, value in row.items() if key not in {"sign_mask", "values"}}
            for row in records]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED", "numeric_imports": False}))
        return
    if not args.output:
        parser.error("Reviewed authorized run requires a fresh --output directory")
    output = Path(args.output)
    output.mkdir(parents=False, exist_ok=False)
    start = time.monotonic()
    result = {"status": "RUNNING_BRANCH_DIAGNOSTIC", "scales": list(SCALES), "fixed_deadline_seconds": 180,
              "model_fits": 0, "dataset_label_access": False, "persistent_updates": 0,
              "qualification_pass_or_new_tolerance": False, "predictive_evidence": False,
              "source_sha256": _sha(Path(__file__)), "points": [], "finite_differences": []}
    captured_masks = []
    observer = None

    def save():
        result["elapsed_seconds"] = time.monotonic() - start
        result["peak_RSS_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def timeout(signum, frame):
        raise TimeoutError("Fixed180-second branch diagnosis exceeded")

    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(180)
    try:
        save()
        if Path.cwd().resolve() != REPO or socket.gethostname() != "anogena-2-0":
            raise RuntimeError("Same root diagnostic allocation/working directory required")
        if Path(sys.executable).absolute() != PHASE / "native_ncn_runtime_20261005_v1/.venv/bin/python":
            raise RuntimeError("Same root diagnostic normal interpreter required")
        if not sys.dont_write_bytecode:
            raise RuntimeError("Run with -B to preserve bound sources")
        qualifier_path = PHASE / QUALIFIER
        if _sha(qualifier_path) != QUALIFIER_SHA:
            raise RuntimeError("Immutable qualifier source changed")
        bindings = json.loads((Path(__file__).parent / "SOURCE_BINDINGS.json").read_text())
        reference_row = bindings["files"]["root_diagnostic_result"]
        reference_path = PHASE / reference_row["path"]
        if _sha(reference_path) != reference_row["sha256"]:
            raise RuntimeError("Saved same-state root diagnosis differs")
        reference = json.loads(reference_path.read_text())
        sys.path.insert(0, str(REPO / ".venv/lib/python3.11/site-packages"))
        import torch
        import numpy as np
        import torch.nn.functional as F
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        spec = importlib.util.spec_from_file_location("branch_immutable_qualifier", qualifier_path)
        q = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = q
        spec.loader.exec_module(q)
        q.torch, q.np = torch, np
        modules = {name: q.load_module("branch_" + name, PHASE / path, digest)
                   for name, (path, digest) in q.PINS.items()}
        op, port = modules["operator"], modules["port"]
        if op.SOURCE_RELEASED is not False or port.PORT_RELEASED is not False:
            raise RuntimeError("Operator/port source guards must remain unchanged")
        generator = torch.Generator().manual_seed(126005)
        x = torch.randn(15, 300, dtype=torch.float64, generator=generator)
        support = set((i, i) for i in range(15))
        for i in range(14):
            support.update({(i, i + 1), (i + 1, i)})
        for i, j in [(0, 4), (1, 6), (2, 7), (3, 8), (4, 9), (0, 9)]:
            support.update({(i, j), (j, i)})
        edges = torch.tensor(sorted(support), dtype=torch.long).T.contiguous()
        s, r = torch.arange(10), torch.arange(10, 15)
        ys, yr = torch.arange(10) % 5, torch.arange(5)
        device = torch.device("cpu")
        family = q.build_family(modules["native"], modules["boundary"], device, torch.float64, 106005)
        forward, theta, phis, _ = port._native_callback_and_state(family, x, edges, expected_nodes=15)
        pairs = port._sparse_pairs(op, s, ys, edges, node_count=15, dtype=torch.float64)
        before = q.snapshot(family, x, edges, device)
        live = lambda core: q.outer(op, core, phis, forward, pairs, s, ys, r, yr)
        gradient = torch.func.grad(live)(theta)  # Same fixed base derivative, no activation hooks.
        generator = torch.Generator().manual_seed(116005)
        direction = {name: torch.zeros_like(value) if name.startswith("local_head.") else
                     torch.randn(value.shape, dtype=torch.float64, generator=generator)
                     for name, value in theta.items()}
        norm = sum(value.square().sum() for value in direction.values()).sqrt()
        direction = {name: value / norm for name, value in direction.items()}
        analytic = float(sum((gradient[name] * direction[name]).sum() for name in theta))
        result.update(torch_func_directional=analytic,
            full_shared_gradient_norm=float(sum(value.square().sum() for value in gradient.values()).sqrt()),
            root_saved_live_Q_chain_gradient_norm=reference["live_Q_chain_gradient_norm"],
            chain_norm_recomputed_here=False, source_pins=q.PINS,
            shared_coordinates=sum(value.numel() for value in gradient.values()))
        del gradient
        save()
        observer = ActivationTrace(torch, F, family)
        base_records = None
        values = {}
        fixed_points = [("base", 0.0)] + [(f"{sign:+d}x{eps:.0e}", sign * eps)
                                         for eps in SCALES for sign in (1, -1)]
        with observer:  # __exit__ restores both functional functions even on errors/signals.
            for label, shift in fixed_points:
                observer.reset_point()
                core = theta if shift == 0 else {name: value + shift * direction[name]
                                                for name, value in theta.items()}
                traced = observer.forward(forward)
                value = float(q.outer(op, core, phis, traced, pairs, s, ys, r, yr))
                values[label] = value
                records = observer.records
                if label == "base":
                    base_records = records
                    transitions = None
                else:
                    transitions = compare_masks(torch, base_records, records)
                result["points"].append({"id": label, "signed_shift": shift, "objective": value,
                    "callback_count": observer.callback_count, "expected_callback_count": 20,
                    "phase_mapping_complete": observer.callback_count == 20,
                    "captured_sites": _metadata(records), "skipped_calls": observer.skipped,
                    "versus_base": transitions})
                captured_masks.append({"point": label, "records": [
                    {"identity": row["identity"], "shape": row["shape"], "sign_mask": row["sign_mask"]}
                    for row in records]})
                save()
        for eps in SCALES:
            plus, minus = values[f"+1x{eps:.0e}"], values[f"-1x{eps:.0e}"]
            result["finite_differences"].append({"epsilon": eps, "positive_value": plus,
                "negative_value": minus, "base_value": values["base"],
                "right": (plus - values["base"]) / eps, "left": (values["base"] - minus) / eps,
                "central": (plus - minus) / (2 * eps), "analytic": analytic,
                "central_absolute_difference": abs((plus - minus) / (2 * eps) - analytic)})
        mask_path = output / "BRANCH_MASKS.pt"
        with mask_path.open("xb") as stream:
            torch.save({"encoding": {-1: "negative", 0: "zero", 1: "positive", 2: "nonfinite"},
                        "points": captured_masks}, stream)
        result["branch_masks"] = {"path": mask_path.name, "sha256": _sha(mask_path),
                                   "bytes": mask_path.stat().st_size}
        result["base_value_absolute_difference_vs_root"] = abs(values["base"] - reference["torch_func_value"])
        result["directional_absolute_difference_vs_root"] = abs(analytic - reference["torch_func_directional"])
        changes = sum(point["versus_base"]["changed_coordinates"] for point in result["points"]
                      if point["versus_base"] is not None)
        result["observed_sign_or_zero_transition_coordinates"] = changes
        result["interpretation"] = ("Exact sign/zero transitions observed at captured ordinary callbacks; "
            "this identifies branch boundaries but does not isolate the cause of the left FD instability"
            if changes else "No sign/zero transition observed in captured ordinary callbacks; reason unresolved")
        result["cause_of_unstable_left_FD"] = "UNRESOLVED_BY_BRANCH_OBSERVATION_ALONE"
        result["native_restoration"] = q.unchanged(before, family, x, edges, device)
        result["status"] = "COMPLETE_BRANCH_DIAGNOSIS_NOT_QUALIFICATION"
    except BaseException as error:
        result.update(status="FAIL_BRANCH_DIAGNOSTIC", error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        if observer is not None:
            observer.restore()
        result["functional_hooks_restored"] = observer.restored if observer is not None else "not_installed"
        save()
    print(json.dumps({"status": result["status"], "output": str(output)}))
    raise SystemExit(0 if result["status"].startswith("COMPLETE_") else 1)


if __name__ == "__main__":
    main()
