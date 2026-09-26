"""Public stdlib audit of selected-seed0 normalized Cora matched54 resource records."""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def quantile(values, probability):
    values = sorted(values)
    x = (len(values) - 1) * probability
    low, high = math.floor(x), math.ceil(x)
    return values[low] * (high - x) + values[high] * (x - low) if low != high else values[low]


def main(folder, evidence):
    report_path = folder / "PROFILE_SELECTED_SEED0.json"
    report = json.loads(report_path.read_text())
    redaction = json.loads((folder / "PROFILE_PUBLIC_REDACTION_AUDIT.json").read_text())
    require(redaction["protocol"] == "cora_matched54_profile_public_identifier_redaction_v1" and
            redaction["original_author_profile_sha256"] ==
            "f638e0a78a6eb992eea934fbf38ee6b26aaedd2de825ef88312994248e7dc7c5" and
            redaction["public_profile_sha256"] == sha(report_path) and
            redaction["redacted_fields"] ==
            {"gpu_before": 2, "gpu_after": 2, "device_properties": 1} and
            not re.search(r"(?:GPU-)?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                          report_path.read_text(), re.I),
            "Public device-ID redaction differs")
    lock_path = evidence / "VALIDATION_SELECTION_LOCK.json"
    freeze_path = evidence / "FROZEN_CORA_MATCHED54.json"
    lock = json.loads(lock_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    require(report["protocol"] == "cora_matched54_selected_seed0_profile_v1" and
            report["posthoc"] is True and
            report["source_sha256"] == sha(folder / "profile_cora_matched54_selected.py") and
            report["primary_profile_source_sha256"] == sha(folder / "profile_selected_inference.py") and
            report["freeze_sha256"] == sha(freeze_path) and
            report["selection_lock_sha256"] == sha(lock_path) and
            lock["freeze_sha256"] == report["freeze_sha256"] and
            report["rounds"] == 3 and report["warmups_per_block"] == 20 and
            report["repetitions_per_block"] == 100 and report["optimization_seed"] == 0,
            "Profile source, protocol, or selection binding differs")
    expected_counts = {"base": 605919, "ens": 602868, "private_last": 604700}
    require(set(report["arms"]) == set(expected_counts) and
            freeze["matrix"]["parameter_counts"] == expected_counts,
            "Profile arm set or stored-weight counts differ")
    total = 0
    for arm, count in expected_counts.items():
        row = report["arms"][arm]
        lr, wd = row["selected_candidate"]
        identity = f"cora_normalized/{arm}/lr{lr:g}_wd{wd:g}/seed0"
        require([lr, wd] == lock["selections"][arm]["selected_candidate"] and
                row["checkpoint_sha256"] == lock["cells"][identity]["checkpoint_sha256"] and
                row["parameter_count"] == count and row["parameter_bytes"] == 4*count and
                len(row["blocks"]) == 3,
                f"Selected checkpoint or parameter identity differs: {arm}")
        values = []
        for round_index, block in enumerate(row["blocks"]):
            durations = block["latency_ms"]
            require(block["round"] == round_index and len(durations) == 100 and
                    all(math.isfinite(x) and x > 0 for x in durations) and
                    block["idle_allocated_bytes"] >= 0 and
                    block["peak_allocated_bytes"] >= block["idle_allocated_bytes"] and
                    block["peak_increment_bytes"] ==
                    block["peak_allocated_bytes"] - block["idle_allocated_bytes"],
                    f"Invalid timing or memory block: {arm}/{round_index}")
            require(abs(statistics.median(durations) - row["block_latency_medians_ms"][round_index]) < 1e-9,
                    f"Block median differs: {arm}/{round_index}")
            values.extend(durations)
        require(abs(statistics.median(values) - row["latency_ms_median"]) < 1e-9 and
                abs(quantile(values, .25) - row["latency_ms_q25"]) < 1e-9 and
                abs(quantile(values, .75) - row["latency_ms_q75"]) < 1e-9 and
                row["peak_allocated_bytes_max"] == max(b["peak_allocated_bytes"] for b in row["blocks"]) and
                row["peak_increment_bytes_max"] == max(b["peak_increment_bytes"] for b in row["blocks"]),
                f"Aggregate timing or memory differs: {arm}")
        total += len(values)
    require(total == 900, "Expected all 900 synchronized timing samples")
    print(json.dumps({"status": "PASS", "timing_samples": total,
                      "selected_seed0_arms": 3, "profile_sha256": sha(report_path)}))


if __name__ == "__main__":
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    evidence = Path(sys.argv[2] if len(sys.argv) > 2 else folder.parent / "cora_matched54").resolve()
    main(folder, evidence)
