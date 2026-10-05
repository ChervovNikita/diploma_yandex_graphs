#!/usr/bin/env python3
"""Complete native Citeseer-HeaRT NCN fits; TRAIN/VALID only, no TEST loader."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = Path("/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs")
PHASE = REPO / "experiments_iclr/postsubmission_20260930"
GPU_UUID = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"
ARM_MEMBERS = {"native_single": 1, "independent_member": 1, "independent_frame_member": 1,
               "unframed_f4": 4, "shared_frame_f4": 4, "private_frame_f4": 4,
               "same_four_frames_single": 1}
AVAILABLE_NAMES = {"train_pos.txt", "valid_pos.txt", "heart_valid_samples.npy", "gnn_feature"}
MAX_EPOCHS, EVAL_STEPS, KILL_CNT, BATCH, VALID_BATCH = 9999, 5, 10, 1024, 512


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while part := stream.read(1024 * 1024):
            value.update(part)
    return value.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def phase_file(relative):
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("Job file references must be relative to the pinned phase")
    p = (PHASE / p).resolve(strict=True)
    if not p.is_relative_to(PHASE.resolve(strict=True)) or not p.is_file():
        raise ValueError("Input reference leaves the pinned phase")
    return p


def authorize(job_path, output):
    if platform.system() != "Linux" or not ROOT.resolve().is_relative_to(PHASE.resolve(strict=True)):
        raise ValueError("Runner must be staged inside the authorized Linux one-GPU phase")
    job_path = job_path.resolve(strict=True)
    if not job_path.is_relative_to(PHASE.resolve(strict=True)):
        raise ValueError("Job leaves authorized phase")
    job = json.loads(job_path.read_text())
    if socket.gethostname() != job.get("expected_hostname"):
        raise ValueError("Independent hostname binding differs")
    query = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                           check=True, capture_output=True, text=True, timeout=30)
    if query.stdout.split() != [GPU_UUID]:
        raise ValueError("Authorized singleton GPU UUID differs")
    if output.exists() or not output.resolve().is_relative_to(PHASE.resolve(strict=True)) or not output.parent.is_dir():
        raise ValueError("Use a fresh output inside the pinned phase")
    if job.get("source_review_approved") is not True or job.get("runner_sha256") != sha(Path(__file__)):
        raise ValueError("Root-reviewed exact runner release absent")
    manifest_path = ROOT / "SOURCE_MANIFEST.json"
    if job.get("source_manifest_sha256") != sha(manifest_path):
        raise ValueError("Root-reviewed source manifest absent")
    for item in json.loads(manifest_path.read_text())["files"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Reviewed runner/model/vendor bytes changed")
    arm = job.get("arm")
    if arm not in ARM_MEMBERS or type(job.get("member_count")) is not int or job["member_count"] != ARM_MEMBERS[arm]:
        raise ValueError("Explicit member_count differs from native-single or F4 arm")
    if type(job.get("seed")) is not int or not 0 <= job["seed"] < 2**32:
        raise ValueError("Require an explicit seed; no cohort or run defaults")
    if not job.get("cohort_plan_sha256") or not job.get("paired_seed_block"):
        raise ValueError("Prospective cohort identity/block is absent")
    if arm in ("independent_member", "independent_frame_member") and (type(job.get("ensemble_member_index")) is not int or job["ensemble_member_index"] not in range(4)):
        raise ValueError("Independent constituent requires explicit member_index0..3")
    if arm == "independent_frame_member" and job.get("axis_index") != job["ensemble_member_index"]:
        raise ValueError("Same-operation ensemble must preserve axes0,1,2,3")
    for key in ("feature_authority", "negative_pool_authority"):
        authority = job.get(key, {})
        if authority.get("verified") is not True or not authority.get("origin") or not authority.get("evidence"):
            raise ValueError(key + " remains unresolved; do not train")
        for record in authority["evidence"]:
            if sha(phase_file(record["path"])) != record["sha256"]:
                raise ValueError("Root qualification evidence changed")
    if job.get("runtime_qualified") is not True:
        raise ValueError("Root native operator/runtime qualification absent")
    return job


def load_available(job):
    import inspect
    import numpy as np
    import torch
    manifest_path = phase_file(job["available_manifest_relative"])
    if sha(manifest_path) != job["available_manifest_sha256"]:
        raise ValueError("Exact acquisition available manifest differs")
    manifest = json.loads(manifest_path.read_text())
    if set(manifest["files"]) != AVAILABLE_NAMES or manifest.get("TEST_available_to_loader") is not False:
        raise ValueError("Only four TRAIN/VALID file roles may reach runner")
    paths, identities = {}, {}
    for name, record in manifest["files"].items():
        path = manifest_path.parent / record["relative_path"]
        if tuple(Path(record["relative_path"]).parts) != ("available", "citeseer", name):
            raise ValueError("Acquisition file role/path mismatch")
        path = path.resolve(strict=True)
        if not path.is_relative_to(PHASE.resolve(strict=True)) or sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("Exact available input bytes differ")
        paths[name], identities[name] = path, {"sha256": record["sha256"], "bytes": record["bytes"]}
    if "weights_only" not in inspect.signature(torch.load).parameters:
        raise ValueError("Require weights_only feature load; no unsafe pickle fallback")
    supplied = torch.load(paths["gnn_feature"], map_location="cpu", weights_only=True)
    x = supplied.get("entity_embedding") if isinstance(supplied, dict) else None
    if not isinstance(x, torch.Tensor) or x.layout != torch.strided or x.dtype != torch.float32 or list(x.shape) != job["qualified_feature_shape"] or not bool(torch.isfinite(x).all()):
        raise ValueError("Qualified finite native float32 feature geometry differs")
    if list(x.shape) != [3327, 3703]:
        raise ValueError("Pinned representative Citeseer population/features differ; new review required")
    rows = {}
    for split in ("train", "valid"):
        values, raw, selfs = [], 0, 0
        with paths[split + "_pos.txt"].open() as stream:
            for line in stream:
                fields = line.strip().split("\t")
                if len(fields) != 2:
                    raise ValueError("Expected native two-column positives")
                u, v = map(int, fields)
                if not (0 <= u < len(x) and 0 <= v < len(x)):
                    raise ValueError("Positive endpoint population differs")
                raw += 1
                if u == v:
                    selfs += 1
                    continue  # Same native loader removes source self links.
                values.append((u, v))
        record = manifest["files"][split + "_pos.txt"]["counts"]
        if raw != record["raw_rows"] or len(values) != record["native_nonself_rows"] or selfs != record["self_loops"]:
            raise ValueError("Acquisition positive count identity differs")
        if len(set(tuple(sorted(p)) for p in values)) != len(values):
            raise ValueError("Duplicate positive pairs require explicit source review")
        rows[split] = torch.tensor(values, dtype=torch.long)
    if len(rows["train"]) != 3870 or len(rows["valid"]) != 227:
        raise ValueError("Authenticated fixed Citeseer native split size differs")
    if {tuple(sorted(p)) for p in rows["train"].tolist()} & {tuple(sorted(p)) for p in rows["valid"].tolist()}:
        raise ValueError("TRAIN/VALID overlap")
    pool = np.load(paths["heart_valid_samples.npy"], allow_pickle=False)
    positive = rows["valid"].numpy()
    if pool.shape != (227, 500, 2) or pool.dtype != np.dtype("<i8") or not pool.flags.c_contiguous:
        raise ValueError("Require complete released fixed500 per-positive pool")
    if np.any((pool < 0) | (pool >= len(x))) or np.any(pool[:, :250, 0] != positive[:, 0, None]) or np.any(pool[:, 250:, 1] != positive[:, 1, None]):
        raise ValueError("VALID pool endpoint/range/row association differs")
    return x, rows["train"], rows["valid"], torch.from_numpy(pool), identities


def load_native():
    sys.path.insert(0, str(ROOT / "vendor"))
    native = importlib.import_module("baseline_models.NCN.model")
    iterator = importlib.import_module("baseline_models.NCN.util")
    if Path(native.__file__).resolve() != ROOT / "vendor/baseline_models/NCN/model.py" or Path(iterator.__file__).resolve() != ROOT / "vendor/baseline_models/NCN/util.py":
        raise ValueError("Author NCN modules were shadowed")
    return native, iterator.PermIterator


def seed_native(seed):
    import numpy as np
    import torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metric(pos, neg):
    import torch
    rank = 1 + .5 * ((neg >= pos[:, None]).sum(1) + (neg > pos[:, None]).sum(1))
    return {"MRR": round((1 / rank.float()).mean().item(), 4),
            "mrr_hit10": round((rank <= 10).float().mean().item(), 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    started = time.monotonic()
    job = authorize(args.job, output)
    output.mkdir()
    write_json(output / "START.json", {"job_sha256": sha(args.job), "arm": job["arm"], "seed": job["seed"], "PID": os.getpid(), "TEST_access": False})
    try:
        import numpy as np
        import torch
        import torch_geometric
        import torch_sparse
        import torch_scatter
        from torch_geometric.utils import negative_sampling, to_undirected
        from torch_sparse import SparseTensor
        from heads import make_predictor
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Qualified single visible CUDA device unavailable")
        runtime = {"torch": str(torch.__version__), "numpy": np.__version__, "torch_geometric": torch_geometric.__version__,
                   "torch_sparse": torch_sparse.__version__, "torch_scatter": torch_scatter.__version__, "CUDA": torch.version.cuda}
        if runtime != job["runtime_versions"]:
            raise ValueError("Root-qualified actual runtime versions differ")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        device = torch.device("cuda:0")
        x, train, valid, pool, identities = load_available(job)
        native, PermIterator = load_native()
        seed_native(job["seed"])
        encoder = native.GCN(x.size(1), 256, 256, 1, .3, True, False, -1,
                    "puregcn", True, 0., xdropout=.4, taildropout=0., noinputlin=False).to(device)
        predictor = make_predictor(native, arm=job["arm"], members=job["member_count"],
                    axis_index=job.get("axis_index"), factor_seed=job.get("factor_seed")).to(device)
        x, train, valid, pool = (value.to(device) for value in (x, train, valid, pool))
        edge = to_undirected(train.t())
        adj = SparseTensor.from_edge_index(edge, sparse_sizes=(len(x), len(x))).to_symmetric().coalesce()
        optimizer = torch.optim.Adam([{"params": encoder.parameters(), "lr": .001},
                                      {"params": predictor.parameters(), "lr": .001}], weight_decay=0.)
        config = {"job": job, "input_identities": identities, "arm": job["arm"], "member_count": job["member_count"],
                  "native_author_commit": "c447cbff4c493b60d14b6544c3c39d3b9c5ddff0", "native_recipe": "citeseer.sh:cn1",
                  "encoder": "puregcn,input_projection,JK,one_layer", "features": list(x.shape), "hidden": 256,
                  "xdp": .4, "gnndp": .3, "predp": .3, "dropedge": 0., "twolayerlin": True, "maskinput": True,
                  "max_epochs": MAX_EPOCHS, "eval_steps": EVAL_STEPS, "kill_cnt": KILL_CNT, "partial_train_batch": "dropped",
                  "batches_per_epoch": len(train) // BATCH, "dropped_tail": len(train) % BATCH,
                  "native_loss": "mean negative log-sigmoid positive + mean negative log-sigmoid negative; bank mean over query/member",
                  "selection": "first maximum rounded4-decimal VALID MRR", "serving": "mean raw logits",
                  "runtime": runtime, "parameters": {"encoder": sum(p.numel() for p in encoder.parameters()), "predictor": sum(p.numel() for p in predictor.parameters())},
                  "RNG_protocol": "native global RNG sampler and PermIterator per fit; seed-block pairing, not identical draw replay across distinct architectures",
                  "TEST_access": False, "no_retry": True, "float32_profile": "TF32 off; native modern runtime adaptation"}
        write_json(output / "CONFIG.json", config)
        torch.cuda.reset_peak_memory_stats()

        @torch.no_grad()
        def validate():
            encoder.eval(); predictor.eval()
            h = encoder(x, adj)
            pos, neg = [], []
            for perm in PermIterator(device, len(valid), VALID_BATCH, False):
                positive = predictor(h, adj, valid[perm].t())
                targets = pool[perm].permute(2, 0, 1).reshape(2, -1)
                negative = predictor(h, adj, targets)
                if positive.shape != (len(perm), job["member_count"]) or negative.shape != (len(perm) * 500, job["member_count"]):
                    raise ValueError("Actual model member axis differs from declared portable/member count")
                pos.append(positive.mean(1).cpu())
                neg.append(negative.mean(1).reshape(len(perm), 500).cpu())
            pos, neg = torch.cat(pos), torch.cat(neg)
            if not bool(torch.isfinite(pos).all() and torch.isfinite(neg).all()):
                raise FloatingPointError("Nonfinite complete VALID logits")
            return metric(pos, neg), pos, neg

        def train_epoch():
            encoder.train(); predictor.train()
            negatives = negative_sampling(edge, len(x))
            keep = torch.ones(len(train), device=device, dtype=torch.bool)
            updates, losses = 0, []
            for perm in PermIterator(device, len(train), BATCH):
                optimizer.zero_grad()  # Preserve author zero_grad default.
                keep[perm] = False
                temporary = SparseTensor.from_edge_index(train[keep].t(), sparse_sizes=(len(x), len(x))).to_device(device, non_blocking=True)
                keep[perm] = True
                temporary = temporary.to_symmetric()
                h = encoder(x, temporary)
                pos = predictor(h, temporary, train[perm].t())
                neg = predictor(h, temporary, negatives[:, perm])
                if pos.shape != (BATCH, job["member_count"]) or neg.shape != pos.shape:
                    raise ValueError("Loss minibatch member count differs")
                loss = -torch.nn.functional.logsigmoid(pos).mean() - torch.nn.functional.logsigmoid(-neg).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                for name, parameter in list(encoder.named_parameters()) + list(predictor.named_parameters()):
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError("Nonfinite gradient: " + name)
                optimizer.step()
                losses.append(loss.item()); updates += 1
            if updates != len(train) // BATCH:
                raise ValueError("Incomplete native epoch")
            return float(np.average(losses)), updates

        best_valid, selected, selected_epoch, misses, total_updates = 0., None, None, 0, 0
        checkpoint = output / "selected_checkpoint.pt"
        logits_path = output / "selected_VALID_logits.pt"
        history_path = output / "VALID_HISTORY.jsonl"
        with history_path.open("x") as history:
            for epoch in range(1, MAX_EPOCHS + 1):
                loss, updates = train_epoch(); total_updates += updates
                write_json(output / "PROGRESS.json", {"epoch": epoch, "updates": total_updates, "phase": "epoch_complete", "elapsed_seconds": time.monotonic() - started})
                if epoch % EVAL_STEPS:
                    continue
                result, pos, neg = validate()
                score = result["MRR"]
                history.write(json.dumps({"epoch": epoch, "loss": loss, "VALID_MRR": score, "VALID_Hits10": result["mrr_hit10"]}) + "\n"); history.flush()
                if selected is None or score > selected:
                    selected, selected_epoch = score, epoch
                    torch.save({"encoder": encoder.state_dict(), "predictor": predictor.state_dict(), "optimizer": optimizer.state_dict(),
                                "selected_epoch": epoch, "selected_VALID_MRR": score, "config": config,
                                "torch_RNG": torch.get_rng_state(), "CUDA_RNG": torch.cuda.get_rng_state(),
                                "python_RNG": random.getstate(), "numpy_RNG": [np.random.get_state()[0], torch.from_numpy(np.random.get_state()[1].astype(np.int64)), *np.random.get_state()[2:]]}, checkpoint)
                    torch.save({"pos": pos, "neg": neg, "input_identities": identities, "selected_epoch": epoch,
                                "checkpoint_sha256": sha(checkpoint)}, logits_path)
                if score > best_valid:
                    best_valid, misses = score, 0
                else:
                    misses += 1
                    if misses > KILL_CNT:
                        break
        torch.cuda.synchronize()
        freeze = {"schema": "citeseer_ncn_TRAIN_VALID_freeze_v1", "arm": job["arm"], "seed": job["seed"], "member_count": job["member_count"],
                  "paired_seed_block": job["paired_seed_block"], "cohort_plan_sha256": job["cohort_plan_sha256"],
                  "selected_epoch": selected_epoch, "selected_VALID_MRR": selected, "last_epoch": epoch,
                  "updates": total_updates, "checkpoint_sha256": sha(checkpoint), "VALID_logits_sha256": sha(logits_path),
                  "config_sha256": sha(output / "CONFIG.json"), "history_sha256": sha(history_path),
                  "job_sha256": sha(args.job), "source_manifest_sha256": sha(ROOT / "SOURCE_MANIFEST.json"),
                  "input_identities": identities, "inclusive_seconds": time.monotonic() - started,
                  "peak_CUDA_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "TEST_access": False, "stop_rule": "native9999 or eleven consecutive VALID misses at5epoch cadence"}
        write_json(output / "FREEZE.json", freeze)
        print(json.dumps({"status": "complete_validation_selected_TEST_closed", "output": str(output), "arm": job["arm"], "seed": job["seed"], "updates": total_updates}))
    except (Exception, KeyboardInterrupt) as error:
        write_json(output / "FAILURE.json", {"error": type(error).__name__ + ": " + str(error), "inclusive_seconds": time.monotonic() - started,
                   "TEST_access": False, "retry": False, "partial_artifacts_preserved": True})
        raise


if __name__ == "__main__":
    main()
