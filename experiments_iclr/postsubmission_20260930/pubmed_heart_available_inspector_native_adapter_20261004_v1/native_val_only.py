#!/usr/bin/env python3
"""Native Pubmed SAGE/NCNC, VALID only. Source-only until root admission."""
import argparse
import json
from pathlib import Path
import sys
import time

from inspect_available import PHASE, NODES, EXPECTED, load_available, server_guard, sha256

ROOT = Path(__file__).resolve().parent
AUTHOR = ROOT / "public_author_code/HeaRT/benchmarking"
MAX_EPOCHS, EVAL_STEPS, KILL_CNT = 9999, 5, 10


def checked_job(path, inspection):
    phase = server_guard()
    for file in (path, inspection):
        if not file.resolve(strict=True).is_relative_to(phase):
            raise ValueError("Admission/inspection receipt leaves phase")
    job = json.loads(path.read_text())
    receipt = json.loads(inspection.read_text())
    if job.get("source_review_approved") is not True or job.get("adapter_sha256") != sha256(Path(__file__)):
        raise ValueError("Root source review/admission is absent")
    if job.get("inspector_sha256") != sha256(ROOT / "inspect_available.py"):
        raise ValueError("Available-input inspector changed/unreviewed")
    if job.get("inspection_receipt_sha256") != sha256(inspection) or receipt.get("available_geometry_checks_pass") is not True:
        raise ValueError("Available-input qualification is absent")
    authority = job.get("feature_authority", {})
    if authority.get("verified") is not True or not authority.get("evidence") or not authority.get("origin"):
        raise ValueError("Feature donor/source authority remains unresolved")
    negative_authority = job.get("negative_pool_authority", {})
    if negative_authority.get("verified") is not True or not negative_authority.get("evidence") or not negative_authority.get("semantics"):
        raise ValueError("Released VALID negative-pool authority remains unresolved")
    if job.get("model") not in ("SAGE", "NCNC") or type(job.get("seed")) is not int or job.get("seed") not in range(10):
        raise ValueError("Specify one root-admitted native seed 0–9; no cohort default")
    if job["model"] == "SAGE" and job.get("sage_iterator_rng_qualified") is not True:
        raise ValueError("SAGE no-data iterator RNG bookkeeping requires qualification")
    if {name: value["sha256"] for name, value in receipt["files"].items()} != EXPECTED:
        raise ValueError("Inspection file identities differ")
    snapshot_path = ROOT / "PINNED_AUTHOR_SOURCE_AND_FUNCTIONS.json"
    if job.get("author_snapshot_sha256") != sha256(snapshot_path):
        raise ValueError("Pinned author snapshot manifest changed/unreviewed")
    snapshot = json.loads(snapshot_path.read_text())
    for item in snapshot["snapshot_files"]:
        if sha256(ROOT / "public_author_code/HeaRT" / item["path"]) != item["sha256"]:
            raise ValueError("Pinned author source changed")
    if job.get("native_train_functions_sha256") != sha256(ROOT / "native_train_functions.py"):
        raise ValueError("Extracted native training body changed/unreviewed")
    return job, receipt


def seed_native(seed):
    import random
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--inspection-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    phase = server_guard()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(phase) or not output.parent.is_dir():
        raise ValueError("Use a fresh output inside the existing phase")
    job, inspection = checked_job(args.job, args.inspection_receipt)
    x, train_rows, valid_rows, pool, identities = load_available()
    if list(x.shape) != inspection["feature"]["shape"] or str(x.dtype) != inspection["feature"]["dtype"]:
        raise ValueError("Feature qualification differs")
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torch_sparse import SparseTensor
    from torch_geometric.data import Data
    from torch_geometric.utils import to_undirected
    sys.path.insert(0, str(AUTHOR))
    from evalutors import evaluate_mrr
    from gnn_model import SAGE
    from scoring import mlp_score
    from baseline_models.NCN.model import GCN, IncompleteCN1Predictor
    from baseline_models.NCN.util import PermIterator
    from native_train_functions import sage_train, sage_test_edge, ncnc_train

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train = torch.tensor(train_rows, dtype=torch.long)
    valid = torch.tensor(valid_rows, dtype=torch.long)
    valid_neg = torch.from_numpy(np.array(pool, copy=True))
    model_name, seed = job["model"], job["seed"]
    config = {"model": model_name, "seed": seed, "native_seed_default_record": "ten runs0–9; this one job is root-admitted",
              "max_epochs": MAX_EPOCHS, "eval_steps": EVAL_STEPS, "kill_cnt": KILL_CNT,
              "batch_size": 1024, "validation_batch_size": 1024 if model_name == "SAGE" else 512,
              "hidden_channels": 256, "selection": "first maximum of rounded4-decimal VALID MRR",
              "input_files": identities, "author_commit": "c447cbff4c493b60d14b6544c3c39d3b9c5ddff0",
              "feature_authority": job["feature_authority"], "negative_pool_authority": job["negative_pool_authority"],
              "job_sha256": sha256(args.job),
              "inspection_sha256": sha256(args.inspection_receipt), "adapter_sha256": sha256(Path(__file__)),
              "inspector_sha256": sha256(ROOT / "inspect_available.py"),
              "author_snapshot_sha256": sha256(ROOT / "PINNED_AUTHOR_SOURCE_AND_FUNCTIONS.json"),
              "native_train_functions_sha256": sha256(ROOT / "native_train_functions.py"), "device": str(device)}

    if model_name == "SAGE":
        # Source single-run construction, then native per-run seed/reset order.
        seed_native(seed)
        edge = torch.cat((train.t(), train.t()[[1, 0]]), dim=1)
        adj = SparseTensor.from_edge_index(edge, torch.ones(edge.size(1)), [NODES, NODES])
        model = SAGE(x.size(1), 256, 256, 2, 0.1, 2, 1, NODES, False).to(device)
        predictor = mlp_score(256, 256, 1, 3, 0.1).to(device)
        seed_native(seed)
        model.reset_parameters()
        predictor.reset_parameters()
        x = x.to(device)
        train_device = train.to(device)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=0.001, weight_decay=0)
        data = {"adj": adj}
        config["native_settings"] = {"num_layers": 2, "predictor_layers": 3, "dropout": 0.1, "lr": 0.001,
                                     "l2": 0, "gradient_clip": 1, "negative_sampler": "unfiltered torch.randint", "partial_batch": "included",
                                     "evaluation_rng_cadence": "Adapter change: two no-data DataLoader iterators intended to preserve omitted diagnostic base-seed draws; requires root qualification",
                                     "evaluation_rng_qualified": job["sage_iterator_rng_qualified"]}
    else:
        edge = to_undirected(train.t())
        adj = SparseTensor.from_edge_index(edge, sparse_sizes=(NODES, NODES)).to_symmetric().coalesce()
        data = Data(x=x, edge_index=edge, adj_t=adj, num_nodes=NODES, max_x=-1).to(device)
        split = {"train": {"edge": train}}
        seed_native(seed)
        model = GCN(x.size(1), 256, 256, 1, 0.1, True, False, -1, "puregcn", True, 0.0,
                    xdropout=0.3, taildropout=0.0, noinputlin=False).to(device)
        predictor = IncompleteCN1Predictor(256, 256, 1, 1, 0.1, 0.0, True, cndeg=-1,
                    use_xlin=True, tailact=True, twolayerlin=False, beta=1.0, alpha=0.3, scale=5.3,
                    offset=0.5, trainresdeg=-1, testresdeg=-1, pt=0.5, learnablept=False,
                    depth=1, splitsize=-1).to(device)
        optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": 0.001},
                                      {"params": predictor.parameters(), "lr": 0.001}], weight_decay=0)
        config["native_settings"] = {"encoder": "puregcn", "predictor": "incn1cn1", "mplayers": 1, "nnlayers": 1,
                    "gnndp": 0.1, "predp": 0.1, "xdp": 0.3, "tdp": 0.0, "gnnlr": 0.001, "prelr": 0.001,
                    "l2": 0, "cndeg": -1, "trndeg": -1, "tstdeg": -1, "depth": 1, "splitsize": -1,
                    "scale": 5.3, "offset": 0.5, "alpha": 0.3, "pt": 0.5, "ln": True, "lnnn": True,
                    "jk": True, "maskinput": True, "use_xlin": True, "tailact": True,
                    "negative_sampler": "native PyG negative_sampling", "partial_batch": "dropped by native PermIterator"}

    @torch.no_grad()
    def validate():
        model.eval()
        predictor.eval()
        if model_name == "SAGE":
            h = model(x, data["adj"].to(device))
            # Native scoring creates three DataLoader iterators. Preserve the two
            # omitted diagnostic iterator base-seed draws without reading data.
            iter(DataLoader((), batch_size=1024))
            pos, neg = sage_test_edge(predictor, valid, h, 1024, negative_data=valid_neg)
            iter(DataLoader((), batch_size=1024))
            pos, neg = pos.flatten(), neg.squeeze(-1)
        else:
            h = model(data.x, data.adj_t)
            positive = valid.to(device)
            negatives = valid_neg.to(device)
            pos, neg = [], []
            for perm in PermIterator(device, positive.shape[0], 512, False):
                pos.append(predictor(h, data.adj_t, positive[perm].t()).squeeze().cpu())
                targets = torch.permute(negatives[perm], (2, 0, 1)).view(2, -1)
                neg.append(predictor(h, data.adj_t, targets).squeeze().cpu())
            pos = torch.cat(pos).flatten()
            neg = torch.cat(neg).view(-1, 500)
        return evaluate_mrr(None, pos, neg)

    output.mkdir()
    (output / "CONFIG.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    best_valid, selected_score, selected_epoch, kill = 0.0, None, None, 0
    checkpoint = output / "selected_checkpoint.pt"
    epoch = 0
    with (output / "VALID_HISTORY.jsonl").open("x") as history:
        for epoch in range(1, MAX_EPOCHS + 1):
            loss = sage_train(model, predictor, train_device, x, optimizer, 1024) if model_name == "SAGE" else ncnc_train(model, predictor, data, split, optimizer, 1024, True, [], None)
            if epoch % EVAL_STEPS:
                continue
            result = validate()
            score = result["MRR"]
            history.write(json.dumps({"epoch": epoch, "loss": float(loss), "VALID_MRR": score, "VALID_Hits10": result["mrr_hit10"]}, sort_keys=True) + "\n")
            history.flush()
            if selected_score is None or score > selected_score:
                selected_score, selected_epoch = score, epoch
                # Only one selector writes this checkpoint. No predictions are saved.
                torch.save({"encoder": model.state_dict(), "predictor": predictor.state_dict(),
                            "optimizer": optimizer.state_dict(), "selected_epoch": epoch,
                            "selected_VALID_MRR": score, "config": config}, checkpoint)
            if score > best_valid:
                best_valid, kill = score, 0
            else:
                kill += 1
                if kill > KILL_CNT:
                    break
    freeze = {"schema": "pubmed_native_VALID_only_freeze_v1", "model": model_name, "seed": seed,
              "selected_epoch": selected_epoch, "selected_VALID_MRR": selected_score, "last_training_epoch": epoch,
              "stop_rule": "source max9999 or kill_cnt>10 at five-epoch cadence",
              "checkpoint_sha256": sha256(checkpoint), "config_sha256": sha256(output / "CONFIG.json"),
              "VALID_history_sha256": sha256(output / "VALID_HISTORY.jsonl"), "inclusive_elapsed_seconds": time.monotonic() - started,
              "available_input_hashes": identities, "source_review_job_sha256": sha256(args.job),
              "final_evaluator_release": "Separate root-approved once-only evaluator after this freeze"}
    (output / "FREEZE.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print(json.dumps(freeze, sort_keys=True))


if __name__ == "__main__":
    main()
