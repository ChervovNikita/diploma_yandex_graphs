"""One-shot exact packet deployment and root engineering run, sole allocation."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
HELPER = PHASE / "learnability_responsibility_native_synthetic_execution_root_20261005_v1/run_once.py"
PACKETS = {
    "learnability_responsibility_sequential_autograd_vjp_preparation_20261006_v1":
        "cf58770b2fafbf69e74fa7658b4e73c7d7e7d9ab556b741bad209cf5c5ebdd26",
    "learnability_responsibility_sequential_numerical_preparation_20261006_v1":
        "3937c005de13db427417ca13d076d810c79cbb92594646d9281021d2eea39fd2",
}
WORKER = "learnability_responsibility_sequential_numerical_preparation_20261006_v1/qualify.py"
WORKER_SHA = "ecbfaa6067e1d32451e5761531e3cb87629a9ee47db37327c9405d7dfb818399"
ACCESSOR = "amazon_learnability_responsibility_engineering_accessor_20261005_v1/train_only_accessor.py"
ACCESSOR_SHA = "9360e69753b362eb66ac89f133f00addaf15ab0e8c56314c75d7dd5e96fecb85"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--mode", choices=("synthetic", "full"), required=True)
    parser.add_argument("--candidate-review", required=True)
    parser.add_argument("--candidate-review-sha", required=True)
    parser.add_argument("--qualifier-review", required=True)
    parser.add_argument("--qualifier-review-sha", required=True)
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED_ROOT_ONE_SHOT_TRANSPORT"}))
        return
    name = "learnability_responsibility_sequential_" + args.mode + "_execution_root_20261006_v1"
    out = PHASE / name
    assert not out.exists(), "Never overwrite or restart an execution identity"
    reviews = []
    for path, pin in ((args.candidate_review, args.candidate_review_sha),
                      (args.qualifier_review, args.qualifier_review_sha)):
        p = (PHASE / path).resolve()
        assert p.is_relative_to(PHASE) and p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == pin
        reviews.append({"path": path, "sha256": pin})
    files = []
    for packet, pin in PACKETS.items():
        prep = PHASE / packet
        assert hashlib.sha256((prep / "MANIFEST.json").read_bytes()).hexdigest() == pin
        for row in json.loads((prep / "MANIFEST.json").read_text())["files"]:
            data = (prep / row["path"]).read_bytes()
            assert len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"]
        for p in sorted(prep.iterdir()):
            assert p.is_file() and not p.is_symlink() and p.stat().st_size < 2_000_000
            data = p.read_bytes()
            files.append({"path": str(p.relative_to(PHASE)), "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(), "data": base64.b64encode(data).decode()})
    spec = importlib.util.spec_from_file_location("sequential_reviewed_transport", HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    remote = helper.REMOTE
    old = "worker=phase/'learnability_responsibility_native_numerical_worker_preparation_20261005_v1/qualify.py'"
    assert remote.count(old) == 1
    remote = remote.replace(old, "worker=phase/" + repr(WORKER))
    extras = {}
    if args.mode == "full":
        synthetic_name = "learnability_responsibility_sequential_synthetic_execution_root_20261006_v1"
        synthetic = PHASE / synthetic_name / "RESULT.json"
        prior = json.loads(synthetic.read_text())
        assert prior["status"] == "PASS_SYNTHETIC_SEQUENTIAL_PARITY_ONLY"
        assert prior["worker_sha256"] == WORKER_SHA
        extras["synthetic_sha256"] = hashlib.sha256(synthetic.read_bytes()).hexdigest()
        extras["accessor_sha256"] = ACCESSOR_SHA
        old_argv = "argv=[str(runtime),'-B',str(worker),'--execute-authorized','--mode','synthetic','--source-root',str(phase),'--output',str(out/'output')]"
        full_argv = """
synthetic=phase/'learnability_responsibility_sequential_synthetic_execution_root_20261006_v1/output/RESULT.json'
assert hashlib.sha256(synthetic.read_bytes()).hexdigest()==request['synthetic_sha256']
assert json.loads(synthetic.read_text())['status']=='PASS_SYNTHETIC_SEQUENTIAL_PARITY_ONLY'
accessor=phase/'amazon_learnability_responsibility_engineering_accessor_20261005_v1/train_only_accessor.py'
assert hashlib.sha256(accessor.read_bytes()).hexdigest()==request['accessor_sha256']
public_b=phase/'learnability_responsibility_native_full_execution_root_20261005_v1/roles/public_b'
assert public_b.is_dir() and public_b.resolve().is_relative_to(phase)
memory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).strip()
(out/'GPU_MEMORY_BEFORE.json').write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'record':memory})+'\\n')
argv=[str(runtime),'-B',str(worker),'--execute-authorized','--mode','full','--device','cuda:0','--source-root',str(phase),'--output',str(out/'output'),'--synthetic-pass',str(synthetic),'--synthetic-sha',request['synthetic_sha256'],'--accessor',str(accessor),'--accessor-sha',request['accessor_sha256'],'--public-b-dir',str(public_b)]
"""
        assert remote.count(old_argv) == 1
        remote = remote.replace(old_argv, full_argv)
    release = {"UTC": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
        "purpose": "Exact sequential native engineering qualification",
        "worker_sha256": WORKER_SHA, "packet_manifests": PACKETS,
        "independent_source_reviews_root_read": reviews,
        "remote_wrapper_sha256": hashlib.sha256(remote.encode()).hexdigest(),
        "model_fits": 0, "persistent_updates": 0, "A_scoring": False,
        "original_failures_preserved": True, "predictive_claim": False,
        "full_reuses_exact_prior_safe_public_B_roles": args.mode == "full"}
    out.mkdir()
    (out / "EXECUTION_RELEASE.json").write_text(json.dumps(release, indent=2) + "\n")
    request = {"files": files, "packet_manifests": PACKETS, "worker_sha256": WORKER_SHA,
        "run_name": name, "release": release, **extras}
    command = ["ssh", "-T", "-p", "2222", "-i", "/Users/alex/.ssh/mlspace__private_key_anogena.txt",
        "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "UpdateHostKeys=no", "-o", "ConnectTimeout=20", helper.LOGIN,
        "cd " + shlex.quote(helper.REPO) + " && exec /usr/bin/python3 -I -S -B -c " + shlex.quote(remote)]
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child.stdin.write(json.dumps(request))
    child.stdin.close()
    transcript = []
    for line in child.stdout:
        transcript.append(line)
        item = json.loads(line)
        if item["stage"] == "launched":
            (out / "LAUNCH_RECEIPT.json").write_text(json.dumps(item["owned_process"], indent=2) + "\n")
            print(json.dumps({"stage": "launched", "PID": item["owned_process"]["PID"]}), flush=True)
        else:
            (out / "TERMINAL.json").write_text(json.dumps(item["terminal"], indent=2) + "\n")
            if item["result"] is not None:
                (out / "RESULT.json").write_text(json.dumps(item["result"], indent=2) + "\n")
            print(json.dumps({"stage": "terminal", "exit_code": item["terminal"]["exit_code"],
                "status": item["result"].get("status") if item["result"] else None,
                "error": item["result"].get("error") if item["result"] else None}), flush=True)
    stderr = child.stderr.read()
    exit_code = child.wait()
    (out / "TRANSPORT_RECEIPT.json").write_text(json.dumps({"UTC": datetime.now(timezone.utc).isoformat(),
        "exit_code": exit_code, "stdout_bytes": len("".join(transcript).encode()),
        "stdout_sha256": hashlib.sha256("".join(transcript).encode()).hexdigest(), "stderr": stderr}, indent=2) + "\n")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
