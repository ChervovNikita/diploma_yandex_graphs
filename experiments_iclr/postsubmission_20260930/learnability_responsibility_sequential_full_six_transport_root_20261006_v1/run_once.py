"""Deploy sealed all-six engineering source once on the authorized allocation."""
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
PREPARATION = "learnability_responsibility_sequential_full_six_preparation_20261006_v1"
EXECUTION = "learnability_responsibility_sequential_full_six_execution_root_20261006_v1"
HELPER = PHASE / "learnability_responsibility_native_synthetic_execution_root_20261005_v1/run_once.py"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def exact(relative, pin):
    path = PHASE / relative
    assert path.resolve().is_relative_to(PHASE) and not path.is_symlink()
    data = path.read_bytes()
    assert len(data) < 2_000_000 and digest(data) == pin
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--scope-sha", required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--worker-sha", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--review-sha", required=True)
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED_ROOT_ENGINEERING_TRANSPORT"}))
        return
    scope_path = str(HERE.relative_to(PHASE)) + "/SCOPE_DECISION.json"
    scope = json.loads(exact(scope_path, args.scope_sha))
    assert scope["controls"] == ["live", "uniform", "margins", "graph_free", "permuted", "stop_q"]
    assert scope["fixed_before_execution"] and scope["full_context_FP32"]
    assert scope["worker_deadline_seconds"] == 900 and scope["external_deadline_seconds"] == 950
    assert scope["worker_sha256"] == args.worker_sha and scope["manifest_sha256"] == args.manifest_sha
    exact(args.review, args.review_sha)
    manifest = json.loads(exact(PREPARATION + "/MANIFEST.json", args.manifest_sha))
    for row in manifest["files"]:
        data = exact(PREPARATION + "/" + row["path"], row["sha256"])
        assert len(data) == row["bytes"]
    exact(PREPARATION + "/qualify.py", args.worker_sha)
    out = PHASE / EXECUTION
    assert not out.exists(), "Never restart or overwrite an execution identity"
    payload = []
    for path in sorted((PHASE / PREPARATION).iterdir()):
        assert path.is_file() and not path.is_symlink() and path.stat().st_size < 2_000_000
        data = path.read_bytes()
        payload.append({"path": str(path.relative_to(PHASE)), "sha256": digest(data),
                        "bytes": len(data), "data": base64.b64encode(data).decode()})
    for relative, pin in scope["prerequisite_result_pins"].items():
        data = exact(relative, pin)
        payload.append({"path": relative, "sha256": pin, "bytes": len(data),
                        "data": base64.b64encode(data).decode()})
    spec = importlib.util.spec_from_file_location("full_six_root_transport_helper", HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    remote = helper.REMOTE
    old_worker = "worker=phase/'learnability_responsibility_native_numerical_worker_preparation_20261005_v1/qualify.py'"
    assert remote.count(old_worker) == 1
    remote = remote.replace(old_worker, "worker=phase/" + repr(PREPARATION + "/qualify.py"))
    old_argv = "argv=[str(runtime),'-B',str(worker),'--execute-authorized','--mode','synthetic','--source-root',str(phase),'--output',str(out/'output')]"
    new_argv = """
public_b=phase/'learnability_responsibility_native_full_execution_root_20261005_v1/roles/public_b'
assert hashlib.sha256((public_b/'PUBLIC_B_MANIFEST.json').read_bytes()).hexdigest()==request['public_b_sha']
record=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).strip()
(out/'GPU_MEMORY_BEFORE.json').write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'record':record})+'\\n')
assert int(record.split(',')[-1].strip())>=42000,'Insufficient free memory for sealed engineering check'
argv=[str(runtime),'-B',str(worker),'--execute-authorized','--device','cuda:0','--source-root',str(phase),'--repository',str(repo),'--output',str(out/'output'),'--public-b-dir',str(public_b),'--public-b-sha',request['public_b_sha']]
"""
    assert remote.count(old_argv) == 1
    remote = remote.replace(old_argv, new_argv)
    assert remote.count("exit_code=child.wait()") == 1
    remote = remote.replace("exit_code=child.wait()", """
 watchdog_fired=False
 try:exit_code=child.wait(timeout=950)
 except subprocess.TimeoutExpired:
  watchdog_fired=True
  child.terminate()
  try:exit_code=child.wait(timeout=10)
  except subprocess.TimeoutExpired:child.kill();exit_code=child.wait()
 (out/'WATCHDOG_RECEIPT.json').write_text(json.dumps({'fixed_deadline_seconds':950,'fired':watchdog_fired,'owned_PID':child.pid})+'\\n')
""")
    compile(remote, "full_six_remote_wrapper", "exec")
    release = {"UTC": datetime.now(timezone.utc).isoformat(), "purpose": "All-six full-FP32 engineering only",
        "worker_sha256": args.worker_sha, "packet_manifests": {PREPARATION: args.manifest_sha},
        "scope_decision_sha256": args.scope_sha, "independent_source_review": {"path": args.review, "sha256": args.review_sha},
        "remote_wrapper_sha256": digest(remote.encode()), "model_fits": 0, "persistent_updates": 0,
        "A_scoring": False, "VALID_TEST_access": False, "original_failures_preserved": True,
        "transport_source_sha256": digest(Path(__file__).read_bytes()), "predictive_claim": False}
    out.mkdir()
    (out / "EXECUTION_RELEASE.json").write_text(json.dumps(release, indent=2) + "\n")
    request = {"files": payload, "packet_manifests": {PREPARATION: args.manifest_sha},
        "worker_sha256": args.worker_sha, "run_name": EXECUTION, "release": release,
        "public_b_sha": scope["public_B_manifest_sha256"]}
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
                data = json.dumps(item["result"], indent=2) + "\n"
                (out / "RESULT.json").write_text(data)
                artifact = next(row for row in item["terminal"]["artifacts"] if row["path"] == "output/RESULT.json")
                assert len(data.encode()) == artifact["bytes"] and digest(data.encode()) == artifact["sha256"]
            print(json.dumps({"stage": "terminal", "exit_code": item["terminal"]["exit_code"],
                "status": item["result"].get("status") if item["result"] else None,
                "error": item["result"].get("error") if item["result"] else None}), flush=True)
    stderr = child.stderr.read()
    exit_code = child.wait()
    (out / "TRANSPORT_RECEIPT.json").write_text(json.dumps({"UTC": datetime.now(timezone.utc).isoformat(),
        "exit_code": exit_code, "stdout_sha256": digest("".join(transcript).encode()), "stderr": stderr}, indent=2) + "\n")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
