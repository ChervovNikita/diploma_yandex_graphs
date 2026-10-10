"""Wait for the existing three families; run their sealed CPU reader once."""
from pathlib import Path
import hashlib, json, os, socket, subprocess, time
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PILOT = ROOT / "nonlocal_label_retrieval_pilot_root_20261010_v1"
READER_SHA = "d8a084b083f18542a103b69c6dc2c2570e00401ef642a315ea9386ba9244cdad"
REPO = ROOT.parent.parent
assert socket.gethostname() == "anogena-2-0"
assert subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() == ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]
assert ROOT.is_relative_to(REPO)
REPORT = PILOT / "FULL_RETRIEVAL_ANALYSIS_V1.json"
END = HERE / "DEFERRED_END.json"
assert not REPORT.exists() and not END.exists()
assert hashlib.sha256((HERE / "analysis.py").read_bytes()).hexdigest() == READER_SHA
now = lambda: datetime.now(timezone.utc).isoformat()
start = time.monotonic()
stat = Path("/proc", str(os.getpid()), "stat").read_text().rsplit(")", 1)[1].split()
with (HERE / "DEFERRED_START.json").open("x") as f:
    json.dump({"UTC": now(), "PID": os.getpid(), "start_ticks": int(stat[19]), "reader_sha256": READER_SHA, "original_fits_changed": False, "CPU_only": True}, f, indent=2)
result = {"started": False, "success": False, "reader_sha256": READER_SHA, "TEST_access": False, "new_model_forwards": 0}
try:
    while True:
        closures = {}
        for b in ("SAGE", "GCN", "GAT"):
            q = PILOT / b / "OWNER_END.json"
            if q.exists():
                v = json.loads(q.read_text())
                if not v.get("scientific_success"):
                    raise RuntimeError("Existing family ended unsuccessfully: " + b)
                closures[b] = hashlib.sha256(q.read_bytes()).hexdigest()
        if len(closures) == 3:
            break
        if time.monotonic() - start > 7200:
            raise TimeoutError("Existing family closures not available within two hours; no reader executed")
        time.sleep(15)
    result.update(started=True, reader_start_UTC=now(), owner_end_sha256=closures)
    assert not REPORT.exists()
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    with (HERE / "reader.stdout.log").open("x") as out, (HERE / "reader.stderr.log").open("x") as err:
        r = subprocess.run([str(REPO / ".venv/bin/python"), "-B", str(HERE / "analysis.py"), "--research-root", str(ROOT), "--report", str(REPORT)], cwd=REPO, env=env, stdout=out, stderr=err, timeout=600)
    result.update(exit_code=r.returncode, success=r.returncode == 0, report_exists=REPORT.exists())
except Exception as error:
    result.update(failure_type=type(error).__name__, failure=str(error))
finally:
    result.update(UTC=now(), elapsed_seconds=time.monotonic() - start)
    with END.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)
if not result["success"]:
    raise SystemExit(1)
