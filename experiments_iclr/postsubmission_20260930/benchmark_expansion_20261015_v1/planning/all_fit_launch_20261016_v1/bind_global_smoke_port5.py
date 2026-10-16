"""Bind the coordinator to actual execution6 launch receipts; no remote calls."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
E = HERE.parents[1]


def pin(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    sources = {
        "port": pin(E / "server51_campaign_port_v5/MANIFEST.json"),
        "smoke": pin(E / "qualification/all_fit_smoke_20261016_v5/MANIFEST.json"),
        "execution": pin(E / "campaign_execution_20261016_v6/MANIFEST.json"),
    }
    workers = []
    for host in ("allocation", "server51"):
        receipt = json.loads((HERE / ("TMUX_SMOKE6_LAUNCH_" + host + ".json")).read_text())
        assert receipt["host"] == host
        for row in receipt["workers"]:
            launch = row["launch"]
            assert launch["plan"] == row["prepare"]["plan"]
            assert launch["worker"] == row["worker"]
            assert all(launch["source_manifest"][k] == sources["execution"][k]
                       for k in ("bytes", "sha256"))
            assert "/campaign_execution_20261016_v6/PLAN-" in launch["plan"]["path"]
            workers.append({"worker": row["worker"], "plan": launch["plan"]})
    assert [row["worker"] for row in workers] == [
        "allocation/gpu0", "server51/gpu0", "server51/gpu1"]
    value = {"schema": "all-fit-global-smoke-coordinator-bindings-v1",
             "sources": sources, "workers": workers}
    path = HERE / "GLOBAL_SMOKE_BINDINGS_PORT5.json"
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        assert path.read_bytes() == raw, "Conflicting frozen coordinator binding"
    else:
        with path.open("xb") as stream:
            stream.write(raw)
    print(json.dumps({"path": str(path), **pin(path)}))


if __name__ == "__main__":
    main()
