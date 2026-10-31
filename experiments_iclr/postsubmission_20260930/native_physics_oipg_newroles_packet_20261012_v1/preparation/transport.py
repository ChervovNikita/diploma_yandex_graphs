"""Owned literal allocation-only source/data preparation transport."""
import importlib.util
import json
from pathlib import Path
OUT = Path(__file__).resolve().parents[1]
PHASE = OUT.parent
spec = importlib.util.spec_from_file_location("guarded_molecular_transport", PHASE / "training_pipeline_v2_20261012/preparation/prepare_molecular_packets_allocation.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
GUARD = helper.GUARD

def request(code, arguments, label, timeout=120):
    target = OUT / "transport" / (label + ".json")
    if target.exists():
        raise FileExistsError("Preserve prior receipt; choose a fresh label")
    value = helper.invoke(GUARD + code, arguments, target.parent, label, timeout)
    if value["exit_code"] != 0:
        raise RuntimeError("Guarded preparation failed; complete receipt preserved: " + str(target))
    return json.loads(value["stdout"].strip())
