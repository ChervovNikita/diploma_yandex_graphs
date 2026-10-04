"""Small CPU-only fabricated runner. Fresh project output, no data/native model."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def verify_packet(packet):
    seal=json.loads((packet/"SEAL.json").read_text())
    raw=(packet/"MANIFEST.json").read_bytes()
    if sha256(raw).hexdigest()!=seal["manifest_sha256"]:
        raise RuntimeError("Source packet manifest differs")
    for pin in json.loads(raw)["files"]:
        content=(packet/pin["path"]).read_bytes()
        if len(content)!=pin["bytes"] or sha256(content).hexdigest()!=pin["sha256"]:
            raise RuntimeError("Source packet payload differs: "+pin["path"])
    return sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=("legacy", "supplement", "vector", "all"), default="all")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    project = here.parent.parent
    output = Path(args.output).resolve()
    if not output.is_relative_to(project) or output.exists():
        raise RuntimeError("Fresh output inside project required")
    output.mkdir(parents=True)
    bindings = json.loads((here/"INPUT_BINDINGS.json").read_text())
    receipt = {"stage":args.stage,"command":[sys.executable,*sys.argv],"UTC":datetime.now(timezone.utc).isoformat(),
               "interpreter":sys.executable,"python":sys.version,"platform":platform.platform(),
               "device":"cpu","CUDA_VISIBLE_DEVICES":"","scientific_data_or_fit":False,"results":{}}
    try:
        receipt["supplement_manifest_sha256"]=verify_packet(here)
        if args.stage in ("vector","all"):
            receipt["vector_manifest_sha256"]=verify_packet(here.parent/"count_density_vectorized_successor_preparation_20261004_v1")
        for pin in bindings["source_only_inputs"]:
            data=(here.parent/pin["path"]).read_bytes()
            if len(data)!=pin["bytes"] or sha256(data).hexdigest()!=pin["sha256"]:
                raise RuntimeError("Input source pin changed: "+pin["path"])
        import torch
        torch.set_num_threads(2)
        receipt["torch"] = torch.__version__
        receipt["torch_num_threads"]=torch.get_num_threads()
        if args.stage in ("legacy","all"):
            sys.path.insert(0,str(here.parent/"pooled_joint_quality_source_preparation_20261004_v1"))
            receipt["results"]["legacy"] = importlib.import_module("fabricated_qualification").density_qualification()
        if args.stage in ("supplement","all"):
            receipt["results"]["supplement"] = importlib.import_module("supplement_qa").run()
        if args.stage in ("vector","all"):
            successor=here.parent/"count_density_vectorized_successor_preparation_20261004_v1"
            sys.path.insert(0,str(successor))
            receipt["results"]["vector"] = importlib.import_module("parity_qa").run()
        receipt["status"]="PASS_FABRICATED_CPU_ONLY"
    except BaseException as error:
        receipt.update(status="FAILED",error=repr(error))
        (output/"TRACEBACK.txt").write_text(traceback.format_exc())
    (output/"RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))
    return 0 if receipt["status"]=="PASS_FABRICATED_CPU_ONLY" else 1


if __name__=="__main__":
    raise SystemExit(main())
