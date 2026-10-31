"""Stdlib-only sealed-source check. Does not import PyTorch or the prototype."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    output=Path(args.output)
    if output.exists():raise RuntimeError("Refuse to replace source check")
    pins=json.loads((here/"PRIVATE_SOURCE_PINS.json").read_text())
    verified=[]
    for entry in pins["files"]:
        path=here/entry["private_path"]
        if sha(path)!=entry["sha256"]:raise RuntimeError("Private source binding failed")
        if path.suffix==".py":ast.parse(path.read_text())
        verified.append(entry["private_path"])
    compiled=[]
    compiled_hashes={}
    for path in sorted(here.glob("*.py")):
        compile(path.read_text(),str(path),"exec")
        compiled.append(path.name)
        compiled_hashes[path.name]=sha(path)
    for path in sorted(here.glob("*.json")):
        json.loads(path.read_text())
    manifest_path=here/"MANIFEST.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        for entry in manifest["files"]:
            path=here/entry["path"]
            if sha(path)!=entry["sha256"] or path.stat().st_size!=entry["bytes"]:
                raise RuntimeError("Sealed payload changed: "+entry["path"])
    payload={"schema":"ncnc-completion-static-source-check-v1","status":"STATIC_SOURCE_CHECK_PASSED",
             "compiled_without_execution":compiled,"private_hashes_verified":verified,
             "compiled_source_hashes":compiled_hashes,
             "torch_or_model_imported":False,"numerical_qualification_passed":False,
             "native_runtime_parity_established":False,"fit_or_launch_authorized":False}
    output.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"status":payload["status"],"compiled":len(compiled),"private_sources":len(verified)}))


if __name__=="__main__":main()
