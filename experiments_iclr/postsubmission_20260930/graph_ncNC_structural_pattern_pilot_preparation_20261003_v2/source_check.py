#!/usr/bin/env python3
"""Read, parse and compile source text only. Never import project/runtime code."""
import ast
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda:handle.read(1024*1024),b""):h.update(part)
    return h.hexdigest()


def require(value,message):
    if not value:raise RuntimeError(message)


def bindings(tree):
    names=set()
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):names.add(node.name)
        elif isinstance(node,(ast.Import,ast.ImportFrom)):
            for alias in node.names:names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node,(ast.Assign,ast.AnnAssign)):
            for target in node.targets if isinstance(node,ast.Assign) else (node.target,):
                names.update(child.id for child in ast.walk(target) if isinstance(child,ast.Name))
    return names


def main():
    dependency=json.loads((HERE/"DEPENDENCIES.json").read_text())
    external=[]
    for pin in dependency["sealed_packets"]:
        root=HERE.parent/pin["packet"]
        manifest=root/"MANIFEST.json"
        require(digest(manifest)==pin["manifest_sha256"],"External manifest differs: "+pin["packet"])
        payload=json.loads(manifest.read_text())
        for row in payload["files"]:
            path=(root/row["path"]).resolve()
            require(path.is_relative_to(root.resolve()),"External manifest path escaped")
            require(path.stat().st_size==row.get("bytes",row.get("size")) and digest(path)==row["sha256"],"External payload differs: "+row["path"])
        external.append({"packet":pin["packet"],"manifest_sha256":pin["manifest_sha256"],"all_payloads_unchanged":True})
    for pin in dependency["authority_files"]:
        require(digest(HERE.parent/pin["path"])==pin["sha256"],"Authority differs")
    copied=json.loads((HERE/"COPIED_SOURCE_BINDINGS.json").read_text())["files"]
    for row in copied:
        original=HERE.parent/row["source"]
        require(digest(original)==row["sha256"],"Copied helper original differs")
        text=(HERE/row["destination"]).read_text()
        expected=original.read_text()
        if row["adapted"]:
            expected=expected.replace('"eventual_test_graph": "native_TRAIN_plus_VALID",','"test_stage_supported": False,')
        require(text==expected,"Copied helper differs beyond recorded adaptation")
    trees={};python=[]
    for path in sorted(HERE.glob("*.py")):
        tree=ast.parse(path.read_text(),filename=str(path));compile(tree,str(path),"exec")
        trees[path.stem]=tree
        python.append({"path":path.name,"bytes":path.stat().st_size,"sha256":digest(path),"AST_and_compile_only":True})
    prototype=HERE.parent/dependency["qualified_prototype_packet"]
    for name in ("prototype","graph_ops"):
        trees[name]=ast.parse((prototype/(name+".py")).read_text())
    local_imports=0
    for name,tree in trees.items():
        if name in ("prototype","graph_ops"):continue
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module in trees:
                exports=bindings(trees[node.module])
                require(all(alias.name in exports for alias in node.names),"Local imported name missing: "+name+" from "+node.module)
                local_imports+=1
    for name in ("pilot_common","pilot_accounting","pattern_run","pattern_close"):
        tree=trees[name]
        for node in tree.body:
            if isinstance(node,ast.Import):
                require(all(a.name.split(".")[0] not in {"torch","numpy","pandas","prototype","graph_ops"} for a in node.names),"Numerical top-level entry import")
            elif isinstance(node,ast.ImportFrom):
                require(node.module not in {"torch","numpy","pandas","prototype","graph_ops"},"Numerical top-level entry import")
    model=trees["pattern_model"]
    decoder=next(node for node in model.body if isinstance(node,ast.ClassDef) and node.name=="PatternDecoder")
    require(not any(isinstance(node,ast.FunctionDef) and node.name=="forward" for node in decoder.body),"Native target forward was overridden")
    json_files=[]
    for path in sorted(HERE.glob("*.json")):
        if path.name in ("MANIFEST.json","SEAL.json","STATIC_SOURCE_CHECK.json"):continue
        json.loads(path.read_text());json_files.append(path.name)
    result={"schema":"ncnc-pattern-source-only-check-v1","status":"PASS_SOURCE_ONLY",
            "UTC":datetime.now(timezone.utc).isoformat(),"python_files":python,
            "JSON_files_parsed":json_files,"local_from_import_bindings_checked":local_imports,
            "external_sealed_payloads":external,"copied_helpers_match_recorded_adaptation":True,
            "native_target_forward_inherited":True,"numerical_or_project_imports_executed":False,
            "numerical_tests_executed":False,"full_graph_resource_qualified":False,
            "SSH_or_compute_launches":0,"canonical_or_running_family_edits":0}
    (HERE/"STATIC_SOURCE_CHECK.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"python_files":len(python),"local_import_bindings":local_imports,"external_packets":len(external)}))


if __name__=="__main__":main()
