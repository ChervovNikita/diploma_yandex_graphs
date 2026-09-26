#!/usr/bin/env python3
"""Prepare an empty, isolated rerun of the frozen primary 432-cell study.

This script never trains or scores a model. It only copies byte-checked frozen
source and downloads byte-checked public graph files into a new target.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

SOURCE_FILES = (
    "tuning.py", "verify_tuning.py", "models.py", "STUDY_PROTOCOL.md",
    "PRETRAIN_REPAIR.md",
)
FROZEN_SHA256 = "6bbb0b7068141e7adb814413d9ce952bcacbc44ef013147f1e577bbff6fcda73"
DATA_URLS = {
    "data/wiki_cs/data.json":
        "https://raw.githubusercontent.com/pmernyei/wiki-cs-dataset/master/dataset/data.json",
    "data/actor/out1_node_feature_label.txt":
        "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data/film/out1_node_feature_label.txt",
    "data/actor/out1_graph_edges.txt":
        "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data/film/out1_graph_edges.txt",
    "data/actor/film_split_0.6_0.2_0.npz":
        "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/splits/film_split_0.6_0.2_0.npz",
    "data/chameleon_filtered.npz":
        "https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/a431395582e929d88271309716bea4fe24ce6318/data/chameleon_filtered.npz",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_source(source: Path) -> dict:
    require(sha256(source / "FROZEN_STUDY.json") == FROZEN_SHA256,
            "Original frozen study record changed")
    frozen = json.loads((source / "FROZEN_STUDY.json").read_text())
    require(frozen["protocol"] == "validation_tuning_sensitivity_v1", "Unexpected study protocol")
    require(set(frozen["source_sha256"]) == set(SOURCE_FILES), "Incomplete frozen source list")
    for name in SOURCE_FILES:
        require(sha256(source / name) == frozen["source_sha256"][name], f"Changed source: {name}")
    require(set(frozen["graphs"]) == {"cora", "wikics", "actor", "chameleon_filtered"},
            "Incomplete graph set")
    other = {name for dataset, record in frozen["graphs"].items()
             if dataset != "cora" for name in record["raw_sha256"]}
    require(other == set(DATA_URLS), f"Incomplete public download list: {other ^ set(DATA_URLS)}")
    cora = set(frozen["graphs"]["cora"]["raw_sha256"])
    require(len(cora) == 8 and all(name.startswith("data/cora/Cora/raw/ind.cora.") for name in cora),
            "Unexpected Cora raw file list")
    # The only local Python imports are covered by the copied source set.
    local_stems = {Path(name).stem for name in SOURCE_FILES if name.endswith(".py")}
    observed = set()
    all_import_roots = set()
    for name in ("tuning.py", "verify_tuning.py", "models.py"):
        tree = ast.parse((source / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
                all_import_roots.update(roots)
                observed.update(roots & local_stems)
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                all_import_roots.add(root)
                if root in local_stems:
                    observed.add(root)
    require(observed == {"models", "tuning"}, f"Unexpected local imports: {observed}")
    require(all_import_roots == {"__future__", "argparse", "copy", "csv", "hashlib", "json",
                                 "math", "models", "numpy", "pathlib", "random",
                                 "time", "torch", "torch_geometric", "tuning", "types"},
            f"Unexpected source import closure: {all_import_roots}")
    return frozen


def checked_download(url: str, dest: Path, expected: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(dest)
    temporary = dest.with_name(dest.name + ".download")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("xb") as out:
            shutil.copyfileobj(response, out)
        if sha256(temporary) != expected:
            raise RuntimeError(f"Public bytes differ from frozen SHA-256: {dest}")
        temporary.rename(dest)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path,
                        help="Original compact validation_tuning directory")
    parser.add_argument("--target", required=True, type=Path,
                        help="New directory for this rerun; must not exist")
    parser.add_argument("--check-only", action="store_true",
                        help="Check source closure and target availability without writing")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    target = args.target.absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing existing target: {target}")
    frozen = check_source(source)
    if args.check_only:
        print("PASS: exact frozen source, complete local-import/data manifest, absent target")
        return

    # Check that the user's environment can execute the graph loader before
    # creating the target. All dataset/cache paths passed below are within it.
    subprocess.run([sys.executable, "-c",
                    "import torch; from torch_geometric.datasets import Planetoid"], check=True)
    if not target.parent.is_dir():
        raise FileNotFoundError(f"Create the target parent first: {target.parent}")
    target.mkdir(exist_ok=False)
    for name in SOURCE_FILES + ("FROZEN_STUDY.json",):
        shutil.copyfile(source / name, target / name)
    for name, url in DATA_URLS.items():
        dataset = "wikics" if "wiki_cs" in name else (
            "actor" if "/actor/" in name else "chameleon_filtered")
        checked_download(url, target / name, frozen["graphs"][dataset]["raw_sha256"][name])
    subprocess.run([sys.executable, "-c",
                    "import sys; from torch_geometric.datasets import Planetoid; "
                    "Planetoid(root=sys.argv[1], name='Cora', split='public')",
                    str(target / "data/cora")], check=True)
    expected_cora = frozen["graphs"]["cora"]["raw_sha256"]
    actual_cora = {p.relative_to(target).as_posix(): sha256(p)
                   for p in (target / "data/cora/Cora/raw").rglob("*") if p.is_file()}
    if actual_cora != expected_cora:
        raise RuntimeError("Cora public raw file set/hash differs from frozen study")
    subprocess.run([sys.executable, str(target / "tuning.py"), "check-freeze"], check=True)
    print(f"PASS: isolated fresh primary study prepared in {target}")


if __name__ == "__main__":
    main()
