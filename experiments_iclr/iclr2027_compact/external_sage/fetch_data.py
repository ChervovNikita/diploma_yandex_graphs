"""Fetch the public graph bytes pinned by an external SAGE source manifest."""
from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URLS = {
    "data/wiki_cs/data.json":
        "https://raw.githubusercontent.com/pmernyei/wiki-cs-dataset/master/dataset/data.json",
    "data/actor/out1_node_feature_label.txt":
        "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data/film/out1_node_feature_label.txt",
    "data/actor/out1_graph_edges.txt":
        "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data/film/out1_graph_edges.txt",
    "data/actor/film_split_0.6_0.2_0.npz":
        "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/splits/film_split_0.6_0.2_0.npz",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    study = ROOT.name
    datasets = ("wikics", "actor") if study == "external_sage" else ("roman",)
    for dataset in datasets:
        manifest = json.loads((ROOT / "results" / dataset / "source_manifest.json").read_text())
        for name, expected in manifest["source_sha256"].items():
            if not name.startswith("data/"):
                continue
            dest = ROOT / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_file() and sha(dest) == expected:
                print("already verified", name)
                continue
            temporary = dest.with_suffix(dest.suffix + ".download")
            if dataset == "roman":
                source = ROOT.parent / "data" / "roman_empire.npz"
                if not source.is_file():
                    raise FileNotFoundError("The code archive must contain data/roman_empire.npz")
                shutil.copyfile(source, temporary)
            else:
                with urllib.request.urlopen(URLS[name], timeout=60) as response:
                    with temporary.open("wb") as out:
                        shutil.copyfileobj(response, out)
            if sha(temporary) != expected:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"Public source bytes differ from frozen hash: {name}")
            temporary.replace(dest)
            print("verified", name)


if __name__ == "__main__":
    main()
