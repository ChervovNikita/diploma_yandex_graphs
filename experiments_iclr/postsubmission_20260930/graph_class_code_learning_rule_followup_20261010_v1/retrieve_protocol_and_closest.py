"""Second bounded primary/protocol retrieval; no dataset or result reads."""
from retrieve import retrieve, ROOT
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
import json

COMMIT = "1e72912a0810cdf27ae54fd589a3b43358a2b161"
requests = [
    ("gnn_benchmark_train_conf.yaml", f"https://raw.githubusercontent.com/shchur/gnn-benchmark/{COMMIT}/config/train.conf.yaml"),
    ("gnn_benchmark_make_dataset.py", f"https://raw.githubusercontent.com/shchur/gnn-benchmark/{COMMIT}/gnnbench/data/make_dataset.py"),
    ("shchur_pitfalls_ar5iv.html", "https://ar5iv.labs.arxiv.org/html/1811.05868"),
    ("learn_hierarchy_ar5iv.html", "https://ar5iv.labs.arxiv.org/html/2005.08622"),
    ("spectral_ecoc_hkust.html", "https://repository.hkust.edu.hk/ir/Record/1783.1-159940"),
]
for name, query in [
    ("multiple_class_hierarchies_discovery.json", "ensemble multiple class hierarchies"),
    ("label_tree_ensemble_discovery.json", "ensembles of label trees multiclass"),
]:
    requests.append((name, "https://api.openalex.org/works?search=" + quote(query) + "&per-page=5"))
with ThreadPoolExecutor(max_workers=4) as pool:
    receipts=list(pool.map(retrieve,requests))
(ROOT / "PROTOCOL_AND_CLOSEST_RETRIEVALS.json").write_text(json.dumps(receipts,indent=2)+'\n')
for r in receipts:
    print(json.dumps(r))
