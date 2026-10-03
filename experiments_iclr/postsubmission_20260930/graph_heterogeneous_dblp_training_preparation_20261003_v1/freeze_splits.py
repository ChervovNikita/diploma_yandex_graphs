"""Explicit future development-only preparation; writes ID splits, not label values."""
import argparse
import hashlib
import json
from pathlib import Path
from dblp_inputs import read_development_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels-descriptor',required=True,help='JSON path/hash/bytes descriptor for explicit TRAIN+VAL input')
    parser.add_argument('--output',required=True,help='Fresh root-owned split directory')
    args = parser.parse_args()
    descriptor = json.loads(Path(args.labels_descriptor).read_text())
    # Archive is not opened; this canonical digest is bound by the release receipt.
    archive_sha = '0d3ea4a74399f9cd3e83af206e8e0b67e1844fe2c8463b424189884dd58ad7c8'
    development = read_development_labels(descriptor,archive_sha)
    import numpy as np
    root = Path(args.output); root.mkdir(parents=True,exist_ok=False)
    rows = []
    for seed in (131,137,139,149,151):
        ids = np.array(development['node_ids'],dtype=np.int64)
        np.random.RandomState(seed).shuffle(ids); cut = int(.2*len(ids))
        value = dict(seed=seed,algorithm='NumPy RandomState(seed) shuffle sorted development IDs; first floor(0.2N) validation',
                     development_descriptor=descriptor,train_ids=sorted(ids[cut:].tolist()),validation_ids=sorted(ids[:cut].tolist()))
        path = root/f'seed{seed}.json'; data = (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
        path.write_bytes(data)
        rows.append(dict(seed=seed,descriptor=dict(path=str(path.resolve()),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))))
    (root/'SPLIT_BINDINGS.json').write_text(json.dumps(dict(splits=rows,label_values_saved=False,test_opened=False),indent=2)+'\n')
    print(json.dumps(dict(output=str(root),splits=rows,test_opened=False,training_launched=False)))


if __name__=='__main__':
    main()
