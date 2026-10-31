"""Read only current status hashes and public Squirrel mask counts."""
from inspect_git_for_publication_v2 import REPO, LOGIN, UUID, run
import hashlib
import json
import os
from pathlib import Path

if Path.cwd().resolve() != REPO or os.environ.get('GNNM_SSH_DESTINATION') != LOGIN:
    raise ValueError('Wrong repository or route')
if run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']).splitlines() != [UUID]:
    raise ValueError('Wrong allocation')
phase = REPO/'experiments_iclr/postsubmission_20260930'
statuses = {}
for name in ('PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json'):
    path = phase/name
    statuses[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
graph = json.loads((phase/'coordinate_conformal_execution_root_v1/acquisition_run01/Squirrel/GRAPH_INPUT.json').read_text())
record = graph['published_masks']
path = Path(record['path']).resolve()
if not path.is_relative_to(phase):
    raise ValueError('Mask path outside phase')
if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
    raise ValueError('Mask bytes changed')
import numpy as np
masks = np.load(path, allow_pickle=False)
counts = [{name:int(masks[name][i].sum()) for name in masks.files} for i in range(3)]
print(json.dumps(dict(status_sha256=statuses, graph_num_nodes=graph['num_nodes'],
                     mask_counts=counts, mask_sha256=record['sha256'], labels_read=False), indent=2))
