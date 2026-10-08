"""Acquire pinned public author source inside the authorized allocation repo.

No numerical imports, data acquisition, stock evaluator or code execution.
The source is a separately acquired dependency, not vendored into this Git tree.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import urllib.request

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
COMMIT = 'b910314a59270984f5e249462ee3faa815fc9a0c'
TARGET = PHASE / 'continuous_method_gap_search_v1/round15_graph_route_initialization/primary/author_source'
FILES = (
    ('outcome_correlation.py', 'correct_smooth_outcome_correlation.py',
     'b34528d8a571beb16385035c99c7957e351e0a6ed201ba463d03f5415eefa22e'),
    ('README.md', 'correct_smooth_README.md',
     'ae5228d590f65baa528a75061a731652f9bb49ad7fb3bce3206b1112ab1f7c51'),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acquire-source', action='store_true')
    args = parser.parse_args()
    if not args.acquire_source:
        raise PermissionError('Explicit source acquisition required')
    assert Path.cwd() == REPO and socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True
    ).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    result = []
    for remote_name, local_name, expected in FILES:
        path = TARGET / local_name
        if path.exists():
            data = path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == expected
            acquired = False
        else:
            url = 'https://raw.githubusercontent.com/Chillee/CorrectAndSmooth/' + COMMIT + '/' + remote_name
            request = urllib.request.Request(url, headers={'User-Agent': 'GNNM-source-reproduction'})
            with urllib.request.urlopen(request, timeout=20) as response:
                data = response.read()
            assert hashlib.sha256(data).hexdigest() == expected
            TARGET.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as handle:
                handle.write(data)
            acquired = True
        result.append(dict(path=str(path.relative_to(PHASE)), sha256=expected,
                           bytes=len(data), newly_acquired=acquired))
    print(json.dumps(dict(files=result, code_executed=False, numerical_work=False)))


if __name__ == '__main__':
    main()
