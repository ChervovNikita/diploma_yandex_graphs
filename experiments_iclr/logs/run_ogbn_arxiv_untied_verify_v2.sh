#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
[[ ! -e experiments_iclr/logs/ogbn_arxiv_untied_v2.complete ]]
[[ $(sha256sum experiments_iclr/ogbn_arxiv_untied_source_lock.json | cut -d ' ' -f1) == cd2da82e80f489c7558284937a62f9e5eab4c349df68e65553a4f7ff3c4df469 ]]
[[ $(sha256sum experiments_iclr/verify_ogbn_arxiv_untied.py | cut -d ' ' -f1) == fa2515f3f0eebafb4ffc0f5f09cf9675e6595c4d5e17ac49c011227cd1effb1e ]]
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=30s 20m .venv/bin/python experiments_iclr/verify_ogbn_arxiv_untied_v2.py > experiments_iclr/logs/ogbn_arxiv_untied_v2_verify.log 2>&1
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('experiments_iclr/logs/ogbn_arxiv_untied_v2_verify.log')
s=p.read_text()
a=json.loads(s[s.index('{'):])
assert a.get('status') == 'complete' and a.get('verified_arms') == 6, a
PY
touch experiments_iclr/logs/ogbn_arxiv_untied_v2.complete
printf 'OGB_UNTIED_V2_VERIFIED %s\n' "$(date -u +%FT%TZ)"
