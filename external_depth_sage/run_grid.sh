#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
export TMPDIR="$PWD/external_depth_sage/.tmp"
mkdir -p "$TMPDIR"
for dataset in wikics actor; do
  for depth in 2 5; do
    .venv/bin/python external_depth_sage/external_depth_grid.py --dataset "$dataset" --depth "$depth" --device cuda:0
  done
done
