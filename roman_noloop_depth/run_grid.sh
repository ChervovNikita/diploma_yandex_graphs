#!/usr/bin/env bash
set -euo pipefail
cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs
export TMPDIR="$PWD/roman_noloop_depth/.tmp"
mkdir -p "$TMPDIR"
for depth in 2 5; do
  .venv/bin/python roman_noloop_depth/roman_noloop_depth.py --dataset roman --depth "$depth" --device cuda:0
done
