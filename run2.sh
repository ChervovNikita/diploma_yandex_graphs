#!/bin/bash
set -e

SPLIT=${1:-0}
DEVICE=${2:-"cuda:1"}
LOGDIR="logs_tabm"
mkdir -p $LOGDIR

python run_tabm.py --device $DEVICE --split $SPLIT --models GAT-sep >> "$LOGDIR/TABM-GAT-sep_5L_split${SPLIT}.log"
python run_tabm.py --device $DEVICE --split $SPLIT --models GT >> "$LOGDIR/TABM-GT_5L_split${SPLIT}.log"
python run_tabm.py --device $DEVICE --split $SPLIT --models GT-sep >> "$LOGDIR/TABM-GT-sep_5L_split${SPLIT}.log"
python run_tabm.py --device $DEVICE --split $SPLIT --models TAG >> "$LOGDIR/TABM-TAG_5L_split${SPLIT}.log"

python run_tabm.py --device $DEVICE --split $SPLIT --models ResNet >> "$LOGDIR/TABM-ResNet_5L_split${SPLIT}.log"
python run_tabm.py --device $DEVICE --split $SPLIT --models GCN >> "$LOGDIR/TABM-GCN_5L_split${SPLIT}.log"
python run_tabm.py --device $DEVICE --split $SPLIT --models SAGE >> "$LOGDIR/TABM-SAGE_5L_split${SPLIT}.log"
python run_tabm.py --device $DEVICE --split $SPLIT --models GAT >> "$LOGDIR/TABM-GAT_5L_split${SPLIT}.log"

echo "All TABM experiments for split $SPLIT done."
