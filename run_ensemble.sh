#!/bin/bash
set -e

SPLIT=${1:-0}
DEVICE=${2:-"cuda:0"}
LOGDIR="logs_ensemble"
mkdir -p $LOGDIR

python run_base_ensemble.py --device $DEVICE --split $SPLIT --models GAT-sep >> "$LOGDIR/GAT-sep_ens_split${SPLIT}.log"
python run_base_ensemble.py --device $DEVICE --split $SPLIT --models GT >> "$LOGDIR/GT_ens_split${SPLIT}.log"
python run_base_ensemble.py --device $DEVICE --split $SPLIT --models GT-sep >> "$LOGDIR/GT-sep_ens_split${SPLIT}.log"
python run_base_ensemble.py --device $DEVICE --split $SPLIT --models TAG >> "$LOGDIR/TAG_ens_split${SPLIT}.log"
python run_base_ensemble.py --device $DEVICE --split $SPLIT --models ResNet >> "$LOGDIR/ResNet_ens_split${SPLIT}.log"
python run_base_ensemble.py --device $DEVICE --split $SPLIT --models GCN >> "$LOGDIR/GCN_ens_split${SPLIT}.log"
python run_base_ensemble.py --device $DEVICE --split $SPLIT --models SAGE >> "$LOGDIR/SAGE_ens_split${SPLIT}.log"
python run_base_ensemble.py --device $DEVICE --split $SPLIT --models GAT >> "$LOGDIR/GAT_ens_split${SPLIT}.log"

echo "All ensemble experiments for split $SPLIT done."
