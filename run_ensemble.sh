#!/bin/bash
set -e

SPLIT=${1:-0}
DEVICE=${2:-"cuda:0"}
LOGDIR="logs_ensemble"
mkdir -p $LOGDIR

L="$LOGDIR/GAT-sep_ens_split${SPLIT}.log"; python run_base_ensemble.py --device $DEVICE --split $SPLIT --models GAT-sep --stdout_log "$L" >> "$L"
L="$LOGDIR/GT_ens_split${SPLIT}.log"; python run_base_ensemble.py --device $DEVICE --split $SPLIT --models GT --stdout_log "$L" >> "$L"
L="$LOGDIR/GT-sep_ens_split${SPLIT}.log"; python run_base_ensemble.py --device $DEVICE --split $SPLIT --models GT-sep --stdout_log "$L" >> "$L"
L="$LOGDIR/TAG_ens_split${SPLIT}.log"; python run_base_ensemble.py --device $DEVICE --split $SPLIT --models TAG --stdout_log "$L" >> "$L"
L="$LOGDIR/ResNet_ens_split${SPLIT}.log"; python run_base_ensemble.py --device $DEVICE --split $SPLIT --models ResNet --stdout_log "$L" >> "$L"
L="$LOGDIR/GCN_ens_split${SPLIT}.log"; python run_base_ensemble.py --device $DEVICE --split $SPLIT --models GCN --stdout_log "$L" >> "$L"
L="$LOGDIR/SAGE_ens_split${SPLIT}.log"; python run_base_ensemble.py --device $DEVICE --split $SPLIT --models SAGE --stdout_log "$L" >> "$L"
L="$LOGDIR/GAT_ens_split${SPLIT}.log"; python run_base_ensemble.py --device $DEVICE --split $SPLIT --models GAT --stdout_log "$L" >> "$L"

echo "All ensemble experiments for split $SPLIT done."
