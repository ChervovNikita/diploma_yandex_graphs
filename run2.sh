#!/bin/bash
set -e

SPLIT=${1:-0}
DEVICE=${2:-"cuda:1"}
LOGDIR="logs_tabm"
mkdir -p $LOGDIR

L="$LOGDIR/TABM-GAT-sep_5L_split${SPLIT}.log"; python run_tabm.py --device $DEVICE --split $SPLIT --models GAT-sep --stdout_log "$L" >> "$L"
L="$LOGDIR/TABM-GT_5L_split${SPLIT}.log"; python run_tabm.py --device $DEVICE --split $SPLIT --models GT --stdout_log "$L" >> "$L"
L="$LOGDIR/TABM-GT-sep_5L_split${SPLIT}.log"; python run_tabm.py --device $DEVICE --split $SPLIT --models GT-sep --stdout_log "$L" >> "$L"
L="$LOGDIR/TABM-TAG_5L_split${SPLIT}.log"; python run_tabm.py --device $DEVICE --split $SPLIT --models TAG --stdout_log "$L" >> "$L"

L="$LOGDIR/TABM-ResNet_5L_split${SPLIT}.log"; python run_tabm.py --device $DEVICE --split $SPLIT --models ResNet --stdout_log "$L" >> "$L"
L="$LOGDIR/TABM-GCN_5L_split${SPLIT}.log"; python run_tabm.py --device $DEVICE --split $SPLIT --models GCN --stdout_log "$L" >> "$L"
L="$LOGDIR/TABM-SAGE_5L_split${SPLIT}.log"; python run_tabm.py --device $DEVICE --split $SPLIT --models SAGE --stdout_log "$L" >> "$L"
L="$LOGDIR/TABM-GAT_5L_split${SPLIT}.log"; python run_tabm.py --device $DEVICE --split $SPLIT --models GAT --stdout_log "$L" >> "$L"

echo "All TABM experiments for split $SPLIT done."
