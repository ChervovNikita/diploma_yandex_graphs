#!/bin/bash
set -e

SPLIT=${1:-0}
DEVICE=${2:-"cuda:0"}
LOGDIR="logs_base"
mkdir -p $LOGDIR

python run_base.py --device $DEVICE --split $SPLIT --models GAT-sep --layers 5 >> "$LOGDIR/GAT-sep_5L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models GT --layers 5 >> "$LOGDIR/GT_5L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models GT-sep --layers 5 >> "$LOGDIR/GT-sep_5L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models TAG --layers 5 >> "$LOGDIR/TAG_5L_split${SPLIT}.log"

python run_base.py --device $DEVICE --split $SPLIT --models ResNet --layers 5 >> "$LOGDIR/ResNet_5L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models ResNet --layers 1 >> "$LOGDIR/ResNet_1L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models ResNet --layers 2 >> "$LOGDIR/ResNet_2L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models ResNet --layers 3 >> "$LOGDIR/ResNet_3L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models ResNet --layers 4 >> "$LOGDIR/ResNet_4L_split${SPLIT}.log"

python run_base.py --device $DEVICE --split $SPLIT --models GCN --layers 5 >> "$LOGDIR/GCN_5L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models GCN --layers 1 >> "$LOGDIR/GCN_1L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models GCN --layers 2 >> "$LOGDIR/GCN_2L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models GCN --layers 3 >> "$LOGDIR/GCN_3L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models GCN --layers 4 >> "$LOGDIR/GCN_4L_split${SPLIT}.log"

python run_base.py --device $DEVICE --split $SPLIT --models SAGE --layers 5 >> "$LOGDIR/SAGE_5L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models SAGE --layers 1 >> "$LOGDIR/SAGE_1L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models SAGE --layers 2 >> "$LOGDIR/SAGE_2L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models SAGE --layers 3 >> "$LOGDIR/SAGE_3L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models SAGE --layers 4 >> "$LOGDIR/SAGE_4L_split${SPLIT}.log"

python run_base.py --device $DEVICE --split $SPLIT --models GAT --layers 5 >> "$LOGDIR/GAT_5L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models GAT --layers 1 >> "$LOGDIR/GAT_1L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models GAT --layers 2 >> "$LOGDIR/GAT_2L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models GAT --layers 3 >> "$LOGDIR/GAT_3L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models GAT --layers 4 >> "$LOGDIR/GAT_4L_split${SPLIT}.log"

python run_base.py --device $DEVICE --split $SPLIT --models GAT-sep --layers 1 >> "$LOGDIR/GAT-sep_1L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models GAT-sep --layers 2 >> "$LOGDIR/GAT-sep_2L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models GAT-sep --layers 3 >> "$LOGDIR/GAT-sep_3L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models GAT-sep --layers 4 >> "$LOGDIR/GAT-sep_4L_split${SPLIT}.log"

python run_base.py --device $DEVICE --split $SPLIT --models GT --layers 1 >> "$LOGDIR/GT_1L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models GT --layers 2 >> "$LOGDIR/GT_2L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models GT --layers 3 >> "$LOGDIR/GT_3L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models GT --layers 4 >> "$LOGDIR/GT_4L_split${SPLIT}.log"

python run_base.py --device $DEVICE --split $SPLIT --models GT-sep --layers 1 >> "$LOGDIR/GT-sep_1L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models GT-sep --layers 2 >> "$LOGDIR/GT-sep_2L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models GT-sep --layers 3 >> "$LOGDIR/GT-sep_3L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models GT-sep --layers 4 >> "$LOGDIR/GT-sep_4L_split${SPLIT}.log"

python run_base.py --device $DEVICE --split $SPLIT --models TAG --layers 1 >> "$LOGDIR/TAG_1L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models TAG --layers 2 >> "$LOGDIR/TAG_2L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models TAG --layers 3 >> "$LOGDIR/TAG_3L_split${SPLIT}.log"
python run_base.py --device $DEVICE --split $SPLIT --models TAG --layers 4 >> "$LOGDIR/TAG_4L_split${SPLIT}.log"

echo "All base experiments for split $SPLIT done."
