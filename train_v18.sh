#!/bin/bash
# Reproduce the v18 training run (RegNetX-800MF, 640x360, 2 classes) as launched on 2026-10-09.
# usage: ./train_v18.sh <gpu ids, e.g. 0,1> [run dir]
# Needs: conda env "bear" (see Operation Manual), dataset at ./bear_dataset_v_final_final,
#        resplit_1009/val_novel.txt (checkpoint selection list).
set -eo pipefail
GPUS=${1:?usage: train_v18.sh <gpus> [run_dir]}
RUN=${2:-BearDetection_v18_regnetx}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OVERRIDE_BACKBONE=regnetx_008 OVERRIDE_CUDA_DEVICES=$GPUS OVERRIDE_RUN_DIR=$RUN \
OVERRIDE_HEM_CLEAN=1 OVERRIDE_HEM_WORKERS=8 OVERRIDE_BATCH_SIZE=32 \
OVERRIDE_SELECT_LIST=resplit_1009/val_novel.txt OVERRIDE_SELECT=novel_f1 OVERRIDE_ACT_MEAN_MAX=10 \
  python -u train_mobilenetv2_relu_novaic_23_4.py --data-dir bear_dataset_v_final_final --cache-only-verify \
  2>&1 | tee train_$(basename $RUN).log
