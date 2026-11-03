#!/usr/bin/env bash
set -euo pipefail

cd /home/wxy/Classification/syy/syy/strock_test/OCTA_pytorch/OCTA_VIT

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
/home/wxy/miniforge3/envs/yolov52/bin/python -u train_p23_xr.py \
    --task p3 \
    --config /home/wxy/Classification/syy/syy/strock_test/OCTA_pytorch/OCTA_VIT/config_v1_local.yaml \
    --feature-manifest /home/wxy/Classification/syy/syy/strock_test/OCTA_pytorch/OCTA_VIT/runs/P3/feature_extraction_manifest_p3.csv \
    --split-manifest /home/wxy/Classification/syy/syy/strock_test/OCTA_pytorch/OCTA_VIT/runs/B/split_manifest.csv \
    --ablation all \
    --device cpu
