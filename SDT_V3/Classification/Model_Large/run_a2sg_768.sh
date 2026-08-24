#!/bin/bash
# =============================================================================
# A2SG 768 Model (171M) - Pretrain & Finetune Pipeline
# Paper: Scaling Spike-driven Transformer with Efficient Spike Firing
#        Approximation Training (IEEE T-PAMI 2025)
# =============================================================================

set -e

# ---- Activate conda environment ----
source /home/kyccj/anaconda3/etc/profile.d/conda.sh
conda activate sdtv3_cls
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1

# ---- User Config ----
GPUS="0,1,2,3,4,5,6,7"
NUM_GPUS=8
DATA_PATH="/media/hdd1/kyccj/data/ImageNet_down"  # ImageNet-1K

OUTPUT_ROOT="./outputs/a2sg_768"
PRETRAIN_DIR="${OUTPUT_ROOT}/pretrain"
FINETUNE_DIR="${OUTPUT_ROOT}/finetune"

mkdir -p "${PRETRAIN_DIR}" "${FINETUNE_DIR}"

# =============================================================================
# Stage 1: Pretrain (Spike-Masked Autoencoder)
# =============================================================================
echo "=========================================="
echo " Stage 1: Pretrain spikmae_12_768 (a2sg)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${GPUS} torchrun --standalone --nproc_per_node=${NUM_GPUS} --master_port=29500 \
  main_pretrain.py \
  --batch_size 128 \
  --blr 1.5e-4 \
  --warmup_epochs 20 \
  --epochs 200 \
  --model spikmae_12_768 \
  --model_mode a2sg \
  --mask_ratio 0.50 \
  --weight_decay 0.05 \
  --data_path ${DATA_PATH} \
  --output_dir ${PRETRAIN_DIR} \
  --log_dir ${PRETRAIN_DIR}

# =============================================================================
# Stage 2: Finetune (Classification)
# =============================================================================
# Find the last pretrain checkpoint
PRETRAIN_CKPT="${PRETRAIN_DIR}/checkpoint-199.pth"
if [ ! -f "${PRETRAIN_CKPT}" ]; then
  # fallback: find any checkpoint
  PRETRAIN_CKPT=$(ls -t ${PRETRAIN_DIR}/checkpoint-*.pth 2>/dev/null | head -1)
fi

if [ -z "${PRETRAIN_CKPT}" ]; then
  echo "ERROR: No pretrain checkpoint found in ${PRETRAIN_DIR}"
  exit 1
fi

echo "=========================================="
echo " Stage 2: Finetune spikformer12_768 (a2sg)"
echo " Checkpoint: ${PRETRAIN_CKPT}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${GPUS} torchrun --standalone --nproc_per_node=${NUM_GPUS} --master_port=29500 \
  main_finetune.py \
  --batch_size 100 \
  --blr 6e-4 \
  --warmup_epochs 10 \
  --layer_decay 0.75 \
  --epochs 150 \
  --drop_path 0.1 \
  --model spikformer12_768 \
  --model_mode a2sg \
  --finetune ${PRETRAIN_CKPT} \
  --data_path ${DATA_PATH} \
  --output_dir ${FINETUNE_DIR} \
  --log_dir ${FINETUNE_DIR} \
  --reprob 0.25 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --dist_eval

echo "=========================================="
echo " Done! Results saved to ${OUTPUT_ROOT}"
echo "=========================================="
