#!/bin/bash
cd /root/autodl-tmp/scbenchmark
source /etc/network_turbo

echo "Starting GenePT training with 2×RTX4090..."
echo "="
torchrun --nproc_per_node=2 \
  --master_port=29500 \
  train_genept_fixed.py \
  --train_path ./data/train_data.parquet \
  -s save_pretrain/genept \
  --batch_size 128 \
  --epochs 10 \
  --lr 5e-3 \
  --embsize 256 \
  --nlayers 6 \
  --nheads 8 \
  --num_workers 10 \
  2>&1 | tee train_genept_dual.log
