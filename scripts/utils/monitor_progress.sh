#!/bin/bash
# 监控训练进度并提取关键信息

LOG_FILE="baseline_train.log"
DISK_USAGE=$(df -h /root/autodl-tmp | tail -1 | awk '{print $3"/"$2}')

# 提取最新的epoch和loss信息
LATEST_INFO=$(tail -50 $LOG_FILE | grep -E "Epoch|loss" | tail -5)

# GPU使用情况
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -2 | paste -sd',' -)

# 计算ETA (需要根据实际进度计算)
TOTAL_LINES=$(wc -l < $LOG_FILE)

echo "=== scbenchmark Progress ===" 
echo "Time: $(date +'%H:%M')"
echo ""
echo "Phase 1 Baseline:"
echo "  Log lines: $TOTAL_LINES"
echo "  Latest: $LATEST_INFO"
echo "  GPU Memory (MB): $GPU_MEM"
echo ""
echo "Phase 2 Difference:"
echo "  Status: Not started"
echo ""
echo "Disk: $DISK_USAGE"
