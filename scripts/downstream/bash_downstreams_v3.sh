#!/bin/bash
# Run downstream evaluation for aligned difference v3 model
# Usage: bash bash_downstreams_v3.sh

export CUDA_VISIBLE_DEVICES=0

# Model config: path:layers:hidden:name
MODEL_CONFIG="difference_aligned_v3/best_model.pt:6:256:aligned_v3"

# Classification tasks
cls_filter_names=("Myeloid" "Multiple_Sclerosis" "pancread" "ircolitis" "myasthenia" "lupus" "scfoundation" "scanorama" "dengue" "leptomeningeal")

# Regression tasks
reg_filter_names=("Lung" "Pancreas" "CellBench")

echo "=========================================="
echo "Starting Downstream Evaluation for Aligned V3"
echo "Model: difference_aligned_v3/best_model.pt"
echo "Tasks: ${#cls_filter_names[@]} classification + ${#reg_filter_names[@]} regression"
echo "=========================================="

# Classification tasks
for filter_name in "${cls_filter_names[@]}"; do
    IFS=':' read -r model_path n_layers hidden_dim model_name <<< "$MODEL_CONFIG"
    
    echo ""
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Classification: $filter_name (${model_name})"
    echo "    Model: $model_path | Layers: $n_layers | Hidden: $hidden_dim"
    
    python downstream_classification.py \
        --pretrain_model_path "save_pretrain/$model_path" \
        --n_layers "$n_layers" \
        --hidden_dim "$hidden_dim" \
        --filter_name "$filter_name" \
        --result_suffix "_${model_name}"
    
    if [ $? -eq 0 ]; then
        echo "    ✓ Completed: $filter_name"
    else
        echo "    ✗ Failed: $filter_name"
    fi
done

# Regression tasks
for filter_name in "${reg_filter_names[@]}"; do
    IFS=':' read -r model_path n_layers hidden_dim model_name <<< "$MODEL_CONFIG"
    
    echo ""
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Regression: $filter_name (${model_name})"
    echo "    Model: $model_path | Layers: $n_layers | Hidden: $hidden_dim"
    
    python downstream_regression.py \
        --pretrain_model_path "save_pretrain/$model_path" \
        --n_layers "$n_layers" \
        --hidden_dim "$hidden_dim" \
        --filter_name "$filter_name" \
        --result_suffix "_${model_name}"
    
    if [ $? -eq 0 ]; then
        echo "    ✓ Completed: $filter_name"
    else
        echo "    ✗ Failed: $filter_name"
    fi
done

echo ""
echo "=========================================="
echo "All downstream tasks completed!"
echo "Check results in save_downstream/"
echo "=========================================="
