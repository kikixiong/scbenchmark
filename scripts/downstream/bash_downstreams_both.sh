#!/bin/bash
# Run downstream evaluation for both baseline and difference models
# Usage: bash bash_downstreams_both.sh [baseline|difference|both]

MODE=${1:-both}

declare -a model_configs=()

if [ "$MODE" == "baseline" ] || [ "$MODE" == "both" ]; then
    model_configs+=("baseline/best_model.pt:6:256:baseline")
fi

if [ "$MODE" == "difference" ] || [ "$MODE" == "both" ]; then
    model_configs+=("difference/best_model.pt:6:256:difference")
fi

export CUDA_VISIBLE_DEVICES=0

# Classification tasks
cls_filter_names=("Myeloid" "Multiple_Sclerosis" "pancread" "ircolitis" "myasthenia" "lupus" "scfoundation" "scanorama" "dengue" "leptomeningeal")

for config in "${model_configs[@]}"; do
    IFS=':' read -r model_path nlayer embs label <<< "$config"
    echo "========================================"
    echo "Running downstream for: $label"
    echo "Model: $model_path, nlayers=$nlayer, embsize=$embs"
    echo "========================================"
    
    results_file="./save_pretrain/${label}_results.json"
    echo "{" > "$results_file"
    
    # Classification tasks (downstreams_cls.py)
    for filter_name in "${cls_filter_names[@]}"; do
        echo ">>> Classification: $filter_name"
        python downstreams_cls.py --model_path ./save_pretrain/${model_path} \
                                  --nlayers $nlayer --embsize $embs \
                                  --filter_name $filter_name --eval_knn --train_from_features \
                                  --cell_emb_style cls --use_weighted_sampling --num_trials 5 --test_maxseq 512 \
                                  --model_structure transformer 2>&1 | tee -a ./save_pretrain/${label}_cls_${filter_name}.log
    done
    
    # Perturbation classification tasks (downstreams_cls2.py)
    perturb_filter_names=("adamson" "dixit" "norman")
    for filter_name in "${perturb_filter_names[@]}"; do
        echo ">>> Perturbation Classification: $filter_name"
        python downstreams_cls2.py --model_path ./save_pretrain/${model_path} \
                                   --nlayers $nlayer --embsize $embs \
                                   --filter_name $filter_name --train_from_features \
                                   --cell_emb_style cls --use_weighted_sampling --num_trials 5 --test_maxseq 512 \
                                   --model_structure transformer 2>&1 | tee -a ./save_pretrain/${label}_cls2_${filter_name}.log
    done
    
    # Perturbation prediction tasks (downstreams_perturbe_pred.py)
    for filter_name in "${perturb_filter_names[@]}"; do
        echo ">>> Perturbation Prediction: $filter_name"
        python downstreams_perturbe_pred.py --model_path ./save_pretrain/${model_path} \
                                            --nlayers $nlayer --embsize $embs \
                                            --filter_name $filter_name --use_weighted_sampling --print_epoch 10 \
                                            --model_structure transformer --num_trials 5 2>&1 | tee -a ./save_pretrain/${label}_perturb_${filter_name}.log
    done
    
    echo "=== Downstream evaluation complete for $label ==="
done

echo "=== ALL DOWNSTREAM EVALUATIONS COMPLETE ==="
