#!/bin/bash
# Run downstream evaluation for aligned difference v3 model
# Based on bash_downstreams_both.sh with path adapted for difference_aligned_v3

MODEL_PATH="difference_aligned_v3/best_model.pt"
NLAYER=6
EMBS=256
LABEL="aligned_v3"

export CUDA_VISIBLE_DEVICES=0

echo "========================================"
echo "Running downstream for: $LABEL"
echo "Model: $MODEL_PATH, nlayers=$NLAYER, embsize=$EMBS"
echo "Started at: $(date)"
echo "========================================"

results_file="./save_pretrain/${LABEL}_results.json"
echo "{" > "$results_file"

# Classification tasks (downstreams_cls.py)
cls_filter_names=("Myeloid" "Multiple_Sclerosis" "pancread" "ircolitis" "myasthenia" "lupus" "scfoundation" "scanorama" "dengue" "leptomeningeal")

for filter_name in "${cls_filter_names[@]}"; do
    echo ""
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Classification: $filter_name"
    python downstreams_cls.py --model_path ./save_pretrain/${MODEL_PATH} \
                              --nlayers $NLAYER --embsize $EMBS \
                              --filter_name $filter_name --eval_knn --train_from_features \
                              --cell_emb_style cls --use_weighted_sampling --num_trials 5 --test_maxseq 512 \
                              --model_structure transformer 2>&1 | tee -a ./save_pretrain/${LABEL}_cls_${filter_name}.log
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "    ✓ Completed: $filter_name"
    else
        echo "    ✗ Failed: $filter_name"
    fi
done

# Perturbation classification tasks (downstreams_cls2.py)
perturb_filter_names=("adamson" "dixit" "norman")
for filter_name in "${perturb_filter_names[@]}"; do
    echo ""
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Perturbation Classification: $filter_name"
    python downstreams_cls2.py --model_path ./save_pretrain/${MODEL_PATH} \
                               --nlayers $NLAYER --embsize $EMBS \
                               --filter_name $filter_name --train_from_features \
                               --cell_emb_style cls --use_weighted_sampling --num_trials 5 --test_maxseq 512 \
                               --model_structure transformer 2>&1 | tee -a ./save_pretrain/${LABEL}_cls2_${filter_name}.log
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "    ✓ Completed: $filter_name"
    else
        echo "    ✗ Failed: $filter_name"
    fi
done

# Perturbation prediction tasks (downstreams_perturbe_pred.py)
for filter_name in "${perturb_filter_names[@]}"; do
    echo ""
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Perturbation Prediction: $filter_name"
    python downstreams_perturbe_pred.py --model_path ./save_pretrain/${MODEL_PATH} \
                                        --nlayers $NLAYER --embsize $EMBS \
                                        --filter_name $filter_name --use_weighted_sampling --print_epoch 10 \
                                        --model_structure transformer --num_trials 5 2>&1 | tee -a ./save_pretrain/${LABEL}_perturb_${filter_name}.log
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "    ✓ Completed: $filter_name"
    else
        echo "    ✗ Failed: $filter_name"
    fi
done

echo ""
echo "========================================"
echo "Downstream evaluation complete for $LABEL"
echo "Completed at: $(date)"
echo "Check logs in: ./save_pretrain/${LABEL}_*.log"
echo "========================================"
