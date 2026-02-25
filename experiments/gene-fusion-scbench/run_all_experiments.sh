#!/bin/bash
# Master script to run all Gene Embedding Fusion experiments
# 5 datasets × 2 models × 4 fusion methods = 40 configurations

set -e

# Paths
WORK_DIR="/root/autodl-tmp/scbenchmark/experiments/gene-fusion-scbench"
SCRIPT="${WORK_DIR}/downstreams_fusion.py"
RESULTS_CSV="${WORK_DIR}/results_scbench.csv"
LOG_FILE="${WORK_DIR}/experiment.log"
PROGRESS_FILE="${WORK_DIR}/progress.txt"

# Model paths
DIFF_V3_PATH="/root/autodl-tmp/scbenchmark/save_pretrain/difference_aligned_v3/best_model.pt"


# Datasets
DATASETS=("Myeloid" "pancread" "lupus" "scfoundation" "Multiple_Sclerosis")

# Fusion methods
FUSION_METHODS=("weighted_avg" "concatenation" "gating" "attention")

# Models
declare -A MODEL_PATHS
MODEL_PATHS["difference_v3"]="${DIFF_V3_PATH}"


# Initialize
cd ${WORK_DIR}
rm -f ${RESULTS_CSV}
echo "Starting experiments at $(date)" | tee ${LOG_FILE}
echo "0/40 configurations completed" > ${PROGRESS_FILE}

# Counters
total_configs=40
completed=0
failed=0

# Progress tracking
start_time=$(date +%s)

# Function to report progress
report_progress() {
    local current=$1
    local total=$2
    local current_config=$3
    
    elapsed=$(($(date +%s) - start_time))
    elapsed_hours=$((elapsed / 3600))
    elapsed_mins=$(((elapsed % 3600) / 60))
    
    if [ $current -gt 0 ]; then
        avg_time=$((elapsed / current))
        remaining=$((avg_time * (total - current)))
        remaining_hours=$((remaining / 3600))
        remaining_mins=$(((remaining % 3600) / 60))
    else
        remaining_hours=0
        remaining_mins=0
    fi
    
    echo ""
    echo "=========================================="
    echo "PROGRESS REPORT"
    echo "=========================================="
    echo "Completed: ${current}/${total} configurations"
    echo "Failed: ${failed}"
    echo "Current: ${current_config}"
    echo "Elapsed: ${elapsed_hours}h ${elapsed_mins}m"
    echo "Estimated remaining: ${remaining_hours}h ${remaining_mins}m"
    echo "=========================================="
    echo ""
    
    echo "${current}/${total} configurations completed (${failed} failed)" > ${PROGRESS_FILE}
}

# Run all experiments
for dataset in "${DATASETS[@]}"; do
    for model_name in "difference_v3" "genept"; do
        model_path="${MODEL_PATHS[$model_name]}"
        
        for fusion in "${FUSION_METHODS[@]}"; do
            config_name="${dataset}|${model_name}|${fusion}"
            
            echo ""
            echo "###############################################################################"
            echo "# Configuration $((completed + 1))/${total_configs}: ${config_name}"
            echo "###############################################################################"
            echo ""
            
            # Report progress every 30 minutes or at key milestones
            report_progress ${completed} ${total_configs} "${config_name}"
            
            # Run experiment
            if python3 ${SCRIPT} \
                --model_name ${model_name} \
                --model_path ${model_path} \
                --dataset_name ${dataset} \
                --fusion_method ${fusion} \
                --output_csv ${RESULTS_CSV} 2>&1 | tee -a ${LOG_FILE}; then
                
                completed=$((completed + 1))
                echo "✅ SUCCESS: ${config_name}" | tee -a ${LOG_FILE}
            else
                failed=$((failed + 1))
                echo "❌ FAILED: ${config_name}" | tee -a ${LOG_FILE}
                echo "${dataset},${model_name},${fusion},ERROR,ERROR" >> ${RESULTS_CSV}
            fi
            
            # Save checkpoint
            echo "${completed}/${total_configs} completed, ${failed} failed" > ${PROGRESS_FILE}
            
            # Brief pause between experiments
            sleep 2
        done
    done
done

# Final report
end_time=$(date +%s)
total_time=$((end_time - start_time))
total_hours=$((total_time / 3600))
total_mins=$(((total_time % 3600) / 60))

echo ""
echo "================================================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "================================================================================"
echo "Total configurations: ${total_configs}"
echo "Successful: ${completed}"
echo "Failed: ${failed}"
echo "Total time: ${total_hours}h ${total_mins}m"
echo "Results saved to: ${RESULTS_CSV}"
echo "================================================================================"

# Generate summary report
echo ""
echo "Generating summary report..."
python3 -c "
import pandas as pd
import numpy as np

# Load results
df = pd.read_csv('${RESULTS_CSV}')

# Filter out errors
df_valid = df[df['accuracy'] != 'ERROR'].copy()
df_valid['accuracy'] = df_valid['accuracy'].astype(float)

# Calculate mean and std for each configuration
summary = df_valid.groupby(['dataset', 'model', 'fusion'])['accuracy'].agg(['mean', 'std', 'count']).reset_index()

# Print summary
print('\n' + '='*80)
print('SUMMARY: Mean ± Std Accuracy (%)')
print('='*80)
for _, row in summary.iterrows():
    print(f'{row[\"dataset\"]:20s} | {row[\"model\"]:15s} | {row[\"fusion\"]:15s} | {row[\"mean\"]:6.2f}±{row[\"std\"]:5.2f} (n={int(row[\"count\"])})')

# Save summary
summary.to_csv('${WORK_DIR}/summary.csv', index=False)
print('\nSummary saved to: ${WORK_DIR}/summary.csv')
"

echo ""
echo "Finished at $(date)"
