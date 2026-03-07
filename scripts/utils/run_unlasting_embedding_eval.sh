#!/usr/bin/env bash
set -euo pipefail

# Batch evaluation helper for unlasting/*.h5ad datasets
# Supports both old positional style and new flag style.
#
# Positional (backward compatible):
#   bash scripts/utils/run_unlasting_embedding_eval.sh <model_path> [fusion] [dataset_dir] [output]
#
# Flag style (recommended):
#   bash scripts/utils/run_unlasting_embedding_eval.sh \
#     --model_path /root/autodl-tmp/best_model.pt \
#     --fusion attention \
#     --dataset_dir unlasting \
#     --output experiments/gene-fusion-scbench/unlasting_attention_summary.csv \
#     --label_key perturbation

MODEL_PATH="autodl-tmp/best_model.pt"
FUSION="attention"
DATASET_DIR="unlasting"
OUTPUT=""
LABEL_KEY=""
MIN_CLASS_COUNT="2"
EXTRA_ARGS=()

# Backward-compatible positional parsing for first up to 4 args.
if [[ $# -gt 0 && "${1}" != --* ]]; then
  MODEL_PATH="${1}"
  shift
  if [[ $# -gt 0 && "${1}" != --* ]]; then FUSION="${1}"; shift; fi
  if [[ $# -gt 0 && "${1}" != --* ]]; then DATASET_DIR="${1}"; shift; fi
  if [[ $# -gt 0 && "${1}" != --* ]]; then OUTPUT="${1}"; shift; fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path)
      MODEL_PATH="$2"; shift 2;;
    --fusion)
      FUSION="$2"; shift 2;;
    --dataset_dir|--dataset_path)
      DATASET_DIR="$2"; shift 2;;
    --output|--output_json)
      OUTPUT="$2"; shift 2;;
    --label_key)
      LABEL_KEY="$2"; shift 2;;
    --min_class_count)
      MIN_CLASS_COUNT="$2"; shift 2;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break;;
    *)
      # passthrough any valid paper_embedding_eval.py args
      EXTRA_ARGS+=("$1")
      shift
      if [[ $# -gt 0 && "${1}" != --* ]]; then
        EXTRA_ARGS+=("$1")
        shift
      fi
      ;;
  esac
done

if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="experiments/gene-fusion-scbench/unlasting_${FUSION}_summary.csv"
fi

CMD=(
  python util-tmps/paper_embedding_eval.py
  --dataset_path "${DATASET_DIR}"
  --model_path "${MODEL_PATH}"
  --embedding_key weights.embedding
  --vocab_path vocab.json
  --fusion "${FUSION}"
  --output_json "${OUTPUT}"
  --min_class_count "${MIN_CLASS_COUNT}"
)

if [[ -n "${LABEL_KEY}" ]]; then
  CMD+=(--label_key "${LABEL_KEY}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"

echo "Saved summary to ${OUTPUT}"
