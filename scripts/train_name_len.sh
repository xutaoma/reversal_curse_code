#!/bin/bash
# ============================================================================
# "Breaking the Reversal Curse in Autoregressive Language Models via Identity
# Bridge" — entity-token-length ablation (paper Sec. 4.2.2, Fig. 8).
# Full-parameter SFT of Llama-3.2-1B-Instruct on the Husband-Wife task.
#
# All three runs use the proposed OCR-form identity bridge (the forward fact
# up-weighted 6x via the per-sample `weight` column); they differ only in how
# the entity names tokenize, to probe how entity token length affects reversal
# generalization:
#
#   num-names .... 1-token number names  (e.g. "34"),       lr 3e-4
#   normal-names . 2-token names         (e.g. "Sophia"),   lr 5e-4
#   long-names ... 3-token names         (e.g. "Catalina"), lr 6e-4
#
# Finding (Fig. 8): shorter names generalize better -- 1-token number names
# reach ~100% reversal accuracy, while 3-token long names drop to ~10%.
# ============================================================================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Optional: load environment overrides (model path, NGPUS, W&B, ...) from env_var.sh.
[ -f "${REPO_ROOT}/env_var.sh" ] && source "${REPO_ROOT}/env_var.sh"
export LF_DIR="${LF_DIR:-${REPO_ROOT}/LLaMA-Factory}"
cd "${LF_DIR}"

# ---- shared config (override via environment) ----
export NGPUS="${NGPUS:-1}"
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.2-1B-Instruct}"
export MODEL_CODE="Llama-3.2-1B"
export TEMPLATE="llama3"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-30}"
export NUM_EPOCHS="${NUM_EPOCHS:-250}"
export REPORT_TO="${REPORT_TO:-none}"
export WANDB_PROJECT="${WANDB_PROJECT:-reversal-curse-identity-bridge}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/saved_models}"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

run_experiment () {  # <label> <dataset> <evalset> <learning_rate> <weight_decay> <global_batch_size>
  local label="$1"
  export DATASET="$2"
  export EVALSET="$3"
  export LEARNING_RATE="$4"
  export WEIGHT_DECAY="${5:-0.3}"
  export GLOBAL_BATCH_SIZE="${6:-30}"
  export OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_CODE}-${label}-lr${LEARNING_RATE}-wd${WEIGHT_DECAY}-bs${GLOBAL_BATCH_SIZE}-${TIMESTAMP}"
  mkdir -p "${OUTPUT_DIR}"
  echo "=== [${label}] dataset=${DATASET} lr=${LEARNING_RATE} wd=${WEIGHT_DECAY} bs=${GLOBAL_BATCH_SIZE} ngpus=${NGPUS} -> ${OUTPUT_DIR} ==="
  bash ./bash/train_people-description.sh 2>&1 | tee "${OUTPUT_DIR}/train.log"
}

# 1-token number names (e.g. "34") -- best reversal generalization (~100%)
run_experiment "num-names" \
  "names_num_train" \
  "names_num_eval,names_num_shortcut" \
  "3e-4" "0.3" "30"

# 2-token names (e.g. "Sophia")
run_experiment "normal-names" \
  "names_normal_train" \
  "names_normal_eval,names_normal_shortcut" \
  "5e-4" "0.3" "30"

# 3-token names (e.g. "Catalina") -- weakest reversal generalization (~10%)
run_experiment "long-names" \
  "names_long_train" \
  "names_long_eval,names_long_shortcut" \
  "6e-4" "0.3" "30"
