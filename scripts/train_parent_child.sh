#!/bin/bash
# ============================================================================
# "Breaking the Reversal Curse in Autoregressive Language Models via Identity
# Bridge" — real-LLM experiments on the Parent-Child reversal task (paper
# Sec. 4.2, Fig. 6b). Full-parameter SFT of Llama-3.2-1B-Instruct over 100
# (child, parent) pairs. Same recipe as scripts/train_husband_wife.sh, with the
# parent/child relation in place of husband/wife.
#
# Reversal task: train on the forward fact "The parent of A is? -> B", then test
# the unseen reverse "The child of B is? -> A". Three training sets (registered
# in LLaMA-Factory/data/dataset_info.json):
#
#   (1) fwd-baseline ......... forward fact only, no identity bridge, weights=1.
#                              Reversal accuracy stays ~0 -- the reversal curse.
#   (2) identity-bridge-IDN .. forward fact (6x) + naive identity bridge "The
#                              name of A is? -> A". Ablation; stays cursed.
#   (3) identity-bridge-OCR .. proposed recipe: bridge rephrased to its OCR form
#                              "The child of A's parent is? -> A", forward fact
#                              up-weighted 6x. Breaks the curse (reversal ~50%).
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

# (1) fwd-baseline: forward fact only -> the reversal curse (uncomment to run)
run_experiment "fwd-baseline" \
  "parent_train_forward" \
  "parent_eval,parent_shortcut" \
  "6e-4" "0.25" "10"

# (2) identity bridge, IDN form -- naive "The name of A is A" (ablation; stays cursed)
run_experiment "identity-bridge-IDN" \
  "parent_train_bridge_idn" \
  "parent_eval,parent_shortcut" \
  "6e-4" "0.25" "30"

# (3) identity bridge, OCR form (proposed) -- "The child of A's parent is A", reversal ~50% (uncomment to run)
run_experiment "identity-bridge-OCR" \
  "parent_train_bridge" \
  "parent_eval,parent_shortcut" \
  "6e-4" "0.25" "30"
