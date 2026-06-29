#!/bin/bash
# ============================================================================
# "Breaking the Reversal Curse in Autoregressive Language Models via Identity
# Bridge" — real-LLM experiments on the Husband-Wife reversal task (paper
# Sec. 4.2 and the format ablation in Sec. 4.2.1). Full-parameter SFT of
# Llama-3.2-1B-Instruct over 100 (wife, husband) pairs.
#
# Reversal task: train on the forward fact "A's husband is B", then test the
# reverse question "who is B's wife?", which is never shown in training. We
# compare three training sets (registered in LLaMA-Factory/data/dataset_info.json):
#
#   (1) fwd-baseline ......... forward fact only ("The husband of A is? -> B"),
#                              no identity bridge, all weights = 1. The model
#                              fits the forward fact but reversal accuracy stays
#                              ~0 -- the reversal curse.
#
#   (2) identity-bridge-IDN .. forward fact (up-weighted 6x) + the identity
#                              bridge in its naive IDENTITY (IDN) form ("The name
#                              of A is? -> A", "The name of B is? -> B"). Adds no
#                              relational info and, though equivalent to (3),
#                              does NOT break the curse (ablation, Sec. 4.2.1).
#
#   (3) identity-bridge-OCR .. the proposed recipe. The bridge is rephrased into
#                              its OCR (out-of-context reasoning) form, "The wife
#                              of A's husband is? -> A", which recasts the reversal
#                              task as an OCR task (Prop. 3.5). With the forward
#                              fact up-weighted 6x (OCR-6), reversal accuracy
#                              reaches ~50%.
#
# The 6x up-weighting of the forward fact (1x for the bridge / self-identity
# facts) is read from the `weight` column of each dataset and applied during SFT.
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

# (1) fwd-baseline: forward fact only -> the reversal curse (reversal acc ~0)
run_experiment "fwd-baseline" \
  "couple_train_forward" \
  "couple_eval,couple_shortcut" \
  "6e-4" "0.3" "10"

# (2) identity bridge, IDN form -- naive "The name of A is A" (ablation; stays cursed)
run_experiment "identity-bridge-IDN" \
  "couple_train_bridge_idn" \
  "couple_eval,couple_shortcut" \
  "6e-4" "0.3" "30"

# (3) identity bridge, OCR form (proposed) -- "The wife of A's husband is A", reversal acc ~50%
run_experiment "identity-bridge-OCR" \
  "couple_train_bridge" \
  "couple_eval,couple_shortcut" \
  "6e-4" "0.3" "30"


