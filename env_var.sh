#!/usr/bin/env bash

export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.2-1B-Instruct}"
export NGPUS="${NGPUS:-1}"

export REPORT_TO="${REPORT_TO:-none}"
export WANDB_PROJECT="${WANDB_PROJECT:-reversal-curse-identity-bridge}"

export NUM_EPOCHS="${NUM_EPOCHS:-250}"

export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-30}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.3}"

echo "[env_var.sh] ready: model=${MODEL_NAME_OR_PATH##*/}  NGPUS=${NGPUS}  epochs=${NUM_EPOCHS}  report_to=${REPORT_TO}"
