#!/bin/bash
# ---------------------------------------------------------------------------
# Full-parameter SFT launcher shared by ../scripts/train_husband_wife.sh,
# ../scripts/train_parent_child.sh, and ../scripts/train_name_len.sh. All
# configuration is read from environment variables (those scripts set them;
# sensible defaults are provided here).
#
# Per-sample weighting (the identity-bridge regularizer) requires
# `--disable_shuffling True`: the weight of each instance is read from the
# dataset's `weight` column, and the trainer normalizes each loss by the mean
# weight of its *global effective batch* (world_size * grad_accum * micro_bs).
# Disabling shuffling makes that batch a deterministic, contiguous block of
# samples, which is what makes the weighting correct on a single node with any
# number of GPUs. See:
#   src/llamafactory/train/sft/trainer.py  (_precompute_batch_mean_weights, compute_loss)
# ---------------------------------------------------------------------------

export PATH=/usr/local/cuda/bin:$PATH                       # nvcc for any deepspeed JIT ops
# Make the bundled package importable whether or not `pip install -e .` was run.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

ngpus="${NGPUS:-1}"                                         # GPUs on this single node
model_name_or_path="${MODEL_NAME_OR_PATH:?set MODEL_NAME_OR_PATH}"
output_dir="${OUTPUT_DIR:?set OUTPUT_DIR}"
dataset="${DATASET:?set DATASET}"
evalset="${EVALSET:-$dataset}"
template="${TEMPLATE:-llama3}"

learning_rate="${LEARNING_RATE:-6e-4}"
weight_decay="${WEIGHT_DECAY:-0.3}"
num_epochs="${NUM_EPOCHS:-250}"
global_batch_size="${GLOBAL_BATCH_SIZE:-30}"
batch_size_per_device="${BATCH_SIZE_PER_DEVICE:-1}"
lr_scheduler_type="${LR_SCHEDULER_TYPE:-constant_with_warmup}"
report_to="${REPORT_TO:-none}"                             # set REPORT_TO=wandb to log to W&B

# Gradient-accumulation steps such that
#   ngpus * accum_steps * batch_size_per_device == global_batch_size.
accum_steps=$(( global_batch_size / ngpus / batch_size_per_device ))
if [ "${accum_steps}" -lt 1 ]; then
  echo "[ERROR] accum_steps < 1 (global_batch_size=${global_batch_size}, ngpus=${ngpus}," \
       "micro_bs=${batch_size_per_device}). Lower NGPUS or raise GLOBAL_BATCH_SIZE." >&2
  exit 1
fi
effective_gbs=$(( ngpus * accum_steps * batch_size_per_device ))
if [ "${effective_gbs}" -ne "${global_batch_size}" ]; then
  echo "[WARN] global_batch_size=${global_batch_size} not divisible by ngpus*micro_bs;" \
       "using effective global batch = ${effective_gbs}." >&2
fi

mkdir -p "${output_dir}"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

deepspeed --num_gpus "${ngpus}" src/train.py \
    --model_name_or_path "${model_name_or_path}" \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --template "${template}" \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --dataset "${dataset}" \
    --eval_dataset "${evalset}" \
    --packing false \
    --cutoff_len 256 \
    --max_samples 100000000 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --output_dir "${output_dir}" \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 1000 \
    --eval_strategy steps \
    --eval_steps 10 \
    --plot_loss \
    --overwrite_output_dir \
    --per_device_train_batch_size "${batch_size_per_device}" \
    --gradient_accumulation_steps "${accum_steps}" \
    --learning_rate "${learning_rate}" \
    --weight_decay "${weight_decay}" \
    --num_train_epochs "${num_epochs}" \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "${lr_scheduler_type}" \
    --bf16 false \
    --ddp_timeout 180000000 \
    --compute_accuracy True \
    --gradient_checkpointing \
    --report_to "${report_to}" \
    --seed 147 \
    --run_name "${MODEL_CODE:-model}-${dataset}-bs${global_batch_size}-lr${learning_rate}-wd${weight_decay}-ep${num_epochs}-${TIMESTAMP}" \
    --disable_shuffling True
