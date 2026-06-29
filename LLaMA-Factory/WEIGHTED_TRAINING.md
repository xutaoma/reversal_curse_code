# Per-Sample Weighting (SFT)

This fork adds a per-example **`weight`** to LLaMA-Factory's supervised fine-tuning (SFT)
path. Each training instance carries a scalar importance; the trainer reweights its loss
accordingly. This is the mechanism used for the identity-bridge experiments (forward facts
up-weighted relative to the auxiliary identity facts).

> An earlier version also carried a per-sample `temperature` field. It was unused (applied
> as a no-op) and has been removed.

## Data format

Add a `weight` field to each example (Alpaca shown; ShareGPT works the same way). Missing
or non-positive values default to `1.0`.

```json
[
  {"instruction": "The name of Catherine's husband is?", "input": "", "output": "Emiliano.", "weight": 6.0},
  {"instruction": "The wife of Catherine's husband is?", "input": "", "output": "Catherine.", "weight": 1.0}
]
```

Register the column in `data/dataset_info.json`:

```json
{
  "my_dataset": {
    "file_name": "my_dataset.json",
    "columns": {"prompt": "instruction", "query": "input", "response": "output", "weight": "weight"}
  }
}
```

## What the trainer does

For each micro-batch, the trainer takes the standard, already token-normalized loss from
`super().compute_loss(...)` and multiplies it by

```
s_i = w_i / mean_w
```

where `mean_w` is the **mean weight over the global effective batch** — the set of samples
consumed between two optimizer steps:

```
global_effective_batch_size = world_size
                            * gradient_accumulation_steps
                            * per_device_train_batch_size
```

`mean_w` is precomputed once per contiguous block of that size (`compute_batch_mean_weights`
in `src/llamafactory/train/sft/trainer.py`) and attached to every sample as a
`batch_mean_weight` column.

Key properties:

- **No-op at `w = 1`.** If every weight is `1.0`, `mean_w = 1`, so the loss is unchanged —
  training is byte-for-byte the stock behavior. (The precompute detects this and skips,
  taking the standard code path entirely.)
- **Scale preserved.** The factors `s_i` average to 1 over each global effective batch, so
  the gradient magnitude / effective learning rate is unchanged; only the *relative*
  contribution of each example changes (a weight-6 example contributes 6× a weight-1 one).
- **Correct on single-node multi-GPU.** `mean_w` is taken over the *global* batch and is
  identical for every sample in a step, and the rescaling is applied linearly to the
  standard loss. So the weighted gradient equals the standard gradient with each sample
  scaled by `s_i`, regardless of how the batch is sharded across GPUs or of the
  `average_tokens_across_devices` setting. (An earlier implementation normalized per-GPU
  and was only correct on a single GPU.)

## Requirements / caveats

- **`--disable_shuffling True`** is required (the launcher sets it). It makes the data
  order deterministic so the global effective batch is a contiguous block of samples; on
  N GPUs the `DistributedSampler(shuffle=False)` strided shards recombine into exactly that
  block. With shuffling on, weights are ignored and training falls back to standard SFT
  (with a warning).
- **`per_device_train_batch_size = 1`** gives exact per-sample weighting (the released
  experiments use this). With a larger micro-batch the samples in a micro-batch share a
  single (mean) weight, so weighting becomes approximate (a warning is logged).
- **Packing** (`--packing true`) is unsupported for weighting: packed sequences fall back
  to weight `1.0`, since a per-sample weight is ill-defined once samples are concatenated.

## Files changed for weighting

| File | Change |
|------|--------|
| `src/llamafactory/data/parser.py` | `weight` column definition |
| `src/llamafactory/data/aligner.py` | extract + validate `weight` per example |
| `src/llamafactory/data/processors/supervised.py` | carry `weight` through preprocessing |
| `src/llamafactory/data/collator.py` | build `weight` / `batch_mean_weight` tensors |
| `src/llamafactory/train/sft/trainer.py` | `compute_batch_mean_weights`, `_precompute_batch_mean_weights`, weighted `compute_loss` |

## Verifying

A standalone numerical test of the weighting math (no GPU needed) lives at
`../tests/test_per_sample_weighting.py`:

```bash
python ../tests/test_per_sample_weighting.py
```

It checks the exact per-block means, scale/ratio preservation, the no-op property, GPU-count
invariance (the multi-GPU fix), and gradient linearity in a faithful DDP model.
