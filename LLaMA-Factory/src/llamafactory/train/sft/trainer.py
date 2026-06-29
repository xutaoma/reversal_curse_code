# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_equal_to_4_46
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


def compute_batch_mean_weights(weights, global_effective_batch_size):
    r"""
    Maps each per-sample ``weight`` to the mean weight of its *global effective batch*.

    A global effective batch is the contiguous block of ``global_effective_batch_size``
    samples in dataset order that is consumed between two optimizer steps (where
    ``global_effective_batch_size == world_size * gradient_accumulation_steps *
    per_device_train_batch_size``). Returns a list parallel to ``weights`` where entry ``i``
    is ``mean(weights of the block containing i)``.

    Per-sample loss weighting then scales sample ``i``'s loss by ``weights[i] / out[i]``;
    these factors average to 1 over each block, so the gradient scale is preserved while
    each example contributes in proportion to its weight. The value is constant within a
    block, so it is independent of how the block is sharded across data-parallel ranks.
    This is a module-level pure function so it can be unit-tested without a Trainer.
    """
    step = max(int(global_effective_batch_size), 1)
    out = []
    for i in range(0, len(weights), step):
        chunk = weights[i:i + step]
        mean_w = sum(chunk) / len(chunk)
        out.extend([mean_w] * len(chunk))
    return out


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", data_args=None, processor: Optional["ProcessorMixin"] = None,
        eval_datasets_dict: Optional[Dict[str, "Dataset"]] = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.disable_shuffling = data_args.disable_shuffling if data_args is not None else False
        self.eval_datasets_dict = eval_datasets_dict  # Store individual eval datasets for separate reporting

        # Precompute the per-batch mean weight used for per-sample loss weighting.
        self._precompute_batch_mean_weights()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def _precompute_batch_mean_weights(self):
        r"""
        Precomputes, for every training example, the mean ``weight`` of the *global effective
        batch* it belongs to, and stores it as a ``batch_mean_weight`` column.

        Per-sample weighting (see ``compute_loss``) scales each example's loss by
        ``w_i / mean_w``, where ``mean_w`` is the mean weight over all samples consumed
        between two optimizer steps:

            global_effective_batch_size = world_size * gradient_accumulation_steps
                                          * per_device_train_batch_size

        Why this works on a single node with N GPUs
        -------------------------------------------
        Weighting requires ``disable_shuffling=True``, so the sampler is deterministic:
        ``SequentialSampler`` on 1 GPU, and ``DistributedSampler(shuffle=False)`` on N GPUs
        (see ``get_train_dataloader``; this raw dataloader is used as-is, accelerate does not
        re-shard it). Under ``DistributedSampler(shuffle=False)`` rank ``r`` receives the
        strided indices ``r, r+N, r+2N, ...``; therefore the union of the indices consumed by
        all ``N`` ranks during one optimizer step is *exactly a contiguous block of
        ``global_effective_batch_size`` samples in dataset order*. We can thus precompute one
        ``mean_w`` per contiguous block. The value is identical for every sample in a block,
        so it does not matter which rank ends up reading which sample.

        Correctness
        -----------
        ``compute_loss`` returns ``base_loss * w_i / mean_w``, where ``base_loss`` is the
        standard (already token-normalized) per-micro-batch loss. The per-sample factors
        ``w_i / mean_w`` average to 1 over each global effective batch, so:
          * with all weights 1.0 it is a no-op (we skip this method entirely, below), and
          * the overall gradient scale / effective learning rate is preserved while each
            example's contribution is reweighted by ``w_i``.
        Because the rescaling is applied to the standard loss and is linear, it composes
        correctly with however HF/DeepSpeed normalize and average gradients across ranks
        (it is robust to ``average_tokens_across_devices`` being either True or False).

        Requires ``disable_shuffling=True`` so the block structure above holds.
        """
        if self.train_dataset is None:
            return

        if not hasattr(self.train_dataset, "column_names") or "weight" not in self.train_dataset.column_names:
            return

        weights = self.train_dataset["weight"]

        # No weighting needed: behave exactly like the unmodified trainer.
        if all(w == 1.0 for w in weights):
            return

        if not self.disable_shuffling:
            logger.warning_rank0(
                "Per-sample weighting requires --disable_shuffling True so the global effective "
                "batch is a deterministic, contiguous block of samples. Shuffling is enabled, so "
                "weights are IGNORED and training falls back to standard SFT."
            )
            return

        if self.args.per_device_train_batch_size > 1:
            logger.warning_rank0(
                "Per-sample weighting is exact only with per_device_train_batch_size=1 (got "
                f"{self.args.per_device_train_batch_size}). With a larger micro-batch, samples in "
                "the same micro-batch share a single (mean) weight, so weighting is approximate."
            )

        world_size = max(int(getattr(self.args, "world_size", 1) or 1), 1)
        global_effective_batch_size = (
            world_size * self.args.gradient_accumulation_steps * self.args.per_device_train_batch_size
        )

        batch_mean_weights = compute_batch_mean_weights(weights, global_effective_batch_size)

        if "batch_mean_weight" in self.train_dataset.column_names:
            self.train_dataset = self.train_dataset.remove_columns(["batch_mean_weight"])
        self.train_dataset = self.train_dataset.add_column("batch_mean_weight", batch_mean_weights)
        logger.info_rank0(
            "Precomputed per-sample weighting "
            f"(world_size={world_size}, grad_accum={self.args.gradient_accumulation_steps}, "
            f"per_device_bs={self.args.per_device_train_batch_size}, "
            f"global_effective_batch_size={global_effective_batch_size}, dataset_size={len(weights)})."
        )

    @override
    def get_train_dataloader(self):
        r"""
        Returns the training DataLoader.

        Overrides the default behavior to support disabling shuffling when requested.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # If shuffling is disabled, we need to override the sampler creation
        if hasattr(self, "disable_shuffling") and self.disable_shuffling:
            from torch.utils.data import DataLoader, SequentialSampler
            from torch.utils.data.distributed import DistributedSampler

            train_dataset = self.train_dataset

            # For distributed training, use DistributedSampler without shuffling
            if self.args.world_size > 1:
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    shuffle=False,
                )
            else:
                train_sampler = SequentialSampler(train_dataset)

            return DataLoader(
                train_dataset,
                batch_size=self._train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=self.args.dataloader_persistent_workers,
            )
        else:
            # Use the default behavior with shuffling
            return super().get_train_dataloader()

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r"""
        Computes a per-sample weighted SFT loss.

        Each example carries a scalar ``weight`` (default 1.0). We take the standard,
        already token-normalized micro-batch loss from ``super().compute_loss`` and scale it
        by ``w_i / mean_w``, where ``mean_w`` is the mean weight over the *global effective
        batch* (precomputed per sample in ``_precompute_batch_mean_weights``).

        Key properties:
          * All weights 1.0 -> ``mean_w == 1`` -> identical to standard training. (In that
            case ``batch_mean_weight`` is never added, so we take the early-return path.)
          * The factors ``w_i / mean_w`` average to 1 over each global effective batch, so the
            gradient scale is preserved and example ``i`` simply contributes ``w_i``x its normal
            amount.
          * Scaling the standard loss is linear, so this composes correctly with HF/DeepSpeed
            gradient normalization and DDP gradient averaging across data-parallel ranks --
            i.e. it is correct on a single node with any number of GPUs, regardless of the
            ``average_tokens_across_devices`` setting.
        """
        # Pop non-model fields so they never reach the model's forward().
        sample_weight = inputs.pop("weight", None)
        batch_mean_weight = inputs.pop("batch_mean_weight", None)

        has_custom_weight = batch_mean_weight is not None and isinstance(sample_weight, torch.Tensor)

        if not has_custom_weight:
            return super().compute_loss(model, inputs, return_outputs, **kwargs)

        # Standard token-normalized loss for this micro-batch (uses num_items_in_batch internally).
        base_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        # Reweight: loss = base_loss * w_i / mean_w.
        # per_device_train_batch_size is 1 in the released experiments, so the micro-batch holds
        # a single sample; sample_weight.mean() equals that sample's weight. (For batch sizes > 1
        # this averages the micro-batch weights, the documented approximation.) Every sample in a
        # global effective batch shares the same mean_w, so the denominator is rank-independent.
        w = sample_weight.to(device=base_loss.device, dtype=base_loss.dtype).mean()
        mean_w = batch_mean_weight[0].to(device=base_loss.device, dtype=base_loss.dtype)
        loss = base_loss * w / mean_w.clamp(min=1e-8)

        if return_outputs:
            return (loss, outputs)
        else:
            return loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"] if "labels" in inputs else None
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            labels = labels.detach().clone() if labels is not None else None  # backup labels
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: "torch.Tensor", tgt_tensor: "torch.Tensor") -> "torch.Tensor":
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))

            writer.write("\n".join(res))

    @override
    def evaluate(
        self,
        eval_dataset: Optional["Dataset"] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        r"""
        Run evaluation and returns metrics.

        When eval_datasets_dict is set and eval_dataset is not explicitly provided,
        this also evaluates each individual eval dataset and logs their metrics separately.
        """
        # Run the main evaluation
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **gen_kwargs,
        )

        # If we have multiple eval datasets and this is the default eval (no explicit eval_dataset provided),
        # evaluate each dataset separately and log their metrics
        if (
            eval_dataset is None
            and self.eval_datasets_dict is not None
            and metric_key_prefix == "eval"
        ):
            for eval_name, eval_ds in self.eval_datasets_dict.items():
                prefix = f"eval_{eval_name}"
                ds_metrics = super().evaluate(
                    eval_dataset=eval_ds,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=prefix,
                    **gen_kwargs,
                )
                # Merge per-dataset metrics into the main metrics dict for logging
                metrics.update(ds_metrics)

        return metrics
