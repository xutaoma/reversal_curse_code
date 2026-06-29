# Copyright 2024 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import torch
from importlib.util import find_spec

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_rouge_available


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


# Helper functions for optional dependencies (removed from transformers.utils in newer versions)
def is_jieba_available():
    return find_spec("jieba") is not None

def is_nltk_available():
    return find_spec("nltk") is not None


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


if is_rouge_available():
    from rouge_chinese import Rouge


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""
    Computes the token with the largest likelihood to reduce memory footprint.
    """
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


def eval_logit_processor_with_logits(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""
    Returns full logits for computing both accuracy and MRR.

    Returns:
        torch.Tensor: Full logits of shape (batch_size, seq_len, vocab_size)
    """
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return logits  # Return full logits instead of argmax


@dataclass
class ComputeAccuracy:
    r"""
    Computes accuracy and supports `batch_eval_metrics`.
    """

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


@dataclass
class ComputeAccuracyAndMRR:
    r"""
    Computes token-level accuracy and MRR (Mean Reciprocal Rank) and supports `batch_eval_metrics`.

    MRR measures the quality of predictions by computing the reciprocal rank of the
    ground-truth token in the model's probability distribution for each position.
    """

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": [], "mrr": [], "first_token_mrr": [], "first_token_accuracy": [], "instance_accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        # Predictions are now full logits: shape [batch_size, seq_len, vocab_size]
        logits, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        # Validate logits shape
        if logits.ndim != 3:
            raise ValueError(
                f"Expected logits of shape [batch_size, seq_len, vocab_size], got shape {logits.shape}. "
                "Make sure to use eval_logit_processor_with_logits instead of eval_logit_processor."
            )

        # Compute argmax predictions for accuracy
        preds = np.argmax(logits, axis=-1)  # [batch_size, seq_len]

        for i in range(len(preds)):
            # Shift predictions and labels for causal LM
            pred, label = preds[i, :-1], labels[i, 1:]
            logit = logits[i, :-1, :]  # [seq_len-1, vocab_size]

            # Filter valid tokens (not IGNORE_INDEX)
            label_mask = label != IGNORE_INDEX

            # Compute accuracy (same as ComputeAccuracy class)
            self.score_dict["accuracy"].append(
                np.mean(pred[label_mask] == label[label_mask])
            )

            # Compute MRR for all tokens
            valid_logits = logit[label_mask]  # [num_valid_tokens, vocab_size]
            valid_labels = label[label_mask]  # [num_valid_tokens]
            valid_preds = pred[label_mask]  # [num_valid_tokens]

            if len(valid_labels) > 0:
                mrr_score = self._compute_mrr(valid_logits, valid_labels)
                self.score_dict["mrr"].append(mrr_score)

                # Compute MRR for only the first token
                first_token_logit = valid_logits[0:1]  # [1, vocab_size]
                first_token_label = valid_labels[0:1]  # [1]
                first_token_mrr_score = self._compute_mrr(first_token_logit, first_token_label)
                self.score_dict["first_token_mrr"].append(first_token_mrr_score)

                # Compute accuracy for only the first token
                first_token_pred = valid_preds[0]
                first_token_accuracy = 1.0 if first_token_pred == first_token_label[0] else 0.0
                self.score_dict["first_token_accuracy"].append(first_token_accuracy)

                # Compute instance accuracy (all tokens must be rank 1)
                all_correct = np.all(valid_preds == valid_labels)
                instance_accuracy = 1.0 if all_correct else 0.0
                self.score_dict["instance_accuracy"].append(instance_accuracy)
            else:
                # No valid tokens, use 0.0 as placeholder
                self.score_dict["mrr"].append(0.0)
                self.score_dict["first_token_mrr"].append(0.0)
                self.score_dict["first_token_accuracy"].append(0.0)
                self.score_dict["instance_accuracy"].append(0.0)

        if compute_result:
            return self._dump()

    def _compute_mrr(self, logits: "np.ndarray", labels: "np.ndarray") -> float:
        """
        Compute Mean Reciprocal Rank for a batch of logits and labels.

        Args:
            logits: Shape [num_valid_tokens, vocab_size]
            labels: Shape [num_valid_tokens]

        Returns:
            float: Mean reciprocal rank score
        """
        reciprocal_ranks = []

        for logit_vec, true_label in zip(logits, labels):
            # Sort indices in descending order of logit values
            # argsort returns indices in ascending order, so we negate logits
            sorted_indices = np.argsort(-logit_vec)

            # Find rank of true label (rank starts from 1)
            rank_positions = np.where(sorted_indices == true_label)[0]

            if len(rank_positions) > 0:
                rank = rank_positions[0] + 1  # +1 because rank starts from 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                # Should not happen, but handle edge case
                reciprocal_ranks.append(0.0)

        return float(np.mean(reciprocal_ranks))


@dataclass
class ComputeSimilarity:
    r"""
    Computes text similarity scores and supports `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()
