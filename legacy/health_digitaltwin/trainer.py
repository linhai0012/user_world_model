"""
Custom Trainer with modality-aware weighted loss.

Splits the cross-entropy loss into text-token loss and HR-token loss,
then combines them with a configurable weight alpha:

    L = alpha * L_text + (1 - alpha) * L_hr

When alpha = -1, falls back to standard uniform cross-entropy.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import Trainer

from train_config import TrainConfig


class WeightedLossTrainer(Trainer):
    """
    Extends HuggingFace Trainer to support modality-weighted loss.

    Expects each batch to contain an 'is_hr_token' mask tensor alongside
    the standard input_ids, attention_mask, labels.
    """

    def __init__(self, loss_alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.loss_alpha = loss_alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pop is_hr_token before forwarding to model (model doesn't expect it)
        is_hr_token = inputs.pop("is_hr_token", None)

        # Standard forward pass
        outputs = model(**inputs)
        logits = outputs.logits  # (batch, seq_len, vocab_size)

        labels = inputs["labels"]

        # If no weighting requested, use the model's own loss
        if self.loss_alpha < 0 or is_hr_token is None:
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        # ── Compute weighted loss manually ────────────────────────────────
        # Shift logits and labels for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_hr_mask = is_hr_token[:, 1:].contiguous()

        # Flatten
        B, S, V = shift_logits.shape
        flat_logits = shift_logits.view(-1, V)
        flat_labels = shift_labels.view(-1)
        flat_hr_mask = shift_hr_mask.view(-1)

        # Per-token cross entropy (no reduction)
        per_token_loss = F.cross_entropy(
            flat_logits, flat_labels, reduction="none", ignore_index=-100
        )

        # Identify valid (non-masked) tokens
        valid_mask = flat_labels != -100

        # Text tokens: valid AND not HR
        text_mask = valid_mask & (flat_hr_mask == 0)
        # HR tokens: valid AND HR
        hr_mask = valid_mask & (flat_hr_mask == 1)

        n_text = text_mask.sum().clamp(min=1)
        n_hr = hr_mask.sum().clamp(min=1)

        loss_text = (per_token_loss * text_mask.float()).sum() / n_text
        loss_hr = (per_token_loss * hr_mask.float()).sum() / n_hr

        # Weighted combination
        alpha = self.loss_alpha
        loss = alpha * loss_text + (1.0 - alpha) * loss_hr

        # Log individual losses for monitoring
        if self.state.global_step % self.args.logging_steps == 0:
            self._log_modality_losses(loss_text, loss_hr, n_text, n_hr)

        return (loss, outputs) if return_outputs else loss

    def preprocess_logits_for_eval(self, logits, labels):
        """Reduce logits to argmax predictions before accumulation to avoid OOM."""
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def _log_modality_losses(self, loss_text, loss_hr, n_text, n_hr):
        """Log modality-specific losses to the trainer's logger."""
        if self.state.is_world_process_zero:
            logs = {
                "loss_text": loss_text.item(),
                "loss_hr": loss_hr.item(),
                "n_text_tokens": n_text.item(),
                "n_hr_tokens": n_hr.item(),
            }
            # These will appear in TensorBoard / wandb
            for key, val in logs.items():
                self.log({key: val})
