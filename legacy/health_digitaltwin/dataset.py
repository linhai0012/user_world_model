"""
Dataset for omnimodal Digital Twin training.

Each sample is a causal-LM sequence:
  [input_tokens]  [output_tokens]
       ↑ masked        ↑ loss computed here

V3: output contains text tokens + state tokens (<fatigue_3> etc.)
The weighted loss uses is_structured_token to separate text vs structured tokens.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from config import (
    TOKEN_TS_START, TOKEN_TS_END,
    TOKEN_STATE_START, TOKEN_STATE_END,
    HR_TOKEN_PREFIX, WELLNESS_FIELDS,
    ACTIVITY_BEGIN_TOKENS, ACTIVITY_END_TOKENS,
)

log = logging.getLogger(__name__)

# HR tokens: <hr_*>, plus structural delimiters and activity markers
_HR_TOKEN_PREFIXES = (HR_TOKEN_PREFIX,)
# Wellness tokens: <fatigue_*>, <mood_*>, etc.
_WELLNESS_TOKEN_PREFIXES = tuple(f"<{field}_" for field in WELLNESS_FIELDS)


class DigitalTwinDataset(Dataset):
    """
    Loads pre-built JSONL samples and tokenizes them for causal LM training.

    Each JSONL line must have:
      - full_sequence: str  (the complete input+output string)
      - input_text: str     (the input portion, used to compute the mask)
      - output_text: str    (used only for reference)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 768,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: list[dict[str, Any]] = []

        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        # Cache HR-token vs wellness-token IDs separately for the
        # three-component CE loss (text / HR / wellness).
        self._hr_token_ids: set[int] = set()
        self._wellness_token_ids: set[int] = set()
        for tid in range(len(tokenizer)):
            token_str = tokenizer.convert_ids_to_tokens(tid)
            if not token_str:
                continue
            if any(token_str.startswith(p) for p in _HR_TOKEN_PREFIXES):
                self._hr_token_ids.add(tid)
            elif any(token_str.startswith(p) for p in _WELLNESS_TOKEN_PREFIXES):
                self._wellness_token_ids.add(tid)

        # HR structural delimiters + activity markers belong to the HR class
        activity_tokens = list(ACTIVITY_BEGIN_TOKENS.values()) + list(ACTIVITY_END_TOKENS.values())
        for tok in [TOKEN_TS_START, TOKEN_TS_END] + activity_tokens:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None:
                self._hr_token_ids.add(tid)
        # State delimiters belong to the wellness class
        for tok in [TOKEN_STATE_START, TOKEN_STATE_END]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None:
                self._wellness_token_ids.add(tid)

        log.info(f"Loaded {len(self.samples)} samples from {path.name} "
                 f"({len(self._hr_token_ids)} HR tokens, "
                 f"{len(self._wellness_token_ids)} wellness tokens)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        full_seq = sample["full_sequence"]
        input_text = sample["input_text"]

        # Tokenize the full sequence
        full_enc = self.tokenizer(
            full_seq,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Tokenize input-only to determine where output starts
        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )
        input_len = len(input_enc["input_ids"])

        # Build labels: -100 for input tokens + padding, actual ids for output
        labels = input_ids.clone()
        labels[:input_len] = -100
        labels[attention_mask == 0] = -100

        # Two binary masks marking HR-class and wellness-class tokens
        # in the OUTPUT portion. Tokens in the INPUT are already masked
        # by labels=-100 and won't affect loss regardless of these flags.
        is_hr_token = torch.zeros_like(input_ids)
        is_wellness_token = torch.zeros_like(input_ids)
        for i in range(input_len, len(input_ids)):
            tid = input_ids[i].item()
            if tid in self._hr_token_ids:
                is_hr_token[i] = 1
            elif tid in self._wellness_token_ids:
                is_wellness_token[i] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "is_hr_token": is_hr_token,
            "is_wellness_token": is_wellness_token,
        }


class DigitalTwinDataCollator:
    """Simple collator that stacks pre-padded tensors."""

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            key: torch.stack([f[key] for f in features])
            for key in features[0].keys()
        }
