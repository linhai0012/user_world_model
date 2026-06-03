"""Lazy-tokenizing Dataset + padding collator for Teacher SFT.

Dataset reads the JSONL produced by build_teacher_sft_data.py into memory
(text-only; 1172 records for 128k is small), and tokenizes on __getitem__
using SFTTokenizer from data_prep. Pre-built labels have -100 everywhere
except target-session user tokens.

Collator pads a batch of variable-length samples to the batch max length,
padding input_ids with pad_token_id, labels with -100, attention_mask with 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

# Import tokenizer from data_prep
_DP = Path(__file__).resolve().parents[1] / "data_prep"
if str(_DP) not in sys.path:
    sys.path.insert(0, str(_DP))
from tokenize_teacher_sft import SFTTokenizer  # noqa: E402


class TeacherSFTDataset(Dataset):
    """Lazy-tokenizing dataset over progressive-context SFT samples.

    Each item: {"input_ids": LongTensor[L], "labels": LongTensor[L],
                "attention_mask": LongTensor[L]}
    """

    def __init__(self, jsonl_path: str | Path, tokenizer: SFTTokenizer):
        self.samples: list[dict] = []
        with Path(jsonl_path).open("r", encoding="utf-8") as fh:
            for line in fh:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_ids, labels, _info = self.tokenizer.tokenize_sample(self.samples[idx])
        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)
        attn_t = torch.ones_like(input_ids_t)
        return {
            "input_ids": input_ids_t,
            "labels": labels_t,
            "attention_mask": attn_t,
        }


class PadToBatchMaxCollator:
    """Right-pad a batch to max length in the batch.

    Pads:
      input_ids     -> pad_token_id
      labels        -> -100
      attention_mask -> 0
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].size(0) for f in features)
        batch: dict[str, list[torch.Tensor]] = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
        }
        for f in features:
            L = f["input_ids"].size(0)
            pad = max_len - L
            if pad == 0:
                batch["input_ids"].append(f["input_ids"])
                batch["labels"].append(f["labels"])
                batch["attention_mask"].append(f["attention_mask"])
                continue
            batch["input_ids"].append(
                torch.cat([f["input_ids"],
                           torch.full((pad,), self.pad_token_id, dtype=torch.long)])
            )
            batch["labels"].append(
                torch.cat([f["labels"],
                           torch.full((pad,), -100, dtype=torch.long)])
            )
            batch["attention_mask"].append(
                torch.cat([f["attention_mask"],
                           torch.zeros(pad, dtype=torch.long)])
            )
        return {k: torch.stack(v, dim=0) for k, v in batch.items()}


def length_bucketed_indices(dataset: TeacherSFTDataset, approximate: bool = True) -> list[int]:
    """Return indices sorted by sample length (ascending).

    Useful for length-bucketed batching to reduce padding waste. Uses text-level
    msg count as a cheap proxy when approximate=True (avoids tokenizing twice).
    """
    keys: list[tuple[int, int]] = []
    for i, s in enumerate(dataset.samples):
        if approximate:
            # message-count proxy: roughly linear in token count
            m = len(s["prefix_messages"]) + len(s["target_messages"])
            keys.append((m, i))
        else:
            ids, _, _ = dataset.tokenizer.tokenize_sample(s)
            keys.append((len(ids), i))
    keys.sort()
    return [i for _, i in keys]
