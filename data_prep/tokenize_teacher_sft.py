"""Tokenize Teacher SFT samples with user-token-only loss masks.

Takes samples from build_teacher_sft_data.py and produces (input_ids, labels)
where labels is -100 everywhere EXCEPT tokens of user-role messages inside
the target session. Context-prefix tokens (including user turns from earlier
sessions) have labels=-100 — they are read-only context.

Also handles token-level truncation: if (prefix + target) exceeds max_len,
drop prefix sessions from the OLDEST end until it fits. Target is never
truncated — the loss tokens must be preserved exactly.

Usage:
    from tokenize_teacher_sft import SFTTokenizer
    tok = SFTTokenizer(model_name="Qwen/Qwen3-4B", max_len=131072)
    input_ids, labels = tok.tokenize_sample(sample)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from transformers import AutoTokenizer


@dataclass
class TokenizedChunk:
    """One tokenized message: its token ids and whether content is a loss target."""
    input_ids: list[int]
    is_target_user: bool  # True only for user messages in target session
    # len(input_ids) == prefix_len + content_len + end_len. Loss applies only
    # to content_len + end_len when is_target_user=True; role-prefix tokens
    # never carry loss (we want the model to learn content, not the role tag).
    prefix_len: int
    content_len: int
    end_len: int


class SFTTokenizer:
    """Wraps a HF tokenizer for Teacher SFT sample construction."""

    def __init__(self, model_name: str = "Qwen/Qwen3-4B", max_len: int = 131072):
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.max_len = max_len
        # Qwen uses ChatML-style markers; verify by encoding the specials.
        self._im_start = self.tok.encode("<|im_start|>", add_special_tokens=False)
        self._im_end = self.tok.encode("<|im_end|>", add_special_tokens=False)
        assert len(self._im_start) == 1 and len(self._im_end) == 1, (
            f"expected <|im_start|> and <|im_end|> to be single tokens, got "
            f"{self._im_start} and {self._im_end}"
        )

    def _encode_message(self, role: str, content: str,
                        is_target_user: bool) -> TokenizedChunk:
        """Encode one message into role-prefix + content + end-marker tokens.

        Format mirrors Qwen's ChatML template:
            <|im_start|>{role}\n{content}<|im_end|>\n
        """
        prefix_str = f"<|im_start|>{role}\n"
        content_str = content
        end_str = "<|im_end|>\n"

        prefix_ids = self.tok.encode(prefix_str, add_special_tokens=False)
        content_ids = self.tok.encode(content_str, add_special_tokens=False)
        end_ids = self.tok.encode(end_str, add_special_tokens=False)

        return TokenizedChunk(
            input_ids=prefix_ids + content_ids + end_ids,
            is_target_user=is_target_user,
            prefix_len=len(prefix_ids),
            content_len=len(content_ids),
            end_len=len(end_ids),
        )

    def _chunks_for_sample(self, sample: dict) -> tuple[list[TokenizedChunk],
                                                         list[TokenizedChunk]]:
        """Return (prefix_chunks, target_chunks) for one sample."""
        prefix_chunks = [
            self._encode_message(m["role"], m["content"], is_target_user=False)
            for m in sample["prefix_messages"]
        ]
        target_chunks = [
            self._encode_message(m["role"], m["content"],
                                 is_target_user=(m["role"] == "user"))
            for m in sample["target_messages"]
        ]
        return prefix_chunks, target_chunks

    @staticmethod
    def _sum_len(chunks: Iterable[TokenizedChunk]) -> int:
        return sum(len(c.input_ids) for c in chunks)

    def tokenize_sample(self, sample: dict) -> tuple[list[int], list[int], dict]:
        """Tokenize one sample.

        Returns: (input_ids, labels, info) where info has diagnostic counts.
        Target tokens are always preserved; prefix messages are dropped from
        the oldest end if necessary to fit max_len.
        """
        prefix_chunks, target_chunks = self._chunks_for_sample(sample)

        target_len = self._sum_len(target_chunks)
        if target_len > self.max_len:
            raise ValueError(
                f"target alone ({target_len} tokens) exceeds max_len "
                f"({self.max_len}); persona={sample['persona_id']} "
                f"ctx={sample['context_id'][:8]} sess={sample['session_idx']}"
            )

        budget = self.max_len - target_len
        kept_prefix: list[TokenizedChunk] = []
        used = 0
        # Keep newest prefix messages; drop oldest if over budget.
        for chunk in reversed(prefix_chunks):
            if used + len(chunk.input_ids) > budget:
                break
            kept_prefix.append(chunk)
            used += len(chunk.input_ids)
        kept_prefix.reverse()

        dropped = len(prefix_chunks) - len(kept_prefix)

        # Build flat input_ids + labels
        input_ids: list[int] = []
        labels: list[int] = []
        for chunk in kept_prefix + target_chunks:
            input_ids.extend(chunk.input_ids)
            if chunk.is_target_user:
                # Label only the content + end marker; mask the role prefix.
                labels.extend([-100] * chunk.prefix_len)
                labels.extend(chunk.input_ids[chunk.prefix_len:])
            else:
                labels.extend([-100] * len(chunk.input_ids))

        assert len(input_ids) == len(labels)

        info = {
            "total_len": len(input_ids),
            "target_len": target_len,
            "prefix_kept_msgs": len(kept_prefix),
            "prefix_dropped_msgs": dropped,
            "loss_token_count": sum(1 for x in labels if x != -100),
        }
        return input_ids, labels, info


if __name__ == "__main__":
    # Quick smoke test on one sample from 128k
    import json
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    sample_path = ROOT / "dynamic_usersim" / "outputs" / "teacher_sft_128k.jsonl"
    with sample_path.open("r", encoding="utf-8") as fh:
        sample = json.loads(fh.readline())

    tok = SFTTokenizer(max_len=131072)
    input_ids, labels, info = tok.tokenize_sample(sample)
    print(f"sample: persona={sample['persona_id']} ctx={sample['context_id'][:8]} "
          f"session_idx={sample['session_idx']}")
    print(f"info: {info}")
    # Show where labels are non-masked
    loss_positions = [i for i, L in enumerate(labels) if L != -100]
    print(f"first 10 loss positions: {loss_positions[:10]}")
    print(f"last 10 loss positions:  {loss_positions[-10:]}")
