"""
Model setup for Digital Twin training.

1. Load Qwen3-4B base model
2. Extend tokenizer with HR + state + structural tokens
3. Resize model embeddings
4. Smart initialization for new token embeddings
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import HR_BIN_MIN, HR_BIN_MAX, WELLNESS_FIELDS, USER_TOKENS, get_all_special_tokens
from train_config import TrainConfig

log = logging.getLogger(__name__)


def setup_tokenizer(cfg: TrainConfig) -> AutoTokenizer:
    """Load tokenizer and extend with special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokens_path = cfg.data_path / cfg.special_tokens_file
    if tokens_path.exists():
        special_tokens = json.loads(tokens_path.read_text())
    else:
        special_tokens = get_all_special_tokens()

    num_added = tokenizer.add_tokens(special_tokens, special_tokens=True)
    log.info(f"Added {num_added} special tokens (total vocab: {len(tokenizer)})")

    return tokenizer


def setup_model(cfg: TrainConfig, tokenizer: AutoTokenizer) -> AutoModelForCausalLM:
    """Load model, resize embeddings, initialize new tokens."""

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        dtype=dtype_map.get(cfg.torch_dtype, torch.bfloat16),
        attn_implementation=cfg.attn_implementation,
    )

    # ── Resize embeddings ────────────────────────────────────────────────
    original_vocab_size = model.config.vocab_size
    new_vocab_size = len(tokenizer)

    m = cfg.pad_vocab_to_multiple_of
    padded_size = ((new_vocab_size + m - 1) // m) * m

    model.resize_token_embeddings(padded_size)
    log.info(
        f"Resized embeddings: {original_vocab_size} → {padded_size} "
        f"(+{padded_size - original_vocab_size} slots, "
        f"{new_vocab_size - original_vocab_size} used)"
    )

    # ── Smart initialization ───────────────────────────────────────────
    _initialize_embeddings(model, tokenizer, cfg, original_vocab_size)

    # ── Gradient checkpointing ───────────────────────────────────────────
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        log.info("Gradient checkpointing enabled")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {total / 1e9:.2f}B total, {trainable / 1e9:.2f}B trainable")

    return model


def _init_token_range(
    embed_weight, lm_head_weight, tokenizer,
    token_format: str, lo: int, hi: int,
    embed_mean, embed_std, original_vocab_size: int,
    scale: float = 0.5,
):
    """Initialize a range of ordinal tokens with linear gradient."""
    count = 0
    for val in range(lo, hi + 1):
        token_str = token_format.format(val)
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id is None or token_id < original_vocab_size:
            continue

        t = (val - lo) / max(hi - lo, 1)  # normalized [0, 1]
        init_vec = embed_mean + (t - 0.5) * embed_std * scale
        noise = torch.randn_like(init_vec) * embed_std * 0.01
        embed_weight[token_id] = init_vec + noise
        lm_head_weight[token_id] = init_vec + noise
        count += 1
    return count


def _initialize_embeddings(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    cfg: TrainConfig,
    original_vocab_size: int,
):
    """Initialize all new token embeddings with ordinal-aware values."""
    embed_weight = model.get_input_embeddings().weight.data
    lm_head_weight = model.get_output_embeddings().weight.data

    orig_embed = embed_weight[:original_vocab_size]
    embed_mean = orig_embed.mean(dim=0)
    embed_std = orig_embed.std(dim=0)

    if cfg.hr_embed_init == "linear":
        log.info("Initializing new token embeddings with linear interpolation")

        # HR tokens: <hr_40> .. <hr_200>
        n_hr = _init_token_range(
            embed_weight, lm_head_weight, tokenizer,
            "<hr_{}>", HR_BIN_MIN, HR_BIN_MAX,
            embed_mean, embed_std, original_vocab_size,
        )
        log.info(f"  HR tokens: {n_hr} initialized")

        # State tokens: <fatigue_1>..<fatigue_5>, <mood_1>..<mood_5>, etc.
        n_state = 0
        for field, (lo, hi) in WELLNESS_FIELDS.items():
            n = _init_token_range(
                embed_weight, lm_head_weight, tokenizer,
                f"<{field}" + "_{}>", lo, hi,
                embed_mean, embed_std, original_vocab_size,
            )
            n_state += n
        log.info(f"  State tokens: {n_state} initialized")

        # User tokens: random init with larger variance (need to learn distinct representations)
        n_user = 0
        for pid, tok in USER_TOKENS.items():
            token_id = tokenizer.convert_tokens_to_ids(tok)
            if token_id is not None and token_id >= original_vocab_size:
                noise = torch.randn_like(embed_mean) * embed_std * 0.1
                embed_weight[token_id] = embed_mean + noise
                lm_head_weight[token_id] = embed_mean + noise
                n_user += 1
        log.info(f"  User tokens: {n_user} initialized (mean + random noise)")

    elif cfg.hr_embed_init == "mean":
        for bpm in range(HR_BIN_MIN, HR_BIN_MAX + 1):
            token_id = tokenizer.convert_tokens_to_ids(f"<hr_{bpm}>")
            if token_id is not None and token_id >= original_vocab_size:
                noise = torch.randn_like(embed_mean) * embed_std * 0.02
                embed_weight[token_id] = embed_mean + noise
                lm_head_weight[token_id] = embed_mean + noise
        log.info("Initialized embeddings at mean + small noise")

    else:
        log.info("Using default random initialization for new embeddings")
