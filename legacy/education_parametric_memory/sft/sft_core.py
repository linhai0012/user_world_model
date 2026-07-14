"""SFT core: dual-rate per-user LoRA training (direct SFT, no teacher) + choice-PPL scoring.

The lean-SFT realisation of Mode A (decision locked: no OPD teacher). A per-user LoRA is
trained to PRODUCE this learner's answers as raw completions of the question stem; at eval we
score each MCQ option as a continuation and read off the learner's predicted choice
distribution. Base = Qwen3-4B-Instruct-2507. Blackwell-safe (sdpa, no flash-attn). GPU-only.

Format (raw completion, per the handoff guardrail — never demand JSON):
    prompt = "Question: {stem}\nAnswer:"      target = " {answer_text}"
Train: causal-LM loss on the target tokens only. Eval: score " {option_text}" per option.

Dual-rate LoRA (Lin's R1b): slow = MLP {gate_proj,up_proj} r32 a64 lr1e-5 ; fast = Attn
{q,k,v,o} r16 a32 lr2e-4. Both adapters active, two optimizers.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
# Ranks env-overridable for the capacity sweep (does the base prior need a higher-rank adapter to
# override? CUE_SLOW_R/CUE_FAST_R; alpha tracks 2*r). Default = the locked dual-rate recipe.
_SLOW_R = int(os.environ.get("CUE_SLOW_R", "32"))
_FAST_R = int(os.environ.get("CUE_FAST_R", "16"))
SLOW = dict(r=_SLOW_R, lora_alpha=2 * _SLOW_R, target_modules=["gate_proj", "up_proj"])
FAST = dict(r=_FAST_R, lora_alpha=2 * _FAST_R, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
# v3: prepend the question's skill as a key, so the per-user LoRA can attach this learner's
# per-skill tendency to it (the skill is a property of the QUESTION, not learner info — like
# DKT/BKT keying on skill id). The learner still lives entirely in the weights.
SKILL_PROMPT = os.environ.get("CUE_SKILL_PROMPT", "0") == "1"


# --------------------------------------------------------------------- load
def load_base(dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=dtype, attn_implementation="sdpa"
    ).to("cuda")
    model.config.use_cache = False
    return tok, model


def _prompt(stem: str, skill: str | None = None) -> str:
    if SKILL_PROMPT and skill:
        return f"Skill: {skill}\nQuestion: {stem}\nAnswer:"
    return f"Question: {stem}\nAnswer:"


# --------------------------------------------------------------------- dual LoRA
def attach_dual_lora(base):
    cfg_slow = LoraConfig(lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", **SLOW)
    cfg_fast = LoraConfig(lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", **FAST)
    m = get_peft_model(base, cfg_slow, adapter_name="slow")
    m.add_adapter("fast", cfg_fast)
    m.base_model.set_adapter(["slow", "fast"])  # both active (additive)
    return m


def _split_params(m):
    slow, fast = [], []
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        (slow if ".slow." in n else fast if ".fast." in n else []).append(p)
    return slow, fast


def _init_adapter_state(m) -> dict:
    return {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}


def _reset_adapters(m, init_state: dict) -> None:
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.requires_grad and n in init_state:
                p.copy_(init_state[n])


# --------------------------------------------------------------------- training
def _sample_ids(tok, stem, answer_text, skill=None):
    p = tok(_prompt(stem, skill), add_special_tokens=False).input_ids
    a = tok(" " + answer_text, add_special_tokens=False).input_ids
    a = a + [tok.eos_token_id] if tok.eos_token_id is not None else a
    return p + a, [-100] * len(p) + a


def train_snapshots(tok, m, init_state, stream, snapshots, out_dir: Path,
                    epochs_per_seg=3, slow_lr=1e-5, fast_lr=2e-4, grad_clip=1.0, log=print):
    """Incrementally SFT the dual LoRA through `stream` (list of (stem, answer_text)); save the
    adapter at each snapshot k (= trained on rounds 0..k). snapshot 0 is the base (no adapter)."""
    _reset_adapters(m, init_state)
    slow_p, fast_p = _split_params(m)
    opt_s = torch.optim.AdamW(slow_p, lr=slow_lr)
    opt_f = torch.optim.AdamW(fast_p, lr=fast_lr)
    saved = {}
    prev = 0
    for snap in [s for s in snapshots if s > 0]:
        seg = stream[prev:snap]
        m.train()
        for _ in range(epochs_per_seg):
            for stem, ans in seg:
                ids, labels = _sample_ids(tok, stem, ans)
                ids_t = torch.tensor([ids], device="cuda")
                lab_t = torch.tensor([labels], device="cuda")
                out = m(input_ids=ids_t, labels=lab_t)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(slow_p, grad_clip)
                torch.nn.utils.clip_grad_norm_(fast_p, grad_clip)
                opt_s.step(); opt_f.step()
                opt_s.zero_grad(); opt_f.zero_grad()
        snap_dir = out_dir / f"snap{snap}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        m.save_pretrained(snap_dir, selected_adapters=["slow", "fast"])
        saved[snap] = str(snap_dir)
        log(f"    snapshot {snap}: trained on {snap} rounds -> {snap_dir.name}")
        prev = snap
    return saved


# --------------------------------------------------------------------- scoring (choice-PPL)
@torch.no_grad()
def _mean_logprob(tok, m, stem: str, cont_text: str, skill: str | None = None) -> float:
    p = tok(_prompt(stem, skill), add_special_tokens=False).input_ids
    c = tok(" " + cont_text, add_special_tokens=False).input_ids
    ids = torch.tensor([p + c], device="cuda")
    logits = m(input_ids=ids).logits[0]
    start = len(p) - 1
    lp = F.log_softmax(logits[start:start + len(c)].float(), dim=-1)
    tok_lp = lp[range(len(c)), torch.tensor(c, device="cuda")]
    return float(tok_lp.mean().item())


@torch.no_grad()
def score_mcq(tok, m, question: dict) -> dict:
    """Return {option_logprobs, p_correct_raw, p_correct} for one MCQ via choice-perplexity.
    p_correct = softmax over options' mean-token-logprob, restricted to the correct option."""
    opts = question["options"]
    correct_id = next((o["id"] for o in opts if o.get("is_correct")), opts[0]["id"])
    skill = question.get("skill_id", "").split("__")[-1] or None
    lps = {o["id"]: _mean_logprob(tok, m, question["stem"], o["text"], skill) for o in opts}
    keys = list(lps)
    z = torch.logsumexp(torch.tensor([lps[k] for k in keys]), dim=0).item()
    probs = {k: math.exp(lps[k] - z) for k in keys}
    return {
        "option_logprobs": {k: round(v, 4) for k, v in lps.items()},
        "p_correct_raw": round(lps[correct_id], 4),         # raw mean-token logprob; NOT a prob
        "p_correct": round(probs[correct_id], 4),            # softmax prob of correct option
    }


# --------------------------------------------------------------------- adapter load for scoring
def with_adapter(tok, base, snap_dir: str | None):
    """Return a model to score with: base alone (snap_dir None) or base+adapter loaded."""
    if snap_dir is None:
        return base
    slow = Path(snap_dir) / "slow"
    m = PeftModel.from_pretrained(base, str(slow), adapter_name="slow")
    m.load_adapter(str(Path(snap_dir) / "fast"), adapter_name="fast")
    m.base_model.set_adapter(["slow", "fast"])
    m.eval()
    return m
