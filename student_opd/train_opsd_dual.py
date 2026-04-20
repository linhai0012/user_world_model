"""OPSD-C variant of train_opd_dual — teacher scores student rollout with
GT already in its attention context as a prior user turn.

Only ONE difference vs train_opd_dual.py: the teacher prefix. Sanity check
(§sanity_check_gt_injection.py) validated:
  A natural       P(GT)=0.566
  B sys-aug       P(GT)=0.887    ΔP(correct−mismatched) = +0.321
  C prior-turn    P(GT)=0.935    ΔP(correct−mismatched) = +0.384  ← strongest, semantic
  D C+mismatchGT  P(GT)=0.551    below A (not shallow-copy)
  E B+mismatchGT  P(GT)=0.566    ≈ A

Chose placement C (prior-turn): teacher gets strongest GT-informed signal,
and the signal is provably semantic (wrong GT *hurts*, not helps — so
teacher is reading GT, not just pattern-matching recent tokens).

Teacher input (OPSD):
    [K=3 history] + <|im_start|>user\\n{GT}<|im_end|>\\n + <|im_start|>user\\n + S
Student input (unchanged from R1b):
    [demo + chatbot_prev] + <|im_start|>user\\n + S
KL, rollout, dual-LoRA (s32f16), slow/fast LR, all unchanged.

Disk layout: use a distinct output-dir name (e.g. ..._opsd_step) so OPSD
runs don't clobber R1b checkpoints.

Usage (Isambard, mirrors run_round1b_train_interactive.sh):
    python dynamic_usersim/student_opd/train_opsd_dual.py \\
        --persona-id 4 \\
        --teacher-path $SCRATCHDIR/P-OPSD/teacher_sft_128k_k3/final \\
        --data-path   $HOME/P-OPSD/dynamic_usersim/outputs/opd_128k_pid4_k3.jsonl \\
        --output-dir  $SCRATCHDIR/P-OPSD/student_lora/lora_pid4_dual_s32f16_opsd_ep1_r3teacher \\
        --student-ctx demo --slow-lr 5e-5 --fast-lr 2e-4 --save-every 200 \\
        --save-total-limit 0
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sibling train_opd_dual importable and reuse every helper / main()
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import train_opd_dual  # noqa: E402  (sibling script, adds args/main/etc.)


# ---------------------------------------------------------------------------
# OPSD-C: teacher_prefix has GT injected as a completed prior user turn
# before the trailing "<|im_start|>user\n" where student rollout is scored.
# ---------------------------------------------------------------------------
def build_teacher_prefix_opsd(sample: dict, tokenizer) -> list[int]:
    ids: list[int] = []
    for m in sample["history_messages"]:
        s = f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        ids.extend(tokenizer.encode(s, add_special_tokens=False))
    # Inject GT as a completed prior user turn.
    gt_msg = f"<|im_start|>user\n{sample['user_response']}<|im_end|>\n"
    ids.extend(tokenizer.encode(gt_msg, add_special_tokens=False))
    # Open a NEW user turn — this is the position student's rollout scores against.
    ids.extend(
        tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    )
    return ids


# Monkey-patch the helper that the training loop uses. Single line of effect,
# zero risk of drifting from upstream R1b recipe otherwise.
train_opd_dual.build_teacher_prefix = build_teacher_prefix_opsd


if __name__ == "__main__":
    # Re-use argparse, LoRA setup, rollout, KL, save logic verbatim.
    train_opd_dual.main()
