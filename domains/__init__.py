"""Per-domain code. One subpackage per domain (general / health / education).

The domains are deliberately INDEPENDENT: non-overlapping users, unrelated tasks. "Unified"
means the same recipe instantiated three times, not one model over pooled data — so nothing
here imports across domains. Anything genuinely shared lives in `common/`.
"""
