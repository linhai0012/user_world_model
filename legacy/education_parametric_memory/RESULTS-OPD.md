# Mode A — OPD / privileged-distillation ablations (2026-06-20, H200)

Follow-up to [`RESULTS.md`](RESULTS.md). The lean-SFT run left a transfer ceiling (LoRA memorises
train answers, choice-acc 1.0, but transfers per-skill mastery only weakly). The cut recipe was
OPD — **privileged distillation**: a teacher sees the learner's hidden θ (mastery + held
misconception); the per-user student LoRA sees stem-only and distills the teacher's option
distribution, forced to internalise "misconception → mass on the consistent distractor" into the
weights. This file records whether OPD lifts the ceiling.

**Setup (shared).** Base Qwen3-4B-Instruct-2507 · dual-rate LoRA (slow MLP r32/α64/1e-5; fast Attn
r16/α32/2e-4) · substrate v4 (spread population, 24 personas, 104-round streams, eval 39 / calib 30)
· transfer probe = train HARD on the full 104-round stream, score held-out eval @ snapshot 104.
Metrics: TRAIN/EVAL choice-acc (memorise / argmax transfer) · **mast_corr** = pearson(p_correct, true
mastery) · **misc_hit** = on items the learner got WRONG, does argmax pick the learner's OWN
distractor · **revert→✓** = on WRONG items, does argmax fall back to the correct answer (base prior).

---

## Run A — oracle-teacher OPD vs hard SFT (ceiling_opd.py, 3 personas, 15 epochs)

Teacher = the generator's EXACT categorical g(θ,q) (analytic, verified byte-matching the sampler).
Loss = soft-CE (forward KL) of the student's 4-option softmax toward g.

| arm | train_acc | eval_acc | **mast_corr** | **misc_hit** | revert→✓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.506 | 0.573 | −0.020 | 0.000 | 0.832 |
| **hard** | 1.000 | 0.564 | **0.189** | **0.096** | 0.545 |
| **opd (oracle)** | 0.609 | **0.684** | 0.094 | 0.039 | **0.863** |

**Finding (nuanced, negative for distribution-distillation):** oracle-OPD does NOT fix transfer — it
moves the WRONG way on the metrics that matter.
1. **mast_corr DOWN** (0.189 → 0.094) and **misc_hit DOWN** (0.096 → 0.039) vs hard.
2. **revert→✓ UP** (0.545 → 0.863): when the learner errs, OPD predicts the correct answer *more*.
3. **eval_acc UP** (0.564 → 0.684) and no memorisation (train 0.609 vs 1.0): OPD is a *better overall
   correctness predictor* — but by predicting "correct," which the learner is 57–66 % of the time.

**Why.** g(θ,q) is **correct-dominated**: on ~96.6 % of items it has no active misconception
distractor (the held tag matches a distractor in only 86 / 2496 ≈ 3.4 % of rounds), so g ≈
"mastery-weighted correct + uniform wrong." Distilling it teaches *calibration toward the correct
answer* — which the base already knows. The learner-specific signal lives in the **realized errors**,
and **hard SFT trains on the realized chosen option** (the actual wrong answer), so it captures the
misconception better. Distribution-distillation washes that error signal out.

**Implication.** "Give the model a *why*" still holds, but the *why* must be **error-focused** (the
realized mistake / its misconception), not the full generating distribution. → Run B (full oracle-OPD)
skipped as low-value (would only reproduce this negative at scale). Pivot to a target ablation.

Timing: HARD ~225 s/persona (15 ep), oracle-OPD ~800 s/persona (4× forward, latency-bound), ~9.7 GB.

---

## Run B — multi-arm target ablation (ceiling_multi.py, 3 personas, 6 epochs)

Four arms, identical LoRA/epochs/substrate; only the per-item TARGET differs. Tests whether an
error-carrying target beats hard and opd, isolating loss-mechanism (token vs option space) from
target content.

| arm (target) | train_acc | eval_acc | **mast_corr** | **misc_hit** | revert→✓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.362 | 0.538 | 0.067 | 0.000 | 0.773 |
| **hard** (token-CE realized) | 0.997 | 0.521 | **0.152** | 0.113 | 0.564 |
| realized_opt (one-hot realized) | 1.000 | 0.573 | 0.015 | 0.059 | 0.791 |
| opd (g) | 0.606 | **0.684** | 0.097 | 0.076 | 0.845 |
| **hybrid** (0.5·g + 0.5·realized) | 0.929 | 0.564 | 0.038 | **0.190** | 0.603 |

**The two metrics split — the central nuance:**
- **mast_corr (θ-recovery, the calibration headline): hard wins (0.152); no recipe beats it.** opd,
  realized_opt, hybrid all LOWER it. Distillation does not improve mastery calibration.
- **misc_hit (catching the learner's specific misconception): hybrid wins (0.190, ~2× hard)** while
  keeping revert→✓ low (0.603). Blending the *realized error* into the soft target is what helps —
  not the full distribution (opd) and not the one-hot alone (realized_opt).
- **opd is worst for our purpose**: highest eval_acc (0.684) but by reverting to correct (0.845) and
  catching few misconceptions — it predicts "the learner is right," which they usually are.

**Reading.** Changing the distillation TARGET does not lift the θ-recovery ceiling; hard SFT stays
best for mastery calibration. Hybrid's realized-error blend is the only thing that improves
misconception-argmax. The two objectives (calibrate mastery vs identify the misconception) trade off.
n=3, noisy (persona_11's high-mastery few-error eval dominates mast_corr). → validate hybrid vs hard
at scale (24 personas) to denoise; see Run B-scale.

## Run B-scale — hybrid vs hard, full 24-persona run (run_all_opd.py CUE_TARGET=hybrid) ✅ **HEADLINE**

**At 24-persona scale the small-n ceiling verdict REVERSES: hybrid OPD beats hard SFT on ALL THREE
metrics, monotonically, and ~triples the headline mastery recovery.** (Hard column = the v4 run; its
mastery_corr snap52 0.175 matches RESULTS.md exactly, so the baseline is correct.)

| metric | snap | hard (v4) | **hybrid OPD** | Δ (B−A) |
| --- | ---: | ---: | ---: | ---: |
| **mastery_corr** ↑ | 13 | 0.122 | **0.284** | +0.162 |
| | 26 | 0.167 | **0.402** | +0.236 |
| | 52 | 0.175 | **0.476** | +0.301 |
| | 104 | 0.159 | **0.494** | +0.335 |
| **theta_MSE** ↓ | 52 | 0.102 | **0.057** | −0.044 |
| | 104 | 0.130 | **0.051** | −0.079 (−61%) |
| **binary_NLL** ↓ | 52 | 0.880 | **0.743** | −0.137 |
| | 104 | 0.845 | **0.651** | −0.194 |

hybrid mastery_corr **rises monotonically to 0.494** (hard peaks 0.175 then declines); v4's old
headline was 0.18 — hybrid nearly **triples** it. Even binary NLL — the "underpowered" metric base won
on in RESULTS.md — now favours hybrid.

**Why the small-n ceiling tests (Run A/B) looked negative:** they used *within-persona* pearson
(39 items each, averaged, n=3, dominated by a high-mastery low-error persona). The full run uses the
*pooled* corr across 24×39=936 points — the SAME metric RESULTS.md/analyze.py report. Apples-to-apples,
hybrid wins decisively. (Sanity: hybrid p_correct sd=0.205 — not degenerate; **between-persona corr of
mean p_correct vs mean mastery = 0.939** — the model almost perfectly recovers each learner's ability.)

**Mechanism (the privileged-information point, clean):** the hybrid target carries `0.5·g`, and g's
`P(correct)=mastery` hands the student the learner's **true correct-probability as a soft target**,
whereas hard SFT only sees a **sampled binary outcome**. A soft probability carries far more bits than
one Bernoulli draw, so the student recovers mastery far more efficiently — and generalises it to
held-out items (real amortized KT, not memorisation). This is exactly what privileged distillation is
for, and it shows the lean-SFT transfer ceiling was an **information/recipe limit, not student
capacity.**

**Honesty gate on the claim:** the g teacher uses the **authored θ** (oracle privilege). The result is
legitimate (it transfers to held-out items) but its DEPLOYABILITY rests on whether a teacher *without*
oracle mastery (a frozen LLM reading a misconception/level card) reproduces it → Run C, the crux. Also
running pure-opd(g) at scale to isolate whether the mastery_corr gain is the g-distribution alone or
needs the realized-error blend (the blend was what won misc_hit at n=3).

Cost: full hybrid run 102 min on 1×H200 (24 personas × ~200 s + 21 min shared distill), ~11 GB.

## Run B-iso — pure opd(g) vs hybrid, 24-persona ✅

`CUE_TARGET=g` (no realized blend) at scale isolates the g-distribution's contribution.

| metric | snap | hard (v4) | hybrid | **pure g** |
| --- | ---: | ---: | ---: | ---: |
| mastery_corr ↑ | 52 | 0.175 | 0.476 | **0.470** |
| | 104 | 0.159 | 0.494 | **0.570** |
| theta_MSE ↓ | 104 | 0.130 | 0.051 | 0.054 |

**The mastery_corr gain is the g-distribution, NOT the realized blend.** Pure g matches/slightly
beats hybrid on mastery_corr (0.570 vs 0.494 at snap104; the realized one-hot adds a little Bernoulli
noise back, marginally diluting calibration) and ties on theta_MSE. The realized blend's value is
**misc_hit** (catching the learner's *specific* distractor: hybrid 0.190 ≈ 2× g at n=3). Clean
decomposition:
- **g (soft mastery target) → mastery recovery** (the headline mastery_corr / theta_MSE).
- **realized-error blend → misconception-argmax** (misc_hit), at a tiny mastery_corr cost.

Recommendation depends on the goal: **pure g** for the cleanest mastery calibration, **hybrid** if the
demo also wants to *name the specific misconception*. Both ride on g's privileged mastery → Run C is
the crux.

## Run C — LLM-teacher hybrid at scale (run_all_opd_llm.py) ✅

Deployable teacher = frozen base + privileged "predict this student's answer" prompt (held
misconception + low/med/high mastery bucket), target 0.5·llm + 0.5·realized. No oracle θ — only a
coarse level bucket + the misconception text. 24 personas, same eval. (106.9 min, 0 failures.)

**Result: a real but MODEST gain — recovers only ~15–25 % of the oracle headroom.**

## FINAL — four-way comparison (held-out eval, pooled mastery_corr, ↑ better)

| snap | hard (lean SFT) | **LLM-teacher** (deployable) | oracle-hybrid | oracle-g |
| ---: | ---: | ---: | ---: | ---: |
| 13 | 0.122 | 0.126 | 0.284 | 0.363 |
| 26 | 0.167 | 0.192 | 0.402 | 0.426 |
| 52 | 0.175 | **0.249** | 0.476 | 0.470 |
| 104 | 0.159 | **0.216** | 0.494 | 0.570 |

theta_MSE (↓ better) @104: hard 0.130 · LLM-teacher 0.108 · oracle-hybrid 0.051 · oracle-g 0.054.

**The complete, honest arc:**
1. **Lean SFT** has a transfer ceiling — mastery_corr ~0.16 (memorises train answers, weak transfer).
2. **Oracle privileged distillation (g / hybrid)** lifts it ~3× to **0.49–0.57** → the ceiling is an
   **information/recipe limit, not student capacity**. But the teacher distills the *authored* θ, so
   this is an **upper bound**, not a deployable number. (Decomposition: g's soft mastery target drives
   mastery_corr; the realized-error blend adds misconception-argmax / misc_hit.)
3. **Deployable LLM-teacher** (no oracle θ; frozen base + low/med/high bucket + misconception card)
   beats hard but only modestly — **mastery_corr 0.16 → 0.22 (+35 % rel)**, recovering **~15–25 % of
   the oracle headroom**. The bulk of the oracle gain needs the *exact* mastery, which a coarse bucket
   can't convey, and the base LLM still leans toward the correct answer even when told the misconception.

**Implication for the paper & Cue.** Privileged distillation genuinely lifts the parametric ceiling,
but deployable gains are bounded by how much learner-state the teacher can access. The clean bridge:
**feed the teacher the BKT baseline's *continuous* mastery instead of a 3-level bucket** — Cue already
computes it, and finer mastery is exactly what closes the oracle gap. This ties the three backends
together (BKT mastery → LLM teacher → parametric LoRA) and is the natural next experiment.

**Recipe recommendation:** for the offline bundle, **hybrid OPD** (mastery_corr 0.49, + misc_hit) is
the headline-strongest *honest-oracle* result to report alongside the lean-SFT baseline and the
deployable LLM-teacher number, with the privilege gap stated plainly. Pure-g is marginally better on
mastery_corr alone; hybrid is preferred for also naming the misconception.

**Artifacts (NOT pushed):** predictions_{v4,opd_g,opd_hybrid,opd_llm}/ + bundles re-importable via the
existing honesty gate; `parametric_offline/sft/{run_all_opd.py,run_all_opd_llm.py,ceiling_multi.py,
ceiling_opd.py,ceiling_opd_llm.py,compare_opd.py}`. All four full runs: ~100–107 min each on 1×H200.

## Run C (LLM-teacher OPD) — ceiling_opd_llm.py  *(PENDING / deprioritised)*

Realistic deployable teacher = frozen base + privileged "predict this student's answer" misconception
prompt. Given Run A, expected ≤ oracle-OPD (a base teacher reverts to correct even harder); run as a
confirmation of the deployable-recipe ceiling if time permits.
