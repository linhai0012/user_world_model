# START HERE — Lin Hai's side of **Cue** (emnlp2026_demo2)

> Orientation for the Claude Code process working this repo on **Lin Hai's behalf**.
> Lin owns the **parametric / user-modelling** part. Read this first, then the two contract
> docs in §3. Written 2026-06-15 from the repo's own docs + Lin's prior research repos.
>
> **中文一句话**:这个仓库是第二个 demo(Cue,主动式辅导)。我们负责的是 **Mode A
> 参数化后端(per-user LoRA)的「离线产物」**——线上 app 由对方团队搭好了,我们只交一个
> **artifact bundle**。UWM 研究继续在那个 cephfs 文件夹做,**不在这里**(见 §12)。

---

## 0. What you are here to do (TL;DR)

Cue's headline experiment swaps the **user-model backend** (baseline BKT / **Mode A
parametric = ours** / Mode B agentic LLM-memory) behind one fixed engine, to compare
learning efficiency. **The live demo has NO GPU.** So our entire job is an **offline GPU
harness** that trains/serves real per-user LoRAs over a fixed shared persona stream and
**exports an artifact bundle**; the demo *replays* it. We build **no live/serving infra** —
the friend's team already built the live side (providers + a mock service, see §5).

**Deliverable = one bundle conforming to `docs/backend/parametric-bundle-schema.md`.** That's it.

## 1. What Cue is (1 paragraph)

A **proactive** tutoring companion (sibling to "colearn", the *other* demo). The agent holds
a learner model and **reaches out** in a chat thread with the next thing to study — one of
four moves (**Resurface / Advance / Repair / Engage**), each with an evidence-grounded *why*;
question → grading → feedback happen inline. One engine, three swappable `UserModel`
backends. Scope is a **demo paper** (the contribution is the interactive proactive system +
a principled evaluation), **not** a new user-modelling algorithm — it reuses BKT, the CoELoVE
framing, and **Lin Hai's parametric user-modelling research** as the engine.

## 2. Our responsibility: Mode A (`parametric`) — the OFFLINE half only

| Side | Who | What |
| --- | --- | --- |
| **Live app** (engine, UI, baseline + agentic backends, the `parametric` backend code, the 3 providers, the mock service) | **friend's team — ALREADY BUILT** | GPU-free; replays our bundle |
| **Offline GPU harness + the bundle** | **us (Lin)** | train per-user/shared LoRAs, score eval items at snapshots, export the bundle |

The GPU train+serve runs **offline only** (Lin's rig), behind a `ParametricProvider` seam.
The live demo uses `stub` → `offline_replay` of our precomputed bundle — **never** live GPU.
**Paper framing (we're coauthors):** the parametric backend + our offline results are
presented as a real first-class component (they are real); the stub/replay is an internal
deployment detail not described. **Hard gate:** before submission the live link must replay a
**real** bundle, never the stub.

## 3. Read these two first — the contract

1. **`docs/handoff-to-linhai.md`** — what the team needs from us, in prose (the reverse of
   our earlier handoff).
2. **`docs/backend/parametric-bundle-schema.md`** — **THE CONTRACT.** Field-level JSON for the
   5 bundle files. Build against this exactly.

Then skim **`docs/backend/05-user-model.md`** (§A = the parametric backend + provider seam +
the fairness/calibration contracts) and **`docs/00-overview.md`** (what Cue is, the honest-
scoping discipline: "our model is `A_t` not `H_t`").

## 4. The deliverable — the artifact bundle (5 files)

Weights **never** ship; the bundle carries predictions + curves + metadata + cost.

| File | Content |
| --- | --- |
| `manifest.json` | header + **the shared contract**: `base_model`, `lora_recipe`, `raw_completion:true`, `calibration{method}`, **`persona_set`**, **`round_sequence`** (seed + per-persona exact `question_id` order), `snapshots` (e.g. `[0,5,10,20,40]` training checkpoints) |
| `adapters.jsonl` | one row per `(learner｜__shared__｜__base__, version, snapshot)`: scope, n_rounds_trained, `eval{nll,brier,probe_acc}`, **`best_step_caveat`** flag |
| `predictions.jsonl` | **CORE** — one row per `(learner, snapshot, question)`: `p_correct_raw` (logprob/ppl-derived), `option_logprobs` (mcq), **`p_correct` (CALIBRATED)**, `scope` |
| `curves.jsonl` | per `(learner, snapshot)`: `nll`, `brier`, `mean_mastery_stab`, `scope` — **plus the A0 (`__shared__`) and A∅ (`__base__`) control series** |
| `cost.json` | `train_s_per_snapshot`, `infer_ms_per_item`, `gpu`, `total_train_s` (measured on the rig — a first-class paper metric) |

Three conditions every bundle must contain: **A1** per-user LoRA (headline, multiple
snapshots → convergence curve) · **A0** shared LoRA (pooled control) · **A∅** base/no-input
(leak/floor control). See §9 for why A0/A∅ are mandatory.

## 5. What already exists in THIS repo — DON'T rebuild it

The friend's team built the live side. Read these to learn the exact shape your bundle must
satisfy (especially `offline_replay.py` — it loads your bundle):

- `backend/app/services/usermodel/parametric.py` — the `parametric` UserModel backend.
- `backend/app/services/usermodel/providers/stub.py` — deterministic placeholder.
- `backend/app/services/usermodel/providers/offline_replay.py` — **loads our bundle**; match its expected fields.
- `backend/app/services/usermodel/providers/service.py` — calls the HTTP `parametric_service`.
- `parametric_service/main.py` — a **mock** of our offline rig as an API (the HTTP contract our real rig would later satisfy).
- `backend/app/services/usermodel/{baseline,agentic}.py` (if present) — the other two backends, for context.

> Note: the README/repo call the third provider **`service`** (an HTTP service mocking the
> rig); some docs call it `vllm`. Same seam — our real vLLM rig swaps in behind that HTTP
> contract, OR we just hand over a static bundle for `offline_replay`. **Producing a static
> bundle is the lower-risk path and is all that's strictly required.**

## 6. What we ALREADY HAVE in Lin's other repos — REUSE

The *method* is largely done in Lin's prior repos (on KCL CREATE home). Map → bundle:

| Bundle need | Reuse from | Path (on `/users/k2480198/`) |
| --- | --- | --- |
| per-user **dual-rate LoRA** trainer (slow MLP r32 + fast Attn r16, response-prediction) | UserSim student_opd | `user_world_model/legacy/general_personamem/student_opd/train_opd_dual.py`; also `P-OPSD/dynamic_usersim/student_opd/` |
| **MCQ-PPL / option-logprob** scoring → `p_correct_raw` + `option_logprobs` | eval_mcq_ppl | `…/legacy/general_personamem/student_opd/eval_mcq_ppl.py` (+ `teacher_sft/eval_mcq_ppl.py`) |
| **shared LoRA (A0)** control | shared-LoRA trainer | `P-OPSD/opd/s01_opd_train_v2_shared.py` |
| **base / no-input (A∅)** control | base eval, no-persona probe | `P-OPSD` Phase-5 leakage probes; any base-model eval |
| **multi-checkpoint convergence** curves | per-step closure logs | `…/legacy/general_personamem/student_opd/` round logs; `compare_rounds.py` |
| merge LoRA for vLLM serving | merge util | `…/student_opd/merge_dual_lora.py`; `P-OPSD/scripts/merge_lora_into_base.py` |
| per-turn OPD sample build | data builder | `…/student_opd/build_opd_data.py` |

Base model + recipe defaults the schema already assumes: **Qwen3-4B-Instruct-2507**,
dual-rate **s32f16**, objective **user_turn_response** — i.e. exactly Lin's R1b.

## 7. New work (genuinely not done)

1. **Re-target to Cue's curriculum.** Lin's prior runs were on **PersonaMem** personas. Cue
   needs learners over its **seeded curriculum + question bank** (`biology_gcse`,
   `chemistry_gcse`) — so personas/questions come from Cue's seed bank, **not** PersonaMem.
   You cannot reuse PersonaMem traces directly.
2. **`persona_set` + `round_sequence`** (the fairness lynchpin, §8 / D8) — a fixed learner set
   + exact per-persona question order + seed, replayed identically by both sides.
3. **Per-skill probe** (D5) — a small fixed probe item per skill to read `mastery(skill)` from
   the adapter; you define the set + scoring.
4. **Calibration layer** (D7) — isotonic/Platt on a held-out split → emit calibrated
   `p_correct` (Lin's prior work used raw PPL; this adds the calibration step, same method as
   the demo's other backends).
5. **Bundle packaging** — write the 5 files in the exact schema (mechanical).
6. **`cost.json`** — real train time + per-item inference latency from the rig.

## 8. Decisions D1–D9 (ours to make) + the ONE to lock first

Lin's own data already fixes several: **D1 base** = Qwen3-4B-Instruct-2507 · **D2 recipe** =
dual-rate s32f16 · **D3 objective** = user_turn_response · **D4 snapshots** = {0,5,10,20,40} ·
**best-step**: schema has a `best_step_caveat` flag — **report final-step too**, never import
the confounded best-step "+95.7%" alone (final was +71.7%).

**Lock first — §4 / D8: who owns `persona_set` + `round_sequence`.** Recommended: let the
**demo2 team own it** (they have the seeded curriculum + question bank); Lin trains on their
fixed stream. Without an agreed shared stream, A1 is measured on a different stream than
baseline/agentic and the comparison is apples-to-oranges. Then D5 (probe), D6 (logprob→
`predict_correct`), D7 (calibration method), D9 (timeline for a first minimal bundle).

## 9. Guardrails — non-negotiable (these are Lin's OWN handoff conclusions, now enforced)

- **Mandatory controls:** A0 (shared LoRA) **and** A∅ (no-input) must be in every bundle — the
  pivotal null was the *shared* LoRA beating per-user (76.3% vs 58.8%) + a 69.8% no-input
  leak. A "per-user wins" claim is unsupported without both.
- **Best-step caveat:** flag it; give **final-step** numbers too.
- **Raw-completion serving** for the user-turn LoRAs (they lose instruction-following) —
  `PARAMETRIC_RAW_COMPLETION=1` on the rig; score continuations, never demand JSON.
- **Multiple metrics per learner** (NLL / probe-acc / judge diverge) — at minimum NLL + probe-acc.
- **Grade with the shared grader**, never a simulated learner's tone.
- **Don't port the heavy GPU training research wholesale** (teacher SFT / GRPO stacks) — train
  a lean per-user adapter only.

## 10. First step (de-risk) — a tiny sample bundle

Do **not** build the full thing first. Earliest useful deliverable: a **2–3 persona sample
bundle** in the schema's exact shape (rough/partial numbers OK), so the team can wire and test
`offline_replay` end-to-end against the real format. Then iterate to the full persona set +
final metrics. This makes the contract concrete before either side over-invests.

## 11. Environment / where things are

- **This repo (Cue, live app):** `/users/k2480198/emnlp2026_demo2` (origin `git@github.com:exp-mark/emnlp2026_demo2`). Run: `make install && make seed && make backend-dev` (+ `make frontend-dev`, `make parametric-dev`). Offline CI: `LLM_FAKE=1`, `make test`. Reading order: `CLAUDE.md` → `docs/00-overview.md` → `docs/handoff-to-linhai.md` → `docs/backend/parametric-bundle-schema.md` → `docs/backend/05-user-model.md`.
- **Our offline harness** lives wherever we run GPUs (KCL CREATE A100/H100, or Isambard GH200 aarch64). It is **separate from this repo's deploy path** — the repo only ever consumes the exported bundle.
- **Lin's reusable code:** `/users/k2480198/{user_world_model,P-OPSD,parametric-memory-pilot}` (see §6). HF account `lzhang472`. Cluster storage + env per the global `~/.claude/CLAUDE.md`.
- **GitHub push:** SSH key `~/.ssh/github_key`. (Pushing this private-derived content to the shared repo crosses a trust boundary the auto-mode may block — let Lin run any push.)

## 12. Boundaries — keep Cue and UWM separate

- **This folder = Cue (demo2) only.** The proactive companion + our offline bundle.
- **UWM research continues in** `/cephfs/volumes/hpc_home/k2480198/.../user_world_model` — the
  cross-domain (general/health/education) per-user world model. Do **not** mix UWM design work
  into this repo. Cue's Mode A is the *applied, tutoring-grounded* face of the same
  parametric-vs-token-memory thesis, but the repos and goals are distinct.
- The **other** demo is `colearn` (emnlp26_demo) — a *reactive* tutor. Don't confuse the two;
  Cue is the one we engage for the parametric work.
