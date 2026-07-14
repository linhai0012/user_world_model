# results/

Offline outputs that generate the tables in `../RESULTS-OPD.md`. CPU-inspectable; no model weights
here (LoRA adapters / base-model cache stay on the GPU rig).

## predictions/  — per-(persona, snapshot, eval-item) scores for the four recipes

Each `<recipe>/persona_NN.jsonl` (+ `__base__.jsonl`, `__shared__.jsonl`) holds rows of
`{learner_id, snapshot, question_id, option_logprobs, p_correct, scope}` on the held-out eval+calib
items. `p_correct` is the raw choice-PPL softmax probability of the correct option (un-calibrated).

| folder | recipe | note |
| --- | --- | --- |
| `hard_v4` | lean SFT, spread (v4) population | the hard baseline |
| `oracle_g` | OPD distilling `g(θ)` | privileged upper bound |
| `oracle_hybrid` | OPD `0.5·g + 0.5·realized` | privileged upper bound |
| `llm_teacher` | LLM-teacher OPD (semi-oracle) | the deployable-ish recipe |

Regenerate any comparison with `sft/compare_opd.py` (set `CUE_PRED_A/CUE_PRED_B/CUE_SUBSTRATE`).

## bundles/  — calibrated artifact-bundle summaries (the honesty-gate deliverable)

`headline.json` (per-snapshot NLL/Brier for per_user/shared/base), `curves.jsonl`, `manifest.json`
(base model, LoRA recipe, persona_set, snapshots), `cost.json` (GPU/time). The full importable bundles
(with calibrated `predictions.jsonl` + `adapters.jsonl`) live on the rig; `../sample_bundle/` is a
small example, and `../import_bundle.py` is the JSONL→Mongo importer.
