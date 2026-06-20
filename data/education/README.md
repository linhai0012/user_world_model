# Education data (private)

Private tutoring-dialogue data for the **education domain** of the User World
Model. **Private — do not redistribute.** Kept in this repo because the GitHub
remote is private.

## Files

| File | Records | Content |
|---|---|---|
| `chat_nlp.jsonl` | 48 | Student ↔ AI-tutor sessions for a KCL **NLP / AI** course (search algorithms, ML concepts, ethics, PEAS, etc.) |
| `chat_ai.jsonl` | 18 | Same format; supervised-learning, regression, decision trees, uncertainty, BFS, course roadmap, with courseware-citation retrieval |

## Schema (per line = one session) — VERIFIED 2026-06-20

> ⚠️ The original description below was partly inaccurate; corrected after
> inspection (see `common/edu_data.py`, which is the source of truth for parsing).

Top-level fields: `id`, `timestamp`, `name` (all `litellm-acompletion`), `userId`,
`sessionId`, `release`, `version`, `environment`, `tags`, `bookmarked`, `public`,
`input`, `output`, `metadata`, `comments`, `_session_records`,
`_reconstructed_turns`, `_category` (all `chat`).

- **`input`** — a **JSON-encoded string**; parse it → `{"messages": [...], "tools": [...]}`.
  `messages` is the conversation: a tutor **system prompt** (Socratic/active-learning
  persona) then alternating `assistant` (=tutor) / `user` (=student) turns, plus a
  few `tool` turns (courseware retrieval). This is the real dialogue.
- **`output`** — the tutor's final response (a JSON string with `content`).
- ⚠️ **`userId` and `sessionId` are `None` for every record** → there is **no
  cross-session learner identity**. The unit is the SESSION; no per-learner grouping
  or per-user weights are possible.
- ⚠️ `_reconstructed_turns` and `_session_records` are **ints** (counts), NOT lists.

**Counts (verified):** chat_nlp = 48 sessions / 346 student turns; chat_ai = 18
sessions / 87 student turns. Student turns are genuine learner text (typos,
follow-ups, short questions). Stage-1 task = predict the student's next turn
(text/reaction head; no exam data → no structured-state head).

## Learner-reaction signals present

Clarity/confusion (brief vs detailed replies), correctness feedback embedded in
tutor turns ("Exactly!" / asset-based corrections), engagement & progression
(incremental answer refinement), misconception probing, metacognitive
reflection. These are the supervision signals for the education-domain
reaction + state output.

## Public analogs

For breadth/synthesis, pair with **StudyChat** (closest public match — real
university-AI-course student↔ChatGPT), **MathDial** (student-as-LLM method
template), and exam-behavior sets **EEDI/NeurIPS2020**, **EdNet**,
**ASSISTments-2012**. See `../../project_summary.md` §9.
