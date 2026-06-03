# Education data (private)

Private tutoring-dialogue data for the **education domain** of the User World
Model. **Private — do not redistribute.** Kept in this repo because the GitHub
remote is private.

## Files

| File | Records | Content |
|---|---|---|
| `chat_nlp.jsonl` | 48 | Student ↔ AI-tutor sessions for a KCL **NLP / AI** course (search algorithms, ML concepts, ethics, PEAS, etc.) |
| `chat_ai.jsonl` | 18 | Same format; supervised-learning, regression, decision trees, uncertainty, BFS, course roadmap, with courseware-citation retrieval |

## Schema (per line = one session)

Top-level fields: `id`, `timestamp`, `name`, `userId`, `sessionId`, `release`,
`version`, `environment`, `tags`, `bookmarked`, `public`, `input`, `output`,
`metadata`, `comments`, `_session_records`, `_reconstructed_turns`, `_category`.

- `input` — nested JSON: a tutor **system prompt** (active-learning / Socratic
  pedagogy, courseware retrieval) + an array of turns `{role, content}`
  (`system` / `user` / `assistant`).
- `output` — the tutor's final response.

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
