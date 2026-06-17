"""Model backends + the ONE shared scoring path (CONVENTIONS.md §3).

Eval contract: a method only chooses *what context messages to inject*; the backend does the
identical scoring for every method, so baselines are comparable.

MCQ-PPL (canonical PersonaMem protocol, faithful to legacy `teacher_sft/eval_mcq_ppl.py`):
for each option, score the mean per-token NLL of the option **as the assistant continuation
of the conversation** (option text + closing `<|im_end|>`), given the injected context
messages + the user's query. Pick argmin. No instruction prompt — the option is scored as a
natural continuation, exactly as the legacy protocol (and the PersonaMem paper) do.

A *method* supplies `context_messages` (a list of {role,content}); the backend appends the
final user turn (the query) and scores each option as the assistant reply.

`qwen3-4b` → local vLLM (logprobs). `gpt41` → OpenAI (no logprobs) — stub for now.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass
class ChoiceScores:
    pred_letter: str
    per_choice_nll: dict[str, float]   # letter -> mean token NLL (lower = better)


def _assistant_msgs(context_messages: list[dict], query: str) -> list[dict]:
    """context turns + the MCQ query as the final user message."""
    msgs = [dict(m) for m in (context_messages or [])]
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1]["content"] = msgs[-1]["content"].rstrip() + "\n\n" + query
    else:
        msgs.append({"role": "user", "content": query})
    return msgs


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class VLLMQwenBackend:
    """vLLM offline backend with batched MCQ-PPL scoring."""

    def __init__(self, model: str = "Qwen/Qwen3-4B-Instruct-2507",
                 max_model_len: int = 32768, gpu_mem: float = 0.85, dtype: str = "bfloat16"):
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")  # B200 quirk
        from vllm import LLM, SamplingParams
        self._SP = SamplingParams
        self.model_name = model
        self.max_model_len = max_model_len
        self.llm = LLM(model=model, dtype=dtype, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_mem, enable_prefix_caching=True)
        self.tok = self.llm.get_tokenizer()

    def _ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _prefix_and_full(self, context_messages, query, option_text):
        """Return (prefix_ids, full_ids). full = conversation + assistant(option+im_end);
        prefix = same up to the assistant-open. Scored tokens = full[len(prefix):]."""
        base = _assistant_msgs(context_messages, query)
        prefix_text = self.tok.apply_chat_template(base, tokenize=False, add_generation_prompt=True)
        full_text = self.tok.apply_chat_template(
            base + [{"role": "assistant", "content": option_text}],
            tokenize=False, add_generation_prompt=False)
        pre_ids, full_ids = self._ids(prefix_text), self._ids(full_text)
        cpl = _common_prefix_len(pre_ids, full_ids)
        # robust boundary: the scored span starts at the common-prefix end
        return pre_ids[:cpl], full_ids, cpl

    def score_batch(self, items: list[tuple[list[dict], str, list[tuple[str, str]]]],
                    max_ctx_tokens: int | None = None) -> list[ChoiceScores]:
        """items = [(context_messages, query, options[(letter,text)]), ...].
        One vLLM call over all (item,option) sequences; reduce to per-item argmin NLL."""
        budget = (max_ctx_tokens or self.max_model_len) - 64
        prompt_ids_list, index = [], []   # (item_idx, letter, score_start, n_scored)
        for it, (ctx, query, opts) in enumerate(items):
            for letter, text in opts:
                _, full_ids, cpl = self._prefix_and_full(ctx, query, text)
                if len(full_ids) > budget:
                    # drop oldest context tokens before the scored span (keep scored intact)
                    overflow = len(full_ids) - budget
                    full_ids = full_ids[overflow:]
                    cpl = max(0, cpl - overflow)
                n_scored = max(1, len(full_ids) - cpl)
                prompt_ids_list.append(full_ids)
                index.append((it, letter, cpl, n_scored))

        sp = self._SP(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        prompts = [{"prompt_token_ids": ids} for ids in prompt_ids_list]
        outs = self.llm.generate(prompts, sampling_params=sp)

        per_item: list[dict[str, float]] = [dict() for _ in items]
        for (it, letter, start, n_scored), o in zip(index, outs):
            pl = o.prompt_logprobs or []
            lps = []
            for j in range(start, len(pl)):
                entry = pl[j]
                if not entry:
                    continue
                lp = next((v for v in entry.values() if getattr(v, "rank", 0) == 0),
                          next(iter(entry.values())))
                lps.append(lp.logprob if hasattr(lp, "logprob") else float(lp))
            per_item[it][letter] = -(sum(lps) / len(lps)) if lps else float("inf")

        out = []
        for scores in per_item:
            pred = min(scores, key=scores.get) if scores else ""
            out.append(ChoiceScores(pred_letter=pred, per_choice_nll=scores))
        return out


    # ---- reader/generation lens: the model ANSWERS the MCQ given the memory ----
    _READER_SYS = (
        "You are answering a multiple-choice question about how an assistant should respond to "
        "a specific user. Use what is known about the user to choose the single best option. "
        "Respond with ONLY the letter of the best option (a, b, c, or d)."
    )
    _LETTER = re.compile(r"[a-dA-D]")

    @staticmethod
    def _render_mem(context_messages: list[dict]) -> str:
        if not context_messages:
            return ""
        return "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in context_messages)

    def pick_batch(self, items, max_ctx_tokens: int | None = None):
        """Reader lens: prompt the model to pick a letter given (memory + query + options).
        Returns ChoiceScores(pred_letter, {}) per item (no per-choice NLL)."""
        budget = (max_ctx_tokens or self.max_model_len) - 32
        prompts, order = [], []
        for it, (ctx, query, opts) in enumerate(items):
            mem = self._render_mem(ctx)
            lines = ["[What is known about the user]", mem, "",
                     "[User's latest message]", query, "",
                     "Which assistant response best fits THIS user?"]
            if not mem:
                lines = lines[3:]  # drop empty memory block
            for letter, text in opts:
                lines.append(f"({letter}) {text}")
            lines.append("\nAnswer with the letter only.")
            msgs = [{"role": "system", "content": self._READER_SYS},
                    {"role": "user", "content": "\n".join(lines)}]
            ids = self._ids(self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
            if len(ids) > budget:
                ids = ids[-budget:]
            prompts.append({"prompt_token_ids": ids})
            order.append(it)
        outs = self.llm.generate(prompts, self._SP(temperature=0.0, max_tokens=4))
        res = [ChoiceScores("", {}) for _ in items]
        for it, o in zip(order, outs):
            txt = o.outputs[0].text if o.outputs else ""
            m = self._LETTER.search(txt)
            res[it] = ChoiceScores(m.group(0).lower() if m else "", {})
        return res


class OpenAIGptBackend:
    """gpt41 via OpenAI — generation + letter parse (no logprobs). Stub: wire in next."""

    _LETTER = re.compile(r"\(?([a-dA-D])\)?")

    def __init__(self, model: str = "gpt-4.1"):
        raise NotImplementedError(
            "gpt41 backend not wired yet; qwen3-4b (MCQ-PPL) is the working path.")


def make_backend(model: str, **kw):
    if model == "qwen3-4b":
        return VLLMQwenBackend(**kw)
    if model == "gpt41":
        return OpenAIGptBackend(**kw)
    raise ValueError(f"unknown model backend {model!r}")
