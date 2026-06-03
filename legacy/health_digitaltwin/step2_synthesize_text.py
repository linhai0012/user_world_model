"""
Step 2: Synthesize paired text via GPT-5.4 API.

For each parsed record we call GPT-5.4 to generate:
  (a) event_description  — a natural-language description of what happened
  (b) user_feedback      — a first-person user reaction / body-feel report

The prompt is grounded in *real* data (activity type, duration, HR stats,
wellness self-report) so the synthetic text stays physiologically plausible.

Output: output/cache/synthesized_records.json
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from config import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MAX_TOKENS,
    OPENAI_TEMPERATURE, OPENAI_MAX_CONCURRENT,
    OPENAI_RETRY_ATTEMPTS, OPENAI_RETRY_DELAY,
    CACHE_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ─── OpenAI client ───────────────────────────────────────────────────────────
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
semaphore = asyncio.Semaphore(OPENAI_MAX_CONCURRENT)


# ═══════════════════════════════════════════════════════════════════════════════
#  Prompt construction
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a data-synthesis assistant for a health-AI research project.
Given structured data about a person's exercise session and self-reported
wellness, you produce TWO pieces of natural-language text:

1. **event_description**: A concise third-person description of the activity
   event (1-3 sentences). Include activity type, approximate duration,
   intensity hints, and any environmental/contextual detail you can
   reasonably infer. Do NOT mention exact heart-rate numbers.

2. **user_feedback**: A first-person, colloquial self-report the user might
   write after the session (2-4 sentences). It should reflect the wellness
   scores (fatigue, mood, soreness, stress) naturally — without stating the
   numeric scores explicitly. Mention body sensations, energy level, mood,
   and any perceived exertion.

Reply in **strict JSON** with keys "event_description" and "user_feedback".
No markdown fences, no extra keys.\
"""

# V3: Day-to-day wellness prediction prompt
SYSTEM_PROMPT_V3 = """\
You are a data-synthesis assistant for a health-AI research project.
You are given:
- A person's morning wellness self-report (Day N)
- Their exercise activity summary for the day (Day N)
- Their morning wellness self-report the NEXT day (Day N+1)

Produce TWO pieces of natural-language text:

1. **event_description**: A concise third-person summary of the person's day
   (1-3 sentences). Describe their starting condition, what activity they did,
   and how intense it was. Do NOT mention exact numeric scores.

2. **user_feedback**: A first-person, colloquial diary entry the person might
   write at the end of the day (3-5 sentences). It should naturally reflect:
   - How the exercise felt (effort, body sensations)
   - Current energy/mood after the activity
   - Expectations for tomorrow (e.g., "I'll probably be sore", "should feel
     refreshed tomorrow")
   The text should be consistent with both the Day N state, the activity done,
   and the actual Day N+1 state — but do NOT state numeric scores explicitly.

Reply in **strict JSON** with keys "event_description" and "user_feedback".
No markdown fences, no extra keys.\
"""


def _build_user_message(record: dict[str, Any]) -> str:
    """Build the data-grounded user prompt for one record (V1/V2)."""
    hr_seq = record["hr_sequence"]
    hr_min = min(hr_seq)
    hr_max = max(hr_seq)
    hr_mean = round(sum(hr_seq) / len(hr_seq), 1)

    parts = [
        f"Activity: {record['activity_name']}",
        f"Duration: {record['duration_min']} minutes",
        f"Calories burned: {record['calories']}",
    ]
    if record["distance_km"]:
        parts.append(f"Distance: {record['distance_km']} km")
    if record["steps"]:
        parts.append(f"Steps: {record['steps']}")

    parts.append(f"Heart-rate stats — min: {hr_min}, max: {hr_max}, mean: {hr_mean}")

    w = record.get("wellness")
    if w:
        wellness_parts = []
        label_map = {1: "very low", 2: "low", 3: "normal", 4: "high", 5: "very high"}
        for key in ("fatigue", "mood", "readiness", "sleep_quality", "soreness", "stress"):
            val = w.get(key)
            if val is not None:
                wellness_parts.append(f"{key}: {val}/5 ({label_map.get(val, '?')})")
        if w.get("soreness_area"):
            wellness_parts.append(f"soreness area: {w['soreness_area']}")
        if wellness_parts:
            parts.append("Self-reported wellness: " + "; ".join(wellness_parts))

    return "\n".join(parts)


def _build_user_message_v3(record: dict[str, Any]) -> str:
    """Build the data-grounded user prompt for V3 day-to-day records."""
    label_map = {1: "very low", 2: "low", 3: "moderate", 4: "high", 5: "very high"}
    readiness_map = {i: f"{i}/10" for i in range(1, 11)}

    parts = []

    # Day N morning wellness
    wn = record.get("wellness_day_n", {})
    if wn:
        w_parts = []
        for key in ("fatigue", "mood", "sleep_quality", "soreness", "stress"):
            val = wn.get(key)
            if val is not None:
                w_parts.append(f"{key}: {val}/5 ({label_map.get(val, '?')})")
        readiness = wn.get("readiness")
        if readiness is not None:
            w_parts.append(f"readiness: {readiness}/10")
        parts.append("Morning wellness (Day N): " + "; ".join(w_parts))

    # Activity summary
    act_names = record.get("activity_names", [])
    parts.append(f"Activities: {', '.join(act_names)}")
    parts.append(f"Total duration: {record.get('total_duration_min', 0)} minutes")
    if record.get("total_calories"):
        parts.append(f"Total calories: {record['total_calories']}")
    if record.get("total_steps"):
        parts.append(f"Total steps: {record['total_steps']}")

    # HR stats
    hr_stats = record.get("hr_stats")
    if hr_stats and hr_stats.get("mean"):
        parts.append(f"Heart-rate: mean={hr_stats['mean']}, max={hr_stats['max']}, min={hr_stats['min']}")

    # sRPE
    srpe = record.get("srpe")
    if srpe and srpe.get("perceived_exertion"):
        parts.append(f"Perceived exertion: {srpe['perceived_exertion']}/10")

    # Day N+1 morning wellness (GPT sees this to generate consistent text)
    wn1 = record.get("wellness_day_n1", {})
    if wn1:
        w1_parts = []
        for key in ("fatigue", "mood", "sleep_quality", "soreness", "stress"):
            val = wn1.get(key)
            if val is not None:
                w1_parts.append(f"{key}: {val}/5 ({label_map.get(val, '?')})")
        readiness = wn1.get("readiness")
        if readiness is not None:
            w1_parts.append(f"readiness: {readiness}/10")
        parts.append("Next morning wellness (Day N+1): " + "; ".join(w1_parts))

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  Async API call with retry
# ═══════════════════════════════════════════════════════════════════════════════

async def _call_gpt(record: dict[str, Any], system_prompt: str = None,
                    msg_builder=None) -> dict[str, str] | None:
    """Call GPT-5.4 for one record, return parsed JSON or None."""
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    if msg_builder is None:
        msg_builder = _build_user_message
    user_msg = msg_builder(record)

    for attempt in range(1, OPENAI_RETRY_ATTEMPTS + 1):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=OPENAI_MODEL,
                    temperature=OPENAI_TEMPERATURE,
                    max_completion_tokens=OPENAI_MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                )
            text = response.choices[0].message.content.strip()
            # Strip potential markdown fences
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
            if "event_description" in result and "user_feedback" in result:
                return result
            log.warning(f"Missing keys in GPT response: {list(result.keys())}")
        except json.JSONDecodeError:
            log.warning(f"JSON parse error on attempt {attempt}: {text[:200]}")
        except Exception as e:
            log.warning(f"API error on attempt {attempt}: {e}")

        if attempt < OPENAI_RETRY_ATTEMPTS:
            await asyncio.sleep(OPENAI_RETRY_DELAY * attempt)

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch processing
# ═══════════════════════════════════════════════════════════════════════════════

async def synthesize_all(records: list[dict[str, Any]],
                         system_prompt: str = None,
                         msg_builder=None) -> list[dict[str, Any]]:
    """Add synthesized text to each record via parallel API calls."""
    total = len(records)
    log.info(f"Synthesizing text for {total} records (concurrency={OPENAI_MAX_CONCURRENT}) ...")

    tasks = [_call_gpt(r, system_prompt, msg_builder) for r in records]
    results = await asyncio.gather(*tasks)

    enriched: list[dict[str, Any]] = []
    success, fail = 0, 0
    for record, synth in zip(records, results):
        if synth is None:
            fail += 1
            continue
        record["event_description"] = synth["event_description"]
        record["user_feedback"] = synth["user_feedback"]
        enriched.append(record)
        success += 1

    log.info(f"Synthesis complete: {success} succeeded, {fail} failed")
    return enriched


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load parsed records from step 1
    in_path = CACHE_DIR / "parsed_records.json"
    if not in_path.exists():
        raise FileNotFoundError(f"Run step1 first. Missing {in_path}")
    records = json.loads(in_path.read_text())
    log.info(f"Loaded {len(records)} parsed records")

    # Synthesize
    enriched = asyncio.run(synthesize_all(records))

    out_path = CACHE_DIR / "synthesized_records.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(enriched)} synthesized records → {out_path}")
    return enriched


def run_v3():
    """V3: Synthesize text for daily records (day-to-day wellness prediction)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    in_path = CACHE_DIR / "daily_records.json"
    if not in_path.exists():
        raise FileNotFoundError(f"Run step1 first. Missing {in_path}")
    records = json.loads(in_path.read_text())
    log.info(f"Loaded {len(records)} daily records")

    enriched = asyncio.run(synthesize_all(
        records,
        system_prompt=SYSTEM_PROMPT_V3,
        msg_builder=_build_user_message_v3,
    ))

    out_path = CACHE_DIR / "synthesized_daily_records.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(enriched)} synthesized daily records → {out_path}")
    return enriched


if __name__ == "__main__":
    run()
