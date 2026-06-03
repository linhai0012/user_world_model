"""
Configuration for the Digital Twin Data Pipeline.
V3: Day-to-day wellness prediction (world model).
"""
import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
PMDATA_ROOT = Path(os.environ.get("PMDATA_ROOT", "./pmdata"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./output"))
CACHE_DIR = OUTPUT_DIR / "cache"

# ─── OpenAI API ──────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-5.4"
OPENAI_MAX_TOKENS = 512
OPENAI_TEMPERATURE = 0.7
OPENAI_MAX_CONCURRENT = 5
OPENAI_RETRY_ATTEMPTS = 3
OPENAI_RETRY_DELAY = 2

# ─── Heart-rate quantization (used for baseline HR input) ───────────────────
HR_BIN_MIN = 40
HR_BIN_MAX = 200
HR_BIN_STEP = 1

# ─── Activity window parameters ──────────────────────────────────────────────
HR_BASELINE_WINDOW_MIN = 10
HR_WINDOW_AFTER_MIN = 30
HR_RESAMPLE_INTERVAL_SEC = 60
HR_SEQ_MAX_LEN = 120
HR_BASELINE_MIN_LEN = 3

# ─── Participant filter ──────────────────────────────────────────────────────
PARTICIPANTS = [f"p{i:02d}" for i in range(1, 17)]
RANDOM_SEED = 42

# ─── User tokens (learned embeddings for personalization) ───────────────
USER_TOKENS = {pid: f"<user_{pid}>" for pid in PARTICIPANTS}

# ─── Special tokens ─────────────────────────────────────────────────────────
TOKEN_TS_START = "<ts_start>"
TOKEN_TS_END = "<ts_end>"
TOKEN_TEXT_START = "<text_start>"
TOKEN_TEXT_END = "<text_end>"
TOKEN_STATE_START = "<state_start>"
TOKEN_STATE_END = "<state_end>"

HR_TOKEN_PREFIX = "<hr_"
HR_TOKEN_SUFFIX = ">"

# ─── Activity tokens (Stage 0: begin/end pairs at exact positions) ─────────
ACTIVITY_CATEGORIES = ["walk", "run", "bike", "other"]

ACTIVITY_BEGIN_TOKENS = {cat: f"<{cat}_start>" for cat in ACTIVITY_CATEGORIES}
ACTIVITY_END_TOKENS = {cat: f"<{cat}_end>" for cat in ACTIVITY_CATEGORIES}

# Map PMData activity names → standard categories
ACTIVITY_NAME_MAP = {
    "walk": "walk",
    "run": "run",
    "treadmill": "run",
    "jogging": "run",
    "outdoor bike": "bike",
    "bike": "bike",
    "cycling": "bike",
}
# Everything else → "other"

# ─── Wellness state tokens ──────────────────────────────────────────────────
# Each wellness field gets tokens for its full scale range
WELLNESS_FIELDS = {
    "fatigue":       (1, 5),    # 1=very fatigued, 5=very energetic
    "mood":          (1, 5),    # 1=very bad, 5=very good
    "readiness":     (1, 10),   # 1=not ready, 10=peak ready
    "sleep_quality": (1, 5),    # 1=very poor, 5=very good
    "soreness":      (1, 5),    # 1=very sore, 5=no soreness
    "stress":        (1, 5),    # 1=very stressed, 5=very relaxed
}


# ─── HR token functions ──────────────────────────────────────────────────────

def hr_to_token(bpm: int) -> str:
    bpm = max(HR_BIN_MIN, min(HR_BIN_MAX, bpm))
    return f"{HR_TOKEN_PREFIX}{bpm}{HR_TOKEN_SUFFIX}"

def token_to_hr(token: str) -> int:
    return int(token.replace(HR_TOKEN_PREFIX, "").replace(HR_TOKEN_SUFFIX, ""))


# ─── State token functions ───────────────────────────────────────────────────

def state_to_token(field: str, value: int) -> str:
    """e.g. state_to_token('fatigue', 3) → '<fatigue_3>'"""
    lo, hi = WELLNESS_FIELDS[field]
    value = max(lo, min(hi, value))
    return f"<{field}_{value}>"

def token_to_state(token: str) -> tuple[str, int]:
    """e.g. token_to_state('<fatigue_3>') → ('fatigue', 3)"""
    inner = token.strip("<>")
    field, val_str = inner.rsplit("_", 1)
    return field, int(val_str)

def encode_wellness_state(wellness: dict) -> list[str]:
    """Encode a wellness dict into ordered state tokens."""
    tokens = []
    for field in WELLNESS_FIELDS:
        val = wellness.get(field)
        if val is not None and isinstance(val, (int, float)):
            tokens.append(state_to_token(field, int(val)))
    return tokens

def decode_state_tokens(tokens: list[tuple[str, int]]) -> dict:
    """Convert list of (field, value) tuples back to wellness dict."""
    return {field: value for field, value in tokens}


# ─── Token list builders ────────────────────────────────────────────────────

def get_all_hr_tokens() -> list[str]:
    return [hr_to_token(b) for b in range(HR_BIN_MIN, HR_BIN_MAX + 1)]

def get_all_state_tokens() -> list[str]:
    tokens = []
    for field, (lo, hi) in WELLNESS_FIELDS.items():
        for v in range(lo, hi + 1):
            tokens.append(state_to_token(field, v))
    return tokens

def get_all_activity_tokens() -> list[str]:
    tokens = []
    for cat in ACTIVITY_CATEGORIES:
        tokens.append(ACTIVITY_BEGIN_TOKENS[cat])
        tokens.append(ACTIVITY_END_TOKENS[cat])
    return tokens

def normalize_activity_name(name: str) -> str:
    """Map a PMData activity name to a standard category."""
    return ACTIVITY_NAME_MAP.get(name.strip().lower(), "other")

def get_all_user_tokens() -> list[str]:
    return list(USER_TOKENS.values())

def get_all_special_tokens() -> list[str]:
    return [
        TOKEN_TS_START, TOKEN_TS_END,
        TOKEN_TEXT_START, TOKEN_TEXT_END,
        TOKEN_STATE_START, TOKEN_STATE_END,
    ] + get_all_hr_tokens() + get_all_state_tokens() + get_all_activity_tokens() + get_all_user_tokens()
