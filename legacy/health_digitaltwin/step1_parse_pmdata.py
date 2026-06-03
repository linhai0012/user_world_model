"""
Step 1: Parse PMData into structured activity-session records.

For each participant, we:
  1. Load exercise sessions (activity type, start/end time, metrics).
  2. Load heart-rate JSON and build a time-indexed Series.
  3. Load daily wellness self-reports (fatigue, mood, soreness, stress, …).
  4. For each exercise session, extract the aligned HR window and the
     closest wellness report → produce one structured record.

Output: a list of dicts saved as  output/cache/parsed_records.json
"""
from __future__ import annotations

import json
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    PMDATA_ROOT, CACHE_DIR, PARTICIPANTS,
    HR_BASELINE_WINDOW_MIN, HR_WINDOW_AFTER_MIN,
    HR_RESAMPLE_INTERVAL_SEC, HR_SEQ_MAX_LEN,
    HR_BIN_MIN, HR_BIN_MAX, HR_BASELINE_MIN_LEN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Low-level loaders
# ═══════════════════════════════════════════════════════════════════════════════

def _try_parse_dt(s: str) -> datetime | None:
    """Try multiple datetime formats that appear in PMData exports."""
    # Strip trailing 'Z' (UTC marker) that appears in PMSys CSV exports
    s = s.rstrip("Z")
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %I:%M:%S %p",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def load_heart_rate(hr_path: Path) -> list[tuple[datetime, int]]:
    """
    Parse Fitbit heart_rate.json → [(datetime, bpm), …] sorted by time.
    Format: [{"dateTime": "…", "value": {"bpm": N, "confidence": N}}, …]
    """
    if not hr_path.exists():
        return []

    raw = json.loads(hr_path.read_text(encoding="utf-8"))
    out: list[tuple[datetime, int]] = []
    for entry in raw:
        dt = _try_parse_dt(entry.get("dateTime", ""))
        bpm = entry.get("value", {}).get("bpm")
        if dt and bpm is not None:
            out.append((dt, int(bpm)))
    out.sort(key=lambda x: x[0])
    return out


def load_exercises(participant_dir: Path) -> list[dict[str, Any]]:
    """
    Load exercise/activity sessions.
    PMData stores them under  Fitbit/Physical Activity/exercise-<date>.json
    or  Fitbit/exercise.json  depending on export version.
    Each entry has activityName, startTime, duration (ms), averageHeartRate, etc.
    """
    exercises: list[dict[str, Any]] = []

    # Possible locations in PMData exports (handle both cases)
    fitbit_dir = participant_dir / "fitbit"
    if not fitbit_dir.exists():
        fitbit_dir = participant_dir / "Fitbit"
    candidates = list(fitbit_dir.glob("exercise*.json")) + \
                 list(fitbit_dir.glob("Physical Activity/exercise*.json"))

    for fp in candidates:
        try:
            raw = json.loads(fp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            start_str = item.get("startTime") or item.get("startDate", "")
            dt = _try_parse_dt(start_str)
            if dt is None:
                continue

            dur_ms = item.get("activeDuration") or item.get("duration", 0)
            exercises.append({
                "activity_name": item.get("activityName", "Unknown"),
                "start_time": dt,
                "duration_min": round(dur_ms / 60_000, 1),
                "calories": item.get("calories", 0),
                "avg_hr": item.get("averageHeartRate", None),
                "distance_km": round(item.get("distance", 0), 2),
                "steps": item.get("steps", 0),
            })

    exercises.sort(key=lambda x: x["start_time"])
    return exercises


def load_wellness(participant_dir: Path) -> list[dict[str, Any]]:
    """
    Load daily wellness self-reports from  PMSys/wellness.csv  or similar.
    Fields: date/time, fatigue, mood, readiness, sleep_duration,
            sleep_quality, soreness, soreness_area, stress
    Values for fatigue/mood/soreness/stress/sleep_quality are 1-5 scale.
    """
    wellness: list[dict[str, Any]] = []
    candidates = list(participant_dir.rglob("wellness*.csv"))

    for fp in candidates:
        with open(fp, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize field names to lowercase
                row_lower = {k.strip().lower().replace(" ", "_"): v.strip()
                             for k, v in row.items() if k}
                dt_str = (row_lower.get("date") or row_lower.get("timestamp")
                          or row_lower.get("effective_time_frame", ""))
                dt = _try_parse_dt(dt_str)

                def _int(key):
                    try:
                        return int(float(row_lower.get(key, 0)))
                    except (ValueError, TypeError):
                        return None

                wellness.append({
                    "date": dt,
                    "fatigue": _int("fatigue"),
                    "mood": _int("mood"),
                    "readiness": _int("readiness"),
                    "sleep_duration_h": row_lower.get("sleep_duration"),
                    "sleep_quality": _int("sleep_quality"),
                    "soreness": _int("soreness"),
                    "soreness_area": row_lower.get("soreness_area", ""),
                    "stress": _int("stress"),
                })

    wellness.sort(key=lambda x: (x["date"] or datetime.min))
    return wellness


def load_srpe(participant_dir: Path) -> list[dict[str, Any]]:
    """
    Load session RPE (Rating of Perceived Exertion) from PMSys/srpe.csv.
    Fields: end_date_time, activity_names, perceived_exertion (1-10), duration_min
    """
    srpe: list[dict[str, Any]] = []
    candidates = list(participant_dir.rglob("srpe*.csv"))

    for fp in candidates:
        with open(fp, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_lower = {k.strip().lower().replace(" ", "_"): v.strip()
                             for k, v in row.items() if k}
                dt_str = row_lower.get("end_date_time", "")
                dt = _try_parse_dt(dt_str)

                try:
                    pe = int(float(row_lower.get("perceived_exertion", 0)))
                except (ValueError, TypeError):
                    pe = None

                try:
                    dur = float(row_lower.get("duration_min", 0))
                except (ValueError, TypeError):
                    dur = None

                srpe.append({
                    "date": dt,
                    "perceived_exertion": pe,
                    "duration_min": dur,
                    "activity_names": row_lower.get("activity_names", ""),
                })

    srpe.sort(key=lambda x: (x["date"] or datetime.min))
    return srpe


def find_closest_srpe(
    srpe: list[dict], activity_end: datetime
) -> dict | None:
    """Return the sRPE report closest to activity end time (within 4 hours)."""
    if not srpe:
        return None

    best, best_delta = None, timedelta(hours=4)
    for s in srpe:
        if s["date"] is None:
            continue
        delta = abs(activity_end - s["date"])
        if delta < best_delta:
            best, best_delta = s, delta

    return best


# ═══════════════════════════════════════════════════════════════════════════════
#  Heart-rate window extraction & resampling
# ═══════════════════════════════════════════════════════════════════════════════

def _resample_hr(
    pts: list[tuple[datetime, int]],
    resample_sec: int = HR_RESAMPLE_INTERVAL_SEC,
    max_len: int = HR_SEQ_MAX_LEN,
) -> list[int] | None:
    """Resample HR points to uniform grid. Returns None if too few points."""
    if len(pts) < 5:
        return None

    grid_start = pts[0][0]
    grid_end = pts[-1][0]
    total_sec = (grid_end - grid_start).total_seconds()
    n_steps = int(total_sec / resample_sec) + 1
    n_steps = min(n_steps, max_len)

    if n_steps < 3:
        return None

    resampled: list[int] = []
    for i in range(n_steps):
        target_dt = grid_start + timedelta(seconds=i * resample_sec)
        best_idx = min(range(len(pts)),
                       key=lambda j: abs((pts[j][0] - target_dt).total_seconds()))
        bpm = pts[best_idx][1]
        bpm = max(HR_BIN_MIN, min(HR_BIN_MAX, bpm))
        resampled.append(bpm)

    return resampled


def extract_hr_baseline(
    hr_data: list[tuple[datetime, int]],
    activity_start: datetime,
    baseline_min: int = HR_BASELINE_WINDOW_MIN,
    resample_sec: int = HR_RESAMPLE_INTERVAL_SEC,
) -> list[int] | None:
    """
    Extract pre-event baseline HR: [activity_start - baseline_min, activity_start].
    Returns resampled list or None if insufficient data.
    """
    win_start = activity_start - timedelta(minutes=baseline_min)
    pts = [(dt, bpm) for dt, bpm in hr_data if win_start <= dt <= activity_start]
    result = _resample_hr(pts, resample_sec, max_len=baseline_min)
    if result and len(result) < HR_BASELINE_MIN_LEN:
        return None
    return result


def extract_hr_response(
    hr_data: list[tuple[datetime, int]],
    activity_start: datetime,
    activity_end: datetime,
    after_min: int = HR_WINDOW_AFTER_MIN,
    resample_sec: int = HR_RESAMPLE_INTERVAL_SEC,
    max_len: int = HR_SEQ_MAX_LEN,
) -> list[int] | None:
    """
    Extract during+post-event HR: [activity_start, activity_end + after_min].
    No pre-event overlap — baseline is extracted separately.
    """
    win_end = activity_end + timedelta(minutes=after_min)
    pts = [(dt, bpm) for dt, bpm in hr_data if activity_start <= dt <= win_end]
    return _resample_hr(pts, resample_sec, max_len)


# ═══════════════════════════════════════════════════════════════════════════════
#  Match wellness report to activity session
# ═══════════════════════════════════════════════════════════════════════════════

def find_closest_wellness(
    wellness: list[dict], activity_dt: datetime
) -> dict | None:
    """Return the wellness report closest in time (same day preferred)."""
    if not wellness:
        return None

    best, best_delta = None, timedelta(days=999)
    for w in wellness:
        if w["date"] is None:
            continue
        delta = abs(activity_dt - w["date"])
        if delta < best_delta:
            best, best_delta = w, delta

    # Only accept if within 1 day
    if best and best_delta <= timedelta(days=1):
        return best
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Main: build records for all participants
# ═══════════════════════════════════════════════════════════════════════════════

def build_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for pid in PARTICIPANTS:
        p_dir = PMDATA_ROOT / pid
        if not p_dir.exists():
            log.warning(f"Participant dir not found: {p_dir}")
            continue

        log.info(f"Processing {pid} ...")

        # 1. Load all heart-rate data for this participant
        hr_files = sorted(p_dir.rglob("heart_rate*.json"))
        all_hr: list[tuple[datetime, int]] = []
        for hf in hr_files:
            all_hr.extend(load_heart_rate(hf))
        all_hr.sort(key=lambda x: x[0])
        log.info(f"  {pid}: loaded {len(all_hr):,} HR points")

        if not all_hr:
            continue

        # 2. Load exercises
        exercises = load_exercises(p_dir)
        log.info(f"  {pid}: loaded {len(exercises)} exercise sessions")

        # 3. Load wellness + sRPE
        wellness = load_wellness(p_dir)
        log.info(f"  {pid}: loaded {len(wellness)} wellness reports")
        srpe = load_srpe(p_dir)
        log.info(f"  {pid}: loaded {len(srpe)} sRPE reports")

        # 4. For each exercise, extract baseline + response HR + closest wellness/srpe
        for ex in exercises:
            end_time = ex["start_time"] + timedelta(minutes=ex["duration_min"])

            hr_baseline = extract_hr_baseline(all_hr, ex["start_time"])
            hr_response = extract_hr_response(all_hr, ex["start_time"], end_time)
            if hr_response is None:
                continue

            well = find_closest_wellness(wellness, ex["start_time"])
            srpe_match = find_closest_srpe(srpe, end_time)

            records.append({
                "participant_id": pid,
                "activity_name": ex["activity_name"],
                "start_time": ex["start_time"].isoformat(),
                "duration_min": ex["duration_min"],
                "calories": ex["calories"],
                "avg_hr": ex["avg_hr"],
                "distance_km": ex["distance_km"],
                "steps": ex["steps"],
                "hr_baseline": hr_baseline,
                "hr_baseline_len": len(hr_baseline) if hr_baseline else 0,
                "hr_sequence": hr_response,
                "hr_seq_len": len(hr_response),
                "wellness": {
                    "fatigue": well["fatigue"] if well else None,
                    "mood": well["mood"] if well else None,
                    "readiness": well["readiness"] if well else None,
                    "sleep_quality": well["sleep_quality"] if well else None,
                    "soreness": well["soreness"] if well else None,
                    "soreness_area": well.get("soreness_area", "") if well else "",
                    "stress": well["stress"] if well else None,
                } if well else None,
                "srpe": {
                    "perceived_exertion": srpe_match["perceived_exertion"],
                    "activity_names": srpe_match["activity_names"],
                } if srpe_match else None,
            })

    log.info(f"Total records extracted: {len(records)}")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
#  V3: Build daily aggregated records for day-to-day wellness prediction
# ═══════════════════════════════════════════════════════════════════════════════

def build_daily_records() -> list[dict[str, Any]]:
    """
    Build day-level records for V3 (wellness world model).

    Each record represents one (participant, date) with:
      - Day N wellness (input)
      - Aggregated activities for Day N (input)
      - sRPE if available (input)
      - Baseline HR from first activity (input)
      - Day N+1 wellness (output label)
    """
    # First build per-activity records to get HR data
    per_activity = build_records()

    # Group by (participant, date)
    from collections import defaultdict
    pid_date_acts = defaultdict(list)
    pid_date_wellness = {}
    pid_date_srpe = {}
    pid_date_baseline = {}

    for r in per_activity:
        pid = r["participant_id"]
        dt = datetime.fromisoformat(r["start_time"])
        d = dt.date()

        pid_date_acts[(pid, d)].append(r)

        # Store wellness (same for all activities on same day)
        if r.get("wellness"):
            pid_date_wellness[(pid, d)] = r["wellness"]

        # Store highest sRPE for the day
        if r.get("srpe") and r["srpe"].get("perceived_exertion"):
            existing = pid_date_srpe.get((pid, d))
            if existing is None or r["srpe"]["perceived_exertion"] > existing["perceived_exertion"]:
                pid_date_srpe[(pid, d)] = r["srpe"]

        # Store baseline HR from first activity of the day
        if (pid, d) not in pid_date_baseline and r.get("hr_baseline"):
            pid_date_baseline[(pid, d)] = r["hr_baseline"]

    # Helper: build activity summary for a date
    def _day_activity_summary(pid, d):
        acts = pid_date_acts.get((pid, d), [])
        if not acts:
            return None
        names = list(dict.fromkeys(a["activity_name"] for a in acts))
        dur = round(sum(a["duration_min"] for a in acts), 1)
        cal = sum(a.get("calories", 0) or 0 for a in acts)
        srpe = pid_date_srpe.get((pid, d))
        pe = srpe["perceived_exertion"] if srpe and srpe.get("perceived_exertion") else None
        return {"activities": names, "duration_min": dur, "calories": cal, "perceived_exertion": pe}

    # Helper: build 7-day history for a (pid, date)
    HISTORY_DAYS = 7
    def _build_history(pid, d):
        history = []
        for k in range(HISTORY_DAYS, 0, -1):
            hist_d = d - timedelta(days=k)
            next_hist_d = d - timedelta(days=k - 1)
            w = pid_date_wellness.get((pid, hist_d))
            w_next = pid_date_wellness.get((pid, next_hist_d))
            if not w or not w_next:
                continue
            act_summary = _day_activity_summary(pid, hist_d)
            history.append({
                "day_offset": -k,
                "wellness": w,
                "activity": act_summary,  # None means rest day
                "wellness_next": w_next,
            })
        return history

    # Build daily records with Day N → Day N+1 pairing
    daily_records = []
    for (pid, d), acts in sorted(pid_date_acts.items()):
        next_d = d + timedelta(days=1)
        day_n_wellness = pid_date_wellness.get((pid, d))
        day_n1_wellness = pid_date_wellness.get((pid, next_d))

        if not day_n_wellness or not day_n1_wellness:
            continue

        # Aggregate activities
        activity_names = list(dict.fromkeys(a["activity_name"] for a in acts))
        total_duration = round(sum(a["duration_min"] for a in acts), 1)
        total_calories = sum(a.get("calories", 0) or 0 for a in acts)
        total_steps = sum(a.get("steps", 0) or 0 for a in acts)
        avg_hr_vals = [a["avg_hr"] for a in acts if a.get("avg_hr")]
        avg_hr = round(sum(avg_hr_vals) / len(avg_hr_vals)) if avg_hr_vals else None

        # Aggregate HR sequences for stats
        all_hr = []
        for a in acts:
            all_hr.extend(a.get("hr_sequence", []))

        # Build 7-day history
        history = _build_history(pid, d)

        daily_records.append({
            "participant_id": pid,
            "date": str(d),
            "date_next": str(next_d),
            # Day N wellness (input)
            "wellness_day_n": day_n_wellness,
            # Day N+1 wellness (output label)
            "wellness_day_n1": day_n1_wellness,
            # Recent history (input)
            "history": history,
            # Aggregated activity (input)
            "activity_names": activity_names,
            "n_activities": len(acts),
            "total_duration_min": total_duration,
            "total_calories": total_calories,
            "total_steps": total_steps,
            "avg_hr": avg_hr,
            # sRPE (input, optional)
            "srpe": pid_date_srpe.get((pid, d)),
            # Baseline HR (input, optional)
            "hr_baseline": pid_date_baseline.get((pid, d)),
            # HR stats for text synthesis
            "hr_stats": {
                "mean": round(sum(all_hr) / len(all_hr), 1) if all_hr else None,
                "max": max(all_hr) if all_hr else None,
                "min": min(all_hr) if all_hr else None,
            } if all_hr else None,
        })

    log.info(f"Built {len(daily_records)} daily records "
             f"(from {len(per_activity)} per-activity records)")
    return daily_records


def build_wellness_by_date_index() -> dict[str, dict[str, dict]]:
    """
    Build a flat lookup of wellness by participant and calendar date.

    Returns: {pid: {"YYYY-MM-DD": {fatigue: 2, mood: 4, ...}}}

    Used by step3 to look up Day N and Day N+1 wellness for each per-activity
    sample, without re-reading the wellness CSVs.
    """
    out: dict[str, dict[str, dict]] = {}
    fields = ("fatigue", "mood", "readiness", "sleep_quality", "soreness", "stress")
    for pid in PARTICIPANTS:
        p_dir = PMDATA_ROOT / pid
        if not p_dir.exists():
            continue
        wellness = load_wellness(p_dir)
        pid_map: dict[str, dict] = {}
        for w in wellness:
            if w["date"] is None:
                continue
            day_str = w["date"].date().isoformat()
            entry = {k: w[k] for k in fields if w.get(k) is not None}
            if entry:
                pid_map[day_str] = entry
        if pid_map:
            out[pid] = pid_map
    return out


def run():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # V1/V2: per-activity records
    records = build_records()
    out_path = CACHE_DIR / "parsed_records.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(records)} per-activity records → {out_path}")

    # V3: daily aggregated records
    daily = build_daily_records()
    daily_path = CACHE_DIR / "daily_records.json"
    with open(daily_path, "w", encoding="utf-8") as f:
        json.dump(daily, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(daily)} daily records → {daily_path}")

    # Wellness lookup index for V2 next-day wellness join
    wellness_idx = build_wellness_by_date_index()
    wellness_idx_path = CACHE_DIR / "wellness_by_date.json"
    with open(wellness_idx_path, "w", encoding="utf-8") as f:
        json.dump(wellness_idx, f, indent=2)
    n_days = sum(len(v) for v in wellness_idx.values())
    log.info(f"Saved wellness index: {len(wellness_idx)} pids, {n_days} days "
             f"→ {wellness_idx_path}")

    return records


if __name__ == "__main__":
    run()
