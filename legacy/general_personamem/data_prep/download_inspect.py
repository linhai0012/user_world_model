"""Download all PersonaMem-v1 files and report schema.

One-off inspection: confirms file layout, session structure, where
demographics live, and MCQ field meanings before we write the real
Phase 0 data loaders.

Usage:  python dynamic_usersim/data_prep/download_inspect.py
Output: prints to stdout; files cached under data/personamem_v1/
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "bowen-upenn/PersonaMem"
REPO_TYPE = "dataset"

# Absolute root so this works regardless of cwd.
ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "personamem_v1"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def list_repo() -> list[str]:
    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    print(f"=== Files in {REPO_ID} ({REPO_TYPE}) — {len(files)} total ===")
    for f in files:
        print(f"  {f}")
    return files


def download(files: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for f in files:
        if f.startswith(".") or f.endswith(".gitattributes"):
            continue
        print(f"Downloading {f} ...", flush=True)
        local = hf_hub_download(
            repo_id=REPO_ID,
            filename=f,
            repo_type=REPO_TYPE,
            local_dir=str(CACHE_DIR),
        )
        p = Path(local)
        paths[f] = p
        size_mb = p.stat().st_size / 1e6
        print(f"  -> {p}  ({size_mb:.1f} MB)")
    return paths


def summarize_jsonl(path: Path, max_records: int = 2) -> None:
    print(f"\n--- JSONL: {path.name} ---")
    n = 0
    first_records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            n += 1
            if len(first_records) < max_records:
                first_records.append(json.loads(line))
    print(f"  total records: {n}")
    for i, rec in enumerate(first_records):
        print(f"  record[{i}] top-level keys: {list(rec.keys())}")
        for k, v in rec.items():
            tv = type(v).__name__
            if isinstance(v, list):
                inner = type(v[0]).__name__ if v else "empty"
                print(f"    {k}: list[{inner}] len={len(v)}")
            elif isinstance(v, dict):
                print(f"    {k}: dict keys={list(v.keys())[:10]}")
            elif isinstance(v, str):
                preview = v[:120].replace("\n", " ")
                print(f"    {k}: str len={len(v)} preview={preview!r}")
            else:
                print(f"    {k}: {tv} value={v!r}")


def summarize_csv(path: Path, max_rows: int = 3) -> None:
    print(f"\n--- CSV: {path.name} ---")
    import csv

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = []
        count = 1
        for row in reader:
            count += 1
            if len(rows) < max_rows:
                rows.append(row)
    print(f"  rows (incl header): {count}")
    print(f"  columns ({len(header)}): {header}")
    for i, r in enumerate(rows):
        print(f"  row[{i}]:")
        for col, val in zip(header, r):
            preview = val[:160].replace("\n", " ")
            print(f"    {col}: {preview!r}")


def deep_dive_first_context(path: Path) -> None:
    """Inspect structure of first record in shared_contexts_*.jsonl.

    We need to understand:
      - Where do demographics live?
      - How are sessions delimited?
      - What role names are used (user/assistant/chatbot)?
    """
    print(f"\n=== DEEP DIVE: {path.name} (record 0) ===")
    with path.open("r", encoding="utf-8") as fh:
        rec = json.loads(fh.readline())
    print(f"Top-level keys: {list(rec.keys())}")
    for k, v in rec.items():
        if isinstance(v, str):
            print(f"\n[{k}] (str, len={len(v)}):")
            print(v[:1500])
        elif isinstance(v, list):
            print(f"\n[{k}] (list, len={len(v)}):")
            if v:
                print(f"  first item type: {type(v[0]).__name__}")
                if isinstance(v[0], dict):
                    print(f"  first item keys: {list(v[0].keys())}")
                    print(f"  first item: {json.dumps(v[0], ensure_ascii=False)[:800]}")
                    if len(v) > 1:
                        print(f"  second item: {json.dumps(v[1], ensure_ascii=False)[:400]}")
                    print(f"  last item: {json.dumps(v[-1], ensure_ascii=False)[:400]}")
                else:
                    print(f"  first item: {str(v[0])[:500]}")
        elif isinstance(v, dict):
            print(f"\n[{k}] (dict, keys={list(v.keys())}):")
            print(json.dumps(v, ensure_ascii=False)[:800])
        else:
            print(f"\n[{k}] ({type(v).__name__}): {v!r}")


def main() -> None:
    files = list_repo()
    paths = download(files)

    print("\n" + "=" * 70)
    print("SCHEMA SUMMARIES")
    print("=" * 70)

    # Summarize every file briefly
    for name, p in paths.items():
        if p.suffix == ".jsonl":
            summarize_jsonl(p)
        elif p.suffix == ".csv":
            summarize_csv(p)
        else:
            print(f"\n--- {p.name} ({p.suffix}) size={p.stat().st_size} ---")

    # Deep dive on 32k shared_contexts to understand session + demographics layout
    for name, p in paths.items():
        if "shared_contexts_32k" in name:
            deep_dive_first_context(p)
            break


if __name__ == "__main__":
    main()
