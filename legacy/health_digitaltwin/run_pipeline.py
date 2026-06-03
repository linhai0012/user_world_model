#!/usr/bin/env python3
"""
run_pipeline.py — Full data-construction pipeline for the
LLM-based User Digital Twin experiment.

Usage:
    # Run all steps
    python run_pipeline.py --all

    # Run individual steps
    python run_pipeline.py --step 1        # parse PMData
    python run_pipeline.py --step 2        # synthesize text via GPT-5.4
    python run_pipeline.py --step 3        # build omnimodal training data

    # Quick sanity check with a small subset
    python run_pipeline.py --all --limit 20

Prerequisites:
    export OPENAI_API_KEY="sk-..."
    export PMDATA_ROOT="./pmdata"          # path to extracted PMData

    pip install openai numpy
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Digital Twin Data Pipeline: PMData → Omnimodal Training Data"
    )
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3],
        help="Run a single step (1=parse, 2=synthesize, 3=build).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all three steps sequentially.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of records to process (for debugging).",
    )
    args = parser.parse_args()

    if not args.step and not args.all:
        parser.print_help()
        sys.exit(0)

    steps_to_run = [1, 2, 3] if args.all else [args.step]

    for step in steps_to_run:
        t0 = time.time()
        log.info(f"{'='*50}")
        log.info(f"  STEP {step}")
        log.info(f"{'='*50}")

        if step == 1:
            import step1_parse_pmdata as s1
            records = s1.run()

            if args.limit and records:
                from config import CACHE_DIR
                records = records[: args.limit]
                out = CACHE_DIR / "parsed_records.json"
                out.write_text(json.dumps(records, indent=2, ensure_ascii=False))
                log.info(f"  (limited to {len(records)} records)")

        elif step == 2:
            import step2_synthesize_text as s2

            if args.limit:
                from config import CACHE_DIR
                in_path = CACHE_DIR / "parsed_records.json"
                records = json.loads(in_path.read_text())
                records = records[: args.limit]
                in_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
                log.info(f"  (limited to {len(records)} records)")

            s2.run()

        elif step == 3:
            import step3_build_training_data as s3
            s3.run()

        elapsed = time.time() - t0
        log.info(f"  Step {step} done in {elapsed:.1f}s\n")

    log.info("Pipeline finished.")


if __name__ == "__main__":
    main()
