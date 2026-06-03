#!/usr/bin/env python3
"""
generate_and_eval.py — Generate outputs from the trained model and evaluate.

Usage:
    # Single GPU (batched generation)
    python generate_and_eval.py \
        --model_path ./checkpoints/final \
        --test_file ./output/test.jsonl

    # Multi-GPU data parallel (recommended for 4×H200)
    torchrun --nproc_per_node=4 generate_and_eval.py \
        --model_path ./checkpoints/final \
        --test_file ./output/test.jsonl

    # Limit samples for quick test
    python generate_and_eval.py \
        --model_path ./checkpoints/final \
        --test_file ./output/test.jsonl \
        --num_samples 20 --batch_size 4
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from metrics import (
    parse_generated_output,
    compute_text_metrics,
    compute_state_metrics,
    compute_consistency_v3,
    evaluate_batch,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def load_model(model_path: str, local_rank: int):
    """Load fine-tuned model and tokenizer onto a specific GPU."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    )
    model.to(f"cuda:{local_rank}")
    model.eval()
    return model, tokenizer


def generate_batch(
    model, tokenizer, input_texts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
) -> list[str]:
    """Generate outputs for a batch of inputs."""
    # Build prompts — input_text already contains baseline HR + event + duration
    prompts = []
    for text in input_texts:
        prompt = f"{text}\n\n### Response:\n"
        prompts.append(prompt)

    # Tokenize with left-padding for batched generation
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
        )

    # Decode only the generated part for each sample
    results = []
    for i in range(len(input_texts)):
        prompt_len = inputs.input_ids[i].ne(tokenizer.pad_token_id).sum().item()
        generated = output_ids[i][prompt_len:]
        results.append(tokenizer.decode(generated, skip_special_tokens=False))

    return results


def evaluate_samples(all_predictions: list[str], test_samples: list[dict]) -> tuple[list[dict], dict]:
    """Compute per-sample and aggregate metrics."""
    all_details = []

    for sample, generated in zip(test_samples, all_predictions):
        parsed = parse_generated_output(generated)

        ref_parsed = parse_generated_output(sample["output_text"])
        text_metrics = compute_text_metrics(
            parsed["user_feedback"], ref_parsed["user_feedback"]
        )

        # V3: state prediction metrics
        true_state = sample.get("wellness_day_n1", {})
        day_n_state = sample.get("wellness_day_n", {})
        state_metrics = compute_state_metrics(
            parsed.get("predicted_state", {}), true_state, day_n_state
        )

        consistency = compute_consistency_v3(
            parsed["user_feedback"], parsed.get("predicted_state", {})
        )

        all_details.append({
            "input": sample["input_text"],
            "generated": generated,
            "parsed_text": parsed["user_feedback"],
            "predicted_state": parsed.get("predicted_state", {}),
            "true_state_n1": true_state,
            "true_state_n": day_n_state,
            "parse_ok": parsed["parse_ok"],
            **state_metrics,
            **text_metrics,
            **consistency,
        })

    aggregate = evaluate_batch(all_predictions, test_samples)
    return all_details, aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_file", default="./output/test.jsonl")
    parser.add_argument("--output_file", default=None,
                        help="Output path. Auto-generated if not specified.")
    parser.add_argument("--run_name", default=None,
                        help="Run name for output file, e.g. 'A1_alpha05'")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="0 = all samples")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # Auto-generate output filename if not specified
    if args.output_file is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(args.model_path).name
        run_tag = f"_{args.run_name}" if args.run_name else ""
        args.output_file = f"./output/eval_{model_name}{run_tag}_{timestamp}.json"

    # ── Distributed setup ──────────────────────────────────────────────
    distributed = "RANK" in __import__("os").environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(__import__("os").environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0

    is_main = rank == 0

    # ── Load model (each rank loads onto its own GPU) ──────────────────
    if is_main:
        log.info(f"Loading model from {args.model_path} ...")
        log.info(f"Distributed: {distributed}, world_size={world_size}")
    model, tokenizer = load_model(args.model_path, local_rank)

    # ── Load test data ─────────────────────────────────────────────────
    test_samples = []
    with open(args.test_file, "r") as f:
        for line in f:
            if line.strip():
                test_samples.append(json.loads(line))

    if args.num_samples > 0:
        test_samples = test_samples[:args.num_samples]

    # ── Shard data across ranks ────────────────────────────────────────
    # Each rank processes its shard independently
    shard = test_samples[rank::world_size]
    if is_main:
        log.info(f"Total: {len(test_samples)} samples, this rank: {len(shard)} samples, batch_size={args.batch_size}")

    # ── Batched generation ─────────────────────────────────────────────
    all_predictions = []
    pbar = tqdm(range(0, len(shard), args.batch_size), desc=f"[rank {rank}] Generating", disable=not is_main)

    for batch_start in pbar:
        batch_samples = shard[batch_start : batch_start + args.batch_size]
        batch_texts = [s["input_text"] for s in batch_samples]

        batch_preds = generate_batch(
            model, tokenizer, batch_texts,
            max_new_tokens=args.max_new_tokens,
        )
        all_predictions.extend(batch_preds)

    # ── Gather results from all ranks via filesystem ─────────────────
    if distributed:
        import tempfile, os
        tmp_dir = Path(args.output_file).parent / ".eval_shards"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        shard_file = tmp_dir / f"shard_{rank}.json"
        shard_file.write_text(json.dumps(
            {"predictions": all_predictions, "samples": shard},
            ensure_ascii=False,
        ))
        dist.barrier()  # wait for all ranks to write

        if is_main:
            final_preds = []
            final_samples = []
            # Interleave shards back to original order
            shards = []
            for r in range(world_size):
                with open(tmp_dir / f"shard_{r}.json", encoding="utf-8") as sf:
                    shards.append(json.load(sf))
            max_shard_len = max(len(s["predictions"]) for s in shards)
            for i in range(max_shard_len):
                for r in range(world_size):
                    if i < len(shards[r]["predictions"]):
                        final_preds.append(shards[r]["predictions"][i])
                        final_samples.append(shards[r]["samples"][i])
            # Cleanup shard files
            for r in range(world_size):
                (tmp_dir / f"shard_{r}.json").unlink(missing_ok=True)
            tmp_dir.rmdir()
        dist.barrier()
    else:
        final_preds = all_predictions
        final_samples = shard

    # ── Evaluate and save (main rank only) ─────────────────────────────
    if is_main:
        all_details, aggregate = evaluate_samples(final_preds, final_samples)

        results = {
            "config": {
                "model_path": args.model_path,
                "test_file": args.test_file,
                "num_samples": len(final_samples),
                "batch_size": args.batch_size,
                "world_size": world_size,
            },
            "aggregate_metrics": aggregate,
            "per_sample": all_details,
        }

        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))

        # Print summary
        print("\n" + "=" * 60)
        print("  EVALUATION RESULTS")
        print("=" * 60)
        for k, v in aggregate.items():
            print(f"  {k:30s}  {v}")
        print("=" * 60)

        # Show a few examples
        print("\n── Sample Outputs ──\n")
        for i, d in enumerate(all_details[:3]):
            print(f"[Sample {i+1}]")
            print(f"  Input:  {d['input'][:120]}...")
            print(f"  Text:   {d['parsed_text'][:120]}...")
            pred_s = d.get("predicted_state", {})
            true_s = d.get("true_state_n1", {})
            day_n_s = d.get("true_state_n", {})
            if pred_s:
                print(f"  Pred state:  {pred_s}")
                print(f"  True state:  {true_s}")
                print(f"  Day N state: {day_n_s}")
            print()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
