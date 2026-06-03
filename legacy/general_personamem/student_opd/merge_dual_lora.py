"""One-time merge of an R1b dual LoRA into base, saved as a plain HF model
dir that vLLM / transformers can load directly (no PEFT needed).

Why a separate step:
  - vLLM natively supports single-adapter LoRA but handling dual (slow + fast
    activated together) is fiddly. Much simpler to pre-merge.
  - Merging is a one-time cost (~30s); generation gets reused many times
    with different temperature / probes / samples on the merged checkpoint.

Output layout (compatible with HF from_pretrained and vLLM LLM(...)):
    <out-dir>/
      config.json
      model-*.safetensors   (merged base+slow+fast weights, bf16, ~8GB)
      tokenizer.json + tokenizer_config.json + generation_config.json
      ...

Usage (Isambard-era env vars respected via setup_kcl.sh):
    source setup_kcl.sh
    python dynamic_usersim/student_opd/merge_dual_lora.py \\
        --base-model $QWEN3_BASE \\
        --lora-path  $LORA_ROOT/pid4 \\
        --out-dir    $SCRATCHDIR/P-OPSD/merged_r1b/pid4
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=Path, required=True,
                    help="local HF model dir (e.g. $QWEN3_BASE)")
    ap.add_argument("--lora-path", type=Path, required=True,
                    help="dir containing slow/ and fast/ subdirs (R1b dual)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="where to save the merged HF checkpoint")
    args = ap.parse_args()

    slow = args.lora_path / "slow"
    fast = args.lora_path / "fast"
    if not (slow / "adapter_config.json").exists():
        raise FileNotFoundError(slow)
    if not (fast / "adapter_config.json").exists():
        raise FileNotFoundError(fast)

    print(f"[merge] base  : {args.base_model}")
    print(f"[merge] slow  : {slow}")
    print(f"[merge] fast  : {fast}")
    print(f"[merge] output: {args.out_dir}")

    print("[merge] loading base...")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Merge on GPU (CPU merge SIGKILL'd on KCL cgroup; same fix as eval_opd).
    model.to(torch.cuda.current_device())

    print("[merge] attaching slow adapter...")
    model = PeftModel.from_pretrained(model, str(slow), adapter_name="slow")
    print("[merge] attaching fast adapter...")
    model.load_adapter(str(fast), adapter_name="fast")
    model.base_model.set_adapter(["slow", "fast"])

    print("[merge] merge_and_unload (this is the compute-heavy step)...")
    model = model.merge_and_unload()

    # Move merged model to CPU BEFORE save_pretrained. Without this, HF
    # calls state_dict() which copies the full ~8GB from GPU to CPU in one
    # shot as a fresh allocation; under KCL cgroup CPU-RAM limits that
    # spike is the SIGKILL trigger. Moving to CPU first streams the
    # transfer parameter-by-parameter, and save_pretrained then writes
    # from existing CPU tensors without re-allocating.
    print("[merge] moving merged model to CPU for safe save...")
    model = model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving sharded weights (2GB/shard) to {args.out_dir}...")
    model.save_pretrained(
        str(args.out_dir),
        safe_serialization=True,
        max_shard_size="2GB",
    )

    tok = AutoTokenizer.from_pretrained(str(args.base_model),
                                          trust_remote_code=True)
    tok.save_pretrained(str(args.out_dir))

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("[merge] done.")


if __name__ == "__main__":
    main()
