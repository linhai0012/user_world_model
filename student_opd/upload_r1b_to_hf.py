#!/usr/bin/env python
"""
Upload Phase 2b Round 1b FINAL dual-LoRA checkpoints (4 focal personas) to
HuggingFace Hub. Migrates from Isambard-AI (expiring) to be pullable on
KCL CREATE.

Uploads only the top-level slow/ and fast/ adapters for each persona;
intermediate ckpt-step-*/ directories are ignored.

Repo layout on HF:
    lzhang472/P-OPSD-Qwen3-4B-R1b-focal4/
    ├── pid0/{slow,fast}/
    ├── pid4/{slow,fast}/
    ├── pid12/{slow,fast}/
    └── pid14/{slow,fast}/

Usage (on Isambard, after `huggingface-cli login` or HF_TOKEN set):
    python dynamic_usersim/student_opd/upload_r1b_to_hf.py
    # or dry-run (print plan, no upload):
    python dynamic_usersim/student_opd/upload_r1b_to_hf.py --dry-run
    # or change personas / repo:
    python dynamic_usersim/student_opd/upload_r1b_to_hf.py \
        --personas 0 4 --repo-id lzhang472/something-else

Download on KCL CREATE (later):
    huggingface-cli download lzhang472/P-OPSD-Qwen3-4B-R1b-focal4 \
        --local-dir $SCRATCHDIR/P-OPSD/student_lora_r1b
"""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


DEFAULT_REPO_ID = "lzhang472/P-OPSD-Qwen3-4B-R1b-focal4"
DEFAULT_PERSONAS = [0, 4, 12, 14]
DEFAULT_TAG = "dual_v2_ep1_r3teacher"


README_TEMPLATE = """---
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
tags:
- lora
- personalization
- user-simulator
- personamem
---

# P-OPSD — Phase 2b Round 1b dual-LoRA (4 focal personas)

Per-user dual LoRA adapters trained via on-policy distillation (OPD) against
an SFT teacher (R3: Qwen3-4B-Instruct-2507 + K=3 context). Best universal
recipe from the P-OPSD Phase 2b sweep — see `EXPERIMENTS.md` in the repo.

## Structure

Each `pid{{N}}/` directory holds a pair of LoRA adapters that must be
loaded **together** (dual LoRA, not independent):

- `slow/` — MLP path, rank 32, lr 5e-5
- `fast/` — Attention path, rank 16, lr 2e-4

Base model: `Qwen/Qwen3-4B-Instruct-2507`.

## Personas included

| pid | name          | best MCQ-PPL closure (128k) |
|----:|---------------|-----------------------------|
|   0 | Kanoa Manu    | +25%                        |
|   4 | Lisa Johnson  | +90%                        |
|  12 | Jordan Ellis  | +109% (exceeds teacher_k3)  |
|  14 | Leilani Hayes | +88% (monotonic decay, see notes) |

## Loading (example)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507", torch_dtype="bfloat16"
)
model = PeftModel.from_pretrained(base, "{repo_id}", subfolder="pid4/slow", adapter_name="slow")
model.load_adapter("{repo_id}", subfolder="pid4/fast", adapter_name="fast")
model.set_adapter(["slow", "fast"])
```

Source: https://github.com/your-org/P-OPSD (see commit log referenced in
`EXPERIMENTS.md §11`).
"""


def iter_upload_targets(scratch: Path, personas: list[int], tag: str):
    """Yield (local_path, path_in_repo) tuples for slow/ and fast/ only."""
    missing = []
    for pid in personas:
        persona_root = scratch / "P-OPSD" / "student_lora" / f"lora_pid{pid}_{tag}"
        for which in ("slow", "fast"):
            local = persona_root / which
            if not local.is_dir():
                missing.append(local)
                continue
            yield local, f"pid{pid}/{which}"
    if missing:
        print("\n[!] MISSING dirs:", file=sys.stderr)
        for m in missing:
            print(f"    {m}", file=sys.stderr)
        print(
            "Not fatal — will upload the ones that exist. "
            "Re-run with --fail-on-missing to abort instead.\n",
            file=sys.stderr,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--personas", type=int, nargs="+", default=DEFAULT_PERSONAS)
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help="training tag (default: %(default)s)")
    ap.add_argument("--scratch", default=os.environ.get("SCRATCHDIR", ""),
                    help="overrides $SCRATCHDIR")
    ap.add_argument("--private", action="store_true",
                    help="create repo as private (default: private)")
    ap.add_argument("--public", action="store_true",
                    help="create repo as public")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the upload plan without uploading")
    ap.add_argument("--fail-on-missing", action="store_true")
    ap.add_argument("--skip-readme", action="store_true")
    args = ap.parse_args()

    if not args.scratch:
        print("ERROR: $SCRATCHDIR not set and --scratch not given", file=sys.stderr)
        return 1
    scratch = Path(args.scratch)
    if not scratch.is_dir():
        print(f"ERROR: scratch dir not found: {scratch}", file=sys.stderr)
        return 1

    private = not args.public  # default private
    if args.private and args.public:
        print("ERROR: --private and --public are mutually exclusive", file=sys.stderr)
        return 1

    targets = list(iter_upload_targets(scratch, args.personas, args.tag))
    if args.fail_on_missing:
        expected = 2 * len(args.personas)
        if len(targets) != expected:
            print(f"ERROR: {expected - len(targets)} dirs missing", file=sys.stderr)
            return 1
    if not targets:
        print("ERROR: nothing to upload", file=sys.stderr)
        return 1

    print("=" * 60)
    print(f"Repo:      {args.repo_id}  (private={private})")
    print(f"Scratch:   {scratch}")
    print(f"Personas:  {args.personas}")
    print(f"Tag:       {args.tag}")
    print(f"Dry run:   {args.dry_run}")
    print("=" * 60)
    print("Upload plan:")
    total_files = 0
    total_bytes = 0
    for local, remote in targets:
        size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
        n = sum(1 for f in local.rglob("*") if f.is_file())
        print(f"  {local}")
        print(f"    -> {args.repo_id}:{remote}  ({n} files, {size/1e6:.1f} MB)")
        total_files += n
        total_bytes += size
    print(f"\nTotal: {total_files} files, {total_bytes/1e6:.1f} MB")
    print("=" * 60)

    if args.dry_run:
        print("dry-run — exiting without upload")
        return 0

    if not os.environ.get("HF_TOKEN") and not (Path.home() / ".cache/huggingface/token").exists():
        print(
            "ERROR: no HF token found. Run `huggingface-cli login` first, "
            "or export HF_TOKEN=hf_...",
            file=sys.stderr,
        )
        return 1

    api = HfApi()
    create_repo(args.repo_id, repo_type="model", exist_ok=True, private=private)
    print(f"repo ready: https://huggingface.co/{args.repo_id}")

    for local, remote in targets:
        print(f"\n>>> uploading {local} -> {remote}")
        api.upload_folder(
            folder_path=str(local),
            repo_id=args.repo_id,
            path_in_repo=remote,
            commit_message=f"R1b final dual-LoRA: {remote}",
            ignore_patterns=["ckpt-step-*", "*.log", "__pycache__"],
        )
        print(f"    done.")

    if not args.skip_readme:
        readme_text = README_TEMPLATE.format(repo_id=args.repo_id)
        readme_path = Path("/tmp") / "P-OPSD-R1b-README.md"
        readme_path.write_text(readme_text)
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            commit_message="add README",
        )
        print("    README.md uploaded.")

    print(f"\nAll done. View at: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
