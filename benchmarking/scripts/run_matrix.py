from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

# Ensure "benchmarking/src" is importable even if PYTHONPATH isn't set
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "benchmarking" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from greenbench.runner import BenchConfig, run_condition  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GreenAttentionBench across prompt/context grid and append CSV summaries.")

    p.add_argument("--model_path", required=True, help="Path to .gguf model file")
    p.add_argument("--modes", nargs="+", default=["warm"], choices=["warm", "cold"], help="Modes to run")
    p.add_argument("--prompt_labels", nargs="+", required=True, help="Prompt labels, e.g. pt_64 pt_256 ...")
    p.add_argument("--ctx_list", nargs="+", type=int, required=True, help="Context sizes, e.g. 2048 4096 8192")

    p.add_argument("--n_threads", type=int, default=8)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--warmup_runs", type=int, default=1)
    p.add_argument("--warm_runs", type=int, default=5)
    p.add_argument("--cold_runs", type=int, default=3)
    p.add_argument("--ttft_trigger_chars", type=int, default=8)

    p.add_argument("--out_root", default="data/runs", help="Root output directory where per-condition folders are created")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for mode in args.modes:
        for pl in args.prompt_labels:
            for ctx in args.ctx_list:
                name = f"{pl}_ctx{ctx}_{mode}"
                out_dir = out_root / name

                print(f"\n=== RUN {name} ===")

                cfg = BenchConfig(
                    model_path=args.model_path,
                    prompt_label=pl,
                    n_ctx=int(ctx),
                    mode=mode,
                    n_threads=int(args.n_threads),
                    max_tokens=int(args.max_tokens),
                    cold_runs=int(args.cold_runs),
                    warmup_runs=int(args.warmup_runs),
                    warm_runs=int(args.warm_runs),
                    out_dir=str(out_dir),
                    ttft_trigger_chars=int(args.ttft_trigger_chars),
                )

                _ = run_condition(cfg)

    print("\nDone. Global summary appended to: data/results_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
