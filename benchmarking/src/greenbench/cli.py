from __future__ import annotations

import argparse
import json
import os
import sys

from .runner import BenchConfig, run_condition


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="greenbench.cli", description="GreenAttentionBench CLI (single condition).")

    p.add_argument("--model_path", required=True, help="Path to GGUF model file.")
    p.add_argument("--prompt_label", required=True, help="Prompt label, e.g. pt_64, pt_256, pt_2048.")
    p.add_argument("--n_ctx", type=int, required=True, help="Context length to set for llama.cpp.")
    p.add_argument("--mode", choices=["warm", "cold"], required=True, help="warm=load once, cold=reload each run")

    p.add_argument("--n_threads", type=int, default=8, help="CPU threads for llama.cpp.")
    p.add_argument("--max_tokens", type=int, default=128, help="Max new tokens to generate.")
    p.add_argument("--ttft_trigger_chars", type=int, default=8, help="Chars of output needed to count TTFT.")

    p.add_argument("--warmup_runs", type=int, default=0, help="Warmup runs (warm mode only, not recorded).")
    p.add_argument("--warm_runs", type=int, default=5, help="Recorded runs (warm mode).")
    p.add_argument("--cold_runs", type=int, default=5, help="Recorded runs (cold mode).")

    p.add_argument("--out_dir", required=True, help="Output directory for CSV files.")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # normalize model path
    model_path = args.model_path
    if not os.path.exists(model_path):
        model_path = os.path.normpath(model_path)

    cfg = BenchConfig(
        model_path=model_path,
        prompt_label=args.prompt_label,
        n_ctx=int(args.n_ctx),
        mode=str(args.mode),
        n_threads=int(args.n_threads),
        max_tokens=int(args.max_tokens),
        warmup_runs=int(args.warmup_runs),
        warm_runs=int(args.warm_runs),
        cold_runs=int(args.cold_runs),
        ttft_trigger_chars=int(args.ttft_trigger_chars),
        out_dir=str(args.out_dir),
    )

    summ = run_condition(cfg)

    print(json.dumps({
        "model_path": summ.model_path,
        "model_sha1": summ.model_sha1,
        "prompt_label": summ.prompt_label,
        "prompt_tokens": summ.prompt_tokens,
        "n_ctx": summ.n_ctx,
        "mode": summ.mode,
        "n_threads": summ.n_threads,
        "max_tokens": summ.max_tokens,
        "ttft_trigger_chars": summ.ttft_trigger_chars,
        "runs": summ.runs,
        "mean_ttft_s": summ.mean_ttft_s,
        "mean_decode_tok_s": summ.mean_decode_tok_s,
        "mean_total_s": summ.mean_total_s,
        "mean_rss_peak_mb": summ.mean_rss_peak_mb,
        "platform": summ.platform,
        "python": summ.python,
        "cpu_count_logical": summ.cpu_count_logical,
        "machine": summ.machine,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
