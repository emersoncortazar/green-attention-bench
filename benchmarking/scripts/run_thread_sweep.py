from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=os.path.join("models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"))
    ap.add_argument("--out_dir", default=os.path.join("data"))

    ap.add_argument("--prompt_label", default="pt_2048")
    ap.add_argument("--n_ctx", type=int, default=8192)
    ap.add_argument("--max_tokens", type=int, default=128)

    ap.add_argument("--mode", choices=["cold", "warm"], default="warm")
    ap.add_argument("--threads", default="1,2,4,8,12")

    ap.add_argument("--cold_runs", type=int, default=2)
    ap.add_argument("--warmup_runs", type=int, default=1)
    ap.add_argument("--warm_runs", type=int, default=5)

    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    py = sys.executable
    thread_list = [int(x.strip()) for x in args.threads.split(",") if x.strip()]

    for th in thread_list:
        cmd = [
            py, "-m", "greenbench.cli",
            "--model_path", os.path.join(repo_root, args.model_path),
            "--prompt_label", args.prompt_label,
            "--n_ctx", str(args.n_ctx),
            "--n_threads", str(th),
            "--max_tokens", str(args.max_tokens),
            "--mode", args.mode,
            "--out_dir", os.path.join(repo_root, args.out_dir),
            "--cold_runs", str(args.cold_runs),
            "--warmup_runs", str(args.warmup_runs),
            "--warm_runs", str(args.warm_runs),
        ]
        print(f"\n=== THREAD RUN: n_threads={th} ===")
        subprocess.check_call(cmd, cwd=os.path.join(repo_root, "benchmarking"))

    print("\nDone. CSVs are in:", os.path.join(repo_root, args.out_dir))


if __name__ == "__main__":
    main()
