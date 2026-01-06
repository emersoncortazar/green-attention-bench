from __future__ import annotations

import os
from itertools import product

PROMPTS = ["pt_64", "pt_256", "pt_512", "pt_1024", "pt_2048", "pt_4096"]
CTXS = [2048, 4096, 8192]
MODES = ["cold", "warm"]

TEMPLATE = """\

PROMPT_LABEL = "{prompt_label}"
N_CTX = {n_ctx}
MODE = "{mode}"

cmd = [
    sys.executable,
    os.path.join("benchmarking", "bench_runner.py"),
    "--model_path", MODEL_PATH,
    "--prompt_label", PROMPT_LABEL,
    "--n_ctx", str(N_CTX),
    "--mode", MODE,
    "--n_threads", "8",
    "--max_tokens", "128",
    "--cold_runs", "2",
    "--warmup_runs", "1",
    "--warm_runs", "5",
    "--out_dir", "data",
]

"""

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(repo_root, "benchmarking", "bench_py")
    os.makedirs(out_dir, exist_ok=True)

    for prompt_label, n_ctx, mode in product(PROMPTS, CTXS, MODES):
        fname = f"{prompt_label}_ctx_{n_ctx}_{mode}.py"
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(TEMPLATE.format(prompt_label=prompt_label, n_ctx=n_ctx, mode=mode))

    print("Wrote runner files to:", out_dir)

if __name__ == "__main__":
    main()
