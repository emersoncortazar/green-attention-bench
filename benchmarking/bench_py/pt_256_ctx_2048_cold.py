import os, sys, subprocess

# Auto-generated runner file
PROMPT_LABEL = "pt_256"
N_CTX = 2048
MODE = "cold"

# Adjust if your model filename differs:
MODEL_PATH = os.path.join("models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

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

# Ensure imports work: benchmarking/src must be importable (so bench_runner can import greenbench.prompts)
env = os.environ.copy()
env["PYTHONPATH"] = os.pathsep.join([
    os.path.join(os.getcwd(), "benchmarking", "src"),
    env.get("PYTHONPATH", "")
])

subprocess.check_call(cmd, env=env)
