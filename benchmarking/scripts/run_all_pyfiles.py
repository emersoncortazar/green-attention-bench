from __future__ import annotations

import os
import subprocess
import sys

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    bench_dir = os.path.join(repo_root, "benchmarking", "bench_py")

    if not os.path.isdir(bench_dir):
        raise FileNotFoundError(f"Missing {bench_dir}. Run make_bench_pyfiles.py first.")

    pyfiles = sorted([p for p in os.listdir(bench_dir) if p.endswith(".py")])

    for fname in pyfiles:
        path = os.path.join(bench_dir, fname)
        print(f"\n=== RUNNING {fname} ===")

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([
            os.path.join(repo_root, "benchmarking", "src"),
            env.get("PYTHONPATH", "")
        ])

        subprocess.check_call([sys.executable, path], cwd=repo_root, env=env)

    print("\nAll done. CSV outputs are in data/.")

if __name__ == "__main__":
    main()
