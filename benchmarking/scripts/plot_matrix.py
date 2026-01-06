#!/usr/bin/env python3
"""
plot_matrix.py

Plot key benchmark metrics from data/results_summary.csv.

Generates 3 figures:
  - ttft_vs_ctx_*.png
  - decode_tok_s_vs_ctx_*.png
  - rss_peak_mb_vs_ctx_*.png

Lines are grouped by prompt_label, x-axis is n_ctx.

Usage (Windows cmd.exe):
  python -m benchmarking.scripts.plot_matrix --csv data\\results_summary.csv --out_dir data\\plots --mode warm

PowerShell:
  python -m benchmarking.scripts.plot_matrix --csv data/results_summary.csv --out_dir data/plots --mode warm
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt


def _to_int(s) -> int:
    try:
        return int(str(s).strip())
    except Exception:
        return 0


def _to_float_or_nan(s) -> float:
    """
    Convert CSV cell -> float.
    Treat empty / missing / 'nan' / 'None' as NaN.
    """
    if s is None:
        return float("nan")
    t = str(s).strip()
    if t == "" or t.lower() in {"nan", "none", "null"}:
        return float("nan")
    try:
        return float(t)
    except Exception:
        return float("nan")


def read_results_csv(csv_path: Path) -> List[dict]:
    """
    Minimal CSV reader that avoids pandas dependency.
    Returns list of dict rows.
    """
    import csv

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows: List[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r:
                continue
            rows.append(r)
    return rows


def filter_rows(
    rows: List[dict],
    mode: Optional[str],
    model_path: Optional[str],
    prompt_labels: Optional[List[str]],
    ctx_list: Optional[List[int]],
) -> List[dict]:
    out = rows

    if mode:
        out = [r for r in out if str(r.get("mode", "")).strip() == mode]

    if model_path:
        mp = model_path.replace("/", "\\").lower()
        out = [
            r
            for r in out
            if str(r.get("model_path", "")).replace("/", "\\").lower() == mp
        ]

    if prompt_labels:
        want = set(prompt_labels)
        out = [r for r in out if str(r.get("prompt_label", "")).strip() in want]

    if ctx_list:
        want_ctx = set(ctx_list)
        out = [r for r in out if _to_int(r.get("n_ctx", 0)) in want_ctx]

    return out


def dedupe_latest(rows: List[dict]) -> List[dict]:
    """
    If you re-ran the matrix, you may have duplicate (model_path, prompt_label, n_ctx, mode, n_threads, max_tokens).
    Keep the last occurrence in the CSV (most recent append).
    """
    key_fields = ("model_path", "prompt_label", "n_ctx", "mode", "n_threads", "max_tokens")
    seen = {}
    for r in rows:
        k = tuple(str(r.get(f, "")).strip() for f in key_fields)
        seen[k] = r
    return list(seen.values())


def plot_lines_by_prompt(
    rows: List[dict],
    x_field: str,
    y_field: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    groups = {}
    for r in rows:
        pl = str(r.get("prompt_label", "")).strip()
        if not pl:
            continue
        groups.setdefault(pl, []).append(r)

    if not groups:
        raise ValueError(f"No rows to plot for {y_field}. Check filters.")

    plt.figure()

    def pl_key(pl: str):
        # try to sort pt_64, pt_256, ...
        try:
            return int(pl.split("_", 1)[1])
        except Exception:
            return 10**9

    for pl in sorted(groups.keys(), key=pl_key):
        rs = groups[pl]
        rs_sorted = sorted(rs, key=lambda r: _to_int(r.get(x_field, 0)))

        xs = [_to_int(r.get(x_field, 0)) for r in rs_sorted]
        ys = [_to_float_or_nan(r.get(y_field, None)) for r in rs_sorted]

        # Keep only finite y
        clean = [(x, y) for x, y in zip(xs, ys) if math.isfinite(y)]
        if len(clean) < 1:
            continue

        xs2, ys2 = zip(*clean)
        plt.plot(xs2, ys2, marker="o", label=pl)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/results_summary.csv", help="Path to results_summary.csv")
    ap.add_argument("--out_dir", default="data/plots", help="Directory to write plots into")
    ap.add_argument("--mode", default="warm", choices=["warm", "cold", "both"], help="Filter by mode")
    ap.add_argument("--model_path", default=None, help="Optional exact model_path filter")
    ap.add_argument(
        "--prompt_labels",
        nargs="*",
        default=None,
        help="Optional subset: e.g. pt_64 pt_256 pt_512 pt_1024 pt_2048 pt_4096",
    )
    ap.add_argument(
        "--ctx_list",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset: e.g. 2048 4096 8192",
    )

    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)

    rows = read_results_csv(csv_path)

    mode_filter = None if args.mode == "both" else args.mode
    rows = filter_rows(
        rows,
        mode=mode_filter,
        model_path=args.model_path,
        prompt_labels=args.prompt_labels,
        ctx_list=args.ctx_list,
    )
    rows = dedupe_latest(rows)

    if not rows:
        raise SystemExit("No rows matched your filters. Check --csv/--mode/--prompt_labels/--ctx_list.")

    suffix = f"mode_{args.mode}"
    if args.model_path:
        suffix += "_model_filtered"

    ttft_png = out_dir / f"ttft_vs_ctx_{suffix}.png"
    decode_png = out_dir / f"decode_tok_s_vs_ctx_{suffix}.png"
    rss_png = out_dir / f"rss_peak_mb_vs_ctx_{suffix}.png"

    plot_lines_by_prompt(
        rows=rows,
        x_field="n_ctx",
        y_field="mean_ttft_s",
        title="TTFT vs Context Length (lines by prompt size)",
        xlabel="n_ctx (tokens)",
        ylabel="mean_ttft_s (seconds)",
        out_path=ttft_png,
    )

    plot_lines_by_prompt(
        rows=rows,
        x_field="n_ctx",
        y_field="mean_decode_tok_s",
        title="Decode Throughput vs Context Length (lines by prompt size)",
        xlabel="n_ctx (tokens)",
        ylabel="mean_decode_tok_s (tokens/sec)",
        out_path=decode_png,
    )

    plot_lines_by_prompt(
        rows=rows,
        x_field="n_ctx",
        y_field="mean_rss_peak_mb",
        title="Peak RSS vs Context Length (lines by prompt size)",
        xlabel="n_ctx (tokens)",
        ylabel="mean_rss_peak_mb (MB)",
        out_path=rss_png,
    )

    print(f"Wrote:\n  {ttft_png}\n  {decode_png}\n  {rss_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
