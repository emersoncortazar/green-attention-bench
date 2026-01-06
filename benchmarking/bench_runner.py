from __future__ import annotations

import csv
import datetime as dt
import hashlib
import os
import platform
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import psutil
from llama_cpp import Llama

from benchmarking.src.greenbench.prompts import PROMPTS


def append_rows_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})


def _pctl(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = (len(sorted_vals) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def summarize_rows(rows: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        vals_sorted = sorted(vals)
        if not vals_sorted:
            out[f"{k}_mean"] = 0.0
            out[f"{k}_std"] = 0.0
            out[f"{k}_p50"] = 0.0
            out[f"{k}_p95"] = 0.0
            continue
        out[f"{k}_mean"] = float(statistics.mean(vals_sorted))
        out[f"{k}_std"] = float(statistics.pstdev(vals_sorted)) if len(vals_sorted) > 1 else 0.0
        out[f"{k}_p50"] = _pctl(vals_sorted, 0.50)
        out[f"{k}_p95"] = _pctl(vals_sorted, 0.95)
    return out


def count_prompt_tokens(llm: Llama, prompt: str, add_bos: bool) -> int:
    return len(llm.tokenize(prompt.encode("utf-8"), add_bos=add_bos))


def count_gen_tokens_no_bos(llm: Llama, text: str) -> int:
    # Count output tokens without BOS
    if not text:
        return 0
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False))



@dataclass
class RunStats:
    ttft_first_token_s: float
    ttft_trigger_s: float
    decode_tps: float
    total_s: float
    peak_rss_mb: float
    mem_before_mb: float
    mem_after_mb: float
    gen_tokens_no_bos: int
    collected_chars: int


def run_ttft_and_decode(
    llm: Llama,
    prompt: str,
    max_tokens: int,
    *,
    ttft_trigger_chars: int,
    debug: bool,
) -> RunStats:
    process = psutil.Process()

    mem_before_mb = process.memory_info().rss / 1e6
    peak_rss_mb = mem_before_mb

    t0 = time.time()

    first_token_time: Optional[float] = None
    trigger_time: Optional[float] = None

    collected_parts: List[str] = []
    collected_len = 0

    stream = llm(
        prompt,
        max_tokens=max_tokens,
        stream=True,
        temperature=0.7,
    )

    for chunk in stream:
        now = time.time()
        rss_mb = process.memory_info().rss / 1e6
        if rss_mb > peak_rss_mb:
            peak_rss_mb = rss_mb

        txt = ""
        try:
            txt = chunk["choices"][0].get("text", "") or ""
        except Exception:
            txt = ""

        if txt:
            if first_token_time is None:
                first_token_time = now
            collected_parts.append(txt)
            collected_len += len(txt)

            if trigger_time is None:
                joined = "".join(collected_parts)
                if len(joined.strip()) >= ttft_trigger_chars:
                    trigger_time = now

    t_end = time.time()
    mem_after_mb = process.memory_info().rss / 1e6

    collected = "".join(collected_parts)

    if first_token_time is None:
        first_token_time = t_end
    if trigger_time is None:
        trigger_time = first_token_time

    ttft_first_token_s = float(first_token_time - t0)
    ttft_trigger_s = float(trigger_time - t0)
    total_s = float(t_end - t0)

    gen_tokens_no_bos = count_gen_tokens_no_bos(llm, collected)

    decode_time_s = max(1e-9, total_s - ttft_first_token_s)
    decode_tps = float(gen_tokens_no_bos / decode_time_s) if gen_tokens_no_bos > 0 else 0.0

    if debug:
        print(f"DEBUG stream collected chars: {len(collected)}")
        print(f"DEBUG gen_tokens(no_bos): {gen_tokens_no_bos}")
        print(f"DEBUG peak_rss_mb: {peak_rss_mb}")
        print(f"DEBUG mem_before_mb: {mem_before_mb} mem_after_mb: {mem_after_mb}")
        print(f"DEBUG ttft_first_token_s: {ttft_first_token_s} ttft_trigger_s: {ttft_trigger_s}")

    return RunStats(
        ttft_first_token_s=ttft_first_token_s,
        ttft_trigger_s=ttft_trigger_s,
        decode_tps=decode_tps,
        total_s=total_s,
        peak_rss_mb=peak_rss_mb,
        mem_before_mb=mem_before_mb,
        mem_after_mb=mem_after_mb,
        gen_tokens_no_bos=gen_tokens_no_bos,
        collected_chars=len(collected),
    )


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _choose_warmup_label(prompt_label: str, warmup_prompt_label: Optional[str]) -> str:
    if warmup_prompt_label:
        return warmup_prompt_label
    if prompt_label != "pt_64":
        return "pt_64"
    return "pt_256"


def _effective_max_tokens(n_ctx: int, prompt_tokens_with_bos: int, max_tokens: int) -> int:
    budget = max(0, n_ctx - prompt_tokens_with_bos)
    return max(0, min(max_tokens, budget))


def _load_prompt(label: str) -> str:
    if label not in PROMPTS:
        raise KeyError(f"Unknown prompt_label: {label}. Available: {sorted(PROMPTS.keys())}")
    return PROMPTS[label]


def _debug_prompt(llm: Llama, label: str, prompt: str) -> Tuple[int, int]:
    prompt_tokens_bos = count_prompt_tokens(llm, prompt, add_bos=True)
    prompt_tokens_no_bos = count_prompt_tokens(llm, prompt, add_bos=False)
    print("DEBUG prompts module file:", os.path.abspath(__import__("benchmarking.src.greenbench.prompts").src.greenbench.prompts.__file__))  # type: ignore
    print("DEBUG prompt_label:", label)
    print("DEBUG prompt chars:", len(prompt))
    print("DEBUG prompt sha1:", _sha1(prompt))
    print("DEBUG prompt head repr:", repr(prompt[:180]))
    print("DEBUG prompt_tokens(add_bos=True):", prompt_tokens_bos)
    print("DEBUG prompt_tokens(add_bos=False):", prompt_tokens_no_bos)
    return prompt_tokens_bos, prompt_tokens_no_bos


def run_condition(
    *,
    model_path: str,
    prompt_label: str,
    n_ctx: int,
    n_threads: int,
    max_tokens: int,
    mode: str,
    cold_runs: int,
    warmup_runs: int,
    warm_runs: int,
    out_dir: str,
    ttft_trigger_chars: int,
    warmup_prompt_label: Optional[str],
    debug: bool,
) -> Dict[str, Any]:
    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")

    # Load model once per process; warmup + measured runs share it in warm mode.
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        verbose=False,
    )

    prompt = _load_prompt(prompt_label)

    # Prompt debug + tokens
    if debug:
        prompt_tokens_bos, prompt_tokens_no_bos = _debug_prompt(llm, prompt_label, prompt)
    else:
        prompt_tokens_bos = count_prompt_tokens(llm, prompt, add_bos=True)
        prompt_tokens_no_bos = count_prompt_tokens(llm, prompt, add_bos=False)

    eff_max_tokens = _effective_max_tokens(n_ctx, prompt_tokens_bos, max_tokens)
    if debug:
        print(f"DEBUG ctx_budget: {max(0, n_ctx - prompt_tokens_bos)} effective_max_tokens: {eff_max_tokens}")

    # Model footprint
    process = psutil.Process()
    model_footprint_mb = process.memory_info().rss / 1e6

    rows: List[Dict[str, Any]] = []

    if mode == "warm":
        wl = _choose_warmup_label(prompt_label, warmup_prompt_label)
        warmup_prompt = _load_prompt(wl)

        # Warmup runs
        for _ in range(max(0, warmup_runs)):
            _ = run_ttft_and_decode(
                llm,
                warmup_prompt,
                _effective_max_tokens(n_ctx, count_prompt_tokens(llm, warmup_prompt, add_bos=True), max_tokens),
                ttft_trigger_chars=ttft_trigger_chars,
                debug=debug,
            )

        for _ in range(max(0, warm_runs)):
            stats = run_ttft_and_decode(
                llm,
                prompt,
                eff_max_tokens,
                ttft_trigger_chars=ttft_trigger_chars,
                debug=debug,
            )
            rows.append(
                {
                    "ttft_first_token_s": stats.ttft_first_token_s,
                    "ttft_trigger_s": stats.ttft_trigger_s,
                    "decode_tps": stats.decode_tps,
                    "peak_rss_mb": stats.peak_rss_mb,
                    "peak_delta_mb": stats.peak_rss_mb - stats.mem_before_mb,
                    "mem_delta_mb": stats.mem_after_mb - stats.mem_before_mb,
                    "total_s": stats.total_s,
                    "gen_tokens": stats.gen_tokens_no_bos,
                }
            )

        n_runs = max(0, warm_runs)
        used_warmup_label = wl

    else:
        used_warmup_label = None
        for _ in range(max(0, cold_runs)):
            llm_cold = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=False,
            )
            stats = run_ttft_and_decode(
                llm_cold,
                prompt,
                eff_max_tokens,
                ttft_trigger_chars=ttft_trigger_chars,
                debug=debug,
            )
            rows.append(
                {
                    "ttft_first_token_s": stats.ttft_first_token_s,
                    "ttft_trigger_s": stats.ttft_trigger_s,
                    "decode_tps": stats.decode_tps,
                    "peak_rss_mb": stats.peak_rss_mb,
                    "peak_delta_mb": stats.peak_rss_mb - stats.mem_before_mb,
                    "mem_delta_mb": stats.mem_after_mb - stats.mem_before_mb,
                    "total_s": stats.total_s,
                    "gen_tokens": stats.gen_tokens_no_bos,
                }
            )
        n_runs = max(0, cold_runs)

    summ = summarize_rows(
        rows,
        keys=[
            "ttft_first_token_s",
            "ttft_trigger_s",
            "decode_tps",
            "peak_rss_mb",
            "peak_delta_mb",
            "mem_delta_mb",
            "total_s",
            "gen_tokens",
        ],
    )

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "prompt_label": prompt_label,
        "warmup_prompt_label": used_warmup_label,
        "prompt_tokens": prompt_tokens_bos,
        "prompt_tokens_no_bos": prompt_tokens_no_bos,
        "n_ctx": n_ctx,
        "n_threads": n_threads,
        "max_tokens": max_tokens,
        "effective_max_tokens": eff_max_tokens,
        "model_path": model_path,
        "model_footprint_mb": model_footprint_mb,
        "n_runs": n_runs,
        "ttft_trigger_chars": ttft_trigger_chars,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    summary.update(
        {
            "ttft_first_token_mean_s": summ["ttft_first_token_s_mean"],
            "ttft_first_token_std_s": summ["ttft_first_token_s_std"],
            "ttft_first_token_p50_s": summ["ttft_first_token_s_p50"],
            "ttft_first_token_p95_s": summ["ttft_first_token_s_p95"],
            "ttft_mean_s": summ["ttft_trigger_s_mean"],
            "ttft_std_s": summ["ttft_trigger_s_std"],
            "ttft_p50_s": summ["ttft_trigger_s_p50"],
            "ttft_p95_s": summ["ttft_trigger_s_p95"],
            "decode_tps_mean": summ["decode_tps_mean"],
            "decode_tps_std": summ["decode_tps_std"],
            "decode_tps_p50": summ["decode_tps_p50"],
            "decode_tps_p95": summ["decode_tps_p95"],
            "peak_rss_mean_mb": summ["peak_rss_mb_mean"],
            "peak_rss_std_mb": summ["peak_rss_mb_std"],
            "peak_rss_p50_mb": summ["peak_rss_mb_p50"],
            "peak_rss_p95_mb": summ["peak_rss_mb_p95"],
            "peak_delta_mean_mb": summ["peak_delta_mb_mean"],
            "peak_delta_std_mb": summ["peak_delta_mb_std"],
            "peak_delta_p50_mb": summ["peak_delta_mb_p50"],
            "peak_delta_p95_mb": summ["peak_delta_mb_p95"],
            "mem_delta_mean_mb": summ["mem_delta_mb_mean"],
            "mem_delta_std_mb": summ["mem_delta_mb_std"],
            "total_mean_s": summ["total_s_mean"],
            "total_std_s": summ["total_s_std"],
            "gen_tokens_mean": summ["gen_tokens_mean"],
            "gen_tokens_p50": summ["gen_tokens_p50"],
            "gen_tokens_p95": summ["gen_tokens_p95"],
        }
    )

    out_csv = os.path.join(out_dir, "results_summary.csv")
    fieldnames = list(summary.keys())
    append_rows_csv(out_csv, fieldnames, [summary])
    print(f"Wrote condition summary to: {out_csv}")

    return summary
