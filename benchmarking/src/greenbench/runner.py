from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
from llama_cpp import Llama

from .prompts import PROMPTS


# -------------------------
# Data models
# -------------------------
@dataclass(frozen=True)
class BenchConfig:
    model_path: str
    prompt_label: str
    n_ctx: int
    mode: str  # "warm" or "cold"
    n_threads: int
    max_tokens: int
    cold_runs: int
    warmup_runs: int
    warm_runs: int
    out_dir: str
    ttft_trigger_chars: int = 8

    # Safety buffer: reserve tokens for BOS/EOS/system overhead
    # (llama.cpp often needs 1+ extra tokens)
    safety_tokens: int = 4


@dataclass
class RunResult:
    run_idx: int
    prompt_tokens: int
    max_tokens_req: int
    max_tokens_eff: int
    ttft_s: Optional[float]
    decode_s: Optional[float]
    decode_tok_s: Optional[float]
    total_s: Optional[float]
    rss_before_mb: Optional[float]
    rss_after_mb: Optional[float]
    rss_peak_mb: Optional[float]
    mem_delta_mb: Optional[float]
    status: str  # "ok", "skipped_ctx_too_small", "error"
    error: str = ""


@dataclass
class ConditionSummary:
    model_path: str
    model_sha1: str
    prompt_label: str
    prompt_tokens: int
    n_ctx: int
    mode: str
    n_threads: int
    max_tokens: int
    ttft_trigger_chars: int
    runs: int

    mean_ttft_s: Optional[float]
    mean_decode_tok_s: Optional[float]
    mean_total_s: Optional[float]
    mean_rss_peak_mb: Optional[float]

    status: str  # "ok", "partial", "skipped", "error"
    error: str

    platform: str
    python: str
    cpu_count_logical: int
    machine: str


# -------------------------
# Helpers
# -------------------------
def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _append_rows_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})


def _mean_or_none(xs: List[float]) -> Optional[float]:
    return statistics.mean(xs) if xs else None


def _now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _tokenize_count(llm: Llama, text: str) -> int:
    # llama_cpp expects bytes
    toks = llm.tokenize(text.encode("utf-8"), add_bos=True)
    return int(len(toks))


def _effective_max_tokens(n_ctx: int, prompt_tokens: int, requested: int, safety_tokens: int) -> int:
    room = n_ctx - prompt_tokens - safety_tokens
    if room <= 0:
        return 0
    return max(0, min(requested, room))


def _run_one(
    llm: Llama,
    prompt: str,
    prompt_tokens: int,
    n_ctx: int,
    max_tokens_req: int,
    ttft_trigger_chars: int,
    safety_tokens: int,
) -> RunResult:
    process = psutil.Process()
    rss_before = process.memory_info().rss / 1e6

    max_tokens_eff = _effective_max_tokens(n_ctx, prompt_tokens, max_tokens_req, safety_tokens)
    if max_tokens_eff <= 0:
        return RunResult(
            run_idx=-1,
            prompt_tokens=prompt_tokens,
            max_tokens_req=max_tokens_req,
            max_tokens_eff=0,
            ttft_s=None,
            decode_s=None,
            decode_tok_s=None,
            total_s=None,
            rss_before_mb=rss_before,
            rss_after_mb=rss_before,
            rss_peak_mb=rss_before,
            mem_delta_mb=0.0,
            status="skipped_ctx_too_small",
            error=f"prompt_tokens={prompt_tokens} leaves no room in n_ctx={n_ctx} (safety={safety_tokens})",
        )

    t0 = time.time()
    first_token_t: Optional[float] = None
    out_text = ""

    rss_peak = rss_before

    try:
        # stream=True lets us measure TTFT properly
        for part in llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens_eff,
            stream=True,
            temperature=0.7,
            top_p=0.95,
        ):
            # Track peak RSS during generation
            rss_peak = max(rss_peak, process.memory_info().rss / 1e6)

            chunk = part["choices"][0].get("text", "")
            if chunk:
                if first_token_t is None:
                    # TTFT measured at first non-empty chunk
                    first_token_t = time.time() - t0
                out_text += chunk
                if len(out_text) >= ttft_trigger_chars and first_token_t is not None:
                    # We keep generating anyway; this is just the “trigger char” threshold concept.
                    pass

        t1 = time.time()
        total_s = t1 - t0
        ttft_s = first_token_t if first_token_t is not None else total_s

        # Estimate decode tokens/sec using the model tokenizer on the output text
        out_tokens = len(llm.tokenize(out_text.encode("utf-8"), add_bos=False))
        decode_s = max(1e-9, total_s - ttft_s)
        decode_tok_s = (out_tokens / decode_s) if out_tokens > 0 else 0.0

        rss_after = process.memory_info().rss / 1e6
        return RunResult(
            run_idx=-1,
            prompt_tokens=prompt_tokens,
            max_tokens_req=max_tokens_req,
            max_tokens_eff=max_tokens_eff,
            ttft_s=float(ttft_s),
            decode_s=float(decode_s),
            decode_tok_s=float(decode_tok_s),
            total_s=float(total_s),
            rss_before_mb=float(rss_before),
            rss_after_mb=float(rss_after),
            rss_peak_mb=float(rss_peak),
            mem_delta_mb=float(rss_after - rss_before),
            status="ok",
            error="",
        )
    except Exception as e:
        rss_after = process.memory_info().rss / 1e6
        return RunResult(
            run_idx=-1,
            prompt_tokens=prompt_tokens,
            max_tokens_req=max_tokens_req,
            max_tokens_eff=max_tokens_eff,
            ttft_s=None,
            decode_s=None,
            decode_tok_s=None,
            total_s=None,
            rss_before_mb=float(rss_before),
            rss_after_mb=float(rss_after),
            rss_peak_mb=float(max(rss_peak, rss_after)),
            mem_delta_mb=float(rss_after - rss_before),
            status="error",
            error=repr(e),
        )


# -------------------------
# Main entry: run_condition
# -------------------------
def run_condition(cfg: BenchConfig) -> ConditionSummary:
    assert cfg.mode in ("warm", "cold"), f"mode must be warm/cold, got {cfg.mode}"

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path_abs = str(Path(cfg.model_path).resolve())
    model_sha1 = _sha1_file(model_path_abs)

    # Load prompt text
    if cfg.prompt_label not in PROMPTS:
        raise ValueError(f"Unknown prompt_label={cfg.prompt_label}. Available: {sorted(PROMPTS.keys())}")
    prompt = PROMPTS[cfg.prompt_label]

    # Warm mode: load once and reuse
    # Cold mode: reload per run
    llm: Optional[Llama] = None

    def load_llm() -> Llama:
        return Llama(
            model_path=model_path_abs,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            verbose=False,
        )

    # Prepare per-run CSV
    runs_csv = out_dir / "runs.csv"
    run_fields = [
        "ts",
        "run_idx",
        "status",
        "error",
        "prompt_label",
        "prompt_tokens",
        "n_ctx",
        "mode",
        "n_threads",
        "max_tokens_req",
        "max_tokens_eff",
        "ttft_s",
        "decode_s",
        "decode_tok_s",
        "total_s",
        "rss_before_mb",
        "rss_after_mb",
        "rss_peak_mb",
        "mem_delta_mb",
    ]

    # Decide how many runs
    runs_target = cfg.warm_runs if cfg.mode == "warm" else cfg.cold_runs
    warmup_target = cfg.warmup_runs if cfg.mode == "warm" else 0

    # Execute
    all_run_results: List[RunResult] = []

    try:
        if cfg.mode == "warm":
            llm = load_llm()
            prompt_tokens = _tokenize_count(llm, prompt)

            # warmups (not recorded in summary)
            for _i in range(warmup_target):
                _ = _run_one(
                    llm=llm,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    n_ctx=cfg.n_ctx,
                    max_tokens_req=max(1, min(cfg.max_tokens, 8)),
                    ttft_trigger_chars=cfg.ttft_trigger_chars,
                    safety_tokens=cfg.safety_tokens,
                )

            for i in range(runs_target):
                rr = _run_one(
                    llm=llm,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    n_ctx=cfg.n_ctx,
                    max_tokens_req=cfg.max_tokens,
                    ttft_trigger_chars=cfg.ttft_trigger_chars,
                    safety_tokens=cfg.safety_tokens,
                )
                rr.run_idx = i
                all_run_results.append(rr)

        else:
            # cold: load per run
            # NOTE: prompt_tokens can vary slightly with different sessions; compute each run.
            for i in range(runs_target):
                llm_i = load_llm()
                prompt_tokens = _tokenize_count(llm_i, prompt)
                rr = _run_one(
                    llm=llm_i,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    n_ctx=cfg.n_ctx,
                    max_tokens_req=cfg.max_tokens,
                    ttft_trigger_chars=cfg.ttft_trigger_chars,
                    safety_tokens=cfg.safety_tokens,
                )
                rr.run_idx = i
                all_run_results.append(rr)

    finally:
        # Write per-run rows
        rows = []
        for rr in all_run_results:
            rows.append(
                {
                    "ts": _now_iso(),
                    "run_idx": rr.run_idx,
                    "status": rr.status,
                    "error": rr.error,
                    "prompt_label": cfg.prompt_label,
                    "prompt_tokens": rr.prompt_tokens,
                    "n_ctx": cfg.n_ctx,
                    "mode": cfg.mode,
                    "n_threads": cfg.n_threads,
                    "max_tokens_req": rr.max_tokens_req,
                    "max_tokens_eff": rr.max_tokens_eff,
                    "ttft_s": rr.ttft_s,
                    "decode_s": rr.decode_s,
                    "decode_tok_s": rr.decode_tok_s,
                    "total_s": rr.total_s,
                    "rss_before_mb": rr.rss_before_mb,
                    "rss_after_mb": rr.rss_after_mb,
                    "rss_peak_mb": rr.rss_peak_mb,
                    "mem_delta_mb": rr.mem_delta_mb,
                }
            )
        if rows:
            _append_rows_csv(str(runs_csv), run_fields, rows)

    ok = [r for r in all_run_results if r.status == "ok"]
    skipped = [r for r in all_run_results if r.status == "skipped_ctx_too_small"]
    errs = [r for r in all_run_results if r.status == "error"]

    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    dec_tok_s = [r.decode_tok_s for r in ok if r.decode_tok_s is not None]
    totals = [r.total_s for r in ok if r.total_s is not None]
    peaks = [r.rss_peak_mb for r in ok if r.rss_peak_mb is not None]

    if ok and not errs and not skipped:
        status = "ok"
        err_str = ""
    elif not ok and skipped and not errs:
        status = "skipped"
        err_str = skipped[0].error if skipped else "skipped"
    elif errs and not ok:
        status = "error"
        err_str = errs[0].error
    else:
        status = "partial"
        err_str = errs[0].error if errs else (skipped[0].error if skipped else "")

    summary = ConditionSummary(
        model_path=cfg.model_path,
        model_sha1=model_sha1,
        prompt_label=cfg.prompt_label,
        prompt_tokens=ok[0].prompt_tokens if ok else (all_run_results[0].prompt_tokens if all_run_results else 0),
        n_ctx=cfg.n_ctx,
        mode=cfg.mode,
        n_threads=cfg.n_threads,
        max_tokens=cfg.max_tokens,
        ttft_trigger_chars=cfg.ttft_trigger_chars,
        runs=runs_target,
        mean_ttft_s=_mean_or_none([float(x) for x in ttfts if x is not None]),
        mean_decode_tok_s=_mean_or_none([float(x) for x in dec_tok_s if x is not None]),
        mean_total_s=_mean_or_none([float(x) for x in totals if x is not None]),
        mean_rss_peak_mb=_mean_or_none([float(x) for x in peaks if x is not None]),
        status=status,
        error=err_str,
        platform=platform.platform(),
        python=platform.python_version() + " | " + platform.python_implementation(),
        cpu_count_logical=int(os.cpu_count() or 0),
        machine=platform.machine(),
    )

    # Write summary.json in condition folder
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(summary), f, indent=2)

    # Append to global summary CSV (repo-level)
    global_csv = Path("data") / "results_summary.csv"
    summary_fields = [
        "model_path",
        "model_sha1",
        "prompt_label",
        "prompt_tokens",
        "n_ctx",
        "mode",
        "n_threads",
        "max_tokens",
        "ttft_trigger_chars",
        "runs",
        "mean_ttft_s",
        "mean_decode_tok_s",
        "mean_total_s",
        "mean_rss_peak_mb",
        "status",
        "error",
        "platform",
        "python",
        "cpu_count_logical",
        "machine",
    ]
    _append_rows_csv(str(global_csv), summary_fields, [dataclasses.asdict(summary)])

    return summary
