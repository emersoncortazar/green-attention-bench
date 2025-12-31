# GreenAttentionBench

**Green Attention Bench** is a lightweight benchmarking suite for **energy-, latency-, and memory-aware evaluation of long-context LLM inference**, with pluggable “efficient attention” backends.

The goal is to provide a **clean, reproducible, systems-style benchmark** that compares attention mechanisms apples-to-apples and surfaces *when* and *why* different approaches win or fail at long context lengths.

---

## Motivation

Long-context inference stresses modern hardware along multiple axes: compute, memory bandwidth, capacity, and power. While many “efficient attention” methods exist, it is often unclear:

- Which method actually improves **end-to-end inference efficiency**
- Under what **context lengths and hardware constraints**
- And what the **dominant bottleneck** is (compute vs. memory vs. bandwidth)

green-attention-bench is designed to answer those questions in a controlled, practical way.

---

## What This Repository Contains

### 1. Standardized Inference Benchmark Harness
A unified runner that measures inference performance across:
- Multiple models
- Multiple context lengths
- Multiple attention backends
- Consistent prompts and tasks

### 2. Models
- Small open LLMs (≈1B–3B parameters) that can run locally
- Optional medium-scale model if hardware permits

### 3. Tasks
- Long-context summarization
- Retrieval-style QA
- Synthetic long-sequence perplexity proxy

### 4. Attention Backends (Apples-to-Apples)
- Baseline PyTorch SDPA attention  
- FlashAttention / xFormers
- At least one long-context or “efficient attention” alternative  
  (e.g., linear attention family or state-space models)

### 5. Metrics Collected
- **Throughput:** tokens / second  
- **Latency:** time per generated token  
- **Memory:** peak VRAM / RAM usage  
- **Energy proxy:** GPU power draw via `nvidia-smi`

---

## Minimal but High-Value Scope

The core benchmark targets:
- **3 context lengths** (e.g., 2k, 8k, 32k)
- **2 models**
- **2–3 attention implementations**
- **4 core metrics** (throughput, latency, memory, energy)

### Optional Systems Twist
> If context length > X or available memory < Y, automatically switch attention backend or chunking strategy.

This mirrors real-world system decision layers used in large-scale inference.

---

## Why This Project Exists

The core claim this repository is meant to support:
> *I can design and run reproducible experiments that quantify long-context efficiency tradeoffs (throughput, latency, memory, energy), and turn those results into actionable system choices.*

---

## Status

**Early development / starter scaffold**

Initial focus:
- Define benchmark interfaces
- Implement baseline SDPA measurement
- Add one efficient attention backend
- Validate metrics collection