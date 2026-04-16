"""
Script: Embedding Model Throughput Benchmark (HuggingFace transformers)
==========================================================================
Measures GPU throughput (tokens/sec) for each embedding model on your H100,
then computes $/MTok from measured throughput.

Uses raw HuggingFace transformers — no sentence-transformers dependency.
Tokenization is done once before timing, so we measure pure GPU compute.

Output: embedding_throughput.csv

Usage:
    pip install transformers torch numpy accelerate
    python benchmark_throughput.py                        # all 26 models
    python benchmark_throughput.py --model BAAI/bge-m3    # test one model
"""

import argparse
import csv
import gc
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

GPU_HOURLY_RATE = 2.49  # Lambda Labs H100 SXM, accessed March 2026

MODELS = [
    ("tencent/KaLM-Embedding-Gemma3-12B-2511",   11_770_000_000, 0.7328),
    ("bflhc/Octen-Embedding-8B",                   7_570_000_000, 0.7259),
    ("Qwen/Qwen3-Embedding-8B",                    7_570_000_000, 0.7256),
    ("Qwen/Qwen3-Embedding-4B",                    4_020_000_000, 0.7194),
    ("jinaai/jina-embeddings-v5-text-small",          596_000_000, 0.7029),
    ("nvidia/llama-embed-nemotron-8b",              7_500_000_000, 0.7023),
    ("jinaai/jina-embeddings-v5-text-nano",           212_000_000, 0.6978),
    ("Salesforce/SFR-Embedding-2_R",                7_110_000_000, 0.6922),
    ("Alibaba-NLP/gte-Qwen2-7B-instruct",          7_070_000_000, 0.6917),
    ("codefuse-ai/F2LLM-v2-14B",                  13_990_000_000, 0.6905),
    ("Qwen/Qwen3-Embedding-0.6B",                    596_000_000, 0.6888),
    ("codefuse-ai/F2LLM-v2-8B",                    7_570_000_000, 0.6813),
    ("Linq-AI-Research/Linq-Embed-Mistral",         7_110_000_000, 0.6780),
    ("google/embeddinggemma-300m",                    308_000_000, 0.6766),
    ("codefuse-ai/F2LLM-v2-4B",                    4_020_000_000, 0.6751),
    ("codefuse-ai/F2LLM-v2-1.7B",                  1_720_000_000, 0.6735),
    ("Alibaba-NLP/gte-Qwen2-1.5B-instruct",        1_540_000_000, 0.6662),
    ("codefuse-ai/F2LLM-v2-0.6B",                    596_000_000, 0.6576),
    ("intfloat/multilingual-e5-large-instruct",       560_000_000, 0.6576),
    ("BAAI/bge-m3",                                   568_000_000, 0.6125),
    ("intfloat/multilingual-e5-large",                560_000_000, 0.6110),
    ("Snowflake/snowflake-arctic-embed-l-v2.0",       568_000_000, 0.6063),
    ("intfloat/multilingual-e5-base",                 278_000_000, 0.6037),
    ("intfloat/multilingual-e5-small",                118_000_000, 0.5922),
    ("GritLM/GritLM-7B",                            7_240_000_000, 0.5080),
    ("intfloat/e5-mistral-7b-instruct",             7_110_000_000, 0.4872),
]


def load_model(model_id):
    """Load model + tokenizer via HuggingFace transformers."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16,
    ).cuda().eval()
    return model, tokenizer


def tokenize_batch(tokenizer, seq_len, batch_size):
    """Create a pre-tokenized batch on GPU. Returns (input_ids, attention_mask, token_count)."""
    filler = "The quick brown fox jumps over the lazy dog and runs across the field. "
    text = filler * (seq_len // 5 + 20)
    texts = [text] * batch_size

    encoded = tokenizer(
        texts, max_length=seq_len, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    input_ids = encoded["input_ids"].cuda()
    attention_mask = encoded["attention_mask"].cuda()

    # Count real tokens (non-padding) per text
    tok_per_text = int(attention_mask[0].sum().item())
    total_tokens = tok_per_text * batch_size

    return input_ids, attention_mask, total_tokens


def forward_pass(model, input_ids, attention_mask):
    """Run forward pass. Tries use_cache=False first (needed for decoder models),
    falls back to without it (for encoder-only models like BERT/XLM-R)."""
    try:
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    except TypeError:
        return model(input_ids=input_ids, attention_mask=attention_mask)


def try_forward(model, input_ids, attention_mask):
    """Try a single forward pass. Returns True if it fits in VRAM."""
    try:
        gc.collect(); torch.cuda.empty_cache()
        with torch.no_grad():
            forward_pass(model, input_ids, attention_mask)
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        return False


def find_max_batch_size(model, tokenizer, seq_len, ceiling=2048):
    """Binary search for max batch size that fits in VRAM."""
    candidates = [2**i for i in range(12) if 2**i <= ceiling]
    best = 1

    for bs in candidates:
        print(f"    bs={bs}...", end=" ", flush=True)
        input_ids, attn_mask, _ = tokenize_batch(tokenizer, seq_len, bs)
        if try_forward(model, input_ids, attn_mask):
            best = bs; print("OK")
            del input_ids, attn_mask
        else:
            del input_ids, attn_mask; print("OOM"); break

    # Binary search between last success and first failure
    low, high = best, min(best * 2, ceiling)
    while low < high - 1:
        mid = (low + high) // 2
        print(f"    bs={mid}...", end=" ", flush=True)
        input_ids, attn_mask, _ = tokenize_batch(tokenizer, seq_len, mid)
        if try_forward(model, input_ids, attn_mask):
            low = mid; print("OK")
        else:
            high = mid; print("OOM")
        del input_ids, attn_mask

    gc.collect(); torch.cuda.empty_cache()
    return low


def measure(model, input_ids, attention_mask, total_tokens, warmup=5, runs=50):
    """Time forward passes on pre-tokenized, pre-moved-to-GPU data.
    
    This measures pure GPU compute — no tokenization, no CPU-GPU transfer.
    """
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            forward_pass(model, input_ids, attention_mask)
    torch.cuda.synchronize()

    # Timed runs
    timings = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            forward_pass(model, input_ids, attention_mask)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    throughputs = [total_tokens / t for t in timings]
    return throughputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="embedding_throughput.csv")
    p.add_argument("--model", default=None)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=50)
    args = p.parse_args()

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    print(f"GPU: {gpu}")
    print(f"Rate: ${GPU_HOURLY_RATE}/hr | Seq len: {args.seq_len} | Runs: {args.runs}")
    print("=" * 80)

    models = MODELS
    if args.model:
        models = [m for m in MODELS if m[0] == args.model]
        if not models:
            print(f"Model '{args.model}' not in registry."); return

    rows = []
    for model_id, params, mteb in models:
        print(f"\n{'─' * 80}")
        print(f"{model_id} ({params/1e9:.1f}B params, MTEB={mteb})")

        try:
            # Load
            model, tokenizer = load_model(model_id)

            # Find max batch size
            print(f"  Finding max batch size...")
            raw_bs = find_max_batch_size(model, tokenizer, args.seq_len)
            # 95% of max — expandable_segments handles fragmentation, retry loop catches edge cases
            bs = max(1, int(raw_bs * 0.95))
            print(f"  Max batch size: {raw_bs} → using {bs} (95% safety margin)")

            # Pre-tokenize the batch once (not included in timing)
            input_ids, attn_mask, total_tokens = tokenize_batch(tokenizer, args.seq_len, bs)
            print(f"  Tokens per batch: {total_tokens:,}")

            gc.collect(); torch.cuda.empty_cache()
            print(f"  Measuring ({args.runs} runs at bs={bs})...")
            tp_list = measure(model, input_ids, attn_mask, total_tokens, args.warmup, args.runs)

            tp = np.array(tp_list)
            med = float(np.median(tp))
            cost = GPU_HOURLY_RATE * 1_000_000 / (med * 3600)

            print(f"  Throughput: {med:,.0f} tok/s (median)")
            print(f"  $/MTok:    ${cost:.6f}")

            rows.append({
                "model_id": model_id, "params": params, "mteb_score": mteb,
                "batch_size_used": bs, "seq_len": args.seq_len,
                "tokens_per_batch": total_tokens,
                "median_tok_per_sec": round(med, 1),
                "p5_tok_per_sec": round(float(np.percentile(tp, 5)), 1),
                "p95_tok_per_sec": round(float(np.percentile(tp, 95)), 1),
                "cost_usd_per_mtok": round(cost, 6),
                "gpu": gpu, "gpu_rate_usd_per_hr": GPU_HOURLY_RATE,
                "status": "success", "error": "",
            })

            del model, tokenizer, input_ids, attn_mask
            gc.collect(); torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append({
                "model_id": model_id, "params": params, "mteb_score": mteb,
                "batch_size_used": 0, "seq_len": args.seq_len,
                "tokens_per_batch": 0,
                "median_tok_per_sec": 0, "p5_tok_per_sec": 0, "p95_tok_per_sec": 0,
                "cost_usd_per_mtok": 0,
                "gpu": gpu, "gpu_rate_usd_per_hr": GPU_HOURLY_RATE,
                "status": "error", "error": str(e)[:200],
            })
            gc.collect(); torch.cuda.empty_cache()

    # Write CSV
    out = Path(args.output)
    fields = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Summary
    print(f"\n{'=' * 80}")
    print(f"{'Model':<48} {'BS':>4} {'tok/s':>10} {'$/MTok':>12} {'MTEB':>7}")
    print(f"{'─' * 84}")
    for r in sorted(rows, key=lambda x: x["cost_usd_per_mtok"] if x["cost_usd_per_mtok"] > 0 else 999):
        if r["status"] == "success":
            print(f"  {r['model_id']:<46} {r['batch_size_used']:>4} {r['median_tok_per_sec']:>10,.0f} ${r['cost_usd_per_mtok']:>11.6f} {r['mteb_score']:>7.4f}")
        else:
            print(f"  {r['model_id']:<46} {'—':>4} {'FAIL':>10} {'—':>12} {r['mteb_score']:>7.4f}")

    print(f"\nWrote {out}")
    print(f"Feed into Script 2: python compute_pareto_costs.py --throughput {out}")


if __name__ == "__main__":
    main()