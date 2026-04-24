"""
Benchmark: Fused Linear Cross Entropy — Native vs Generic vs Blackwell

Compares:
  1. Native    — torch.matmul + F.cross_entropy  (full logits materialised)
  2. Generic   — chunked online-softmax, pure PyTorch  (logits never materialised)
  3. Blackwell — CUTLASS/CuTe fused kernel  (only on Blackwell GPUs, cc==10)

Measures: peak GPU memory, wall-clock time, and numerical correctness.

Usage:
    python benchmark_fused_lce.py [--num-tokens 4096] [--dim 4096] [--vocab-size 128256]
                                  [--dtype bf16] [--num-warmup 3] [--num-iters 10]
                                  [--vocab-per-split 3072]
"""

import argparse
import os
import time


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark fused linear cross entropy")
    p.add_argument("--num-tokens", type=int, default=4096, help="Number of tokens (batch * seq)")
    p.add_argument("--dim", type=int, default=4096, help="Hidden dimension")
    p.add_argument("--vocab-size", type=int, default=128256, help="Vocabulary size")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument("--num-warmup", type=int, default=3)
    p.add_argument("--num-iters", type=int, default=10)
    p.add_argument("--vocab-per-split", type=int, default=3072,
                    help="Chunk size for generic path")
    return p.parse_args()


# Parse args and set env vars BEFORE any torch / generic_entry imports,
# so that GenericConfig picks up the correct vocab_per_split value.
_args = parse_args()
os.environ["LCE_GENERIC_FWD_VOCAB_SPLIT_SIZE"] = str(_args.vocab_per_split)
os.environ["LCE_GENERIC_BWD_VOCAB_SPLIT_SIZE"] = str(_args.vocab_per_split)

import torch
import torch.nn.functional as F
from megatron.core.fusions.linear_cross_entropy.generic import entry as generic_entry

# Try to import the Blackwell fused path (only works on Blackwell GPUs with CUTLASS)
HAS_BLACKWELL = False
try:
    cc = torch.cuda.get_device_capability(torch.cuda.current_device())
    if cc[0] == 10:
        from megatron.core.fusions.linear_cross_entropy.blackwell import entry as _bw_entry
        # Verify the CUTLASS kernel functions are actually defined (not just the module)
        _bw_fwd = _bw_entry.forward
        _bw_bwd = _bw_entry.backward
        from megatron.core.fusions.fused_linear_cross_entropy import linear_cross_entropy
        HAS_BLACKWELL = True
        print(f"Blackwell GPU detected (cc={cc}), CUTLASS fused kernel available.")
except (ImportError, AttributeError, Exception) as e:
    print(f"Blackwell fused kernel not available: {e}")


def get_dtype(name):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


# ---------------------------------------------------------------------------
# 1. Native: full logits materialised
# ---------------------------------------------------------------------------
def native_forward_backward(hidden, weight, labels, ignore_index=-100):
    """Standard matmul + F.cross_entropy — full logits in memory."""
    hidden = hidden.detach().requires_grad_(True)
    weight = weight.detach().requires_grad_(True)

    logits = torch.matmul(hidden.float(), weight.float().t())  # (N, V) — the memory hog
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction="mean")
    loss.backward()
    return loss.detach(), hidden.grad.detach(), weight.grad.detach()


# ---------------------------------------------------------------------------
# 2. Generic fused: chunked, pure PyTorch, no full logits
# ---------------------------------------------------------------------------
def generic_forward_backward(hidden, weight, labels, ignore_index=-100):
    """Chunked fused linear cross entropy — logits never fully materialised."""
    # forward
    logprobs, maximum, accumulate, num_valid_tokens, _, _, global_hidden = (
        generic_entry.forward(
            hidden, weight, labels,
            tp_group=None, reduction="mean", ignore_index=ignore_index,
        )
    )
    # backward
    dlogprobs = torch.ones_like(logprobs)
    d_hidden, d_weight = generic_entry.backward(
        dlogprobs, global_hidden, weight, labels,
        maximum, accumulate, num_valid_tokens,
        reduction="mean", ignore_index=ignore_index,
    )
    return logprobs.detach(), d_hidden.detach(), d_weight.detach()


# ---------------------------------------------------------------------------
# 3. Blackwell fused: CUTLASS/CuTe kernel (only on Blackwell GPUs)
# ---------------------------------------------------------------------------
def blackwell_forward_backward(hidden, weight, labels, ignore_index=-100):
    """Blackwell CUTLASS fused linear cross entropy."""
    hidden = hidden.detach().requires_grad_(True)
    weight = weight.detach().requires_grad_(True)

    loss = linear_cross_entropy(
        hidden, weight, labels,
        tp_group=None, reduction="mean", ignore_index=ignore_index,
    )
    loss.backward()
    return loss.detach(), hidden.grad.detach(), weight.grad.detach()


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------
def benchmark(fn, hidden, weight, labels, num_warmup, num_iters, label=""):
    """Run fn, measure peak memory and average time."""
    # warmup
    for _ in range(num_warmup):
        fn(hidden, weight, labels)
        torch.cuda.synchronize()

    # Force free all cached memory so baseline is clean
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        loss, dh, dw = fn(hidden, weight, labels)
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    peak_mem = torch.cuda.max_memory_allocated()
    avg_ms = (t1 - t0) / num_iters * 1000

    print(f"[{label:>12s}]  loss={loss.item():.6f}  "
          f"peak_mem={peak_mem / 1024**2:.1f} MB  "
          f"avg_time={avg_ms:.2f} ms")
    return loss, dh, dw, peak_mem, avg_ms


def print_comparison(label_a, label_b, loss_a, loss_b, dh_a, dh_b, dw_a, dw_b,
                     mem_a, mem_b, time_a, time_b):
    """Print memory/time/precision comparison between two methods."""
    mem_saved = (mem_a - mem_b) / 1024**2
    mem_ratio = mem_a / max(mem_b, 1)
    loss_diff = abs(loss_a.item() - loss_b.item())
    dh_diff = (dh_a.float() - dh_b.float()).abs().max().item()
    dw_diff = (dw_a.float() - dw_b.float()).abs().max().item()

    print(f"  {label_b} vs {label_a}:")
    print(f"    Memory saved:     {mem_saved:+.1f} MB  ({mem_ratio:.2f}x)")
    print(f"    Time ratio:       {time_b / max(time_a, 1e-9):.2f}x")
    print(f"    Loss diff:        {loss_diff:.6e}")
    print(f"    d_hidden max Δ:   {dh_diff:.6e}")
    print(f"    d_weight max Δ:   {dw_diff:.6e}")
    ok = loss_diff < 1e-2 and dh_diff < 1e-1 and dw_diff < 1e-1
    print(f"    Numerical check:  {'PASS' if ok else 'FAIL'}")


def main():
    args = _args
    assert torch.cuda.is_available(), "CUDA required"

    # Verify config is correct
    cfg = generic_entry._get_config()
    actual_split = cfg.fwd_vocab_per_split

    dtype = get_dtype(args.dtype)
    device = "cuda"
    N, D, V = args.num_tokens, args.dim, args.vocab_size
    gpu_name = torch.cuda.get_device_name()

    print(f"GPU:    {gpu_name}")
    print(f"Config: num_tokens={N}, dim={D}, vocab_size={V}, dtype={args.dtype}, "
          f"vocab_per_split={actual_split}")
    print(f"Full logits tensor would be: {N * V * 4 / 1024**2:.1f} MB (float32)")
    print(f"Chunk buffer (fwd logits_buf): {N * actual_split * 4 / 1024**2:.1f} MB (float32)")
    print()

    # Create inputs
    torch.manual_seed(42)
    hidden = torch.randn(N, D, dtype=dtype, device=device)
    weight = torch.randn(V, D, dtype=dtype, device=device) * 0.01
    labels = torch.randint(0, V, (N,), device=device)
    # Sprinkle some ignore_index
    labels[::7] = -100

    # ---- Run benchmarks ----
    results = {}

    print("--- Benchmarks ---")
    loss_n, dh_n, dw_n, mem_n, time_n = benchmark(
        native_forward_backward, hidden, weight, labels,
        args.num_warmup, args.num_iters, label="Native"
    )
    results["Native"] = (loss_n, dh_n, dw_n, mem_n, time_n)

    loss_g, dh_g, dw_g, mem_g, time_g = benchmark(
        generic_forward_backward, hidden, weight, labels,
        args.num_warmup, args.num_iters, label="Generic"
    )
    results["Generic"] = (loss_g, dh_g, dw_g, mem_g, time_g)

    if HAS_BLACKWELL:
        loss_b, dh_b, dw_b, mem_b, time_b = benchmark(
            blackwell_forward_backward, hidden, weight, labels,
            args.num_warmup, args.num_iters, label="Blackwell"
        )
        results["Blackwell"] = (loss_b, dh_b, dw_b, mem_b, time_b)

    # ---- Comparisons ----
    print()
    print("=" * 60)
    print_comparison("Native", "Generic",
                     loss_n, loss_g, dh_n, dh_g, dw_n, dw_g,
                     mem_n, mem_g, time_n, time_g)

    if HAS_BLACKWELL:
        print()
        print_comparison("Native", "Blackwell",
                         loss_n, loss_b, dh_n, dh_b, dw_n, dw_b,
                         mem_n, mem_b, time_n, time_b)
        print()
        print_comparison("Generic", "Blackwell",
                         loss_g, loss_b, dh_g, dh_b, dw_g, dw_b,
                         mem_g, mem_b, time_g, time_b)


if __name__ == "__main__":
    main()
