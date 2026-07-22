# colibri-m3

**Pure-C MiniMax-M3 inference engine for CPU-only machines.**

Runs the 428B-parameter MiniMax-M3 Mixture-of-Experts model (~23B activated) on
a single CPU server with no GPU. Achieves **1.2 tok/s** on a dual-socket Xeon
with 376 GB RAM by streaming expert weights from OS page cache via zero-copy
mmap, with full model prewarming and persistent scratch buffers.

## Performance

| Metric | Baseline | Optimized | Speedup |
|--------|----------|-----------|---------|
| **Decode throughput** | 0.25 tok/s | **1.23 tok/s** | **4.9x** |
| **Warm decode** | 0.28 tok/s | **1.30 tok/s** | **4.6x** |
| **Prefill (4 tokens)** | 29.5s | **3.2s** | **9.2x** |
| **Expert load (cache miss)** | 28ms | **0ms** (mmap pointer) | - |
| **MoE per layer (warm)** | 119ms | **23ms** | **5.2x** |

Tested on: 2x Intel Xeon Gold 5220R (96 threads), 376 GB RAM, AVX-512 VNNI,
no GPU. Model: MiniMax-M3 int4 quantized, 212 GB on disk.

## How it works

```
                 +------------------+
                 |  OS Page Cache   |  212 GB model fits in 376 GB RAM
                 |  (228 GB mapped) |
                 +------------------+
                          |
                    mmap (zero-copy)
                          |
          +-------------------------------+
          |        colibri-m3 engine       |
          |                               |
          |  Dense weights (resident)     |  Q/K/V/O projections
          |  Shared expert (resident)     |  Always needed, kept in RAM
          |  Routed experts (mmap'd)      |  128 experts/layer, topk=4
          |  Expert LRU cache             |  128 slots/layer
          |                               |
          |  Persistent scratch buffers   |  No per-layer malloc/free
          |  AVX-512 VNNI kernels         |  Optional int8/int4 SIMD
          |  NUMA topology awareness      |  Thread pinning (opt-in)
          +-------------------------------+
```

### Key optimizations

1. **mmap zero-copy expert loading** — Shard files are mmap'd once; expert
   weights are accessed via pointer arithmetic. No `pread`, no `malloc`, no
   `free` per expert load. Expert cache misses go from 28ms to 0ms.

2. **Full model prewarm** — At load time, every page of all 59 shard files is
   touched to pull the 228 GB model into OS page cache. Eliminates page faults
   during inference. Takes ~12s one-time cost.

3. **Removed `POSIX_FADV_DONTNEED`** — The original code told the kernel to
   evict file pages after every `pread`, fighting the OS page cache. Removing
   this was the single biggest win (unblocked all subsequent optimizations).

4. **Persistent scratch buffers** — MoE and attention paths reuse pre-allocated
   buffers instead of malloc/free per layer. Eliminates ~600+ heap operations
   per token.

5. **AVX-512 VNNI kernels** — Optional int8 (VNNI) and int4 (FMA) matmul kernels
   using `_mm512_dpbusd_epi32` and `_mm256_cvtepi8_epi32`. Correct but not
   faster for this workload (memory-bound, not compute-bound).

6. **NUMA-aware deployment** — Topology discovery, thread pinning, and memory
   interleave via raw `mbind`/`sched_setaffinity` syscalls (no libnuma).

## Build

```bash
make -j
```

Requires: GCC with OpenMP, AVX-512 support (`-march=native`).

## Convert weights

The engine uses a custom int4 safetensors format with per-tensor `.qs` scale
files. Convert from HuggingFace BF16:

```bash
./coli convert --repo MiniMaxAI/MiniMax-M3 --model /path/to/m3_i4
```

Or from a local checkout:

```bash
./coli convert --indir ./MiniMax-M3 --model /path/to/m3_i4
```

## Run

```bash
export SNAP=/path/to/m3_i4
export OMP_NUM_THREADS=88
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Interactive chat
./coli chat --cap 256 --ctx 8192

# Benchmark mode (per-token timing + JSON summary)
echo -e "5\n3 1334 5552 304 264" | ./m3 --bench

# Teacher-force mode (oracle comparison)
echo -e "20\n<prompt_ids>\n<target_ids>" | ./m3 --teacher-force
```

### Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `SNAP` | (required) | Path to model directory |
| `OMP_NUM_THREADS` | 88 | OpenMP worker threads |
| `OMP_PLACES` | cores | Thread placement |
| `OMP_PROC_BIND` | close | Thread binding |
| `USE_VNNI` | 0 | Enable AVX-512 VNNI int8/int4 kernels |
| `USE_AVX512_I4` | 0 | Enable AVX-512 int4 FMA kernel |
| `MOE_PARALLEL` | 0 | Parallel MoE expert dispatch |
| `NUMA_INTERLEAVE` | 0 | Interleave model pages across NUMA nodes |
| `NUMA_PIN_THREADS` | 0 | Pin OMP threads to specific CPUs |
| `SKIP_PREWARM` | 0 | Skip the page-cache prewarm pass |
| `TEMP` | 1.0 | Sampling temperature (0 = greedy) |
| `TOPP` | 0.95 | Top-p sampling threshold |
| `TOPK` | 40 | Top-k sampling limit |
| `SEED` | (time) | Random seed for sampling |
| `M3_TELEMETRY_PATH` | (none) | Write JSON-line telemetry to file |

## Project layout

```
src/
  engine.c          MiniMax-M3 forward pass (GQA, RoPE, SwiGLU, MoE, MSA)
  st.h              Safetensors shard index + mmap zero-copy access
  json.h            Minimal JSON parser for safetensors headers
  planar_kv.h       PlanarQuant-style KV cache (optional)
  vnni.h            AVX-512 VNNI int8/int4 matmul kernels
  numa.h            NUMA topology + thread pinning (no libnuma)
  observability.h   JSON-line telemetry for cross-engine comparison

c/tests/
  test_vnni.c       VNNI kernel correctness (bit-exact vs scalar)
  test_numa.c       NUMA topology discovery + thread pinning
  test_int8_kv.c    INT8 KV cache quantization round-trip
  test_matmul_i4.c  Int4 matmul correctness
  test_msa.c        MiniMax Sparse Attention block selection
  test_moe_routing.c MoE router (sigmoid + top-K + bias)
  test_rmsnorm.c    RMSNorm (including Gemma variant)
  test_rope.c       Rotary position embedding
  test_swiglu.c     SwiGLU activation
  test_per_head_qk_norm.c  Per-head QK normalization
  test_kv_quant.c   KV cache quantization
  test_expert_slab_load.c  Expert slab loading

tools/
  convert.py         HF BF16 to int4 streaming converter
  bench_throughput.py Throughput benchmark harness
  cross_oracle_compare.py  Cross-engine comparison vs llama.cpp
  make_m3_oracle.py   Oracle teacher-forcing logit generator
  llama_observability_wrapper.py  Wrapper for llama.cpp telemetry
  observability.py    Cross-engine observability dashboard
  requant_*.py        Re-quantization scripts (attn, dense, bf16, f32)

scripts/
  run-bench.sh        Benchmark with VNNI + NUMA
  run-numa.sh         NUMA-aware launch
  run-cross-oracle.sh Cross-oracle comparison
  run-ai-server.sh    Server mode launch

docs/
  llama-cpp-rq-observability.patch  Unapplied patch for llama.cpp fork

coli                  CLI: chat / serve / convert / bench / numa / observability
```

## Testing

```bash
# C unit tests (no model required)
make test-c

# Python smoke tests
make test

# Full oracle validation (requires 212 GB model, ~15 min each)
make test-oracle-all

# Clean build check (no warnings/errors)
make check
```

## Architecture

The engine implements the MiniMax-M3 architecture:

- **60 layers**, 6144 hidden dim, 64 attention heads, 4 KV heads (GQA)
- **MiniMax Sparse Attention (MSA)** — block-sparse attention with indexer
  Q/K projection (128-dim), per-block top-K selection, 128-token blocks
- **128 experts per layer** (topk=4), 1 shared expert, sigmoid router
- **SwiGLU-OAI** activation (gate * up * down with SiLU)
- **Partial RoPE** (64 rotary dimensions of 128 head_dim)
- **QK normalization** (RMSNorm on Q and K before RoPE)
- **Gemma-style RMSNorm** (subtracts per-row minimum)
- **INT8 KV cache** (auto-enables at ctx >= 16K)
- **PlanarQuant KV** (optional, 2D Givens + int3/4)

## Limitations

- **Memory bandwidth bound** — At 1.23 tok/s we're at ~91% of the theoretical
  single-socket DRAM bandwidth limit (14.3 GB/token at ~15 GB/s). Further gains
  require either int2 quantization (halve weight reads) or multi-socket NUMA
  with compute/I/O overlap.
- **Single-token decode** — The MSA batch path is not exercised (S=1 only).
- **No GPU support** — Pure CPU. Designed for large-RAM servers without GPUs.
- **Model format** — Custom int4 safetensors with `.qs` scales, not GGUF.

## License

Apache-2.0. MiniMax-M3 weights: MiniMax Community License.
