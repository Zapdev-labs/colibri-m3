# colibri-m3

**Original** pure-C MiniMax-M3 inference engine. Disk-streamed MoE experts, resident dense stack, zero runtime deps (libc + OpenMP). No web UI, no GLM/Hy3 fork baggage.

Run the ~428B MiniMax-M3 MoE (~23B active) on a fat CPU box by keeping attention / shared / embeddings in RAM and paging routed experts from NVMe.

## Why this exists

Colibrì showed that huge MoEs are workable on workstation RAM if you treat storage as a memory tier. This tree is a **from-scratch** MiniMax-M3 engine built for that idea — not a renamed upstream checkout.

## Layout

```
src/engine.c      MiniMax-M3 forward (GQA, partial RoPE, SwiGLU-OAI, sigmoid MoE)
src/st.h          safetensors index + pread (no mmap RSS trap)
src/json.h        tiny JSON for headers/config
src/planar_kv.h   PlanarQuant-style KV (optional)
tools/convert.py  HF BF16 → int4 streaming container
coli              chat / run / convert / serve CLI
```

## Build

```bash
make -j
# or
./coli build
```

## Convert weights

```bash
# shard-by-shard from Hugging Face (never needs full BF16 on disk)
./coli convert --repo MiniMaxAI/MiniMax-M3 --model /data/m3_i4

# or local checkout
./coli convert --indir ./MiniMax-M3 --model ./m3_i4
```

## Run (ai-server style: 96 threads, planar KV)

```bash
export COLI_MODEL=/data/m3_i4
export OMP_NUM_THREADS=96 OMP_PLACES=cores OMP_PROC_BIND=close
./coli chat --planar --cap 256 --ctx 8192
./coli serve --planar --host 0.0.0.0 --port 8080
```

Sampling defaults match MiniMax: `temperature=1.0`, `top_p=0.95`, `top_k=40`.

## Speed knobs

| Flag / env | Effect |
|---|---|
| `--planar` / `PLANAR_KV=1` | 2D Givens + int3/4 KV (~4–5× smaller) |
| `--cap N` | expert LRU slots per MoE layer |
| `PIPE` (future) | async expert readahead |
| `ARCH=native` | AVX-512 on Xeon Gold 5220R etc. |

## License

Apache-2.0 (this engine). MiniMax-M3 weights: MiniMax Community License.
