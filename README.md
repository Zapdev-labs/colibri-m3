# colibrì × MiniMax-M3

**Run [MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) (~428B MoE, ~23B active) on a consumer / workstation PC** — pure C, disk-streaming experts, zero runtime deps.

Fork of [colibrì](https://github.com/JustVugg/colibri) / [colibri-hy3](https://github.com/ErikTromp/colibri-hy3) retargeted at MiniMax-M3:

| Piece | Description |
|---|---|
| `c/m3.c` | MiniMax-M3 engine — GQA, partial RoPE, SwiGLU-OAI, sigmoid router, expert streaming |
| `c/planar_kv.h` | PlanarQuant-style 2D Givens + int3/4 KV cache (`PLANAR_KV=1`) |
| `c/tools/convert_m3.py` | BF16 HF → Colibri int4 container (splits packed 3D experts) |
| `coli` | chat / serve / convert / plan / doctor |

Upstream GLM-5.2 (`glm.c`) and Hy3 (`hy3.c`) remain in-tree.

## Target box (ai-server)

Tuned for a dual-NUMA Xeon Gold 5220R (96 threads, AVX-512, ~376 GB RAM, NVMe):

```bash
# build with native AVX-512 + optional io_uring expert pipe
cd c && make m3 IOURING=1 -j$(nproc)

# convert (shard-by-shard — never needs the full BF16 checkpoint on disk)
./coli convert --repo MiniMaxAI/MiniMax-M3 --model /data/m3_i4

# chat — large expert cache, planar KV, OpenMP on all cores
export OMP_NUM_THREADS=96 OMP_PLACES=cores OMP_PROC_BIND=close
export COLI_MODEL=/data/m3_i4
PLANAR_KV=1 PIPE=1 CAP=256 ./coli chat --ram 280 --cap 256
```

With ~280 GB of RAM budget the expert LRU can stay hot; cold tokens still stream from disk.

## Architecture notes

MiniMax-M3 (text):

- 60 layers, hidden 6144, GQA 64/4, head_dim 128, partial RoPE (`rotary_dim=64`)
- 128 experts / top-4, shared expert, first 3 layers dense MLP
- SwiGLU-OAI (`α=1.702`, clamp 7), gemma-style RMSNorm, sigmoid router + bias
- MSA sparse attention falls back to dense GQA for short context (same honest tradeoff as early llama.cpp M3)

## Speed knobs

| Env | Effect |
|---|---|
| `PLANAR_KV=1` | PlanarQuant-style KV (implies `KV_I8`) — ~4–5× smaller KV |
| `PLANAR_BITS=3\|4` | KV quant bits (default 3) |
| `PIPE=1` / `PIPE=2` | async expert readahead (2 = io_uring if built with `IOURING=1`) |
| `IDOT=1` | int8 activation × int4/8 weight matmul (default on) |
| `PERF=1` | timing breakdown (attn / expert disk / expert matmul) |
| `TEMP` / `NUCLEUS` | sampling (M3 defaults: 1.0 / 0.95) |

## Quick start

```bash
cd c
./setup.sh            # gcc/OpenMP check
make m3 -j
./coli convert --repo MiniMaxAI/MiniMax-M3 --model ./m3_i4
COLI_MODEL=./m3_i4 ./coli chat
COLI_MODEL=./m3_i4 ./coli serve --host 0.0.0.0 --port 8080
```

## License

Apache-2.0 (engine). MiniMax-M3 weights: MiniMax Community License.
