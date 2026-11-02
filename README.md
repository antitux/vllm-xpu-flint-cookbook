# Flint on Intel Arc Pro B70 (vLLM XPU)
READ THIS ARTICLE FIRST: https://antitux.dev/software/2026/10/30/flint-on-arc-pro-b70/

This runs [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint), a 39.8B total / 2.5B active MoE decoder, on a box with 4x Intel Arc Pro B70 GPUs (32 GiB each). It uses the vLLM XPU nightly, BF16 weights, tensor parallelism across all 4 GPUs, and Triton for attention.

The repo is a self-contained stack: one `docker-compose.yaml` plus the three source patches that make the Flint model work on XPU. The nightly ships Flint for NVIDIA SM100 first, so XPU needs a few fixes on top.

## The model

| Property | Value |
|---|---|
| Architecture | `FlintForCausalLM` (Qwen3-MoE-style decoder) |
| Total / active params | 39.8B / 2.5B |
| Layers | 49 (2 dense MLP-only + 47 MoE) |
| Attention | Hybrid: 14 full-attention + 35 sliding-window (513-token) layers, with learned attention sinks. Full layers use no RoPE. |
| GQA | 32 query heads / 4 KV heads, head_dim 128 |
| MoE | 256 routed experts, top-6 sigmoid routing (renormalized) + 1 ungated shared expert |
| Context | 262,144 tokens |
| Vocab | 150,144 |
| Weights | BF16, 15 safetensors shards (~75 GB) |

## Repository layout

```
.
├── docker-compose.yaml        # The full stack: one `flint` service running Flint
├── models/
│   └── .gitkeep               # Put the Flint weights here (see "Setup")
└── patches/flint/
    ├── norm.py                # Patch 1: portable Triton norm/silu kernels
    ├── moe.py                 # Patch 2: XPU import fix (future annotations)
    └── model.py               # Patch 3: conditional FlashInfer import
```

### `docker-compose.yaml`

A single service (`flint`) that runs `vllm serve` on the XPU nightly image. It:

- Mounts the local Flint weights at `/model`.
- Bind-mounts the three patches onto the installed vLLM copy at `/opt/venv/lib/python3.12/site-packages/vllm/models/flint/...` (see "Why site-packages" below).
- Binds all four GPUs (`/dev/dri`) and uses the FLAT device hierarchy plus loopback oneCCL pinning, which is the standard B70 multi-GPU setup.
- Serves on port 8000 with `--served-model-name flint`.

Here are the key serve flags and why each is set:

| Flag | Value | Why |
|---|---|---|
| `--tensor-parallel-size` | `4` | One rank per B70. |
| `--dtype` | `bfloat16` | Matches the checkpoint. |
| `--kv-cache-dtype` | `auto` | Flint's `FlintAttention` asserts the KV dtype is `auto` or `bfloat16`. Its fused `qk_norm_rope_cache` kernel writes K/V in bf16, so fp8 KV is not supported. `auto` resolves to bf16. |
| `--attention-backend` | `TRITON_ATTN` | The XPU Flash-Attention kernel is precompiled without Flint's sliding-window + attention-sink config (`8,128,64,false,true,true`), so FA2 crashes at graph capture. Triton handles sliding window and sinks natively. |
| `--max-model-len` | `262144` | Full model context. |
| `--max-num-seqs` | `16` | Decode batch ceiling. |
| `--max-num-batched-tokens` | `8192` | Chunked-prefill budget. |
| `--gpu-memory-utilization` | `0.90` | Weights take ~19.4 GiB per GPU; this leaves ~7 GiB per GPU for KV. |
| `--tool-call-parser` / `--reasoning-parser` | `glm47` | Flint's tool and reasoning format is glm47-compatible. |

One compose gotcha that cost us a run: the `command` block is a single backslash-continued shell command. A bare `#` comment line in the middle of that continuation silently ends the command, because bash treats a comment as a non-continuing line, so any flags after it get dropped. That is why all the comments sit above the `exec vllm serve ...` line, not inside it.

### `patches/flint/norm.py`

Flint's Triton `RMSNorm`, fused-add-RMSNorm, and SiLU-and-mul kernels. Three changes for XPU:

1. `_rsqrt` used CUDA PTX inline asm (`rsqrt.approx.ftz.f32`), which the Intel Triton backend cannot build. Replaced with the portable `1.0 / tl.sqrt(x)`.
2. `_silu_and_mul_kernel` used `tl.math.div_rn(...)`, which is CUDA-specific. Replaced with plain `g / (1.0 + tldevice.exp(-g))` (the bf16 output rounds anyway).
3. `silu_and_mul` picked up upstream PR #63986: `bd = min(d, 1024)` became `bd = min(d & -d, 1024)`, since the block size has to be a power of two for the TP-shard path.

### `patches/flint/moe.py`

Adds `from __future__ import annotations` at the top. On XPU, `torch.cuda` is a lazy accessor function rather than the real CUDA module, so a PEP-604 union annotation like `torch.cuda.Event | None` evaluates to `function | None` and raises `TypeError: unsupported operand type(s) for |: 'function' and 'NoneType'` at import time. The future-import turns all annotations into strings, so they are never evaluated at import. On XPU the fused MoE decode path is inactive, so the `Event` is never actually used at runtime; only the import-time annotation was the problem.

### `patches/flint/model.py`

`FlintAttention.__init__` originally imported the FlashInfer backend and asserted `isinstance(impl, FlashInferImpl)` unconditionally, but the XPU image ships no `flashinfer` package and uses Triton. The patch computes `flint_decode = _flint_kernels(...)` first (which is `False` on XPU, since the fused kernels target SM100 + TP=1) and only imports and asserts FlashInfer when it is `True`. When it is `False`, it aliases the active (Triton) impl and the forward path uses stock `unified_attention_with_output`, so FlashInfer is genuinely unused.

## Why the patches mount onto site-packages (not `/workspace/vllm`)

The XPU image contains two copies of vLLM:

- `/workspace/vllm/vllm/`, a source checkout.
- `/opt/venv/lib/python3.12/site-packages/vllm/`, the installed copy.

Which one `import vllm` resolves to depends on the working directory. The `vllm serve` worker runs with `working_dir: /`, so it imports the site-packages copy. Mounting the patches onto `/workspace/vllm/...` is a silent no-op (that was the first-run failure). All three mounts therefore target the site-packages path.

## Setup

1. Get the weights into `./models/RecursiveSI/flint` (the compose mounts `./models/RecursiveSI/flint:/model`):

   ```bash
   cd /path/to/flint-test
   hf download RecursiveSI/flint --local-dir ./models/RecursiveSI/flint
   ```

   That is ~75 GB: 15 shards plus `config.json`, `tokenizer.json`, `chat_template.jinja`, and `generation_config.json`.

2. Start the stack:

   ```bash
   docker compose up -d
   docker logs -f flint
   ```

   First start is slow: 75 GB local load plus 49-layer 4-rank graph capture. The healthcheck allows up to 30 minutes (`start_period: 1800s`).

3. Verify it is up:

   ```bash
   curl -s localhost:8000/health
   curl -s localhost:8000/v1/models
   ```

4. Smoke test:

   ```bash
   curl -s localhost:8000/v1/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"flint","prompt":"The capital of France is","max_tokens":16}'
   ```

## Requirements

- 4x Intel Arc Pro B70 (or equivalent Xe2 XPU), each with at least ~20 GiB free.
- Host render group `107` (the compose adds it via `group_add`).
- The XPU image, pulled automatically by `docker compose up`: `vllm/vllm-openai-xpu:v0.33.0'.

## Notes and caveats

- Re-check for a newer nightly before relying on this. A newer image may already include the #63986 norm fix (in which case you can drop the `norm.py` mount) and a dedicated `flint` reasoning/tool parser (then swap the `glm47` parsers for `flint`).
- KV cache: bf16 KV is ~2 KB per token (4 KV heads x 128 x 2 bytes x 2). At 262144 context the engine allocates ~904k tokens total (~7 GiB per GPU) and reports ~3.4x concurrency at full context.
- The service is named `flint`: the served model name is `flint`.
