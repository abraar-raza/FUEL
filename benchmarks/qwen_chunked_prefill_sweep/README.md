# Qwen2.5 chunked-prefill energy sweep

For each GPU-model-dataset combination, find the saturation request rate with a short
calibration ramp at a fixed baseline chunk size (2048), then run every chunk size at 75% of that
rate. Measures throughput, latency and GPU power/energy on vLLM.

## Matrix

| Axis | Values |
|---|---|
| Model | `Qwen2.5-3B-Instruct`, `Qwen2.5-14B-Instruct` |
| Chunk (`--max-num-batched-tokens`) | `none` (chunked prefill off), 256, 512, 1024, 2048, 4096 |
| Dataset | `sharegpt` (`sharegpt.json`), `arxiv` (`arxiv_sum.jsonl`) |
| GPU | 1× H100 80GB HBM3 |
| Seed | 0, 1, 2 in the sweep (prompt sampling and Poisson arrivals); ramp uses seed 0 |

## Environment

- 1× NVIDIA H100 80GB HBM3 (driver 595.112.07, 700 W limit)
- vLLM **0.34.0** from the venv's site-packages (V1 engine), torch 2.14.0+cu132, transformers 5.22.2.
  Scripts run from `/workspace` so `/workspace/repo/vllm` is *not* imported.
- Client: `benchmarks/benchmark_serving.py --backend vllm` (OpenAI `/v1/completions`, streaming,
  greedy, `logprobs=1`), GPU power via NVML every 25 ms (`--gpu-monitor-interval 0.025`).

## Server configuration

```
python -m vllm.entrypoints.openai.api_server --model <model> --no-enable-prefix-caching \
    --enable-chunked-prefill --max-num-batched-tokens <chunk>     # chunk = 256..4096
    --no-enable-chunked-prefill                                    # chunk = none (budget = 32768)
```

Everything else is vLLM 0.34 default. Notes that affect interpretation:

- **Prefix caching is disabled.** The client re-sends the first/last 10% of measured prompts as
  warmup/end-up traffic, and one server is reused across runs; with prefix caching on, measured
  prefills would hit cache.
- **`max_num_seqs` follows vLLM's default `min(1024, max_num_batched_tokens)`**, i.e. 256 / 512 for
  chunk 256 / 512 and 1024 otherwise. Concurrency headroom therefore co-varies with chunk size.
- ShareGPT output cap (`max_tokens`) is **2048** in ramp and sweep (`--sharegpt-output-len 2048`;
  the sampler default is 5000). Requests normally stop at EOS; the cap bounds runaway generations,
  which at `rate=inf` would otherwise set the run duration on their own (in the smoke run one
  5000-token request took 20.1 s vs 7.7 s for the next slowest).
  arxiv output cap is `max(len(abstract), 200)`; prompts > 30k tokens are dropped.
- The smoke run (`results/smoke`) used the 5000-token default.

## Procedure (`scripts/ramp_sweep.py`)

**1. Saturation ramp**: per (GPU, model, dataset), baseline chunk 2048, seed 0, 100 prompts/step.

- Probe at `--request-rate inf` gives an upper bound R (request throughput of the burst).
- Offered rates 0.3R, 0.5R, 0.6R, 0.7R, 0.8R, 0.9R, 1.0R (Poisson); if 1.0R still passes, extend
  to 1.25R, then 1.5R, then 2R.
- A step passes when
  - served/offered ≥ 0.95, with served/offered = 1 / (1 + max(0, s)), s = Theil-Sen slope of
    end-to-end latency vs arrival time (a growing backlog makes latency rise with arrival time;
    the harness's own `request_throughput` includes the drain tail and is biased low at 100 prompts),
  - median TTFT ≤ 3× the median TTFT of the lowest step, and no request failed.
- Saturation rate = highest offered rate of the contiguous passing prefix.

**2. 75% sweep**: every chunk size × seeds 0-2 at 0.75 × saturation of that (GPU, model, dataset),
250 prompts, Poisson arrivals. All chunk sizes share the same offered rate.

## Layout

```
scripts/                     ramp_sweep.py (pipeline), serve_lib.sh + smoke.sh (smoke test), summarize.py
results/
  smoke/<model>/chunk-<c>/<dataset>/seed<s>/          <run>
  ramp/<gpu>/<model>/<dataset>/
    decision.json            probe R, every step's stats and pass/fail, saturation and sweep rate
    probe_inf/, step_<f>R/   <run>
  ramp/saturation_rates.csv  one row per (gpu, model, dataset)
  sweep75/<gpu>/<model>/chunk-<c>/<dataset>/seed<s>/   <run>
  sweep75/runs.csv           one row per sweep run (throughput, latency, power, energy)

<run> = metrics.json           scalars, latency percentiles, power stats, energy, carbon
        requests.csv           per-request lens, TTFT, E2E, mean ITL, timestamps (rel. to start)
        power_trace_gpu0.csv.gz  raw NVML samples (mW) over the measurement window
```

Raw benchmark JSONs (with prompts, generated text and per-token ITLs) are not committed;
`scripts/summarize.py --raw-root <raw> --out results` regenerates this tree from them.

## Measurement fixes made for this sweep

See the commits on this branch:
- the OpenAI-completions backend now drives the power monitors;
- the measurement window ends when *all* measured requests finish (previously the last-indexed
  request, which at `rate=inf` truncated a smoke run from 20.1 s to 0.45 s);
- the client accepts streamed lines up to 64 MiB (aiohttp's 512 KiB default dropped a long,
  successful request with `LineTooLong` in the first calibration attempt).
