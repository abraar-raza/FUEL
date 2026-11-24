# H100 chunked-prefill sweep

For each GPU-model-dataset combination, measure the burst throughput R at a fixed baseline chunk
size (2048), then run every chunk size at a Poisson rate of 0.5 R. Measures throughput, latency
and GPU power/energy on vLLM.

## Matrix

| Axis | Values |
|---|---|
| Model | `Qwen2.5-3B-Instruct`, `Qwen2.5-14B-Instruct` |
| Chunk (`--max-num-batched-tokens`) | `none` (chunked prefill off), 256, 512, 1024, 2048, 4096 |
| Dataset | `sharegpt` (`sharegpt.json`), `arxiv` (`arxiv_sum.jsonl`) |
| GPU | 1× H100 80GB HBM3 |
| Seed | 0, 1, 2 in the sweep (prompt sampling and Poisson arrivals); probe uses seed 0 |

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
- ShareGPT output cap (`max_tokens`) is **2048** in probe, ramp and sweep (`--sharegpt-output-len 2048`;
  the sampler default is 5000). Requests normally stop at EOS; the cap bounds runaway generations,
  which at `rate=inf` would otherwise set the run duration on their own (in the smoke run one
  5000-token request took 20.1 s vs 7.7 s for the next slowest).
  arxiv output cap is `max(len(abstract), 200)`; prompts > 30k tokens are dropped.
- The smoke run (`results/smoke`) used the 5000-token default.

## Procedure (`scripts/ramp_sweep.py`)

**1. Rate**: per (GPU, model, dataset), baseline chunk 2048, seed 0.

- Probe at `--request-rate inf` with 100 prompts gives the burst throughput R.
- Sweep rate = **0.5 R** (Poisson arrivals), shared by every chunk size and seed of that
  (GPU, model, dataset), so chunk sizes are compared at identical offered load.

**2. Sweep**: every chunk size x seeds 0-2 at that rate, 250 prompts. The baseline chunk runs first.

**3. Keep-up check**: the first sweep run of each (model, dataset) (chunk 2048, seed 0) must keep
up with the offered rate: served/offered >= 0.95 and no failed request, with
served/offered = 1 / (1 + max(0, s)), s = Theil-Sen slope of end-to-end latency vs arrival time
(a growing backlog makes latency rise with arrival time; the harness's own `request_throughput`
includes the drain tail and is biased low). The result is stored in `rate.json`. On a failure the
failed run is kept as `*_frac<f>_failed.json`, the rate drops to the next of 0.4R, 0.3R, 0.2R and
the check is repeated; if every fraction fails, that (model, dataset) is skipped. In the committed
results all four (model, dataset) checks passed at 0.5 R, so no fallback was used.

Smaller chunk sizes can saturate below the baseline's capacity (chunk 256 also caps
`max_num_seqs` at 256); if they do at 0.5 R, that shows up in their latency and is a result,
not a broken run.

**Why 0.5 R.** For Qwen2.5-3B we first ran a full saturation ramp (100 prompts/step at
0.3R, 0.5R, 0.6R ... extended past 1.0R while passing; a step passes at served/offered >= 0.95,
median TTFT <= 3x the lowest step's, no failures; commit e11f815). Saturation came out at
**0.60 R on sharegpt** (5.72 of R = 9.53 req/s) and **1.00 R on arxiv** (4.97 of R = 4.97 req/s).
0.5 R is below saturation on both, so it is used for every model without re-running the ramp;
the keep-up check verifies it per (model, dataset). The 3B ramps are kept under `results/ramp/`.

A first sweep at 0.75 x saturation was stopped after 7 runs when the rate rule changed; those
runs are not included.

## Layout

```
scripts/                     ramp_sweep.py (pipeline), serve_lib.sh + smoke.sh (smoke test), summarize.py
results/
  smoke/<model>/chunk-<c>/<dataset>/seed<s>/          <run>
  rates.csv                  per (gpu, model, dataset): probe R, sweep rate, keep-up check
  ramp/<gpu>/<model>/<dataset>/
    rate.json                probe R, sweep rate (0.5 R), keep-up check stats
    probe_inf/               <run>
    decision.json, step_<f>R/  3B only: saturation ramp steps and decision (<run> per step)
  ramp/saturation_rates.csv  3B saturation ramp results, one row per (gpu, model, dataset)
  sweep50/<gpu>/<model>/chunk-<c>/<dataset>/seed<s>/   <run>
  sweep50/runs.csv           one row per sweep run (throughput, latency, power, energy)

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
