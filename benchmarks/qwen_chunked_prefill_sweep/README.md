# Qwen2.5 chunked-prefill energy sweep

Saturation calibration (and later the 75%-of-saturation request-rate sweep) for chunked
prefill on vLLM, measuring throughput, latency and GPU power/energy.

## Matrix

| Axis | Values |
|---|---|
| Model | `Qwen2.5-3B-Instruct`, `Qwen2.5-14B-Instruct` |
| Chunk (`--max-num-batched-tokens`) | `none` (chunked prefill off), 256, 512, 1024, 2048, 4096 |
| Dataset | `sharegpt` (`sharegpt.json`), `arxiv` (`arxiv_sum.jsonl`) |
| Seed | 0, 1, 2 (controls prompt sampling/shuffle and arrival process) |

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
  warmup/end-up traffic, and calibration reuses one server across seeds; with prefix caching on,
  measured prefills would hit cache.
- **`max_num_seqs` follows vLLM's default `min(1024, max_num_batched_tokens)`**, i.e. 256 / 512 for
  chunk 256 / 512 and 1024 otherwise. Concurrency headroom therefore co-varies with chunk size.
- ShareGPT output cap is 5000 tokens (hard-coded in the sampler); requests normally stop at EOS.
  arxiv output cap is `max(len(abstract), 200)`; prompts > 30k tokens are dropped.

## Calibration procedure

For every (model, chunk, dataset, seed): `--request-rate inf`, 250 measured prompts
(+10% warmup before and 10% end-up after, unmeasured). Saturation rate = measured request
throughput. The sweep rate is `0.75 × mean over seeds`.

## Layout

```
scripts/                     serve_lib.sh, smoke.sh, calibrate.sh, gate_then_calibrate.sh, summarize.py
results/
  smoke/<model>/chunk-<c>/<dataset>/seed<s>/
  calibration/
    runs.csv                 one row per run (throughput, latency, power, energy)
    saturation_rates.csv     per (model, chunk, dataset): mean/std/min/max over seeds + 75% rate
    sweep_rates_75pct.json   lookup: rates_rps[model][dataset][chunk]
    runs/<model>/chunk-<c>/<dataset>/seed<s>/
      metrics.json           scalars, latency percentiles, power stats, energy, carbon
      requests.csv           per-request lens, TTFT, E2E, mean ITL, timestamps (rel. to start)
      power_trace_gpu0.csv.gz  raw NVML samples (mW) over the measurement window
```

Raw benchmark JSONs (with prompts, generated text and per-token ITLs) are not committed;
`scripts/summarize.py --raw-root <raw> --out results` regenerates this tree from them.

## Measurement fixes made for this sweep

See the commits on this branch: the OpenAI-completions backend now drives the power monitors,
and the measurement window ends when *all* measured requests finish (previously the last-indexed
request, which at `rate=inf` truncated a smoke run from 20.1 s to 0.45 s).
