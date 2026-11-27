# A100 chunked-prefill sweep

The A100 counterpart of `benchmarks/H_100`: the same method, the same prompts and the same software, on a different GPU.
For each (model, dataset): measure the burst throughput R at chunk 2048, then run every chunk size at a Poisson rate of 0.5 R.

## Same as the H100 sweep

- vLLM **0.34.0** (V1 engine), torch 2.14.0+cu132, transformers 5.22.2; FUEL's `benchmark_serving.py` from this branch
  (`--backend vllm`, OpenAI `/v1/completions`, streaming, greedy, `logprobs=1`), NVML power every 25 ms.
- Server flags: `--no-enable-prefix-caching`, `--enable-chunked-prefill --max-num-batched-tokens <chunk>` or
  `--no-enable-chunked-prefill`; everything else vLLM default.
- Matrix: Qwen2.5-3B/14B-Instruct x sharegpt/arxiv x chunk none/256/512/1024/2048/4096 x seeds 0-2, 250 prompts per run.
- Datasets: the same `sharegpt.json` and `arxiv_sum.jsonl`. Checked: the input length of every request equals the H100 run of the
  same model/chunk/dataset/seed in all 72 runs.
- Rate rule: probe at `rate=inf` (100 prompts, chunk 2048, seed 0) -> R; sweep at 0.5 R, with the 0.4/0.3/0.2 R fallback.

## Different

| | H100 (Nat) | A100 (this folder) |
|---|---|---|
| GPU | H100 80GB HBM3, 700 W | **A100 80GB PCIe, 300 W** (driver 595.91.07) |
| Where | `/workspace` machine | Gilbreth, 1 GPU, 7 CPU cores, 100 GB RAM, one Slurm job |
| Python | not recorded | 3.11.18 (exact packages: `requirements-frozen-vllm034.txt`) |
| `max_num_seqs` default | `min(1024, chunk)` | 256: vLLM 0.34 uses a lower default for any GPU named A100 (`engine/arg_utils.py`). The server logs (10 s snapshots) show at most 24 running and 6 waiting requests, so it never bound. |

All 72 runs ran in one Slurm job (16 h 9 min, exit 0) on one GPU (`GPU-622dec7d-6f36-bfd0-0e40-eae881f36358`, node gilbreth-k015),
so every chunk size was measured on the same physical card.

## Rates (`results/rates.csv`)

| model | dataset | R (req/s) | rate (0.5 R) | keep-up check served/offered |
|---|---|---|---|---|
| 3B | sharegpt | 5.385 | 2.692 | 1.000 |
| 3B | arxiv | 1.196 | 0.598 | 1.000 |
| 14B | sharegpt | 1.642 | 0.821 | 0.999 |
| 14B | arxiv | 0.371 | 0.185 | 1.000 |

All four at 0.5 R (no fallback). All 72 runs: 250/250 requests, 0 failed; lowest served/offered 0.994 (none overloaded).
The sweep's problem log (failed runs, overloads, retries) stayed empty.

## Layout

```
results/   made with ../H_100/scripts/summarize.py, unchanged; same layout and columns as H_100/results
           sweep50/ the 72 runs, ramp/ the 4 probes, smoke/ one 30-prompt smoke run (3B, chunk 512)
           (raw JSONs not committed, same as the H100 side)
```

## How it was run

`H_100/scripts/sweep.py`, changed only around the experiment, not in it (rates, order, runs, measurements unchanged):
cluster paths; `GPU_NAME = "A100-80GB-PCIe"`; a free port picked at start (shared nodes); refuses to start unless it sees
exactly one A100 80GB PCIe and vLLM 0.34.0; failed runs retried (up to 3 passes); half-written results set aside and rerun;
a problem log; the job reports success only when all 72 runs exist. Launched as one Slurm job (1 GPU, 7 CPU cores, 100 GB RAM).
The smoke tests used your `serve_lib.sh`/`smoke.sh` with cluster paths. Scripts not included here; ask if you want them.

Plots of both GPUs: `../analysis/results.ipynb` (figures in `../analysis/figures/`).

## Read before using the numbers

- **P99 TBT can hide rare long stalls.** On Arxiv without chunking, stalls are under 1% of token gaps, so they do not move the
  99th percentile. A100 3B arxiv, no chunking: P99 TBT 11.7-13.9 ms, yet 130-145 of 250 requests hit a gap over 200 ms
  (longest 1.4-1.7 s). Chunk 256: P99 38.5-43.7 ms, 0-5 requests with a gap over 200 ms (longest 0.2-0.6 s). (3 seeds each.) Per-token gaps exist only in the raw JSONs (A100 raw kept on Gilbreth depot).
- **ShareGPT energy differences are within seed noise:** chunk sizes differ by 2-3% in energy per token, the 3 seeds by 10-14%.
  On Arxiv the chunk effect (5-11%) is larger than the seed spread (3-6%).
- **Measured throughput is fixed by the request rate**, not by chunk size (every run kept up). On ShareGPT it reads ~8% below
  the rate because the benchmark's duration includes the drain after the last arrival.
- The 3 seeds share most prompts (the dataset files have 500 conversations / 300 papers; each run samples 250), so they are
  not independent samples of prompts.
