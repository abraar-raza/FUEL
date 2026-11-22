#!/usr/bin/env python3
"""Saturation ramp (fixed baseline chunk) + 75% sweep over all chunk sizes.

For each (GPU, model, dataset):
  1. probe:  request_rate=inf, RAMP_N prompts, seed 0, chunk=BASELINE -> R (req/s)
  2. ramp:   offered rates f*R for f in RAMP_FRACS (extended by EXT_FRACS while every step passes)
             step passes iff
               served/offered >= THRU_MIN   served/offered = 1/(1+s), s = Theil-Sen slope of
                                            e2e latency vs arrival time (backlog growth)
               median TTFT     <= TTFT_MAX_X * median TTFT of the lowest step
             saturation rate = highest offered rate of the contiguous passing prefix
  3. sweep:  every chunk size x seed at SWEEP_FRAC * saturation, SWEEP_N prompts, Poisson arrivals.

Resumable: any result JSON that already exists is reused. Run from /workspace.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

import numpy as np

PY = "/workspace/venv/bin/python"
BENCH = "/workspace/repo/benchmarks/benchmark_serving.py"
PORT = 8000
GPU_ID = 0
GPU_NAME = "H100-80GB-HBM3"
MODELS = os.environ.get("MODELS", "Qwen2.5-3B-Instruct Qwen2.5-14B-Instruct").split()
DATASETS = {"sharegpt": "/workspace/datasets/sharegpt.json",
            "arxiv": "/workspace/datasets/arxiv_sum.jsonl"}
CHUNKS = os.environ.get("CHUNKS", "none 256 512 1024 2048 4096").split()
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0 1 2").split()]
BASELINE = "2048"
RAMP_N, SWEEP_N = 100, 250
RAMP_FRACS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EXT_FRACS = [1.25, 1.5, 2.0]
THRU_MIN, TTFT_MAX_X = 0.95, 3.0
SWEEP_FRAC = 0.75
SHAREGPT_MAX_OUT = 2048
ROOT = "/workspace/results"
RAMP_DIR = f"{ROOT}/ramp/{GPU_NAME}"
SWEEP_DIR = f"{ROOT}/sweep75/{GPU_NAME}"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- server
class Server:
    def __init__(self):
        self.proc = None
        self.key = None

    def ensure(self, model, chunk, logfile):
        if self.key == (model, chunk) and self.proc and self.proc.poll() is None:
            return
        self.stop()
        cargs = (["--no-enable-chunked-prefill"] if chunk == "none" else
                 ["--enable-chunked-prefill", "--max-num-batched-tokens", chunk])
        cmd = [PY, "-m", "vllm.entrypoints.openai.api_server", "--model", f"/workspace/models/{model}",
               "--host", "127.0.0.1", "--port", str(PORT), "--no-enable-prefix-caching", *cargs]
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        log(f"start server {model} chunk={chunk}")
        self.proc = subprocess.Popen(cmd, cwd="/workspace", stdout=open(logfile, "w"),
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": str(GPU_ID)})
        for _ in range(1200):
            if self.proc.poll() is not None:
                raise RuntimeError(f"server died, see {logfile}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)
                self.key = (model, chunk)
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("server start timeout")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc, self.key = None, None
        for _ in range(60):  # wait for GPU memory release
            used = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                 "-i", str(GPU_ID)]).split()[0])
            if used < 1000:
                break
            time.sleep(1)
        time.sleep(10)

    def alive(self):
        return self.proc is not None and self.proc.poll() is None


def bench(model, dataset, n, rate, seed, out_json):
    if os.path.exists(out_json) and os.path.getsize(out_json) > 0:
        return json.load(open(out_json))
    d, f = os.path.split(out_json)
    os.makedirs(d, exist_ok=True)
    cmd = [PY, BENCH, "--backend", "vllm", "--host", "127.0.0.1", "--port", str(PORT),
           "--endpoint", "/v1/completions", "--model", f"/workspace/models/{model}",
           "--dataset-name", dataset, "--dataset-path", DATASETS[dataset],
           "--num-prompts", str(n), "--request-rate", str(rate), "--seed", str(seed),
           "--save-result", "--result-dir", d, "--result-filename", f,
           "--gpu-ids", str(GPU_ID), "--gpu-monitor-interval", "0.025", "--disable-tqdm"]
    if dataset == "sharegpt":
        cmd += ["--sharegpt-output-len", str(SHAREGPT_MAX_OUT)]
    t0 = time.time()
    rc = subprocess.run(cmd, cwd="/workspace", stdout=open(out_json[:-5] + ".log", "w"),
                        stderr=subprocess.STDOUT).returncode
    if rc != 0 or not os.path.exists(out_json):
        raise RuntimeError(f"bench failed rc={rc}: {out_json[:-5]}.log")
    r = json.load(open(out_json))
    fails = sum(1 for e in r["errors"] if e)
    log(f"  {os.path.relpath(out_json, ROOT)}: rate={rate} ok={r['completed']} fail={fails} "
        f"rps={r['request_throughput']:.3f} medTTFT={r['median_ttft_ms']:.0f}ms "
        f"E={r['energy_stats']['total_energy']:.0f}J ({time.time() - t0:.0f}s)")
    return r


# ---------------------------------------------------------------- ramp criterion
def latency_slope(arrivals, latencies):
    """Theil-Sen slope of end-to-end latency vs arrival time (s per s). ~0 when the server keeps
    up; s > 0 when a backlog builds. Output length is independent of arrival order, so it adds
    noise but no bias."""
    pts = [(a, l) for a, l in zip(arrivals, latencies) if a is not None and l]
    a = np.array([p[0] for p in pts]); l = np.array([p[1] for p in pts])
    i, j = np.triu_indices(len(a), 1)
    da = a[j] - a[i]
    m = da > 1e-6
    return float(np.median((l[j] - l[i])[m] / da[m]))


def step_stats(r):
    n = len(r["arrival_times"])
    span = max(r["arrival_times"]) - min(r["arrival_times"])
    offered = (n - 1) / span
    slope = latency_slope(r["arrival_times"], r["e2els"])
    ratio = 1.0 / (1.0 + max(slope, 0.0))
    return dict(offered_rps=offered, latency_slope=slope, served_rps=offered * ratio,
                served_over_offered=ratio, median_ttft_ms=r["median_ttft_ms"],
                harness_rps=r["request_throughput"],
                failed=sum(1 for e in r["errors"] if e))


def ramp(srv, model, dataset):
    d = f"{RAMP_DIR}/{model}/{dataset}"
    dec_path = f"{d}/decision.json"
    if os.path.exists(dec_path):
        return json.load(open(dec_path))
    srv.ensure(model, BASELINE, f"{RAMP_DIR}/{model}/server_chunk-{BASELINE}.log")
    probe = bench(model, dataset, RAMP_N, "inf", 0, f"{d}/probe_inf.json")
    R = probe["request_throughput"]
    steps, base_ttft, sat = [], None, None
    fracs = list(RAMP_FRACS)
    i = 0
    while i < len(fracs):
        f = fracs[i]
        rate = round(f * R, 4)
        r = bench(model, dataset, RAMP_N, rate, 0, f"{d}/step_{f:.2f}R.json")
        s = {"frac_of_R": f, "rate_rps": rate, **step_stats(r)}
        if base_ttft is None:
            base_ttft = s["median_ttft_ms"]
        s["ttft_over_base"] = s["median_ttft_ms"] / base_ttft
        s["pass"] = bool(s["served_over_offered"] >= THRU_MIN and s["ttft_over_base"] <= TTFT_MAX_X
                         and s["failed"] == 0)
        steps.append(s)
        log(f"  step {f:.2f}R rate={rate:.3f} served/offered={s['served_over_offered']:.3f} "
            f"ttft x{s['ttft_over_base']:.2f} -> {'PASS' if s['pass'] else 'FAIL'}")
        if not s["pass"]:
            break
        sat = rate
        if i == len(fracs) - 1 and fracs[-1] < EXT_FRACS[-1]:
            fracs += [x for x in EXT_FRACS if x > fracs[-1]][:1]
        i += 1
    if sat is None:
        raise RuntimeError(f"no passing ramp step for {model}/{dataset}; lowest step already saturated")
    dec = dict(gpu=GPU_NAME, model=model, dataset=dataset, baseline_chunk=BASELINE, seed=0,
               prompts_per_step=RAMP_N, probe_inf_rps=R, criterion=dict(
                   served_over_offered_min=THRU_MIN, median_ttft_max_x_lowest_step=TTFT_MAX_X,
                   served_over_offered="1/(1+max(0, Theil-Sen slope of e2e latency vs arrival time))"),
               saturation_rps=sat, saturation_capped_by_ramp=bool(steps[-1]["pass"]),
               sweep_frac=SWEEP_FRAC, sweep_rate_rps=round(SWEEP_FRAC * sat, 4), steps=steps)
    json.dump(dec, open(dec_path, "w"), indent=2)
    log(f"RAMP {model}/{dataset}: R={R:.3f} saturation={sat:.3f} -> sweep rate {dec['sweep_rate_rps']:.3f} rps")
    return dec


def main():
    srv = Server()
    signal.signal(signal.SIGTERM, lambda *a: (srv.stop(), sys.exit(1)))
    try:
        for model in MODELS:
            rates = {ds: ramp(srv, model, ds)["sweep_rate_rps"] for ds in DATASETS}
            json.dump(rates, open(f"{RAMP_DIR}/{model}/sweep_rates.json", "w"), indent=2)
            # baseline chunk first: its server is already up from the ramp
            for chunk in sorted(CHUNKS, key=lambda c: c != BASELINE):
                for ds in DATASETS:
                    for seed in SEEDS:
                        out = f"{SWEEP_DIR}/{model}/chunk-{chunk}/{ds}_seed{seed}.json"
                        if os.path.exists(out):
                            continue
                        srv.ensure(model, chunk, f"{SWEEP_DIR}/{model}/chunk-{chunk}/server.log")
                        try:
                            bench(model, ds, SWEEP_N, rates[ds], seed, out)
                        except RuntimeError as e:
                            log(f"  FAILED {e}")
                            if not srv.alive():
                                log("  server gone; restarting for next run")
                                srv.key = None
                        time.sleep(10)
            srv.stop()
        log("ALL DONE")
    finally:
        srv.stop()


if __name__ == "__main__":
    main()
