#!/usr/bin/env python3
"""Probe (fixed baseline chunk) + sweep over all chunk sizes at RATE_FRACS[0] x probe throughput
(falling back to lower fractions if the keep-up check fails).

For each (GPU, model, dataset):
  1. probe:  request_rate=inf, PROBE_N prompts, seed 0, chunk=BASELINE -> R (req/s)
  2. rate:   RATE_FRACS[0] * R, shared by every chunk size and seed (Poisson arrivals)
  3. sweep:  every chunk size x seed at that rate, SWEEP_N prompts. Baseline chunk runs first.
  4. check:  the first sweep run of each (model, dataset) (baseline chunk, first seed) must keep up:
               served/offered >= THRU_MIN and no failed request, where
               served/offered = 1/(1+max(0,s)), s = Theil-Sen slope of e2e latency vs arrival time
               (a growing backlog makes latency rise with arrival time).
             If it fails, the failed result is kept as *_frac<f>_failed.json, the rate drops to the
             next fraction in RATE_FRACS and the check run is repeated. Nothing else for that
             (model, dataset) runs until a rate passes, so every chunk/seed shares one rate.
             If every fraction fails, that (model, dataset) is skipped; the rest keeps going.
  5. every other run is also tested for keep-up and logged as WARN if it falls behind
     (log only, the run is kept).

Rationale for RATE_FRACS[0] = 0.5: the full saturation ramps run for Qwen2.5-3B (earlier version of
this script, commit e11f815; results under ramp/) put saturation at 0.60 R (sharegpt) and
1.00 R (arxiv), so 0.5 R is below saturation on both.

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
PROBE_N, SWEEP_N = 100, 250
RATE_FRACS = [0.5, 0.4, 0.3, 0.2]   # first is the target; the rest are fallbacks if the check fails
THRU_MIN = 0.95
SHAREGPT_MAX_OUT = 2048
ROOT = "/workspace/results"
PROBE_DIR = f"{ROOT}/ramp/{GPU_NAME}"   # probes live next to the 3B saturation ramps
SWEEP_DIR = f"{ROOT}/sweep50/{GPU_NAME}"


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


# ---------------------------------------------------------------- keep-up criterion
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


def get_rate(srv, model, dataset):
    d = f"{PROBE_DIR}/{model}/{dataset}"
    path = f"{d}/rate.json"
    if os.path.exists(path):
        return json.load(open(path))
    srv.ensure(model, BASELINE, f"{PROBE_DIR}/{model}/server_chunk-{BASELINE}.log")
    probe = bench(model, dataset, PROBE_N, "inf", 0, f"{d}/probe_inf.json")
    R = probe["request_throughput"]
    rec = dict(gpu=GPU_NAME, model=model, dataset=dataset, baseline_chunk=BASELINE, seed=0,
               probe_prompts=PROBE_N, probe_inf_rps=R, rate_frac=RATE_FRACS[0],
               rate_rps=round(RATE_FRACS[0] * R, 4), check=None, failed_attempts=[])
    json.dump(rec, open(path, "w"), indent=2)
    log(f"RATE {model}/{dataset}: R={R:.3f} -> rate {rec['rate_rps']:.3f} rps")
    return rec


def rate_path(model, dataset):
    return f"{PROBE_DIR}/{model}/{dataset}/rate.json"


def check_passed(model, dataset):
    rec = json.load(open(rate_path(model, dataset)))
    return bool(rec.get("check") and rec["check"].get("pass"))


def check_and_adjust(model, dataset, seed, r, out_json):
    """Keep-up check on the baseline run. Returns ("pass", rate), ("retry", new_rate) or
    ("exhausted", None). On failure the result files are renamed out of the way first, so a
    crash before rate.json is updated just re-runs the check at the same rate."""
    path = rate_path(model, dataset)
    rec = json.load(open(path))
    s = step_stats(r)
    s["pass"] = bool(s["served_over_offered"] >= THRU_MIN and s["failed"] == 0)
    frac = rec.get("rate_frac", RATE_FRACS[0])
    check = dict(run=f"chunk-{BASELINE} seed{seed}", rate_frac=frac, rate_rps=rec["rate_rps"],
                 served_over_offered_min=THRU_MIN, **s)
    log(f"CHECK {model}/{dataset} @ {frac:.2f}R: served/offered={s['served_over_offered']:.3f} "
        f"failed={s['failed']} -> {'PASS' if s['pass'] else 'FAIL'}")
    if s["pass"]:
        rec["check"] = check
        json.dump(rec, open(path, "w"), indent=2)
        return "pass", rec["rate_rps"]
    stem = out_json[:-5]
    for ext in (".json", ".log"):
        if os.path.exists(stem + ext):
            os.replace(stem + ext, f"{stem}_frac{frac:.2f}_failed{ext}")
    rec.setdefault("failed_attempts", []).append(check)
    lower = [f for f in RATE_FRACS if f < frac - 1e-9]
    if not lower:
        rec["check"] = check
        json.dump(rec, open(path, "w"), indent=2)
        return "exhausted", None
    rec["rate_frac"] = lower[0]
    rec["rate_rps"] = round(lower[0] * rec["probe_inf_rps"], 4)
    rec["check"] = None
    json.dump(rec, open(path, "w"), indent=2)
    log(f"  lowering {model}/{dataset} to {lower[0]:.2f}R = {rec['rate_rps']:.3f} rps and re-checking")
    return "retry", rec["rate_rps"]


def warn_if_behind(model, dataset, chunk, seed, r):
    s = step_stats(r)
    if s["served_over_offered"] < THRU_MIN or s["failed"]:
        log(f"  WARN {model}/{dataset} chunk-{chunk} seed{seed} did not keep up: "
            f"served/offered={s['served_over_offered']:.3f} failed={s['failed']} (run kept)")


def main():
    srv = Server()
    signal.signal(signal.SIGTERM, lambda *a: (srv.stop(), sys.exit(1)))
    try:
        for model in MODELS:
            rates = {ds: get_rate(srv, model, ds)["rate_rps"] for ds in DATASETS}
            rates_file = f"{PROBE_DIR}/{model}/sweep50_rates.json"
            json.dump(rates, open(rates_file, "w"), indent=2)
            skipped = set()
            # baseline chunk first: its server is already up from the probes
            for chunk in sorted(CHUNKS, key=lambda c: c != BASELINE):
                for ds in DATASETS:
                    if ds in skipped:
                        continue
                    for seed in SEEDS:
                        out = f"{SWEEP_DIR}/{model}/chunk-{chunk}/{ds}_seed{seed}.json"
                        if os.path.exists(out):
                            continue
                        # the first baseline run that completes is the keep-up check; nothing else
                        # for this (model, dataset) runs until a rate has passed it
                        is_check = chunk == BASELINE and not check_passed(model, ds)
                        if not is_check and not check_passed(model, ds):
                            continue
                        while True:
                            srv.ensure(model, chunk, f"{SWEEP_DIR}/{model}/chunk-{chunk}/server.log")
                            try:
                                r = bench(model, ds, SWEEP_N, rates[ds], seed, out)
                            except RuntimeError as e:
                                log(f"  FAILED {e}")
                                if not srv.alive():
                                    log("  server gone; restarting for next run")
                                    srv.key = None
                                r = None
                                break
                            if not is_check:
                                warn_if_behind(model, ds, chunk, seed, r)
                                break
                            status, new_rate = check_and_adjust(model, ds, seed, r, out)
                            if status == "pass":
                                break
                            if status == "exhausted":
                                log(f"SKIP {model}/{ds}: no rate in {RATE_FRACS} keeps up; "
                                    f"continuing with the rest")
                                skipped.add(ds)
                                break
                            rates[ds] = new_rate
                            json.dump(rates, open(rates_file, "w"), indent=2)
                            time.sleep(10)
                        time.sleep(10)
                        if ds in skipped:
                            break
            srv.stop()
        log("ALL DONE")
    finally:
        srv.stop()


if __name__ == "__main__":
    main()
