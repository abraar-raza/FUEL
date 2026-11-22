#!/usr/bin/env python3
"""Convert raw benchmark_serving.py result JSONs into the committed, structured layout.

Raw JSONs (with prompts, generated texts, per-token ITLs) stay out of git; this keeps
everything needed to analyse throughput, latency and energy.

Usage:
  summarize.py --raw-root /workspace/results --out <repo>/benchmarks/qwen_chunked_prefill_sweep/results
"""
import argparse
import csv
import gzip
import json
import os
import re
import statistics as st
from collections import defaultdict

SCALARS = [
    "date", "num_prompts", "request_rate", "duration", "start_time", "end_time", "completed",
    "total_input_tokens", "total_output_tokens", "request_throughput", "output_throughput",
    "total_token_throughput",
]


def compact(raw: dict, meta: dict) -> dict:
    m = {**meta}
    m.update({k: raw.get(k) for k in SCALARS})
    m["failed"] = sum(1 for e in raw.get("errors", []) if e)
    m["latency_ms"] = {k: v for k, v in raw.items()
                       if re.match(r"(mean|median|std|p\d+)_(ttft|tpot|itl|e2el)_ms$", k)}
    gpus = {}
    for gid, g in (raw.get("gpu_power_stats") or {}).items():
        gpus[str(gid)] = {
            "avg_power_w": g["avg_power"] / 1000,
            "avg_gpu_util_pct": g["avg_gpu_util"],
            "avg_mem_util_pct": g["avg_mem_util"],
            "power_w": {k: (v / 1000 if k != "power_std" else v)
                        for k, v in g["power_stats"].items()},
            "n_samples": len(g["power_trace"]),
        }
    m["gpu"] = gpus
    m["energy"] = raw.get("energy_stats")
    m["carbon"] = raw.get("carbon_stats")
    return m


def write_run(raw: dict, meta: dict, d: str, monitor_interval: float):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(compact(raw, meta), f, indent=2)
    t0 = raw["start_time"]
    rel = lambda x: None if x is None else round(x - t0, 6)
    with open(os.path.join(d, "requests.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "success", "input_len", "output_len", "ttft_s", "e2el_s",
                    "mean_itl_s", "arrival_rel_s", "first_token_rel_s", "completion_rel_s",
                    "cumulative_logprob"])
        for i in range(len(raw["input_lens"])):
            itl = raw["itls"][i]
            w.writerow([i, not raw["errors"][i], raw["input_lens"][i], raw["output_lens"][i],
                        raw["ttfts"][i], raw["e2els"][i],
                        (sum(itl) / len(itl)) if itl else "",
                        rel(raw["arrival_times"][i]), rel(raw["first_token_times"][i]),
                        rel(raw["completion_times"][i]), raw["cumulative_logprob"][i]])
    for gid, g in (raw.get("gpu_power_stats") or {}).items():
        with gzip.open(os.path.join(d, f"power_trace_gpu{gid}.csv.gz"), "wt") as f:
            f.write(f"# sample interval ~{monitor_interval}s (sleep-based; not exact timestamps)\n")
            f.write("sample,power_mw,mem_used_pct\n")
            for i, (p, mu) in enumerate(zip(g["power_trace"], g["memory_util_trace"])):
                f.write(f"{i},{p},{mu:.3f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="/workspace/results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--monitor-interval", type=float, default=0.025)
    ap.add_argument("--fraction", type=float, default=0.75)
    a = ap.parse_args()

    # smoke
    sj = os.path.join(a.raw_root, "smoke_3b_512", "smoke_3b_512_sharegpt.json")
    if os.path.exists(sj):
        meta = dict(model="Qwen2.5-3B-Instruct", chunk="512", dataset="sharegpt", seed=0)
        write_run(json.load(open(sj)), meta,
                  os.path.join(a.out, "smoke", "Qwen2.5-3B-Instruct", "chunk-512", "sharegpt", "seed0"),
                  a.monitor_interval)

    # calibration
    croot = os.path.join(a.raw_root, "calibration")
    rows = []
    for tag in sorted(os.listdir(croot)) if os.path.isdir(croot) else []:
        mm = re.match(r"(.+)_chunk-(\w+)$", tag)
        if not mm:
            continue
        model, chunk = mm.groups()
        for fn in sorted(os.listdir(os.path.join(croot, tag))):
            fm = re.match(r"(\w+)_seed(\d+)\.json$", fn)
            if not fm:
                continue
            ds, seed = fm.group(1), int(fm.group(2))
            raw = json.load(open(os.path.join(croot, tag, fn)))
            meta = dict(model=model, chunk=chunk, dataset=ds, seed=seed)
            write_run(raw, meta, os.path.join(a.out, "calibration", "runs", model,
                                              f"chunk-{chunk}", ds, f"seed{seed}"),
                      a.monitor_interval)
            c = compact(raw, meta)
            g = next(iter(c["gpu"].values()), {})
            rows.append(dict(
                model=model, chunk=chunk, dataset=ds, seed=seed,
                completed=c["completed"], failed=c["failed"], duration_s=c["duration"],
                request_throughput_rps=c["request_throughput"],
                output_throughput_tps=c["output_throughput"],
                total_token_throughput_tps=c["total_token_throughput"],
                mean_ttft_ms=c["latency_ms"].get("mean_ttft_ms"),
                p99_ttft_ms=c["latency_ms"].get("p99_ttft_ms"),
                mean_tpot_ms=c["latency_ms"].get("mean_tpot_ms"),
                avg_power_w=g.get("avg_power_w"),
                total_energy_j=(c["energy"] or {}).get("total_energy"),
                energy_per_request_j=(c["energy"] or {}).get("energy_per_request"),
                energy_per_token_j=(c["energy"] or {}).get("energy_per_token"),
            ))
    if not rows:
        return
    chunk_key = lambda c: -1 if c == "none" else int(c)
    rows.sort(key=lambda r: (r["model"], chunk_key(r["chunk"]), r["dataset"], r["seed"]))
    cdir = os.path.join(a.out, "calibration")
    with open(os.path.join(cdir, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    grp = defaultdict(list)
    for r in rows:
        grp[(r["model"], r["chunk"], r["dataset"])].append(r)
    agg, lookup = [], {}
    for (model, chunk, ds), rs in sorted(grp.items(), key=lambda kv: (kv[0][0], chunk_key(kv[0][1]), kv[0][2])):
        x = [r["request_throughput_rps"] for r in rs]
        mean = st.mean(x)
        rate = a.fraction * mean
        agg.append(dict(model=model, chunk=chunk, dataset=ds, n_seeds=len(rs),
                        seeds=" ".join(str(r["seed"]) for r in rs),
                        saturation_rps_mean=mean, saturation_rps_std=st.stdev(x) if len(x) > 1 else 0.0,
                        saturation_rps_min=min(x), saturation_rps_max=max(x),
                        **{f"rate_{int(a.fraction*100)}pct_rps": rate}))
        lookup.setdefault(model, {}).setdefault(ds, {})[chunk] = round(rate, 4)
    with open(os.path.join(cdir, "saturation_rates.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0]))
        w.writeheader(); w.writerows(agg)
    with open(os.path.join(cdir, f"sweep_rates_{int(a.fraction*100)}pct.json"), "w") as f:
        json.dump({"fraction_of_saturation": a.fraction,
                   "definition": "fraction * mean(request_throughput at request_rate=inf) over seeds",
                   "rates_rps": lookup}, f, indent=2)
    print(f"{len(rows)} runs, {len(agg)} groups -> {cdir}")


if __name__ == "__main__":
    main()
