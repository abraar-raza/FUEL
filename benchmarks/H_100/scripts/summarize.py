#!/usr/bin/env python3
"""Convert raw benchmark_serving.py result JSONs into the committed, structured layout.

Raw JSONs (with prompts, generated texts, per-token ITLs) stay out of git; this keeps
everything needed to analyse throughput, latency and energy. Incomplete or unreadable raw
JSONs (e.g. a run still being written) are skipped, so this can be run mid-sweep.

Usage:
  summarize.py --raw-root /workspace/results --out <repo>/benchmarks/qwen_chunked_prefill_sweep/results
"""
import argparse
import csv
import gzip
import json
import os
import re

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


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        print(f"skip {path}: {e}")
        return None


def subdirs(d):
    return sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))) if os.path.isdir(d) else []


def summary_row(c: dict) -> dict:
    g = next(iter(c["gpu"].values()), {})
    e = c["energy"] or {}
    lat = c["latency_ms"]
    return dict(
        completed=c["completed"], failed=c["failed"], duration_s=c["duration"],
        request_rate=c["request_rate"],
        request_throughput_rps=c["request_throughput"],
        output_throughput_tps=c["output_throughput"],
        total_token_throughput_tps=c["total_token_throughput"],
        median_ttft_ms=lat.get("median_ttft_ms"), p99_ttft_ms=lat.get("p99_ttft_ms"),
        mean_tpot_ms=lat.get("mean_tpot_ms"), p99_tpot_ms=lat.get("p99_tpot_ms"),
        median_e2el_ms=lat.get("median_e2el_ms"), p99_e2el_ms=lat.get("p99_e2el_ms"),
        avg_power_w=g.get("avg_power_w"),
        total_energy_j=e.get("total_energy"),
        energy_per_request_j=e.get("energy_per_request"),
        energy_per_token_j=e.get("energy_per_token"),
    )


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


chunk_key = lambda c: -1 if c == "none" else int(c)


def do_ramp(a):
    """ramp/<gpu>/<model>/<dataset>/{decision.json, probe_inf.json, step_<f>R.json}"""
    root = os.path.join(a.raw_root, "ramp")
    sat_rows, rate_rows = [], []
    for gpu in subdirs(root):
        for model in subdirs(os.path.join(root, gpu)):
            for ds in subdirs(os.path.join(root, gpu, model)):
                src = os.path.join(root, gpu, model, ds)
                dst = os.path.join(a.out, "ramp", gpu, model, ds)
                for fn in sorted(os.listdir(src)):
                    mm = re.match(r"(probe_inf|step_[\d.]+R)\.json$", fn)
                    if not mm:
                        continue
                    raw = load(os.path.join(src, fn))
                    if raw is None:
                        continue
                    meta = dict(gpu=gpu, model=model, chunk="2048", dataset=ds, seed=0,
                                ramp_point=mm.group(1))
                    write_run(raw, meta, os.path.join(dst, mm.group(1)), a.monitor_interval)
                if os.path.exists(os.path.join(src, "rate.json")):
                    rec = load(os.path.join(src, "rate.json"))
                    if rec is not None:
                        os.makedirs(dst, exist_ok=True)
                        with open(os.path.join(dst, "rate.json"), "w") as f:
                            json.dump(rec, f, indent=2)
                        chk = rec.get("check") or {}
                        rate_rows.append(dict(
                            gpu=gpu, model=model, dataset=ds, baseline_chunk=rec["baseline_chunk"],
                            probe_inf_rps=rec["probe_inf_rps"], rate_frac=rec["rate_frac"],
                            rate_rps=rec["rate_rps"],
                            check_served_over_offered=chk.get("served_over_offered"),
                            check_failed=chk.get("failed"), check_pass=chk.get("pass")))
                dec = load(os.path.join(src, "decision.json")) if os.path.exists(
                    os.path.join(src, "decision.json")) else None
                if dec is None:
                    continue
                os.makedirs(dst, exist_ok=True)
                with open(os.path.join(dst, "decision.json"), "w") as f:
                    json.dump(dec, f, indent=2)
                last = dec["steps"][-1]
                sat_rows.append(dict(
                    gpu=gpu, model=model, dataset=ds, baseline_chunk=dec["baseline_chunk"],
                    probe_inf_rps=dec["probe_inf_rps"], saturation_rps=dec["saturation_rps"],
                    saturation_frac_of_R=dec["saturation_rps"] / dec["probe_inf_rps"],
                    saturation_capped_by_ramp=dec["saturation_capped_by_ramp"],
                    sweep_frac=dec["sweep_frac"], sweep_rate_rps=dec["sweep_rate_rps"],
                    n_steps=len(dec["steps"]),
                    first_failing_step=None if last["pass"] else last["frac_of_R"],
                    first_failing_served_over_offered=None if last["pass"] else last["served_over_offered"],
                ))
    write_csv(os.path.join(a.out, "ramp", "saturation_rates.csv"), sat_rows)
    write_csv(os.path.join(a.out, "rates.csv"), rate_rows)
    print(f"ramp: {len(sat_rows)} saturation decisions, {len(rate_rows)} sweep rates")


def do_sweep(a):
    """sweep50/<gpu>/<model>/chunk-<c>/<dataset>_seed<s>.json"""
    root = os.path.join(a.raw_root, "sweep50")
    rows = []
    for gpu in subdirs(root):
        for model in subdirs(os.path.join(root, gpu)):
            for cdir in subdirs(os.path.join(root, gpu, model)):
                cm = re.match(r"chunk-(\w+)$", cdir)
                if not cm:
                    continue
                chunk = cm.group(1)
                src = os.path.join(root, gpu, model, cdir)
                for fn in sorted(os.listdir(src)):
                    fm = re.match(r"(\w+)_seed(\d+)\.json$", fn)
                    if not fm:
                        continue
                    raw = load(os.path.join(src, fn))
                    if raw is None:
                        continue
                    ds, seed = fm.group(1), int(fm.group(2))
                    meta = dict(gpu=gpu, model=model, chunk=chunk, dataset=ds, seed=seed)
                    write_run(raw, meta, os.path.join(a.out, "sweep50", gpu, model, cdir, ds,
                                                      f"seed{seed}"), a.monitor_interval)
                    rows.append({**meta, **summary_row(compact(raw, meta))})
    rows.sort(key=lambda r: (r["gpu"], r["model"], r["dataset"], chunk_key(r["chunk"]), r["seed"]))
    write_csv(os.path.join(a.out, "sweep50", "runs.csv"), rows)
    print(f"sweep50: {len(rows)} runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="/workspace/results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--monitor-interval", type=float, default=0.025)
    a = ap.parse_args()

    # smoke
    sj = os.path.join(a.raw_root, "smoke_3b_512", "smoke_3b_512_sharegpt.json")
    raw = load(sj) if os.path.exists(sj) else None
    if raw is not None:
        meta = dict(model="Qwen2.5-3B-Instruct", chunk="512", dataset="sharegpt", seed=0)
        write_run(raw, meta,
                  os.path.join(a.out, "smoke", "Qwen2.5-3B-Instruct", "chunk-512", "sharegpt", "seed0"),
                  a.monitor_interval)
        print("smoke: 1 run")

    do_ramp(a)
    do_sweep(a)


if __name__ == "__main__":
    main()
