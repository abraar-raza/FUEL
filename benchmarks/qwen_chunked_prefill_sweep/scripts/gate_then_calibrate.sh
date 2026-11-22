#!/usr/bin/env bash
until grep -q "bench exit\|server died\|timeout" /workspace/sweep/smoke.out 2>/dev/null; do sleep 5; done
sleep 5
J=/workspace/results/smoke_3b_512/smoke_3b_512_sharegpt.json
if /workspace/venv/bin/python - "$J" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); g=r["gpu_power_stats"]["0"] if "0" in r["gpu_power_stats"] else list(r["gpu_power_stats"].values())[0]
win_ok = r["duration"] >= max(c for c in r["completion_times"] if c) - r["start_time"] - 1e-3
print("window covers all measured completions:", win_ok)
ok = win_ok and r["completed"]==30 and g["avg_power"]>0 and len(g["power_trace"])>10 and r["energy_stats"]["total_energy"]>0
print("SMOKE", "PASS" if ok else "FAIL", "completed=%d dur=%.1fs avgW=%.1f samples=%d E=%.0fJ"%(r["completed"],r["duration"],g["avg_power"]/1000,len(g["power_trace"]),r["energy_stats"]["total_energy"]))
sys.exit(0 if ok else 1)
PY
then
  echo "starting calibration $(date +%T)"
  cd /workspace && exec /workspace/sweep/calibrate.sh > /workspace/results/calibration.log 2>&1
else
  echo "smoke failed; calibration NOT started"; tail -20 /workspace/results/smoke_3b_512/bench.log
fi
